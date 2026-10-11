# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Residual detail paths that preserve the six-CSAR backbone and existing checkpoint keys."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .block import Proto26MultiLabel
from .conv import Conv, DWConv
from .shadow_dent import LuminanceDetail, Segment26MultiLabelShadow

__all__ = ("DualPathP2Proto", "LuminanceResidualStem", "Segment26MultiLabelExtent")


def _validate_scale(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("detail_scale must be a finite number.")


class LuminanceResidualStem(Conv):
    """Keep the original RGB Conv and add a gated luminance/contrast/gradient branch.

    Inherited conv/bn/act names allow all original stem weights to be loaded. Signed local contrast can expose
    low-contrast shape cues, but is not a physical depth or shadow estimate.
    """

    def __init__(self, c1, c2, k=3, s=2, detail_scale=0.1):
        if c1 != 3:
            raise ValueError("LuminanceResidualStem requires RGB input with three channels.")
        _validate_scale(detail_scale)
        super().__init__(c1, c2, k, s)
        self.luminance_detail = LuminanceDetail()
        self.detail_stem = nn.Sequential(Conv(5, c2, k, s), DWConv(c2, c2, 3))
        self.detail_gate = nn.Conv2d(2 * c2, 1, 1)
        self.detail_scale = nn.Parameter(torch.tensor(float(detail_scale)))
        nn.init.zeros_(self.detail_gate.weight)
        nn.init.zeros_(self.detail_gate.bias)

    def _add_detail(self, x, base):
        detail = self.detail_stem(self.luminance_detail(x))
        gate = self.detail_gate(torch.cat((base, detail), dim=1)).sigmoid()
        return base + self.detail_scale.tanh() * gate * detail

    def forward(self, x):
        return self._add_detail(x, super().forward(x))

    def forward_fuse(self, x):
        """Retain the residual when BaseModel.fuse replaces a Conv's forward method."""
        return self._add_detail(x, super().forward_fuse(x))


class DualPathP2Proto(Proto26MultiLabel):
    """Add a native P2 detail decoder alongside the complete original prototype path.

    The original path first combines scales at P3 then upsamples to P2. This second path projects P2 directly and gates
    its contribution with resized P4 context. No original transpose-convolution or auxiliary localization weights are
    removed or renamed.
    """

    def __init__(self, ch=(), c_=256, c2=32, nc=80, detail_scale=0.1):
        if len(ch) != 4:
            raise ValueError("DualPathP2Proto requires four features ordered P3/P2/P4/P5.")
        _validate_scale(detail_scale)
        super().__init__(ch, c_, c2, nc)
        self.native_p2 = Conv(ch[1], c_, 1)
        self.native_context = Conv(ch[2], c_, 1)
        self.native_gate = nn.Conv2d(2 * c_, 1, 1)
        self.native_refine = DWConv(c_, c_, 3)
        self.native_output = nn.Conv2d(c_, c2, 1)
        self.native_scale = nn.Parameter(torch.tensor(float(detail_scale)))
        nn.init.zeros_(self.native_gate.weight)
        nn.init.zeros_(self.native_gate.bias)

    def forward(self, x):
        if len(x) != 4 or x[1].shape[-2:] != tuple(2 * n for n in x[0].shape[-2:]):
            raise ValueError("DualPathP2Proto expects P3/P2/P4/P5 with native P2 twice the P3 resolution.")
        original = super().forward(x)
        base = original[0] if isinstance(original, tuple) else original
        detail = self.native_p2(x[1])
        context = F.interpolate(self.native_context(x[2]), size=detail.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.native_gate(torch.cat((detail, context), dim=1)).sigmoid()
        delta = self.native_output(self.native_refine(detail + context))
        prototypes = base + self.native_scale.tanh() * gate * delta
        return (prototypes, original[1]) if isinstance(original, tuple) else prototypes


class Segment26MultiLabelExtent(Segment26MultiLabelShadow):
    """Four-scale multi-label head with shallow bypasses and a second P2 prototype path.

    Inputs: four enhanced features, four independent class-logit maps, then shallow P3/P2. The detection and
    mask-coefficient heads and their state keys are inherited unchanged.
    """

    def __init__(
        self,
        nc=80,
        nm=32,
        npr=256,
        multilabel_gain=0.5,
        cooccurrence_weight=2.0,
        detail_scale=0.1,
        reg_max=16,
        end2end=False,
        ch=(),
    ):
        if len(ch) != 10:
            raise ValueError("Segment26MultiLabelExtent requires four features, four logits, and shallow P3/P2.")
        _validate_scale(detail_scale)
        super().__init__(nc, nm, npr, multilabel_gain, cooccurrence_weight, detail_scale, reg_max, end2end, ch)
        self.proto = DualPathP2Proto(ch[:4], self.npr, self.nm, nc, detail_scale)
