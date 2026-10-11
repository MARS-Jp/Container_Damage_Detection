# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Boundary-guided shape fusion and native P2 prototypes for dent segmentation experiments."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .block import Proto, Proto26MultiLabel
from .conv import Conv, DWConv
from .head import Segment26MultiLabel
from .shadow_dent import DentDetailFusion, _local_mean

__all__ = ("Segment26MultiLabelBoundary",)


class ModulatedShapeConv(nn.Module):
    """Optional depthwise DCNv2 residual, using learned offsets and sigmoid modulation masks.

    Torchvision native operators are required only when this branch is enabled. Sampling uses FP32, including under
    autocast/model.half(), then returns to the incoming feature dtype.
    """

    def __init__(self, channels: int):
        super().__init__()
        from torchvision.extension import _assert_has_ops

        _assert_has_ops()  # Fail explicitly: never silently replace DCNv2 with an ordinary convolution.
        self.offset_mask = nn.Conv2d(channels, 27, 3, padding=1)
        self.weight = nn.Parameter(torch.empty(channels, 1, 3, 3))
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()
        self.align_scale = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from torchvision.ops import deform_conv2d

        with torch.autocast(device_type=x.device.type, enabled=False):
            offset_mask = F.conv2d(x.float(), self.offset_mask.weight.float(), self.offset_mask.bias.float(), padding=1)
            offset, mask = offset_mask[:, :18], offset_mask[:, 18:].sigmoid()
            aligned = deform_conv2d(x.float(), offset, self.weight.float(), padding=(1, 1), mask=mask)
        return x + self.align_scale.tanh() * self.act(self.norm(aligned.to(x.dtype)))


class BoundaryGuidedDentFusion(DentDetailFusion):
    """Add deep semantic context and explicitly supervised boundaries to the 0915 shallow path.

    The neck feature always retains an identity/residual path. Boundary evidence only modulates the extra detail, with a
    nonzero floor; neither rust nor an edge threshold is required for D.
    """

    def __init__(self, ch, c2, context_channels, nc, detail_scale=0.1, use_dcn=False):
        super().__init__(ch, c2, detail_scale)
        self.context = Conv(context_channels, c2, 1)
        self.semantic_gate = nn.Conv2d(c2, 1, 1)
        self.shape_align = ModulatedShapeConv(c2) if use_dcn else nn.Identity()
        self.boundary_head = nn.Sequential(DWConv(c2, c2, 3), nn.Conv2d(c2, nc, 1))
        nn.init.zeros_(self.semantic_gate.weight)
        nn.init.zeros_(self.semantic_gate.bias)

    def forward(self, x):
        if len(x) != 3 or x[0].shape[-2:] != x[1].shape[-2:]:
            raise ValueError("BoundaryGuidedDentFusion expects matched neck/shallow features and a context feature.")
        base, shallow = self.base(x[0]), self.shallow(x[1])
        context = F.interpolate(self.context(x[2]), size=base.shape[-2:], mode="bilinear", align_corners=False)
        with torch.autocast(device_type=shallow.device.type, enabled=False):
            signal = shallow.float()
            fine = signal - _local_mean(signal, 3)
            coarse = signal - _local_mean(signal, 7)
        detail = self.shape_align(
            self.refine(torch.cat((shallow, fine.to(shallow.dtype), coarse.to(shallow.dtype)), dim=1))
        )
        boundary = self.boundary_head(detail + context)
        semantic_gate = (self.gate(torch.cat((base, shallow), dim=1)) + self.semantic_gate(context)).sigmoid()
        boundary_gate = 0.5 + 0.5 * boundary.sigmoid().amax(dim=1, keepdim=True)
        fused = base + self.detail_scale.tanh() * semantic_gate * boundary_gate * detail
        return fused, boundary


class NativeP2Proto(Proto26MultiLabel):
    """Fuse at the second input (P2) without downsampling it or upsampling prototypes to P1.

    Input feature order remains P3/P2/P4/P5/P6. Existing projection/refinement parameter names and shapes are retained;
    only the old transpose-convolution parameters are removed.
    """

    def __init__(self, ch=(), c_=256, c2=32, nc=80):
        if len(ch) < 2:
            raise ValueError("NativeP2Proto requires P3 followed by P2 features.")
        super().__init__(ch, c_, c2, nc)
        self.upsample = nn.Identity()

    def forward(self, x):
        size = x[1].shape[-2:]
        feat = F.interpolate(x[0], size=size, mode="nearest")
        for i, refine in enumerate(self.feat_refine):
            projected = refine(x[i + 1])
            if i:  # P2 itself is projected at its native size, never reduced to P3.
                projected = F.interpolate(projected, size=size, mode="nearest")
            feat = feat + projected
        prototypes = Proto.forward(self, self.feat_fuse(feat))
        if self.training:
            return prototypes, {"heatmap": self.heatmap(feat), "seedmap": self.seedmap(feat)}
        return prototypes


class Segment26MultiLabelBoundary(Segment26MultiLabel):
    """0915-compatible input layout plus boundary-guided detail and native P2 mask decoding."""

    def __init__(
        self,
        nc=80,
        nm=32,
        npr=256,
        multilabel_gain=0.5,
        cooccurrence_weight=2.0,
        detail_scale=0.1,
        boundary_gain=0.2,
        use_dcn=False,
        reg_max=16,
        end2end=False,
        ch=(),
    ):
        if len(ch) != 12:
            raise ValueError("Segment26MultiLabelBoundary expects five features, five class logits, shallow P3/P2.")
        if boundary_gain < 0:
            raise ValueError("boundary_gain must be nonnegative.")
        if not isinstance(use_dcn, bool):
            raise ValueError("use_dcn must be a YAML boolean, not a quoted string.")
        super().__init__(nc, nm, npr, multilabel_gain, cooccurrence_weight, reg_max, end2end, ch[:-2])
        self.boundary_gain = float(boundary_gain)
        self.use_dcn = use_dcn
        self.proto = NativeP2Proto(ch[: self.num_feature_levels], self.npr, self.nm, nc)
        self.detail_fusion = nn.ModuleList(
            BoundaryGuidedDentFusion([ch[i], ch[-2 + i]], ch[i], ch[2], nc, detail_scale, use_dcn) for i in range(2)
        )

    def forward(self, x):
        inputs, boundaries = list(x[:-2]), []
        for i, fusion in enumerate(self.detail_fusion):
            inputs[i], boundary = fusion([inputs[i], x[-2 + i], inputs[2]])
            boundaries.append(boundary)
        outputs = super().forward(inputs)
        # Explicit outputs also support validation loss and avoid retaining graph tensors on modules/EMA.
        if self.training:
            outputs["boundary_logits"] = boundaries
        elif not self.export:
            outputs[1]["boundary_logits"] = boundaries
        return outputs
