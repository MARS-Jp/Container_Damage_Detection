# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Luminance detail and shallow feature paths for low-contrast dent experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .conv import Conv, DWConv, IIMStem
from .head import Segment26MultiLabel

__all__ = ("Segment26MultiLabelShadow", "ShadowIIMStem")


def _local_mean(x: torch.Tensor, kernel: int) -> torch.Tensor:
    """Average with replicated borders so uniform surfaces have no artificial edge response."""
    pad = kernel // 2
    return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel, stride=1)


class LuminanceDetail(nn.Module):
    """Keep signed achromatic shape cues alongside, rather than inside, RGB-pair IIM features.

    The five channels are luminance, fine/coarse relative contrast, and signed x/y gradients. Dark pixels use a fixed
    denominator floor and tanh bounds; darkness itself is not a dent label.
    """

    def __init__(self, fine_kernel: int = 5, coarse_kernel: int = 15, eps: float = 1.0 / 255.0):
        super().__init__()
        if fine_kernel < 3 or coarse_kernel <= fine_kernel or fine_kernel % 2 == 0 or coarse_kernel % 2 == 0:
            raise ValueError("LuminanceDetail requires odd kernels with 3 <= fine_kernel < coarse_kernel.")
        if eps <= 0:
            raise ValueError("LuminanceDetail requires eps > 0.")
        self.fine_kernel, self.coarse_kernel, self.eps = fine_kernel, coarse_kernel, float(eps)
        self.register_buffer("luma_weights", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        self.register_buffer(
            "sobel",
            torch.tensor(
                [
                    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                ]
            ).unsqueeze(1)
            / 8.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("LuminanceDetail expects a BCHW RGB tensor in [0, 1].")
        # The explicit casts also support model.half() at inference without changing module state.
        with torch.autocast(device_type=x.device.type, enabled=False):
            luminance = (x.float() * self.luma_weights.float()).sum(dim=1, keepdim=True)
            fine = _local_mean(luminance, self.fine_kernel)
            coarse = _local_mean(luminance, self.coarse_kernel)
            fine_contrast = ((luminance - fine) / (fine + self.eps)).tanh()
            coarse_contrast = ((luminance - coarse) / (coarse + self.eps)).tanh()
            gradient = F.conv2d(F.pad(luminance, (1, 1, 1, 1), mode="replicate"), self.sobel.float())
            gradient = (gradient / (fine + self.eps)).tanh()
            detail = torch.cat((luminance, fine_contrast, coarse_contrast, gradient), dim=1)
        return detail.to(dtype=x.dtype)


class ShadowIIMStem(IIMStem):
    """Extend the existing RGB/IIM stem with a gated luminance-detail residual.

    Inheriting IIMStem preserves its parameter names and tensor shapes for warm-starting. The extra branch retains local
    shading that a channel-pair difference can cancel.
    """

    def __init__(self, c1, c2, k=3, s=2, kernel_nums=8, kernel_size=3, detail_scale=0.1):
        super().__init__(c1, c2, k, s, kernel_nums, kernel_size)
        self.luminance_detail = LuminanceDetail()
        self.detail_stem = nn.Sequential(Conv(5, c2, k, s), DWConv(c2, c2, 3))
        self.detail_gate = nn.Conv2d(2 * c2, 1, 1)
        self.detail_scale = nn.Parameter(torch.tensor(float(detail_scale)))
        nn.init.zeros_(self.detail_gate.weight)
        nn.init.zeros_(self.detail_gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = super().forward(x)
        detail = self.detail_stem(self.luminance_detail(x))
        gate = self.detail_gate(torch.cat((base, detail), dim=1)).sigmoid()
        return base + self.detail_scale.tanh() * gate * detail


class DentDetailFusion(nn.Module):
    """Reintroduce shallow P2/P3 features and signed local differences at the prediction head."""

    def __init__(self, ch: list[int] | tuple[int, int], c2: int, detail_scale: float = 0.1):
        super().__init__()
        if len(ch) != 2:
            raise ValueError("DentDetailFusion expects [neck_feature, shallow_feature].")
        self.base = nn.Identity() if ch[0] == c2 else Conv(ch[0], c2, 1)
        self.shallow = Conv(ch[1], c2, 1)
        self.refine = nn.Sequential(Conv(3 * c2, c2, 1), DWConv(c2, c2, 3))
        self.gate = nn.Conv2d(2 * c2, 1, 1)
        self.detail_scale = nn.Parameter(torch.tensor(float(detail_scale)))
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if len(x) != 2 or x[0].shape[-2:] != x[1].shape[-2:]:
            raise ValueError("DentDetailFusion requires two features at the same spatial resolution.")
        base, shallow = self.base(x[0]), self.shallow(x[1])
        with torch.autocast(device_type=shallow.device.type, enabled=False):
            signal = shallow.float()
            fine = signal - _local_mean(signal, 3)
            coarse = signal - _local_mean(signal, 7)
        detail = self.refine(torch.cat((shallow, fine.to(shallow.dtype), coarse.to(shallow.dtype)), dim=1))
        gate = self.gate(torch.cat((base, shallow), dim=1)).sigmoid()
        return base + self.detail_scale.tanh() * gate * detail


class Segment26MultiLabelShadow(Segment26MultiLabel):
    """Multi-label head with shallow detail bypasses and unchanged auxiliary class supervision.

    Input layout: [fused P3, P2, ..., matching class logits..., shallow P3, shallow P2]. Rust evidence is retained by
    the existing network; no gate requires rust for a dent prediction.
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
        if len(ch) < 6 or (len(ch) - 2) % 2:
            raise ValueError("Segment26MultiLabelShadow requires paired features/logits plus two shallow inputs.")
        super().__init__(nc, nm, npr, multilabel_gain, cooccurrence_weight, reg_max, end2end, ch[:-2])
        self.detail_fusion = nn.ModuleList(DentDetailFusion([ch[i], ch[-2 + i]], ch[i], detail_scale) for i in range(2))

    def forward(self, x):
        inputs = list(x[:-2])
        for i, fusion in enumerate(self.detail_fusion):
            inputs[i] = fusion([inputs[i], x[-2 + i]])
        return super().forward(inputs)
