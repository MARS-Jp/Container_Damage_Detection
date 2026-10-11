# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Adaptive RGB/IIM stem used only by the 0910 IIM experiment."""

from __future__ import annotations

import torch
from torch import nn

from .conv import Conv, IlluminationInvariantConv

__all__ = ("AdaptiveGatedIIMStem0910",)


class AdaptiveGatedIIMStem0910(nn.Module):
    """Fuse RGB and illumination-invariant features with a learned spatial gate.

    A zero-initialized gate starts at 0.5 everywhere. The factors of two make the initial forward pass equivalent in
    scale to the original concatenation stem while allowing darker regions to favor IIM features after learning.
    """

    def __init__(self, c1, c2, k=3, s=2, kernel_nums=8, kernel_size=3, gate_bias=0.0):
        """Build RGB/IIM branches with the same public arguments as ``IIMStem``."""
        super().__init__()
        if c1 != 3:
            raise ValueError(f"AdaptiveGatedIIMStem0910 requires RGB input with 3 channels, but received {c1}.")
        rgb_channels = c2 // 2
        iim_channels = c2 - rgb_channels
        self.rgb_stem = Conv(c1, rgb_channels, k, s)
        self.iim = IlluminationInvariantConv(kernel_nums, kernel_size)
        self.iim_stem = Conv(3 * kernel_nums, iim_channels, k, s)
        self.gate = nn.Conv2d(c2, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.fuse = Conv(c2, c2, 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_bias))

    def branch_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return RGB features, IIM features, and the learned IIM preference map."""
        rgb = self.rgb_stem(x)
        iim = self.iim_stem(self.iim(x))
        preference = self.gate(torch.cat((rgb, iim), dim=1)).sigmoid()
        return rgb, iim, preference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply complementary spatial weights and fuse both feature branches."""
        rgb, iim, preference = self.branch_features(x)
        balanced = torch.cat((2.0 * (1.0 - preference) * rgb, 2.0 * preference * iim), dim=1)
        return self.fuse(balanced)
