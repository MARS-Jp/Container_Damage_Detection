# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import LayerNorm2d, TransformerBlock

__all__ = (
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "CSAR",
    "DFL",
    "ELAN1",
    "MSAT",
    "PSA",
    "SCA",
    "SETA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "AConv",
    "ADown",
    "Attention",
    "AttentionResiduals",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ContrastiveHead",
    "CrossScaleAttention",
    "FSAttentionResiduals",
    "FSNetShuffle",
    "FeatureShuffle",
    "GhostBottleneck",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "MSATMultiLabel",
    "MultiStateCSAR",
    "PatchCSAR",
    "Proto",
    "Proto26MultiLabel",
    "RepC3",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "ScaleShuffle",
    "TorchVision",
)


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of pooling iterations.
            shortcut (bool): Whether to use shortcut connection.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """Initialize RepC3 module with RepConv blocks.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and addition to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 4 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """Forward pass of ImagePoolingAttn.

        Args:
            x (list[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    @staticmethod
    def forward_fuse(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes image features through unchanged after fusing."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (list[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx: list[int]):
        """Initialize CBFuse module.

        Args:
            idx (list[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through CBFuse layer.

        Args:
            xs (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize CSP bottleneck layer with three convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class AttentionResiduals2d(nn.Module):
    """Attention Residuals mixer for 2D feature maps.

    This adapts depth-wise Attention Residuals to CNN feature states with shape [B, C, H, W], applying a learned
    pseudo-query over previous states at each spatial location.
    """

    def __init__(self, c: int, eps: float = 1e-6):
        """Initialize the mixer.

        Args:
            c (int): Number of channels in each state.
            eps (float): Numerical stability term for RMS normalization.
        """
        super().__init__()
        self.query = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Mix previous states with learned softmax weights over depth."""
        if len(xs) == 1:
            return xs[0]
        v = torch.stack(xs, dim=0)
        k = v * torch.rsqrt(v.pow(2).mean(dim=2, keepdim=True) + self.eps)
        logits = torch.einsum("c,nbchw->nbhw", self.query, k)
        weights = logits.float().softmax(dim=0).to(dtype=v.dtype)
        return (weights.unsqueeze(2) * v).sum(dim=0)


class AttentionResiduals(nn.Module):
    """CNN block that directly replaces a residual block with Attention Residuals."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5, k: int = 3):
        """Initialize an Attention Residuals block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of feature transformation layers.
            e (float): Hidden channel expansion ratio.
            k (int): Kernel size for feature transformations.
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.blocks = nn.ModuleList(Conv(c_, c_, k, 1) for _ in range(n))
        self.attn_res = nn.ModuleList(AttentionResiduals2d(c_) for _ in range(n))
        self.out_attn_res = AttentionResiduals2d(c_)
        self.cv2 = Conv(c_, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with learned attention over previous feature states."""
        states = [self.cv1(x)]
        for mixer, block in zip(self.attn_res, self.blocks):
            states.append(block(mixer(states)))
        return self.cv2(self.out_attn_res(states))


class SETACore(nn.Module):
    """Scale-Equilibrium Transport Self-Attention for one 2D feature scale.

    The module conserves local evidence with windowed doubly-stochastic attention, models global semantics with a
    compact anchor grid, transports evidence in both directions between local tokens and anchors, and routes the three
    outputs with token-wise learned weights.
    """

    def __init__(
        self,
        c: int,
        num_heads: int = 4,
        window_size: int = 4,
        anchor_grid: int = 4,
        sinkhorn_iters: int = 3,
    ):
        """Initialize a SETA attention core.

        Args:
            c (int): Input and output channels.
            num_heads (int): Number of attention heads.
            window_size (int): Side length of each local attention window.
            anchor_grid (int): Side length of the pooled global anchor grid.
            sinkhorn_iters (int): Number of alternating Sinkhorn normalization iterations.
        """
        super().__init__()
        self.num_heads = max(1, min(int(num_heads), c))
        while c % self.num_heads:
            self.num_heads -= 1
        self.head_dim = c // self.num_heads
        self.scale = self.head_dim**-0.5
        self.window_size = max(int(window_size), 1)
        self.window_tokens = self.window_size**2
        self.anchor_grid = max(int(anchor_grid), 1)
        self.sinkhorn_iters = max(int(sinkhorn_iters), 1)

        self.local_qkv = nn.Conv2d(c, 3 * c, 1, bias=False)
        self.global_q = nn.Conv2d(c, c, 1, bias=False)
        self.transport_q = nn.Conv2d(c, c, 1, bias=False)
        self.anchor_kv = nn.Conv2d(c, 2 * c, 1, bias=False)
        self.updated_anchor_v = nn.Conv2d(c, c, 1, bias=False)
        self.proj = nn.Conv2d(c, c, 1, bias=False)

        relative_positions = (2 * self.window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(torch.zeros(relative_positions, self.num_heads))
        coords = torch.stack(
            torch.meshgrid(torch.arange(self.window_size), torch.arange(self.window_size), indexing="ij")
        ).flatten(1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1), persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        router_hidden = max(c // 8, 8)
        self.router_norm = nn.LayerNorm(5)
        self.router = nn.Sequential(
            nn.Linear(5, router_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(router_hidden, 3),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    @staticmethod
    def _square_sinkhorn(logits: torch.Tensor, iterations: int) -> torch.Tensor:
        """Normalize square local attention matrices toward doubly-stochastic plans."""
        log_plan = logits.float()
        for _ in range(iterations):
            log_plan = log_plan - torch.logsumexp(log_plan, dim=-1, keepdim=True)
            log_plan = log_plan - torch.logsumexp(log_plan, dim=-2, keepdim=True)
        log_plan = log_plan - torch.logsumexp(log_plan, dim=-1, keepdim=True)
        return log_plan.exp().to(dtype=logits.dtype)

    @staticmethod
    def _rectangular_sinkhorn(logits: torch.Tensor, iterations: int) -> torch.Tensor:
        """Normalize an N-by-M transport plan with row mass 1 and column mass N/M.

        The target column mass is derived from the mean column mass after every
        row normalization, avoiding the inconsistent row=1 and column=1 constraints
        when the token and anchor counts differ.
        """
        log_plan = logits.float()
        for _ in range(iterations):
            log_plan = log_plan - torch.logsumexp(log_plan, dim=-1, keepdim=True)
            log_column_mass = torch.logsumexp(log_plan, dim=-2, keepdim=True)
            log_target_mass = log_column_mass.exp().mean(dim=-1, keepdim=True).clamp_min(1e-8).log()
            log_plan = log_plan - log_column_mass + log_target_mass
        return log_plan.exp().to(dtype=logits.dtype)

    @staticmethod
    def _window_partition(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, int, int]:
        """Partition [B, heads, dim, H, W] features into flattened windows."""
        b, heads, dim, h, w = x.shape
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = h + pad_h, w + pad_w
        x = x.view(b, heads, dim, hp // window_size, window_size, wp // window_size, window_size)
        x = x.permute(0, 3, 5, 1, 4, 6, 2).contiguous()
        return x.view(-1, heads, window_size**2, dim), pad_h, pad_w

    @staticmethod
    def _window_reverse(
        windows: torch.Tensor,
        batch: int,
        height: int,
        width: int,
        window_size: int,
        pad_h: int,
        pad_w: int,
    ) -> torch.Tensor:
        """Reverse flattened windows to a [B, heads, dim, H, W] feature map."""
        heads, dim = windows.shape[1], windows.shape[-1]
        hp, wp = height + pad_h, width + pad_w
        windows = windows.view(
            batch,
            hp // window_size,
            wp // window_size,
            heads,
            window_size,
            window_size,
            dim,
        )
        x = windows.permute(0, 3, 6, 1, 4, 2, 5).contiguous().view(batch, heads, dim, hp, wp)
        return x[..., :height, :width]

    @staticmethod
    def _valid_window_tokens(
        reference: torch.Tensor,
        batch: int,
        height: int,
        width: int,
        window_size: int,
        pad_h: int,
        pad_w: int,
    ) -> torch.Tensor:
        """Return a validity mask for tokens in padded local windows."""
        valid = reference.new_ones((1, 1, height, width))
        if pad_h or pad_w:
            valid = F.pad(valid, (0, pad_w, 0, pad_h), value=0)
        hp, wp = height + pad_h, width + pad_w
        valid = valid.view(1, 1, hp // window_size, window_size, wp // window_size, window_size)
        valid = valid.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, window_size**2)
        return valid.bool().repeat(batch, 1)

    def _local_attention(self, local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply local window Sinkhorn attention and return output, entropy, and local values."""
        b, _c, h, w = local.shape
        qkv = self.local_qkv(local).view(b, 3, self.num_heads, self.head_dim, h, w)
        q, k, value_map = qkv.unbind(dim=1)
        q, pad_h, pad_w = self._window_partition(q, self.window_size)
        k, _, _ = self._window_partition(k, self.window_size)
        value, _, _ = self._window_partition(value_map, self.window_size)

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.view(self.window_tokens, self.window_tokens, self.num_heads).permute(2, 0, 1)
        logits = logits + bias.unsqueeze(0).to(dtype=logits.dtype)

        valid = self._valid_window_tokens(local, b, h, w, self.window_size, pad_h, pad_w)
        valid_pair = valid[:, None, :, None] & valid[:, None, None, :]
        invalid = ~valid
        identity = torch.eye(self.window_tokens, dtype=torch.bool, device=local.device)[None, None]
        dummy_identity = invalid[:, None, :, None] & invalid[:, None, None, :] & identity
        logits = logits.masked_fill(~valid_pair, float("-inf"))
        logits = torch.where(dummy_identity, torch.zeros((), dtype=logits.dtype, device=logits.device), logits)

        attention = self._square_sinkhorn(logits, self.sinkhorn_iters)
        output = torch.matmul(attention, value)
        output = self._window_reverse(output, b, h, w, self.window_size, pad_h, pad_w)

        entropy = attention.float().clamp_min(1e-8)
        entropy = -(entropy * entropy.log()).sum(dim=-1) / torch.log(
            entropy.new_tensor(float(max(self.window_tokens, 2)))
        )
        entropy = entropy.mean(dim=1, keepdim=True).unsqueeze(-1)
        entropy = self._window_reverse(entropy, b, h, w, self.window_size, pad_h, pad_w)[:, 0, 0]
        return output, entropy, value_map

    def _as_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [B, C, H, W] into [B, heads, HW, head_dim]."""
        b = x.shape[0]
        return x.reshape(b, self.num_heads, self.head_dim, -1).transpose(-2, -1)

    def _as_map(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Convert [B, heads, HW, head_dim] into [B, C, H, W]."""
        b = x.shape[0]
        return x.transpose(-2, -1).contiguous().view(b, self.num_heads * self.head_dim, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local/global attention, bidirectional transport, and scale routing."""
        b, _, h, w = x.shape
        anchor_map = F.adaptive_avg_pool2d(x, (self.anchor_grid, self.anchor_grid))
        global_map = F.interpolate(anchor_map, size=(h, w), mode="bilinear", align_corners=False)
        local_map = x - global_map

        z_local_map, h_local_map, local_value_map = self._local_attention(local_map)
        z_local = self._as_tokens(z_local_map)
        local_values = self._as_tokens(local_value_map)

        global_query = self._as_tokens(self.global_q(global_map))
        transport_query = self._as_tokens(self.transport_q(local_map))
        anchor_kv = self.anchor_kv(anchor_map).view(
            b, 2, self.num_heads, self.head_dim, self.anchor_grid, self.anchor_grid
        )
        anchor_key, anchor_value = anchor_kv.unbind(dim=1)
        anchor_key = anchor_key.flatten(-2).transpose(-2, -1)
        anchor_value = anchor_value.flatten(-2).transpose(-2, -1)

        global_logits = torch.matmul(global_query, anchor_key.transpose(-2, -1)) * self.scale
        global_attention = global_logits.float().softmax(dim=-1).to(dtype=global_logits.dtype)
        z_global = torch.matmul(global_attention, anchor_value)

        transport_logits = torch.matmul(transport_query, anchor_key.transpose(-2, -1)) * self.scale
        transport = self._rectangular_sinkhorn(transport_logits, self.sinkhorn_iters)
        row_transport = transport / transport.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        column_mass = transport.sum(dim=-2).unsqueeze(-1).clamp_min(1e-6)
        anchor_evidence = torch.matmul(transport.transpose(-2, -1), local_values) / column_mass
        anchor_state = self._as_tokens(anchor_map)
        updated_anchor = anchor_state + anchor_evidence
        updated_anchor_map = self._as_map(updated_anchor, self.anchor_grid, self.anchor_grid)
        updated_anchor_value = self._as_tokens(self.updated_anchor_v(updated_anchor_map))
        delta_local = torch.matmul(row_transport, updated_anchor_value)

        local_float = local_map.float()
        local_energy = (local_float.square().mean(dim=1) + 1e-6).sqrt().flatten(1)
        delta_x = F.pad((local_float[..., 1:] - local_float[..., :-1]).abs(), (0, 1, 0, 0))
        delta_y = F.pad((local_float[..., 1:, :] - local_float[..., :-1, :]).abs(), (0, 0, 0, 1))
        variation = (delta_x + delta_y).mean(dim=1).flatten(1)
        h_local = h_local_map.flatten(1).float()

        global_probability = global_attention.float().clamp_min(1e-8)
        h_global = -(global_probability * global_probability.log()).sum(dim=-1)
        h_global = h_global.mean(dim=1) / torch.log(global_probability.new_tensor(float(max(anchor_key.shape[-2], 2))))
        transport_probability = row_transport.float().clamp_min(1e-8)
        h_transport = -(transport_probability * transport_probability.log()).sum(dim=-1)
        h_transport = h_transport.mean(dim=1) / torch.log(
            transport_probability.new_tensor(float(max(anchor_key.shape[-2], 2)))
        )
        confidence = 1.0 - h_transport

        descriptor = torch.stack((local_energy, h_local, h_global, variation, confidence), dim=-1)
        route = self.router(self.router_norm(descriptor)).softmax(dim=-1).to(dtype=z_local.dtype)
        output = (
            route[:, None, :, 0, None] * z_local
            + route[:, None, :, 1, None] * z_global
            + route[:, None, :, 2, None] * delta_local
        )
        return self.proj(self._as_map(output, h, w))


class SETALayer(nn.Module):
    """Residual SETA attention and convolutional feed-forward layer."""

    def __init__(
        self,
        c: int,
        num_heads: int = 4,
        window_size: int = 4,
        anchor_grid: int = 4,
        sinkhorn_iters: int = 3,
        ffn_ratio: float = 2.0,
        layer_scale_init: float = 1e-3,
    ):
        """Initialize one residual SETA layer."""
        super().__init__()
        ffn_channels = max(int(c * ffn_ratio), c)
        self.norm1 = LayerNorm2d(c)
        self.attn = SETACore(c, num_heads, window_size, anchor_grid, sinkhorn_iters)
        self.norm2 = LayerNorm2d(c)
        self.ffn = nn.Sequential(
            nn.Conv2d(c, ffn_channels, 1),
            nn.GELU(),
            nn.Conv2d(ffn_channels, ffn_channels, 3, padding=1, groups=ffn_channels),
            nn.GELU(),
            nn.Conv2d(ffn_channels, c, 1),
        )
        self.gamma1 = nn.Parameter(torch.full((1, c, 1, 1), float(layer_scale_init)))
        self.gamma2 = nn.Parameter(torch.full((1, c, 1, 1), float(layer_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual transport attention followed by a residual FFN."""
        x = x + self.gamma1 * self.attn(self.norm1(x))
        return x + self.gamma2 * self.ffn(self.norm2(x))


class SETA(nn.Module):
    """YOLO block that replaces AttentionResiduals with scale-equilibrium transport attention."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        e: float = 0.5,
        num_heads: int = 4,
        window_size: int = 4,
        anchor_grid: int = 4,
        sinkhorn_iters: int = 3,
        ffn_ratio: float = 2.0,
    ):
        """Initialize a repeatable SETA block with the AttentionResiduals YAML interface."""
        super().__init__()
        hidden = max(int(c2 * e), 1)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.layers = nn.Sequential(
            *(
                SETALayer(hidden, num_heads, window_size, anchor_grid, sinkhorn_iters, ffn_ratio)
                for _ in range(max(int(n), 1))
            )
        )
        self.cv2 = Conv(hidden, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project, refine with SETA layers, and restore the requested output channels."""
        return self.cv2(self.layers(self.cv1(x)))


class FeatureShuffle(nn.Module):
    """Feature shuffle mixer with a local branch and a down-up context branch."""

    def __init__(self, c: int, groups: int = 2):
        """Initialize feature shuffle mixer.

        Args:
            c (int): Number of input/output channels.
            groups (int): Number of channel shuffle groups.
        """
        super().__init__()
        self.c1 = c - c // 2
        self.c2 = c // 2
        self.groups = groups
        self.context = Conv(self.c2, self.c2, 3, 1) if self.c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix local and context features, then shuffle channels."""
        if self.c2 == 0:
            return x
        local, context = x.split((self.c1, self.c2), dim=1)
        if min(context.shape[-2:]) > 1:
            context = F.avg_pool2d(context, kernel_size=2, stride=2, ceil_mode=True)
            context = self.context(context)
            context = F.interpolate(context, size=local.shape[-2:], mode="nearest")
        else:
            context = self.context(context)
        return self.channel_shuffle(torch.cat((local, context), dim=1), self.groups)

    @staticmethod
    def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
        """Shuffle channels across groups."""
        b, c, h, w = x.shape
        if groups <= 1 or c % groups:
            return x
        x = x.view(b, groups, c // groups, h, w)
        return x.transpose(1, 2).contiguous().view(b, c, h, w)


class ScaleShuffle(FeatureShuffle):
    """Backward-compatible name for checkpoints saved with the old ScaleShuffle class."""

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor):
        """Fuse old multi-scale list inputs while retaining FeatureShuffle tensor behavior."""
        if isinstance(x, (list, tuple)):
            xs = list(x)
            size = xs[0].shape[-2:]
            xs = [xi if xi.shape[-2:] == size else F.interpolate(xi, size=size, mode="nearest") for xi in xs]

            def fit_channels(t: torch.Tensor, channels: int) -> torch.Tensor:
                if t.shape[1] == channels:
                    return t
                if t.shape[1] > channels:
                    return t[:, :channels]
                pad = t.new_zeros(t.shape[0], channels - t.shape[1], *t.shape[2:])
                return torch.cat((t, pad), dim=1)

            local = fit_channels(xs[0], self.c1)
            context_source = torch.cat(xs[1:], dim=1) if len(xs) > 1 else xs[0][:, self.c1 :]
            context = fit_channels(context_source, self.c2)
            return self.channel_shuffle(torch.cat((local, context), dim=1), self.groups)
        return super().forward(x)


class FSNetShuffle(nn.Module):
    """Multi-input FSNet-style shuffle layer for exchanging features across scales.

    The layer returns one target-resolution tensor. Use multiple YAML rows with different target indices when several
    shuffled scale outputs are needed.
    """

    def __init__(self, ch: list[int], c2: int, target: int = 0, groups: int = 2, k: int = 3, refine: bool = True):
        """Initialize the multi-scale shuffle layer.

        Args:
            ch (list[int]): Input channel dimensions for each source feature.
            c2 (int): Output channels.
            target (int): Input index whose spatial size is used for the output.
            groups (int): Number of channel-shuffle groups.
            k (int): Kernel size for the optional output refinement convolution.
            refine (bool): Whether to refine the shuffled output with a convolution.
        """
        super().__init__()
        if not ch:
            raise ValueError("FSNetShuffle requires at least one input feature map.")
        self.target = target % len(ch)
        self.groups = groups

        base = c2 // len(ch)
        splits = [base] * len(ch)
        for i in range(c2 - base * len(ch)):
            splits[i] += 1

        self.proj = nn.ModuleList(Conv(c, out_c, 1, 1) for c, out_c in zip(ch, splits))
        self.refine = Conv(c2, c2, k, 1) if refine else nn.Identity()

    @staticmethod
    def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Resize a feature map to the target spatial size."""
        if x.shape[-2:] == size:
            return x
        if x.shape[-2] >= size[0] and x.shape[-1] >= size[1]:
            return F.adaptive_avg_pool2d(x, size)
        return F.interpolate(x, size=size, mode="nearest")

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor) -> torch.Tensor:
        """Shuffle features from multiple scales into the target scale."""
        xs = [x] if isinstance(x, torch.Tensor) else list(x)
        if len(xs) != len(self.proj):
            raise ValueError(f"FSNetShuffle expected {len(self.proj)} inputs, but received {len(xs)}.")
        size = xs[self.target].shape[-2:]
        parts = [self._resize(proj(xi), size) for proj, xi in zip(self.proj, xs)]
        return self.refine(FeatureShuffle.channel_shuffle(torch.cat(parts, dim=1), self.groups))


class FSAttentionResiduals(nn.Module):
    """Attention Residuals block with FSNet-style feature shuffle before each transform."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5, k: int = 3, groups: int = 2):
        """Initialize an FSNet-enhanced Attention Residuals block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of feature transformation layers.
            e (float): Hidden channel expansion ratio.
            k (int): Kernel size for feature transformations.
            groups (int): Number of channel shuffle groups.
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.shuffle = nn.ModuleList(FeatureShuffle(c_, groups) for _ in range(n))
        self.blocks = nn.ModuleList(Conv(c_, c_, k, 1) for _ in range(n))
        self.attn_res = nn.ModuleList(AttentionResiduals2d(c_) for _ in range(n))
        self.out_shuffle = FeatureShuffle(c_, groups)
        self.out_attn_res = AttentionResiduals2d(c_)
        self.cv2 = Conv(c_, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with feature shuffle and learned attention over feature states."""
        states = [self.cv1(x)]
        for shuffle, mixer, block in zip(self.shuffle, self.attn_res, self.blocks):
            states.append(block(shuffle(mixer(states))))
        return self.cv2(self.out_shuffle(self.out_attn_res(states)))


class SCA(nn.Module):
    """Scale-aware Channel Attention for multi-scale YOLO feature maps.

    This module adapts the COP-Net SCA idea to the Ultralytics YAML graph. It receives multiple feature maps, aligns
    them to a target scale, computes a target-guided channel attention vector, and fuses the reweighted features into
    one refined target-resolution output.
    """

    def __init__(
        self,
        ch: list[int],
        c2: int,
        target: int = 0,
        reduction: int = 4,
        shortcut: bool = True,
    ):
        """Initialize SCA.

        Args:
            ch (list[int]): Input channel dimensions for each source feature.
            c2 (int): Output channels.
            target (int): Source index used for output resolution and channel guidance.
            reduction (int): Channel reduction factor in the attention MLP.
            shortcut (bool): Whether to add the target feature shortcut.
        """
        super().__init__()
        if not ch:
            raise ValueError("SCA requires at least one input feature map.")
        self.target = target % len(ch)
        hidden = max(c2 // max(int(reduction), 1), 8)
        self.proj = nn.ModuleList(Conv(c, c2, 1, 1) for c in ch)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = Conv(c2 * len(ch), c2, 3, 1)
        self.shortcut = (
            nn.Identity()
            if shortcut and ch[self.target] == c2
            else Conv(ch[self.target], c2, 1, act=False)
            if shortcut
            else None
        )

    @staticmethod
    def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Resize a feature map to the target spatial size."""
        return x if x.shape[-2:] == size else F.interpolate(x, size=size, mode="nearest")

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor) -> torch.Tensor:
        """Fuse multi-scale inputs using target-guided channel attention."""
        xs = [x] if isinstance(x, torch.Tensor) else list(x)
        if len(xs) != len(self.proj):
            raise ValueError(f"SCA expected {len(self.proj)} inputs, but received {len(xs)}.")
        ref = xs[self.target]
        size = ref.shape[-2:]
        aligned = [self._resize(proj(xi), size) for proj, xi in zip(self.proj, xs)]
        channel_weight = self.attn(aligned[self.target])
        y = self.fuse(torch.cat([xi * channel_weight for xi in aligned], dim=1))
        return y + self.shortcut(ref) if self.shortcut is not None else y


class CSAR(nn.Module):
    """Cross-Scale Attention Residual fusion for YOLO feature maps.

    This module receives a list of feature maps, aligns them to a target feature resolution, builds 1x1-conv
    query/key/value projections, attends over the scale dimension, and adds a residual shortcut from the target feature.
    """

    def __init__(
        self,
        ch: list[int],
        c2: int,
        num_heads: int = 4,
        target: int = -1,
        attn_ratio: float = 0.5,
        shortcut: bool = True,
    ):
        """Initialize CSAR.

        Args:
            ch (list[int]): Input channel dimensions for each source feature.
            c2 (int): Output channels.
            num_heads (int): Number of attention heads.
            target (int): Source index used as query, output size, and residual.
            attn_ratio (float): Key/query dimension ratio relative to each value head.
            shortcut (bool): Whether to add the target feature shortcut.
        """
        super().__init__()
        if not ch:
            raise ValueError("CSAR requires at least one input feature map.")
        self.target = target % len(ch)
        self.num_heads = max(1, min(int(num_heads), c2))
        while c2 % self.num_heads:
            self.num_heads -= 1
        self.head_dim = c2 // self.num_heads
        self.key_dim = max(1, int(self.head_dim * attn_ratio))
        self.scale = self.key_dim**-0.5

        key_channels = self.key_dim * self.num_heads
        self.q = Conv(ch[self.target], key_channels, 1, act=False)
        self.k = nn.ModuleList(Conv(c, key_channels, 1, act=False) for c in ch)
        self.v = nn.ModuleList(Conv(c, c2, 1, act=False) for c in ch)
        self.pe = Conv(c2, c2, 3, 1, g=c2, act=False)
        self.proj = Conv(c2, c2, 1, act=False)
        self.shortcut = (
            nn.Identity()
            if shortcut and ch[self.target] == c2
            else Conv(ch[self.target], c2, 1, act=False)
            if shortcut
            else None
        )

    @staticmethod
    def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Resize a feature map to the target spatial size."""
        return x if x.shape[-2:] == size else F.interpolate(x, size=size, mode="nearest")

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor) -> torch.Tensor:
        """Fuse multi-scale features with attention over source scales."""
        xs = [x] if isinstance(x, torch.Tensor) else list(x)
        if len(xs) != len(self.k):
            raise ValueError(f"CSAR expected {len(self.k)} inputs, but received {len(xs)}.")
        ref = xs[self.target]
        b, _, h, w = ref.shape
        size = (h, w)
        aligned = [self._resize(xi, size) for xi in xs]

        q = self.q(ref).view(b, self.num_heads, self.key_dim, h, w)
        k = torch.stack(
            [proj(xi).view(b, self.num_heads, self.key_dim, h, w) for proj, xi in zip(self.k, aligned)], dim=0
        )
        v = torch.stack(
            [proj(xi).view(b, self.num_heads, self.head_dim, h, w) for proj, xi in zip(self.v, aligned)], dim=0
        )

        attn = (q.unsqueeze(0) * k).sum(dim=3) * self.scale
        attn = attn.float().softmax(dim=0).to(dtype=v.dtype)
        y = (attn.unsqueeze(3) * v).sum(dim=0).reshape(b, self.num_heads * self.head_dim, h, w)
        y = self.proj(y + self.pe(v[self.target].reshape(b, self.num_heads * self.head_dim, h, w)))
        return y + self.shortcut(ref) if self.shortcut is not None else y


class MultiStateCSAR(CSAR):
    """CSAR that keeps each input scale as a token state before learned state interaction.

    Each aligned source feature is treated as a different state of the same spatial token. A small MLP mixes the
    explicit state axis, while a zero-initialized residual gate makes the module start with standard CSAR behavior. The
    mixed states are aggregated only after query-key attention has assigned a content-dependent weight to each state.
    """

    def __init__(
        self,
        ch: list[int],
        c2: int,
        num_heads: int = 4,
        target: int = -1,
        attn_ratio: float = 0.5,
        shortcut: bool = True,
        state_expansion: float = 2.0,
    ):
        """Initialize multi-state cross-scale attention.

        Args:
            ch (list[int]): Input channel dimensions; every input represents one token state.
            c2 (int): Output channels.
            num_heads (int): Number of attention heads.
            target (int): State used for the query, output resolution, positional encoding, and residual.
            attn_ratio (float): Key/query dimension ratio relative to each value head.
            shortcut (bool): Whether to add a shortcut from the target state.
            state_expansion (float): Hidden expansion ratio of the state-axis MLP.
        """
        super().__init__(ch, c2, num_heads, target, attn_ratio, shortcut)
        self.num_states = len(ch)
        hidden_states = max(self.num_states, round(self.num_states * state_expansion))
        self.state_mixer = nn.Sequential(
            nn.Linear(self.num_states, hidden_states, bias=False),
            nn.GELU(),
            nn.Linear(hidden_states, self.num_states, bias=False),
        )
        # ReZero-style per-head gates preserve ordinary CSAR behavior at initialization.
        self.state_gate = nn.Parameter(torch.zeros(self.num_heads))
        self.state_bias = nn.Parameter(torch.zeros(self.num_states, self.num_heads))

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor) -> torch.Tensor:
        """Interact over the explicit state axis, then aggregate states into the target feature map."""
        xs = [x] if isinstance(x, torch.Tensor) else list(x)
        if len(xs) != self.num_states:
            raise ValueError(f"MultiStateCSAR expected {self.num_states} inputs, but received {len(xs)}.")
        ref = xs[self.target]
        b, _, h, w = ref.shape
        aligned = [self._resize(xi, (h, w)) for xi in xs]

        q = self.q(ref).view(b, self.num_heads, self.key_dim, h, w)
        k = torch.stack(
            [proj(xi).view(b, self.num_heads, self.key_dim, h, w) for proj, xi in zip(self.k, aligned)], dim=0
        )
        states = torch.stack(
            [proj(xi).view(b, self.num_heads, self.head_dim, h, w) for proj, xi in zip(self.v, aligned)], dim=0
        )

        # [state, batch, head, channel, height, width] -> [..., state] for the shared state-axis MLP.
        state_tokens = states.permute(1, 2, 3, 4, 5, 0)
        mixed_tokens = self.state_mixer(state_tokens)
        gate = torch.tanh(self.state_gate).view(1, self.num_heads, 1, 1, 1, 1)
        state_tokens = state_tokens + gate * mixed_tokens
        states = state_tokens.permute(5, 0, 1, 2, 3, 4)

        logits = (q.unsqueeze(0) * k).sum(dim=3) * self.scale
        logits = logits + self.state_bias[:, None, :, None, None]
        weights = logits.float().softmax(dim=0).to(dtype=states.dtype)
        y = (weights.unsqueeze(3) * states).sum(dim=0).reshape(b, self.num_heads * self.head_dim, h, w)
        target_state = states[self.target].reshape(b, self.num_heads * self.head_dim, h, w)
        y = self.proj(y + self.pe(target_state))
        return y + self.shortcut(ref) if self.shortcut is not None else y


class _MSATStateLayer(nn.Module):
    """Pre-normalized self-attention and FFN over the state axis."""

    def __init__(self, channels: int, num_heads: int, ffn_ratio: float, layer_scale_init: float):
        """Initialize a state-axis Transformer layer."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5
        hidden_channels = max(channels, round(channels * ffn_ratio))
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )
        self.gamma1 = nn.Parameter(torch.full((channels,), float(layer_scale_init)))
        self.gamma2 = nn.Parameter(torch.full((channels,), float(layer_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention independently to the states of every spatial token."""
        b, h, w, num_states, channels = x.shape
        state_tokens = self.norm1(x).reshape(b * h * w, num_states, channels)
        qkv = state_tokens.reshape(-1, num_states, channels)
        qkv = self.qkv(qkv).view(-1, num_states, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.float().softmax(dim=-1).to(dtype=v.dtype)
        output = (attention @ v).transpose(1, 2).reshape(-1, num_states, channels)
        output = self.proj(output).view(b, h, w, num_states, channels)
        x = x + self.gamma1 * output
        return x + self.gamma2 * self.ffn(self.norm2(x))


class _MSATWindowSpatialLayer(nn.Module):
    """Pre-normalized window self-attention and FFN over spatial tokens."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int,
        ffn_ratio: float,
        layer_scale_init: float,
    ):
        """Initialize a spatial window Transformer layer."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5
        self.window_size = max(1, int(window_size))
        self.window_tokens = self.window_size**2
        hidden_channels = max(channels, round(channels * ffn_ratio))

        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )
        self.gamma1 = nn.Parameter(torch.full((channels,), float(layer_scale_init)))
        self.gamma2 = nn.Parameter(torch.full((channels,), float(layer_scale_init)))

        relative_positions = (2 * self.window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(torch.zeros(relative_positions, self.num_heads))
        coords = torch.stack(
            torch.meshgrid(torch.arange(self.window_size), torch.arange(self.window_size), indexing="ij")
        ).flatten(1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1), persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _partition(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Partition channels-last feature maps into flattened spatial windows."""
        batch, height, width, channels = x.shape
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        padded_h, padded_w = height + pad_h, width + pad_w
        x = x.view(
            batch,
            padded_h // self.window_size,
            self.window_size,
            padded_w // self.window_size,
            self.window_size,
            channels,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(-1, self.window_tokens, channels), pad_h, pad_w

    def _reverse(
        self,
        windows: torch.Tensor,
        batch: int,
        height: int,
        width: int,
        pad_h: int,
        pad_w: int,
    ) -> torch.Tensor:
        """Reverse flattened windows to channels-last feature maps."""
        channels = windows.shape[-1]
        padded_h, padded_w = height + pad_h, width + pad_w
        x = windows.view(
            batch,
            padded_h // self.window_size,
            padded_w // self.window_size,
            self.window_size,
            self.window_size,
            channels,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, padded_h, padded_w, channels)
        return x[:, :height, :width]

    def _valid_window_tokens(
        self,
        reference: torch.Tensor,
        batch: int,
        height: int,
        width: int,
        pad_h: int,
        pad_w: int,
    ) -> torch.Tensor:
        """Return the valid key positions for every possibly padded window."""
        valid = reference.new_ones((1, height, width, 1))
        if pad_h or pad_w:
            valid = F.pad(valid, (0, 0, 0, pad_w, 0, pad_h), value=0)
        padded_h, padded_w = height + pad_h, width + pad_w
        valid = valid.view(
            1,
            padded_h // self.window_size,
            self.window_size,
            padded_w // self.window_size,
            self.window_size,
            1,
        )
        valid = valid.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_tokens)
        return valid.bool().repeat(batch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local spatial attention independently to every state."""
        b, h, w, num_states, channels = x.shape
        spatial_tokens = self.norm1(x).permute(0, 3, 1, 2, 4).reshape(b * num_states, h, w, channels)
        windows, pad_h, pad_w = self._partition(spatial_tokens)
        qkv = self.qkv(windows).view(-1, self.window_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

        attention = (q @ k.transpose(-2, -1)) * self.scale
        relative_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        relative_bias = relative_bias.view(self.window_tokens, self.window_tokens, self.num_heads)
        relative_bias = relative_bias.permute(2, 0, 1).to(dtype=attention.dtype)
        attention = attention + relative_bias.unsqueeze(0)

        valid = self._valid_window_tokens(spatial_tokens, b * num_states, h, w, pad_h, pad_w)
        attention = attention.masked_fill(~valid[:, None, None, :], torch.finfo(attention.dtype).min)
        attention = attention.float().softmax(dim=-1).to(dtype=v.dtype)
        output = (attention @ v).transpose(1, 2).reshape(-1, self.window_tokens, channels)
        output = self.proj(output)
        output = self._reverse(output, b * num_states, h, w, pad_h, pad_w)
        output = output.view(b, num_states, h, w, channels).permute(0, 2, 3, 1, 4).contiguous()

        x = x + self.gamma1 * output
        return x + self.gamma2 * self.ffn(self.norm2(x))


class _MSATStatePool(nn.Module):
    """Pool multiple states into one target feature using cross-attention."""

    def __init__(self, channels: int, num_heads: int, target: int):
        """Initialize target-query attention pooling."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5
        self.target = target
        self.norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(channels, 2 * channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use the target state to attend to all states at each spatial location."""
        b, h, w, num_states, channels = x.shape
        states = self.norm(x).reshape(b * h * w, num_states, channels)
        query = self.q(states[:, self.target : self.target + 1])
        key, value = self.kv(states).chunk(2, dim=-1)
        query = query.view(-1, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(-1, num_states, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(-1, num_states, self.num_heads, self.head_dim).transpose(1, 2)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.float().softmax(dim=-1).to(dtype=value.dtype)
        pooled = (attention @ value).transpose(1, 2).reshape(-1, channels)
        pooled = self.proj(pooled).view(b, h, w, channels)
        return x[..., self.target, :] + pooled


class MSAT(nn.Module):
    """Multi-State Axial Transformer with state-first and windowed-spatial self-attention.

    Input feature scales are aligned to the target resolution and retained as explicit states of each spatial token.
    State MHSA first exchanges information across scales at the same location. Windowed spatial MHSA then exchanges
    local evidence between locations while preserving the state axis. Target-query attention pools the states only after
    both Transformer axes have completed.
    """

    def __init__(
        self,
        ch: list[int],
        c2: int,
        num_heads: int = 4,
        target: int = -1,
        embed_channels: int = 0,
        ffn_ratio: float = 2.0,
        window_size: int = 8,
        shortcut: bool = True,
        layer_scale_init: float = 1e-3,
    ):
        """Initialize MSAT V1.

        Args:
            ch (list[int]): Input channels; each input is one scale state.
            c2 (int): Output channels.
            num_heads (int): Number of state and spatial attention heads.
            target (int): Input state defining output resolution and residual.
            embed_channels (int): Internal Transformer width, or 0 to use min(c2, 256).
            ffn_ratio (float): Hidden expansion ratio for Transformer FFNs.
            window_size (int): Side length of each local spatial attention window.
            shortcut (bool): Whether to add a residual from the target input.
            layer_scale_init (float): Initial residual scale inside Transformer layers.
        """
        super().__init__()
        if not ch:
            raise ValueError("MSAT requires at least one input feature map.")
        self.num_states = len(ch)
        self.target = int(target) % self.num_states
        channels = int(embed_channels) if int(embed_channels) > 0 else min(int(c2), 256)
        heads = max(1, min(int(num_heads), channels))
        while channels % heads:
            heads -= 1
        self.embed_channels = channels
        self.num_heads = heads

        self.input_proj = nn.ModuleList(
            nn.Sequential(
                Conv(input_channels, channels, 1),
                Conv(channels, channels, 3, g=channels),
            )
            for input_channels in ch
        )
        self.state_embedding = nn.Parameter(torch.zeros(1, 1, 1, self.num_states, channels))
        nn.init.trunc_normal_(self.state_embedding, std=0.02)
        self.state_layer = _MSATStateLayer(channels, heads, ffn_ratio, layer_scale_init)
        self.spatial_layer = _MSATWindowSpatialLayer(
            channels,
            heads,
            window_size,
            ffn_ratio,
            layer_scale_init,
        )
        self.state_pool = _MSATStatePool(channels, heads, self.target)
        self.pe = Conv(channels, channels, 3, g=channels, act=False)
        self.proj = Conv(channels, c2, 1, act=False)
        self.shortcut = (
            nn.Identity()
            if shortcut and ch[self.target] == c2
            else Conv(ch[self.target], c2, 1, act=False)
            if shortcut
            else None
        )
        self.output_scale = nn.Parameter(torch.ones(())) if shortcut else None

    @staticmethod
    def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Resize a state with interpolation suited to its scale direction."""
        if x.shape[-2:] == size:
            return x
        mode = "bilinear" if x.shape[-2] < size[0] or x.shape[-1] < size[1] else "area"
        return F.interpolate(x, size=size, mode=mode, align_corners=False if mode == "bilinear" else None)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor) -> torch.Tensor:
        """Apply state MHSA, windowed spatial MHSA, and state-aware pooling."""
        xs = [x] if isinstance(x, torch.Tensor) else list(x)
        if len(xs) != self.num_states:
            raise ValueError(f"MSAT expected {self.num_states} inputs, but received {len(xs)}.")
        ref = xs[self.target]
        size = ref.shape[-2:]
        states = [projection(self._resize(feature, size)) for projection, feature in zip(self.input_proj, xs)]
        tokens = torch.stack(states, dim=1).permute(0, 3, 4, 1, 2).contiguous()
        tokens = tokens + self.state_embedding
        tokens = self.state_layer(tokens)
        tokens = self.spatial_layer(tokens)
        pooled = self.state_pool(tokens).permute(0, 3, 1, 2).contiguous()
        target_state = tokens[..., self.target, :].permute(0, 3, 1, 2).contiguous()
        output = self.proj(pooled + self.pe(target_state))
        if self.shortcut is None:
            return output
        return self.shortcut(ref) + self.output_scale * output


class _MSATMultiLabelAuxHead(nn.Module):
    """Class-conditioned state readout for independent per-pixel damage labels."""

    def __init__(self, channels: int, num_heads: int, num_classes: int):
        """Initialize class queries and state-evidence projections."""
        super().__init__()
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5
        self.norm = nn.LayerNorm(channels)
        self.key = nn.Linear(channels, channels)
        self.evidence = nn.Linear(channels, num_classes)
        self.class_queries = nn.Parameter(torch.empty(num_heads, num_classes, self.head_dim))
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.trunc_normal_(self.class_queries, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict independent class logits by attending each class query over the state axis."""
        b, h, w, num_states, channels = x.shape
        states = self.norm(x).reshape(b * h * w, num_states, channels)
        keys = self.key(states).view(-1, num_states, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attention = torch.einsum("hcd,nhsd->nhcs", self.class_queries, keys) * self.scale
        attention = attention.float().softmax(dim=-1).to(dtype=states.dtype).mean(dim=1)
        evidence = self.evidence(states).transpose(1, 2)
        logits = (attention * evidence).sum(dim=-1) + self.class_bias
        return logits.view(b, h, w, self.num_classes).permute(0, 3, 1, 2).contiguous()


class MSATMultiLabel(MSAT):
    """MSAT V2 with a class-conditioned multi-label auxiliary prediction for every spatial token.

    This class is intentionally separate from MSAT V1. It returns the fused feature and the state-derived auxiliary
    logits as explicit graph outputs so training, validation, EMA, checkpointing, and export never depend on mutable
    tensor attributes.
    """

    def __init__(
        self,
        ch: list[int],
        c2: int,
        num_classes: int,
        num_heads: int = 4,
        target: int = -1,
        embed_channels: int = 0,
        ffn_ratio: float = 2.0,
        window_size: int = 8,
        shortcut: bool = True,
        layer_scale_init: float = 1e-3,
    ):
        """Initialize MSAT V2 without changing the MSAT V1 implementation."""
        super().__init__(
            ch,
            c2,
            num_heads,
            target,
            embed_channels,
            ffn_ratio,
            window_size,
            shortcut,
            layer_scale_init,
        )
        if int(num_classes) < 1:
            raise ValueError("MSATMultiLabel requires at least one damage class.")
        self.num_classes = int(num_classes)
        self.multi_label_head = _MSATMultiLabelAuxHead(
            self.embed_channels,
            self.num_heads,
            self.num_classes,
        )
        self.is_multilabel_state_module = True

    def forward(
        self,
        x: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the fused target feature and class-conditioned state logits."""
        xs = [x] if isinstance(x, torch.Tensor) else list(x)
        if len(xs) != self.num_states:
            raise ValueError(f"MSATMultiLabel expected {self.num_states} inputs, but received {len(xs)}.")
        ref = xs[self.target]
        size = ref.shape[-2:]
        states = [projection(self._resize(feature, size)) for projection, feature in zip(self.input_proj, xs)]
        tokens = torch.stack(states, dim=1).permute(0, 3, 4, 1, 2).contiguous()
        tokens = tokens + self.state_embedding
        tokens = self.state_layer(tokens)
        tokens = self.spatial_layer(tokens)

        auxiliary_logits = self.multi_label_head(tokens)
        pooled = self.state_pool(tokens).permute(0, 3, 1, 2).contiguous()
        target_state = tokens[..., self.target, :].permute(0, 3, 1, 2).contiguous()
        output = self.proj(pooled + self.pe(target_state))
        if self.shortcut is not None:
            output = self.shortcut(ref) + self.output_scale * output
        return output, auxiliary_logits


class PatchCSAR(CSAR):
    """Overlapping patch-based Cross-Scale Attention Residual fusion.

    This is a lightweight adaptation of COP-Net's PCA/COSA mechanism. It runs CSAR attention on overlapping patches at
    the same relative locations across scales, then averages overlapping predictions back into the target feature map.
    """

    def __init__(
        self,
        ch: list[int],
        c2: int,
        num_heads: int = 4,
        target: int = -1,
        attn_ratio: float = 0.5,
        shortcut: bool = True,
        patch_ratio: float = 0.75,
        division: str = "fud",
    ):
        """Initialize PatchCSAR.

        Args:
            ch (list[int]): Input channel dimensions for each source feature.
            c2 (int): Output channels.
            num_heads (int): Number of attention heads.
            target (int): Source index used as query, output size, and residual.
            attn_ratio (float): Key/query dimension ratio relative to each value head.
            shortcut (bool): Whether to add the target feature shortcut.
            patch_ratio (float): Patch height/width ratio relative to target feature size.
            division (str): Patch splitting scheme, one of 'qud', 'ced', or 'fud'.
        """
        super().__init__(ch, c2, num_heads, target, attn_ratio, shortcut)
        self.patch_ratio = float(patch_ratio)
        self.division = str(division).lower()

    @staticmethod
    def _unique_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        """Remove duplicate patch boxes while preserving order."""
        seen = set()
        unique = []
        for box in boxes:
            if box not in seen:
                unique.append(box)
                seen.add(box)
        return unique

    def _patch_boxes(self, h: int, w: int) -> list[tuple[int, int, int, int]]:
        """Build overlapping patch boxes for QuD, CeD, or FuD splitting."""
        ratio = min(max(self.patch_ratio, 0.1), 1.0)
        ph = max(1, min(h, round(h * ratio)))
        pw = max(1, min(w, round(w * ratio)))
        cy = max((h - ph) // 2, 0)
        cx = max((w - pw) // 2, 0)
        bottom = h - ph
        right = w - pw
        boxes = [
            (0, ph, 0, pw),
            (0, ph, right, w),
            (bottom, h, 0, pw),
            (bottom, h, right, w),
        ]
        if self.division in {"ced", "fud", "full"}:
            boxes.append((cy, cy + ph, cx, cx + pw))
        if self.division in {"fud", "full"}:
            boxes.extend(
                [
                    (0, ph, cx, cx + pw),
                    (bottom, h, cx, cx + pw),
                    (cy, cy + ph, 0, pw),
                    (cy, cy + ph, right, w),
                ]
            )
        return self._unique_boxes(boxes)

    def _attend_patch(self, aligned: list[torch.Tensor], ref_patch: torch.Tensor) -> torch.Tensor:
        """Apply CSAR attention to one aligned patch group."""
        b, _, h, w = ref_patch.shape
        q = self.q(ref_patch).view(b, self.num_heads, self.key_dim, h, w)
        k = torch.stack(
            [proj(xi).view(b, self.num_heads, self.key_dim, h, w) for proj, xi in zip(self.k, aligned)], dim=0
        )
        v = torch.stack(
            [proj(xi).view(b, self.num_heads, self.head_dim, h, w) for proj, xi in zip(self.v, aligned)], dim=0
        )
        attn = (q.unsqueeze(0) * k).sum(dim=3) * self.scale
        attn = attn.float().softmax(dim=0).to(dtype=v.dtype)
        y = (attn.unsqueeze(3) * v).sum(dim=0).reshape(b, self.num_heads * self.head_dim, h, w)
        y = self.proj(y + self.pe(v[self.target].reshape(b, self.num_heads * self.head_dim, h, w)))
        return y + self.shortcut(ref_patch) if self.shortcut is not None else y

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor) -> torch.Tensor:
        """Fuse multi-scale features using overlapping patch attention."""
        xs = [x] if isinstance(x, torch.Tensor) else list(x)
        if len(xs) != len(self.k):
            raise ValueError(f"PatchCSAR expected {len(self.k)} inputs, but received {len(xs)}.")
        ref = xs[self.target]
        b, _, h, w = ref.shape
        size = (h, w)
        aligned = [self._resize(xi, size) for xi in xs]
        out_channels = self.num_heads * self.head_dim
        out = ref.new_zeros(b, out_channels, h, w)
        weight = ref.new_zeros(1, 1, h, w)

        for y1, y2, x1, x2 in self._patch_boxes(h, w):
            patch_inputs = [xi[..., y1:y2, x1:x2] for xi in aligned]
            patch = self._attend_patch(patch_inputs, ref[..., y1:y2, x1:x2])
            out[..., y1:y2, x1:x2] += patch
            weight[..., y1:y2, x1:x2] += 1
        return out / weight.clamp_min(1)


class CrossScaleAttention(CSAR):
    """Backward-compatible CrossScaleAttention for checkpoints saved before CSAR was added."""

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor) -> torch.Tensor:
        """Fuse old multi-scale inputs with the modules stored in legacy checkpoints."""
        if all(hasattr(self, attr) for attr in ("num_heads", "key_dim", "head_dim", "pe", "shortcut")):
            return super().forward(x)

        xs = [x] if isinstance(x, torch.Tensor) else list(x)
        if not xs:
            raise ValueError("CrossScaleAttention requires at least one input feature map.")

        target = getattr(self, "target", 0) % len(xs)
        ref = xs[target]
        _b, _, h, w = ref.shape
        size = (h, w)
        aligned = [CSAR._resize(xi, size) for xi in xs]

        if len(aligned) != len(self.k):
            raise ValueError(f"CrossScaleAttention expected {len(self.k)} inputs, but received {len(aligned)}.")

        q = self.q(ref)
        k = torch.stack([proj(xi) for proj, xi in zip(self.k, aligned)], dim=0)
        v = torch.stack([proj(xi) for proj, xi in zip(self.v, aligned)], dim=0)

        scale = getattr(self, "scale", q.shape[1] ** -0.5)
        attn = (q.unsqueeze(0) * k).sum(dim=2, keepdim=True) * scale
        attn = attn.float().softmax(dim=0).to(dtype=v.dtype)
        return self.proj((attn * v).sum(dim=0))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            attn (bool): Whether to use attention blocks.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth-wise convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth-wise convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the fused RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth-wise convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        if not hasattr(self, "conv1"):
            return  # already fused
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """Compact Inverted Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use large kernel. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use large kernel.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature
    extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c2.
        m (nn.ModuleList): List of PSABlock modules for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n))


class SCDown(nn.Module):
    """SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and
    customize the model by truncating or unwrapping layers.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): Unwraps the model to a sequential containing all but the last `truncate` layers.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | list[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided into.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.all_head_dim = all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, all_head_dim, 7, 1, 3, g=all_head_dim, act=False)

    def __setstate__(self, state):
        """Add missing all_head_dim attribute to old checkpoints."""
        super().__setstate__(state)
        if not hasattr(self, "all_head_dim"):
            self.all_head_dim = self.head_dim * self.num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, _, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, self.all_head_dim * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, self.all_head_dim)
            v = v.reshape(B // self.area, N * self.area, self.all_head_dim)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided into.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: list[int], c3: int, embed: int):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (list[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        score = F.softmax(score, dim=-1).to(y.dtype)
        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)


class Proto26(Proto):
    """Ultralytics YOLO26 models mask Proto module for segmentation models."""

    def __init__(self, ch: tuple = (), c_: int = 256, c2: int = 32, nc: int = 80):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            ch (tuple): Tuple of channel sizes from backbone feature maps.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
            nc (int): Number of classes for semantic segmentation.
        """
        super().__init__(c_, c_, c2)
        self.feat_refine = nn.ModuleList(Conv(x, ch[0], k=1) for x in ch[1:])
        self.feat_fuse = Conv(ch[0], c_, k=3)
        self.semseg = nn.Sequential(Conv(ch[0], c_, k=3), Conv(c_, c_, k=3), nn.Conv2d(c_, nc, 1))
        self.heatmap = nn.Sequential(Conv(ch[0], c_, k=3), nn.Conv2d(c_, nc, 1))
        self.seedmap = nn.Sequential(Conv(ch[0], c_, k=3), nn.Conv2d(c_, nc, 1))

    def forward(self, x: torch.Tensor, return_semantic: bool = True) -> torch.Tensor:
        """Perform a forward pass by fusing multi-scale feature maps and generating proto masks."""
        feat = x[0]
        for i, f in enumerate(self.feat_refine):
            up_feat = f(x[i + 1])
            up_feat = F.interpolate(up_feat, size=feat.shape[2:], mode="nearest")
            feat = feat + up_feat
        p = super().forward(self.feat_fuse(feat))
        if self.training and return_semantic:
            dense = {
                "semantic": self.semseg(feat),
                "heatmap": self.heatmap(feat),
                "seedmap": self.seedmap(feat),
            }
            return (p, dense)
        return p

    def fuse(self):
        """Fuse the model for inference by removing the semantic segmentation head."""
        self.semseg = None
        self.heatmap = None
        self.seedmap = None


class Proto26MultiLabel(Proto26):
    """Proto26 variant that leaves semantic supervision to the MSAT multi-label state heads."""

    def __init__(self, ch: tuple = (), c_: int = 256, c2: int = 32, nc: int = 80):
        """Initialize multi-scale prototypes without the mutually exclusive semantic branch."""
        super().__init__(ch, c_, c2, nc)
        self.semseg = None

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Fuse multi-scale features and return prototypes plus compatible dense localization targets."""
        feat = x[0]
        for i, refine in enumerate(self.feat_refine):
            up_feat = refine(x[i + 1])
            up_feat = F.interpolate(up_feat, size=feat.shape[2:], mode="nearest")
            feat = feat + up_feat
        prototypes = Proto.forward(self, self.feat_fuse(feat))
        if self.training:
            dense = {
                "heatmap": self.heatmap(feat),
                "seedmap": self.seedmap(feat),
            }
            return prototypes, dense
        return prototypes


class RealNVP(nn.Module):
    """RealNVP: a flow-based generative model.

    References:
        https://arxiv.org/abs/1605.08803
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/realnvp.py
    """

    @staticmethod
    def nets():
        """Get the scale model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def nett():
        """Get the translation model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))

    @property
    def prior(self):
        """The prior distribution."""
        return torch.distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self):
        super().__init__()

        self.register_buffer("loc", torch.zeros(2))
        self.register_buffer("cov", torch.eye(2))
        self.register_buffer("mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))

        self.s = torch.nn.ModuleList([self.nets() for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList([self.nett() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """Apply mapping from the data space to the latent space and calculate the log determinant of the Jacobian
        matrix.
        """
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """Calculate the log probability of given sample in data space."""
        if x.dtype == torch.float32 and self.s[0][0].weight.dtype != torch.float32:
            self.float()
        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det
