# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Damage queries that read visual evidence and return independent per-pixel responses."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ("DamageSemanticAttention",)


class DamageSemanticAttention(nn.Module):
    """Refine a feature map with supervised damage tokens and token-to-visual cross-attention.

    Each row of ``damage_tokens`` corresponds to a dataset class ID. These are learned class embeddings, not pretrained
    text embeddings. Queries attend over spatial positions independently for each class; the dense response uses sigmoid
    rather than a softmax over classes, allowing damage overlap.

    Returns ``(enhanced_feature, multilabel_logits)`` for explicit routing through YAML Index layers. The residual
    visual path feeds the YOLO head, while both segmentation and multi-label losses train the semantic branch. Attention
    storage grows as O(num_classes * H * W), not O((H * W) ** 2).
    """

    def __init__(
        self,
        c1: int,
        num_classes: int,
        num_heads: int = 4,
        embed_channels: int = 128,
        residual_scale: float = 1e-3,
    ):
        """Keep input channels/resolution and use a fixed embedding width across model scales."""
        super().__init__()
        if c1 < 1 or num_classes < 1:
            raise ValueError("DamageSemanticAttention requires positive input channels and damage classes.")
        if num_heads < 1 or embed_channels < 1 or embed_channels % num_heads:
            raise ValueError("DamageSemanticAttention requires embed_channels divisible by positive num_heads.")
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.embed_channels = embed_channels
        self.head_dim = embed_channels // num_heads
        self.scale = self.head_dim**-0.5

        self.visual_proj = nn.Conv2d(c1, embed_channels, 1)
        self.visual_norm = nn.LayerNorm(embed_channels)
        self.damage_tokens = nn.Parameter(torch.empty(num_classes, embed_channels))
        nn.init.trunc_normal_(self.damage_tokens, std=0.02)
        self.token_norm = nn.LayerNorm(embed_channels)
        self.query = nn.Linear(embed_channels, embed_channels)
        self.key = nn.Linear(embed_channels, embed_channels)
        self.value = nn.Linear(embed_channels, embed_channels)
        self.context_proj = nn.Linear(embed_channels, embed_channels)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_channels),
            nn.Linear(embed_channels, 2 * embed_channels),
            nn.GELU(),
            nn.Linear(2 * embed_channels, embed_channels),
        )
        self.response_norm = nn.LayerNorm(embed_channels)
        self.response_query = nn.Linear(embed_channels, embed_channels)
        self.response_key = nn.Linear(embed_channels, embed_channels)
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        self.semantic_value = nn.Linear(embed_channels, embed_channels)
        self.output_proj = nn.Conv2d(embed_channels, c1, 1)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Let every damage query read the same visual map, then broadcast its independent response."""
        b, _, h, w = x.shape
        visual = self.visual_norm(self.visual_proj(x).flatten(2).transpose(1, 2))
        tokens = self.damage_tokens.unsqueeze(0).expand(b, -1, -1)
        queries = self.query(self.token_norm(tokens))
        queries = queries.reshape(b, self.num_classes, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key(visual).reshape(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value(visual).reshape(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)

        # Softmax is over visual positions: Rust and Dent never share a class normalization.
        attention = (queries @ keys.transpose(-2, -1)) * self.scale
        attention = attention.float().softmax(dim=-1).to(dtype=values.dtype)
        context = (attention @ values).transpose(1, 2).reshape(b, self.num_classes, self.embed_channels)
        tokens = tokens + self.context_proj(context)
        tokens = self.response_norm(tokens + self.ffn(tokens))

        logits = self.response_query(tokens) @ self.response_key(visual).transpose(1, 2)
        logits = logits * self.embed_channels**-0.5 + self.class_bias[None, :, None]
        response = logits.sigmoid()
        semantic = (response.transpose(1, 2) @ self.semantic_value(tokens)) / self.num_classes
        semantic = semantic.transpose(1, 2).reshape(b, self.embed_channels, h, w)
        enhanced = x + self.residual_scale * self.output_proj(semantic)
        return enhanced, logits.reshape(b, self.num_classes, h, w)
