# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the Multi-State Axial Transformer V1 neck."""

from pathlib import Path

import torch

from ultralytics.nn.modules import MSAT
from ultralytics.nn.tasks import SegmentationModel

MODEL_CFG = Path("ultralytics/cfg/models/11_myself/yolo11-MSAT-V1.yaml")


def test_msat_forward_backward():
    """Verify multi-scale alignment, padded windows, output shape, and attention gradients."""
    module = MSAT(
        [16, 24, 32, 48, 64],
        c2=24,
        num_heads=4,
        target=1,
        embed_channels=32,
        window_size=8,
    )
    features = [
        torch.randn(2, 16, 31, 35, requires_grad=True),
        torch.randn(2, 24, 17, 19, requires_grad=True),
        torch.randn(2, 32, 9, 10, requires_grad=True),
        torch.randn(2, 48, 5, 5, requires_grad=True),
        torch.randn(2, 64, 3, 3, requires_grad=True),
    ]

    output = module(features)
    assert output.shape == (2, 24, 17, 19)
    output.mean().backward()
    assert all(feature.grad is not None and torch.isfinite(feature.grad).all() for feature in features)
    assert module.state_layer.qkv.weight.grad is not None
    assert module.spatial_layer.qkv.weight.grad is not None


def test_msat_model_yaml():
    """Verify that the independent MSAT model YAML builds and performs inference."""
    model = SegmentationModel(MODEL_CFG, ch=3, nc=3, verbose=False)
    assert sum(isinstance(layer, MSAT) for layer in model.modules()) == 5

    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 128, 128))
    assert output is not None
