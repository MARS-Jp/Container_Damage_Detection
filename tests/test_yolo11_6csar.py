# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Regression tests for the two-stage CSAR segmentation architecture."""

from pathlib import Path

import torch

from ultralytics.nn.modules import CSAR, Segment26
from ultralytics.nn.tasks import SegmentationModel

MODEL_CFG = Path("ultralytics/cfg/models/11_myself/yolo11-6csar.yaml")


def test_yolo11_6csar_two_stage_head_and_mask_resolution():
    """Verify the diagrammed two-stage CSAR graph and validator-compatible prototypes."""
    model = SegmentationModel(MODEL_CFG, ch=3, nc=3, verbose=False)
    csar_layers = [layer for layer in model.model if type(layer) is CSAR]

    assert len(csar_layers) == 6
    assert [layer.f for layer in csar_layers] == [
        [3, 6, 9, 14],
        [3, 6, 9, 14],
        [3, 6, 9, 14],
        [3, 15, 16, 17],
        [3, 15, 16, 17],
        [3, 15, 16, 17],
    ]
    assert [layer.target for layer in csar_layers] == [1, 2, 3, 1, 2, 3]
    assert isinstance(model.model[-1], Segment26)
    assert model.model[-1].f == [18, 19, 20]
    assert model.stride.tolist() == [8.0, 16.0, 32.0]

    model.eval()
    image = torch.randn(1, 3, 128, 192)
    with torch.no_grad():
        output = model(image)
    proto = output[0][1]
    assert proto.shape == (1, 32, image.shape[-2] // 4, image.shape[-1] // 4)
