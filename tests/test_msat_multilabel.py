# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for class-conditioned multi-label supervision in MSAT V2."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import MSAT, MSATMultiLabel, MultiStateCSAR, Segment26MultiLabel
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.loss import v8MultiLabelSegmentationLoss
from ultralytics.utils.torch_utils import ModelEMA

MODEL_CFG = Path("ultralytics/cfg/models/11_myself/yolo11-MSAT-V2-MultiLabel.yaml")
V1_CFG = Path("ultralytics/cfg/models/11_myself/yolo11-MSAT-V1.yaml")
LEGACY_CFG = Path("ultralytics/cfg/models/11_myself/yolo11-MultiStateToken.yaml")


def _overlapping_batch() -> dict[str, torch.Tensor]:
    """Create two independently stored damage masks with a genuine multi-label overlap."""
    masks = torch.zeros(2, 32, 32)
    masks[0, 6:23, 5:21] = 1
    masks[1, 12:29, 12:27] = 1
    return {
        "img": torch.randn(1, 3, 128, 128),
        "batch_idx": torch.tensor([0.0, 0.0]),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor(
            [
                [0.40625, 0.453125, 0.5, 0.53125],
                [0.609375, 0.640625, 0.46875, 0.53125],
            ]
        ),
        "masks": masks,
        "heatmaps": torch.zeros(1, 3, 32, 32),
        "seedmaps": torch.zeros(1, 3, 32, 32),
    }


def test_msat_multilabel_target_loss_backward():
    """Verify multi-hot overlap targets, finite loss, and gradients to every class-conditioned state head."""
    model = SegmentationModel(MODEL_CFG, ch=3, nc=3, verbose=False)
    model.args = get_cfg()
    model.args.overlap_mask = False
    model.train()
    batch = _overlapping_batch()

    predictions = model(batch["img"])
    assert [tuple(logits.shape) for logits in predictions["multilabel_logits"]] == [
        (1, 3, 16, 16),
        (1, 3, 32, 32),
        (1, 3, 8, 8),
        (1, 3, 4, 4),
        (1, 3, 2, 2),
    ]
    criterion = model.init_criterion()
    assert isinstance(criterion, v8MultiLabelSegmentationLoss)
    target = criterion.build_multilabel_target(batch, 1, (32, 32), torch.float32)
    assert target[0, :, 15, 15].tolist() == [1.0, 1.0, 0.0]

    loss, items = criterion(predictions, batch)
    assert loss.shape == (7,) and items.shape == (7,)
    assert torch.isfinite(loss).all() and items[4] > 0
    loss.sum().backward()

    modules = [module for module in model.modules() if isinstance(module, MSATMultiLabel)]
    assert len(modules) == 5
    assert all(module.multi_label_head.class_queries.grad is not None for module in modules)
    assert all(torch.isfinite(module.multi_label_head.class_queries.grad).all() for module in modules)
    assert isinstance(model.model[-1], Segment26MultiLabel)


def test_msat_versions_remain_independent():
    """Verify V2, V1, and legacy YAML files instantiate only their intended state modules."""
    v2_model = SegmentationModel(MODEL_CFG, ch=3, nc=3, verbose=False)
    v1_model = SegmentationModel(V1_CFG, ch=3, nc=3, verbose=False)
    legacy_model = SegmentationModel(LEGACY_CFG, ch=3, nc=3, verbose=False)

    assert sum(isinstance(module, MSATMultiLabel) for module in v2_model.modules()) == 5
    assert sum(type(module) is MSAT for module in v1_model.modules()) == 5
    assert sum(isinstance(module, MSATMultiLabel) for module in v1_model.modules()) == 0
    assert sum(isinstance(module, MultiStateCSAR) for module in legacy_model.modules()) == 5
    assert sum(isinstance(module, MSAT) for module in legacy_model.modules()) == 0


def test_msat_multilabel_requires_independent_masks():
    """Verify that merged instance masks fail early instead of silently losing class co-occurrence."""
    model = SegmentationModel(MODEL_CFG, ch=3, nc=3, verbose=False)
    model.args = get_cfg()
    model.args.overlap_mask = True
    try:
        model.init_criterion()
    except ValueError as error:
        assert "overlap_mask=False" in str(error)
    else:
        raise AssertionError("Expected multi-label loss initialization to reject overlap_mask=True.")


def test_msat_multilabel_supports_ema_deepcopy():
    """Verify explicit auxiliary outputs never leave graph tensors in module state before EMA construction."""
    model = SegmentationModel(MODEL_CFG, ch=3, nc=3, verbose=False)
    model.args = get_cfg()
    model.args.overlap_mask = False
    model.train()
    batch = _overlapping_batch()
    predictions = model(batch["img"])
    assert "multilabel_logits" in predictions
    ema = ModelEMA(model)
    assert ema.ema is not model

    train_loss, _ = model.loss(batch)
    assert torch.isfinite(train_loss).all()
    deepcopy(model)

    ema.ema.eval()
    with torch.no_grad():
        validation_predictions = ema.ema(batch["img"])
        validation_loss, _ = ema.ema.loss(batch, validation_predictions)
    assert torch.isfinite(validation_loss).all()
    deepcopy(ema.ema)
