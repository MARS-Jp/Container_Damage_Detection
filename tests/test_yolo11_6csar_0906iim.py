# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the standalone yolo11-6csar-0906iim MMS + ASL variant."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from tools.train_yolo11_6csar_0906iim import (
    MODEL_CFG,
    AsymmetricLoss,
    MMSASLSegmentationModel,
    MMSASLTrainer,
)
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel


def _overlapping_batch() -> dict[str, torch.Tensor]:
    """Create two independent masks with a real two-class overlap."""
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


def test_asl_matches_bce_when_asymmetric_terms_are_disabled():
    """Check the numerically useful ASL-to-BCE limiting case."""
    logits = torch.tensor([[-2.0, 0.0, 1.5]])
    target = torch.tensor([[0.0, 1.0, 1.0]])
    actual = AsymmetricLoss(gamma_pos=0, gamma_neg=0, clip=0)(logits, target)
    expected = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    assert torch.allclose(actual, expected, atol=1e-6)


def test_new_yaml_mms_asl_forward_and_full_loss_backward():
    """Build the new YAML and backpropagate full loss through all three independent MMS maps."""
    model = MMSASLSegmentationModel(MODEL_CFG, ch=3, nc=3, verbose=False)
    model.args = get_cfg()
    model.args.overlap_mask = False
    model.train()
    batch = _overlapping_batch()

    predictions = model(batch["img"])
    assert [tuple(logits.shape) for logits in predictions["multilabel_logits"]] == [
        (1, 3, 16, 16),
        (1, 3, 8, 8),
        (1, 3, 4, 4),
    ]

    criterion = model.init_criterion()
    target = criterion.build_multilabel_target(batch, 1, (16, 16), torch.float32)
    assert target[0, :, 7, 7].tolist() == [1.0, 1.0, 0.0]
    loss, items = criterion(predictions, batch)
    assert loss.shape == (7,) and items.shape == (7,)
    assert torch.isfinite(loss).all() and items[4] > 0

    loss.sum().backward()
    assert all(model.model[index].conv.weight.grad is not None for index in (21, 23, 25))


def test_trainer_serializes_a_portable_standard_model_class():
    """Ensure checkpoint saving temporarily exposes EMA as SegmentationModel and then restores its runtime class."""
    model = MMSASLSegmentationModel(MODEL_CFG, ch=3, nc=3, verbose=False)
    trainer = object.__new__(MMSASLTrainer)
    trainer.ema = SimpleNamespace(ema=model)
    observed = {}
    original_save_model = SegmentationTrainer.save_model

    def capture_model_class(self):
        observed["saved_class"] = type(self.ema.ema)
        return True

    SegmentationTrainer.save_model = capture_model_class
    try:
        assert trainer.save_model() is True
    finally:
        SegmentationTrainer.save_model = original_save_model

    assert observed["saved_class"] is SegmentationModel
    assert type(trainer.ema.ema) is MMSASLSegmentationModel
