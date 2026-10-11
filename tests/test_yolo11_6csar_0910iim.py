# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the isolated 0910 adaptive-IIM, P2, MMS class-query model."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tools.train_yolo11_6csar_0910iim import (
    MODEL_CFG,
    IIMQueryASLSegmentationModel,
    IIMQueryASLTrainer,
    build_overrides,
    parse_args,
)
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.nn import tasks as nn_tasks
from ultralytics.nn.modules.adaptive_iim_0910 import AdaptiveGatedIIMStem0910
from ultralytics.nn.modules.conv import IIMStem
from ultralytics.nn.modules.head import Segment26MultiLabel
from ultralytics.nn.modules.segment_query_0910 import P2Proto26MultiLabel0910, Segment26ClassQuery0910
from ultralytics.nn.tasks import SegmentationModel, yaml_model_load
from ultralytics.utils import ops


def _overlapping_batch() -> dict[str, torch.Tensor]:
    """Create two independently stored masks with a two-class overlap."""
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


def test_adaptive_iim_starts_balanced_and_backpropagates_to_both_branches():
    """The adaptive stem must preserve RGB and IIM features at initialization."""
    stem = AdaptiveGatedIIMStem0910(3, 32, 3, 2, 4, 3)
    image = torch.rand(2, 3, 64, 64, requires_grad=True)
    rgb, iim, preference = stem.branch_features(image)
    output = stem(image)

    assert rgb.shape == iim.shape == (2, 16, 32, 32)
    assert output.shape == (2, 32, 32, 32)
    assert torch.allclose(preference, torch.full_like(preference, 0.5))
    output.mean().backward()
    assert stem.rgb_stem.conv.weight.grad is not None
    assert stem.iim_stem.conv.weight.grad is not None
    assert stem.gate.weight.grad is not None


def test_0910_yaml_keeps_six_csar_and_adds_p2_query_conditioning():
    """Build four output scales without adding more CSAR blocks."""
    config = yaml_model_load(MODEL_CFG)
    layers = config["backbone"] + config["head"]
    assert sum(layer[2] == "CSAR" for layer in layers) == 6
    assert config["head"][-1][0][:4] == [3, 18, 19, 20]

    original_iim = nn_tasks.IIMStem
    original_head = nn_tasks.Segment26MultiLabel
    model = IIMQueryASLSegmentationModel(MODEL_CFG, ch=3, nc=3, verbose=False)
    assert nn_tasks.IIMStem is original_iim
    assert nn_tasks.Segment26MultiLabel is original_head
    assert isinstance(model.model[0], AdaptiveGatedIIMStem0910)
    assert isinstance(model.model[-1], Segment26ClassQuery0910)
    assert isinstance(model.model[-1].proto, P2Proto26MultiLabel0910)
    assert model.model[-1].num_feature_levels == 4
    assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    assert not isinstance(model.model[0], IIMStem)
    assert isinstance(model.model[-1], Segment26MultiLabel)


def test_0910_full_asl_loss_reaches_p2_p3_p4_p5_mms_projections():
    """Backpropagate the full main-class and MMS ASL loss through all four maps."""
    model = IIMQueryASLSegmentationModel(MODEL_CFG, ch=3, nc=3, verbose=False)
    model.args = get_cfg()
    model.args.overlap_mask = False
    model.train()
    batch = _overlapping_batch()

    predictions = model(batch["img"])
    assert [tuple(logits.shape) for logits in predictions["multilabel_logits"]] == [
        (1, 3, 32, 32),
        (1, 3, 16, 16),
        (1, 3, 8, 8),
        (1, 3, 4, 4),
    ]
    prototypes = predictions["proto"][0]
    assert prototypes.shape[-2:] == batch["masks"].shape[-2:] == (32, 32)
    criterion = model.init_criterion()
    assert criterion.main_cls_asl_enabled is True
    loss, items = criterion(predictions, batch)
    assert loss.shape == (7,) and items.shape == (7,)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert all(model.model[index].conv.weight.grad is not None for index in (21, 23, 25, 27))
    assert all(adapter.weight.grad is not None for adapter in model.model[-1].class_adapters)


def test_standard_validator_receives_matching_stride_four_masks():
    """Exercise the exact validation path that previously failed during mask IoU."""
    model = IIMQueryASLSegmentationModel(MODEL_CFG, ch=3, nc=3, verbose=False).eval()
    image = torch.rand(1, 3, 128, 128)
    raw_predictions = model(image)
    args = get_cfg(overrides={"task": "segment", "mode": "val", "conf": 0.001, "iou": 0.7, "overlap_mask": False})
    validator = SegmentationValidator(args=args)
    validator.device = torch.device("cpu")
    validator.nc = 3
    validator.end2end = False
    validator.process = ops.process_mask
    predictions = validator.postprocess(raw_predictions)
    batch = {
        **_overlapping_batch(),
        "ori_shape": [(128, 128)],
        "ratio_pad": [((1.0, 1.0), (0.0, 0.0))],
        "im_file": ["synthetic.jpg"],
    }
    prepared = validator._prepare_batch(0, batch)

    assert prepared["masks"].shape[-2:] == predictions[0]["masks"].shape[-2:] == (32, 32)
    assert prepared["masks"].flatten(1).shape[1] == predictions[0]["masks"].flatten(1).shape[1]


def test_training_overrides_use_independent_masks_and_no_early_stop():
    """Expose the requested training controls without changing the YAML graph."""
    args = parse_args(["--data", "/home/d11405003/dataset/data.yaml"])
    overrides = build_overrides(args)
    assert overrides["overlap_mask"] is False
    assert overrides["patience"] == 0
    assert overrides["batch"] == 8
    assert overrides["mask_ratio"] == 4
    assert overrides["cls_pw"] == 0.5
    assert overrides["seed"] == 0
    assert overrides["deterministic"] is True
    assert "project" not in overrides


def test_training_rejects_a_mask_ratio_that_cannot_match_p2_prototypes():
    """Keep P2-aware prototypes and standard validation targets at stride four."""
    args = parse_args(["--data", "/home/d11405003/dataset/data.yaml", "--mask-ratio", "2"])
    try:
        build_overrides(args)
    except ValueError as error:
        assert "requires --mask-ratio 4" in str(error)
    else:
        raise AssertionError("mask_ratio=2 should be rejected for the P2-aware prototype head")


def test_0910_trainer_serializes_the_portable_standard_model_class():
    """Future checkpoints must load without a class defined under __main__."""
    model = IIMQueryASLSegmentationModel(MODEL_CFG, ch=3, nc=3, verbose=False)
    trainer = object.__new__(IIMQueryASLTrainer)
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
    assert type(trainer.ema.ema) is IIMQueryASLSegmentationModel
