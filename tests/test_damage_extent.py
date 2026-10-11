"""Check complete-instance supervision and the ordered 0928 training experiments."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import YAML
from ultralytics.utils.damage_extent import instance_extent_loss
from ultralytics.utils.loss import v8MultiLabelSegmentationLoss
from ultralytics.utils.torch_utils import ModelEMA
from ultralytics.utils.training_aux_0918 import parse_training_aux

CFG = Path("ultralytics/cfg/models/11_myself")
BASE = CFG / "yolo11-6csar-semantic-token.yaml"
SUFFIXES = ("01-baseline", "02-shadow", "03-consistency", "04-extent", "05-boundary", "06-detail960")


def configured(suffix=None):
    path = BASE if suffix is None else CFG / f"yolo11-6csar-extent-0928-{suffix}.yaml"
    model = SegmentationModel(path, nc=8, verbose=False)
    model.args = get_cfg(overrides={"overlap_mask": False, "compile": False})
    model.names = dict(enumerate(("B", "C", "D", "H", "O", "R", "X", "U")))
    return model


def batch(empty=False):
    masks = torch.zeros(2, 32, 32)
    masks[0, 6:23, 5:21] = 1
    masks[1, 12:29, 12:27] = 1
    n = 0 if empty else 2
    return {
        "img": torch.rand(1, 3, 128, 128),
        "masks": masks[:n],
        "batch_idx": torch.zeros(n),
        "cls": torch.tensor([[2.0], [5.0]])[:n],
        "bboxes": torch.tensor([[0.40625, 0.453125, 0.5, 0.53125], [0.609375, 0.640625, 0.46875, 0.53125]])[:n],
        "heatmaps": torch.zeros(1, 8, 32, 32),
        "seedmaps": torch.zeros(1, 8, 32, 32),
    }


def test_extent_rewards_full_mask_and_penalizes_spill_and_missed_holes():
    target = torch.zeros(1, 9, 13)
    target[:, 2:7, 3:10] = 1
    target[:, 4, 6] = 0  # a genuine hole must not be filled
    box = torch.tensor([[3.0, 2.0, 10.0, 7.0]])
    perfect = torch.where(target.bool(), 8.0, -8.0)
    partial = torch.full_like(target, -8.0)
    partial[:, 3:5, 4:6] = 8
    filled = torch.full_like(target, 8.0)
    good = instance_extent_loss(perfect, target, box)
    assert good < instance_extent_loss(partial, target, box)
    assert good < instance_extent_loss(filled, target, box)
    assert good < instance_extent_loss(perfect.masked_fill(~target.bool(), 8), target, box)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_extent_has_recovery_and_background_gradients_without_mutating_targets(dtype):
    target = torch.zeros(1, 8, 8, dtype=dtype)
    target[:, 2:6, 2:6] = 1
    saved = target.clone()
    logits = torch.full_like(target, -20, requires_grad=True)
    box = torch.tensor([[2.0, 2.0, 6.0, 6.0]])
    loss = instance_extent_loss(logits, target, box)
    loss.backward()
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert torch.equal(saved, target) and torch.isfinite(logits.grad).all()
    assert logits.grad[target.bool()].sum() < 0
    assert logits.grad[0, 0, 0] == 0  # outside the expanded GT ROI
    filled = torch.full_like(target, 20, requires_grad=True)
    instance_extent_loss(filled, target, box).backward()
    assert filled.grad[0, 1, 1] > 0


@pytest.mark.parametrize("n", [0, 2])
def test_extent_empty_foreground_is_differentiable_zero(n):
    logits = torch.randn(n, 7, 11, requires_grad=True)
    target = torch.zeros_like(logits)
    boxes = torch.tensor([[1.0, 1.0, 8.0, 6.0]]).repeat(n, 1)
    loss = instance_extent_loss(logits, target, boxes)
    assert loss.item() == 0
    loss.backward()
    assert logits.grad is not None and torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize(
    "extent",
    [
        {"gain": -1},
        {"fn_weight": 0},
        {"fn_weight": 1},
        {"margin": -1},
        {"fn_weight": float("nan")},
        {"gain": True},
        {"typo": 1},
    ],
)
def test_invalid_extent_config(extent):
    with pytest.raises(ValueError):
        parse_training_aux({"extent": extent})


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_ordered_configs_forward_backward_and_unchanged_inference_graph(suffix):
    original = YAML.load(BASE)
    path = CFG / f"yolo11-6csar-extent-0928-{suffix}.yaml"
    config = YAML.load(path)
    assert config["backbone"] == original["backbone"] and config["head"] == original["head"]
    recipe = get_cfg(f"ultralytics/cfg/experiments/damage-extent-0928/{suffix}.yaml")
    assert recipe.overlap_mask is False and recipe.mask_ratio == 4 and recipe.compile is False
    assert Path(recipe.model) == path
    assert recipe.imgsz == (960 if suffix == "06-detail960" else 640)
    model = configured(suffix).train()
    data = batch()
    saved = {k: v.clone() for k, v in data.items()}
    loss, items = model(data)
    assert loss.shape == items.shape == (7,)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    for module in model.modules():
        if hasattr(module, "damage_tokens"):
            assert module.damage_tokens.grad is not None and torch.isfinite(module.damage_tokens.grad).all()
    assert all(torch.equal(data[k], v) for k, v in saved.items())
    assert model.stride.tolist() == [8.0, 4.0, 16.0, 32.0]
    ModelEMA(model)
    with torch.no_grad():
        output = deepcopy(model).eval()(torch.rand(1, 3, 128, 192))
    assert output[0][1].shape == (1, 32, 32, 48)


def test_baseline_matches_original_exactly():
    old, new = configured().train(), configured("01-baseline").train()
    new.load_state_dict(old.state_dict(), strict=True)
    data = batch()
    a, _ = old(data)
    b, _ = new(data)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    with torch.no_grad():
        a = old.eval()(data["img"])
        b = new.eval()(data["img"])
    torch.testing.assert_close(a[0][0], b[0][0], rtol=0, atol=0)
    torch.testing.assert_close(a[0][1], b[0][1], rtol=0, atol=0)


def test_extent_reaches_main_mask_parameters_and_is_disabled_in_eval():
    model = configured("04-extent")
    criterion = model.init_criterion()
    target = torch.zeros(1, 8, 8)
    target[:, 2:6, 2:6] = 1
    coeff = torch.randn(1, 2, requires_grad=True)
    proto = torch.randn(2, 8, 8, requires_grad=True)
    box, area = torch.tensor([[2.0, 2.0, 6.0, 6.0]]), torch.tensor([0.25])
    original = criterion.single_mask_loss(target, coeff, proto, box, area)
    criterion._training_terms = True
    extra = criterion.single_mask_loss(target, coeff, proto, box, area) - original
    assert extra > 0
    grads = torch.autograd.grad(extra, (coeff, proto))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    model.criterion = criterion
    model.eval()
    data = batch()
    with torch.no_grad():
        raw = model(data["img"])
        got, _ = model.loss(data, raw)
        expected, _ = v8MultiLabelSegmentationLoss(model)(raw, data)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_extent_empty_training_batch_backward():
    model = configured("05-boundary").train()
    loss, _ = model(batch(empty=True))
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
