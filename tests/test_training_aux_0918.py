# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Check training-only shadow, boundary, and overlap constraints without inference graph changes."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import Segment26MultiLabel
from ultralytics.nn.modules.conv import IIMStem
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import YAML
from ultralytics.utils.loss import v8MultiLabelSegmentationLoss
from ultralytics.utils.torch_utils import ModelEMA
from ultralytics.utils.training_aux_0918 import TrainingAuxSegmentationLoss, parse_training_aux, resolve_overlap_pairs
from ultralytics.utils.training_aux_ops import (
    consistency_loss,
    overlap_pair_loss,
    resize_presence,
    shadow_view,
    soft_boundary,
)

CFG_DIR = Path("ultralytics/cfg/models/11_myself")
BASE_CFG = CFG_DIR / "yolo11-10csar-dr-msat-0911.yaml"
VARIANTS = ("baseline", "shadow-aug", "consistency", "boundary", "overlap", "shadow-boundary", "full")
NAMES = {0: "B", 1: "C", 2: "D", 3: "H", 4: "O", 5: "R", 6: "X", 7: "U"}


def variant_cfg(variant):
    """Return one explicit ablation YAML instead of silently changing the baseline."""
    return CFG_DIR / f"yolo11-10csar-dr-trainonly-0918-{variant}.yaml"


def make_batch(empty=False):
    """Independent D/R masks overlap, while black and low-light image regions remain valid input."""
    image = torch.rand(1, 3, 128, 128) * 0.08
    image[:, :, :24] = 0
    masks = torch.zeros(2, 32, 32)
    masks[0, 6:23, 5:21] = 1
    masks[1, 12:29, 12:27] = 1
    batch = {
        "img": image,
        "batch_idx": torch.tensor([0.0, 0.0]),
        "cls": torch.tensor([[2.0], [5.0]]),
        "bboxes": torch.tensor([[0.40625, 0.453125, 0.5, 0.53125], [0.609375, 0.640625, 0.46875, 0.53125]]),
        "masks": masks,
        "heatmaps": torch.zeros(1, 8, 32, 32),
        "seedmaps": torch.zeros(1, 8, 32, 32),
    }
    if empty:
        for key in ("batch_idx", "cls", "bboxes", "masks"):
            batch[key] = batch[key][:0]
    return batch


def configured_model(cfg):
    """Simulate dataset class names and training arguments supplied by the normal trainer."""
    model = SegmentationModel(cfg, ch=3, nc=8, verbose=False)
    model.names = dict(NAMES)
    model.args = get_cfg(overrides={"overlap_mask": False})
    return model


@pytest.fixture(scope="module")
def reference_model():
    """Share graph inspection costs without keeping any training graph alive."""
    return configured_model(BASE_CFG).eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_shadow_view_black_low_light_finite_no_inplace(dtype):
    """Photometric-only shadows preserve pixels' coordinates, dtype, and the caller's tensor."""
    image = (torch.rand(2, 3, 31, 47) * 0.08).to(dtype)
    image[:, :, :10] = 0
    original = image.clone()
    view, darkness = shadow_view(image)
    assert view.shape == image.shape and view.dtype == dtype
    assert darkness.shape == (2, 1, 31, 47)
    assert torch.equal(image, original)
    assert torch.isfinite(view).all() and torch.isfinite(darkness).all()
    assert (view >= 0).all() and (view <= image).all()
    assert torch.equal(view[:, :, :10], image[:, :, :10])
    assert (view[:, :, 10:] < image[:, :, 10:]).any()


def test_shadow_view_reproducible_and_probability_zero_identity():
    """Torch seeding reproduces views; disabling augmentation cannot alter the image."""
    image = torch.rand(3, 3, 32, 48)
    torch.manual_seed(918)
    first, first_darkness = shadow_view(image)
    torch.manual_seed(918)
    second, second_darkness = shadow_view(image)
    assert torch.equal(first, second) and torch.equal(first_darkness, second_darkness)
    disabled, darkness = shadow_view(image, probability=0.0)
    assert torch.equal(disabled, image)
    assert torch.count_nonzero(darkness) == 0


def test_soft_boundary_constant_surfaces_and_gradient():
    """Image borders are not fabricated damage boundaries; genuine mask edges remain differentiable."""
    for value in (0.0, 1.0):
        flat = torch.full((1, 2, 9, 13), value)
        assert torch.count_nonzero(soft_boundary(flat)) == 0
    mask = torch.zeros(1, 2, 9, 13)
    mask[:, :, 2:7, 3:10] = 1
    mask.requires_grad_()
    boundary = soft_boundary(mask)
    assert boundary.shape == mask.shape
    assert torch.isfinite(boundary).all() and boundary.sum() > 0
    assert boundary[0, 0, 4, 6] == 0
    boundary.square().mean().backward()
    assert mask.grad is not None and torch.isfinite(mask.grad).all()


def test_presence_resize_keeps_small_independent_classes():
    """Presence pooling must retain a small dent and co-located rust independently."""
    target = torch.zeros(1, 8, 8, 8)
    target[:, 2, 2, 2] = 1
    target[:, 5, 2, 2] = 1
    original = target.clone()
    resized = resize_presence(target, (2, 2))
    assert resized.shape == (1, 8, 2, 2)
    assert resized[:, 2].max() == 1 and resized[:, 5].max() == 1
    assert torch.equal(target, original)
    assert resize_presence(target, (16, 16)).shape == (1, 8, 16, 16)


def test_consistency_teacher_detached_confidence_and_ground_truth_agreement():
    """Do not copy confident misses or false positives from the clean-view teacher."""
    teacher = torch.tensor([8.0, -8.0, 0.0, 8.0, -8.0]).reshape(1, 1, 1, 5).requires_grad_()
    student = torch.zeros_like(teacher, requires_grad=True)
    target = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]).reshape_as(teacher)
    loss = consistency_loss(student, teacher, target, confidence=0.7)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert teacher.grad is None
    assert student.grad[0, 0, 0, 0] < 0
    assert student.grad[0, 0, 0, 1] > 0
    assert torch.count_nonzero(student.grad[..., 2:]) == 0


def test_consistency_no_valid_pixels_is_differentiable_zero():
    """An uncertain teacher cannot force an arbitrary background target on missed dents."""
    teacher = torch.zeros(1, 2, 3, 4, requires_grad=True)
    student = torch.randn_like(teacher, requires_grad=True)
    loss = consistency_loss(student, teacher, torch.ones_like(teacher), confidence=0.7)
    assert loss.item() == 0
    loss.backward()
    assert teacher.grad is None
    assert student.grad is not None and torch.count_nonzero(student.grad) == 0


def test_half_precision_auxiliary_losses_have_finite_gradients():
    """Loss statistics run in FP32 even when existing logits arrive in half precision."""
    teacher = torch.full((1, 8, 8, 8), 8.0, dtype=torch.float16, requires_grad=True)
    student = torch.zeros_like(teacher, requires_grad=True)
    target = torch.zeros_like(teacher)
    target[:, 2, 2:6, 2:6] = 1
    target[:, 5, 3:7, 3:7] = 1
    consistency = consistency_loss(student, teacher, target)
    overlap = overlap_pair_loss(student, target, [(2, 5)])
    boundary = soft_boundary(student.sigmoid()).square().mean()
    loss = consistency + overlap + boundary
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert teacher.grad is None
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_overlap_pair_loss_rewards_both_labels_at_true_overlap():
    """A real D/R overlap should improve only when both labels are predicted correctly."""
    target = torch.zeros(1, 8, 8, 8)
    target[:, 2, 2:6, 2:6] = 1
    target[:, 5, 3:7, 3:7] = 1
    good = torch.full((1, 8, 4, 4), 8.0, requires_grad=True)
    missed = torch.full((1, 8, 4, 4), -8.0, requires_grad=True)
    good_loss = overlap_pair_loss(good, target, [(2, 5)])
    missed_loss = overlap_pair_loss(missed, target, [(2, 5)])
    assert torch.isfinite(good_loss) and good_loss < missed_loss
    missed_loss.backward()
    assert torch.isfinite(missed.grad).all()
    assert missed.grad[:, 2].sum() < 0 and missed.grad[:, 5].sum() < 0
    assert torch.count_nonzero(missed.grad[:, [0, 1, 3, 4, 6, 7]]) == 0


@pytest.mark.parametrize("case", ["empty", "dent-only", "adjacent"])
def test_overlap_pair_loss_does_not_invent_cooccurrence(case):
    """Intersect native masks before pooling: adjacent classes may share a coarse cell but not a pixel."""
    target = torch.zeros(1, 8, 8, 8)
    if case != "empty":
        target[:, 2, :, :4] = 1
    if case == "adjacent":
        target[:, 5, :, 4:] = 1
        pooled = resize_presence(target, (1, 1))
        assert pooled[0, 2, 0, 0] == 1 and pooled[0, 5, 0, 0] == 1
    logits = torch.zeros(1, 8, 1, 1, requires_grad=True)
    loss = overlap_pair_loss(logits, target, [(2, 5)])
    assert loss.item() == 0
    loss.backward()
    assert logits.grad is not None and torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize(
    "invalid_config",
    [
        {"typo": {}},
        {"shadow": {"enabled": "true"}},
        {"shadow": {"strength": [0.1, 1.0]}},
        {"boundary": {"gain": -0.1}},
        {"overlap": {"gain": float("nan")}},
        {"consistency": {"gain": 0.2}},
    ],
)
def test_invalid_training_config_fails_explicitly(invalid_config):
    """Misspelled or inconsistent ablations must not silently run a different experiment."""
    with pytest.raises(ValueError):
        parse_training_aux(invalid_config)


def test_overlap_class_names_follow_dataset_and_missing_names_fail():
    """Dataset order can change: D/R are resolved by names, not fixed numeric IDs."""
    assert resolve_overlap_pairs([["D", "R"]], {0: "R", 1: "D"}, 2) == [(0, 1)]
    with pytest.raises(ValueError, match="absent"):
        resolve_overlap_pairs([["D", "R"]], {0: "D", 1: "H"}, 2)


def test_boundary_constraint_reaches_main_mask_parameters():
    """The new term must train actual coefficients/prototypes, not an unused auxiliary edge map."""
    model = configured_model(variant_cfg("boundary"))
    criterion = model.init_criterion()
    target = torch.zeros(1, 8, 8)
    target[:, 2:6, 2:6] = 1
    coefficients = torch.randn(1, 2, requires_grad=True)
    prototypes = torch.randn(2, 8, 8, requires_grad=True)
    boxes, area = torch.tensor([[2.0, 2.0, 6.0, 6.0]]), torch.tensor([0.25])
    original = criterion.single_mask_loss(target, coefficients, prototypes, boxes, area)
    criterion._training_terms = True
    extra = criterion.single_mask_loss(target, coefficients, prototypes, boxes, area) - original
    assert extra > 0
    grads = torch.autograd.grad(extra, (coefficients, prototypes))
    assert all(torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in grads)


@pytest.mark.parametrize("variant", VARIANTS)
def test_ablation_yaml_preserves_original_graph_and_cooccurrence(variant):
    """Only training_aux may distinguish variants: do not relabel existing overlap loss as new."""
    original, variant_yaml = YAML.load(BASE_CFG), YAML.load(variant_cfg(variant))
    for key in ("nc", "scales", "backbone", "head"):
        assert variant_yaml[key] == original[key]
    auxiliary = variant_yaml["training_aux"]
    assert auxiliary["version"] == 1
    assert auxiliary["shadow"]["enabled"] == (variant in {"shadow-aug", "consistency", "shadow-boundary", "full"})
    assert (auxiliary["consistency"]["gain"] > 0) == (variant in {"consistency", "shadow-boundary", "full"})
    assert (auxiliary["boundary"]["gain"] > 0) == (variant in {"boundary", "shadow-boundary", "full"})
    assert (auxiliary["overlap"]["gain"] > 0) == (variant in {"overlap", "full"})


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_variant_forward_backward_batch_unchanged_and_one_network_pass(variant, reference_model):
    """Each runnable YAML retains the baseline state while training its explicit auxiliary constraints."""
    model = configured_model(variant_cfg(variant)).train()
    reference_state, new_state = reference_model.state_dict(), model.state_dict()
    assert reference_state.keys() == new_state.keys()
    assert all(value.shape == new_state[key].shape for key, value in reference_state.items())
    assert len(model.model) == 47
    assert type(model.model[0]) is IIMStem
    assert type(model.model[-1]) is Segment26MultiLabel
    assert model.stride.tolist() == [8.0, 4.0, 16.0, 32.0, 64.0]
    original_keys = set(new_state)
    del new_state, reference_state
    batch = make_batch()
    original_batch = {key: value.clone() for key, value in batch.items()}
    observed_batch_sizes = []
    handle = model.model[0].register_forward_pre_hook(
        lambda _module, args: observed_batch_sizes.append(args[0].shape[0])
    )
    try:
        loss, items = model(batch)
    finally:
        handle.remove()
    assert isinstance(model.criterion, v8MultiLabelSegmentationLoss)
    if variant != "baseline":
        assert isinstance(model.criterion, TrainingAuxSegmentationLoss)
    expected_batch = 2 if model.yaml["training_aux"]["shadow"]["enabled"] else 1
    assert observed_batch_sizes == [expected_batch]
    assert loss.numel() == 7 and items.numel() == 7
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.sum().backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert any(grad.abs().sum() > 0 for grad in grads)
    assert all(torch.equal(batch[key], value) for key, value in original_batch.items())
    assert set(model.state_dict()) == original_keys
    # Storing live prediction tensors or the parent model in the criterion breaks model deepcopy/EMA.
    assert not any(isinstance(value, SegmentationModel) for value in vars(model.criterion).values())
    clone = deepcopy(model).eval()
    with torch.no_grad():
        output = clone(torch.rand(1, 3, 128, 192) * 0.05)
    assert output[0][1].shape == (1, 32, 32, 48)
    assert torch.isfinite(output[0][0]).all() and torch.isfinite(output[0][1]).all()


def test_all_off_matches_baseline_loss_and_prediction(reference_model):
    """Ablation baseline must be numerically identical, not merely another similar architecture."""
    baseline = deepcopy(reference_model).train()
    model = configured_model(variant_cfg("baseline")).train()
    model.load_state_dict(baseline.state_dict(), strict=True)
    batch = make_batch()
    original_loss, original_items = baseline(batch)
    new_loss, new_items = model(batch)
    torch.testing.assert_close(new_loss, original_loss, rtol=0, atol=0)
    torch.testing.assert_close(new_items, original_items, rtol=0, atol=0)
    baseline.eval()
    model.eval()
    with torch.no_grad():
        original_predictions = baseline(batch["img"])
        new_predictions = model(batch["img"])
    torch.testing.assert_close(new_predictions[0][0], original_predictions[0][0], rtol=0, atol=0)
    torch.testing.assert_close(new_predictions[0][1], original_predictions[0][1], rtol=0, atol=0)


def test_full_empty_batch_backward_and_ema():
    """All-background images remain trainable and checkpoint/EMA operations remain safe after backward."""
    model = configured_model(variant_cfg("full")).train()
    batch = make_batch(empty=True)
    loss, items = model(batch)
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.sum().backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    ema = ModelEMA(model)
    ema.update(model)
    assert ema.ema.state_dict().keys() == model.state_dict().keys()


@pytest.mark.parametrize("variant,slot", [("boundary", 1), ("overlap", 4)])
def test_training_terms_actually_contribute_to_expected_loss_slot(variant, slot):
    """The YAML switches must change the supervised objective, not merely parse successfully."""
    model = configured_model(variant_cfg(variant)).train()
    batch = make_batch()
    with torch.no_grad():
        raw_predictions = model(batch["img"])
        _, original_items = v8MultiLabelSegmentationLoss(model)(raw_predictions, batch)
        _, auxiliary_items = model.loss(batch, preds=raw_predictions)
    assert auxiliary_items[slot] > original_items[slot]
    other_slots = [index for index in range(7) if index != slot]
    torch.testing.assert_close(auxiliary_items[other_slots], original_items[other_slots], rtol=0, atol=0)


def test_shadow_training_rejects_precomputed_predictions(reference_model):
    """A compiled/precomputed single-view route cannot silently skip paired-view supervision."""
    model = configured_model(variant_cfg("consistency")).train()
    batch = make_batch()
    with torch.no_grad():
        predictions = model(batch["img"])
    with pytest.raises(ValueError, match="compile=False"):
        model.loss(batch, preds=predictions)


def test_full_eval_has_no_shadow_pass_or_auxiliary_loss(reference_model):
    """Validation uses one unmodified view and the same baseline metric loss, even after training initialization."""
    model = configured_model(variant_cfg("full"))
    model.load_state_dict(reference_model.state_dict(), strict=True)
    model.criterion = model.init_criterion()
    model.eval()
    batch = make_batch()
    observed_batch_sizes = []
    handle = model.model[0].register_forward_pre_hook(
        lambda _module, args: observed_batch_sizes.append(args[0].shape[0])
    )
    try:
        with torch.no_grad():
            predictions = model(batch["img"])
            actual_loss, actual_items = model.loss(batch, preds=predictions)
            expected_loss, expected_items = v8MultiLabelSegmentationLoss(model)(predictions, batch)
            original_predictions = reference_model(batch["img"])
    finally:
        handle.remove()
    assert observed_batch_sizes == [1]
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(actual_items, expected_items, rtol=0, atol=0)
    torch.testing.assert_close(predictions[0][0], original_predictions[0][0], rtol=0, atol=0)
    torch.testing.assert_close(predictions[0][1], original_predictions[0][1], rtol=0, atol=0)
