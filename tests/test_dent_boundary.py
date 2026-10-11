# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Boundary target semantics, native P2 decoding, optional DCNv2, and full-model regression tests."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.dent_boundary import (
    BoundaryGuidedDentFusion,
    ModulatedShapeConv,
    NativeP2Proto,
    Segment26MultiLabelBoundary,
)
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import YAML
from ultralytics.utils.dent_boundary_loss import v8BoundaryMultiLabelSegmentationLoss
from ultralytics.utils.loss import MultiChannelDiceLoss

CFG_DIR = Path("ultralytics/cfg/models/11_myself")
MODEL_CFG = CFG_DIR / "yolo11-10csar-dr-boundary-p2-0915.yaml"
DCN_CFG = CFG_DIR / "yolo11-10csar-dr-boundary-p2-dcn-0915.yaml"
SHADOW_CFG = CFG_DIR / "yolo11-10csar-dr-shadow-0915.yaml"


@pytest.fixture
def target_builder():
    """Exercise targets without repeatedly constructing the backbone."""
    criterion = v8BoundaryMultiLabelSegmentationLoss.__new__(v8BoundaryMultiLabelSegmentationLoss)
    criterion.nc, criterion.device = 8, torch.device("cpu")
    criterion.multilabel_dice = MultiChannelDiceLoss(smooth=1)
    return criterion


def test_boundary_overlap_and_same_class_interfaces(target_builder):
    masks = torch.zeros(3, 20, 24)
    masks[0, 4:16, 3:10] = 1
    masks[1, 4:16, 10:18] = 1  # Touching D instances: retain the interface at columns 9/10.
    masks[2, 4:16, 3:10] = 1  # R co-located with the first D: both class targets must survive.
    batch = {"masks": masks, "cls": torch.tensor([[2], [2], [5]]), "batch_idx": torch.zeros(3)}
    target = target_builder.build_boundary_target(batch, 2, (20, 24), torch.float32)
    assert target.shape == (2, 8, 20, 24)
    assert target[0, 2, 10, 9:11].eq(1).all()
    assert target[0, [2, 5], 4, 5].eq(1).all()
    assert target[0, 2, 10, 6] == 0  # Interior, not an edge.
    assert not target[0, [0, 1, 3, 4, 6, 7]].any()
    assert not target[1].any()
    assert torch.equal(batch["masks"], masks)  # Do not replace masks used by instance loss.
    coarse = target_builder.build_boundary_target(batch, 2, (10, 12), torch.float32)
    assert coarse[0, [2, 5]].sum() > 0


@pytest.mark.parametrize("size", [(20, 24), (10, 12), (40, 48)])
def test_boundary_empty_background_has_finite_loss(target_builder, size):
    batch = {"masks": torch.zeros(0, 20, 24), "cls": torch.zeros(0, 1), "batch_idx": torch.zeros(0)}
    target = target_builder.build_boundary_target(batch, 2, size, torch.float32)
    assert not target.any()
    logits = torch.zeros_like(target, requires_grad=True)
    loss = target_builder._boundary_loss(logits, target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad.min() > 0  # Negative examples penalize false boundaries.


def test_boundary_rejects_merged_instance_masks(target_builder):
    batch = {"masks": torch.zeros(1, 20, 24), "cls": torch.tensor([[2], [5]]), "batch_idx": torch.zeros(2)}
    with pytest.raises(ValueError, match="independent instance masks"):
        target_builder.build_boundary_target(batch, 1, (20, 24), torch.float32)


def test_dent_boundary_does_not_require_rust(target_builder):
    masks = torch.zeros(1, 20, 24)
    masks[0, 5:15, 6:18] = 1
    batch = {"masks": masks, "cls": torch.tensor([[2]]), "batch_idx": torch.zeros(1)}
    target = target_builder.build_boundary_target(batch, 1, (20, 24), torch.float32)
    assert target[0, 2].any() and not target[0, 5].any()
    good_logits = (target * 2 - 1) * 5
    assert target_builder._boundary_loss(good_logits, target) < target_builder._boundary_loss(-good_logits, target)


def test_native_p2_retains_sub_p3_detail():
    """Isolate the resize path: adjacent P2 pixels must not collapse to one P3 sample."""
    proto = NativeP2Proto((1, 1, 1), c_=8, c2=1, nc=2).eval()
    proto.feat_refine = nn.ModuleList([nn.Identity(), nn.Identity()])
    proto.feat_fuse = proto.cv1 = proto.cv2 = proto.cv3 = nn.Identity()
    p2 = torch.zeros(1, 1, 8, 12)
    p2[0, 0, 3, 5] = 1
    output = proto([torch.zeros(1, 1, 4, 6), p2, torch.zeros(1, 1, 2, 3)])
    assert isinstance(proto.upsample, nn.Identity)
    assert torch.equal(output, p2)


@pytest.mark.parametrize("use_dcn", [False, True])
def test_guided_shape_gate_and_gradients(use_dcn):
    torch.manual_seed(123)
    module = BoundaryGuidedDentFusion([8, 4], 8, 16, nc=8, detail_scale=0.0, use_dcn=use_dcn).eval()
    neck = torch.randn(2, 8, 12, 16, requires_grad=True)
    shallow = torch.randn(2, 4, 12, 16, requires_grad=True)
    context = torch.randn(2, 16, 3, 4, requires_grad=True)
    assert torch.equal(module([neck, shallow, context])[0], neck)
    with torch.no_grad():
        module.detail_scale.fill_(0.1)
    fused, boundary = module([neck, shallow, context])
    assert boundary.shape == (2, 8, 12, 16)
    (fused.square().mean() + boundary.square().mean()).backward()
    for tensor in (neck, shallow, context):
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0
    for parameter in module.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    # Gates actually influence inference, not just an unused auxiliary prediction.
    with torch.no_grad():
        reference = module([neck, shallow, context])[0]
        module.boundary_head[-1].bias.fill_(15)
        assert not torch.equal(reference, module([neck, shallow, context])[0])
        module.semantic_gate.bias.fill_(15)
        assert not torch.equal(reference, module([neck, shallow, context])[0])
    if use_dcn:
        grad = module.shape_align.offset_mask.weight.grad
        assert grad[:18].abs().sum() > 0 and grad[18:].abs().sum() > 0


def test_dcn_half_precision_and_dark_input():
    module = ModulatedShapeConv(4).eval().half()
    image = torch.rand(1, 4, 12, 16).half() * 0.01
    image[:, :, :4] = 0
    with torch.no_grad():
        output = module(image)
    assert output.dtype == image.dtype and torch.isfinite(output).all()


def test_yaml_graph_and_0915_weight_compatibility():
    cfg, old_cfg, dcn_cfg = YAML.load(MODEL_CFG), YAML.load(SHADOW_CFG), YAML.load(DCN_CFG)
    assert cfg["backbone"] == old_cfg["backbone"]
    assert cfg["head"][:-1] == old_cfg["head"][:-1]
    assert cfg["head"][-1][0] == old_cfg["head"][-1][0]
    dcn_cfg["head"][-1][3][-1] = False
    assert cfg == dcn_cfg  # Only the DCN boolean differs, not the training graph elsewhere.
    old = SegmentationModel(SHADOW_CFG, nc=8, verbose=False)
    model = SegmentationModel(MODEL_CFG, nc=8, verbose=False)
    old_state, new_state = old.state_dict(), model.state_dict()
    removed = {"model.46.proto.upsample.weight", "model.46.proto.upsample.bias"}
    assert set(old_state) - set(new_state) == removed
    assert all(value.shape == new_state[key].shape for key, value in old_state.items() if key not in removed)
    incompatible = model.load_state_dict(old_state, strict=False)
    assert set(incompatible.unexpected_keys) == removed
    assert incompatible.missing_keys
    model.args = get_cfg(overrides={"overlap_mask": True})
    with pytest.raises(ValueError, match="overlap_mask=False"):
        model.init_criterion()


def _batch(empty=False):
    image = torch.rand(1, 3, 128, 128) * 0.08
    image[:, :, :24] = 0
    masks = torch.zeros(2, 32, 32)
    masks[0, 6:23, 5:21] = 1
    masks[1, 12:29, 12:27] = 1
    return {
        "img": image,
        "batch_idx": torch.zeros(0 if empty else 2),
        "cls": torch.zeros(0, 1) if empty else torch.tensor([[2.0], [5.0]]),
        "bboxes": torch.zeros(0, 4)
        if empty
        else torch.tensor([[0.40625, 0.453125, 0.5, 0.53125], [0.609375, 0.640625, 0.46875, 0.53125]]),
        "masks": masks[:0] if empty else masks,
        "heatmaps": torch.zeros(1, 8, 32, 32),
        "seedmaps": torch.zeros(1, 8, 32, 32),
    }


@pytest.mark.parametrize("cfg", [MODEL_CFG, DCN_CFG])
@pytest.mark.parametrize("empty", [False, True])
def test_full_model_loss_backward_eval_and_fusion(cfg, empty):
    model = SegmentationModel(cfg, nc=8, verbose=False)
    model.args = get_cfg(overrides={"overlap_mask": False})
    assert len(model.model) == 47
    assert isinstance(model.model[-1], Segment26MultiLabelBoundary)
    assert model.stride.tolist() == [8.0, 4.0, 16.0, 32.0, 64.0]
    loss, items = model(_batch(empty))
    assert isinstance(model.criterion, v8BoundaryMultiLabelSegmentationLoss)
    assert loss.shape == items.shape == (7,)
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.sum().backward()
    for fusion in model.model[-1].detail_fusion:
        for parameter in fusion.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert fusion.boundary_head[-1].weight.grad.abs().sum() > 0
    clone = deepcopy(model).eval()  # No graph tensors retained as module attributes (EMA/checkpointing).
    image = torch.rand(1, 3, 128, 192) * 0.05
    with torch.no_grad():
        validation_batch = _batch(empty)
        validation_preds = clone(validation_batch["img"])[1]
        validation_loss, _ = clone.loss(validation_batch, validation_preds)
        assert torch.isfinite(validation_loss).all()  # Validation still sees explicit boundary logits.
        outputs = clone(image)
        assert outputs[0][1].shape == (1, 32, 32, 48)
        assert [x.shape[-2:] for x in outputs[1]["boundary_logits"]] == [(16, 24), (32, 48)]
        assert torch.isfinite(outputs[0][0]).all() and torch.isfinite(outputs[0][1]).all()
        fused = clone.fuse(verbose=False)(image)
        assert torch.allclose(outputs[0][1], fused[0][1], atol=1e-4, rtol=1e-4)
        assert torch.allclose(outputs[0][0], fused[0][0], atol=1e-3, rtol=1e-4)
        clone.model[-1].export = True
        clone.model[-1].format = "onnx"
        export_output = clone(image)
        assert len(export_output) == 2 and all(isinstance(tensor, torch.Tensor) for tensor in export_output)
        assert export_output[1].shape == (1, 32, 32, 48)
