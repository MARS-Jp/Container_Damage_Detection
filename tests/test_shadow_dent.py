# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Check shadow detail behavior, checkpoint compatibility, and segmentation gradients."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import Segment26MultiLabelShadow, ShadowIIMStem
from ultralytics.nn.modules.conv import IIMStem, IlluminationInvariantConv
from ultralytics.nn.modules.shadow_dent import DentDetailFusion, LuminanceDetail
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import YAML
from ultralytics.utils.loss import v8MultiLabelSegmentationLoss

MODEL_CFG = Path("ultralytics/cfg/models/11_myself/yolo11-10csar-dr-shadow-0915.yaml")
BASE_CFG = Path("ultralytics/cfg/models/11_myself/yolo11-10csar-dr-msat-0911.yaml")


@pytest.mark.parametrize("brightness", [0.0, 1e-7, 0.03, 0.5, 1.0])
def test_detail_uniform_surfaces_and_black_input(brightness):
    """Uniform surfaces, including borders, must not turn into shape cues or non-finite gradients."""
    image = torch.full((1, 3, 25, 31), brightness, requires_grad=True)
    detail = LuminanceDetail()(image)
    assert detail.shape == (1, 5, 25, 31)
    assert torch.isfinite(detail).all()
    assert detail[:, 1:].abs().max() < 2e-5
    detail.square().mean().backward()
    assert torch.isfinite(image.grad).all()


def test_detail_keeps_neutral_shading_cancelled_by_channel_pairs():
    """A synthetic achromatic depression-like intensity pattern survives the new branch in shade."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 41), torch.linspace(-1, 1, 41), indexing="ij")
    luminance = 0.1 - 0.05 * torch.exp(-8 * (xx.square() + yy.square()))
    image = luminance[None, None].repeat(1, 3, 1, 1)
    with torch.no_grad():
        iim = IlluminationInvariantConv(4, 3).eval()(image)
        detail = LuminanceDetail()(image)
    assert iim.abs().max() < 1e-6
    assert detail[:, 1:3].abs().max() > 0.01
    assert detail[:, 3:].abs().max() > 0.01
    assert detail[:, 3:].min() < 0 and detail[:, 3:].max() > 0


def test_detail_half_precision_black_and_low_light():
    """Statistics remain finite when a saved model and input are converted to half precision."""
    module = LuminanceDetail().half()
    image = torch.rand(2, 3, 24, 32).half() * 0.01
    image[:, :, :8] = 0
    output = module(image)
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    assert output.abs().max() <= 1


def test_stem_loads_old_weights_and_can_disable_residual():
    """Warm-start from the old stem without renaming or discarding RGB/IIM weights."""
    original = IIMStem(3, 16).eval()
    shadow = ShadowIIMStem(3, 16, detail_scale=0.0).eval()
    incompatible = shadow.load_state_dict(original.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(key.startswith(("detail_", "luminance_detail.")) for key in incompatible.missing_keys)
    image = torch.rand(2, 3, 32, 48)
    with torch.no_grad():
        assert torch.allclose(original(image), shadow(image), atol=1e-6)


def test_fusion_preserves_neck_and_learns_shallow_detail():
    """Zero gain is an identity; a small nonzero gain sends gradients through the bypass."""
    fusion = DentDetailFusion([16, 8], 16, detail_scale=0.0)
    neck = torch.randn(2, 16, 16, 24, requires_grad=True)
    shallow = torch.randn(2, 8, 16, 24, requires_grad=True)
    assert torch.equal(fusion([neck, shallow]), neck)
    with torch.no_grad():
        fusion.detail_scale.fill_(0.1)
    fusion([neck, shallow]).square().mean().backward()
    assert shallow.grad.abs().sum() > 0 and torch.isfinite(shallow.grad).all()
    assert torch.isfinite(neck.grad).all()


def test_model_compatibility_loss_backward_and_eval():
    """Preserve all old state keys and verify independent D/R masks train the added branches."""
    cfg = YAML.load(MODEL_CFG)
    base_cfg = YAML.load(BASE_CFG)
    assert cfg["backbone"][1:] == base_cfg["backbone"][1:]
    assert cfg["head"][:-1] == base_cfg["head"][:-1]
    assert cfg["head"][-1][0] == base_cfg["head"][-1][0] + [6, 3]

    base = SegmentationModel(BASE_CFG, ch=3, nc=8, verbose=False)
    model = SegmentationModel(MODEL_CFG, ch=3, nc=8, verbose=False)
    old_state, new_state = base.state_dict(), model.state_dict()
    assert all(key in new_state and value.shape == new_state[key].shape for key, value in old_state.items())
    incompatible = model.load_state_dict(old_state, strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    del base
    assert len(model.model) == 47
    assert isinstance(model.model[0], ShadowIIMStem)
    assert isinstance(model.model[-1], Segment26MultiLabelShadow)
    assert model.stride.tolist() == [8.0, 4.0, 16.0, 32.0, 64.0]

    model.args = get_cfg(overrides={"overlap_mask": False})
    model.train()
    image = torch.rand(1, 3, 128, 128) * 0.08
    image[:, :, :24] = 0
    masks = torch.zeros(2, 32, 32)
    masks[0, 6:23, 5:21] = 1
    masks[1, 12:29, 12:27] = 1
    batch = {
        "img": image,
        "batch_idx": torch.tensor([0.0, 0.0]),
        "cls": torch.tensor([[2.0], [5.0]]),  # D and R
        "bboxes": torch.tensor([[0.40625, 0.453125, 0.5, 0.53125], [0.609375, 0.640625, 0.46875, 0.53125]]),
        "masks": masks,
        "heatmaps": torch.zeros(1, 8, 32, 32),
        "seedmaps": torch.zeros(1, 8, 32, 32),
    }
    loss, items = model(batch)
    assert isinstance(model.criterion, v8MultiLabelSegmentationLoss)
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.sum().backward()
    added_branches = [model.model[0].detail_stem, *model.model[-1].detail_fusion]
    for branch in added_branches:
        grads = [param.grad for param in branch.parameters() if param.requires_grad]
        assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
        assert any(grad.abs().sum() > 0 for grad in grads)
    # Catch graph tensors accidentally retained in modules, which can break EMA/checkpointing.
    clone = deepcopy(model).eval()
    with torch.no_grad():
        output = clone(torch.rand(1, 3, 128, 192) * 0.05)
    assert output[0][1].shape == (1, 32, 32, 48)
    assert torch.isfinite(output[0][0]).all() and torch.isfinite(output[0][1]).all()
