# YOLO11 10-CSAR Dent-Rust Relation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以 `yolo11-10csar-0911.yaml` 為固定主幹，新增能利用 D（凹洞）與 R（生鏽）空間共現訊號的高解析度關係閘門，提高伴隨生鏽之凹洞的召回率，同時維持單獨凹洞的辨識能力。

**Architecture:** 完整保留 IIMStem、五尺度第一階段 CSAR、FSNetShuffle 與五尺度第二階段 CSAR（全域層 `0–30`）。在第二階段的 P2、P3、P4 後各加入一個 `DentRustRelationGate`，輸出增強特徵以及 D、R、D-near-R 三張輔助圖；`Segment26DentRust` 使用增強後的五尺度特徵做原有實例分割，專用 loss 則用獨立 instance masks 監督三張關係圖。關係訊號只以零初始化、有界的 residual gate 增強特徵，不把「存在 R」設成預測 D 的必要條件。

**Tech Stack:** Python 3.10、PyTorch、Ultralytics YOLO11 instance segmentation、OpenCV、PyYAML、pytest。

**Spec:** `ultralytics/cfg/models/11_myself/yolo11-10csar-0911.yaml`

## Global Constraints

- 不修改 `ultralytics/cfg/models/11_myself/yolo11-10csar-0911.yaml`；它是所有實驗的 immutable baseline。
- 新模型的全域層 `0–30` 必須與 baseline 的 `backbone + head` 前 31 列完全相同，包括 `from`、repeats、module 與 args。
- 固定類別對應 `{0:B, 1:C, 2:D, 3:H, 4:O, 5:R, 6:X, 7:U}`；訓練前仍須從 dataset YAML 驗證 D=`2`、R=`5`，不符時直接停止。
- 保留五個 Segment26 feature levels，順序固定為 P3、P2、P4、P5、P6；P3 必須排第一，確保 Proto26 產生 stride-4 masks。
- 關係閘門只放在 P2、P3、P4；P5/P6 保留全域語意，不增加粗尺度的 D/R shortcut。
- `DentRustRelationGate` 初始 forward 必須等同 identity；新模組不得在載入 baseline 權重後立即破壞既有預測。
- D 通道必須監督所有凹洞；D-near-R 通道只監督與 R 相交或距離小於等於 16 個輸入影像 pixels 的凹洞區域。
- 訓練固定 `overlap_mask=False`，讓同一 pixel 的 D 與 R instance masks 能同時存在；若收到 merged masks，loss 必須提早拋出錯誤。
- 第一輪只比較 baseline 與一個完整架構版本；在架構版本通過驗收前，不進行大範圍超參數搜尋。
- 第一輪共同設定為 `imgsz=640`、`batch=8`、`seed=0`、`deterministic=True`、`patience=0`、`mask_ratio=4`、`cls_pw=0.5`；兩個模型使用相同 split、初始化權重與訓練 epochs。
- 驗證 split 不得與 train split 指向同一批影像；若 pair-aware validation 中少於 50 個 D-near-R ground-truth instances，結果只標記為 preliminary，不宣稱架構勝出。
- 不新增第三方 Python dependency；使用專案既有的 PyTorch、Ultralytics、OpenCV 與 YAML 工具。

## Architecture Decisions

- 類別專屬特徵與明確共現關係是本次主要改動。這和 [Query2Label](https://arxiv.org/abs/2107.10834) 用 label-specific query 取得局部辨識特徵，以及 [SSGRL](https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_Learning_Semantic-Specific_Graph_Representation_for_Multi-Label_Image_Recognition_ICCV_2019_paper.html) 顯式建模 label co-occurrence 的方向一致。
- 使用 residual gate 而非把 R probability 直接乘到 D score；[Contextual Debiasing](https://www.openaccess.thecvf.com/content/CVPR2022/papers/Liu_Contextual_Debiasing_for_Visual_Recognition_With_Causal_Mechanisms_CVPR_2022_paper.pdf) 指出過度依賴共現背景會形成 context shortcut，因此驗收必須同時檢查 D-with-R 與 D-without-R。
- 關係輔助 loss 沿用專案已有的 ASL 思路；[Asymmetric Loss](https://arxiv.org/abs/2009.14119) 專門降低大量 easy negatives 對稀有 multi-label positives 的壓制。

## File Structure

- Keep `ultralytics/cfg/models/11_myself/yolo11-10csar-0911.yaml`: immutable baseline graph and experiment spec.
- Create `ultralytics/cfg/models/11_myself/yolo11-10csar-dr-0911.yaml`: baseline layers `0–30` plus three D/R relation gates and `Segment26DentRust`.
- Create `ultralytics/nn/modules/dent_rust.py`: focused P2/P3/P4 relation feature gate.
- Modify `ultralytics/nn/modules/__init__.py`: export `DentRustRelationGate` and `Segment26DentRust`.
- Modify `ultralytics/nn/modules/head.py`: package five detection features and three relation-logit maps in a Segment26-compatible head.
- Modify `ultralytics/nn/tasks.py`: allow the YAML parser to construct the new gate and head.
- Create `ultralytics/utils/dent_rust.py`: strict class-ID resolution and reusable D/R/D-near-R target construction.
- Create `tools/train_yolo11_10csar_dr_0911.py`: specialized relation loss, warm-start remapping, portable checkpoint saving, and reproducible CLI.
- Create `tools/eval_dent_rust_0911.py`: pair-stratified D metrics and baseline/candidate comparison JSON.
- Create `tests/test_yolo11_10csar_dr_0911.py`: graph, gate, target, loss, checkpoint, and evaluation regression coverage.

---

### Task 1: Freeze the Baseline and Build Pair Targets

**Files:**

- Keep: `ultralytics/cfg/models/11_myself/yolo11-10csar-0911.yaml`
- Create: `ultralytics/utils/dent_rust.py`
- Create: `tests/test_yolo11_10csar_dr_0911.py`

**Interfaces:**

- Consumes: baseline YAML and independent instance masks shaped `[N, H, W]`.
- Produces: `resolve_dent_rust_ids(names) -> tuple[int, int]` and `build_dent_rust_targets(masks, classes, batch_indices, batch_size, size, dent_class, rust_class, radius_cells) -> Tensor[B, 3, H, W]` with channel order D, R, D-near-R.

- [ ] **Step 1: Add the baseline characterization test**

```python
from pathlib import Path

import torch

from ultralytics.nn.tasks import yaml_model_load

ROOT = Path(__file__).resolve().parents[1]
BASE_CFG = ROOT / "ultralytics/cfg/models/11_myself/yolo11-10csar-0911.yaml"
BASE_CFG = ROOT / "ultralytics/cfg/models/11_myself/yolo11-10csar-0911.yaml"
DR_CFG = ROOT / "ultralytics/cfg/models/11_myself/yolo11-10csar-dr-0911.yaml"


def test_0911_baseline_is_the_expected_iim_ten_csar_graph():
    config = yaml_model_load(BASE_CFG)
    layers = config["backbone"] + config["head"]
    assert layers[0][2] == "IIMStem"
    assert sum(layer[2] == "CSAR" for layer in layers) == 10
    assert [layers[index][2] for index in range(21, 26)] == ["FSNetShuffle"] * 5
    assert layers[31][2] == "Segment26"
    assert layers[31][0] == [27, 26, 28, 29, 30]
```

- [ ] **Step 2: Run the characterization test**

Run: `pytest tests/test_yolo11_10csar_dr_0911.py::test_0911_baseline_is_the_expected_iim_ten_csar_graph -v`

Expected: PASS. If it fails, stop because the implementation spec has changed.

- [ ] **Step 3: Add failing tests for class mapping and pair-target semantics**

```python
from ultralytics.utils.dent_rust import build_dent_rust_targets, resolve_dent_rust_ids


def test_dent_rust_ids_are_resolved_strictly():
    names = {0: "B", 1: "C", 2: "D", 3: "H", 4: "O", 5: "R", 6: "X", 7: "U"}
    assert resolve_dent_rust_ids(names) == (2, 5)


def test_pair_target_keeps_standalone_dent_positive_without_marking_it_as_pair():
    masks = torch.zeros(3, 32, 32)
    masks[0, 4:12, 4:12] = 1  # D near R
    masks[1, 8:16, 10:18] = 1  # R
    masks[2, 22:29, 22:29] = 1  # standalone D
    classes = torch.tensor([2, 5, 2])
    batch_indices = torch.zeros(3, dtype=torch.long)
    target = build_dent_rust_targets(
        masks, classes, batch_indices, batch_size=1, size=(32, 32), dent_class=2, rust_class=5, radius_cells=2
    )
    assert target.shape == (1, 3, 32, 32)
    assert target[0, 0, 25, 25] == 1  # all D remains supervised
    assert target[0, 2, 25, 25] == 0  # distant D is not D-near-R
    assert target[0, 2, 9, 10] == 1  # D pixel near R is pair-positive
```

- [ ] **Step 4: Run the target tests and verify the missing module fails**

Run: `pytest tests/test_yolo11_10csar_dr_0911.py -k "dent_rust_ids or pair_target" -v`

Expected: FAIL with `ModuleNotFoundError: ultralytics.utils.dent_rust`.

- [ ] **Step 5: Implement the reusable target utility**

Create `ultralytics/utils/dent_rust.py` with:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


def resolve_dent_rust_ids(names: Mapping[int, str] | Sequence[str]) -> tuple[int, int]:
    items = names.items() if isinstance(names, Mapping) else enumerate(names)
    normalized = {str(name).strip().upper(): int(index) for index, name in items}
    if normalized.get("D") != 2 or normalized.get("R") != 5:
        raise ValueError(f"Expected dataset classes D=2 and R=5, received {normalized}.")
    return 2, 5


def build_dent_rust_targets(
    masks: torch.Tensor,
    classes: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_size: int,
    size: tuple[int, int],
    dent_class: int,
    rust_class: int,
    radius_cells: int,
) -> torch.Tensor:
    classes = classes.reshape(-1).long()
    batch_indices = batch_indices.reshape(-1).long()
    if masks.ndim != 3 or masks.shape[0] != classes.numel() or classes.numel() != batch_indices.numel():
        raise ValueError("Dent/rust targets require one independent binary mask per labeled instance.")
    if radius_cells < 0:
        raise ValueError("radius_cells must be non-negative.")

    masks = masks.float().unsqueeze(1)
    if masks.shape[-2] >= size[0] and masks.shape[-1] >= size[1]:
        masks = F.adaptive_max_pool2d(masks, size)
    else:
        masks = F.interpolate(masks, size=size, mode="nearest")
    masks = masks[:, 0]

    dent = masks.new_zeros((batch_size, *size))
    rust = masks.new_zeros((batch_size, *size))
    for image_index in range(batch_size):
        image_rows = batch_indices == image_index
        dent_rows = image_rows & (classes == dent_class)
        rust_rows = image_rows & (classes == rust_class)
        if dent_rows.any():
            dent[image_index] = masks[dent_rows].amax(dim=0)
        if rust_rows.any():
            rust[image_index] = masks[rust_rows].amax(dim=0)

    if radius_cells:
        kernel = 2 * radius_cells + 1
        rust_near = F.max_pool2d(rust.unsqueeze(1), kernel, stride=1, padding=radius_cells)[:, 0]
    else:
        rust_near = rust
    pair = dent * rust_near
    return torch.stack((dent, rust, pair), dim=1)
```

- [ ] **Step 6: Run the focused tests**

Run: `pytest tests/test_yolo11_10csar_dr_0911.py -k "baseline or dent_rust_ids or pair_target" -v`

Expected: PASS.

- [ ] **Step 7: Commit the frozen spec and target contract**

```bash
git add ultralytics/cfg/models/11_myself/yolo11-10csar-0911.yaml ultralytics/utils/dent_rust.py tests/test_yolo11_10csar_dr_0911.py
git commit -m "test: freeze 0911 ten-csar dent-rust baseline"
```

### Task 2: Add the High-Resolution Dent-Rust Relation Gate

**Files:**

- Create: `ultralytics/nn/modules/dent_rust.py`
- Modify: `ultralytics/nn/modules/__init__.py`
- Modify: `ultralytics/nn/tasks.py`
- Modify: `tests/test_yolo11_10csar_dr_0911.py`

**Interfaces:**

- Consumes: one P2, P3, or P4 feature tensor `[B, C, H, W]`.
- Produces: `DentRustRelationGate.forward(x) -> tuple[enhanced_feature, relation_logits]`; logits always have channels `[D, R, D-near-R]`.

- [ ] **Step 1: Write the failing identity-and-gradient test**

```python
from ultralytics.nn.modules import DentRustRelationGate


def test_relation_gate_starts_as_identity_and_both_paths_receive_gradients():
    gate = DentRustRelationGate(32, 32, hidden=16, neighborhood=5, init_gain=0.1)
    image_feature = torch.randn(2, 32, 24, 32, requires_grad=True)
    enhanced, logits = gate(image_feature)
    assert enhanced.shape == image_feature.shape
    assert logits.shape == (2, 3, 24, 32)
    assert torch.allclose(enhanced, image_feature, atol=1e-6)
    (enhanced.mean() + logits.mean()).backward()
    assert gate.modulator.weight.grad is not None
    assert gate.evidence.weight.grad is not None
    assert gate.relation.weight.grad is not None
```

- [ ] **Step 2: Run the test and verify the missing export fails**

Run: `pytest tests/test_yolo11_10csar_dr_0911.py::test_relation_gate_starts_as_identity_and_both_paths_receive_gradients -v`

Expected: FAIL because `DentRustRelationGate` is not exported.

- [ ] **Step 3: Implement the relation gate**

Create `ultralytics/nn/modules/dent_rust.py` with:

```python
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .conv import Conv

__all__ = ("DentRustRelationGate",)


class DentRustRelationGate(nn.Module):
    """Predict D/R/local-pair evidence and feed it back through a bounded residual gate."""

    def __init__(
        self,
        c1: int,
        c2: int,
        hidden: int = 64,
        neighborhood: int = 5,
        init_gain: float = 0.1,
    ):
        super().__init__()
        if neighborhood < 1 or neighborhood % 2 == 0:
            raise ValueError("neighborhood must be a positive odd integer.")
        if not 0.0 <= init_gain < 1.0:
            raise ValueError("init_gain must be in [0, 1).")
        self.neighborhood = int(neighborhood)
        self.feature_proj = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.context = nn.Sequential(Conv(c2, hidden, 3), Conv(hidden, hidden, 3))
        self.evidence = nn.Conv2d(hidden, 2, 1)
        self.relation = nn.Conv2d(hidden + 3, 1, 1)
        self.modulator = nn.Conv2d(3, c2, 1)
        self.gain_logit = nn.Parameter(torch.tensor(math.atanh(init_gain)))
        nn.init.zeros_(self.modulator.weight)
        nn.init.zeros_(self.modulator.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.feature_proj(x)
        context = self.context(feature)
        evidence_logits = self.evidence(context)
        dent_probability, rust_probability = evidence_logits.sigmoid().chunk(2, dim=1)
        padding = self.neighborhood // 2
        rust_near = F.max_pool2d(rust_probability, self.neighborhood, stride=1, padding=padding)
        interaction = dent_probability * rust_near
        pair_logit = self.relation(torch.cat((context, dent_probability, rust_near, interaction), dim=1))
        relation_logits = torch.cat((evidence_logits, pair_logit), dim=1)
        modulation = torch.tanh(self.modulator(relation_logits.sigmoid()))
        enhanced = feature * (1.0 + torch.tanh(self.gain_logit) * modulation)
        return enhanced, relation_logits
```

- [ ] **Step 4: Export and register the YAML module**

In `ultralytics/nn/modules/__init__.py`, import `DentRustRelationGate` from `.dent_rust` and add its name to `__all__`. In `ultralytics/nn/tasks.py`, import the name from `ultralytics.nn.modules` and add it to `base_modules`; this makes the parser prepend `c1`, scale `c2`, and retain `c2` for subsequent `Index` layers.

```python
# ultralytics/nn/modules/__init__.py
from .dent_rust import DentRustRelationGate

__all__ = ("DentRustRelationGate",)

# ultralytics/nn/tasks.py: ultralytics.nn.modules import list
(DentRustRelationGate,)

# ultralytics/nn/tasks.py: parse_model() base_modules
(DentRustRelationGate,)
```

- [ ] **Step 5: Run module and parser regression tests**

Run: `pytest tests/test_yolo11_10csar_dr_0911.py::test_relation_gate_starts_as_identity_and_both_paths_receive_gradients tests/test_msat.py tests/test_msat_multilabel.py -v`

Expected: PASS; existing MSAT parser behavior remains unchanged.

- [ ] **Step 6: Commit the isolated architecture primitive**

```bash
git add ultralytics/nn/modules/dent_rust.py ultralytics/nn/modules/__init__.py ultralytics/nn/tasks.py tests/test_yolo11_10csar_dr_0911.py
git commit -m "feat: add bounded dent-rust relation gate"
```

### Task 3: Add a Segment26 Head That Carries Relation Logits

**Files:**

- Modify: `ultralytics/nn/modules/head.py`
- Modify: `ultralytics/nn/modules/__init__.py`
- Modify: `ultralytics/nn/tasks.py`
- Modify: `tests/test_yolo11_10csar_dr_0911.py`

**Interfaces:**

- Consumes: five feature maps followed by three relation-logit maps.
- Produces: ordinary Segment26 predictions plus `preds["dent_rust_logits"]` during training.

- [ ] **Step 1: Write the failing packaging test**

```python
from ultralytics.nn.modules import Segment26DentRust


def test_segment26_dent_rust_packages_three_relation_levels_during_training():
    head = Segment26DentRust(
        nc=8,
        nm=32,
        npr=64,
        dent_class=2,
        rust_class=5,
        num_feature_levels=5,
        num_relation_levels=3,
        relation_loss_gain=0.25,
        relation_radii=(2, 4, 1),
        ch=(64, 32, 128, 256, 256, 3, 3, 3),
    ).train()
    features = [
        torch.randn(1, 64, 16, 16),
        torch.randn(1, 32, 32, 32),
        torch.randn(1, 128, 8, 8),
        torch.randn(1, 256, 4, 4),
        torch.randn(1, 256, 2, 2),
    ]
    relation_logits = [
        torch.randn(1, 3, 16, 16),
        torch.randn(1, 3, 32, 32),
        torch.randn(1, 3, 8, 8),
    ]
    predictions = head([*features, *relation_logits])
    assert all(actual is expected for actual, expected in zip(predictions["dent_rust_logits"], relation_logits))
    assert head.dent_class == 2 and head.rust_class == 5
```

- [ ] **Step 2: Run the test and verify the missing head fails**

Run: `pytest tests/test_yolo11_10csar_dr_0911.py::test_segment26_dent_rust_packages_three_relation_levels_during_training -v`

Expected: FAIL because `Segment26DentRust` does not exist.

- [ ] **Step 3: Implement the head beside `Segment26MultiLabel`**

Add to `ultralytics/nn/modules/head.py`:

```python
from __future__ import annotations


class Segment26DentRust(Segment26):
    """Segment26 head with explicit high-resolution dent/rust relation supervision."""

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        dent_class: int = 2,
        rust_class: int = 5,
        num_feature_levels: int = 5,
        num_relation_levels: int = 3,
        relation_loss_gain: float = 0.25,
        relation_radii: tuple[int, int, int] = (2, 4, 1),
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        if end2end:
            raise ValueError("Segment26DentRust does not support end2end mode.")
        if len(ch) != num_feature_levels + num_relation_levels:
            raise ValueError("Segment26DentRust expects five features followed by three relation maps.")
        if len(relation_radii) != num_relation_levels:
            raise ValueError("One relation radius is required for every relation-logit level.")
        if max(dent_class, rust_class) >= nc or min(dent_class, rust_class) < 0:
            raise ValueError("Dent and rust class indices must be valid dataset classes.")
        self.num_feature_levels = int(num_feature_levels)
        self.num_relation_levels = int(num_relation_levels)
        self.dent_class = int(dent_class)
        self.rust_class = int(rust_class)
        self.relation_loss_gain = float(relation_loss_gain)
        self.relation_radii = tuple(int(radius) for radius in relation_radii)
        super().__init__(nc, nm, npr, reg_max, end2end, ch[: self.num_feature_levels])

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        expected = self.num_feature_levels + self.num_relation_levels
        if len(x) != expected:
            raise ValueError(f"Segment26DentRust expected {expected} inputs, but received {len(x)}.")
        features = x[: self.num_feature_levels]
        relation_logits = x[self.num_feature_levels :]
        outputs = super().forward(features)
        if self.training:
            outputs["dent_rust_logits"] = relation_logits
        return outputs
```

- [ ] **Step 4: Export and register the head**

Add `Segment26DentRust` to the imports and `__all__` in `ultralytics/nn/modules/__init__.py`. In `ultralytics/nn/tasks.py`, import it and add it to both the Detect/Segment constructor set and the set whose prototype width `args[2]` is compound-scaled.

```python
# ultralytics/nn/modules/__init__.py: .head import list and __all__
(Segment26DentRust,)

# ultralytics/nn/tasks.py: ultralytics.nn.modules import list
(Segment26DentRust,)

# ultralytics/nn/tasks.py: Detect/Segment constructor set
(Segment26DentRust,)

# ultralytics/nn/tasks.py: prototype-width scaling set
(Segment26DentRust,)
```

- [ ] **Step 5: Run the focused head test and existing Segment26 tests**

Run: `pytest tests/test_yolo11_10csar_dr_0911.py::test_segment26_dent_rust_packages_three_relation_levels_during_training tests/test_yolo11_6csar.py tests/test_yolo11_6csar_iim.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the explicit-output head**

```bash
git add ultralytics/nn/modules/head.py ultralytics/nn/modules/__init__.py ultralytics/nn/tasks.py tests/test_yolo11_10csar_dr_0911.py
git commit -m "feat: carry dent-rust relation maps through segment26"
```

### Task 4: Wire the Relation Architecture into a New YAML

**Files:**

- Create: `ultralytics/cfg/models/11_myself/yolo11-10csar-dr-0911.yaml`
- Modify: `tests/test_yolo11_10csar_dr_0911.py`

**Interfaces:**

- Consumes: global layers `0–30` from the immutable baseline.
- Produces: a 41-layer model whose layers `31–39` are three relation gates plus their explicit outputs, and layer `40` is `Segment26DentRust`.

- [ ] **Step 1: Write the failing graph-preservation test**

```python
from ultralytics.nn.modules import DentRustRelationGate, Segment26DentRust
from ultralytics.nn.tasks import SegmentationModel


def test_dr_yaml_preserves_layers_zero_to_thirty_and_adds_only_relation_layers():
    base = yaml_model_load(BASE_CFG)
    candidate = yaml_model_load(DR_CFG)
    base_layers = base["backbone"] + base["head"]
    candidate_layers = candidate["backbone"] + candidate["head"]
    assert candidate_layers[:31] == base_layers[:31]
    assert [row[2] for row in candidate_layers[31:41]] == [
        "DentRustRelationGate",
        "Index",
        "Index",
        "DentRustRelationGate",
        "Index",
        "Index",
        "DentRustRelationGate",
        "Index",
        "Index",
        "Segment26DentRust",
    ]
    model = SegmentationModel(DR_CFG, ch=3, nc=8, verbose=False)
    assert sum(isinstance(module, DentRustRelationGate) for module in model.modules()) == 3
    assert isinstance(model.model[-1], Segment26DentRust)
    assert model.model[-1].f == [35, 32, 38, 29, 30, 36, 33, 39]
    assert model.stride.tolist() == [8.0, 4.0, 16.0, 32.0, 64.0]
```

- [ ] **Step 2: Run the graph test and verify the missing YAML fails**

Run: `pytest tests/test_yolo11_10csar_dr_0911.py::test_dr_yaml_preserves_layers_zero_to_thirty_and_adds_only_relation_layers -v`

Expected: FAIL with `FileNotFoundError` for `yolo11-10csar-dr-0911.yaml`.

- [ ] **Step 3: Create the candidate YAML from the baseline**

Copy the baseline file, retain `nc`, `scales`, the complete `backbone`, and head rows for global layers `15–30`. Remove only the baseline global layer `31` Segment26 row. Add this configuration block above `backbone`:

```yaml
dent_rust:
    dent_class: 2
    rust_class: 5
    relation_loss_gain: 0.25
    cooccurrence_weight: 2.0
    gamma_pos: 1.0
    gamma_neg: 4.0
    clip: 0.05
```

Append exactly these rows to `head`:

```yaml
# P2 relation gate: 5x5 at stride 4 preserves fine dent boundaries.
- [26, 1, DentRustRelationGate, [128, 64, 5, 0.10]] # 31
- [31, 1, Index, [-1, 0]] # 32-P2 enhanced
- [31, 1, Index, [3, 1]] # 33-P2 D/R/pair logits

# P3 relation gate: primary Segment26 feature and local context.
- [27, 1, DentRustRelationGate, [256, 96, 5, 0.10]] # 34
- [34, 1, Index, [-1, 0]] # 35-P3 enhanced
- [34, 1, Index, [3, 1]] # 36-P3 D/R/pair logits

# P4 relation gate: use 3x3 to avoid excessive coarse-scale dilation.
- [28, 1, DentRustRelationGate, [512, 128, 3, 0.10]] # 37
- [37, 1, Index, [-1, 0]] # 38-P4 enhanced
- [37, 1, Index, [3, 1]] # 39-P4 D/R/pair logits

# Five Segment features first; relation maps follow in P3, P2, P4 order.
# Radii [2, 4, 1] correspond to approximately 16 input pixels at strides [8, 4, 16].
- [[35, 32, 38, 29, 30, 36, 33, 39], 1, Segment26DentRust, [nc, 32, 256, 2, 5, 5, 3, 0.25, [2, 4, 1]]] # 40
```

- [ ] **Step 4: Add a forward-shape regression test**

```python
def test_dr_model_forward_has_stride_four_prototypes_and_three_relation_scales():
    model = SegmentationModel(DR_CFG, ch=3, nc=8, verbose=False).train()
    image = torch.randn(1, 3, 128, 128)
    predictions = model(image)
    assert [tuple(item.shape) for item in predictions["dent_rust_logits"]] == [
        (1, 3, 16, 16),
        (1, 3, 32, 32),
        (1, 3, 8, 8),
    ]
    prototypes = predictions["proto"][0]
    assert prototypes.shape[-2:] == (32, 32)
```

- [ ] **Step 5: Run graph and forward tests**

Run: `pytest tests/test_yolo11_10csar_dr_0911.py -k "dr_yaml or dr_model_forward" -v`

Expected: PASS.

- [ ] **Step 6: Commit the architecture variant**

```bash
git add ultralytics/cfg/models/11_myself/yolo11-10csar-dr-0911.yaml tests/test_yolo11_10csar_dr_0911.py
git commit -m "feat: add ten-csar dent-rust relation architecture"
```

### Task 5: Train the Relation Maps and Warm-Start the New Head

**Files:**

- Create: `tools/train_yolo11_10csar_dr_0911.py`
- Modify: `tests/test_yolo11_10csar_dr_0911.py`

**Interfaces:**

- Consumes: standard segmentation batch, candidate YAML, optional baseline checkpoint.
- Produces: `DentRustRelationLoss`, `DentRustSegmentationModel`, `DentRustTrainer`, `prepare_common_initialization`, `remap_base_state_dict`, `parse_args`, and `build_overrides`.

- [ ] **Step 1: Write failing loss-target and backward tests**

```python
from __future__ import annotations

from tools.train_yolo11_10csar_dr_0911 import DentRustSegmentationModel

from ultralytics.cfg import get_cfg


def _dent_rust_batch() -> dict[str, torch.Tensor]:
    masks = torch.zeros(3, 32, 32)
    masks[0, 5:20, 4:18] = 1
    masks[1, 10:25, 10:24] = 1
    masks[2, 23:30, 2:9] = 1
    return {
        "img": torch.randn(1, 3, 128, 128),
        "batch_idx": torch.tensor([0.0, 0.0, 0.0]),
        "cls": torch.tensor([[2.0], [5.0], [2.0]]),
        "bboxes": torch.tensor(
            [
                [0.34375, 0.390625, 0.4375, 0.46875],
                [0.53125, 0.546875, 0.4375, 0.46875],
                [0.171875, 0.828125, 0.21875, 0.21875],
            ]
        ),
        "masks": masks,
        "sem_masks": torch.zeros(1, 32, 32),
        "heatmaps": torch.zeros(1, 8, 32, 32),
        "seedmaps": torch.zeros(1, 8, 32, 32),
    }


def test_relation_loss_preserves_standalone_d_and_backpropagates_to_all_gates():
    model = DentRustSegmentationModel(DR_CFG, ch=3, nc=8, verbose=False)
    model.args = get_cfg(overrides={"overlap_mask": False})
    model.train()
    batch = _dent_rust_batch()
    predictions = model(batch["img"])
    criterion = model.init_criterion()
    target = criterion.build_target(batch, batch_size=1, size=(32, 32), radius_cells=4, dtype=torch.float32)
    assert target[0, 0, 26, 5] == 1
    assert target[0, 2, 26, 5] == 0
    loss, items = criterion(predictions, batch)
    assert loss.shape == items.shape == (7,)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    gates = [model.model[index] for index in (31, 34, 37)]
    assert all(gate.evidence.weight.grad is not None for gate in gates)
    assert all(gate.relation.weight.grad is not None for gate in gates)
    assert all(gate.modulator.weight.grad is not None for gate in gates)
```

- [ ] **Step 2: Run the loss test and verify the missing trainer fails**

Run: `pytest tests/test_yolo11_10csar_dr_0911.py::test_relation_loss_preserves_standalone_d_and_backpropagates_to_all_gates -v`

Expected: FAIL because `tools.train_yolo11_10csar_dr_0911` does not exist.

- [ ] **Step 3: Implement the architecture-specific loss**

In `tools/train_yolo11_10csar_dr_0911.py`, reuse `AsymmetricLoss` from the tested 0906 trainer and implement:

```python
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from ultralytics.utils.dent_rust import build_dent_rust_targets, resolve_dent_rust_ids

from tools.train_yolo11_6csar_0906iim import AsymmetricLoss
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel, load_checkpoint
from ultralytics.utils import RANK
from ultralytics.utils.loss import MultiChannelDiceLoss, v8SegmentationLoss
from ultralytics.utils.torch_utils import unwrap_model

ROOT = Path(__file__).resolve().parents[1]


class DentRustRelationLoss(v8SegmentationLoss):
    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk, tal_topk2)
        if self.overlap:
            raise ValueError("Dent/rust relation training requires overlap_mask=False.")
        head = model.model[-1]
        config = model.yaml.get("dent_rust", {})
        self.dent_class = head.dent_class
        self.rust_class = head.rust_class
        self.relation_radii = head.relation_radii
        self.relation_loss_gain = float(config.get("relation_loss_gain", head.relation_loss_gain))
        self.cooccurrence_weight = float(config.get("cooccurrence_weight", 2.0))
        self.asl = AsymmetricLoss(
            gamma_pos=float(config.get("gamma_pos", 1.0)),
            gamma_neg=float(config.get("gamma_neg", 4.0)),
            clip=float(config.get("clip", 0.05)),
        )
        self.dice = MultiChannelDiceLoss(smooth=1)

    def build_target(self, batch, batch_size, size, radius_cells, dtype):
        return build_dent_rust_targets(
            batch["masks"].to(self.device),
            batch["cls"].to(self.device),
            batch["batch_idx"].to(self.device),
            batch_size,
            size,
            self.dent_class,
            self.rust_class,
            radius_cells,
        ).to(dtype=dtype)

    def relation_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        elementwise = self.asl(logits, target)
        pair_pixel = target[:, 2:3]
        pixel_weight = 1.0 + self.cooccurrence_weight * pair_pixel
        weighted_asl = (elementwise * pixel_weight).sum()
        weighted_asl /= (pixel_weight.sum() * 3).clamp_min(1.0)
        return 0.5 * weighted_asl + 0.5 * self.dice(logits, target)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        scaled, detached = super().loss(preds, batch)
        logits_by_level = preds.get("dent_rust_logits")
        if logits_by_level is None or len(logits_by_level) != len(self.relation_radii):
            raise RuntimeError("Segment26DentRust did not return all configured relation maps.")
        batch_size = preds["boxes"].shape[0]
        auxiliary = torch.zeros((), device=self.device)
        for logits, radius in zip(logits_by_level, self.relation_radii):
            target = self.build_target(batch, batch_size, logits.shape[-2:], radius, logits.dtype)
            auxiliary = auxiliary + self.relation_loss(logits, target)
        auxiliary = auxiliary * (self.relation_loss_gain / len(logits_by_level))
        scaled[4] += auxiliary * batch_size
        detached[4] += auxiliary.detach()
        return scaled, detached
```

- [ ] **Step 4: Implement the specialized model and dataset-name guard**

```python
class DentRustSegmentationModel(SegmentationModel):
    def init_criterion(self):
        return DentRustRelationLoss(self)


class DentRustTrainer(SegmentationTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        dent_class, rust_class = resolve_dent_rust_ids(self.data["names"])
        model = DentRustSegmentationModel(
            cfg,
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )
        head = model.model[-1]
        if (head.dent_class, head.rust_class) != (dent_class, rust_class):
            raise ValueError("Model and dataset disagree on D/R class indices.")
        if weights:
            source, _ = load_checkpoint(weights, device="cpu")
            transfer = remap_base_state_dict(source.float().state_dict(), model.state_dict())
            model.load_state_dict(transfer, strict=False)
        return model

    def save_model(self):
        ema_model = unwrap_model(self.ema.ema)
        runtime_class = ema_model.__class__
        ema_model.__class__ = SegmentationModel
        try:
            return super().save_model()
        finally:
            ema_model.__class__ = runtime_class
```

- [ ] **Step 5: Add and test baseline-head weight remapping**

```python
from __future__ import annotations


def remap_base_state_dict(source: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    transfer = {}
    for source_key, tensor in source.items():
        target_key = (
            source_key.replace("model.31.", "model.40.", 1) if source_key.startswith("model.31.") else source_key
        )
        if target_key in target and target[target_key].shape == tensor.shape:
            transfer[target_key] = tensor
    if not any(key.startswith("model.40.") for key in transfer):
        raise ValueError("Baseline Segment26 weights were not remapped to candidate layer 40.")
    return transfer


def prepare_common_initialization(source_weights: Path | None, output: Path) -> Path:
    torch.manual_seed(0)
    baseline = SegmentationModel(BASE_CFG, ch=3, nc=8, verbose=False)
    if source_weights is not None:
        baseline.load(source_weights)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": deepcopy(baseline).half(), "train_args": {}}, output)
    return output
```

Test it without disk I/O:

```python
def test_baseline_warm_start_remaps_segment26_from_31_to_40():
    base = SegmentationModel(BASE_CFG, ch=3, nc=8, verbose=False)
    candidate = DentRustSegmentationModel(DR_CFG, ch=3, nc=8, verbose=False)
    transfer = remap_base_state_dict(base.state_dict(), candidate.state_dict())
    assert "model.40.cv2.0.0.conv.weight" in transfer
    assert torch.equal(transfer["model.40.cv2.0.0.conv.weight"], base.state_dict()["model.31.cv2.0.0.conv.weight"])


def test_identity_gates_preserve_baseline_predictions_after_weight_remap():
    base = SegmentationModel(BASE_CFG, ch=3, nc=8, verbose=False).eval()
    candidate = DentRustSegmentationModel(DR_CFG, ch=3, nc=8, verbose=False).eval()
    candidate.load_state_dict(remap_base_state_dict(base.state_dict(), candidate.state_dict()), strict=False)
    image = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        base_raw = base(image)[1]
        candidate_raw = candidate(image)[1]
    for key in ("boxes", "scores", "mask_coefficient"):
        assert torch.allclose(candidate_raw[key], base_raw[key], atol=1e-6, rtol=1e-5)
```

- [ ] **Step 6: Add the minimal reproducible CLI**

Expose `--data`, `--pretrained`, `--epochs`, `--imgsz`, `--batch`, `--device`, `--workers`, `--project`, and `--name`. `build_overrides()` must return:

```python
from __future__ import annotations

MODEL_CFG = ROOT / "ultralytics/cfg/models/11_myself/yolo11-10csar-dr-0911.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 0911 ten-CSAR dent/rust relation model.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default="yolo11-10csar-dr-0911")
    return parser.parse_args(argv)


def build_overrides(args: argparse.Namespace) -> dict:
    if args.imgsz != 640 or args.batch != 8:
        raise ValueError("The first controlled comparison requires imgsz=640 and batch=8.")
    overrides = {
        "model": str(MODEL_CFG),
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": 640,
        "batch": 8,
        "device": args.device,
        "workers": args.workers,
        "overlap_mask": False,
        "mask_ratio": 4,
        "cls_pw": 0.5,
        "seed": 0,
        "deterministic": True,
        "patience": 0,
        "name": args.name,
    }
    if args.pretrained is not None:
        overrides["pretrained"] = str(args.pretrained)
    if args.project is not None:
        overrides["project"] = args.project
    return overrides


def main() -> None:
    trainer = DentRustTrainer(overrides=build_overrides(parse_args()))
    trainer.train()


if __name__ == "__main__":
    main()
```

`mask_ratio` and `cls_pw` are deliberately not exposed as CLI flags during the first controlled comparison.

- [ ] **Step 7: Run all relation tests and the existing multi-label tests**

Run: `pytest tests/test_yolo11_10csar_dr_0911.py tests/test_yolo11_6csar_0906iim.py tests/test_msat_multilabel.py -v`

Expected: PASS with finite seven-component loss and gradients in all three relation gates.

- [ ] **Step 8: Commit the trainable architecture**

```bash
git add tools/train_yolo11_10csar_dr_0911.py tests/test_yolo11_10csar_dr_0911.py
git commit -m "feat: train dent-rust relation supervision"
```

### Task 6: Add Pair-Stratified Evaluation and Run the Controlled Comparison

**Files:**

- Create: `tools/eval_dent_rust_0911.py`
- Modify: `tests/test_yolo11_10csar_dr_0911.py`
- Create at run time: `runs/segment/0911-dent-rust-comparison/metrics.json`

**Interfaces:**

- Consumes: baseline checkpoint, candidate checkpoint, validation dataset YAML, `association_radius_px=16`, and `mask_iou=0.50`.
- Produces: standard validation metrics plus D overall, D-with-R, D-without-R, R, and rust-only false-D statistics for both checkpoints.

- [ ] **Step 1: Write the failing association and metric tests**

```python
from tools.eval_dent_rust_0911 import classify_dent_instances, summarize_pair_recall


def test_dents_are_partitioned_into_with_r_and_without_r():
    dent_masks = torch.zeros(2, 32, 32, dtype=torch.bool)
    rust_masks = torch.zeros(1, 32, 32, dtype=torch.bool)
    dent_masks[0, 4:10, 4:10] = True
    dent_masks[1, 24:30, 24:30] = True
    rust_masks[0, 8:14, 8:14] = True
    associated = classify_dent_instances(dent_masks, rust_masks, radius_px=2)
    assert associated.tolist() == [True, False]


def test_pair_recall_summary_reports_both_dent_subsets():
    summary = summarize_pair_recall(
        associated=torch.tensor([True, True, False]),
        matched=torch.tensor([True, False, True]),
    )
    assert summary == {
        "dent_with_r_total": 2,
        "dent_with_r_matched": 1,
        "dent_with_r_recall": 0.5,
        "dent_without_r_total": 1,
        "dent_without_r_matched": 1,
        "dent_without_r_recall": 1.0,
    }
```

- [ ] **Step 2: Run the metric tests and verify the missing evaluator fails**

Run: `pytest tests/test_yolo11_10csar_dr_0911.py -k "partitioned or pair_recall_summary" -v`

Expected: FAIL because `tools.eval_dent_rust_0911` does not exist.

- [ ] **Step 3: Implement the association contract**

```python
from __future__ import annotations


def classify_dent_instances(dent_masks: torch.Tensor, rust_masks: torch.Tensor, radius_px: int) -> torch.Tensor:
    if dent_masks.ndim != 3 or rust_masks.ndim != 3:
        raise ValueError("Expected dent and rust masks shaped [N, H, W].")
    if not len(rust_masks):
        return torch.zeros(len(dent_masks), dtype=torch.bool, device=dent_masks.device)
    rust_union = rust_masks.any(dim=0, keepdim=True).float().unsqueeze(0)
    kernel = 2 * radius_px + 1
    rust_near = torch.nn.functional.max_pool2d(rust_union, kernel, stride=1, padding=radius_px)[0, 0].bool()
    return (dent_masks.bool() & rust_near).flatten(1).any(dim=1)


def summarize_pair_recall(associated: torch.Tensor, matched: torch.Tensor) -> dict[str, int | float]:
    with_r_total = int(associated.sum())
    without_r_total = int((~associated).sum())
    with_r_matched = int((associated & matched).sum())
    without_r_matched = int(((~associated) & matched).sum())
    return {
        "dent_with_r_total": with_r_total,
        "dent_with_r_matched": with_r_matched,
        "dent_with_r_recall": with_r_matched / with_r_total if with_r_total else 0.0,
        "dent_without_r_total": without_r_total,
        "dent_without_r_matched": without_r_matched,
        "dent_without_r_recall": without_r_matched / without_r_total if without_r_total else 0.0,
    }
```

- [ ] **Step 4: Complete the evaluator using standard model outputs**

The CLI must require `--base`, `--candidate`, `--data`, and `--output`; defaults are `--association-radius 16`, `--mask-iou 0.50`, `--conf 0.25`, and `--imgsz 640`. Use `conf=0.25` for the pair-aware operating-point metrics, while the separate `model.val()` call uses `conf=0.001` for standard AP calculation. Implement the mask I/O and one-to-one matching with these concrete helpers:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.utils.dent_rust import resolve_dent_rust_ids

from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, img2label_paths, polygon2mask
from ultralytics.utils.checks import check_det_dataset


def split_roots(value: str | list[str]) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    return [Path(item).resolve() for item in values]


def validation_images(value: str | list[str]) -> list[Path]:
    images = []
    for root in split_roots(value):
        if root.is_dir():
            images.extend(path for path in root.rglob("*") if path.suffix.lower().lstrip(".") in IMG_FORMATS)
        elif root.is_file():
            base = root.parent
            images.extend((base / line.strip()).resolve() for line in root.read_text().splitlines() if line.strip())
        else:
            raise FileNotFoundError(root)
    return sorted(set(images))


def ground_truth_masks(image_path: Path, label_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    masks, classes = [], []
    for line in label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []:
        values = [float(value) for value in line.split()]
        if len(values) < 7 or (len(values) - 1) % 2:
            raise ValueError(f"Expected a segmentation polygon in {label_path}: {line}")
        points = np.asarray(values[1:], dtype=np.float32).reshape(-1, 2)
        points *= np.asarray([width, height], dtype=np.float32)
        masks.append(torch.from_numpy(polygon2mask((height, width), [points.reshape(-1)], color=1)).bool())
        classes.append(int(values[0]))
    empty = torch.zeros((0, height, width), dtype=torch.bool)
    return (torch.stack(masks) if masks else empty), torch.tensor(classes, dtype=torch.long)


def prediction_masks(result, class_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = result.orig_shape
    if result.masks is None or result.boxes is None:
        return torch.zeros((0, height, width), dtype=torch.bool), torch.zeros(0)
    masks, scores = [], []
    for predicted_class, confidence, polygon in zip(result.boxes.cls, result.boxes.conf, result.masks.xy):
        if int(predicted_class) != class_id:
            continue
        mask = polygon2mask((height, width), [np.asarray(polygon, dtype=np.float32).reshape(-1)], color=1)
        masks.append(torch.from_numpy(mask).bool())
        scores.append(float(confidence))
    empty = torch.zeros((0, height, width), dtype=torch.bool)
    return (torch.stack(masks) if masks else empty), torch.tensor(scores)


def greedy_mask_matches(
    predicted_masks: torch.Tensor,
    scores: torch.Tensor,
    target_masks: torch.Tensor,
    iou_threshold: float,
) -> tuple[torch.Tensor, int]:
    matched_targets = torch.zeros(len(target_masks), dtype=torch.bool)
    used_targets: set[int] = set()
    false_positives = 0
    for prediction_index in scores.argsort(descending=True).tolist():
        prediction = predicted_masks[prediction_index]
        best_target, best_iou = -1, 0.0
        for target_index, target in enumerate(target_masks):
            if target_index in used_targets:
                continue
            intersection = (prediction & target).sum().item()
            union = (prediction | target).sum().item()
            iou = intersection / union if union else 0.0
            if iou > best_iou:
                best_target, best_iou = target_index, iou
        if best_target >= 0 and best_iou >= iou_threshold:
            used_targets.add(best_target)
            matched_targets[best_target] = True
        else:
            false_positives += 1
    return matched_targets, false_positives
```

For each validation image:

1. Rasterize ground-truth YOLO polygons at the original image size with `ultralytics.data.utils.polygon2mask`.
2. Split GT masks into D and R by the validated class IDs.
3. Classify each GT D mask with `classify_dent_instances`.
4. Match class-D predictions to GT D masks greedily by descending confidence and mask IoU `>=0.50`, using every prediction and GT at most once.
5. Count D false positives on images containing R but no D.
6. Call `model.val(data=args.data, split="val", imgsz=640, conf=0.001, plots=False)` for standard per-class metrics. Resolve D and R row positions through `metrics.seg.ap_class_index` before calling `metrics.seg.class_result(row)`; each result returns precision, recall, AP50, and AP50-95.
7. Write one JSON object with keys `dataset`, `base`, `candidate`, and `delta`; each model section must contain `map50_95`, `dent_ap50`, `dent_recall`, `dent_with_r_recall`, `dent_without_r_recall`, `rust_ap50`, `rust_recall`, `false_d_on_r_only`, `parameters`, and `latency_ms`.

Use this aggregation function so the operating-point and standard AP metrics cannot be mixed accidentally:

```python
from __future__ import annotations


def evaluate_checkpoint(
    checkpoint: Path,
    data_yaml: Path,
    images: list[Path],
    labels: list[Path],
    dent_class: int,
    rust_class: int,
    association_radius: int,
    mask_iou: float,
    confidence: float,
    imgsz: int,
) -> dict[str, int | float]:
    model = YOLO(checkpoint)
    associated_rows, matched_rows = [], []
    false_d_on_r_only = 0
    for image_path, label_path in zip(images, labels):
        target_masks, target_classes = ground_truth_masks(image_path, label_path)
        dent_targets = target_masks[target_classes == dent_class]
        rust_targets = target_masks[target_classes == rust_class]
        associated = classify_dent_instances(dent_targets, rust_targets, association_radius)
        result = model.predict(image_path, imgsz=imgsz, conf=confidence, verbose=False)[0]
        dent_predictions, dent_scores = prediction_masks(result, dent_class)
        matched, false_positives = greedy_mask_matches(dent_predictions, dent_scores, dent_targets, mask_iou)
        associated_rows.append(associated.cpu())
        matched_rows.append(matched.cpu())
        if len(rust_targets) and not len(dent_targets):
            false_d_on_r_only += false_positives

    associated = torch.cat(associated_rows) if associated_rows else torch.zeros(0, dtype=torch.bool)
    matched = torch.cat(matched_rows) if matched_rows else torch.zeros(0, dtype=torch.bool)
    pair_metrics = summarize_pair_recall(associated, matched)
    standard = model.val(data=str(data_yaml), split="val", imgsz=imgsz, conf=0.001, plots=False)
    dent_row = list(standard.seg.ap_class_index).index(dent_class)
    rust_row = list(standard.seg.ap_class_index).index(rust_class)
    _, dent_recall, dent_ap50, _ = standard.seg.class_result(dent_row)
    _, rust_recall, rust_ap50, _ = standard.seg.class_result(rust_row)
    return {
        **pair_metrics,
        "map50_95": float(standard.seg.map),
        "dent_ap50": float(dent_ap50),
        "dent_recall": float(dent_recall),
        "rust_ap50": float(rust_ap50),
        "rust_recall": float(rust_recall),
        "false_d_on_r_only": int(false_d_on_r_only),
        "parameters": int(sum(parameter.numel() for parameter in model.model.parameters())),
        "latency_ms": float(standard.speed["inference"]),
    }


def numeric_delta(candidate: dict, baseline: dict) -> dict[str, float]:
    return {
        key: float(candidate[key] - baseline[key])
        for key in candidate.keys() & baseline.keys()
        if isinstance(candidate[key], (int, float)) and isinstance(baseline[key], (int, float))
    }
```

- [ ] **Step 5: Reject invalid evaluation splits before inference**

Resolve `train` and `val` entries relative to the dataset YAML `path`. If their normalized absolute image paths are equal, raise:

```text
Pair-aware evaluation requires a held-out val split; train and val resolve to the same images.
```

Add a unit test using a temporary YAML whose train and val both equal `images/train`.

Use this exact guard after `check_det_dataset(args.data)` resolves the paths:

```python
data = check_det_dataset(args.data)
resolve_dent_rust_ids(data["names"])
if split_roots(data["train"]) == split_roots(data["val"]):
    raise ValueError("Pair-aware evaluation requires a held-out val split; train and val resolve to the same images.")
images = validation_images(data["val"])
labels = [Path(path) for path in img2label_paths([str(image) for image in images])]
```

Complete `main()` with an explicit shared dataset pass and JSON write:

```python
from __future__ import annotations


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare D recall with and without nearby R.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--association-radius", type=int, default=16)
    parser.add_argument("--mask-iou", type=float, default=0.50)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    data = check_det_dataset(str(args.data))
    dent_class, rust_class = resolve_dent_rust_ids(data["names"])
    if split_roots(data["train"]) == split_roots(data["val"]):
        raise ValueError(
            "Pair-aware evaluation requires a held-out val split; train and val resolve to the same images."
        )
    images = validation_images(data["val"])
    labels = [Path(path) for path in img2label_paths([str(image) for image in images])]
    common = (
        args.data,
        images,
        labels,
        dent_class,
        rust_class,
        args.association_radius,
        args.mask_iou,
        args.conf,
        args.imgsz,
    )
    baseline = evaluate_checkpoint(args.base, *common)
    candidate = evaluate_checkpoint(args.candidate, *common)
    payload = {
        "dataset": {"yaml": str(args.data.resolve()), "validation_images": len(images)},
        "base": baseline,
        "candidate": candidate,
        "delta": numeric_delta(candidate, baseline),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the complete unit suite**

Run: `pytest tests/test_yolo11_10csar_dr_0911.py tests/test_yolo11_6csar_iim.py tests/test_yolo11_6csar_0906iim.py tests/test_yolo11_6csar_0910iim.py tests/test_msat.py tests/test_msat_multilabel.py -v`

Expected: PASS.

- [ ] **Step 7: Train baseline and candidate under identical settings**

First create one common 10-CSAR initialization. This imports every compatible tensor from the older checkpoint once, stores the baseline Segment26 at layer 31, and becomes the only initialization source for both runs:

```bash
python -c "from pathlib import Path; from tools.train_yolo11_10csar_dr_0911 import prepare_common_initialization; prepare_common_initialization(Path('/home/d11405003/weights/best0905-6csar-p2.pt'), Path('runs/segment/0911-dent-rust-comparison/init/common-10csar.pt'))"
```

Baseline command:

```bash
yolo segment train model=ultralytics/cfg/models/11_myself/yolo11-10csar-0911.yaml data=/home/d11405003/dataset/data.yaml pretrained=runs/segment/0911-dent-rust-comparison/init/common-10csar.pt epochs=300 imgsz=640 batch=8 device=0 workers=8 overlap_mask=False mask_ratio=4 cls_pw=0.5 seed=0 deterministic=True patience=0 project=runs/segment/0911-dent-rust-comparison name=baseline
```

Candidate command:

```bash
python tools/train_yolo11_10csar_dr_0911.py --data /home/d11405003/dataset/data.yaml --pretrained runs/segment/0911-dent-rust-comparison/init/common-10csar.pt --epochs 300 --device 0 --workers 8 --project runs/segment/0911-dent-rust-comparison --name relation
```

- [ ] **Step 8: Generate the comparison artifact**

```bash
python tools/eval_dent_rust_0911.py --base runs/segment/0911-dent-rust-comparison/baseline/weights/best.pt --candidate runs/segment/0911-dent-rust-comparison/relation/weights/best.pt --data /home/d11405003/dataset/data.yaml --association-radius 16 --mask-iou 0.50 --conf 0.25 --imgsz 640 --output runs/segment/0911-dent-rust-comparison/metrics.json
```

Expected: the JSON contains non-zero `dent_with_r_total` and `dent_without_r_total`, and the validation paths differ from the training paths.

- [ ] **Step 9: Apply architecture-first acceptance gates**

Accept the relation architecture only when all conditions hold:

- `candidate.dent_with_r_recall - base.dent_with_r_recall >= 0.05`.
- `candidate.dent_recall >= base.dent_recall`.
- `candidate.dent_without_r_recall >= base.dent_without_r_recall - 0.02`.
- `candidate.rust_ap50 >= base.rust_ap50 - 0.02`.
- `candidate.map50_95 >= base.map50_95 - 0.01`.
- `candidate.false_d_on_r_only <= base.false_d_on_r_only * 1.10 + 1`.
- Candidate parameter count increases by at most 10%, and median latency increases by at most 15% on the same device and batch size.

If the architecture passes, run only this secondary parameter ablation: `relation_loss_gain ∈ {0.15, 0.25, 0.40}` with every other value fixed. Select by D-with-R recall subject to the same D-without-R and all-class gates.

- [ ] **Step 10: Commit evaluator and record the winning configuration**

```bash
git add tools/eval_dent_rust_0911.py tests/test_yolo11_10csar_dr_0911.py
git commit -m "test: evaluate dent recall by rust co-occurrence"
```

After a winning run, copy the exact command, dataset YAML SHA-256, git commit, checkpoint path, `metrics.json`, parameters, and latency into `runs/segment/0911-dent-rust-comparison/README.md`. Do not overwrite either baseline or candidate checkpoint.

## Self-Review Results

- **Spec coverage:** The plan freezes the user-selected 10-CSAR/IIM architecture, adds architecture-level D/R feature feedback, keeps D-alone supervision, makes parameters secondary, and defines pair-specific evaluation.
- **Placeholder scan:** Every code-producing task includes concrete interfaces, test bodies, implementation bodies, commands, and expected outcomes; no deferred implementation marker remains.
- **Type consistency:** Relation outputs are always `tuple[Tensor, Tensor]`; `Index` exposes three-channel logits; `Segment26DentRust` receives five features plus three logits; the loss uses the head's three radii in P3/P2/P4 order.
