# YOLO11 6-CSAR D-Class Shadow Recall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve class `D` recall, especially in shadowed container regions, without changing the IIM backbone, either 6-CSAR stage, or the Segment26 network topology.

**Architecture:** Keep the `backbone` and `head` layer lists identical to `yolo11-6csar-0906iim.yaml`. Make only training-time changes: explicitly control where ASL replaces BCE, reduce auxiliary-loss interference, use existing inverse-frequency class weighting, add optional local-shadow augmentation, support warm-start weights, and apply per-class inference thresholds. Treat raw MMS-map inference as a separate follow-up project because it changes the meaning of the delivered output from instance masks to semantic multi-label maps.

**Tech Stack:** Python 3.10, PyTorch, Ultralytics segmentation trainer, OpenCV, pytest, YAML.

**Spec:** `ultralytics/cfg/models/11_myself/yolo11-6csar-0906iim.yaml`

## Global Constraints

- Do not modify `ultralytics/cfg/models/11_myself/yolo11-6csar-0906iim.yaml`.
- Do not modify `tools/train_yolo11_6csar_0906iim.py` or `tools/predict_seg_by_class_0906iim.py`.
- Keep every entry in the new YAML `backbone` and `head` lists identical to the 0906 YAML, including layer order, `from`, repeats, modules, and arguments.
- Keep `overlap_mask=False`; MMS targets require one mask tensor per instance.
- Apply ASL only to class decisions: the MMS multi-label pixel classification and, when enabled, the main detection classification term. Do not replace the instance mask reconstruction BCE.
- Use class index `2` for `D`, matching checkpoint names `{0:B, 1:C, 2:D, 3:H, 4:O, 5:R, 6:X, 7:U}`.
- Use the absolute dataset file `/home/d11405003/dataset/data.yaml` for every controlled run.
- Keep `seed=0`, `deterministic=True`, and `patience=0` across ablations so one variable changes at a time and training is not interrupted.
- Save new work under 0909-specific filenames; never overwrite existing checkpoints or output directories.

---

## File Structure

- Create `ultralytics/cfg/models/11_myself/yolo11-6csar-0909iim-tuned.yaml`: immutable copy of the 0906 graph with conservative MMS/ASL defaults.
- Create `tools/train_yolo11_6csar_0909iim.py`: standalone trainer for main-class ASL, warm-start loading, class weighting, and reproducible command-line overrides.
- Create `tools/shadow_augmentation_0909iim.py`: tensor-only local-shadow augmentation with no dependency on model code.
- Create `tools/predict_seg_by_class_0909iim.py`: portable checkpoint loader plus class-specific confidence filtering.
- Create `tests/test_yolo11_6csar_0909iim.py`: architecture identity, loss-scope, augmentation, trainer-override, checkpoint portability, and threshold tests.
- Do not change shared files under `ultralytics/nn/`, `ultralytics/utils/`, or `ultralytics/engine/`.

### Task 1: Freeze the Network Graph in a New YAML

**Files:**

- Create: `ultralytics/cfg/models/11_myself/yolo11-6csar-0909iim-tuned.yaml`
- Test: `tests/test_yolo11_6csar_0909iim.py`

**Interfaces:**

- Consumes: `yaml_model_load(path) -> dict` and the complete `backbone`/`head` lists from the 0906 YAML.
- Produces: `MODEL_CFG_0909: Path` pointing to a graph-identical model configuration.

- [ ] **Step 1: Write the failing architecture-identity test**

```python
from pathlib import Path

from ultralytics.nn.tasks import yaml_model_load

ROOT = Path(__file__).resolve().parents[1]
CFG_0906 = ROOT / "ultralytics/cfg/models/11_myself/yolo11-6csar-0906iim.yaml"
CFG_0909 = ROOT / "ultralytics/cfg/models/11_myself/yolo11-6csar-0909iim-tuned.yaml"


def test_0909_keeps_the_0906_network_graph_exactly():
    old = yaml_model_load(CFG_0906)
    new = yaml_model_load(CFG_0909)
    assert new["backbone"] == old["backbone"]
    assert new["head"] == old["head"]
    assert new["scales"] == old["scales"]
```

- [ ] **Step 2: Run the test and verify that the missing YAML fails**

Run: `pytest tests/test_yolo11_6csar_0909iim.py::test_0909_keeps_the_0906_network_graph_exactly -v`

Expected: FAIL with `FileNotFoundError` for `yolo11-6csar-0909iim-tuned.yaml`.

- [ ] **Step 3: Create the graph-identical YAML**

Copy `nc`, `scales`, `backbone`, and `head` exactly from the 0906 YAML. Change only the comment header and `mms` block to:

```yaml
mms:
    loss: asl
    gamma_pos: 1.0
    gamma_neg: 4.0
    clip: 0.05
    pixel_gain: 0.15
    cooccurrence_weight: 2.0
    main_cls_asl: false
    main_gamma_pos: 1.0
    main_gamma_neg: 4.0
    main_clip: 0.05
```

The lower `pixel_gain` prevents the auxiliary branch from dominating the standard instance-segmentation objective. `main_cls_asl` starts disabled so the first 0909 run remains an interpretable auxiliary-loss baseline.

- [ ] **Step 4: Run the architecture test**

Run: `pytest tests/test_yolo11_6csar_0909iim.py::test_0909_keeps_the_0906_network_graph_exactly -v`

Expected: PASS.

- [ ] **Step 5: Commit the immutable configuration**

```bash
git add ultralytics/cfg/models/11_myself/yolo11-6csar-0909iim-tuned.yaml tests/test_yolo11_6csar_0909iim.py
git commit -m "test: freeze 0909 model graph"
```

### Task 2: Make ASL Scope Explicit Without Changing Mask Reconstruction

**Files:**

- Create: `tools/train_yolo11_6csar_0909iim.py`
- Modify: `tests/test_yolo11_6csar_0909iim.py`

**Interfaces:**

- Consumes: `AsymmetricLoss`, `MMSASLSegmentationLoss`, `MMSASLSegmentationModel`, and `MMSASLTrainer` from `tools.train_yolo11_6csar_0906iim`.
- Produces: `TunedMMSASLLoss`, `TunedMMSASLModel`, and `TunedMMSASLTrainer`.

- [ ] **Step 1: Write a failing test for main-class ASL opt-in**

```python
from tools.train_yolo11_6csar_0909iim import TunedMMSASLModel

from tools.train_yolo11_6csar_0906iim import AsymmetricLoss


def test_main_classification_asl_is_opt_in_and_mask_loss_is_unchanged():
    model = TunedMMSASLModel(CFG_0909, ch=3, nc=3, verbose=False)
    model.args = get_cfg()
    model.args.overlap_mask = False
    model.yaml["mms"]["main_cls_asl"] = True
    criterion = model.init_criterion()
    assert isinstance(criterion.bce, AsymmetricLoss)
    assert criterion.main_cls_asl_enabled is True
    assert not hasattr(criterion, "mask_asl")
```

- [ ] **Step 2: Run the test and verify the missing classes fail**

Run: `pytest tests/test_yolo11_6csar_0909iim.py::test_main_classification_asl_is_opt_in_and_mask_loss_is_unchanged -v`

Expected: FAIL with `ModuleNotFoundError` or missing `TunedMMSASLModel`.

- [ ] **Step 3: Implement the tuned criterion and model**

Use the existing tested ASL implementation; do not duplicate its formula.

```python
class TunedMMSASLLoss(MMSASLSegmentationLoss):
    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        config = model.yaml.get("mms", {})
        self.main_cls_asl_enabled = bool(config.get("main_cls_asl", False))
        if self.main_cls_asl_enabled:
            self.bce = AsymmetricLoss(
                gamma_pos=float(config.get("main_gamma_pos", 1.0)),
                gamma_neg=float(config.get("main_gamma_neg", 4.0)),
                clip=float(config.get("main_clip", 0.05)),
            )


class TunedMMSASLModel(MMSASLSegmentationModel):
    def init_criterion(self):
        return TunedMMSASLLoss(self)
```

Assigning `criterion.bce` changes the inherited main anchor-classification loss only. The inherited instance mask path calls `F.binary_cross_entropy_with_logits` directly and therefore remains unchanged.

- [ ] **Step 4: Add a full forward/backward test with soft assignment targets**

Reuse `_overlapping_batch()` from the 0906 tests. Set `main_cls_asl=True`, run `model(batch["img"])`, calculate all seven loss items, call `loss.sum().backward()`, and assert every loss and every MMS projection gradient is finite.

- [ ] **Step 5: Run the focused loss tests**

Run: `pytest tests/test_yolo11_6csar_0909iim.py -k "main_classification or forward_backward" -v`

Expected: PASS.

- [ ] **Step 6: Commit the isolated loss change**

```bash
git add tools/train_yolo11_6csar_0909iim.py tests/test_yolo11_6csar_0909iim.py
git commit -m "feat: make main classification ASL optional"
```

### Task 3: Add Reproducible Training and Warm-Start Controls

**Files:**

- Modify: `tools/train_yolo11_6csar_0909iim.py`
- Modify: `tests/test_yolo11_6csar_0909iim.py`

**Interfaces:**

- Consumes: command-line values and `MMSASLTrainer.get_model(cfg, weights, verbose)`.
- Produces: `parse_args(argv=None) -> argparse.Namespace` and `build_overrides(args) -> dict`.

- [ ] **Step 1: Write the failing override test**

```python
def test_training_overrides_are_reproducible_and_keep_independent_masks():
    args = parse_args(
        [
            "--data",
            "/home/d11405003/dataset/data.yaml",
            "--pretrained-weights",
            "/home/d11405003/weights/best0905-6csar-p2.pt",
            "--batch",
            "8",
            "--cls-pw",
            "0.5",
        ]
    )
    overrides = build_overrides(args)
    assert overrides["overlap_mask"] is False
    assert overrides["patience"] == 0
    assert overrides["seed"] == 0
    assert overrides["deterministic"] is True
    assert overrides["pretrained"].endswith("best0905-6csar-p2.pt")
    assert overrides["cls_pw"] == 0.5
```

- [ ] **Step 2: Run the test and verify the parser/override failure**

Run: `pytest tests/test_yolo11_6csar_0909iim.py::test_training_overrides_are_reproducible_and_keep_independent_masks -v`

Expected: FAIL because `parse_args(argv)` and `build_overrides` do not yet exist.

- [ ] **Step 3: Add explicit training arguments**

Implement these defaults:

```text
--epochs 600
--imgsz 640
--batch 8
--patience 0
--optimizer AdamW
--lr0 0.001
--lrf 0.01
--cls-pw 0.0
--hsv-v 0.4
--multi-scale 0.0
--mask-ratio 4
--close-mosaic 10
--seed 0
--workers 8
--pretrained-weights optional path
--main-cls-asl store_true
```

Map `--pretrained-weights` to the Ultralytics `pretrained` override as a path string. Do not map it to `model`; `model` must remain the 0909 YAML so the graph does not revert to the checkpoint's old YAML.

- [ ] **Step 4: Apply runtime loss flags before criterion construction**

Give `TunedMMSASLTrainer` a `loss_options: dict`. In `get_model()`, construct `TunedMMSASLModel`, update `model.yaml["mms"]` with `loss_options`, and only then load compatible weights. This makes each checkpoint record the exact loss settings used by its run.

- [ ] **Step 5: Retain portable checkpoint serialization**

Inherit the tested `MMSASLTrainer.save_model()` behavior. Add a test that intercepts the parent save call and verifies the EMA object is serialized as `SegmentationModel` and restored to `TunedMMSASLModel` afterward.

- [ ] **Step 6: Run parser, model-load, and serialization tests**

Run: `pytest tests/test_yolo11_6csar_0909iim.py -k "overrides or pretrained or portable" -v`

Expected: PASS. The pretrained test must report at least one intersecting tensor loaded and must not change the 0909 layer count.

- [ ] **Step 7: Commit the controlled trainer**

```bash
git add tools/train_yolo11_6csar_0909iim.py tests/test_yolo11_6csar_0909iim.py
git commit -m "feat: add controlled 0909 segmentation trainer"
```

### Task 4: Add Optional Local-Shadow Augmentation

**Files:**

- Create: `tools/shadow_augmentation_0909iim.py`
- Modify: `tools/train_yolo11_6csar_0909iim.py`
- Modify: `tests/test_yolo11_6csar_0909iim.py`

**Interfaces:**

- Produces: `apply_random_shadow(images, probability, strength_min, strength_max, generator=None) -> torch.Tensor`.
- Consumes: normalized image tensors shaped `[B, C, H, W]` in `[0, 1]` after the parent trainer's `preprocess_batch()`.

- [ ] **Step 1: Write failing unit tests for the augmentation contract**

```python
def test_shadow_augmentation_is_deterministic_and_preserves_tensor_contract():
    images = torch.ones(2, 3, 64, 64)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    first = apply_random_shadow(images, 1.0, 0.35, 0.65, g1)
    second = apply_random_shadow(images, 1.0, 0.35, 0.65, g2)
    assert first.shape == images.shape
    assert first.dtype == images.dtype
    assert torch.equal(first, second)
    assert 0.0 <= first.min() <= first.max() <= 1.0
    assert (first < images).any()
    assert (first == images).any()
```

- [ ] **Step 2: Run the test and verify the missing function fails**

Run: `pytest tests/test_yolo11_6csar_0909iim.py::test_shadow_augmentation_is_deterministic_and_preserves_tensor_contract -v`

Expected: FAIL with an import error.

- [ ] **Step 3: Implement tensor-only local shadows**

For each selected image, generate two x-coordinates at the top and bottom edges, randomly select the left or right half-plane, multiply that spatial region by a factor sampled from `[strength_min, strength_max]`, then clamp to `[0, 1]`. Use only PyTorch operations so the transform works on CPU and CUDA. Reject invalid probabilities or strength ranges with `ValueError`.

- [ ] **Step 4: Integrate it after parent preprocessing**

Override `TunedMMSASLTrainer.preprocess_batch()`:

```python
def preprocess_batch(self, batch):
    batch = super().preprocess_batch(batch)
    if self.shadow_probability > 0:
        batch["img"] = apply_random_shadow(
            batch["img"],
            self.shadow_probability,
            self.shadow_strength_min,
            self.shadow_strength_max,
        )
    return batch
```

Add CLI defaults `--shadow-probability 0.0`, `--shadow-strength-min 0.35`, and `--shadow-strength-max 0.65`. The first ablation keeps this disabled; the shadow experiment uses probability `0.30`.

- [ ] **Step 5: Verify labels and masks are untouched**

Construct a batch containing `img`, `masks`, `cls`, and `bboxes`. Call the overridden preprocessing and assert only `img` changes; `masks`, `cls`, and `bboxes` remain tensor-equal after accounting for the normal device transfer.

- [ ] **Step 6: Run all augmentation tests**

Run: `pytest tests/test_yolo11_6csar_0909iim.py -k shadow -v`

Expected: PASS on CPU and, when available, CUDA.

- [ ] **Step 7: Commit the opt-in augmentation**

```bash
git add tools/shadow_augmentation_0909iim.py tools/train_yolo11_6csar_0909iim.py tests/test_yolo11_6csar_0909iim.py
git commit -m "feat: add opt-in local shadow augmentation"
```

### Task 5: Add Per-Class Confidence Thresholds for D and R

**Files:**

- Create: `tools/predict_seg_by_class_0909iim.py`
- Modify: `tests/test_yolo11_6csar_0909iim.py`

**Interfaces:**

- Consumes: normal Ultralytics `Results`, model class names, and the legacy checkpoint alias behavior from the 0906 predictor.
- Produces: `parse_class_thresholds(values, names) -> dict[int, float]` and `keep_prediction(class_id, score, global_conf, thresholds) -> bool`.

- [ ] **Step 1: Write failing threshold parser tests**

```python
def test_class_thresholds_accept_names_and_ids():
    names = {0: "B", 1: "C", 2: "D", 3: "H", 4: "O", 5: "R", 6: "X", 7: "U"}
    actual = parse_class_thresholds(["D=0.15", "5=0.35"], names)
    assert actual == {2: 0.15, 5: 0.35}
    assert keep_prediction(2, 0.18, 0.25, actual)
    assert not keep_prediction(5, 0.30, 0.25, actual)
```

- [ ] **Step 2: Run the test and verify the missing predictor fails**

Run: `pytest tests/test_yolo11_6csar_0909iim.py::test_class_thresholds_accept_names_and_ids -v`

Expected: FAIL with an import error.

- [ ] **Step 3: Implement class-specific post-filtering**

Add repeatable `--class-conf NAME=VALUE`. Pass the smallest requested threshold to `model.predict(conf=...)` so low-confidence D candidates survive Ultralytics NMS, then filter each rendered prediction with its class-specific threshold. Reject unknown class names, duplicated class entries, and thresholds outside `[0, 1]`.

- [ ] **Step 4: Keep compatibility and output conventions**

Install `__main__.MMSASLSegmentationModel = SegmentationModel` before loading legacy checkpoints. Continue writing eight class-specific images and one `__all.jpg`. Include the class name in the drawn label, for example `D 0.31`, so an all-class image cannot confuse orange D with green R.

- [ ] **Step 5: Test filtering and labels without invoking a GPU**

Use synthetic masks, boxes, class IDs `[2, 5]`, and scores `[0.18, 0.30]`. With `D=0.15,R=0.35`, assert the rendered output contains D and excludes R. Test the parser error cases separately.

- [ ] **Step 6: Run predictor tests**

Run: `pytest tests/test_yolo11_6csar_0909iim.py -k "threshold or prediction_label" -v`

Expected: PASS.

- [ ] **Step 7: Commit the inference-only change**

```bash
git add tools/predict_seg_by_class_0909iim.py tests/test_yolo11_6csar_0909iim.py
git commit -m "feat: add class-specific segmentation thresholds"
```

### Task 6: Run One-Variable-at-a-Time Ablations

**Files:**

- Verify: `/home/d11405003/dataset/data.yaml`
- Verify outputs: `runs/segment/0909-*`
- No source files change in this task.

**Interfaces:**

- Consumes: the 0909 trainer and fixed dataset YAML.
- Produces: directly comparable `results.csv`, `best.pt`, D-class validation metrics, and shadow-image predictions.

- [ ] **Step 1: Record dataset identity before every run**

Run:

```bash
readlink -f /home/d11405003/dataset/data.yaml
sha256sum /home/d11405003/dataset/data.yaml
```

Expected: every run records the same resolved path and SHA-256 value in its log. Stop the comparison if either changes.

- [ ] **Step 2: Run the conservative 0909 baseline**

```bash
python tools/train_yolo11_6csar_0909iim.py \
  --data /home/d11405003/dataset/data.yaml \
  --epochs 600 --imgsz 640 --batch 8 --patience 0 \
  --device 0 --cls-pw 0.0 \
  --shadow-probability 0.0 \
  --name 0909-baseline
```

Expected: all 600 epochs run, checkpoint loads in a fresh Python process, and the logged graph is identical to Task 1.

- [ ] **Step 3: Test existing inverse-frequency class weighting only**

Repeat the baseline with `--cls-pw 0.5 --name 0909-clspw050`. Do not enable main ASL or shadow augmentation.

Expected: trainer prints normalized class weights; D's weight is determined from label frequency rather than hard-coded.

- [ ] **Step 4: Test main classification ASL only**

Repeat the better of Steps 2 and 3 with `--main-cls-asl --name 0909-main-asl`. Keep the same batch, image size, seed, and augmentation values.

Expected: the checkpoint YAML records `main_cls_asl: true` and the training log includes a finite main classification loss from epoch 1 onward.

- [ ] **Step 5: Test local-shadow augmentation only after selecting the better loss**

Repeat the current winner with `--shadow-probability 0.30 --shadow-strength-min 0.35 --shadow-strength-max 0.65 --name 0909-shadow030`.

Expected: no NaN/Inf losses and no modification to mask counts or class counts.

- [ ] **Step 6: Test higher resolution last**

Repeat the current winner with `--imgsz 960 --batch 8 --name 0909-img960`. Reduce batch only if CUDA memory is insufficient, and record the new batch value explicitly.

Expected: the run completes without out-of-memory errors. `retina_masks` is not a training substitute and is evaluated only during inference.

- [ ] **Step 7: Validate every best checkpoint with the same settings**

Use `conf=0.001`, `iou=0.7`, `imgsz` matching the run, the same data YAML, and save per-class metrics. Select a candidate only if:

- D mask recall improves by at least 15% relative to the 0909 baseline;
- overall mask mAP50 does not decrease by more than 0.02 absolute;
- the fixed image `CBHU0708054-A` gains at least one D prediction in the shadowed right-hand region at `D=0.15`;
- the number of R predictions on that fixed image does not increase at `R=0.35`.

- [ ] **Step 8: Run the fixed-image inference comparison**

```bash
python tools/predict_seg_by_class_0909iim.py \
  --model runs/segment/0909-shadow030/weights/best.pt \
  --source /home/d11405003/dataset/val/images/CBHU0708054-A.JPG \
  --output runs/segment_by_class/0909-shadow030 \
  --imgsz 960 --conf 0.25 \
  --class-conf D=0.15 --class-conf R=0.35 \
  --device 0 --retina-masks
```

Expected: the all-class image labels include class names, and the D-only image uses the D threshold without admitting low-confidence R predictions.

- [ ] **Step 9: Record the winning command and reject non-improving variants**

Add the complete winning command, dataset SHA-256, git commit, and four acceptance measurements to the experiment notes under the winning run directory. Do not combine two variants unless each independently passed Step 7.

---

## Deferred Separate Plan: Raw MMS Inference

Do not silently fuse MMS logits into standard YOLO instance masks in this plan. A later, separate plan should expose P3/P4/P5 `multilabel_logits`, upsample and calibrate them per class, and save semantic multi-label heatmaps beside—not in place of—the standard instance results. This separation is required because MMS heatmaps and YOLO instance masks have different semantics and validation metrics.
