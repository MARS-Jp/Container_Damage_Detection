"""Simulate the first-stage Dent/Rust feature-suppression check.

The script follows the experiment described in the linked HackMD note:

1. find Dent-only, Rust-only, and genuinely overlapping Dent+Rust masks;
2. build per-layer feature prototypes from the single-class masks;
3. compare overlap-ROI features with both prototypes;
4. retain pre-NMS Detect-head and MSAT multi-label scores in the same ROI; and
5. create class-specific Grad-CAM views for one representative overlap ROI.

This is a diagnostic simulation, not a statistical proof of causal suppression.
The output report states the sample sizes and the explicit heuristic used for
candidate flags so a larger validation run can reuse the exact same procedure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "runs" / ".matplotlib"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics import YOLO

DEFAULT_WEIGHT = Path(r"C:\Users\USER\Downloads\last0912.pt")
DEFAULT_DATASET = Path(r"C:\Users\USER\Desktop\碩論\dataset0608\dataset0608\dataset")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Instance:
    """One normalized YOLO polygon instance."""

    class_id: int
    points: np.ndarray


@dataclass(frozen=True)
class Sample:
    """One labeled image and its parsed instances."""

    image_path: Path
    label_path: Path
    instances: tuple[Instance, ...]


@dataclass(frozen=True)
class LayerSpec:
    """A model layer included in the prototype comparison."""

    name: str
    index: int
    group: str
    scale: str


LAYER_SPECS = (
    LayerSpec("attention_p2", 2, "AttentionResidual", "P2"),
    LayerSpec("attention_p3", 5, "AttentionResidual", "P3"),
    LayerSpec("attention_p4", 8, "AttentionResidual", "P4"),
    LayerSpec("attention_p5", 11, "AttentionResidual", "P5"),
    LayerSpec("fsnet_backbone_p2", 3, "FSNet-backbone", "P2"),
    LayerSpec("fsnet_backbone_p3", 6, "FSNet-backbone", "P3"),
    LayerSpec("fsnet_backbone_p4", 9, "FSNet-backbone", "P4"),
    LayerSpec("fsnet_backbone_p5", 12, "FSNet-backbone", "P5"),
    LayerSpec("csar_stage1_p2", 16, "CSAR-stage1", "P2"),
    LayerSpec("csar_stage1_p3", 17, "CSAR-stage1", "P3"),
    LayerSpec("csar_stage1_p4", 18, "CSAR-stage1", "P4"),
    LayerSpec("csar_stage1_p5", 19, "CSAR-stage1", "P5"),
    LayerSpec("csar_stage1_p6", 20, "CSAR-stage1", "P6"),
    LayerSpec("fsnet_neck_p2", 21, "FSNet-neck", "P2"),
    LayerSpec("fsnet_neck_p3", 22, "FSNet-neck", "P3"),
    LayerSpec("fsnet_neck_p4", 23, "FSNet-neck", "P4"),
    LayerSpec("fsnet_neck_p5", 24, "FSNet-neck", "P5"),
    LayerSpec("fsnet_neck_p6", 25, "FSNet-neck", "P6"),
    LayerSpec("csar_stage2_p2", 26, "CSAR-stage2", "P2"),
    LayerSpec("csar_stage2_p3", 27, "CSAR-stage2", "P3"),
    LayerSpec("csar_stage2_p4", 28, "CSAR-stage2", "P4"),
    LayerSpec("csar_stage2_p5", 29, "CSAR-stage2", "P5"),
    LayerSpec("csar_stage2_p6", 30, "CSAR-stage2", "P6"),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--images", type=Path, default=None, help="Override <dataset>/images/train.")
    parser.add_argument("--labels", type=Path, default=None, help="Override <dataset>/labels/train.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "feature_suppression_0912")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dent-class", type=int, default=2)
    parser.add_argument("--rust-class", type=int, default=5)
    parser.add_argument("--max-pure-images", type=int, default=6)
    parser.add_argument("--max-overlap-images", type=int, default=6)
    parser.add_argument("--max-overlap-rois", type=int, default=12)
    parser.add_argument("--max-rois-per-overlap-image", type=int, default=2)
    parser.add_argument("--scan-size", type=int, default=384)
    parser.add_argument("--min-overlap-pixels", type=int, default=16)
    parser.add_argument("--similarity-drop", type=float, default=0.10)
    parser.add_argument("--score-ratio", type=float, default=0.75)
    parser.add_argument("--no-gradcam", action="store_true")
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    """Read a Unicode-safe BGR image."""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def parse_label(path: Path) -> tuple[Instance, ...]:
    """Parse YOLO box or polygon labels as normalized polygons."""
    instances: list[Instance] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.strip().split()
        if not fields:
            continue
        try:
            class_id = int(float(fields[0]))
            coords = np.asarray([float(value) for value in fields[1:]], dtype=np.float32)
        except ValueError as error:
            raise ValueError(f"Invalid label at {path}:{line_number}") from error
        if len(coords) == 4:
            cx, cy, width, height = coords
            points = np.asarray(
                [
                    [cx - width / 2, cy - height / 2],
                    [cx + width / 2, cy - height / 2],
                    [cx + width / 2, cy + height / 2],
                    [cx - width / 2, cy + height / 2],
                ],
                dtype=np.float32,
            )
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            points = coords.reshape(-1, 2)
        else:
            raise ValueError(f"Expected a box or polygon at {path}:{line_number}")
        instances.append(Instance(class_id, np.clip(points, 0.0, 1.0)))
    return tuple(instances)


def build_image_index(images_dir: Path) -> dict[str, Path]:
    """Map label-relative stems to image paths."""
    index: dict[str, Path] = {}
    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            relative = path.relative_to(images_dir)
            key = str(relative.with_suffix("")).replace("\\", "/").lower()
            index[key] = path
    return index


def load_samples(images_dir: Path, labels_dir: Path) -> list[Sample]:
    """Load labels that have a matching image."""
    image_index = build_image_index(images_dir)
    samples: list[Sample] = []
    for label_path in sorted(labels_dir.rglob("*.txt")):
        key = str(label_path.relative_to(labels_dir).with_suffix("")).replace("\\", "/").lower()
        image_path = image_index.get(key)
        if image_path is None:
            continue
        instances = parse_label(label_path)
        if instances:
            samples.append(Sample(image_path, label_path, instances))
    return samples


def normalized_mask(instance: Instance, size: int) -> np.ndarray:
    """Rasterize a normalized polygon for sample discovery."""
    mask = np.zeros((size, size), dtype=np.uint8)
    points = np.rint(instance.points * np.asarray([size - 1, size - 1])).astype(np.int32)
    if len(points) >= 3:
        cv2.fillPoly(mask, [points], 1)
    return mask


def normalized_area(instance: Instance, size: int = 1024) -> int:
    """Return a stable rasterized proxy for normalized polygon area."""
    return int(normalized_mask(instance, size).sum())


def discover_samples(
    samples: Iterable[Sample],
    dent_class: int,
    rust_class: int,
    scan_size: int,
    min_overlap_pixels: int,
) -> tuple[list[tuple[int, Sample]], list[tuple[int, Sample]], list[tuple[float, Sample, int, int]]]:
    """Split the dataset into pure-class and true polygon-overlap candidates."""
    dent_only: list[tuple[int, Sample]] = []
    rust_only: list[tuple[int, Sample]] = []
    overlaps: list[tuple[float, Sample, int, int]] = []

    for sample in samples:
        dent_indices = [i for i, item in enumerate(sample.instances) if item.class_id == dent_class]
        rust_indices = [i for i, item in enumerate(sample.instances) if item.class_id == rust_class]
        if dent_indices and not rust_indices:
            area = sum(normalized_area(sample.instances[i], scan_size) for i in dent_indices)
            dent_only.append((area, sample))
        if rust_indices and not dent_indices:
            area = sum(normalized_area(sample.instances[i], scan_size) for i in rust_indices)
            rust_only.append((area, sample))
        if not dent_indices or not rust_indices:
            continue
        masks = {i: normalized_mask(sample.instances[i], scan_size) for i in dent_indices + rust_indices}
        for dent_index in dent_indices:
            dent_mask = masks[dent_index]
            dent_area = max(int(dent_mask.sum()), 1)
            for rust_index in rust_indices:
                rust_mask = masks[rust_index]
                intersection = int(np.logical_and(dent_mask, rust_mask).sum())
                if intersection < min_overlap_pixels:
                    continue
                union = max(int(np.logical_or(dent_mask, rust_mask).sum()), 1)
                smaller = max(min(dent_area, int(rust_mask.sum())), 1)
                relative_overlap = intersection / smaller + intersection / union
                # Prefer an ROI with enough spatial support at coarse feature scales.
                # The tiny relative-overlap tie breaker keeps deterministic ordering
                # without letting a one-pixel perfect overlap outrank a large region.
                rank = intersection / float(scan_size * scan_size) + relative_overlap * 1e-9
                overlaps.append((rank, sample, dent_index, rust_index))

    dent_only.sort(key=lambda item: (-item[0], str(item[1].image_path).lower()))
    rust_only.sort(key=lambda item: (-item[0], str(item[1].image_path).lower()))
    overlaps.sort(key=lambda item: (-item[0], str(item[1].image_path).lower(), item[2], item[3]))
    return dent_only, rust_only, overlaps


def select_unique_samples(candidates: list[tuple[int, Sample]], limit: int) -> list[Sample]:
    """Select high-area samples while avoiding duplicated image stems across source folders."""
    selected: list[Sample] = []
    seen_stems: set[str] = set()
    for _area, sample in candidates:
        stem = sample.image_path.stem.casefold()
        if stem in seen_stems:
            continue
        selected.append(sample)
        seen_stems.add(stem)
        if len(selected) >= limit:
            break
    return selected


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Apply the standard centered YOLO letterbox transform."""
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_width, new_height = round(width * ratio), round(height * ratio)
    pad_width = (size - new_width) / 2
    pad_height = (size - new_height) / 2
    left, right = round(pad_width - 0.1), round(pad_width + 0.1)
    top, bottom = round(pad_height - 0.1), round(pad_height + 0.1)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    transformed = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return transformed, ratio, left, top


def instance_mask(
    instance: Instance,
    original_shape: tuple[int, int],
    size: int,
    ratio: float,
    left: int,
    top: int,
) -> np.ndarray:
    """Rasterize an instance in letterboxed model-input coordinates."""
    original_height, original_width = original_shape
    points = instance.points * np.asarray([original_width, original_height], dtype=np.float32)
    points = points * ratio + np.asarray([left, top], dtype=np.float32)
    points = np.rint(points).astype(np.int32)
    mask = np.zeros((size, size), dtype=np.uint8)
    if len(points) >= 3:
        cv2.fillPoly(mask, [points], 1)
    return mask


def prepare_sample(sample: Sample, size: int) -> tuple[np.ndarray, torch.Tensor, list[np.ndarray]]:
    """Return letterboxed BGR image, RGB model tensor, and transformed masks."""
    original = read_image(sample.image_path)
    transformed, ratio, left, top = letterbox(original, size)
    masks = [instance_mask(item, original.shape[:2], size, ratio, left, top) for item in sample.instances]
    rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().unsqueeze(0) / 255.0
    return transformed, tensor, masks


def tensor_from_output(output: Any) -> torch.Tensor:
    """Get the feature tensor from a hooked output."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported hooked output type: {type(output)!r}")


def extract_prediction_dict(output: Any) -> dict[str, Any]:
    """Find the raw pre-NMS prediction dictionary in a model output."""
    if isinstance(output, dict) and "scores" in output:
        return output
    if isinstance(output, (tuple, list)):
        for item in reversed(output):
            try:
                return extract_prediction_dict(item)
            except (TypeError, KeyError):
                pass
    raise TypeError("Could not find a pre-NMS prediction dictionary in model output.")


def masked_pool(feature: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    """Pool one BCHW feature tensor over a possibly small ROI mask."""
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError(f"Expected a single BCHW feature tensor, got {tuple(feature.shape)}")
    weights = torch.from_numpy(mask).to(device=feature.device, dtype=feature.dtype)[None, None]
    weights = F.interpolate(weights, size=feature.shape[-2:], mode="area")
    denominator = weights.sum().clamp_min(1e-6)
    return (feature * weights).sum(dim=(-2, -1)).squeeze(0) / denominator


def roi_values(grid: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    """Select CxN grid values whose cells overlap an ROI."""
    weights = torch.from_numpy(mask).to(device=grid.device, dtype=torch.float32)[None, None]
    weights = F.interpolate(weights, size=grid.shape[-2:], mode="area").flatten()
    selected = weights > 1e-6
    if not bool(selected.any()):
        selected[weights.argmax()] = True
    return grid.squeeze(0).flatten(1)[:, selected]


def topk_mean(values: torch.Tensor, fraction: float = 0.10, minimum: int = 1, maximum: int = 20) -> float:
    """Average the highest ROI scores without letting background cells dominate."""
    count = max(minimum, min(maximum, math.ceil(values.numel() * fraction)))
    return float(values.topk(min(count, values.numel())).values.mean().item())


def roi_raw_scores(predictions: dict[str, Any], mask: np.ndarray, class_ids: tuple[int, int]) -> dict[str, float]:
    """Pool Detect-head and MSAT probabilities before NMS in the same ROI."""
    scores = predictions["scores"].sigmoid()
    features = predictions["feats"]
    head_by_class: dict[int, list[torch.Tensor]] = {class_id: [] for class_id in class_ids}
    offset = 0
    for feature in features:
        height, width = feature.shape[-2:]
        count = height * width
        level_scores = scores[:, :, offset : offset + count].reshape(1, scores.shape[1], height, width)
        selected = roi_values(level_scores, mask)
        for class_id in class_ids:
            head_by_class[class_id].append(selected[class_id])
        offset += count

    auxiliary = predictions.get("multilabel_logits") or []
    aux_by_class: dict[int, list[torch.Tensor]] = {class_id: [] for class_id in class_ids}
    for logits in auxiliary:
        selected = roi_values(logits.sigmoid(), mask)
        for class_id in class_ids:
            aux_by_class[class_id].append(selected[class_id])

    metrics: dict[str, float] = {}
    labels = ("dent", "rust")
    for label, class_id in zip(labels, class_ids):
        head_values = torch.cat(head_by_class[class_id])
        metrics[f"head_{label}_max"] = float(head_values.max().item())
        metrics[f"head_{label}_topk"] = topk_mean(head_values)
        if aux_by_class[class_id]:
            aux_values = torch.cat(aux_by_class[class_id])
            metrics[f"aux_{label}_max"] = float(aux_values.max().item())
            metrics[f"aux_{label}_topk"] = topk_mean(aux_values)
    return metrics


def forward_capture(
    network: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Run one no-grad forward pass and capture configured layer outputs."""
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for spec in LAYER_SPECS:

        def record(_module, _inputs, output, name=spec.name):
            captured[name] = tensor_from_output(output).detach()

        handles.append(network.model[spec.index].register_forward_hook(record))
    try:
        with torch.inference_mode():
            output = network(tensor.to(device))
        predictions = extract_prediction_dict(output)
        return captured, predictions
    finally:
        for handle in handles:
            handle.remove()


def cosine(vector: torch.Tensor, prototype: torch.Tensor) -> float:
    """Return cosine similarity as a Python float."""
    return float(F.cosine_similarity(vector.float(), prototype.float(), dim=0).item())


def make_prototype(vectors: list[torch.Tensor]) -> torch.Tensor:
    """Build a normalized mean prototype from normalized ROI vectors."""
    stacked = torch.stack([F.normalize(vector.float(), dim=0) for vector in vectors])
    return F.normalize(stacked.mean(dim=0), dim=0)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries with the union of their fields."""
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_sample_manifest(
    output: Path,
    dent_samples: list[Sample],
    rust_samples: list[Sample],
    overlap_pairs: list[tuple[float, Sample, int, int]],
) -> list[dict[str, Any]]:
    """Record exactly which images and instance indices were used."""
    rows: list[dict[str, Any]] = []
    for category, selected in (("dent_only", dent_samples), ("rust_only", rust_samples)):
        rows.extend(
            {"category": category, "image": str(sample.image_path), "label": str(sample.label_path)}
            for sample in selected
        )
    rows.extend(
        {
            "category": "dent_rust_overlap",
            "image": str(sample.image_path),
            "label": str(sample.label_path),
            "dent_instance": dent_index,
            "rust_instance": rust_index,
            "overlap_rank": rank,
        }
        for rank, sample, dent_index, rust_index in overlap_pairs
    )
    write_csv(output / "sample_manifest.csv", rows)
    return rows


def plot_selected_samples(
    output: Path,
    dent_samples: list[Sample],
    rust_samples: list[Sample],
    overlap_pairs: list[tuple[float, Sample, int, int]],
    dent_class: int,
    rust_class: int,
) -> None:
    """Render one representative input from each test category."""
    if not dent_samples or not rust_samples or not overlap_pairs:
        return
    selections = [
        ("Dent only", dent_samples[0], None),
        ("Rust only", rust_samples[0], None),
        ("Dent + Rust overlap", overlap_pairs[0][1], overlap_pairs[0][2:]),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {dent_class: (0, 1, 1), rust_class: (1, 0.35, 0)}
    for axis, (title, sample, pair) in zip(axes, selections):
        image = cv2.cvtColor(read_image(sample.image_path), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        axis.imshow(image)
        indices = pair if pair is not None else range(len(sample.instances))
        for index in indices:
            instance = sample.instances[index]
            if instance.class_id not in colors:
                continue
            points = instance.points * np.asarray([width, height])
            closed = np.vstack([points, points[0]])
            axis.plot(closed[:, 0], closed[:, 1], color=colors[instance.class_id], linewidth=2)
        axis.set_title(f"{title}\n{sample.image_path.name}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output / "selected_samples.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def objective_from_predictions(predictions: dict[str, Any], mask: np.ndarray, class_id: int) -> torch.Tensor:
    """Build a differentiable top-ROI pre-NMS class objective."""
    scores = predictions["scores"].sigmoid()
    features = predictions["feats"]
    selected_values: list[torch.Tensor] = []
    offset = 0
    for feature in features:
        height, width = feature.shape[-2:]
        count = height * width
        grid = scores[:, :, offset : offset + count].reshape(1, scores.shape[1], height, width)
        selected_values.append(roi_values(grid, mask)[class_id])
        offset += count
    values = torch.cat(selected_values)
    count = min(max(1, math.ceil(values.numel() * 0.05)), 12)
    return values.topk(count).values.mean()


def gradcam_for_class(
    network: torch.nn.Module,
    tensor: torch.Tensor,
    mask: np.ndarray,
    class_id: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], float]:
    """Compute Grad-CAM maps for all configured feature layers."""
    activations: dict[str, torch.Tensor] = {}
    handles = []
    for spec in LAYER_SPECS:

        def record(_module, _inputs, output, name=spec.name):
            feature = tensor_from_output(output)
            feature.retain_grad()
            activations[name] = feature

        handles.append(network.model[spec.index].register_forward_hook(record))
    try:
        network.zero_grad(set_to_none=True)
        # A previous torch.inference_mode() pass may leave the Detect head's cached
        # anchor tensors marked as inference tensors. Force a normal autograd-time
        # rebuild before computing Grad-CAM; model parameters remain unchanged.
        head = network.model[-1]
        if hasattr(head, "shape"):
            head.shape = None
        model_input = tensor.to(device).detach().clone().requires_grad_(True)
        output = network(model_input)
        predictions = extract_prediction_dict(output)
        objective = objective_from_predictions(predictions, mask, class_id)
        objective.backward()
        cams: dict[str, np.ndarray] = {}
        for spec in LAYER_SPECS:
            activation = activations[spec.name]
            gradient = activation.grad
            if gradient is None:
                continue
            weights = gradient.mean(dim=(-2, -1), keepdim=True)
            cam = (weights * activation).sum(dim=1, keepdim=True).relu()
            cam = F.interpolate(cam, size=mask.shape, mode="bilinear", align_corners=False)[0, 0]
            cam -= cam.min()
            cam /= cam.max().clamp_min(1e-6)
            cams[spec.name] = cam.detach().cpu().numpy()
        return cams, float(objective.detach().item())
    finally:
        network.zero_grad(set_to_none=True)
        for handle in handles:
            handle.remove()


def aggregate_cams(cams: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Average layer CAMs within each architectural stage."""
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for spec in LAYER_SPECS:
        if spec.name in cams:
            grouped[spec.group].append(cams[spec.name])
    return {group: np.mean(values, axis=0) for group, values in grouped.items()}


def plot_gradcam(
    output: Path,
    image: np.ndarray,
    mask: np.ndarray,
    rust_cams: dict[str, np.ndarray],
    dent_cams: dict[str, np.ndarray],
    rust_score: float,
    dent_score: float,
    filename: str,
    title: str,
) -> None:
    """Save a two-class, stage-by-stage Grad-CAM comparison."""
    stage_order = ["AttentionResidual", "FSNet-backbone", "CSAR-stage1", "FSNet-neck", "CSAR-stage2"]
    rows = [("Dent", aggregate_cams(dent_cams), dent_score), ("Rust", aggregate_cams(rust_cams), rust_score)]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    figure, axes = plt.subplots(2, len(stage_order) + 1, figsize=(19, 7))
    contour = mask.astype(np.float32)
    for row_index, (class_name, cams, score) in enumerate(rows):
        axes[row_index, 0].imshow(rgb)
        axes[row_index, 0].contour(contour, levels=[0.5], colors=["cyan"], linewidths=1.2)
        axes[row_index, 0].set_title(f"{class_name} ROI\nobjective={score:.3f}")
        axes[row_index, 0].axis("off")
        for column, stage in enumerate(stage_order, 1):
            axes[row_index, column].imshow(rgb)
            cam = cams.get(stage)
            if cam is not None:
                axes[row_index, column].imshow(cam, cmap="jet", alpha=0.48, vmin=0, vmax=1)
            axes[row_index, column].contour(contour, levels=[0.5], colors=["white"], linewidths=0.8)
            axes[row_index, column].set_title(stage)
            axes[row_index, column].axis("off")
    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    figure.savefig(output / filename, dpi=180, bbox_inches="tight")
    plt.close(figure)


def safe_ratio(value: float, reference: float) -> float:
    """Compute a stable nonnegative ratio."""
    return value / max(reference, 1e-8)


def main() -> int:
    """Run the complete first-step diagnostic simulation."""
    args = parse_args()
    images_dir = (args.images or args.dataset / "images" / "train").resolve()
    labels_dir = (args.labels or args.dataset / "labels" / "train").resolve()
    weight_path = args.weights.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "runs" / ".matplotlib").mkdir(parents=True, exist_ok=True)

    for path, label in ((weight_path, "weight"), (images_dir, "images"), (labels_dir, "labels")):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    samples = load_samples(images_dir, labels_dir)
    dent_candidates, rust_candidates, overlap_candidates = discover_samples(
        samples,
        args.dent_class,
        args.rust_class,
        args.scan_size,
        args.min_overlap_pixels,
    )
    dent_samples = select_unique_samples(dent_candidates, args.max_pure_images)
    rust_samples = select_unique_samples(rust_candidates, args.max_pure_images)

    overlap_pairs: list[tuple[float, Sample, int, int]] = []
    used_overlap_images: set[Path] = set()
    used_overlap_stems: set[str] = set()
    overlap_count_by_stem: dict[str, int] = defaultdict(int)
    for candidate in overlap_candidates:
        _, sample, _, _ = candidate
        stem = sample.image_path.stem.casefold()
        if overlap_count_by_stem[stem] >= args.max_rois_per_overlap_image:
            continue
        if sample.image_path not in used_overlap_images:
            if stem in used_overlap_stems or len(used_overlap_images) >= args.max_overlap_images:
                continue
            used_overlap_stems.add(stem)
        overlap_pairs.append(candidate)
        used_overlap_images.add(sample.image_path)
        overlap_count_by_stem[stem] += 1
        if len(overlap_pairs) >= args.max_overlap_rois:
            break

    if not dent_samples or not rust_samples or not overlap_pairs:
        raise RuntimeError(
            "The scan did not find all three required groups: "
            f"Dent-only={len(dent_samples)}, Rust-only={len(rust_samples)}, overlap={len(overlap_pairs)}"
        )

    print(
        f"Scanned {len(samples)} labeled images; selected "
        f"{len(dent_samples)} Dent-only, {len(rust_samples)} Rust-only, "
        f"{len(used_overlap_images)} overlap images / {len(overlap_pairs)} overlap ROIs."
    )
    manifest_rows = save_sample_manifest(output, dent_samples, rust_samples, overlap_pairs)
    plot_selected_samples(output, dent_samples, rust_samples, overlap_pairs, args.dent_class, args.rust_class)

    device = torch.device(args.device)
    yolo = YOLO(str(weight_path))
    network = yolo.model.to(device).eval().float()
    class_names = {int(key): value for key, value in yolo.names.items()}
    if class_names.get(args.dent_class) != "D" or class_names.get(args.rust_class) != "R":
        print(
            "Warning: requested class mapping differs from checkpoint names: "
            f"Dent={class_names.get(args.dent_class)!r}, Rust={class_names.get(args.rust_class)!r}"
        )

    pure_vectors: dict[str, dict[str, list[torch.Tensor]]] = {
        "dent": defaultdict(list),
        "rust": defaultdict(list),
    }
    pure_score_rows: list[dict[str, Any]] = []
    pure_categories = (("dent", args.dent_class, dent_samples), ("rust", args.rust_class, rust_samples))
    for category, class_id, selected_samples in pure_categories:
        for sample in selected_samples:
            _image, tensor, masks = prepare_sample(sample, args.imgsz)
            features, predictions = forward_capture(network, tensor, device)
            for instance_index, instance in enumerate(sample.instances):
                if instance.class_id != class_id:
                    continue
                mask = masks[instance_index]
                if not mask.any():
                    continue
                for spec in LAYER_SPECS:
                    pure_vectors[category][spec.name].append(masked_pool(features[spec.name], mask).cpu())
                scores = roi_raw_scores(predictions, mask, (args.dent_class, args.rust_class))
                pure_score_rows.append(
                    {
                        "category": category,
                        "image": str(sample.image_path),
                        "instance_index": instance_index,
                        **scores,
                    }
                )
            print(f"Prototype forward: {category:4s} {sample.image_path.name}")

    prototypes: dict[str, dict[str, torch.Tensor]] = {"dent": {}, "rust": {}}
    reference_rows: list[dict[str, Any]] = []
    for category in ("dent", "rust"):
        for spec in LAYER_SPECS:
            vectors = pure_vectors[category][spec.name]
            if not vectors:
                raise RuntimeError(f"No {category} prototype vectors for {spec.name}")
            prototype = make_prototype(vectors)
            prototypes[category][spec.name] = prototype
            similarities = [cosine(vector, prototype) for vector in vectors]
            reference_rows.append(
                {
                    "category": category,
                    "layer": spec.name,
                    "group": spec.group,
                    "scale": spec.scale,
                    "roi_count": len(vectors),
                    "similarity_mean": float(np.mean(similarities)),
                    "similarity_median": float(median(similarities)),
                    "similarity_min": float(min(similarities)),
                }
            )

    score_reference: dict[str, float] = {}
    for category in ("dent", "rust"):
        rows = [row for row in pure_score_rows if row["category"] == category]
        own_label = category
        for source in ("head", "aux"):
            key = f"{source}_{own_label}_topk"
            values = [float(row[key]) for row in rows if key in row]
            if values:
                score_reference[f"{category}_{source}"] = float(median(values))

    write_csv(output / "prototype_reference.csv", reference_rows)
    write_csv(output / "pure_roi_scores.csv", pure_score_rows)

    reference_lookup = {(row["category"], row["layer"]): float(row["similarity_median"]) for row in reference_rows}
    detailed_rows: list[dict[str, Any]] = []
    overlap_score_rows: list[dict[str, Any]] = []
    gradcam_payloads: dict[str, tuple[np.ndarray, torch.Tensor, np.ndarray]] = {}
    largest_overlap_roi_id: str | None = None
    overlap_cache: dict[
        Path, tuple[np.ndarray, torch.Tensor, list[np.ndarray], dict[str, torch.Tensor], dict[str, Any]]
    ] = {}
    for roi_index, (rank, sample, dent_index, rust_index) in enumerate(overlap_pairs):
        if sample.image_path not in overlap_cache:
            image, tensor, masks = prepare_sample(sample, args.imgsz)
            features, predictions = forward_capture(network, tensor, device)
            overlap_cache[sample.image_path] = (image, tensor, masks, features, predictions)
            print(f"Overlap forward:       {sample.image_path.name}")
        image, tensor, masks, features, predictions = overlap_cache[sample.image_path]
        overlap_mask = np.logical_and(masks[dent_index], masks[rust_index]).astype(np.uint8)
        if not overlap_mask.any():
            continue
        roi_id = f"{sample.image_path.stem}_d{dent_index}_r{rust_index}"
        scores = roi_raw_scores(predictions, overlap_mask, (args.dent_class, args.rust_class))
        score_row = {
            "roi_id": roi_id,
            "image": str(sample.image_path),
            "dent_instance": dent_index,
            "rust_instance": rust_index,
            "input_overlap_pixels": int(overlap_mask.sum()),
            "overlap_rank": rank,
            **scores,
        }
        for category in ("dent", "rust"):
            for source in ("head", "aux"):
                key = f"{source}_{category}_topk"
                reference = score_reference.get(f"{category}_{source}")
                if reference is not None and key in score_row:
                    score_row[f"{source}_{category}_pure_median"] = reference
                    score_row[f"{source}_{category}_ratio"] = safe_ratio(float(score_row[key]), reference)
        overlap_score_rows.append(score_row)

        for spec in LAYER_SPECS:
            vector = masked_pool(features[spec.name], overlap_mask).cpu()
            dent_similarity = cosine(vector, prototypes["dent"][spec.name])
            rust_similarity = cosine(vector, prototypes["rust"][spec.name])
            detailed_rows.append(
                {
                    "roi_id": roi_id,
                    "image": str(sample.image_path),
                    "layer": spec.name,
                    "layer_index": spec.index,
                    "group": spec.group,
                    "scale": spec.scale,
                    "dent_similarity": dent_similarity,
                    "rust_similarity": rust_similarity,
                    "dent_pure_median": reference_lookup[("dent", spec.name)],
                    "rust_pure_median": reference_lookup[("rust", spec.name)],
                    "dent_similarity_delta": dent_similarity - reference_lookup[("dent", spec.name)],
                    "rust_similarity_delta": rust_similarity - reference_lookup[("rust", spec.name)],
                }
            )
        gradcam_payloads[roi_id] = (image, tensor, overlap_mask)
        if largest_overlap_roi_id is None:
            largest_overlap_roi_id = roi_id

    if not detailed_rows:
        raise RuntimeError("No overlap ROI survived the model-input transform.")
    write_csv(output / "feature_similarity.csv", detailed_rows)
    write_csv(output / "overlap_roi_scores.csv", overlap_score_rows)

    group_rows: list[dict[str, Any]] = []
    groups = list(dict.fromkeys(spec.group for spec in LAYER_SPECS))
    roi_ids = list(dict.fromkeys(row["roi_id"] for row in detailed_rows))
    for roi_id in roi_ids:
        score_row = next(row for row in overlap_score_rows if row["roi_id"] == roi_id)
        for group in groups:
            rows = [row for row in detailed_rows if row["roi_id"] == roi_id and row["group"] == group]
            dent_delta = float(np.mean([row["dent_similarity_delta"] for row in rows]))
            rust_delta = float(np.mean([row["rust_similarity_delta"] for row in rows]))
            dent_ratio = float(score_row.get("head_dent_ratio", float("nan")))
            rust_ratio = float(score_row.get("head_rust_ratio", float("nan")))
            dent_candidate = (
                dent_delta <= -args.similarity_drop
                and dent_ratio <= args.score_ratio
                and dent_delta <= rust_delta - 0.05
            )
            rust_candidate = (
                rust_delta <= -args.similarity_drop
                and rust_ratio <= args.score_ratio
                and rust_delta <= dent_delta - 0.05
            )
            group_rows.append(
                {
                    "roi_id": roi_id,
                    "group": group,
                    "dent_similarity": float(np.mean([row["dent_similarity"] for row in rows])),
                    "rust_similarity": float(np.mean([row["rust_similarity"] for row in rows])),
                    "dent_similarity_delta": dent_delta,
                    "rust_similarity_delta": rust_delta,
                    "head_dent_ratio": dent_ratio,
                    "head_rust_ratio": rust_ratio,
                    "candidate_dent_suppression": dent_candidate,
                    "candidate_rust_suppression": rust_candidate,
                }
            )
    write_csv(output / "group_roi_summary.csv", group_rows)

    aggregate_rows: list[dict[str, Any]] = []
    for group in groups:
        rows = [row for row in group_rows if row["group"] == group]
        aggregate_rows.append(
            {
                "group": group,
                "roi_count": len(rows),
                "dent_similarity_mean": float(np.mean([row["dent_similarity"] for row in rows])),
                "rust_similarity_mean": float(np.mean([row["rust_similarity"] for row in rows])),
                "dent_similarity_delta_mean": float(np.mean([row["dent_similarity_delta"] for row in rows])),
                "rust_similarity_delta_mean": float(np.mean([row["rust_similarity_delta"] for row in rows])),
                "head_dent_ratio_mean": float(np.mean([row["head_dent_ratio"] for row in rows])),
                "head_rust_ratio_mean": float(np.mean([row["head_rust_ratio"] for row in rows])),
                "dent_suppression_flags": sum(bool(row["candidate_dent_suppression"]) for row in rows),
                "rust_suppression_flags": sum(bool(row["candidate_rust_suppression"]) for row in rows),
            }
        )
    write_csv(output / "stage_summary.csv", aggregate_rows)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    positions = np.arange(len(aggregate_rows))
    width = 0.36
    axes[0].bar(
        positions - width / 2,
        [row["dent_similarity_delta_mean"] for row in aggregate_rows],
        width,
        label="Dent",
        color="#00a6a6",
    )
    axes[0].bar(
        positions + width / 2,
        [row["rust_similarity_delta_mean"] for row in aggregate_rows],
        width,
        label="Rust",
        color="#d75b25",
    )
    axes[0].axhline(-args.similarity_drop, color="black", linestyle="--", linewidth=1, label="flag threshold")
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].set_title("Overlap similarity minus pure-class reference")
    axes[0].set_ylabel("cosine similarity delta")
    axes[0].set_xticks(positions, [row["group"] for row in aggregate_rows], rotation=25, ha="right")
    axes[0].legend()

    dent_ratios = [row["head_dent_ratio"] for row in overlap_score_rows]
    rust_ratios = [row["head_rust_ratio"] for row in overlap_score_rows]
    axes[1].bar([0, 1], [np.mean(dent_ratios), np.mean(rust_ratios)], color=["#00a6a6", "#d75b25"])
    axes[1].axhline(1, color="gray", linewidth=0.8)
    axes[1].axhline(args.score_ratio, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks([0, 1], ["Dent", "Rust"])
    axes[1].set_ylabel("overlap / pure median pre-NMS score")
    axes[1].set_title("Detect-head raw score ratio in the same ROI")
    figure.tight_layout()
    figure.savefig(output / "suppression_overview.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    gradcam_status = "skipped by --no-gradcam"
    if not args.no_gradcam and largest_overlap_roi_id is not None:
        image, tensor, overlap_mask = gradcam_payloads[largest_overlap_roi_id]
        dent_cams, dent_objective = gradcam_for_class(network, tensor, overlap_mask, args.dent_class, device)
        rust_cams, rust_objective = gradcam_for_class(network, tensor, overlap_mask, args.rust_class, device)
        plot_gradcam(
            output,
            image,
            overlap_mask,
            rust_cams,
            dent_cams,
            rust_objective,
            dent_objective,
            "gradcam_stage_comparison.png",
            f"Large-overlap control: {largest_overlap_roi_id}",
        )
        flagged_rows = [
            row for row in group_rows if row["candidate_dent_suppression"] or row["candidate_rust_suppression"]
        ]
        if flagged_rows:

            def candidate_strength(row: dict[str, Any]) -> float:
                values = []
                if row["candidate_dent_suppression"]:
                    values.append(float(row["dent_similarity_delta"]))
                if row["candidate_rust_suppression"]:
                    values.append(float(row["rust_similarity_delta"]))
                return min(values)

            candidate_roi_id = min(flagged_rows, key=candidate_strength)["roi_id"]
            candidate_image, candidate_tensor, candidate_mask = gradcam_payloads[candidate_roi_id]
            candidate_dent_cams, candidate_dent_objective = gradcam_for_class(
                network, candidate_tensor, candidate_mask, args.dent_class, device
            )
            candidate_rust_cams, candidate_rust_objective = gradcam_for_class(
                network, candidate_tensor, candidate_mask, args.rust_class, device
            )
            plot_gradcam(
                output,
                candidate_image,
                candidate_mask,
                candidate_rust_cams,
                candidate_dent_cams,
                candidate_rust_objective,
                candidate_dent_objective,
                "gradcam_candidate_comparison.png",
                f"Strongest suppression candidate: {candidate_roi_id}",
            )
            gradcam_status = (
                f"generated for large-overlap control `{largest_overlap_roi_id}` and candidate `{candidate_roi_id}`"
            )
        else:
            gradcam_status = f"generated for large-overlap control `{largest_overlap_roi_id}`; no candidate was flagged"

    metadata = {
        "weights": str(weight_path),
        "checkpoint_names": class_names,
        "images": str(images_dir),
        "labels": str(labels_dir),
        "imgsz": args.imgsz,
        "device": str(device),
        "scanned_labeled_images": len(samples),
        "available_dent_only": len(dent_candidates),
        "available_rust_only": len(rust_candidates),
        "available_overlap_pairs": len(overlap_candidates),
        "selected_manifest_rows": len(manifest_rows),
        "dent_prototype_images": len(dent_samples),
        "rust_prototype_images": len(rust_samples),
        "overlap_images": len(used_overlap_images),
        "overlap_rois": len(roi_ids),
        "max_rois_per_overlap_image": args.max_rois_per_overlap_image,
        "similarity_drop_threshold": args.similarity_drop,
        "score_ratio_threshold": args.score_ratio,
        "gradcam": gradcam_status,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    dent_flags = sum(row["dent_suppression_flags"] for row in aggregate_rows)
    rust_flags = sum(row["rust_suppression_flags"] for row in aggregate_rows)
    dent_flag_rois = {row["roi_id"] for row in group_rows if row["candidate_dent_suppression"]}
    rust_flag_rois = {row["roi_id"] for row in group_rows if row["candidate_rust_suppression"]}
    mean_dent_score_ratio = float(np.mean([row["head_dent_ratio"] for row in overlap_score_rows]))
    mean_rust_score_ratio = float(np.mean([row["head_rust_ratio"] for row in overlap_score_rows]))
    if mean_dent_score_ratio > args.score_ratio and mean_rust_score_ratio > args.score_ratio:
        screening_conclusion = (
            "The mean pre-NMS ratios do not support consistent dataset-level suppression of either class; "
            "the flags below are ROI-specific candidates for follow-up."
        )
    elif mean_dent_score_ratio <= args.score_ratio < mean_rust_score_ratio:
        screening_conclusion = "The aggregate screen supports a Dent-suppression candidate relative to Rust."
    elif mean_rust_score_ratio <= args.score_ratio < mean_dent_score_ratio:
        screening_conclusion = "The aggregate screen supports a Rust-suppression candidate relative to Dent."
    else:
        screening_conclusion = (
            "Both classes lose aggregate pre-NMS response, so this screen cannot isolate one suppressor."
        )
    report_lines = [
        "# 0912 Dent/Rust feature-suppression simulation",
        "",
        "## Scope",
        "",
        f"- Weight: `{weight_path}`",
        f"- Input size/device: `{args.imgsz}` / `{device}`",
        f"- Prototype images: Dent-only `{len(dent_samples)}`, Rust-only `{len(rust_samples)}`",
        f"- Evaluated overlap data: `{len(used_overlap_images)}` images, `{len(roi_ids)}` intersecting instance pairs",
        "- Features: AttentionResidual, backbone/neck FSNet, first/second CSAR stages at every available scale",
        "- Raw scores: sigmoid probabilities from the Detect classification head and MSAT auxiliary logits before NMS",
        f"- Grad-CAM: {gradcam_status}",
        "",
        "## Exploratory result",
        "",
        f"- Candidate Dent-suppression flags across ROI × stage comparisons: `{dent_flags}`",
        f"- Candidate Rust-suppression flags across ROI × stage comparisons: `{rust_flags}`",
        f"- Unique Dent-candidate ROIs: `{len(dent_flag_rois)}/{len(roi_ids)}`",
        f"- Unique Rust-candidate ROIs: `{len(rust_flag_rois)}/{len(roi_ids)}`",
        f"- Mean Dent/Rust pre-NMS score ratios: `{mean_dent_score_ratio:.4f}` / `{mean_rust_score_ratio:.4f}`",
        f"- Screening conclusion: {screening_conclusion}",
        "",
        (
            "A flag requires all three conditions: the class similarity is at least "
            f"`{args.similarity_drop:.2f}` below its pure-class median, its pre-NMS score ratio is at most "
            f"`{args.score_ratio:.2f}`, and its similarity drop is at least `0.05` worse than the other class."
        ),
        "This is an explicit screening heuristic, not a trained decision boundary or proof of causality.",
        "",
        "## Stage means",
        "",
        "| Stage | Dent similarity Δ | Rust similarity Δ | Dent score ratio | Rust score ratio | D flags | R flags |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    report_lines.extend(
        "| {group} | {dent_similarity_delta_mean:.4f} | {rust_similarity_delta_mean:.4f} | "
        "{head_dent_ratio_mean:.4f} | {head_rust_ratio_mean:.4f} | "
        "{dent_suppression_flags} | {rust_suppression_flags} |".format(**row)
        for row in aggregate_rows
    )
    report_lines.extend(
        [
            "",
            "## Files",
            "",
            "- `suppression_overview.png`: stage similarity changes and raw-score ratios.",
            "- `gradcam_stage_comparison.png`: class-specific attention on one identical overlap ROI.",
            "- `gradcam_candidate_comparison.png`: class-specific attention on the strongest flagged ROI, when present.",
            "- `feature_similarity.csv`: per-layer, per-ROI cosine similarities.",
            "- `overlap_roi_scores.csv`: same-ROI pre-NMS Detect/MSAT scores.",
            "- `prototype_reference.csv`: pure-class reference distributions.",
            "- `sample_manifest.csv`: exact images and instance indices used.",
            "",
            "## Interpretation limit",
            "",
            (
                "This run only establishes whether the supplied checkpoint shows a repeatable suppression candidate in "
                "the selected labeled ROIs. A publication-grade conclusion needs a fixed held-out split, bootstrap "
                "confidence intervals, and controls for ROI size, damage severity, and illumination."
            ),
        ]
    )
    (output / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote report and figures to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
