"""Run the Dent/Rust suppression screen on one specified co-occurrence image.

Unlike ``simulate_feature_suppression.py``, this entry point does not require
the selected image to contain a pixel-level Dent/Rust mask intersection. It
reports true overlap separately, then compares each class's own annotated ROI
with pure-class feature prototypes and pre-NMS score references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "runs" / ".matplotlib"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulate_feature_suppression import (
    DEFAULT_DATASET,
    DEFAULT_WEIGHT,
    LAYER_SPECS,
    Sample,
    cosine,
    discover_samples,
    forward_capture,
    gradcam_for_class,
    load_samples,
    make_prototype,
    masked_pool,
    parse_label,
    prepare_sample,
    roi_raw_scores,
    safe_ratio,
    select_unique_samples,
    write_csv,
)

from ultralytics import YOLO

DEFAULT_IMAGE = Path(r"C:\Users\USER\Desktop\已標記\images\train\B\CBHU0708054-A.JPG")
DEFAULT_LABEL = Path(r"C:\Users\USER\Desktop\碩論\dataset0608\dataset0608\dataset\labels\train\B\CBHU0708054-A.txt")


def parse_args() -> argparse.Namespace:
    """Parse single-image experiment arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--label", type=Path, default=DEFAULT_LABEL)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--prototype-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "feature_suppression_CBHU0708054-A",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dent-class", type=int, default=2)
    parser.add_argument("--rust-class", type=int, default=5)
    parser.add_argument("--max-pure-images", type=int, default=6)
    parser.add_argument("--similarity-drop", type=float, default=0.10)
    parser.add_argument("--score-ratio", type=float, default=0.75)
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows through the shared CSV helper."""
    write_csv(path, rows)


def plot_annotated_input(
    output: Path,
    sample: Sample,
    dent_class: int,
    rust_class: int,
) -> None:
    """Draw all Dent and Rust polygons on the original selected image."""
    data = np.fromfile(sample.image_path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read {sample.image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    figure, axis = plt.subplots(figsize=(11, 8))
    axis.imshow(image)
    for instance in sample.instances:
        if instance.class_id not in (dent_class, rust_class):
            continue
        points = instance.points * np.asarray([width, height], dtype=np.float32)
        points = np.vstack([points, points[0]])
        color = "cyan" if instance.class_id == dent_class else "orangered"
        axis.plot(points[:, 0], points[:, 1], color=color, linewidth=1.8)
    axis.plot([], [], color="cyan", linewidth=2, label="Dent")
    axis.plot([], [], color="orangered", linewidth=2, label="Rust")
    axis.legend(loc="upper right")
    axis.set_title(f"Selected image annotations: {sample.image_path.name}")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output / "annotated_input.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_dual_mask_gradcam(
    output: Path,
    image: np.ndarray,
    dent_mask: np.ndarray,
    rust_mask: np.ndarray,
    dent_cams: dict[str, np.ndarray],
    rust_cams: dict[str, np.ndarray],
    dent_score: float,
    rust_score: float,
    group_map: dict[str, str],
) -> None:
    """Plot each class's Grad-CAM against that class's own annotated mask union."""
    stage_order = list(dict.fromkeys(group_map[spec.name] for spec in LAYER_SPECS))

    def grouped_cams(cams: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for spec in LAYER_SPECS:
            if spec.name in cams:
                grouped[group_map[spec.name]].append(cams[spec.name])
        return {group: np.mean(values, axis=0) for group, values in grouped.items()}

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rows = (
        ("Dent", dent_mask, grouped_cams(dent_cams), dent_score, "cyan"),
        ("Rust", rust_mask, grouped_cams(rust_cams), rust_score, "orangered"),
    )
    figure, axes = plt.subplots(2, len(stage_order) + 1, figsize=(19, 7))
    for row_index, (class_name, mask, cams, score, color) in enumerate(rows):
        axes[row_index, 0].imshow(rgb)
        axes[row_index, 0].contour(mask.astype(np.float32), levels=[0.5], colors=[color], linewidths=1.0)
        axes[row_index, 0].set_title(f"{class_name} union ROI\nobjective={score:.3f}")
        axes[row_index, 0].axis("off")
        for column, stage in enumerate(stage_order, 1):
            axes[row_index, column].imshow(rgb)
            cam = cams.get(stage)
            if cam is not None:
                axes[row_index, column].imshow(cam, cmap="jet", alpha=0.48, vmin=0, vmax=1)
            axes[row_index, column].contour(mask.astype(np.float32), levels=[0.5], colors=["white"], linewidths=0.7)
            axes[row_index, column].set_title(stage)
            axes[row_index, column].axis("off")
    figure.suptitle("Single-image Dent/Rust co-occurrence Grad-CAM (class-specific ROIs)", fontsize=14)
    figure.tight_layout()
    figure.savefig(output / "gradcam_cooccurrence.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    """Run the single-image co-occurrence diagnostic."""
    args = parse_args()
    image_path = args.image.resolve()
    label_path = args.label.resolve()
    weight_path = args.weights.resolve()
    prototype_images = (args.prototype_dataset / "images" / "train").resolve()
    prototype_labels = (args.prototype_dataset / "labels" / "train").resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    for path, description in (
        (image_path, "image"),
        (label_path, "polygon label"),
        (weight_path, "weight"),
        (prototype_images, "prototype images"),
        (prototype_labels, "prototype labels"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")

    target = Sample(image_path, label_path, parse_label(label_path))
    dent_indices = [i for i, item in enumerate(target.instances) if item.class_id == args.dent_class]
    rust_indices = [i for i, item in enumerate(target.instances) if item.class_id == args.rust_class]
    if not dent_indices or not rust_indices:
        raise RuntimeError(f"Selected image needs both classes; Dent={len(dent_indices)}, Rust={len(rust_indices)}")

    _dent, _rust, true_overlaps = discover_samples(
        [target], args.dent_class, args.rust_class, scan_size=384, min_overlap_pixels=1
    )
    plot_annotated_input(output, target, args.dent_class, args.rust_class)

    samples = load_samples(prototype_images, prototype_labels)
    dent_candidates, rust_candidates, _overlap_candidates = discover_samples(
        samples, args.dent_class, args.rust_class, scan_size=384, min_overlap_pixels=16
    )
    dent_samples = select_unique_samples(dent_candidates, args.max_pure_images)
    rust_samples = select_unique_samples(rust_candidates, args.max_pure_images)
    if not dent_samples or not rust_samples:
        raise RuntimeError("Unable to build both pure-class prototype groups.")

    device = torch.device(args.device)
    yolo = YOLO(str(weight_path))
    network = yolo.model.to(device).eval().float()
    group_map = {
        spec.name: (
            "MultiStateCSAR"
            if spec.group == "CSAR-stage2" and network.model[spec.index].__class__.__name__ == "MultiStateCSAR"
            else spec.group
        )
        for spec in LAYER_SPECS
    }

    pure_vectors: dict[str, dict[str, list[torch.Tensor]]] = {
        "dent": defaultdict(list),
        "rust": defaultdict(list),
    }
    pure_score_rows: list[dict[str, Any]] = []
    for category, class_id, selected in (
        ("dent", args.dent_class, dent_samples),
        ("rust", args.rust_class, rust_samples),
    ):
        for sample in selected:
            _image, tensor, masks = prepare_sample(sample, args.imgsz)
            features, predictions = forward_capture(network, tensor, device)
            for instance_index, instance in enumerate(sample.instances):
                if instance.class_id != class_id or not masks[instance_index].any():
                    continue
                for spec in LAYER_SPECS:
                    pure_vectors[category][spec.name].append(
                        masked_pool(features[spec.name], masks[instance_index]).cpu()
                    )
                scores = roi_raw_scores(predictions, masks[instance_index], (args.dent_class, args.rust_class))
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
    reference_similarity: dict[tuple[str, str], float] = {}
    prototype_rows: list[dict[str, Any]] = []
    for category in ("dent", "rust"):
        for spec in LAYER_SPECS:
            vectors = pure_vectors[category][spec.name]
            prototype = make_prototype(vectors)
            prototypes[category][spec.name] = prototype
            similarities = [cosine(vector, prototype) for vector in vectors]
            value = float(median(similarities))
            reference_similarity[(category, spec.name)] = value
            prototype_rows.append(
                {
                    "category": category,
                    "layer": spec.name,
                    "group": group_map[spec.name],
                    "scale": spec.scale,
                    "roi_count": len(vectors),
                    "similarity_median": value,
                }
            )

    score_reference: dict[str, float] = {}
    for category in ("dent", "rust"):
        rows = [row for row in pure_score_rows if row["category"] == category]
        for source in ("head", "aux"):
            key = f"{source}_{category}_topk"
            values = [float(row[key]) for row in rows if key in row]
            if values:
                score_reference[f"{category}_{source}"] = float(median(values))

    image, tensor, masks = prepare_sample(target, args.imgsz)
    features, predictions = forward_capture(network, tensor, device)
    dent_mask = np.logical_or.reduce([masks[index].astype(bool) for index in dent_indices]).astype(np.uint8)
    rust_mask = np.logical_or.reduce([masks[index].astype(bool) for index in rust_indices]).astype(np.uint8)
    intersection_pixels = int(np.logical_and(dent_mask, rust_mask).sum())

    union_masks = {"dent": dent_mask, "rust": rust_mask}
    layer_rows: list[dict[str, Any]] = []
    for spec in LAYER_SPECS:
        dent_vector = masked_pool(features[spec.name], dent_mask).cpu()
        rust_vector = masked_pool(features[spec.name], rust_mask).cpu()
        dent_similarity = cosine(dent_vector, prototypes["dent"][spec.name])
        rust_similarity = cosine(rust_vector, prototypes["rust"][spec.name])
        layer_rows.append(
            {
                "layer": spec.name,
                "layer_index": spec.index,
                "group": group_map[spec.name],
                "scale": spec.scale,
                "dent_similarity_to_dent": dent_similarity,
                "dent_similarity_to_rust": cosine(dent_vector, prototypes["rust"][spec.name]),
                "rust_similarity_to_rust": rust_similarity,
                "rust_similarity_to_dent": cosine(rust_vector, prototypes["dent"][spec.name]),
                "dent_pure_median": reference_similarity[("dent", spec.name)],
                "rust_pure_median": reference_similarity[("rust", spec.name)],
                "dent_similarity_delta": dent_similarity - reference_similarity[("dent", spec.name)],
                "rust_similarity_delta": rust_similarity - reference_similarity[("rust", spec.name)],
            }
        )

    union_score_rows: list[dict[str, Any]] = []
    for category in ("dent", "rust"):
        scores = roi_raw_scores(predictions, union_masks[category], (args.dent_class, args.rust_class))
        own_head = float(scores[f"head_{category}_topk"])
        row = {
            "roi": f"{category}_union",
            **scores,
            "own_head_pure_median": score_reference[f"{category}_head"],
            "own_head_ratio": safe_ratio(own_head, score_reference[f"{category}_head"]),
        }
        aux_key = f"aux_{category}_topk"
        aux_reference_key = f"{category}_aux"
        if aux_key in scores and aux_reference_key in score_reference:
            own_aux = float(scores[aux_key])
            row["own_aux_pure_median"] = score_reference[aux_reference_key]
            row["own_aux_ratio"] = safe_ratio(own_aux, score_reference[aux_reference_key])
        union_score_rows.append(row)

    instance_rows: list[dict[str, Any]] = []
    for index in dent_indices + rust_indices:
        category = "dent" if target.instances[index].class_id == args.dent_class else "rust"
        mask = masks[index]
        if not mask.any():
            continue
        scores = roi_raw_scores(predictions, mask, (args.dent_class, args.rust_class))
        layer_similarities = [
            cosine(masked_pool(features[spec.name], mask).cpu(), prototypes[category][spec.name])
            for spec in LAYER_SPECS
        ]
        own_head = float(scores[f"head_{category}_topk"])
        instance_rows.append(
            {
                "instance_index": index,
                "class": category,
                "input_pixels": int(mask.sum()),
                "mean_own_prototype_similarity": float(np.mean(layer_similarities)),
                f"head_{category}_topk": own_head,
                "own_head_ratio": safe_ratio(own_head, score_reference[f"{category}_head"]),
            }
        )

    dent_instance_rows = [row for row in instance_rows if row["class"] == "dent"]
    rust_instance_rows = [row for row in instance_rows if row["class"] == "rust"]
    dent_weak_instances = sum(float(row["own_head_ratio"]) <= args.score_ratio for row in dent_instance_rows)
    rust_weak_instances = sum(float(row["own_head_ratio"]) <= args.score_ratio for row in rust_instance_rows)

    groups = list(dict.fromkeys(group_map[spec.name] for spec in LAYER_SPECS))
    stage_rows: list[dict[str, Any]] = []
    for group in groups:
        rows = [row for row in layer_rows if row["group"] == group]
        stage_rows.append(
            {
                "group": group,
                "dent_similarity": float(np.mean([row["dent_similarity_to_dent"] for row in rows])),
                "rust_similarity": float(np.mean([row["rust_similarity_to_rust"] for row in rows])),
                "dent_similarity_delta": float(np.mean([row["dent_similarity_delta"] for row in rows])),
                "rust_similarity_delta": float(np.mean([row["rust_similarity_delta"] for row in rows])),
            }
        )

    dent_scores = next(row for row in union_score_rows if row["roi"] == "dent_union")
    rust_scores = next(row for row in union_score_rows if row["roi"] == "rust_union")
    dent_delta = float(np.mean([row["dent_similarity_delta"] for row in layer_rows]))
    rust_delta = float(np.mean([row["rust_similarity_delta"] for row in layer_rows]))
    dent_candidate = dent_delta <= -args.similarity_drop and dent_scores["own_head_ratio"] <= args.score_ratio
    rust_candidate = rust_delta <= -args.similarity_drop and rust_scores["own_head_ratio"] <= args.score_ratio

    dent_cams, dent_objective = gradcam_for_class(network, tensor, dent_mask, args.dent_class, device)
    rust_cams, rust_objective = gradcam_for_class(network, tensor, rust_mask, args.rust_class, device)
    plot_dual_mask_gradcam(
        output,
        image,
        dent_mask,
        rust_mask,
        dent_cams,
        rust_cams,
        dent_objective,
        rust_objective,
        group_map,
    )

    write_rows(output / "layer_similarity.csv", layer_rows)
    write_rows(output / "stage_summary.csv", stage_rows)
    write_rows(output / "union_roi_scores.csv", union_score_rows)
    write_rows(output / "instance_metrics.csv", instance_rows)
    write_rows(output / "prototype_reference.csv", prototype_rows)
    write_rows(output / "pure_roi_scores.csv", pure_score_rows)

    metadata = {
        "image": str(image_path),
        "image_sha256": sha256(image_path),
        "polygon_label": str(label_path),
        "weights": str(weight_path),
        "imgsz": args.imgsz,
        "device": str(device),
        "dent_instances": len(dent_indices),
        "rust_instances": len(rust_indices),
        "true_polygon_overlap_pairs": len(true_overlaps),
        "input_intersection_pixels": intersection_pixels,
        "prototype_dent_images": len(dent_samples),
        "prototype_rust_images": len(rust_samples),
        "final_feature_modules": sorted({network.model[spec.index].__class__.__name__ for spec in LAYER_SPECS}),
        "multilabel_aux_available": "dent_aux" in score_reference and "rust_aux" in score_reference,
        "dent_instances_below_score_ratio": dent_weak_instances,
        "rust_instances_below_score_ratio": rust_weak_instances,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if true_overlaps:
        scope_conclusion = "The selected image contains true pixel-level Dent/Rust overlap."
    else:
        scope_conclusion = (
            "The selected image contains both classes but no polygon intersection; this is a co-occurrence screen, "
            "not a direct overlap-suppression test."
        )
    if dent_candidate or rust_candidate:
        candidates = ", ".join(name for name, flag in (("Dent", dent_candidate), ("Rust", rust_candidate)) if flag)
        result_conclusion = f"The aggregate co-occurrence heuristic flags {candidates} for follow-up."
    else:
        result_conclusion = (
            "The aggregate co-occurrence heuristic does not flag either class as consistently suppressed."
        )

    def format_optional_ratio(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.4f}"

    report = [
        "# CBHU0708054-A single-image suppression simulation",
        "",
        "## Scope",
        "",
        f"- Image: `{image_path}`",
        f"- Weight: `{weight_path}`",
        f"- Instances: Dent `{len(dent_indices)}`, Rust `{len(rust_indices)}`",
        f"- True polygon-overlap pairs: `{len(true_overlaps)}`; transformed overlap pixels: `{intersection_pixels}`",
        f"- Pure prototypes: Dent-only `{len(dent_samples)}` images, Rust-only `{len(rust_samples)}` images",
        f"- Scope conclusion: {scope_conclusion}",
        "",
        "## Aggregate result",
        "",
        f"- Dent mean feature-similarity delta: `{dent_delta:.4f}`",
        f"- Rust mean feature-similarity delta: `{rust_delta:.4f}`",
        (
            f"- Dent Detect-head ratio / auxiliary ratio: `{dent_scores['own_head_ratio']:.4f}` / "
            f"`{format_optional_ratio(dent_scores.get('own_aux_ratio'))}`"
        ),
        (
            f"- Rust Detect-head ratio / auxiliary ratio: `{rust_scores['own_head_ratio']:.4f}` / "
            f"`{format_optional_ratio(rust_scores.get('own_aux_ratio'))}`"
        ),
        f"- Individual Dent ROIs below `{args.score_ratio:.2f}`: `{dent_weak_instances}/{len(dent_instance_rows)}`",
        f"- Individual Rust ROIs below `{args.score_ratio:.2f}`: `{rust_weak_instances}/{len(rust_instance_rows)}`",
        f"- Screening conclusion: {result_conclusion}",
        "",
        (
            "A candidate requires its mean prototype-similarity delta to be at most "
            f"`-{args.similarity_drop:.2f}` and its Detect-head ratio to be at most `{args.score_ratio:.2f}`."
        ),
        "",
        "## Stage means",
        "",
        "| Stage | Dent similarity | Dent Δ | Rust similarity | Rust Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    report.extend(
        f"| {row['group']} | {row['dent_similarity']:.4f} | {row['dent_similarity_delta']:.4f} | "
        f"{row['rust_similarity']:.4f} | {row['rust_similarity_delta']:.4f} |"
        for row in stage_rows
    )
    report.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Because the masks do not intersect, a low score can indicate co-occurrence, small-object difficulty, "
                "illumination, or appearance shift; it cannot by itself demonstrate that one damage type suppresses the other."
            ),
        ]
    )
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote single-image report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
