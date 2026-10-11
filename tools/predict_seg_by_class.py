"""Run YOLO segmentation inference and save N per-class images plus one combined image.

For every source image, this script always writes exactly N + 1 visualizations,
where N is the number of class names stored in the model. The per-class image
contains only predictions of that class; the ``all`` image contains every class.

Example (PowerShell):
    python tools/predict_seg_by_class.py `
        --model runs/segment/train/weights/best.pt `
        --source C:/data/images `
        --output runs/segment_by_class
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Allow direct execution from a source checkout even when the package is not installed.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Save one YOLO segmentation result per class and one result containing all classes."
    )
    parser.add_argument("--model", required=True, type=Path, help="Path to a trained YOLO segmentation model (.pt).")
    parser.add_argument("--source", required=True, help="Image, image directory, glob, URL, or other YOLO source.")
    parser.add_argument("--output", type=Path, default=Path("runs/segment_by_class"), help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--class-conf", nargs="+", default=[], help="Per-class thresholds, e.g. D=0.08 R=0.20.")
    parser.add_argument("--save-metrics", action="store_true", help="Save prediction counts and mask areas as JSON.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--device", default=None, help="Inference device, e.g. 0, 0,1, or cpu.")
    parser.add_argument("--alpha", type=float, default=0.35, help="Mask fill opacity (0 to 1).")
    parser.add_argument("--line-width", type=int, default=2, help="Contour and box line width.")
    parser.add_argument("--show-boxes", action="store_true", help="Also draw bounding boxes (disabled by default).")
    score_display = parser.add_mutually_exclusive_group()
    score_display.add_argument("--hide-labels", dest="hide_labels", action="store_true", help="Hide confidence scores (default).")
    score_display.add_argument("--show-scores", dest="hide_labels", action="store_false", help="Draw confidence scores.")
    parser.set_defaults(hide_labels=True)
    parser.add_argument("--no-fill", action="store_true", help="Draw mask contours without transparent mask fill.")
    parser.add_argument("--retina-masks", action="store_true", help="Use masks at original image resolution.")
    return parser.parse_args()


def normalize_names(raw_names: Any) -> dict[int, str]:
    """Convert model class names to a sorted integer-keyed dictionary."""
    if isinstance(raw_names, dict):
        return dict(sorted((int(class_id), str(name)) for class_id, name in raw_names.items()))
    if isinstance(raw_names, (list, tuple)):
        return {class_id: str(name) for class_id, name in enumerate(raw_names)}
    raise ValueError("The model does not contain valid class names.")


def safe_name(value: str) -> str:
    """Return a filename-safe class name while retaining Unicode letters."""
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", value.strip())
    value = value.rstrip(". ")
    return value or "unnamed"


def class_thresholds(entries: list[str], names: dict[int, str], default: float) -> dict[int, float]:
    """Resolve class names or IDs, preserving the global threshold for all other classes."""
    if not 0.0 <= default <= 1.0:
        raise ValueError("--conf must be between 0 and 1.")
    thresholds = dict.fromkeys(names, default)
    assigned = set()
    for entry in entries:
        key, separator, value = entry.partition("=")
        if not separator:
            raise ValueError(f"Expected CLASS=THRESHOLD, received {entry!r}.")
        matches = [cid for cid, name in names.items() if key.casefold() in {str(cid), name.casefold()}]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous class {key!r}; model names: {names}.")
        threshold = float(value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Class threshold must be between 0 and 1: {entry!r}.")
        class_id = matches[0]
        if class_id in assigned:
            raise ValueError(f"Duplicate threshold for class {names[class_id]!r}.")
        thresholds[class_id] = threshold
        assigned.add(class_id)
    return thresholds


def filter_class_scores(result: Any, thresholds: dict[int, float]) -> Any:
    """Filter the complete Results object so boxes and masks retain the same indices."""
    if result.boxes is None or len(result.boxes) == 0:
        return result
    scores = result.boxes.conf
    limits = scores.new_tensor([thresholds[int(cid)] for cid in result.boxes.cls.tolist()])
    return result[scores >= limits]


def prediction_metrics(
    masks: np.ndarray, classes: np.ndarray, names: dict[int, str], shape: tuple[int, int],
) -> dict:
    """Describe predictions without treating mask coverage or object counts as accuracy."""
    height, width = shape
    unions = {}
    metrics = {}
    for class_id, name in names.items():
        selected = masks[classes == class_id]
        union = selected.any(axis=0) if len(selected) else np.zeros(shape, dtype=bool)
        pixels = int(union.sum())
        unions[class_id] = union
        metrics[str(class_id)] = {
            "name": name, "count": len(selected), "union_pixels": pixels,
            "coverage_percent": round(100 * pixels / (height * width), 4),
        }
    overlaps = {
        f"{left}:{right}": int(np.logical_and(unions[left], unions[right]).sum())
        for left in names for right in names if left < right
        if np.any(unions[left] & unions[right])
    }
    return {"width": width, "height": height, "classes": metrics, "overlap_pixels": overlaps}


def class_color(class_id: int) -> tuple[int, int, int]:
    """Return a stable BGR color for a class ID."""
    palette_rgb = (
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
        (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
        (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
        (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    )
    red, green, blue = palette_rgb[class_id % len(palette_rgb)]
    return blue, green, red


def result_masks(result: Any) -> np.ndarray:
    """Convert predicted masks to boolean arrays in original-image size."""
    height, width = result.orig_img.shape[:2]
    if result.masks is None or result.masks.data is None:
        return np.zeros((0, height, width), dtype=bool)

    masks = result.masks.data.detach().cpu().numpy()
    resized_masks = []
    for mask in masks:
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        resized_masks.append(mask > 0.5)
    return np.stack(resized_masks) if resized_masks else np.zeros((0, height, width), dtype=bool)


def draw_predictions(
    image: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    selected_class: int | None,
    args: argparse.Namespace,
) -> np.ndarray:
    """Draw all predictions, or predictions belonging to one selected class."""
    output = image.copy()
    for mask, box, class_id, score in zip(masks, boxes, classes, scores):
        class_id = int(class_id)
        if selected_class is not None and class_id != selected_class:
            continue

        color = class_color(class_id)
        mask_u8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not args.no_fill and contours:
            overlay = output.copy()
            cv2.drawContours(overlay, contours, -1, color, cv2.FILLED)
            cv2.addWeighted(overlay, args.alpha, output, 1.0 - args.alpha, 0, dst=output)
        if contours and args.no_fill:
            cv2.drawContours(output, contours, -1, color, args.line_width, lineType=cv2.LINE_AA)

        x1, y1, x2, y2 = np.rint(box).astype(int)
        if args.show_boxes:
            cv2.rectangle(output, (x1, y1), (x2, y2), color, args.line_width, lineType=cv2.LINE_AA)
        if args.hide_labels:
            continue

        label = f"{float(score):.2f}"
        font_scale = max(0.45, min(image.shape[:2]) / 1000.0)
        thickness = max(1, args.line_width - 1)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_x = max(0, min(x1, output.shape[1] - text_width - 7))
        label_y = max(text_height + baseline + 5, y1)
        cv2.rectangle(
            output,
            (label_x, label_y - text_height - baseline - 5),
            (label_x + text_width + 6, label_y),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            output,
            label,
            (label_x + 3, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return output


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image reliably, including when its Windows path contains Unicode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} else ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise OSError(f"Failed to encode output image: {path}")
    encoded.tofile(path)


def save_result(result: Any, names: dict[int, str], output_dir: Path, index: int, args: argparse.Namespace) -> int:
    """Save N class-specific images and one combined image for a prediction result."""
    image = result.orig_img
    masks = result_masks(result)
    prediction_count = len(masks)

    if result.boxes is None or prediction_count == 0:
        boxes = np.zeros((0, 4), dtype=np.float32)
        classes = np.zeros(0, dtype=np.int32)
        scores = np.zeros(0, dtype=np.float32)
    else:
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(np.int32)
        scores = result.boxes.conf.detach().cpu().numpy()
        if not (len(boxes) == len(classes) == len(scores) == prediction_count):
            raise RuntimeError("The number of masks and boxes returned by the model does not match.")

    source_path = Path(str(result.path))
    stem = safe_name(source_path.stem or f"image_{index:06d}")
    image_dir = output_dir / f"{index:06d}_{stem}"

    for class_id, class_name in names.items():
        rendered = draw_predictions(image, masks, boxes, classes, scores, class_id, args)
        filename = f"{stem}__class_{class_id:03d}_{safe_name(class_name)}.jpg"
        write_image(image_dir / filename, rendered)

    combined = draw_predictions(image, masks, boxes, classes, scores, None, args)
    write_image(image_dir / f"{stem}__all.jpg", combined)
    if getattr(args, "save_metrics", False):
        metrics = prediction_metrics(masks, classes, names, image.shape[:2])
        metrics["source"] = str(result.path)
        (image_dir / "prediction_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    return prediction_count


def main() -> int:
    """Run inference and produce class-separated visualizations."""
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1.")
    if args.line_width < 1:
        raise ValueError("--line-width must be at least 1.")

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    if model.task != "segment":
        raise ValueError(f"Expected a segmentation model, but the loaded model task is {model.task!r}.")
    names = normalize_names(model.names)
    if not names:
        raise ValueError("The model has no classes.")
    thresholds = class_thresholds(args.class_conf, names, args.conf)

    args.output.mkdir(parents=True, exist_ok=True)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=min(thresholds.values()),
        iou=args.iou,
        device=args.device,
        retina_masks=args.retina_masks,
        stream=True,
        save=False,
        verbose=True,
    )

    image_count = 0
    total_predictions = 0
    for image_count, result in enumerate(results, start=1):
        if args.class_conf:
            result = filter_class_scores(result, thresholds)
        total_predictions += save_result(result, names, args.output, image_count - 1, args)

    if image_count == 0:
        raise FileNotFoundError(f"No images were found for source: {args.source}")

    outputs_per_image = len(names) + 1
    print(f"Processed {image_count} image(s), {total_predictions} prediction(s).")
    print(f"Saved {outputs_per_image} files per image ({len(names)} classes + all) to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
