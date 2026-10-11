"""Compare low-score, shadow-enhanced, and local-crop inference using predict-2.py defaults.

    python tools/diagnose_shadow_inference.py
    python tools/diagnose_shadow_inference.py --weights C:/Users/USER/Downloads/last0912.pt --versions S08_diagnostic_only

Selecting S08 runs the original image once and writes only S08. Without
--versions, all eight profiles and the original nine input views are retained.
--source accepts one or more image folders, including their subfolders.
--dry-run previews the image count without loading a model or creating results.
Results are grouped by the source dataset split (train/val/test/other).

ROIs below apply to CBHU0708054-A. Override them with --left-roi/--right-roi and
--left-context/--right-context for other images. Coordinates are normalized xyxy.
Predictions are mapped back without dilation or forced mask unions. Existing
baseline predictions are preserved; only same-class duplicate additions are removed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import runpy
import re
import sys

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from tools.predict_seg_by_class import draw_predictions, normalize_names, prediction_metrics, safe_name, write_image
from tools.predict_shadow import Detection, collect_images, enhance_shadow, extract_detections

BASE = runpy.run_path(str(SCRIPT_DIR / "predict-2.py"))
# Keep the right inspection region below the stepped shadow edge; bright pixels
# above that edge must not count as evidence that shadow detections improved.
REGIONS = {"left": (0.03, 0.20, 0.54, 0.90), "right": (0.65, 0.70, 0.805, 0.89)}
CONTEXTS = {"left": (0.00, 0.10, 0.59, 0.97), "right": (0.55, 0.25, 0.90, 0.97)}
VERSION_NAMES = (
    "S01_baseline", "S02_gamma", "S03_clahe", "S04_crops", "S05_crop_clahe",
    "S06_crop_mix", "S07_crop_recall", "S08_diagnostic_only",
)


def infer_dataset_split(source: Path) -> str:
    """Use the closest named dataset directory, including 'val 驗證'."""
    aliases = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
    for part in reversed(source.parent.parts):
        match = re.fullmatch(r"(train|val|valid|validation|test)(?:[\s_-].*|[^\x00-\x7f].*)?", part.casefold())
        if match:
            return aliases[match.group(1)]
    return "other"


def pixel_bounds(bounds: tuple, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    left, top, right, bottom = bounds
    if not 0 <= left < right <= 1 or not 0 <= top < bottom <= 1:
        raise ValueError(f"Invalid normalized xyxy: {bounds}")
    result = (round(left * width), round(top * height), round(right * width), round(bottom * height))
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"Region is smaller than one pixel: {bounds}")
    return result


def map_detection(detection: Detection, offset: tuple[int, int], shape: tuple[int, int]) -> Detection:
    """Translate crop coordinates, placing the original binary mask on a full-size canvas."""
    x, y = offset
    height, width = detection.mask.shape
    if x < 0 or y < 0 or x + width > shape[1] or y + height > shape[0]:
        raise ValueError("Crop does not fit within the original image")
    box = detection.box.copy()
    box[:4] += np.array([x, y, x, y])
    if (x, y) == (0, 0) and detection.mask.shape == shape:
        return Detection(box, detection.mask, detection.view)
    mask = np.zeros(shape, dtype=bool)
    mask[y:y + height, x:x + width] = detection.mask
    return Detection(box, mask, detection.view)


def in_region(detection: Detection, bounds: tuple[int, int, int, int]) -> bool:
    """Require >=5 mask pixels and at least half the mask inside the inspected region."""
    x1, y1, x2, y2 = bounds
    inside = int(detection.mask[y1:y2, x1:x2].sum())
    return inside >= 5 and inside >= 0.5 * int(detection.mask.sum())


def append_candidates(baseline: list[Detection], candidates: list[Detection], iou: float) -> list[Detection]:
    """Never replace a baseline mask; deduplicate same-class additions with mask IoU."""
    kept = list(baseline)
    areas = [int(d.mask.sum()) for d in kept]
    for detection in sorted(candidates, key=lambda d: -float(d.box[4])):
        area = int(detection.mask.sum())
        duplicate = False
        for other, other_area in zip(kept, areas):
            if detection.box[5] != other.box[5]:
                continue
            intersection = int(np.logical_and(detection.mask, other.mask).sum())
            union = area + other_area - intersection
            if union and intersection / union > iou:
                duplicate = True
                break
        if not duplicate:
            kept.append(detection)
            areas.append(area)
    return kept


def arrays(detections: list[Detection], shape: tuple[int, int]):
    if not detections:
        return (np.zeros((0, *shape), dtype=bool), np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0))
    boxes = np.stack([d.box for d in detections])
    return np.stack([d.mask for d in detections]), boxes[:, :4], boxes[:, 5].astype(int), boxes[:, 4]


def regional_metrics(detections: list[Detection], bounds: tuple, names: dict) -> dict:
    x1, y1, x2, y2 = bounds
    output = {}
    for class_id, name in names.items():
        if name not in {"D", "R"}:
            continue
        selected = [d for d in detections if int(d.box[5]) == class_id and in_region(d, bounds)]
        union = np.zeros((y2 - y1, x2 - x1), dtype=bool)
        for detection in selected:
            union |= detection.mask[y1:y2, x1:x2]
        output[name] = {"count": len(selected), "roi_pixels": int(union.sum()),
                        "max_conf": max((float(d.box[4]) for d in selected), default=0.0)}
    return output


def selected_profiles(args, regions):
    crops_original = [f"{region}_original" for region in regions]
    crops_clahe = [f"{region}_clahe" for region in regions]
    profiles = [
        ("S01_baseline", "原圖基準", [], args.conf, args.conf),
        ("S02_gamma", "原圖＋暗部 Gamma", ["full_gamma"], args.conf, args.conf),
        ("S03_clahe", "原圖＋暗部 CLAHE", ["full_clahe"], args.conf, args.conf),
        ("S04_crops", "原圖＋左右局部裁切", crops_original, args.conf, args.conf),
        ("S05_crop_clahe", "原圖＋裁切 CLAHE", crops_clahe, args.conf, args.conf),
        ("S06_crop_mix", "裁切混合 D:0.10 R:0.15", crops_original + crops_clahe, 0.10, 0.15),
        ("S07_crop_recall", "裁切混合 D:0.08 R:0.10", crops_original + crops_clahe, 0.08, 0.10),
        ("S08_diagnostic_only", "原圖低分檢查，非推薦設定", ["full_original"], args.diagnostic_conf, args.diagnostic_conf),
    ]
    versions = getattr(args, "versions", None)
    return [p for p in profiles if versions is None or p[0] in versions]


def evaluate(model, image: np.ndarray, args, regions: dict, contexts: dict, required_views=None):
    """Evaluate each input once; keep low-score provenance separate from usable additions."""
    names = normalize_names(model.names)
    shape = image.shape[:2]
    views = [("full_original", image, (0, 0), None)]
    for enhancement in ("gamma", "clahe"):
        label = f"full_{enhancement}"
        if required_views is None or label in required_views:
            views.append((label, enhance_shadow(image, enhancement, args), (0, 0), None))
    for region, (x1, y1, x2, y2) in contexts.items():
        crop = image[y1:y2, x1:x2]
        for enhancement in ("original", "gamma", "clahe"):
            label = f"{region}_{enhancement}"
            if required_views is None or label in required_views:
                views.append((label, enhance_shadow(crop, enhancement, args), (x1, y1), region))
    kept_views, diagnostic_rows = {}, []
    for label, input_image, offset, target_region in views:
        result = model.predict(
            source=input_image, imgsz=args.imgsz, conf=args.diagnostic_conf, iou=args.iou,
            device=args.device, retina_masks=True, agnostic_nms=False, max_det=1000,
            save=False, verbose=False,
        )[0]
        detections = [map_detection(d, offset, shape) for d in extract_detections(result, label, input_image.shape[:2])]
        regional = {name: regional_metrics(detections, bounds, names) for name, bounds in regions.items()}
        diagnostic_rows.append({"view": label, "regions": regional, "candidate_count": len(detections),
                                "at_max_det": len(result.boxes) >= 1000})
        print(f"  {label}: " + json.dumps(regional, ensure_ascii=True), flush=True)
        # Low-score masks are retained only for the original-input diagnostic control.
        floor = args.diagnostic_conf if label == "full_original" else min(args.conf, 0.08)
        relevant = [d for d in detections if float(d.box[4]) >= floor]
        if target_region:
            relevant = [d for d in relevant if in_region(d, regions[target_region])]
        kept_views[label] = relevant
        write_image(args.image_output / "inputs" / f"{label}.png", input_image)
        del result, detections
    return kept_views, diagnostic_rows


def process_image(model, source: Path, output: Path, args) -> tuple[list[dict], list[dict]]:
    image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read {source}")
    shape = image.shape[:2]
    names = normalize_names(model.names)
    regions = {name: pixel_bounds(getattr(args, f"{name}_roi"), shape) for name in args.regions}
    contexts = {name: pixel_bounds(getattr(args, f"{name}_context"), shape) for name in args.regions}
    args.image_output = output
    profiles = selected_profiles(args, regions)
    required_views = None
    if getattr(args, "versions", None) is not None:
        required_views = {"full_original"}
        required_views.update(view for profile in profiles for view in profile[2])
    views, diagnostics = evaluate(model, image, args, regions, contexts, required_views)
    baseline = [d for d in views["full_original"] if float(d.box[4]) >= args.conf]
    drawing = argparse.Namespace(no_fill=False, alpha=0.35, line_width=2, show_boxes=False, hide_labels=True)
    records, summary_rows = [], []
    for version, title, selected_views, dent_conf, rust_conf in profiles:
        candidates = [
            d for view in selected_views for d in views[view]
            if names[int(d.box[5])] in {"D", "R"}
            and float(d.box[4]) >= (dent_conf if names[int(d.box[5])] == "D" else rust_conf)
            and any(in_region(d, bounds) for bounds in regions.values())
        ]
        merged = append_candidates(baseline, candidates, args.merge_iou)
        masks, boxes, classes, scores = arrays(merged, shape)
        version_dir = output / version
        stem = safe_name(source.stem)
        for class_id, name in names.items():
            plotted = draw_predictions(image, masks, boxes, classes, scores, class_id, drawing)
            write_image(version_dir / f"{stem}__class_{class_id:03d}_{safe_name(name)}.jpg", plotted)
        plotted = draw_predictions(image, masks, boxes, classes, scores, None, drawing)
        image_path = version_dir / f"{stem}__all.jpg"
        write_image(image_path, plotted)
        stats = prediction_metrics(masks, classes, names, shape)
        stats.update(source=str(source), regions={name: regional_metrics(merged, bounds, names) for name, bounds in regions.items()})
        (version_dir / "prediction_metrics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        provenance = {
            "version": version, "title": title, "baseline_conf": args.conf,
            "D_addition_conf": dent_conf, "R_addition_conf": rust_conf, "merge_iou": args.merge_iou,
            "regions_xyxy": regions, "contexts_xyxy": contexts, "selected_views": selected_views,
            "detections": [{"class": names[int(d.box[5])], "score": float(d.box[4]), "view": d.view,
                            "box_xyxy": d.box[:4].tolist(), "mask_pixels": int(d.mask.sum())} for d in merged],
        }
        (version_dir / "predictions.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
        if getattr(args, "save_masks", False):
            # Lossless original-coordinate masks are required for pixel evaluation.
            np.savez_compressed(
                version_dir / "prediction_masks.npz", schema=np.asarray("instance_bool_packbits_v1"),
                packed_masks=np.packbits(masks, axis=-1), width=shape[1], height=shape[0],
                class_names=np.asarray([names[int(class_id)] for class_id in classes], dtype=str),
                class_ids=classes, scores=scores, source=np.asarray(str(source)), version=np.asarray(version),
            )
        count_labels = []
        for name, metric in stats["regions"].items():
            prefix = "左" if name == "left" else "右"
            count_labels.append(f"{prefix} D{metric.get('D', {}).get('count', 0)}/R{metric.get('R', {}).get('count', 0)}")
            for cls in ("D", "R"):
                summary_rows.append({"source": str(source), "version": version, "region": name,
                                     "class": cls, **metric.get(cls, {"count": 0, "roi_pixels": 0, "max_conf": 0})})
        records.append({"source": str(source), "image": image_path, "metrics": stats,
                        "caption": f"{version[:3]}｜{title}\n區內預測：" + "　".join(count_labels)})
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
    return records, summary_rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", nargs="+", default=BASE["WEIGHT_SOURCES"])
    parser.add_argument("--source", nargs="+", default=BASE["DATASET_SOURCES"],
                        help="One or more images or image folders; folders include subfolders")
    parser.add_argument("--output", default=str(Path(BASE["OUTPUT_DIR"]) / "shadow_diagnostics"))
    parser.add_argument("--imgsz", type=int, default=BASE["IMAGE_SIZE"])
    parser.add_argument("--conf", type=float, default=BASE["CONFIDENCE"])
    parser.add_argument("--iou", type=float, default=BASE["IOU_THRESHOLD"])
    parser.add_argument("--device", default=BASE["DEVICE"])
    parser.add_argument("--versions", nargs="+", choices=VERSION_NAMES,
                        help="Run only these profiles and their required input views; default: all eight")
    parser.add_argument("--dry-run", action="store_true", help="Preview selected images and expected inference count without running a model")
    parser.add_argument("--continue-on-error", action="store_true", help="Record failed images and continue processing the dataset")
    parser.add_argument("--save-masks", action="store_true", help="Save lossless prediction_masks.npz for true pixel-level evaluation")
    parser.add_argument("--diagnostic-conf", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.55)
    parser.add_argument("--clahe-clip", type=float, default=3.0)
    parser.add_argument("--shadow-threshold", type=int, default=180)
    parser.add_argument("--merge-iou", type=float, default=0.60)
    parser.add_argument("--regions", nargs="+", choices=("left", "right"), default=["left", "right"])
    for region in REGIONS:
        parser.add_argument(f"--{region}-roi", nargs=4, type=float, default=REGIONS[region])
        parser.add_argument(f"--{region}-context", nargs=4, type=float, default=CONTEXTS[region])
    args = parser.parse_args(argv)
    if not 0 < args.diagnostic_conf <= min(args.conf, 0.08) or not 0 <= args.conf <= 1:
        parser.error("diagnostic-conf must be positive and <= min(conf, 0.08); conf must be <=1")
    if not 0 < args.gamma <= 1 or not 0 < args.clahe_clip < float("inf") or not 1 <= args.shadow_threshold <= 255:
        parser.error("Invalid shadow enhancement settings")
    if args.imgsz <= 0 or not 0 <= args.iou <= 1 or not 0 <= args.merge_iou <= 1:
        parser.error("Invalid inference size or IoU")
    for region in args.regions:
        roi = pixel_bounds(getattr(args, f"{region}_roi"), (10000, 10000))
        context = pixel_bounds(getattr(args, f"{region}_context"), (10000, 10000))
        if not (context[0] <= roi[0] and context[1] <= roi[1] and context[2] >= roi[2] and context[3] >= roi[3]):
            parser.error(f"{region} context must contain the inspected ROI")
    weights = [BASE["resolve_path"](value) for value in args.weights]
    if not weights or any(not weight.is_file() for weight in weights):
        raise FileNotFoundError("Missing model weights")
    sources = collect_images([BASE["resolve_path"](value) for value in args.source])
    if not sources:
        raise ValueError("No source images configured")
    dataset_splits = [infer_dataset_split(source) for source in sources]
    split_counts = Counter(dataset_splits)
    output_names = [Path(split) / f"{index:02d}_{safe_name(source.stem)}"
                    for index, (source, split) in enumerate(zip(sources, dataset_splits), 1)]
    profiles = selected_profiles(args, {name: None for name in args.regions})
    if args.versions is None:
        views_per_image = 3 + 3 * len(args.regions)
    else:
        views_per_image = len({"full_original"} | {view for profile in profiles for view in profile[2]})
    print(f"Models: {len(weights)}; images: {len(sources)}; versions: {len(profiles)}; "
          f"model calls per image: {views_per_image}; total model calls: {len(weights) * len(sources) * views_per_image}", flush=True)
    print("Dataset splits: " + "; ".join(f"{split}: {split_counts[split]}" for split in ("train", "val", "test", "other")
                                        if split_counts[split]), flush=True)
    if args.dry_run:
        for source, relative_output in zip(sources[:10], output_names[:10]):
            print(f"  {source} -> {relative_output}", flush=True)
        if len(sources) > 10:
            print(f"  ... {len(sources) - 10} more image(s)", flush=True)
        return 0
    output = BASE["resolve_path"](args.output) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    (output / "settings.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "source_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        manifest = csv.DictWriter(handle, fieldnames=("image_index", "source", "dataset_split", "output_name"))
        manifest.writeheader()
        manifest.writerows({"image_index": index, "source": str(source),
                             "dataset_split": split, "output_name": relative_output.as_posix()}
                            for index, (source, split, relative_output) in enumerate(zip(sources, dataset_splits, output_names), 1))
    from ultralytics import YOLO
    from tools.compare_parameter_sweep import draw_figure
    from matplotlib import font_manager, pyplot as plt

    font = Path("C:/Windows/Fonts/msjh.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    failures = 0
    print(f"Output: {output}", flush=True)
    with (output / "region_summary.csv").open("w", encoding="utf-8-sig", newline="") as regional_handle, \
            (output / "image_summary.csv").open("w", encoding="utf-8-sig", newline="") as image_handle:
        regional_writer = csv.DictWriter(regional_handle, fieldnames=("weight", "source", "dataset_split", "version", "region", "class", "count", "roi_pixels", "max_conf"))
        image_writer = csv.DictWriter(image_handle, fieldnames=("weight", "source", "dataset_split", "output", "status", "error"))
        regional_writer.writeheader()
        image_writer.writeheader()
        for model_index, weight in enumerate(weights, 1):
            model = YOLO(str(weight))
            if model.task != "segment":
                raise ValueError(f"Expected segmentation weights: {weight}")
            for source_index, (source, split, relative_output) in enumerate(zip(sources, dataset_splits, output_names), 1):
                folder = output / f"{model_index:02d}_{safe_name(weight.stem)}" / relative_output
                print(f"[{model_index}/{len(weights)}][{source_index}/{len(sources)}] {source}", flush=True)
                entry = {"weight": str(weight), "source": str(source), "dataset_split": split, "output": str(folder)}
                try:
                    records, rows = process_image(model, source, folder, args)
                    regional_writer.writerows({"weight": str(weight), **row, "dataset_split": split} for row in rows)
                    if len(records) > 1:
                        comparison_dir = folder / "comparisons"
                        comparison_dir.mkdir(parents=True, exist_ok=True)
                        draw_figure(records, comparison_dir / "all_variants.png", "all")
                        for region in args.regions:
                            for kind in ("all", "D", "R"):
                                draw_figure(records, comparison_dir / f"{region}_{kind}.png", kind,
                                            tuple(getattr(args, f"{region}_roi")),
                                            crop_label="左側暗部放大" if region == "left" else "右側陰影內部放大")
                except Exception as error:
                    failures += 1
                    image_writer.writerow({**entry, "status": "failed", "error": f"{type(error).__name__}: {error}"})
                    image_handle.flush()
                    regional_handle.flush()
                    print(f"  FAILED: {type(error).__name__}: {error}", flush=True)
                    if not args.continue_on_error:
                        raise
                else:
                    image_writer.writerow({**entry, "status": "ok", "error": ""})
                    image_handle.flush()
                    regional_handle.flush()
            del model
    print(f"Completed {len(profiles)} variant(s) per image; failed images: {failures}. Results: {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
