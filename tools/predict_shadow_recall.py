"""Inspect missed dents near MSC and damage in the lower-right shadow.

    python tools/predict_shadow_recall.py

Uses predict-2.py models/sources and baseline settings. Adds contextual crops and
overlapping tiles at a separate inference size, preserving existing shadow masks.
Six versions are saved for inspection; more predictions do not establish recall.
Default regions are specific to CBHU0708054-A; override them for other images.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import runpy
import sys

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from tools import predict_shadow as shadow
from tools.diagnose_shadow_inference import append_candidates, arrays, in_region, map_detection, pixel_bounds, regional_metrics
from tools.predict_seg_by_class import normalize_names, prediction_metrics, safe_name, write_image

BASE = runpy.run_path(str(SCRIPT_DIR / "predict-2.py"))
REGIONS = {"left": (0.03, 0.20, 0.54, 0.90), "right": (0.64, 0.58, 0.85, 0.94)}
CONTEXTS = {"left": (0.00, 0.10, 0.59, 0.97), "right": (0.55, 0.25, 0.86, 0.98)}


def tile_boxes(bounds: tuple[int, int, int, int], fraction: float, overlap: float) -> list[tuple]:
    """Cover the complete context, including its far edges, with overlapping tiles."""
    if not 0 < fraction < 1 or not 0 <= overlap < 1:
        raise ValueError("tile fraction must be in (0,1); overlap in [0,1)")
    x1, y1, x2, y2 = bounds
    width, height = x2 - x1, y2 - y1
    if min(width, height) < 2:
        return []
    tile_width, tile_height = max(2, round(width * fraction)), max(2, round(height * fraction))

    def starts(length, size):
        step = max(1, round(size * (1 - overlap)))
        return sorted(set([*range(0, length - size + 1, step), length - size]))

    return [(x1 + x, y1 + y, x1 + x + tile_width, y1 + y + tile_height)
            for y in starts(height, tile_height) for x in starts(width, tile_width)]


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", nargs="+", default=BASE["WEIGHT_SOURCES"])
    parser.add_argument("--source", nargs="+", default=BASE["DATASET_SOURCES"])
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR.parent / "runs/segment_shadow/recall_tests")
    parser.add_argument("--annotation", type=Path, help="Optional XML for box evaluation after all six versions finish")
    parser.add_argument("--imgsz", type=int, default=BASE["IMAGE_SIZE"])
    parser.add_argument("--crop-imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=BASE["CONFIDENCE"])
    parser.add_argument("--iou", type=float, default=BASE["IOU_THRESHOLD"])
    parser.add_argument("--device", default=BASE["DEVICE"])
    parser.add_argument("--tile-fraction", type=float, default=0.60)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--diagnostic-conf", type=float, default=0.01)
    exclusions = parser.add_mutually_exclusive_group()
    exclusions.add_argument("--exclude-regions", type=Path, default=shadow.EXCLUSION_REGIONS)
    exclusions.add_argument("--no-exclusions", dest="exclude_regions", action="store_const", const=None)
    for name in REGIONS:
        parser.add_argument(f"--{name}-roi", nargs=4, type=float, default=REGIONS[name])
        parser.add_argument(f"--{name}-context", nargs=4, type=float, default=CONTEXTS[name])
    args = parser.parse_args(argv)
    if args.crop_imgsz < 32 or args.crop_imgsz % 32:
        parser.error("crop-imgsz must be a positive multiple of 32")
    if not 0 < args.diagnostic_conf <= min(args.conf, 0.05):
        parser.error("diagnostic-conf must be positive and <= min(conf, 0.05)")
    # Restrict tile settings to practical values rather than allowing accidental thousands of crops.
    if not 0.4 <= args.tile_fraction <= 0.8 or not 0 <= args.overlap <= 0.5:
        parser.error("tile-fraction must be in [0.4,0.8]; overlap in [0,0.5]")
    for name in REGIONS:
        roi = pixel_bounds(tuple(getattr(args, f"{name}_roi")), (10000, 10000))
        context = pixel_bounds(tuple(getattr(args, f"{name}_context")), (10000, 10000))
        if not (context[0] <= roi[0] < roi[2] <= context[2] and context[1] <= roi[1] < roi[3] <= context[3]):
            parser.error(f"{name} context must contain the ROI")
    common = ["--imgsz", str(args.imgsz), "--conf", str(args.conf), "--iou", str(args.iou),
              "--device", args.device, "--max-det", "300"]
    common += ["--exclude-regions", str(args.exclude_regions)] if args.exclude_regions else ["--no-exclusions"]
    settings = shadow.parse_args(common)
    # Preserve the shadow script's enhancement and drawing defaults, recording them with this run.
    for key, value in vars(settings).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def predict_view(model, image, label, args, size, conf, max_det):
    result = model.predict(source=image, imgsz=size, conf=conf, iou=args.iou,
                           device=args.device, retina_masks=True, agnostic_nms=False,
                           max_det=max_det, save=False, verbose=False)[0]
    detections = shadow.extract_detections(result, label, image.shape[:2])
    capped = result.boxes is not None and len(result.boxes) >= max_det
    return detections, capped


def select_additions(candidates, names, dent_conf, rust_conf):
    """Class thresholds only gate actual model predictions; no masks are invented."""
    thresholds = {"D": dent_conf, "R": rust_conf}
    return [d for d in candidates if names[int(d.box[5])] in thresholds
            and float(d.box[4]) >= thresholds[names[int(d.box[5])]]]


def process_image(model, source: Path, output: Path, args) -> list[dict]:
    image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {source}")
    output.mkdir(parents=True, exist_ok=True)
    shape = image.shape[:2]
    names = normalize_names(model.names)
    excluded, matched = shadow.exclusion_masks(source, shape, names, args.exclusion_rules)
    regions = {name: pixel_bounds(tuple(getattr(args, f"{name}_roi")), shape) for name in REGIONS}
    contexts = {name: pixel_bounds(tuple(getattr(args, f"{name}_context")), shape) for name in REGIONS}
    views, diagnostics = {}, []
    for view in args.views:
        enhanced = shadow.enhance_shadow(image, view, args)
        detections, capped = predict_view(model, enhanced, view, args, args.imgsz, args.conf, args.max_det)
        views[view] = shadow.exclude_background(detections, excluded)
        diagnostics.append({"view": view, "kind": "full", "imgsz": args.imgsz, "conf": args.conf,
                            "at_max_det": capped, "count": len(views[view])})
    baseline = shadow.merge_detections([d for values in views.values() for d in values], args.merge_iou)
    candidates = {"context": [], "tile": []}
    for name, bounds in contexts.items():
        boxes = [("context", bounds), *[("tile", box) for box in tile_boxes(bounds, args.tile_fraction, args.overlap)]]
        for index, (kind, (x1, y1, x2, y2)) in enumerate(boxes):
            crop = image[y1:y2, x1:x2]
            for view in args.views:
                label = f"{name}_{kind}{index:02d}_{view}"
                enhanced = shadow.enhance_shadow(crop, view, args)
                raw, capped = predict_view(model, enhanced, label, args, args.crop_imgsz, args.diagnostic_conf, 1000)
                diagnostics.append({
                    "view": label, "kind": kind, "region": name, "crop_xyxy": [x1, y1, x2, y2],
                    "imgsz": args.crop_imgsz, "conf": args.diagnostic_conf, "at_max_det": capped,
                    "raw_count": len(raw),
                })
                mapped = shadow.exclude_background([map_detection(d, (x1, y1), shape) for d in raw], excluded)
                eligible = [d for d in mapped if in_region(d, regions[name])]
                diagnostics[-1]["region_metrics"] = regional_metrics(eligible, regions[name], names)
                # Retain the lowest threshold used in the comparison; low-score diagnostics remain statistics only.
                additions = select_additions(eligible, names, min(args.conf, 0.05), min(args.conf, 0.08))
                candidates[kind].extend(additions)
                print(f"  {label}: ROI candidates={len(eligible)}, saved={len(additions)}", flush=True)
                del raw, mapped, eligible

    profiles = [
        ("R01_original", "原圖基準", views["original"], [], args.conf, args.conf),
        ("R02_previous_shadow", "既有三視圖基準", baseline, [], args.conf, args.conf),
        ("R03_context", "加入局部背景裁切", baseline, ["context"], args.conf, args.conf),
        ("R04_tiles", "加入重疊裁切，同門檻", baseline, ["context", "tile"], args.conf, args.conf),
        ("R05_recall", "局部補檢 D0.08 R0.10", baseline, ["context", "tile"], 0.08, 0.10),
        ("R06_low_threshold", "低門檻檢查 D0.05 R0.08", baseline, ["context", "tile"], 0.05, 0.08),
    ]
    records, summary = [], []
    for version, title, existing, kinds, dent_conf, rust_conf in profiles:
        additions = select_additions([d for kind in kinds for d in candidates[kind]], names, dent_conf, rust_conf)
        detections = append_candidates(existing, additions, args.merge_iou)
        added = detections[len(existing):]
        folder = output / version
        stem = safe_name(source.stem)
        image_path = folder / f"{stem}__all.jpg"
        write_image(image_path, shadow.render(image, detections, names, args))
        write_image(folder / f"{stem}__additions.jpg", shadow.render(image, added, names, args))
        for class_id, class_name in names.items():
            selected = [d for d in detections if int(d.box[5]) == class_id]
            write_image(folder / f"{stem}__class_{class_id:03d}_{safe_name(class_name)}.jpg",
                        shadow.render(image, selected, names, args))
        masks, _, classes, _ = arrays(detections, shape)
        metrics = prediction_metrics(masks, classes, names, shape)
        stats = {name: regional_metrics(detections, bounds, names) for name, bounds in regions.items()}
        report = {
            "source": str(source), "version": version, "baseline_count": len(existing), "added_count": len(added),
            "merged_count": len(detections), "metrics": metrics, "regions": stats, "excluded_regions": matched,
            "D_addition_conf": dent_conf, "R_addition_conf": rust_conf,
            "regions_xyxy": regions, "contexts_xyxy": contexts, "selected_kinds": kinds,
            "detections": [{"class_id": int(d.box[5]), "class_name": names[int(d.box[5])],
                            "confidence": float(d.box[4]), "view": d.view, "box_xyxy": d.box[:4].tolist(),
                            "mask_pixels": int(d.mask.sum()), "is_addition": index >= len(existing)}
                           for index, d in enumerate(detections)],
        }
        (folder / "predictions.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        caption = f"{version[:3]} {title}\n全圖 {args.imgsz} / 裁切 {args.crop_imgsz} / 新增 {len(added)}"
        records.append({"source": str(source), "image": image_path, "metrics": metrics, "caption": caption})
        for name, region in stats.items():
            for class_name, values in region.items():
                summary.append({"version": version, "region": name, "class": class_name,
                                "added_total": len(added), **values})
        print(f"  {version}: total={len(detections)}, added={len(added)}", flush=True)
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "region_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    from tools.compare_parameter_sweep import draw_figure, font_manager, plt

    font = Path("C:/Windows/Fonts/msjh.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    comparison = output / "comparisons"
    comparison.mkdir(exist_ok=True)
    draw_figure(records, comparison / "all_variants.png", "all")
    for name in REGIONS:
        for kind in ("all", "D", "R"):
            draw_figure(records, comparison / f"{name}_{kind}.png", kind,
                        tuple(getattr(args, f"{name}_roi")),
                        crop_label="MSC 附近凹陷區" if name == "left" else "右下陰影內部")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    weights = list(dict.fromkeys(BASE["resolve_path"](p) for p in args.weights))
    if not weights or any(not p.is_file() for p in weights):
        raise FileNotFoundError("Missing model weights")
    sources = shadow.collect_images([BASE["resolve_path"](p) for p in args.source])
    if not sources:
        raise ValueError("No source images configured")
    annotation = None
    if args.annotation:
        from tools.evaluate_shadow_annotations import read_annotation

        annotation = read_annotation(BASE["resolve_path"](args.annotation))
        if not any(source.name.casefold() == annotation["filename"].casefold() for source in sources):
            raise ValueError("XML filename does not match any source image")
    import torch
    from ultralytics import YOLO

    if args.device == "auto":
        args.device = "0" if torch.cuda.is_available() else "cpu"
    output = BASE["resolve_path"](args.output) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    (output / "settings.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Output: {output}", flush=True)
    for model_index, weight in enumerate(weights, 1):
        model = YOLO(str(weight))
        if model.task != "segment":
            raise ValueError(f"Expected segmentation model: {weight}")
        for image_index, source in enumerate(sources, 1):
            folder = output / f"{model_index:02d}_{safe_name(weight.stem)}" / f"{image_index:02d}_{safe_name(source.stem)}"
            print(f"Inspecting {source}", flush=True)
            process_image(model, source, folder, args)
        del model
    print(f"Completed 6 versions per image: {output}", flush=True)
    if annotation:
        from tools.evaluate_shadow_annotations import evaluate_runs

        evaluate_runs(annotation, [output], output / "annotation_evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
