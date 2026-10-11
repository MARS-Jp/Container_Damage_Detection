"""Run reproducible shadow-recovery experiments, evaluated against a VOC XML.

Defaults are read from predict-2.py. XML is read only for evaluation and label
auditing; uniform tiles and predictions are generated without annotation hints.
All result photographs contain only segmentation fill on the original image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import runpy
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from tools.damage_segmentation_predictor import DamageSegmentationPredictor
from tools.diagnose_shadow_inference import map_detection, pixel_bounds
from tools.evaluate_damage_xml import box_iou, evaluate_boxes, read_voc
from tools.predict_seg_by_class import normalize_names, prediction_metrics, safe_name, write_image
from tools.predict_shadow import (
    Detection,
    enhance_shadow,
    exclude_background,
    exclusion_masks,
    extract_detections,
    load_exclusion_rules,
    render,
)

BASE = runpy.run_path(str(SCRIPT_DIR / "predict-2.py"))
REGIONS = {"left": (0.03, 0.20, 0.54, 0.90), "right": (0.64, 0.70, 0.85, 0.89)}


def tile_bounds(shape: tuple, size: int, overlap: float) -> list[tuple]:
    """Cover every pixel, including image edges, with ordinary overlapping tiles."""
    if size < 32 or not 0 <= overlap < 1:
        raise ValueError("Tile size must be >=32; overlap must be in [0,1)")
    height, width = shape
    step = max(1, round(size * (1 - overlap)))

    def starts(length):
        last = max(0, length - size)
        return sorted({*range(0, last + 1, step), last})

    return [(x, y, min(width, x + size), min(height, y + size)) for y in starts(height) for x in starts(width)]


def auxiliary_detections(maps: np.ndarray, threshold: float, view: str, min_area=12) -> list[Detection]:
    """Use connected pixels from a trained auxiliary head without expanding masks."""
    result = []
    for class_id in (2, 5):
        probability = maps[class_id]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (probability >= threshold).astype(np.uint8), connectivity=8
        )
        for component in range(1, count):
            x, y, w, h, area = stats[component]
            if area < min_area:
                continue
            mask = labels == component
            score = float(probability[mask].mean())
            result.append(Detection(np.array([x, y, x + w, y + h, score, class_id]), mask, view))
    return result


def deduplicate(baseline: list[Detection], additions: list[Detection]) -> list[Detection]:
    """Preserve baseline predictions, removing same-class duplicates among additions."""
    kept = list(baseline)
    for candidate in sorted(additions, key=lambda d: -float(d.box[4])):
        duplicate = False
        for other in kept:
            if int(candidate.box[5]) != int(other.box[5]):
                continue
            if box_iou(candidate.box[:4], other.box[:4]) >= 0.5:
                duplicate = True
                break
            intersection = int(np.count_nonzero(candidate.mask & other.mask))
            union = int(candidate.mask.sum()) + int(other.mask.sum()) - intersection
            if union and intersection / union >= 0.6:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def serialize(detections, names):
    return [
        {
            "class": names[int(d.box[5])],
            "score": float(d.box[4]),
            "box": d.box[:4].tolist(),
            "view": d.view,
            "mask_pixels": int(d.mask.sum()),
            "score_type": "mean_auxiliary_pixel_probability" if "aux" in d.view else "instance_confidence",
        }
        for d in detections
    ]


def audit_labels(labels: Path | None, truth: list, names: dict, shape: tuple, output: Path):
    if not labels:
        return
    height, width = shape
    existing = []
    for index, line in enumerate(labels.read_text(encoding="utf-8").splitlines()):
        fields = line.split()
        points = np.asarray([float(v) for v in fields[1:]]).reshape(-1, 2) * [width, height]
        if len(points) < 3:
            raise ValueError("Expected segmentation polygons, not YOLO detection rows")
        existing.append(
            {"index": index, "class": names[int(fields[0])], "box": np.r_[points.min(0), points.max(0)].tolist()}
        )
    differences = []
    for target in truth:
        same_class = [old for old in existing if old["class"] == target["class"]]
        overlap, old = max(
            ((box_iou(target["box"], old["box"]), old) for old in same_class), default=(0, None), key=lambda p: p[0]
        )
        differences.append(
            {
                **target,
                "existing_polygon_index": old["index"] if old else None,
                "max_box_iou_with_existing_polygon": overlap,
            }
        )
    audit = {
        "local_labels": str(labels),
        "local_counts": dict(Counter(o["class"] for o in existing)),
        "xml_counts": dict(Counter(o["class"] for o in truth)),
        "objects": differences,
        "caveat": "Local labels are a matched-image copy; checkpoint does not record their hash. "
        "XML contains boxes, not pixel boundaries; do not convert boxes to segmentation truth.",
    }
    (output / "annotation_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(labels, output / "existing_segmentation_labels.txt")


def infer_bank(model, image, args, excluded, size=None, enhancement="original", multi=True, auxiliary=False):
    shape = image.shape[:2]
    bounds = tile_bounds(shape, size, args.tile_overlap) if size else [(0, 0, shape[1], shape[0])]
    detections, auxiliary_banks = [], {"p3_050": [], "p3_030": [], "mean_050": []}
    label = f"{'tile' + str(size) if size else 'full'}_{enhancement}_{'multi' if multi else 'single'}"
    for index, (x1, y1, x2, y2) in enumerate(bounds):
        view = enhance_shadow(image[y1:y2, x1:x2], enhancement, args)
        DamageSegmentationPredictor.multi_label = multi
        DamageSegmentationPredictor.collect_auxiliary = auxiliary
        if model.predictor:
            model.predictor.multi_label = multi
            model.predictor.collect_auxiliary = auxiliary
        result = model.predict(
            view,
            predictor=DamageSegmentationPredictor,
            imgsz=args.imgsz,
            conf=0.03,
            iou=args.iou,
            device=args.device,
            retina_masks=True,
            max_det=1000,
            verbose=False,
            save=False,
        )[0]
        provenance = f"{label}_{index:02d}"
        local = extract_detections(result, provenance, view.shape[:2])
        detections.extend(exclude_background([map_detection(d, (x1, y1), shape) for d in local], excluded))
        if auxiliary:
            maps = model.predictor.auxiliary_maps
            if len(maps) < 2:
                raise ValueError("Checkpoint does not expose the expected multi-label auxiliary heads")
            # This checkpoint's head order is P3, P2, P4, P5, P6 (see YAML).
            for key, probabilities, threshold in (
                ("p3_050", maps[0][0], 0.5),
                ("p3_030", maps[0][0], 0.3),
                ("mean_050", (maps[0][0] + maps[1][0]) / 2, 0.5),
            ):
                candidates = auxiliary_detections(probabilities, threshold, provenance + "_aux_" + key)
                auxiliary_banks[key].extend(
                    exclude_background([map_detection(d, (x1, y1), shape) for d in candidates], excluded)
                )
        del result
    print(f"{label}: {len(bounds)} views, {len(detections)} instance candidates", flush=True)
    return detections, auxiliary_banks


def select(bank, names, dent, rust):
    return [
        d
        for d in bank
        if names[int(d.box[5])] in {"D", "R"} and float(d.box[4]) >= (dent if names[int(d.box[5])] == "D" else rust)
    ]


def shadow_rust_candidates(candidates, image, regions):
    """A conservative R-only recovery: keep majority-in-ROI, dark-mask additions."""
    from tools.diagnose_shadow_inference import in_region

    luminance = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    return [
        d
        for d in candidates
        if int(d.box[5]) == 5
        and any(in_region(d, roi) for roi in regions.values())
        and float(np.median(luminance[d.mask])) <= 128
    ]


def save_version(output, name, title, image, source, names, detections, truth, regions, args):
    folder = output / name
    folder.mkdir(exist_ok=True)
    stem = safe_name(source.stem)
    write_image(folder / f"{stem}__all.jpg", render(image, detections, names, args))
    for class_id, class_name in names.items():
        chosen = [d for d in detections if int(d.box[5]) == class_id]
        write_image(folder / f"{stem}__class_{class_id:03d}_{class_name}.jpg", render(image, chosen, names, args))
    predicted = serialize(detections, names)
    metrics = evaluate_boxes(predicted, truth, regions)
    (folder / "evaluation.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (folder / "predictions.json").write_text(
        json.dumps({"source": str(source), "title": title, "detections": predicted}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    masks = np.stack([d.mask for d in detections]) if detections else np.zeros((0, *image.shape[:2]), bool)
    classes = np.array([int(d.box[5]) for d in detections], dtype=int)
    np.savez_compressed(folder / "masks.npz", masks=masks, classes=classes)
    stats = prediction_metrics(masks, classes, names, image.shape[:2])
    (folder / "prediction_metrics.json").write_text(
        json.dumps({**stats, "source": str(source)}, indent=2), encoding="utf-8"
    )
    print(name, json.dumps(metrics["0.5"]["all"]["classes"], ensure_ascii=True), flush=True)
    return {
        "name": name,
        "title": title,
        "image": folder / f"{stem}__all.jpg",
        "evaluation": metrics,
        "metrics": stats,
        "source": str(source),
    }


def comparisons(records, output):
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    from tools.compare_parameter_sweep import draw_figure

    font_path = Path("C:/Windows/Fonts/msjh.ttc")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    panels = []
    for record in records:
        metrics = record["evaluation"]["0.5"]["all"]["classes"]
        panels.append(
            {
                **record,
                "caption": f"{record['name'][:3]} {record['title']}\n"
                f"框 IoU 0.50：D {metrics['D']['tp']}/{metrics['D']['gt']}，"
                f"R {metrics['R']['tp']}/{metrics['R']['gt']}；"
                f"多餘 D {metrics['D']['fp']} / R {metrics['R']['fp']}",
            }
        )
    folder = output / "comparisons"
    folder.mkdir(exist_ok=True)
    for filename, kind, crop, label in (
        ("all_variants.png", "all", None, "全圖"),
        ("left_all.png", "all", REGIONS["left"], "左側暗部"),
        ("left_D.png", "D", REGIONS["left"], "左側暗部"),
        ("right_all.png", "all", REGIONS["right"], "右下陰影"),
        ("right_D.png", "D", REGIONS["right"], "右下陰影"),
        ("right_R.png", "R", REGIONS["right"], "右下陰影"),
    ):
        draw_figure(panels, folder / filename, kind, crop, label)
    # Compact comparison of the baseline, instance recovery, and auxiliary branch.
    chosen = [p for p in panels if p["name"] in {"T01_baseline", "T03_tile512", "T10_aux_mean_050", "T13_shadow_rust"}]
    draw_figure(chosen, folder / "selected_right.png", "all", REGIONS["right"], "右下陰影")


def report(records, output, xml):
    rows = []
    for record in records:
        for threshold, evaluation in record["evaluation"].items():
            for area, groups in [("all", evaluation["all"]["classes"]), *evaluation["regions"].items()]:
                for class_name, stats in groups.items():
                    rows.append(
                        {"version": record["name"], "box_iou": threshold, "region": area, "class": class_name, **stats}
                    )
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "evaluation_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# 陰影漏檢修復實驗",
        "",
        f"評估標註：`{xml}`（25 D、69 R）。",
        "所有推論圖只顯示原图上的遮罩塗色，顏色和透明度 0.35 固定。",
        "XML 僅用於評估；所有裁切位置採固定網格，推論不讀取 XML 座標。",
        "",
        "## 單張圖的框評估（IoU 0.50）",
        "",
        "| 版本 | D 命中/25 | D 多餘 | R 命中/69 | R 多餘 | 右陰影 D 命中 | 右陰影 R 命中 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        metrics = record["evaluation"]["0.5"]
        d, r = metrics["all"]["classes"]["D"], metrics["all"]["classes"]["R"]
        rd, rr = metrics["regions"]["right"]["D"], metrics["regions"]["right"]["R"]
        lines.append(
            f"| {record['name']} | {d['tp']} | {d['fp']} | {r['tp']} | {r['fp']} | {rd['tp']}/{rd['gt']} | {rr['tp']}/{rr['gt']} |"
        )
    lines.extend(
        [
            "",
            "## 解讀限制",
            "",
            "同類別依分數排序、一對一配對。重複候選只能算一次命中，其餘記為多餘。",
            "區域以 XML 物件框中心分組；右陰影範圍 x512..680、y420..534。",
            "VOC 框起點預設減 1（與 evaluate_shadow_annotations.py 一致）；使用 --coordinate-origin 0 可改為原座標。",
            "IoU 0.30 的小物件診斷另存 CSV/JSON；不可與 IoU 0.50 的數值混用。",
            "XML 只有框，無法計算分割精度；遮罩仍需人工核對。多餘預測是未匹配 XML 的候選，不保證都是誤檢。",
            "這張圖用於選參數，數值不代表其他圖片的泛化效果。",
            "輔助分割版本直接使用已訓練的 multi-label 像素輸出，不是新增權重或重新訓練；像素機率與偵測信心分數不同。",
            "原分割標註只有 19 R、12 D，與新 XML 內容不同；checkpoint 未記錄實際訓練標註 hash，故無法斷言它使用的就是本機版本。",
            "新增分割訓練標註需畫出損傷輪廓，不能把整個 XML 框當成損傷遮罩。",
            "T13 以 T03 為候選來源，只補 R 遮罩至少半數落在左右 ROI、遮罩亮度中位數不高於 128 的預測；基準 D 保持原樣。此篩選不使用 XML 座標。",
            "",
            "[全版本比較](comparisons/all_variants.png) · [右下陰影](comparisons/selected_right.png) · [逐類與逐區數據](evaluation_summary.csv)",
            "",
            "各版本 evaluation.json 的 missed_indices 可對應 annotation.xml 第 0 起算物件索引。",
            "masks.npz 保存未從 JPEG 反推的二值遮罩；predictions.json 記錄來源視圖與分數類型。",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def label_review(records, truth, output):
    """Export annotation disagreements and source-centered review crops, not fake masks."""
    audit_path = output / "annotation_audit.json"
    if not audit_path.is_file():
        return
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    baseline = records[0]["evaluation"]["0.5"]["all"]
    base_indices = {match["truth_index"] for match in baseline["matches"]}
    rows = []
    for obj in audit["objects"]:
        hit_versions = [
            record["name"]
            for record in records
            if any(match["truth_index"] == obj["index"] for match in record["evaluation"]["0.5"]["all"]["matches"])
        ]
        rows.append(
            {
                "xml_object_id": obj["xml_object_id"],
                "class": obj["class"],
                "xmin": obj["box"][0],
                "ymin": obj["box"][1],
                "xmax": obj["box"][2],
                "ymax": obj["box"][3],
                "existing_polygon_index": obj["existing_polygon_index"],
                "existing_polygon_box_iou": obj["max_box_iou_with_existing_polygon"],
                "baseline_matched_050": obj["index"] in base_indices,
                "matched_versions_050": ";".join(hit_versions),
            }
        )
    with (output / "label_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source = Path(records[0]["source"])
    image = cv2.imdecode(np.fromfile(source, np.uint8), 1)
    names = {0: "B", 1: "C", 2: "D", 3: "H", 4: "O", 5: "R", 6: "X", 7: "U"}
    old_masks = []
    for line in (output / "existing_segmentation_labels.txt").read_text().splitlines():
        fields = line.split()
        points = np.array([float(v) for v in fields[1:]]).reshape(-1, 2) * [image.shape[1], image.shape[0]]
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        box = np.r_[points.min(0), points.max(0), 1.0, int(fields[0])]
        old_masks.append(Detection(box, mask.astype(bool), "manual_local_polygon"))
    write_image(
        output / "existing_polygons_reference.jpg", render(image, old_masks, names, argparse.Namespace(alpha=0.35))
    )
    from PIL import Image

    from tools.compare_parameter_sweep import font_manager, plt

    font_path = Path("C:/Windows/Fonts/msjh.ttc")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    # Include all right-shadow D and the least represented left D/R examples.
    height, width = image.shape[:2]

    def inside(obj, region):
        from tools.evaluate_damage_xml import in_roi

        return in_roi(obj["box"], pixel_bounds(REGIONS[region], (height, width)))

    selected = [o for o in audit["objects"] if o["class"] == "D" and inside(o, "right")]
    for name in ("D", "R"):
        candidates = sorted(
            [o for o in audit["objects"] if o["class"] == name and inside(o, "left")],
            key=lambda o: o["max_box_iou_with_existing_polygon"],
        )
        selected += candidates[:2]
    selected += sorted(
        [o for o in audit["objects"] if o["class"] == "R" and inside(o, "right")],
        key=lambda o: o["max_box_iou_with_existing_polygon"],
    )[:2]
    conservative = next(record for record in records if record["name"] == "T03_tile512")
    paths = [source, output / "existing_polygons_reference.jpg", records[0]["image"], conservative["image"]]
    images = [Image.open(path).convert("RGB") for path in paths]
    figure, axes = plt.subplots(len(selected), 4, figsize=(13, 2.2 * len(selected)), squeeze=False)
    titles = ("原圖", "本機舊分割標註", "T01 基準", "T03 裁切 512")
    for row, obj in enumerate(selected):
        x1, y1, x2, y2 = obj["box"]
        crop = (max(0, int(x1) - 12), max(0, int(y1) - 12), min(width, int(x2) + 12), min(height, int(y2) + 12))
        for col, (view, title) in enumerate(zip(images, titles)):
            axes[row, col].imshow(view.crop(crop), interpolation="nearest")
            axes[row, col].set_title(f"XML #{obj['xml_object_id']} {obj['class']}｜{title}", fontsize=9)
            axes[row, col].axis("off")
    figure.suptitle("左右陰影標註差異｜XML 只指定檢查位置，沒有把框填成分割結果", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output / "comparisons/annotation_review.png", dpi=150)
    plt.close(figure)
    for view in images:
        view.close()


def rebuild_report(output: Path, origin: int):
    """Re-evaluate saved predictions without re-running the model."""
    settings = json.loads((output / "settings.json").read_text(encoding="utf-8"))
    source = Path(settings["source"])
    image = cv2.imdecode(np.fromfile(source, np.uint8), 1)
    truth = read_voc(Path(settings["xml"]), image.shape[:2], source.name, origin)
    regions = {key: tuple(value) for key, value in settings["regions"].items()}
    audit_labels(
        Path(settings["labels"]),
        truth,
        {0: "B", 1: "C", 2: "D", 3: "H", 4: "O", 5: "R", 6: "X", 7: "U"},
        image.shape[:2],
        output,
    )
    records = []
    for path in sorted(output.glob("T*/predictions.json")):
        if path.parent.name == "T13_shadow_rust":
            continue
        predicted = json.loads(path.read_text(encoding="utf-8"))
        evaluation = evaluate_boxes(predicted["detections"], truth, regions)
        (path.parent / "evaluation.json").write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        metrics = json.loads((path.parent / "prediction_metrics.json").read_text())
        records.append(
            {
                "name": path.parent.name,
                "title": predicted["title"],
                "image": path.parent / f"{source.stem}__all.jpg",
                "source": str(source),
                "metrics": metrics,
                "evaluation": evaluation,
            }
        )
    names = {0: "B", 1: "C", 2: "D", 3: "H", 4: "O", 5: "R", 6: "X", 7: "U"}

    def saved_detections(version):
        record = json.loads((output / version / "predictions.json").read_text(encoding="utf-8"))
        with np.load(output / version / "masks.npz") as data:
            return [
                Detection(np.array([*d["box"], d["score"], int(class_id)]), mask, d["view"])
                for d, mask, class_id in zip(record["detections"], data["masks"], data["classes"])
            ]

    baseline = saved_detections("T01_baseline")
    tile = saved_detections("T03_tile512")
    additions = shadow_rust_candidates(tile[len(baseline) :], image, regions)
    records.append(
        save_version(
            output,
            "T13_shadow_rust",
            "512 裁切，只補暗部 R",
            image,
            source,
            names,
            baseline + additions,
            truth,
            regions,
            argparse.Namespace(alpha=0.35),
        )
    )
    settings["coordinate_origin"] = origin
    (output / "settings.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    report(records, output, Path(settings["xml"]))
    comparisons(records, output)
    label_review(records, truth, output)
    print(f"REBUILT={output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=BASE["WEIGHT_SOURCES"][0])
    parser.add_argument("--source", type=Path, default=BASE["DATASET_SOURCES"][0])
    parser.add_argument("--xml", type=Path, default="C:/Users/USER/Desktop/0928-1/CBHU0708054-A.xml")
    parser.add_argument(
        "--labels",
        type=Path,
        default="C:/Users/USER/Desktop/碩論/dataset0608/dataset0608/dataset/labels/train/B/CBHU0708054-A.txt",
    )
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR.parent / "runs/segment_shadow_recovery")
    parser.add_argument("--imgsz", type=int, default=BASE["IMAGE_SIZE"])
    parser.add_argument("--conf", type=float, default=BASE["CONFIDENCE"])
    parser.add_argument("--iou", type=float, default=BASE["IOU_THRESHOLD"])
    parser.add_argument("--device", default=BASE["DEVICE"])
    parser.add_argument("--tile-overlap", type=float, default=0.4)
    parser.add_argument("--coordinate-origin", type=int, choices=(0, 1), default=1)
    parser.add_argument("--no-auxiliary", action="store_true")
    parser.add_argument("--rebuild-report", type=Path)
    args = parser.parse_args()
    if args.rebuild_report:
        rebuild_report(args.rebuild_report.resolve(), args.coordinate_origin)
        return
    args.alpha, args.gamma, args.clahe_clip, args.shadow_threshold = 0.35, 0.55, 3.0, 180
    if not 0.03 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error("conf must be in [0.03,1], IoU in [0,1]")
    source = args.source.resolve()
    image = cv2.imdecode(np.fromfile(source, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read {source}")
    truth = read_voc(args.xml, image.shape[:2], source.name, args.coordinate_origin)
    output = args.output.resolve() / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    args.regions = {k: pixel_bounds(v, image.shape[:2]) for k, v in REGIONS.items()}
    settings = {
        **vars(args),
        "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "xml_sha256": hashlib.sha256(args.xml.read_bytes()).hexdigest(),
    }
    (output / "settings.json").write_text(
        json.dumps(settings, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    shutil.copy2(args.xml, output / "annotation.xml")
    print(f"OUTPUT={output}", flush=True)
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    if model.task != "segment" or getattr(model.model.model[-1], "end2end", False):
        raise ValueError("Expected a conventional segmentation checkpoint")
    names = normalize_names(model.names)
    if names.get(2) != "D" or names.get(5) != "R":
        raise ValueError("This experiment expects D=2 and R=5")
    audit_labels(args.labels, truth, names, image.shape[:2], output)
    rules = load_exclusion_rules(SCRIPT_DIR / "shadow_exclusion_regions.json")
    excluded, matched = exclusion_masks(source, image.shape[:2], names, rules)
    (output / "exclusion_regions.json").write_text(
        json.dumps({"regions": matched}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    baseline_bank, _ = infer_bank(model, image, args, excluded, multi=False)
    baseline = [d for d in baseline_bank if float(d.box[4]) >= args.conf]
    full, full_aux = infer_bank(model, image, args, excluded, auxiliary=not args.no_auxiliary)
    tiles = {}
    for size in (512, 400, 320):
        tiles[size], _ = infer_bank(model, image, args, excluded, size=size)
    gamma, _ = infer_bank(model, image, args, excluded, size=400, enhancement="gamma")
    clahe, tile_aux = infer_bank(
        model, image, args, excluded, size=400, enhancement="clahe", auxiliary=not args.no_auxiliary
    )
    mixed = tiles[400] + gamma + clahe
    versions = [
        ("T01_baseline", "原圖基準", []),
        ("T02_multi_label", "保留多類別候選", select(full, names, args.conf, args.conf)),
        ("T03_tile512", "512 重疊裁切", select(tiles[512], names, args.conf, args.conf)),
        ("T04_tile400", "400 重疊裁切", select(tiles[400], names, args.conf, args.conf)),
        ("T05_tile320", "320 重疊裁切", select(tiles[320], names, args.conf, args.conf)),
        ("T06_tile_mix", "400 裁切＋提亮", select(mixed, names, args.conf, args.conf)),
        ("T07_tile_mix_recall", "裁切提亮 D:0.08 R:0.10", select(mixed, names, 0.08, 0.10)),
    ]
    if not args.no_auxiliary:
        versions += [
            ("T08_aux_p3_050", "裁切輔助 P3 像素:0.50", tile_aux["p3_050"]),
            ("T09_aux_p3_030", "裁切輔助 P3 像素:0.30", tile_aux["p3_030"]),
            ("T10_aux_mean_050", "輔助 P2/P3 均值:0.50", tile_aux["mean_050"]),
            (
                "T11_instance_aux",
                "裁切偵測＋輔助 P3:0.50",
                select(mixed, names, args.conf, args.conf) + tile_aux["p3_050"],
            ),
            ("T12_full_aux", "原圖輔助 P3:0.50", full_aux["p3_050"]),
        ]
    records = [
        save_version(
            output, name, title, image, source, names, deduplicate(baseline, additions), truth, args.regions, args
        )
        for name, title, additions in versions
    ]
    conservative = deduplicate(baseline, select(tiles[512], names, args.conf, args.conf))
    additions = shadow_rust_candidates(conservative[len(baseline) :], image, args.regions)
    records.append(
        save_version(
            output,
            "T13_shadow_rust",
            "512 裁切，只補暗部 R",
            image,
            source,
            names,
            baseline + additions,
            truth,
            args.regions,
            args,
        )
    )
    report(records, output, args.xml)
    comparisons(records, output)
    label_review(records, truth, output)
    print(f"COMPLETE={output}", flush=True)


if __name__ == "__main__":
    main()
