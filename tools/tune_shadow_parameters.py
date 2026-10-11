"""Compare local inference parameters and cross-view support, using predict-2.py.

The four input views are original, gentle Gamma, gentle CLAHE, and horizontal
flip. The XML is used only for evaluation; it supplies no inference locations.
All masks are mapped back to the original image and rendered as fill only.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.damage_segmentation_predictor import DamageSegmentationPredictor
from tools.diagnose_shadow_inference import CONTEXTS, in_region, map_detection, pixel_bounds
from tools.evaluate_damage_xml import box_iou, read_voc
from tools.predict_seg_by_class import normalize_names
from tools.predict_shadow import Detection, enhance_shadow, exclude_background, exclusion_masks, extract_detections, load_exclusion_rules
from tools.predict_shadow_recovery import BASE, REGIONS, save_version, serialize


def unflip(detection: Detection) -> Detection:
    width = detection.mask.shape[1]
    box = detection.box.copy()
    box[0], box[2] = width - detection.box[2], width - detection.box[0]
    return Detection(box, np.ascontiguousarray(detection.mask[:, ::-1]), detection.view)


def pairwise_iou(boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.zeros((0, 0))
    low = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    high = np.minimum(boxes[:, None, 2:4], boxes[None, :, 2:4])
    intersection = np.maximum(high - low, 0).prod(-1)
    area = np.maximum(boxes[:, 2:4] - boxes[:, :2], 0).prod(-1)
    union = area[:, None] + area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def consensus_candidates(candidates, gate, agreement_iou, min_area):
    selected = [d for d in candidates if int(d.box[5]) in (2, 5)
                and float(d.box[4]) >= gate["D" if int(d.box[5]) == 2 else "R"]
                and int(d.mask.sum()) >= min_area]
    if not selected:
        return [], []
    boxes = np.stack([d.box for d in selected])
    overlap = pairwise_iou(boxes[:, :4])
    same_class = boxes[:, 5, None] == boxes[None, :, 5]
    # A duplicate prediction from one input view must not count as independent
    # evidence. The suffix identifies the photometric/flip view, not the ROI.
    view_types = [d.view.rsplit(":", 1)[-1] for d in selected]
    support = [len({view_types[j] for j in np.flatnonzero(same_class[i] & (overlap[i] >= agreement_iou))})
               for i in range(len(selected))]
    indices = [i for i, count in enumerate(support) if count >= gate["min_support"]]
    return [selected[i] for i in indices], [support[i] for i in indices]


def merge_preserving_baseline(baseline, candidates, iou):
    kept = list(baseline)
    for candidate in sorted(candidates, key=lambda d: -float(d.box[4])):
        if any(int(candidate.box[5]) == int(other.box[5])
               and box_iou(candidate.box[:4], other.box[:4]) >= iou for other in kept):
            continue
        kept.append(candidate)
    return kept


def merge_local_candidates(baseline, candidates, iou, policy, regions):
    """Allow a stronger crop to refine a shadow detection, keeping exterior masks."""
    if policy == "preserve":
        return merge_preserving_baseline(baseline, candidates, iou)
    if policy != "highest_confidence":
        raise ValueError(f"Unknown merge policy: {policy}")
    protected = [d for d in baseline if not any(in_region(d, roi) for roi in regions.values())]
    local = [d for d in baseline if any(in_region(d, roi) for roi in regions.values())]
    return merge_preserving_baseline(protected, local + candidates, iou)


def save_bank(folder, detections, names, shape, logs):
    folder.mkdir(parents=True)
    masks = np.stack([d.mask for d in detections]) if detections else np.zeros((0, *shape), bool)
    np.savez_compressed(folder / "candidate_masks.npz", bits=np.packbits(masks.reshape(len(masks), int(np.prod(shape))), axis=1),
                        shape=np.array(masks.shape, dtype=int))
    (folder / "candidates.json").write_text(json.dumps({"detections": serialize(detections, names), "views": logs},
        indent=2, ensure_ascii=False), encoding="utf-8")


def infer_bank(model, image, bank, config, args, contexts, regions, excluded):
    detections, logs = [], []
    for region, (x1, y1, x2, y2) in contexts.items():
        crop = image[y1:y2, x1:x2]
        for view in ("original", "gamma", "clahe", "flip"):
            processed = np.ascontiguousarray(crop[:, ::-1]) if view == "flip" else enhance_shadow(crop, view, args)
            model.predictor.multi_label = True
            result = model.predict(processed, imgsz=bank["imgsz"], rect=bank["rect"],
                conf=config["candidate_conf"], iou=bank["iou"], device=args.device,
                retina_masks=True, classes=[2, 5], max_det=config["max_det"], verbose=False, save=False)[0]
            raw = extract_detections(result, f"{region}:{view}", crop.shape[:2])
            if view == "flip":
                raw = [unflip(d) for d in raw]
            mapped = exclude_background([map_detection(d, (x1, y1), image.shape[:2]) for d in raw], excluded)
            local = [d for d in mapped if in_region(d, regions[region])]
            logs.append({"region": region, "view": view, "raw_count": len(raw), "roi_count": len(local),
                         "hit_max_det": len(result.boxes) >= config["max_det"]})
            detections.extend(local)
            print(f"  {bank['name']} {region}:{view} {len(raw)} -> {len(local)}", flush=True)
            del result
    return detections, logs


def performance(record):
    metrics = record["evaluation"]["0.5"]
    classes = metrics["all"]["classes"]
    return {"macro_f1": sum(classes[c]["f1"] for c in ("D", "R")) / 2,
            "D_tp": classes["D"]["tp"], "D_fp": classes["D"]["fp"],
            "R_tp": classes["R"]["tp"], "R_fp": classes["R"]["fp"],
            "left_D_tp": metrics["regions"]["left"]["D"]["tp"],
            "left_R_tp": metrics["regions"]["left"]["R"]["tp"],
            "right_D_tp": metrics["regions"]["right"]["D"]["tp"],
            "right_R_tp": metrics["regions"]["right"]["R"]["tp"]}


def export_comparisons(records, output, selected):
    from tools.compare_parameter_sweep import draw_figure, font_manager, plt
    font = Path("C:/Windows/Fonts/msjh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    panels = []
    for record in records:
        p = performance(record)
        panels.append({**record, "caption": f"{record['name'][:3]} {record['title']}\n"
            f"框 IoU0.50：D {p['D_tp']}/25，R {p['R_tp']}/69；未匹配 {p['D_fp']}/{p['R_fp']}"})
    comparison = output / "comparisons"
    comparison.mkdir()
    for filename, kind, crop, label in (
        ("all_versions.png", "all", None, "全圖"),
        ("left_D.png", "D", REGIONS["left"], "左側暗部"),
        ("right_D.png", "D", REGIONS["right"], "右下陰影"),
        ("right_R.png", "R", REGIONS["right"], "右下陰影"),
    ):
        draw_figure(panels, comparison / filename, kind, crop, label)
    chosen = [p for p in panels if p["name"] in selected]
    draw_figure(chosen, comparison / "selected_all.png", "all")
    draw_figure(chosen, comparison / "selected_right.png", "all", REGIONS["right"], "右下陰影")


def validate_config(config):
    if not 0 < config["candidate_conf"] < 1 or not 0 < config["agreement_iou"] <= 1:
        raise ValueError("Invalid candidate confidence/agreement IoU")
    for gate in config["gates"]:
        if not all(config["candidate_conf"] <= gate[c] <= 1 for c in ("D", "R")):
            raise ValueError("Class thresholds must be >= candidate_conf")
        if not 1 <= gate["min_support"] <= 4 or not 0 < gate["merge_iou"] <= 1:
            raise ValueError("Invalid view support or merge IoU")
    for bank in config["banks"]:
        if bank["imgsz"] % 32 or not 0 < bank["iou"] <= 1 or not isinstance(bank["rect"], bool):
            raise ValueError("Invalid bank image size, IoU or padding")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "tools/shadow_parameter_profiles.json")
    parser.add_argument("--model", type=Path, default=BASE["WEIGHT_SOURCES"][0])
    parser.add_argument("--source", type=Path, default=BASE["DATASET_SOURCES"][0])
    parser.add_argument("--xml", type=Path, default="C:/Users/USER/Desktop/0928-1/CBHU0708054-A.xml")
    parser.add_argument("--device", default=BASE["DEVICE"])
    parser.add_argument("--output", type=Path, default=ROOT / "runs/segment_shadow_parameter_tuning")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    args.gamma, args.clahe_clip, args.shadow_threshold, args.alpha = (
        config["gamma"], config["clahe_clip"], config["shadow_threshold"], .35)
    image = cv2.imdecode(np.fromfile(args.source, np.uint8), 1)
    if image is None:
        raise ValueError(f"Cannot read {args.source}")
    truth = read_voc(args.xml, image.shape[:2], args.source.name)
    regions = {name: pixel_bounds(bounds, image.shape[:2]) for name, bounds in REGIONS.items()}
    contexts = {name: pixel_bounds(bounds, image.shape[:2]) for name, bounds in config.get("contexts", CONTEXTS).items()}
    if set(contexts) != set(regions):
        raise ValueError("Every shadow region needs a matching context crop")
    output = args.output / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    settings = {**vars(args), "parameters": config, "regions": regions, "contexts": contexts,
        "baseline": {"imgsz": BASE["IMAGE_SIZE"], "conf": BASE["CONFIDENCE"], "iou": BASE["IOU_THRESHOLD"]},
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "image_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "xml_sha256": hashlib.sha256(args.xml.read_bytes()).hexdigest(), "coordinate_origin": 1}
    (output / "settings.json").write_text(json.dumps(settings, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(args.xml, output / "annotation.xml")
    shutil.copy2(args.config, output / "profiles.json")
    print(f"OUTPUT={output}", flush=True)
    from ultralytics import YOLO
    model = YOLO(str(args.model))
    names = normalize_names(model.names)
    if names.get(2) != "D" or names.get(5) != "R" or model.task != "segment":
        raise ValueError("Expected segmentation checkpoint with D=2, R=5")
    rules = load_exclusion_rules(ROOT / "tools/shadow_exclusion_regions.json")
    excluded, matched = exclusion_masks(args.source, image.shape[:2], names, rules)
    (output / "exclusion_regions.json").write_text(json.dumps({"regions": matched}, indent=2, ensure_ascii=False), encoding="utf-8")
    DamageSegmentationPredictor.multi_label = False
    DamageSegmentationPredictor.collect_auxiliary = False
    result = model.predict(image, predictor=DamageSegmentationPredictor, imgsz=BASE["IMAGE_SIZE"],
        conf=BASE["CONFIDENCE"], iou=BASE["IOU_THRESHOLD"], rect=True, device=args.device,
        retina_masks=True, max_det=1000, verbose=False, save=False)[0]
    baseline = exclude_background(extract_detections(result, "baseline", image.shape[:2]), excluded)
    del result
    records = [save_version(output, "P01_baseline", "原圖基準", image, args.source, names, baseline, truth, regions, args)]
    parameters = [{"version": "P01_baseline", **settings["baseline"]}]
    for bank in config["banks"]:
        candidates, logs = infer_bank(model, image, bank, config, args, contexts, regions, excluded)
        save_bank(output / "candidate_banks" / bank["name"], candidates, names, image.shape[:2], logs)
        for gate in config["gates"]:
            accepted, support = consensus_candidates(candidates, gate, config["agreement_iou"], config["min_mask_area"])
            combined = merge_preserving_baseline(baseline, accepted, gate["merge_iou"])
            version = f"P{len(records) + 1:02d}_{bank['name']}_{gate['name']}"
            title = f"{bank['imgsz']} D:{gate['D']:g} R:{gate['R']:g} 支持:{gate['min_support']}"
            records.append(save_version(output, version, title, image, args.source, names, combined, truth, regions, args))
            parameter = {"version": version, **bank, **gate, "bank_name": bank["name"], "gate_name": gate["name"], "candidate_count": len(candidates),
                         "accepted_before_merge": len(accepted), "support_counts": support}
            parameters.append(parameter)
            (output / version / "parameters.json").write_text(json.dumps(parameter, indent=2), encoding="utf-8")
        del candidates
    (output / "parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    scores = [{"version": r["name"], **performance(r)} for r in records]
    with (output / "comparison_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    # A balanced result must preserve baseline matched counts. This selection
    # uses the same image's XML and is not independent validation.
    base = scores[0]
    eligible = [s for s in scores if s["D_tp"] >= base["D_tp"] and s["R_tp"] >= base["R_tp"]]
    balanced = max(eligible, key=lambda s: (s["macro_f1"], -(s["D_fp"] + s["R_fp"])))
    recall = max(scores, key=lambda s: (s["right_D_tp"], s["left_D_tp"], s["right_R_tp"], s["left_R_tp"], s["macro_f1"]))
    alternatives = [s for s in eligible if s["version"] != base["version"]]
    candidate = max(alternatives, key=lambda s: (s["macro_f1"], -(s["D_fp"] + s["R_fp"]))) if alternatives else balanced
    selected = {"balanced": balanced, "recall": recall, "best_new_candidate": candidate,
        "scope": "parameter selection on this single annotated image; no independent validation"}
    (output / "selected_parameters.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    export_comparisons(records, output, {base["version"], balanced["version"], recall["version"], candidate["version"]})
    lines = ["# 陰影參數調整比較", "", f"{len(records)} 版：原圖基準＋{len(config['banks'])} 組局部尺寸/IoU × {len(config['gates'])} 組門檻/視圖支持。",
        "所有版保留基準遮罩，只增加左右陰影內的實際模型遮罩；維持純塗色。",
        "新增候選先各自滿足 D/R 門檻，再要求原圖、Gamma、CLAHE、翻轉中不同視圖支持。",
        "多個同視圖候選不重複計票；D 與 R 不互相抑制。XML 只用於框評估。",
        "", "| 版本 | D 命中/25 | D 未匹配 | R 命中/69 | R 未匹配 | 左 D | 右 D | 右 R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for score in scores:
        lines.append(f"| {score['version']} | {score['D_tp']} | {score['D_fp']} | {score['R_tp']} | {score['R_fp']} | {score['left_D_tp']}/22 | {score['right_D_tp']}/2 | {score['right_R_tp']}/9 |")
    lines += ["", f"保留命中數後的 F1 較佳版本：{balanced['version']}。", f"陰影召回優先版本：{recall['version']}。",
        "未匹配項可能是誤檢、重複或定位偏差。門檻選擇使用了這張圖的 XML，不能作為泛化效果。",
        "框 IoU 固定 0.50；IoU 0.30 另存在各版 evaluation.json，沒有調低評估標準。",
        "未改動權重或重新訓練。所有參數記錄在 profiles.json/settings.json/parameters.json。",
        "[重點比較](comparisons/selected_all.png) · [右下比較](comparisons/selected_right.png) · [完整數據](comparison_summary.csv)"]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("SELECTED=" + json.dumps(selected), flush=True)
    print(f"COMPLETE={output}", flush=True)


if __name__ == "__main__":
    main()
