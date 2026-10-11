"""Refine per-class confidence/support/merge parameters from cached candidates.

Inference locations are already fixed in the source run. The XML evaluates and
selects parameter recipes; it never edits a predicted box or a mask.
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
from tools.diagnose_shadow_inference import in_region
from tools.evaluate_damage_xml import read_voc
from tools.predict_shadow import Detection
from tools.predict_shadow_recovery import save_version
from tools.tune_shadow_parameters import pairwise_iou, performance


def rectangular_iou(a, b):
    intersection = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:])
                              - np.maximum(a[:, None, :2], b[None, :, :2]), 0).prod(-1)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(-1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(-1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def metadata_selection(base, candidates, gate, config, protected, overlap=None):
    """The same score, view-vote and box merge rules used by the inference runner."""
    count = len(base)
    all_d = base + candidates
    protected = np.asarray(protected, dtype=bool)
    if overlap is None:
        overlap = pairwise_iou(np.asarray([d["box"] for d in all_d]).reshape(-1, 4))
    scores = np.array([d["score"] for d in candidates])
    valid = np.flatnonzero((scores >= gate["threshold"])
                          & (np.array([d["mask_pixels"] for d in candidates]) >= config["min_mask_area"]))
    views = [d["view"].rsplit(":", 1)[-1] for d in candidates]
    support = np.zeros(len(valid), dtype=int)
    for view in set(views):
        same_view = [j for j in valid if views[j] == view]
        support += np.any(overlap[np.ix_(count + valid, count + np.array(same_view, dtype=int))]
                          >= config["agreement_iou"], axis=1)
    accepted = (count + valid[support >= gate["min_support"]]).tolist()
    if gate["merge_policy"] == "preserve":
        kept = list(range(count))
        remaining = accepted
    elif gate["merge_policy"] == "highest_confidence":
        kept = list(np.flatnonzero(protected))
        remaining = list(np.flatnonzero(~protected)) + accepted
    else:
        raise ValueError(f"Unknown merge policy: {gate['merge_policy']}")
    remaining.sort(key=lambda i: -all_d[i]["score"])
    for index in remaining:
        if not kept or not np.any(overlap[index, kept] >= gate["merge_iou"]):
            kept.append(index)
    return kept, all_d


def fast_match(detections, truth, regions):
    if not detections:
        return {"tp": 0, "fp": 0, "f1": 0., "matched_ids": [], "left_tp": 0, "right_tp": 0}
    if not truth:
        return {"tp": 0, "fp": len(detections), "f1": 0., "matched_ids": [], "left_tp": 0, "right_tp": 0}
    iou = rectangular_iou(np.array([d["box"] for d in detections]), np.array([g["box"] for g in truth]))
    used = set()
    for index in sorted(range(len(detections)), key=lambda i: -detections[i]["score"]):
        available = [j for j in range(len(truth)) if j not in used]
        if not available:
            break
        target = max(available, key=lambda j: iou[index, j])
        if iou[index, target] >= .5:
            used.add(target)
    centers = [(np.asarray(g["box"][:2]) + np.asarray(g["box"][2:])) / 2 for g in truth]
    result = {"tp": len(used), "fp": len(detections) - len(used),
              "f1": 2 * len(used) / (len(detections) + len(truth)),
              "matched_ids": [truth[j]["xml_object_id"] for j in sorted(used)]}
    for name, roi in regions.items():
        result[name + "_tp"] = int(sum(roi[0] <= centers[j][0] < roi[2] and roi[1] <= centers[j][1] < roi[3] for j in used))
    return result


def load_detection_masks(predictions, archive):
    with np.load(archive) as data:
        if "bits" in data:
            count, height, width = data["shape"].tolist()
            masks = np.unpackbits(data["bits"], axis=1, count=height * width).reshape(count, height, width).astype(bool)
            classes = [2 if d["class"] == "D" else 5 for d in predictions]
        else:
            masks, classes = data["masks"], data["classes"]
        return [Detection(np.array([*d["box"], d["score"], int(cid)]), mask, d["view"])
                for d, mask, cid in zip(predictions, masks, classes)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/segment_shadow_refined")
    parser.add_argument("--no-update-preset", action="store_true", help="Keep the current executable preset when evaluating an alternative run")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    settings = json.loads((run / "settings.json").read_text(encoding="utf-8"))
    config = settings["parameters"]
    source = Path(settings["source"])
    for key, path in (("image_sha256", source), ("xml_sha256", Path(settings["xml"]))):
        if hashlib.sha256(path.read_bytes()).hexdigest() != settings[key]:
            raise ValueError(f"{path} changed since candidate inference; generate candidates again")
    image = cv2.imdecode(np.fromfile(source, np.uint8), 1)
    truth = read_voc(Path(settings["xml"]), image.shape[:2], source.name)
    regions = {k: tuple(v) for k, v in settings["regions"].items()}
    base = json.loads((run / "P01_baseline/predictions.json").read_text(encoding="utf-8"))["detections"]
    base_masks = load_detection_masks(base, run / "P01_baseline/masks.npz")
    mask_lookup = {id(d): mask for d, mask in zip(base, base_masks)}
    bank_data = {}
    for bank in config["banks"]:
        path = run / "candidate_banks" / bank["name"]
        bank_data[bank["name"]] = json.loads((path / "candidates.json").read_text(encoding="utf-8"))["detections"]
    results, balanced, recall = [], {}, {}
    for class_name in ("D", "R"):
        old = [d for d in base if d["class"] == class_name]
        gt = [g for g in truth if g["class"] == class_name]
        protected = np.array([not any(in_region(mask_lookup[id(d)], roi) for roi in regions.values()) for d in old])
        reference = fast_match(old, gt, regions)
        class_rows = []
        thresholds = (.005, .01, .02, .04, .08, .12, .15, .2, .3, .4, .6, .8) if class_name == "D" else (.06, .08, .1, .12, .15, .18, .2, .25, .3)
        for bank in config["banks"]:
            candidates = [d for d in bank_data[bank["name"]] if d["class"] == class_name]
            overlap = pairwise_iou(np.asarray([d["box"] for d in old + candidates]).reshape(-1, 4))
            for threshold in thresholds:
                for votes in (1, 2, 3):
                    for merge in (.5, .7, .85, .95):
                        for policy in ("preserve", "highest_confidence"):
                            recipe = {"class": class_name, "bank": bank["name"], "threshold": threshold,
                                      "min_support": votes, "merge_iou": merge, "merge_policy": policy}
                            indices, all_d = metadata_selection(old, candidates, recipe, config, protected, overlap)
                            score = fast_match([all_d[i] for i in indices], gt, regions)
                            class_rows.append({**recipe, **score})
            print(f"screened {class_name} {bank['name']}", flush=True)
        # Retain every object the baseline already matched. This is an offline
        # parameter selection condition, never a per-prediction annotation filter.
        eligible = [r for r in class_rows if set(reference["matched_ids"]) <= set(r["matched_ids"])]
        balanced[class_name] = max(eligible, key=lambda r: (r["f1"], r["right_tp"], r["left_tp"], -r["fp"], r["threshold"]))
        recall[class_name] = max(eligible, key=lambda r: (r["right_tp"], r["left_tp"], r["f1"], -r["fp"]))
        results += class_rows
        print("BEST " + class_name + " " + json.dumps(balanced[class_name]), flush=True)
    output = args.output / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    shutil.copy2(run / "settings.json", output / "input_provenance.json")
    with (output / "parameter_screen.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    cache = {}
    def compose(recipes):
        combined = [d for d in base_masks if int(d.box[5]) not in (2, 5)]
        for name in ("D", "R"):
            if name not in recipes:
                combined += [m for d, m in zip(base, base_masks) if d["class"] == name]
                continue
            recipe = recipes[name]
            bank_name = recipe["bank"]
            if bank_name not in cache:
                full = bank_data[bank_name]
                masks = load_detection_masks(full, run / "candidate_banks" / bank_name / "candidate_masks.npz")
                cache[bank_name] = {id(d): m for d, m in zip(full, masks)}
            old = [d for d in base if d["class"] == name]
            candidates = [d for d in bank_data[bank_name] if d["class"] == name]
            protected = np.array([not any(in_region(mask_lookup[id(d)], roi) for roi in regions.values()) for d in old])
            indices, all_d = metadata_selection(old, candidates, recipe, config, protected)
            for index in indices:
                d = all_d[index]
                combined.append(mask_lookup[id(d)] if index < len(old) else cache[bank_name][id(d)])
        return sorted(combined, key=lambda d: -float(d.box[4]))
    names = {0: "B", 1: "C", 2: "D", 3: "H", 4: "O", 5: "R", 6: "X", 7: "U"}
    drawing = argparse.Namespace(alpha=.35)
    records = [save_version(output, "Q01_baseline", "原圖基準", image, source, names, base_masks, truth, regions, drawing)]
    for version, title, recipes in (
        ("Q02_D_refined", "僅調整 D", {"D": balanced["D"]}),
        ("Q03_R_refined", "僅調整 R", {"R": balanced["R"]}),
        ("Q04_balanced", "D/R 各自調整", balanced),
        ("Q05_recall", "陰影召回優先診斷", recall),
    ):
        records.append(save_version(output, version, title, image, source, names, compose(recipes), truth, regions, drawing))
        (output / version / "parameters.json").write_text(json.dumps(recipes, indent=2), encoding="utf-8")
    preset = {"source_filename": source.name, "source_path": str(source), "model_path": settings["model"],
              "image_shape": list(image.shape[:2]), "model_sha256": settings["model_sha256"],
              "baseline": settings["baseline"], "regions": settings["regions"], "contexts": settings["contexts"],
              "enhancement": {k: config[k] for k in ("gamma", "clahe_clip", "shadow_threshold")},
              "candidate_conf": config["candidate_conf"], "max_det": config["max_det"],
              "agreement_iou": config["agreement_iou"], "min_mask_area": config["min_mask_area"],
              "banks": [b for b in config["banks"] if b["name"] in {r["bank"] for r in balanced.values()}],
              "classes": balanced, "selected_from": str(run),
              "selection_scope": "same-image parameter tuning using XML; not independent validation"}
    (output / "selected_preset.json").write_text(json.dumps(preset, indent=2), encoding="utf-8")
    scores = [{"version": r["name"], **performance(r)} for r in records]
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    from tools.compare_parameter_sweep import draw_figure, font_manager, plt
    font = Path("C:/Windows/Fonts/msjh.ttc")
    font_manager.fontManager.addfont(str(font)); plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    panels = []
    for record in records:
        score = performance(record)
        panels.append({**record, "caption": f"{record['name'][:3]} {record['title']}\n框 IoU0.50：D {score['D_tp']}/25，R {score['R_tp']}/69；未匹配 {score['D_fp']}/{score['R_fp']}"})
    comp = output / "comparisons"; comp.mkdir()
    draw_figure(panels, comp / "all.png", "all")
    draw_figure(panels, comp / "left_D.png", "D", REGIONS_LEFT, "左側暗部")
    draw_figure(panels, comp / "right_all.png", "all", REGIONS_RIGHT, "右下陰影")
    lines = ["# 分類門檻與合併參數調整", "", f"篩選 {len(results)} 組每類參數；沒有重新跑模型或改動遮罩像素。",
        "門檻、視圖支持和合併規則套用模型已保存的候選。XML 只用於評估與選參數。",
        "最高分合併只允許更新左右檢查區內的基準預測；區外遮罩保留。",
        "", "| 版本 | D 命中/25 | D 未匹配 | R 命中/69 | R 未匹配 | 左 D/22 | 右 D/2 | 右 R/9 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in scores:
        lines.append(f"| {s['version']} | {s['D_tp']} | {s['D_fp']} | {s['R_tp']} | {s['R_fp']} | {s['left_D_tp']} | {s['right_D_tp']} | {s['right_R_tp']} |")
    lines += ["", "每類最佳參數：", "", "```json", json.dumps(balanced, indent=2), "```", "",
        "分數選擇是此圖回測，不能作為未見圖片效果。框 IoU 固定 0.50；沒有將 XML 框作為塗色或重訓。",
        "Q05 是召回優先診斷；低分新增項可能有大量誤檢。Q04 是同時保持原命中物件、以 F1 選取的參數。",
        "[全圖比較](comparisons/all.png) · [左 D](comparisons/left_D.png) · [右下比較](comparisons/right_all.png)"]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not args.no_update_preset:
        shutil.copy2(output / "selected_preset.json", ROOT / "tools/shadow_tuned_parameters.json")
    print("OUTPUT=" + str(output), flush=True)
    print("SELECTED=" + json.dumps(balanced), flush=True)


REGIONS_LEFT = (.03, .20, .54, .90)
REGIONS_RIGHT = (.64, .70, .85, .89)

if __name__ == "__main__":
    main()
