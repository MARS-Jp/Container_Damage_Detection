"""Run the selected shadow parameters without reading annotations.

PowerShell (in the repository):
    & 'C:/Users/USER/anaconda3/envs/yolov11/python.exe' -B tools/predict_tuned_shadow.py

Each run creates a new folder with pure mask-fill images and prediction metadata.
The default preset is specific to CBHU0708054-A and last0912.pt.
"""

from __future__ import annotations

import argparse
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
from tools.predict_seg_by_class import normalize_names, prediction_metrics, safe_name, write_image
from tools.predict_shadow import exclude_background, exclusion_masks, extract_detections, load_exclusion_rules, render
from tools.predict_shadow_recovery import BASE, serialize
from tools.tune_shadow_parameters import consensus_candidates, infer_bank, merge_local_candidates


def validate_preset(preset, source, shape):
    if source.name.casefold() != preset["source_filename"].casefold():
        raise ValueError("This preset's fixed shadow regions are for " + preset["source_filename"])
    if tuple(preset["image_shape"]) != tuple(shape):
        raise ValueError("Source dimensions do not match the preset's pixel coordinates")
    height, width = shape
    if set(preset["regions"]) != set(preset["contexts"]):
        raise ValueError("Each shadow region needs its own context crop")
    for bounds in [*preset["regions"].values(), *preset["contexts"].values()]:
        x1, y1, x2, y2 = bounds
        if not all(isinstance(v, int) for v in bounds) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Invalid preset crop coordinates")
    banks = {b["name"]: b for b in preset["banks"]}
    if len(banks) != len(preset["banks"]) or set(preset["classes"]) != {"D", "R"}:
        raise ValueError("Expected unique banks and per-class D/R recipes")
    if not 0 < preset["candidate_conf"] <= 1 or not 0 < preset["agreement_iou"] <= 1 or preset["min_mask_area"] < 1:
        raise ValueError("Invalid candidate/vote/area parameters")
    for bank in banks.values():
        if bank["imgsz"] < 32 or bank["imgsz"] % 32 or not isinstance(bank["rect"], bool) or not 0 < bank["iou"] <= 1:
            raise ValueError("Invalid inference bank")
    for recipe in preset["classes"].values():
        if recipe["bank"] not in banks or not preset["candidate_conf"] <= recipe["threshold"] <= 1:
            raise ValueError("Class recipe needs an existing bank and threshold >= candidate_conf")
        if not 1 <= recipe["min_support"] <= 4 or not 0 < recipe["merge_iou"] <= 1:
            raise ValueError("Invalid support/merge parameters")
        if recipe["merge_policy"] not in {"preserve", "highest_confidence"}:
            raise ValueError("Invalid merge policy")


def apply_recipes(baseline, candidate_banks, preset):
    """Use only inference parameters, never the offline XML matching metadata."""
    combined = [d for d in baseline if int(d.box[5]) not in (2, 5)]
    logs = {}
    for name, cid in (("D", 2), ("R", 5)):
        recipe = preset["classes"][name]
        old = [d for d in baseline if int(d.box[5]) == cid]
        raw = [d for d in candidate_banks[recipe["bank"]] if int(d.box[5]) == cid]
        gate = {"D": 1., "R": 1., "min_support": recipe["min_support"], name: recipe["threshold"]}
        accepted, support = consensus_candidates(raw, gate, preset["agreement_iou"], preset["min_mask_area"])
        selected = merge_local_candidates(old, accepted, recipe["merge_iou"], recipe["merge_policy"], preset["regions"])
        combined += selected
        logs[name] = {"baseline_count": len(old), "raw_candidates": len(raw), "accepted_before_merge": len(accepted),
                      "support_counts": support, "merged_count": len(selected)}
    return sorted(combined, key=lambda d: -float(d.box[4])), logs


def save_predictions(folder, image, source, names, detections, drawing):
    folder.mkdir(parents=True)
    stem = safe_name(source.stem)
    write_image(folder / f"{stem}__all.jpg", render(image, detections, names, drawing))
    for cid, name in names.items():
        chosen = [d for d in detections if int(d.box[5]) == cid]
        write_image(folder / f"{stem}__class_{cid:03d}_{safe_name(name)}.jpg", render(image, chosen, names, drawing))
    masks = np.stack([d.mask for d in detections]) if detections else np.zeros((0, *image.shape[:2]), bool)
    classes = np.array([int(d.box[5]) for d in detections], dtype=int)
    np.savez_compressed(folder / "masks.npz", masks=masks, classes=classes)
    report = {"source": str(source), "detections": serialize(detections, names)}
    (folder / "predictions.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics = {**prediction_metrics(masks, classes, names, image.shape[:2]), "source": str(source)}
    (folder / "prediction_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", type=Path, default=ROOT / "tools/shadow_tuned_parameters.json")
    parser.add_argument("--model", type=Path, help="Override the preset's checkpoint path")
    parser.add_argument("--source", type=Path, help="Override the preset's source path")
    parser.add_argument("--device", default=BASE["DEVICE"])
    parser.add_argument("--output", type=Path, default=ROOT / "runs/segment_shadow_tuned")
    args = parser.parse_args()
    preset = json.loads(args.preset.read_text(encoding="utf-8"))
    args.model = args.model or Path(preset["model_path"])
    args.source = args.source or Path(preset["source_path"])
    image = cv2.imdecode(np.fromfile(args.source, np.uint8), 1)
    if image is None:
        raise ValueError(f"Cannot decode source image: {args.source}")
    validate_preset(preset, args.source, image.shape[:2])
    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if model_hash != preset["model_sha256"]:
        raise ValueError("Checkpoint differs from the model used to select this preset")
    args.alpha = .35
    for name, value in preset["enhancement"].items():
        setattr(args, name, value)
    from ultralytics import YOLO
    model = YOLO(str(args.model))
    names = normalize_names(model.names)
    if model.task != "segment" or names.get(2) != "D" or names.get(5) != "R":
        raise ValueError("Expected segmentation checkpoint with D=2 and R=5")
    rules = load_exclusion_rules(ROOT / "tools/shadow_exclusion_regions.json")
    excluded, matched = exclusion_masks(args.source, image.shape[:2], names, rules)
    output = args.output / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    shutil.copy2(args.preset, output / "preset.json")
    settings = {**vars(args), "model_sha256": model_hash, "image_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "excluded_regions": matched, "retina_masks": True, "alpha": .35, "views": ["original", "gamma", "clahe", "flip"],
                "baseline_multi_label": False, "crop_multi_label": True, "annotation_used_at_inference": False}
    (output / "settings.json").write_text(json.dumps(settings, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"OUTPUT={output}", flush=True)
    DamageSegmentationPredictor.multi_label = False
    DamageSegmentationPredictor.collect_auxiliary = False
    result = model.predict(image, predictor=DamageSegmentationPredictor, **preset["baseline"],
        rect=True, device=args.device, retina_masks=True, max_det=1000, classes=None, verbose=False, save=False)[0]
    baseline = exclude_background(extract_detections(result, "baseline", image.shape[:2]), excluded)
    del result
    save_predictions(output / "baseline", image, args.source, names, baseline, args)
    banks, view_logs = {}, {}
    for bank in preset["banks"]:
        banks[bank["name"]], view_logs[bank["name"]] = infer_bank(model, image, bank, preset, args,
            preset["contexts"], preset["regions"], excluded)
    selected, selection_logs = apply_recipes(baseline, banks, preset)
    save_predictions(output / "tuned", image, args.source, names, selected, args)
    (output / "inference_log.json").write_text(json.dumps({"views": view_logs, "selection": selection_logs}, indent=2), encoding="utf-8")
    print("COUNTS=" + json.dumps({name: data["merged_count"] for name, data in selection_logs.items()}), flush=True)
    print(f"COMPLETE={output}", flush=True)


if __name__ == "__main__":
    main()
