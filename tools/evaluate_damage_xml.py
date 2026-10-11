"""Evaluate predicted boxes against a VOC XML without using XML at inference.

XML boxes cannot validate mask boundaries. Matching is per-class, one-to-one,
in descending prediction score order. Report both IoU 0.30 and 0.50 for small
objects; neither is called segmentation accuracy.
"""

from pathlib import Path

import numpy as np
from tools.evaluate_shadow_annotations import read_annotation


def read_voc(path: Path, shape: tuple[int, int], filename: str, origin: int = 1) -> list[dict]:
    annotation = read_annotation(path, origin)
    size = (annotation["height"], annotation["width"])
    if size != shape or annotation["filename"].casefold() != filename.casefold():
        raise ValueError("XML filename/dimensions do not match the source image")
    objects = []
    for index, obj in enumerate(annotation["objects"]):
        box = np.array(obj["box_xyxy"])
        if not np.isfinite(box).all() or not (0 <= box[0] < box[2] <= shape[1]
                                              and 0 <= box[1] < box[3] <= shape[0]):
            raise ValueError(f"Invalid XML box: object {index}")
        objects.append({"index": index, "xml_object_id": obj["id"],
                        "class": obj["class_name"], "box": box.tolist()})
    return objects


def box_iou(first, second) -> float:
    a, b = np.asarray(first), np.asarray(second)
    intersection = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0).prod()
    union = np.maximum(a[2:] - a[:2], 0).prod() + np.maximum(b[2:] - b[:2], 0).prod() - intersection
    return float(intersection / union) if union else 0.0


def match_boxes(predictions: list[dict], truth: list[dict], threshold: float) -> dict:
    matches, used, unmatched = [], set(), []
    for prediction in sorted(predictions, key=lambda p: -p["score"]):
        available = [(box_iou(prediction["box"], target["box"]), target)
                     for target in truth if target["class"] == prediction["class"]
                     and target["index"] not in used]
        best_iou, target = max(available, key=lambda pair: pair[0], default=(0, None))
        if target is not None and best_iou >= threshold:
            used.add(target["index"])
            matches.append({"truth_index": target["index"], "iou": best_iou,
                            "score": prediction["score"], "class": prediction["class"]})
        else:
            unmatched.append(prediction)
    metrics = {}
    for name in ("D", "R"):
        total = sum(t["class"] == name for t in truth)
        predicted = sum(p["class"] == name for p in predictions)
        tp = sum(m["class"] == name for m in matches)
        metrics[name] = {"gt": total, "tp": tp, "fp": predicted - tp, "fn": total - tp,
                         "precision": tp / predicted if predicted else 0.0,
                         "recall": tp / total if total else 0.0,
                         "f1": 2 * tp / (predicted + total) if predicted + total else 0.0}
    return {"classes": metrics, "matches": matches,
            "missed_indices": [t["index"] for t in truth if t["index"] not in used],
            "unmatched_predictions": unmatched}


def in_roi(box: list, roi: tuple) -> bool:
    x, y = (np.asarray(box[:2]) + np.asarray(box[2:])) / 2
    return roi[0] <= x < roi[2] and roi[1] <= y < roi[3]


def evaluate_boxes(predictions: list[dict], truth: list[dict], regions: dict) -> dict:
    result = {}
    for threshold in (0.3, 0.5):
        # Match globally first so a large object crossing an ROI boundary cannot
        # be counted as multiple objects. Regional TP is assigned by GT center.
        matched = match_boxes(predictions, truth, threshold)
        result[str(threshold)] = {"all": matched, "regions": {}}
        for region, roi in regions.items():
            selected = [t for t in truth if in_roi(t["box"], roi)]
            indices = {t["index"] for t in selected}
            regional = {}
            for name in ("D", "R"):
                total = sum(t["class"] == name for t in selected)
                tp = sum(m["class"] == name and m["truth_index"] in indices for m in matched["matches"])
                regional[name] = {"gt": total, "tp": tp, "fn": total - tp,
                                  "recall": tp / total if total else 0.0}
            result[str(threshold)]["regions"][region] = regional
    return result
