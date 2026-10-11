"""Nine pixel metrics for independent foreground-class binary masks.

No true negatives are inferred from object-detection match counts. Undefined
ratios are None, never fabricated zeros. Macro means omit absent classes.
"""

from __future__ import annotations

import math
import numpy as np

METRIC_COLUMNS = ("Accuracy", "Precision", "Recall", "Dice", "IoU", "mIoU", "Mean Dice", "FPR", "FNR")
COUNT_COLUMNS = ("TP", "FP", "FN", "TN")


def confusion_counts(prediction, truth):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    if prediction.dtype != np.bool_ or truth.dtype != np.bool_ or prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("Prediction and ground truth must be bool HxW masks of the same size")
    tp = int(np.count_nonzero(prediction & truth))
    predicted = int(np.count_nonzero(prediction))
    annotated = int(np.count_nonzero(truth))
    fp, fn = predicted - tp, annotated - tp
    return {"TP": tp, "FP": fp, "FN": fn, "TN": truth.size - tp - fp - fn}


def metric_values(counts):
    values = [counts[key] for key in COUNT_COLUMNS]
    if any(not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("Confusion counts must be nonnegative integers")
    tp, fp, fn, tn = values
    def divide(numerator, denominator):
        return numerator / denominator if denominator else None
    return {"Accuracy": divide(tp + tn, tp + fp + fn + tn),
            "Precision": divide(tp, tp + fp), "Recall": divide(tp, tp + fn),
            "Dice": divide(2 * tp, 2 * tp + fp + fn), "IoU": divide(tp, tp + fp + fn),
            "FPR": divide(fp, fp + tn), "FNR": divide(fn, tp + fn)}


def mean_defined(values):
    selected = [value for value in values if value is not None]
    return sum(selected) / len(selected) if selected else None


def summarize_counts(class_counts):
    """Pooled pixel micro scores plus class macro mIoU/Mean Dice.

Returns every requested column for both the overall row and each class row.
Macro averages apply only to the overall scope, not to an individual class.
"""
    by_class = {name: {**counts, **metric_values(counts)} for name, counts in class_counts.items()}
    macro = {"mIoU": mean_defined(row["IoU"] for row in by_class.values()),
             "Mean Dice": mean_defined(row["Dice"] for row in by_class.values())}
    total = {key: sum(counts[key] for counts in class_counts.values()) for key in COUNT_COLUMNS}
    overall = {**total, **metric_values(total), **macro}
    return overall, {name: {**row, "mIoU": None, "Mean Dice": None} for name, row in by_class.items()}


def rasterize_boxes(objects, names, height, width):
    """EXPLICIT box-area proxy only; does not generate real segmentation GT.

Tests image pixel centers against zero-based continuous [x1,y1,x2,y2) boxes.
Instances of the same class are united; different classes may overlap.
"""
    if height <= 0 or width <= 0:
        raise ValueError("Invalid raster dimensions")
    masks = {name: np.zeros((height, width), dtype=bool) for name in names}
    for entry in objects:
        name, box = entry["class"], entry["box"]
        if name not in masks or len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError("Unknown class or invalid box")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Box is outside the image")
        left, top = max(0, math.ceil(x1 - .5)), max(0, math.ceil(y1 - .5))
        right, bottom = min(width, math.ceil(x2 - .5)), min(height, math.ceil(y2 - .5))
        masks[name][top:bottom, left:right] = True
    return masks
