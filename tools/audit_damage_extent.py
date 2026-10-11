"""Compare VOC damage boxes with YOLO instance polygons without changing either label source.

This is a geometry audit, not a mask-quality assessment or a label conversion tool.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def audit(xml_path: Path, labels_path: Path, names: list[str], min_iou=0.1, min_extent=0.75) -> dict:
    """Return same-class one-to-one bbox comparisons; ratios only flag manual review."""
    xml_path, labels_path = Path(xml_path), Path(labels_path)
    if not names or len(names) != len(set(names)):
        raise ValueError("Class names must be a nonempty list of unique values.")
    for value, key in ((min_iou, "min_iou"), (min_extent, "min_extent")):
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{key} must be in (0, 1].")
    root = ET.parse(xml_path).getroot()
    width, height = float(root.findtext("size/width")), float(root.findtext("size/height"))
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("XML image dimensions must be finite and positive.")
    truth, counts = [], Counter()
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in names:
            raise ValueError(f"Unknown XML class {name!r}.")
        box = [float(obj.findtext(f"bndbox/{key}")) for key in ("xmin", "ymin", "xmax", "ymax")]
        if not all(math.isfinite(v) for v in box) or not (
            0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height
        ):
            raise ValueError(f"Invalid XML box: {box}.")
        counts[name] += 1
        truth.append({"xml_id": f"{name}{counts[name]:02d}", "class": name, "box": box})
    polygons = []
    for line_number, line in enumerate(labels_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        values = np.asarray([float(v) for v in line.split()], dtype=float)
        if len(values) < 7 or len(values) % 2 != 1 or not np.isfinite(values).all():
            raise ValueError(f"Line {line_number}: expected class ID plus at least three finite polygon vertices.")
        class_id = int(values[0])
        if values[0] != class_id or not 0 <= class_id < len(names):
            raise ValueError(f"Line {line_number}: invalid polygon class ID.")
        points = values[1:].reshape(-1, 2)
        if (points < 0).any() or (points > 1).any():
            raise ValueError(f"Line {line_number}: polygon coordinates must be normalized to [0, 1].")
        points = points * np.array([width, height])
        box = np.concatenate((points.min(0), points.max(0)))
        if (box[2:] <= box[:2]).any():
            raise ValueError(f"Line {line_number}: polygon has an empty bounding box.")
        polygons.append({"label_line": line_number, "class": names[class_id], "box": box.tolist()})
    matches, used_truth, used_polygons = [], set(), set()
    for name in names:
        ti = [i for i, g in enumerate(truth) if g["class"] == name]
        pi = [i for i, p in enumerate(polygons) if p["class"] == name]
        if not ti or not pi:
            continue
        matrix = np.asarray([[_iou(truth[i]["box"], polygons[j]["box"]) for j in pi] for i in ti])
        # Dummy rows/columns allow unmatched objects instead of forcing a below-threshold pair.
        weights = np.zeros((len(ti) + len(pi), len(ti) + len(pi)))
        weights[: len(ti), : len(pi)] = np.where(matrix >= min_iou, matrix, 0)
        rows, cols = linear_sum_assignment(-weights)
        for row, col in zip(rows, cols):
            if row >= len(ti) or col >= len(pi) or matrix[row, col] < min_iou:
                continue
            i, j = ti[row], pi[col]
            used_truth.add(i)
            used_polygons.add(j)
            a, b = np.asarray(truth[i]["box"]), np.asarray(polygons[j]["box"])
            ratios = (b[2:] - b[:2]) / (a[2:] - a[:2])
            matches.append(
                {
                    "xml_id": truth[i]["xml_id"],
                    "class": name,
                    "label_line": polygons[j]["label_line"],
                    "xml_box": a.tolist(),
                    "polygon_bbox": b.tolist(),
                    "bbox_iou": float(matrix[row, col]),
                    "width_ratio": float(ratios[0]),
                    "height_ratio": float(ratios[1]),
                    "review_extent": bool((ratios < min_extent).any()),
                }
            )
    return {
        "xml": str(xml_path.resolve()),
        "labels": str(labels_path.resolve()),
        "image_size": [width, height],
        "class_names": names,
        "xml_counts": dict(counts),
        "polygon_counts": dict(Counter(p["class"] for p in polygons)),
        "min_iou": min_iou,
        "min_extent": min_extent,
        "interpretation": "Bounding-box comparison only; polygon mask accuracy and remote training provenance are unverified.",
        "matches": matches,
        "unmatched_xml": [g for i, g in enumerate(truth) if i not in used_truth],
        "unmatched_polygons": [p for i, p in enumerate(polygons) if i not in used_polygons],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--names", required=True, nargs="+", help="Dataset names in class-ID order.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-iou", type=float, default=0.1)
    parser.add_argument("--min-extent", type=float, default=0.75)
    args = parser.parse_args()
    if args.out.resolve() in (args.xml.resolve(), args.labels.resolve()):
        parser.error("--out must not overwrite either annotation source.")
    report = audit(args.xml, args.labels, args.names, args.min_iou, args.min_extent)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "xml_counts": report["xml_counts"],
                "polygon_counts": report["polygon_counts"],
                "matched": len(report["matches"]),
                "unmatched_xml": len(report["unmatched_xml"]),
                "extent_review": sum(m["review_extent"] for m in report["matches"]),
                "report": str(args.out.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
