"""Evaluate saved predictions against one Pascal VOC box-annotation file.

Class-aware, confidence-ordered, one-to-one matching at IoU >= 0.5. VOC's
one-based inclusive boxes are converted to zero-based continuous xyxy. This is
single-image box evaluation, not segmentation IoU or independent validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REGIONS = {"left_MSC": (0.03, 0.20, 0.54, 0.90), "right_shadow": (0.64, 0.58, 0.85, 0.94)}


def read_annotation(path: Path, origin: int = 1) -> dict:
    root = ET.parse(path).getroot()
    width, height = int(root.findtext("size/width")), int(root.findtext("size/height"))
    if min(width, height) <= 0 or origin not in (0, 1):
        raise ValueError("Invalid image size or coordinate origin")
    objects = []
    for index, obj in enumerate(root.findall("object"), 1):
        node = obj.find("bndbox")
        if node is None:
            raise ValueError(f"Object {index} has no box")
        box = [float(node.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")]
        box[0] -= origin
        box[1] -= origin
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError(f"Invalid box {index}: {box}")
        if obj.findtext("difficult", "0") != "0":
            raise ValueError("Difficult annotations require an explicit ignore policy; this evaluator expects none.")
        objects.append({"id": index, "class_name": obj.findtext("name"), "box_xyxy": box})
    return {
        "filename": root.findtext("filename"),
        "width": width,
        "height": height,
        "annotation": str(path.resolve()),
        "coordinate_origin": origin,
        "objects": objects,
    }


def box_iou(left, right) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(annotations: list[dict], detections: list[dict], threshold: float = 0.5) -> dict[int, int]:
    """Map GT indices to prediction indices, allowing each box to match only once."""
    matched = {}
    for prediction in sorted(range(len(detections)), key=lambda i: -detections[i]["confidence"]):
        candidates = [
            i
            for i, gt in enumerate(annotations)
            if i not in matched and gt["class_name"] == detections[prediction]["class_name"]
        ]
        if not candidates:
            continue
        target = max(candidates, key=lambda i: box_iou(annotations[i]["box_xyxy"], detections[prediction]["box_xyxy"]))
        if box_iou(annotations[target]["box_xyxy"], detections[prediction]["box_xyxy"]) >= threshold:
            matched[target] = prediction
    return matched


def in_roi(box, bounds, width, height):
    x, y = (box[0] + box[2]) / (2 * width), (box[1] + box[3]) / (2 * height)
    return bounds[0] <= x < bounds[2] and bounds[1] <= y < bounds[3]


def evaluate(annotation: dict, report: dict, threshold: float = 0.5) -> tuple[list[dict], list[dict]]:
    gt, detections = annotation["objects"], report["detections"]
    matched = match_boxes(gt, detections, threshold)
    rows, missed = [], []
    classes = sorted({obj["class_name"] for obj in gt} | {d["class_name"] for d in detections})
    for region, bounds in {"full": (0, 0, 1.001, 1.001), **REGIONS}.items():
        for name in classes:
            selected = [
                i
                for i, obj in enumerate(gt)
                if obj["class_name"] == name
                and in_roi(obj["box_xyxy"], bounds, annotation["width"], annotation["height"])
            ]
            tp = sum(index in matched for index in selected)
            predicted = sum(d["class_name"] == name for d in detections)
            # Regional recall is defined by the GT center, using the same global one-to-one matches.
            # Precision is only reported for the full image to avoid boundary-assignment ambiguity.
            rows.append(
                {
                    "region": region,
                    "class": name,
                    "annotated": len(selected),
                    "matched": tp,
                    "missed": len(selected) - tp,
                    "recall": round(tp / len(selected), 4) if selected else "",
                    "predicted": predicted if region == "full" else "",
                    "unmatched_predictions": predicted - tp if region == "full" else "",
                    "precision": round(tp / predicted, 4) if region == "full" and predicted else "",
                }
            )
    for index, obj in enumerate(gt):
        if index in matched:
            continue
        same = [d for d in detections if d["class_name"] == obj["class_name"]]
        best = max(same, key=lambda d: box_iou(obj["box_xyxy"], d["box_xyxy"]), default=None)
        missed.append(
            {
                **obj,
                "best_same_class_iou": round(box_iou(obj["box_xyxy"], best["box_xyxy"]), 4) if best else 0,
                "best_same_class_score": best["confidence"] if best else 0,
                "region": "+".join(
                    name
                    for name, roi in REGIONS.items()
                    if in_roi(obj["box_xyxy"], roi, annotation["width"], annotation["height"])
                )
                or "other",
            }
        )
    return rows, missed


def evaluate_runs(annotation: dict, runs: list[Path], output: Path, threshold: float = 0.5) -> list[dict]:
    from PIL import Image

    rows, missed_all, evaluated = [], [], []
    checked_sources = set()
    for run in runs:
        jobs_path = run / "parameters.json"
        jobs = json.loads(jobs_path.read_text(encoding="utf-8")) if jobs_path.is_file() else []
        jobs = jobs if isinstance(jobs, list) else []
        for path in sorted(run.rglob("predictions.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            source = Path(report.get("source", ""))
            if source.name.casefold() != annotation["filename"].casefold():
                continue
            if source not in checked_sources:
                with Image.open(source) as image:
                    if image.size != (annotation["width"], annotation["height"]):
                        raise ValueError(f"Image dimensions disagree with XML: {source}")
                checked_sources.add(source)
            job = next((j for j in jobs if Path(j["output"]) in path.parents), {})
            identity = {
                "run": run.name,
                "group": job.get("group", "recall"),
                "version": job.get("version", report.get("version", path.parent.name)),
                "source": str(source),
                "prediction_file": str(path.resolve()),
            }
            scores, missed = evaluate(annotation, report, threshold)
            rows.extend({**identity, **score} for score in scores)
            missed_all.extend({**identity, **item} for item in missed)
            evaluated.append({**identity, "metrics": scores, "missed": missed})
    if not rows:
        raise ValueError("No prediction reports matched this annotation filename")
    output.mkdir(parents=True, exist_ok=False)
    (output / "annotation_snapshot.json").write_text(
        json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "evaluation.json").write_text(
        json.dumps(
            {
                "iou_threshold": threshold,
                "matching": "same-class confidence-ordered one-to-one",
                "scope": "single-image bbox evaluation; no segmentation ground truth",
                "results": evaluated,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for filename, values in (("summary.csv", rows), ("missed_annotations.csv", missed_all)):
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            if values:
                writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    lines = [
        "# XML 標記比對",
        "",
        f"標記：`{annotation['annotation']}`；{len(annotation['objects'])} 個框。",
        f"同類別依信心分數排序、一對一匹配，IoU ≥ {threshold:g}；VOC 起始座標使用 {annotation['coordinate_origin']}。",
        "XML 只有框，不能計算分割遮罩準確率。這是同一張圖片上的回測，不代表驗證集表現。",
        "區域召回依標記框中心位置分組；未匹配預測可能是誤檢、重複或定位不夠準。",
        "",
        "| 組別／版本 | 左 MSC D 命中／標記 | 右下 D 命中／標記 | 右下 R 命中／標記 | 全圖 D 命中／預測 | 全圖 R 命中／預測 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in evaluated:
        values = {(r["region"], r["class"]): r for r in result["metrics"]}
        cells = []
        for region, name in (
            ("left_MSC", "D"),
            ("right_shadow", "D"),
            ("right_shadow", "R"),
            ("full", "D"),
            ("full", "R"),
        ):
            metric = values.get((region, name), {"matched": 0, "annotated": 0, "predicted": 0})
            denominator = metric["predicted"] if region == "full" else metric["annotated"]
            cells.append(f"{metric['matched']} / {denominator}")
        lines.append(f"| {result['group']}/{result['version']} | " + " | ".join(cells) + " |")
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Evaluated {len(evaluated)} versions: {output}", flush=True)
    return rows


def compare_targets(annotation: dict, baseline_path: Path, candidate_path: Path, output: Path) -> Path:
    """Show annotation-centered crops of original and model outputs; never draw GT as predictions."""
    from PIL import Image

    from tools.compare_parameter_sweep import font_manager, plt

    before = json.loads(baseline_path.read_text(encoding="utf-8"))
    after = json.loads(candidate_path.read_text(encoding="utf-8"))
    if before["source"] != after["source"]:
        raise ValueError("Comparison predictions must refer to the same source")
    gt = annotation["objects"]
    matched_before, matched_after = match_boxes(gt, before["detections"]), match_boxes(gt, after["detections"])
    changed = [i for i in matched_after if i not in matched_before and gt[i]["class_name"] == "D"][:2]
    left = [
        i
        for i, obj in enumerate(gt)
        if i not in matched_after
        and obj["class_name"] == "D"
        and in_roi(obj["box_xyxy"], REGIONS["left_MSC"], annotation["width"], annotation["height"])
    ][:2]
    right = [
        i
        for i, obj in enumerate(gt)
        if obj["class_name"] == "D"
        and in_roi(obj["box_xyxy"], REGIONS["right_shadow"], annotation["width"], annotation["height"])
    ][:2]
    rust = [
        i
        for i, obj in enumerate(gt)
        if i not in matched_after
        and obj["class_name"] == "R"
        and in_roi(obj["box_xyxy"], REGIONS["right_shadow"], annotation["width"], annotation["height"])
    ][:1]
    selected = list(dict.fromkeys(changed + left + right + rust))
    if not selected:
        raise ValueError("No target annotations in the configured regions")
    font = Path("C:/Windows/Fonts/msjh.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    paths = [
        Path(before["source"]),
        next(baseline_path.parent.glob("*__all.jpg")),
        next(candidate_path.parent.glob("*__all.jpg")),
    ]
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    figure, axes = plt.subplots(len(selected), 3, figsize=(12, 2.5 * len(selected)), squeeze=False)
    for row, index in enumerate(selected):
        obj = gt[index]
        x1, y1, x2, y2 = obj["box_xyxy"]
        bounds = (
            max(0, int(x1) - 8),
            max(0, int(y1) - 8),
            min(annotation["width"], int(x2) + 8),
            min(annotation["height"], int(y2) + 8),
        )
        for column, (image, label) in enumerate(zip(images, ("原圖", "既有三視圖", "局部低門檻診斷"))):
            axis = axes[row, column]
            axis.imshow(image.crop(bounds), interpolation="nearest")
            state = ""
            if column:
                matched = matched_before if column == 1 else matched_after
                state = "｜框匹配成功" if index in matched else "｜框未匹配"
            axis.set_title(f"標記 #{obj['id']} {obj['class_name']}｜{label}{state}", fontsize=10)
            axis.axis("off")
    figure.suptitle("指定標記附近的原圖與預測｜文字在圖外，未將標記框畫成預測", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/segment_shadow/annotation_evaluation")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--coordinate-origin", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    if not 0 < args.iou <= 1:
        parser.error("iou must be in (0,1]")
    annotation = read_annotation(args.annotation, args.coordinate_origin)
    output = args.output.resolve() / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    evaluate_runs(annotation, [p.resolve() for p in args.run_dir], output, args.iou)


if __name__ == "__main__":
    main()
