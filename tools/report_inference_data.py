r"""Report saved segmentation outputs: train, val, and train + val.

Run with C:\Users\USER\anaconda3\python.exe tools/report_inference_data.py
Use --results for another inference folder and --annotations for VOC XML folders.
No model is loaded. XML boxes evaluate saved boxes, not segmentation boundaries.
AP uses 101 recall samples and IoU 0.50:0.05:0.95 on the retained predictions.
Dependencies: NumPy and Matplotlib. --open opens the completed HTML report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import webbrowser
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.evaluate_shadow_annotations import box_iou, read_annotation

# Change these defaults or pass --results / --annotations on the command line.
DEFAULT_RESULTS = REPO / "tools/runs/segment_by_class/shadow_diagnostics/run_20261002_232129_326448/01_last0912"
DEFAULT_ANNOTATIONS = Path("C:/Users/USER/Desktop/已標記/labels")
DEFAULT_OUTPUT = REPO / "tools/runs/inference_reports"
GROUPS = {"train": ("train",), "val": ("val",), "train_and_val": ("train", "val")}
TITLES = {"train": "只有 train", "val": "只有 val", "train_and_val": "綜合 train + val"}
AP_IOUS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))


def canonical_split(name):
    match = re.fullmatch(r"(train|val|valid|validation)(?:[\s_-].*|[^\x00-\x7f].*)?", name.casefold())
    return ("train" if match.group(1) == "train" else "val") if match else None


def split_relative(path):
    """Return canonical split and the tail after its closest directory."""
    for index in range(len(path.parts) - 2, -1, -1):
        split = canonical_split(path.parts[index])
        if split:
            return split, Path(*path.parts[index + 1 :])
    return None, None


class AnnotationIndex:
    def __init__(self, roots):
        self.by_relative, self.by_filename = defaultdict(set), defaultdict(set)
        self.cached, self.scan_errors = {}, []
        self.files = set()
        for root in roots:
            root = Path(root).expanduser().resolve()
            if not root.exists():
                self.scan_errors.append({"annotation": str(root), "error": "Annotation path does not exist"})
                continue
            paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix.casefold() == ".xml")
            for path in paths:
                if path.suffix.casefold() != ".xml" or path in self.files:
                    continue
                self.files.add(path)
                split, tail = split_relative(path)
                if tail is not None:
                    self.by_relative[(split, tail.as_posix().casefold())].add(path)
                # Index by the actual XML filename; validate its declared image later.
                self.by_filename[(split, path.stem.casefold())].add(path)

    def resolve(self, source, split, width, height, origin):
        _, tail = split_relative(source)
        exact = self.by_relative.get((split, tail.with_suffix(".xml").as_posix().casefold()), set()) if tail else set()
        candidates = exact or self.by_filename.get((split, source.stem.casefold()), set())
        candidates = candidates or self.by_filename.get((None, source.stem.casefold()), set())
        if not candidates:
            return "missing_annotation", None, [], "No corresponding XML"
        if len(candidates) != 1:
            return "ambiguous_annotation", None, [], "Multiple XMLs: " + " | ".join(map(str, sorted(candidates)))
        path = next(iter(candidates))
        try:
            key = (path, origin)
            if key not in self.cached:
                self.cached[key] = read_annotation(path, origin)
            annotation = self.cached[key]
            if (annotation["width"], annotation["height"]) != (width, height):
                raise ValueError("XML dimensions do not match the inference image")
            if (annotation["filename"] or "").casefold() != source.name.casefold():
                raise ValueError(f"XML filename {annotation['filename']!r} does not match {source.name!r}")
            truth = [{"class": obj["class_name"], "box": obj["box_xyxy"]} for obj in annotation["objects"]]
            return "ok", path, truth, ""
        except (OSError, ValueError, TypeError, ET.ParseError) as error:
            # Bad XMLs affect their image's evaluation only, not prediction statistics.
            return "invalid_annotation", path, [], f"{type(error).__name__}: {error}"


def read_predictions(path, width, height, expected_counts):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    predictions = []
    for entry in data["detections"]:
        box = list(map(float, entry["box_xyxy"]))
        score = float(entry["score"])
        if len(box) != 4 or not all(math.isfinite(value) for value in [*box, score]):
            raise ValueError("Non-finite prediction or invalid box length")
        if not (
            0 <= score <= 1
            and -0.001 <= box[0] < box[2] <= width + 0.001
            and -0.001 <= box[1] < box[3] <= height + 0.001
        ):
            raise ValueError("Prediction score or box is outside its valid range")
        box = [max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3])]
        predictions.append({"class": str(entry["class"]), "score": score, "box": box})
    counts = Counter(prediction["class"] for prediction in predictions)
    if counts != Counter({name: count for name, count in expected_counts.items() if count}):
        raise ValueError("predictions.json counts differ from prediction_metrics.json")
    return predictions


def match_predictions(predictions, truth, threshold):
    """Confidence-ordered, class-aware, one-to-one box matching per image."""
    used, matched = set(), [False] * len(predictions)
    for index in sorted(range(len(predictions)), key=lambda index: -predictions[index]["score"]):
        prediction = predictions[index]
        candidates = [
            (box_iou(prediction["box"], target["box"]), target_index)
            for target_index, target in enumerate(truth)
            if target_index not in used and target["class"] == prediction["class"]
        ]
        iou, target_index = max(candidates, default=(0.0, -1))
        if target_index >= 0 and iou >= threshold:
            used.add(target_index)
            matched[index] = True
    return matched


def average_precision(scored_matches, gt_count):
    """101-point precision-envelope average; classes without GT are undefined."""
    if not gt_count:
        return None
    if not scored_matches:
        return 0.0
    ranked = sorted(scored_matches, key=lambda pair: -pair[0])
    true_positive = np.cumsum([int(pair[1]) for pair in ranked])
    recall = true_positive / gt_count
    precision = true_positive / np.arange(1, len(ranked) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    positions = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    sampled = np.zeros(101)
    valid = positions < len(envelope)
    sampled[valid] = envelope[positions[valid]]
    return float(sampled.mean())


def ratios(tp, fp, fn):
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def load_records(results, annotations, versions=None, origin=1, eval_conf=0.0, match_iou=0.5):
    records, issues = [], []
    seen = set()
    paths = sorted(results.rglob("prediction_metrics.json"))
    for path in paths:
        version = path.parent.name
        if versions and version not in versions:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            source = Path(data["source"])
            split, _ = split_relative(source)
            result_split = next((parent for parent in path.parents if canonical_split(parent.name)), None)
            if result_split:
                output_split = canonical_split(result_split.name)
                if split and split != output_split:
                    raise ValueError("Output split conflicts with original image split")
                split = output_split
                model = result_split.parent.name
            else:
                relative = path.relative_to(results)
                model = relative.parts[0] if len(relative.parts) >= 4 else results.name
            if split not in {"train", "val"}:
                raise ValueError("Cannot identify train/val split")
            width, height = int(data["width"]), int(data["height"])
            if min(width, height) <= 0:
                raise ValueError("Invalid image dimensions")
            classes = {}
            for class_id, entry in data["classes"].items():
                name, count, pixels = str(entry["name"]), entry["count"], entry["union_pixels"]
                if (
                    name in classes
                    or not isinstance(count, int)
                    or count < 0
                    or not isinstance(pixels, int)
                    or not 0 <= pixels <= width * height
                ):
                    raise ValueError("Invalid or duplicate class statistics")
                classes[name] = {"id": int(class_id), "count": count, "union_pixels": pixels}
            key = (model, version, str(source).casefold())
            if key in seen:
                raise ValueError("Duplicate model/version/source; refusing to count it twice")
            seen.add(key)
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append({"metrics_file": str(path), "status": "invalid_metrics", "error": str(error)})
            continue
        status, annotation_path, truth, message = annotations.resolve(source, split, width, height, origin)
        predictions, flags = [], {}
        if status == "ok" and any(target["class"] not in classes for target in truth):
            status, message = "invalid_annotation", "XML class name is not in the model class table"
        if status == "ok":
            try:
                predictions = read_predictions(
                    path.with_name("predictions.json"),
                    width,
                    height,
                    {name: entry["count"] for name, entry in classes.items()},
                )
                predictions = [prediction for prediction in predictions if prediction["score"] >= eval_conf]
                flags = {iou: match_predictions(predictions, truth, iou) for iou in set(AP_IOUS) | {match_iou}}
            except (OSError, ValueError, KeyError, TypeError) as error:
                status, message = "invalid_predictions", str(error)
        record = {
            "model": model,
            "version": version,
            "split": split,
            "source": str(source),
            "width": width,
            "height": height,
            "classes": classes,
            "regions": data.get("regions", {}),
            "metrics_file": str(path),
            "annotation": str(annotation_path) if annotation_path else "",
            "evaluation_status": status,
            "evaluation_error": message,
            "truth": truth,
            "predictions": predictions,
            "flags": flags,
        }
        records.append(record)
        if status != "ok":
            issues.append(
                {
                    "model": model,
                    "version": version,
                    "split": split,
                    "source": str(source),
                    "metrics_file": str(path),
                    "annotation": record["annotation"],
                    "status": status,
                    "error": message,
                }
            )
    return records, issues


def summarize(records, group, match_iou=0.5):
    eligible = [record for record in records if record["evaluation_status"] == "ok"]
    names = sorted(
        {name for record in records for name in record["classes"]},
        key=lambda name: min(record["classes"][name]["id"] for record in records if name in record["classes"]),
    )
    total_pixels = sum(record["width"] * record["height"] for record in records)
    detections = sum(sum(entry["count"] for entry in record["classes"].values()) for record in records)
    class_rows = []
    for name in names:
        counts = [record["classes"].get(name, {}).get("count", 0) for record in records]
        union_pixels = sum(record["classes"].get(name, {}).get("union_pixels", 0) for record in records)
        gt = sum(sum(target["class"] == name for target in record["truth"]) for record in eligible)
        predicted = sum(sum(prediction["class"] == name for prediction in record["predictions"]) for record in eligible)
        tp = sum(
            sum(
                prediction["class"] == name and hit
                for prediction, hit in zip(record["predictions"], record["flags"][match_iou])
            )
            for record in eligible
        )
        aps = [
            average_precision(
                [
                    (prediction["score"], hit)
                    for record in eligible
                    for prediction, hit in zip(record["predictions"], record["flags"][iou])
                    if prediction["class"] == name
                ],
                gt,
            )
            for iou in AP_IOUS
        ]
        evaluation = (
            {
                "gt_objects": gt,
                "evaluated_predictions": predicted,
                "tp": tp,
                "fp": predicted - tp,
                "fn": gt - tp,
                **ratios(tp, predicted - tp, gt - tp),
                "box_ap50": aps[0],
                "box_ap50_95": sum(aps) / len(aps) if gt else None,
            }
            if eligible
            else {
                field: None
                for field in (
                    "gt_objects",
                    "evaluated_predictions",
                    "tp",
                    "fp",
                    "fn",
                    "precision",
                    "recall",
                    "f1",
                    "box_ap50",
                    "box_ap50_95",
                )
            }
        )
        class_rows.append(
            {
                "class": name,
                "detections": sum(counts),
                "images_with_class": sum(count > 0 for count in counts),
                "images_with_class_percent": 100 * sum(count > 0 for count in counts) / len(records) if records else 0,
                "detections_per_image": sum(counts) / len(records) if records else 0,
                "union_pixels": union_pixels,
                "weighted_coverage_percent": 100 * union_pixels / total_pixels if total_pixels else 0,
                **evaluation,
            }
        )
    tp = sum(row["tp"] or 0 for row in class_rows)
    fp = sum(row["fp"] or 0 for row in class_rows)
    fn = sum(row["fn"] or 0 for row in class_rows)
    defined_aps = [row for row in class_rows if row["box_ap50"] is not None]
    summary = {
        "group": group,
        "image_count": len(records),
        "train_images": sum(record["split"] == "train" for record in records),
        "val_images": sum(record["split"] == "val" for record in records),
        "images_with_predictions": sum(
            any(entry["count"] for entry in record["classes"].values()) for record in records
        ),
        "empty_prediction_images": sum(
            not any(entry["count"] for entry in record["classes"].values()) for record in records
        ),
        "detections": detections,
        "detections_per_image": detections / len(records) if records else 0,
        "evaluated_images": len(eligible),
        "excluded_from_evaluation": len(records) - len(eligible),
        "evaluation_coverage_percent": 100 * len(eligible) / len(records) if records else 0,
        "evaluation_iou": match_iou,
        "tp": tp if eligible else None,
        "fp": fp if eligible else None,
        "fn": fn if eligible else None,
        **(ratios(tp, fp, fn) if eligible else {field: None for field in ("precision", "recall", "f1")}),
        "box_map50": sum(row["box_ap50"] for row in defined_aps) / len(defined_aps) if defined_aps else None,
        "box_map50_95": sum(row["box_ap50_95"] for row in defined_aps) / len(defined_aps) if defined_aps else None,
    }
    return {"summary": summary, "classes": class_rows}


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def display(value, percent=False):
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value * 100:.2f}%" if percent else f"{value:,.2f}"


def html_table(rows, columns):
    header = "".join(f"<th>{escape(label)}</th>" for _, label, _ in columns)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{escape(display(row.get(field), percent) if isinstance(row.get(field), (int, float)) or row.get(field) is None else str(row[field]))}</td>"
            for field, _, percent in columns
        )
        + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


CSS = """
body{margin:0;background:#f3f5f9;color:#172238;font:16px/1.6 'Microsoft JhengHei',sans-serif}
main{max-width:1280px;margin:auto;padding:32px 24px}h1{font-size:30px}h2{margin-top:32px}h3{margin-top:28px}
.tabs{display:flex;gap:10px;margin:22px 0;flex-wrap:wrap}button,.link{border:1px solid #cad2df;border-radius:8px;background:white;padding:10px 18px;cursor:pointer;color:#172238;font:inherit;text-decoration:none}
button.active{background:#174eae;color:white;border-color:#174eae}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px}.card{background:white;border:1px solid #dee4ee;border-radius:10px;padding:16px}.card strong{display:block;font-size:27px}.note{background:#eaf0fc;padding:16px;border-radius:8px}.table-wrap{overflow-x:auto;background:white;border:1px solid #dee4ee;border-radius:8px}table{border-collapse:collapse;width:100%;white-space:nowrap}th,td{padding:10px 13px;border-bottom:1px solid #e5e9f0;text-align:right}th{background:#eef2f8}th:first-child,td:first-child{text-align:left}img{width:100%;height:auto;background:white;border-radius:10px}.muted{color:#54637a}.tab-panel[hidden]{display:none}a{color:#174eae}
"""


def panel_html(group, series, prefix=""):
    pieces = [f"<h2>{TITLES[group]}</h2>"]
    if not series:
        return "".join(pieces) + "<p>此組沒有可讀取的推論結果。</p>"
    for index, report in enumerate(series, 1):
        summary = report["summary"]
        pieces.append(f"<h3>{escape(report['model'])} ／ {escape(report['version'])}</h3>")
        cards = [
            ("推論圖片", str(summary["image_count"])),
            ("偵測數量", str(summary["detections"])),
            ("每張平均偵測", f"{summary['detections_per_image']:.2f}"),
            ("實際評估圖片", f"{summary['evaluated_images']} / {summary['image_count']}"),
            ("框 Precision（micro）", display(summary["precision"], True)),
            ("框 Recall（micro）", display(summary["recall"], True)),
            ("框 F1（micro）", display(summary["f1"], True)),
            ("框 mAP50", display(summary["box_map50"], True)),
        ]
        pieces.append(
            '<div class="cards">'
            + "".join(
                f'<div class="card">{escape(label)}<strong>{escape(value)}</strong></div>' for label, value in cards
            )
            + "</div>"
        )
        pieces.append(
            f"<p class='muted'>train {summary['train_images']} 張、val {summary['val_images']} 張；"
            f"無偵測 {summary['empty_prediction_images']} 張；未納入評估 {summary['excluded_from_evaluation']} 張；"
            f"評估涵蓋率 {summary['evaluation_coverage_percent']:.2f}%；"
            f"框 mAP50–95 {display(summary['box_map50_95'], True)}。</p>"
        )
        pieces.append(
            "<h3>推論統計（涵蓋此組全部推論圖片）</h3>"
            + html_table(
                report["classes"],
                [
                    ("class", "類別", False),
                    ("detections", "偵測數", False),
                    ("images_with_class", "有此類別的圖片", False),
                    ("images_with_class_percent", "圖片占比（%）", False),
                    ("detections_per_image", "每張平均偵測", False),
                    ("weighted_coverage_percent", "遮罩覆蓋率（%）", False),
                ],
            )
        )
        pieces.append(
            f"<h3>XML 框評估（IoU ≥ {summary['evaluation_iou']:.2f}）</h3>"
            + html_table(
                report["classes"],
                [
                    ("class", "類別", False),
                    ("gt_objects", "GT", False),
                    ("evaluated_predictions", "評估預測數", False),
                    ("tp", "TP", False),
                    ("fp", "FP", False),
                    ("fn", "FN", False),
                    ("precision", "Precision", True),
                    ("recall", "Recall", True),
                    ("f1", "F1", True),
                    ("box_ap50", "框 AP50", True),
                    ("box_ap50_95", "框 AP50–95", True),
                ],
            )
        )
        pieces.append(
            f'<p><a href="{prefix}class_summary.csv">下載類別 CSV</a> · <a href="{prefix}images.csv">逐張資料</a> · <a href="{prefix}evaluation_exclusions.csv">未評估清單</a></p>'
        )
        pieces.append(f'<img src="{prefix}series_{index:02d}.png" alt="{escape(TITLES[group])}統計與評估圖表">')
    return "".join(pieces)


def write_html(path, content, title, config, tabs=False):
    notes = (
        "推論統計包含全部可讀取的推論結果；評估只包含檔名、尺寸及資料集對應正確的 XML。"
        "沒有標記的圖片不當作背景，缺少數值以 — 表示。遮罩覆蓋率為同類遮罩聯集像素／全部圖片像素，類別之間可能重疊。"
        "XML 只有矩形框，因此這裡呈現框 Precision／Recall／F1／AP，沒有計算分割 IoU 或分割 mAP。"
        "框 AP 使用已保存、已篩選的預測，以 101 個 recall 點計算；mAP50–95 使用 IoU 0.50 到 0.95、間距 0.05，"
        "僅平均有 GT 的類別。這不是重新執行 model.val() 的成績。"
        "綜合組合併原始 TP／FP／FN，並重新排序全部預測計算 AP；train 與 val 的模型／版本分別呈現。"
    )
    script = (
        """<script>function selectGroup(key){document.querySelectorAll('.tab-panel').forEach(p=>p.hidden=p.id!==key);document.querySelectorAll('[data-group]').forEach(b=>b.classList.toggle('active',b.dataset.group===key));}selectGroup('train');</script>"""
        if tabs
        else ""
    )
    notice = "推論統計包含全部可讀取的圖片；框評估僅使用通過 XML 對應檢查的圖片。各頁均顯示實際評估張數。"
    path.write_text(
        f'<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{CSS}</style></head><body><main><h1>{escape(title)}</h1><p class="muted">建立時間：{escape(config["created_at"])} ／ 評估信心門檻：{config["eval_conf"]:.3f}</p><div class="note">{notice}</div>{content}<details><summary>數據定義與評估方式</summary><p>{notes}</p></details></main>{script}</body></html>',
        encoding="utf-8",
    )


def plot_series(report, path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    font = Path("C:/Windows/Fonts/msjh.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    rows, summary = report["classes"], report["summary"]
    names = [row["class"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].bar(names, [row["detections"] for row in rows], color="#3687ca")
    axes[0, 0].set_title("各類偵測數量")
    axes[0, 1].bar(names, [row["images_with_class_percent"] for row in rows], color="#409c89")
    axes[0, 1].set_title("有此類別的圖片占比（%）")
    axes[0, 1].set_ylim(0, 105)
    axes[1, 0].bar(names, [row["weighted_coverage_percent"] for row in rows], color="#8b77bd")
    axes[1, 0].set_title("各類遮罩覆蓋率（%）")
    x = np.arange(len(names))
    for offset, field, label, color in (
        (-1, "precision", "Precision", "#3687ca"),
        (0, "recall", "Recall", "#ed986a"),
        (1, "f1", "F1", "#409c89"),
    ):
        axes[1, 1].bar(
            x + offset * 0.24,
            [row[field] * 100 if row[field] is not None else np.nan for row in rows],
            width=0.24,
            label=label,
            color=color,
        )
    axes[1, 1].set_xticks(x, names)
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].set_title(f"框評估（%，{summary['evaluated_images']}/{summary['image_count']} 張）")
    axes[1, 1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9, frameon=False)
    if not summary["evaluated_images"]:
        axes[1, 1].text(0.5, 0.5, "沒有可評估的標記", ha="center", transform=axes[1, 1].transAxes)
    for axis in axes.flat:
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"{TITLES[summary['group']]}｜{report['model']}｜{report['version']}\n推論 {summary['image_count']} 張；框 IoU >= {summary['evaluation_iou']:.2f}"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_reports(records, issues, output, config):
    output.mkdir(parents=True, exist_ok=False)
    all_reports = {}
    for group, splits in GROUPS.items():
        folder = output / group
        folder.mkdir()
        selected = [record for record in records if record["split"] in splits]
        grouped = defaultdict(list)
        for record in selected:
            grouped[(record["model"], record["version"])].append(record)
        series, class_rows, image_rows, region_rows = [], [], [], []
        for index, ((model, version), subset) in enumerate(sorted(grouped.items()), 1):
            report = {"model": model, "version": version, **summarize(subset, group, config["iou"])}
            series.append(report)
            class_rows.extend({"model": model, "version": version, **row} for row in report["classes"])
            plot_series(report, folder / f"series_{index:02d}.png")
        for record in selected:
            total = sum(entry["count"] for entry in record["classes"].values())
            image_rows.append(
                {
                    field: record[field]
                    for field in (
                        "model",
                        "version",
                        "split",
                        "source",
                        "width",
                        "height",
                        "metrics_file",
                        "annotation",
                        "evaluation_status",
                        "evaluation_error",
                    )
                }
                | {
                    "detections": total,
                    **{f"{name}_count": entry["count"] for name, entry in record["classes"].items()},
                }
            )
            for region, classes in record["regions"].items():
                for name, entry in classes.items():
                    region_rows.append(
                        {
                            "model": record["model"],
                            "version": record["version"],
                            "split": record["split"],
                            "source": record["source"],
                            "region": region,
                            "class": name,
                            **entry,
                        }
                    )
        summaries = [{"model": report["model"], "version": report["version"], **report["summary"]} for report in series]
        exclusions = [row for row in image_rows if row["evaluation_status"] != "ok"]
        image_fields = list(dict.fromkeys(key for row in image_rows for key in row)) or ["source", "evaluation_status"]
        write_csv(
            folder / "summary.csv",
            summaries,
            list(summaries[0]) if summaries else ["model", "version", "image_count", "evaluated_images"],
        )
        write_csv(
            folder / "class_summary.csv",
            class_rows,
            list(class_rows[0]) if class_rows else ["model", "version", "class", "detections"],
        )
        write_csv(folder / "images.csv", image_rows, image_fields)
        write_csv(
            folder / "region_summary.csv", region_rows, None if region_rows else ["source", "region", "class", "count"]
        )
        write_csv(folder / "evaluation_exclusions.csv", exclusions, image_fields)
        (folder / "summary.json").write_text(
            json.dumps({"group": group, "series": series}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_html(folder / "report.html", panel_html(group, series), TITLES[group], config)
        all_reports[group] = series
        print(
            f"{group}: {len(selected)} result(s), evaluated {sum(record['evaluation_status'] == 'ok' for record in selected)}",
            flush=True,
        )
    content = (
        '<nav class="tabs">'
        + "".join(
            f'<button type="button" data-group="{group}" onclick="selectGroup(\'{group}\')">{TITLES[group]}</button>'
            for group in GROUPS
        )
        + "</nav>"
    )
    content += "".join(
        f'<section class="tab-panel" id="{group}">{panel_html(group, all_reports[group], group + "/")}</section>'
        for group in GROUPS
    )
    write_html(output / "index.html", content, "推論統計與標註評估", config, tabs=True)
    write_csv(output / "data_quality.csv", issues, None if issues else ["source", "status", "error"])
    (output / "settings.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return all_reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results", type=Path, default=DEFAULT_RESULTS, help="推論結果模型資料夾或包含多個模型的 run 資料夾"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        nargs="+",
        default=[DEFAULT_ANNOTATIONS],
        help="VOC XML 標記資料夾或個別 XML；遞迴搜尋",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="報表存放位置；每次新增 report_時間 資料夾")
    parser.add_argument("--versions", nargs="+", help="只統計指定版本；預設全部版本，各版本分開呈現")
    parser.add_argument("--iou", type=float, default=0.5, help="Precision/Recall/F1 框匹配 IoU 門檻，預設 0.5")
    parser.add_argument("--eval-conf", type=float, default=0.0, help="評估額外信心門檻；0 表示使用全部已保存預測")
    parser.add_argument("--xml-origin", type=int, choices=(0, 1), default=1, help="VOC 框左上角座標起點，預設 1")
    parser.add_argument("--open", action="store_true", help="產生完成後在瀏覽器開啟三組報表")
    args = parser.parse_args(argv)
    if not 0 < args.iou <= 1 or not 0 <= args.eval_conf <= 1:
        parser.error("IoU must be in (0, 1]; confidence must be in [0, 1]")
    results = args.results.expanduser().resolve()
    if not results.is_dir():
        parser.error(f"Results folder does not exist: {results}")
    annotations = AnnotationIndex(args.annotations)
    records, issues = load_records(results, annotations, args.versions, args.xml_origin, args.eval_conf, args.iou)
    issues.extend({"status": "annotation_scan_error", **row} for row in annotations.scan_errors)
    if not records:
        parser.error("No valid train/val prediction_metrics.json files found")
    created = datetime.now(timezone(timedelta(hours=8)))
    output = args.output.expanduser().resolve() / created.strftime("report_%Y%m%d_%H%M%S_%f")
    config = {
        "created_at": created.isoformat(),
        "results": str(results),
        "annotations": [str(path.expanduser().resolve()) for path in args.annotations],
        "annotation_files": len(annotations.files),
        "iou": args.iou,
        "eval_conf": args.eval_conf,
        "xml_origin": args.xml_origin,
        "versions": args.versions,
        "evaluation_type": "saved_prediction_boxes_vs_VOC_XML",
        "ap_method": "101_recall_points_precision_envelope",
        "ap_iou_thresholds": AP_IOUS,
    }
    generate_reports(records, issues, output, config)
    print(f"Report: {output / 'index.html'}", flush=True)
    if args.open:
        webbrowser.open((output / "index.html").as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
