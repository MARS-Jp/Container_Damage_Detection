r"""Export all nine metrics for every image, class, and dataset split.

Choose an evaluation basis explicitly:
  python tools/report_nine_metrics.py --mode boxes --open
  python tools/report_nine_metrics.py --mode masks --mask-annotations MASK_FOLDER

boxes = XML rectangle area versus predicted rectangle area (not segmentation).
masks = true ground-truth masks/polygons versus saved prediction_masks.npz.
Missing inputs and undefined ratios are NA; all source images remain in CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.pixel_metrics import COUNT_COLUMNS, METRIC_COLUMNS, confusion_counts, rasterize_boxes, summarize_counts
from tools.report_inference_data import (
    CSS,
    DEFAULT_ANNOTATIONS,
    DEFAULT_RESULTS,
    GROUPS,
    TITLES,
    AnnotationIndex,
    html_table,
    load_records,
    split_relative,
    write_csv,
)

DEFAULT_OUTPUT = REPO / "tools/runs/nine_metric_reports"
BASES = {"boxes": "框區域像素評估（非分割精度）", "masks": "真實分割遮罩像素評估"}
EMPTY_VALUES = {field: None for field in (*COUNT_COLUMNS, *METRIC_COLUMNS)}
CONTEXT_COLUMNS = (
    "metric_scope",
    "macro_scope",
    "gt_positive_pixels",
    "predicted_positive_pixels",
    "always_background_accuracy",
    "accuracy_gain_over_background",
    "gt_positive_images",
    "predicted_positive_images",
)


def metric_context(values, scope, gt_images=None, predicted_images=None):
    """Expose support and the trivial negative-only baseline alongside scores."""
    tp, fp, fn, tn = (values[key] for key in COUNT_COLUMNS)
    valid = tp is not None
    total = tp + fp + fn + tn if valid else 0
    background = (fp + tn) / total if total else None
    return {
        "metric_scope": scope,
        "macro_scope": "not_applicable_single_class"
        if scope.endswith("class")
        else "foreground_classes_with_nonzero_union",
        "gt_positive_pixels": tp + fn if valid else None,
        "predicted_positive_pixels": tp + fp if valid else None,
        "always_background_accuracy": background,
        "accuracy_gain_over_background": values["Accuracy"] - background if background is not None else None,
        "gt_positive_images": gt_images,
        "predicted_positive_images": predicted_images,
    }


def load_mask_archive(path, names, height, width, confidence=None):
    """Read lossless instance/per-class masks with pickle disabled."""
    with np.load(path, allow_pickle=False) as archive:
        if "packed_masks" in archive:
            packed = archive["packed_masks"]
            if packed.dtype != np.uint8 or packed.ndim != 3 or packed.shape[1:] != (height, (width + 7) // 8):
                raise ValueError("Invalid packed mask array dimensions/type")
            if int(archive["width"]) != width or int(archive["height"]) != height:
                raise ValueError("Mask archive dimensions do not match the inference image")
            masks = np.unpackbits(packed, axis=-1, count=width).astype(bool)
        else:
            masks = archive["masks"]
            if masks.dtype != np.bool_ or masks.ndim != 3 or masks.shape[1:] != (height, width):
                raise ValueError("masks must be bool NxHxW in original image coordinates")
        if "class_names" not in archive:
            raise ValueError("Mask archives must declare class_names")
        class_names = archive["class_names"]
        if class_names.ndim != 1 or len(class_names) != len(masks):
            raise ValueError("class_names and masks lengths differ")
        class_names = [str(value) for value in class_names]
        if any(name not in names for name in class_names):
            raise ValueError("Mask archive uses unknown class names")
        if confidence is not None:
            if "scores" not in archive:
                raise ValueError("Prediction masks must retain scores")
            scores = archive["scores"]
            if scores.shape != (len(masks),) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError("Invalid prediction mask scores")
            keep = scores >= confidence
        else:
            keep = np.ones(len(masks), dtype=bool)
        output = {name: np.zeros((height, width), dtype=bool) for name in names}
        for name, mask, selected in zip(class_names, masks, keep):
            if selected:
                output[name] |= mask
        return output


def load_polygon_labels(path, classes, height, width):
    """YOLO normalized segmentation polygons; empty existing TXT is background."""
    try:
        import cv2
    except ImportError as error:
        raise ValueError("YOLO polygon labels require OpenCV; use the yolov11 Python environment") from error
    by_id = {entry["id"]: name for name, entry in classes.items()}
    masks = {name: np.zeros((height, width), dtype=np.uint8) for name in classes}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        values = np.asarray([float(value) for value in line.split()])
        if len(values) < 7 or len(values) % 2 != 1 or not np.isfinite(values).all():
            raise ValueError(f"Invalid segmentation polygon on line {line_number}; box TXT is not accepted")
        if values[0] != int(values[0]) or int(values[0]) not in by_id or np.any((values[1:] < 0) | (values[1:] > 1)):
            raise ValueError(f"Unknown class/out-of-range polygon on line {line_number}")
        points = values[1:].reshape(-1, 2)
        if len(np.unique(points, axis=0)) < 3 or cv2.contourArea(points.astype(np.float32)) <= 0:
            raise ValueError(f"Degenerate polygon on line {line_number}")
        pixels = (points * [width, height]).astype(np.int32)
        cv2.fillPoly(masks[by_id[int(values[0])]], [pixels], 1)
    return {name: mask.astype(bool) for name, mask in masks.items()}


class MaskIndex:
    def __init__(self, roots):
        self.relative, self.filename = defaultdict(set), defaultdict(set)
        for root in roots:
            root = Path(root).expanduser().resolve()
            if not root.exists():
                raise FileNotFoundError(f"Missing ground-truth mask folder: {root}")
            paths = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
            for path in paths:
                if path.suffix.casefold() not in {".npz", ".txt"}:
                    continue
                split, tail = split_relative(path)
                if tail:
                    self.relative[(split, tail.with_suffix("").as_posix().casefold())].add(path)
                self.filename[(split, path.stem.casefold())].add(path)

    def resolve(self, source, split):
        _, tail = split_relative(source)
        exact = self.relative.get((split, tail.with_suffix("").as_posix().casefold()), set()) if tail else set()
        candidates = exact or self.filename.get((split, source.stem.casefold()), set())
        candidates = candidates or self.filename.get((None, source.stem.casefold()), set())
        if not candidates:
            raise FileNotFoundError("Missing corresponding ground-truth mask/polygon label")
        if len(candidates) != 1:
            raise ValueError("Ambiguous ground-truth masks: " + " | ".join(map(str, sorted(candidates))))
        return next(iter(candidates))


def evaluate_images(records, mode, mask_roots=(), confidence=0.0):
    mask_index = MaskIndex(mask_roots) if mode == "masks" else None
    for index, record in enumerate(records, 1):
        names, height, width = list(record["classes"]), record["height"], record["width"]
        record["pixel_counts"] = None
        record["pixel_truth"] = record["annotation"] if mode == "boxes" else ""
        try:
            if mode == "boxes":
                if record["evaluation_status"] != "ok":
                    raise ValueError(record["evaluation_status"] + ": " + record["evaluation_error"])
                truth = rasterize_boxes(record["truth"], names, height, width)
                prediction = rasterize_boxes(record["predictions"], names, height, width)
            else:
                ground_truth = mask_index.resolve(Path(record["source"]), record["split"])
                record["pixel_truth"] = str(ground_truth)
                truth = (
                    load_polygon_labels(ground_truth, record["classes"], height, width)
                    if ground_truth.suffix.casefold() == ".txt"
                    else load_mask_archive(ground_truth, names, height, width)
                )
                prediction_path = Path(record["metrics_file"]).with_name("prediction_masks.npz")
                prediction = load_mask_archive(prediction_path, names, height, width, confidence)
            counts = {name: confusion_counts(prediction[name], truth[name]) for name in names}
            overall, by_class = summarize_counts(counts)
            record.update(
                pixel_counts=counts, pixel_overall=overall, pixel_classes=by_class, pixel_status="ok", pixel_reason=""
            )
            del prediction, truth
        except (OSError, ValueError, KeyError, TypeError) as error:
            record.update(
                pixel_overall=dict(EMPTY_VALUES),
                pixel_classes={name: dict(EMPTY_VALUES) for name in names},
                pixel_status="not_evaluated",
                pixel_reason=f"{type(error).__name__}: {error}",
            )
        if index % 200 == 0:
            print(f"Pixel evaluation: {index}/{len(records)}", flush=True)


def pooled_metrics(records):
    eligible = [record for record in records if record["pixel_counts"] is not None]
    names = list(dict.fromkeys(name for record in records for name in record["classes"]))
    if not eligible:
        return dict(EMPTY_VALUES), {name: dict(EMPTY_VALUES) for name in names}
    pooled = {
        name: {
            field: sum(record["pixel_counts"].get(name, {}).get(field, 0) for record in eligible)
            for field in COUNT_COLUMNS
        }
        for name in names
    }
    return summarize_counts(pooled)


def write_metric_csv(path, rows, fields):
    """Keep every column and source row; represent undefined values explicitly."""
    output = []
    for row in rows:
        missing = [field for field in METRIC_COLUMNS if row[field] is None]
        reason = ""
        if missing:
            if row.get("pixel_status") == "not_evaluated":
                reason = row["pixel_reason"]
            elif row.get("evaluated_images") == 0:
                reason = "No valid evaluation inputs"
            else:
                macro_fields = [
                    field
                    for field in missing
                    if field in {"mIoU", "Mean Dice"} and row.get("macro_scope") == "not_applicable_single_class"
                ]
                undefined = [field for field in missing if field not in macro_fields]
                reasons = []
                if macro_fields:
                    reasons.append("Not applicable to a single class: " + ", ".join(macro_fields))
                if undefined:
                    reasons.append("Zero denominator: " + ", ".join(undefined))
                reason = "; ".join(reasons)
        output.append({**{key: "NA" if value is None else value for key, value in row.items()}, "NA_reason": reason})
    write_csv(path, output, [*fields, "NA_reason"])


METRIC_TABLE = [
    (
        field,
        field + (" ↓" if field in {"FPR", "FNR"} else " ↑" if field in {"Dice", "IoU", "mIoU", "Mean Dice"} else ""),
        True,
    )
    for field in METRIC_COLUMNS
]


def report_panel(group, summaries, classes, images, prefix=""):
    columns = [
        ("model", "模型", False),
        ("version", "版本", False),
        ("evaluated_images", "已評估圖片", False),
        *METRIC_TABLE,
        ("always_background_accuracy", "全部預測背景的 Accuracy", True),
    ]
    class_columns = [
        ("model", "模型", False),
        ("version", "版本", False),
        ("class", "類別", False),
        ("gt_positive_images", "有標記圖片", False),
        ("predicted_positive_images", "有預測圖片", False),
        *METRIC_TABLE,
    ]
    image_columns = [
        ("image_id", "圖片（含來源子資料夾）", False),
        ("model", "模型", False),
        ("version", "版本", False),
        ("pixel_status", "評估狀態", False),
        *METRIC_TABLE,
    ]
    parts = [
        f"<h2>{TITLES[group]}</h2><h3>整體：九個指標</h3>",
        html_table(summaries, columns),
        "<h3>逐類別：九個指標</h3>",
        html_table(classes, class_columns),
        (
            f'<p><a href="{prefix}summary.csv">整體 CSV</a> · <a href="{prefix}class_metrics.csv">逐類 CSV</a> · '
            f'<a href="{prefix}image_metrics.csv">全部逐張 CSV</a> · <a href="{prefix}image_class_metrics.csv">全部逐張逐類 CSV</a> · '
            f'<a href="{prefix}not_evaluated.csv">未評估原因</a></p>'
        ),
        f"<h3>逐張：九個指標（{len(images)} 筆，每頁 50 筆）</h3>",
        (
            '<div class="image-list"><input type="search" placeholder="搜尋圖片檔名或資料集" oninput="refreshImages(this.closest(\'.image-list\'),0)"> '
            "<button onclick=\"stepImages(this.closest('.image-list'),-1)\">上一頁</button> "
            '<button onclick="stepImages(this.closest(\'.image-list\'),1)">下一頁</button> <span class="page-info"></span>'
        ),
        html_table(images, image_columns),
        "</div>",
    ]
    return "".join(parts).replace("<td>—</td>", "<td>NA</td>")


SCRIPT = """<script>
function selectGroup(key){document.querySelectorAll('.tab-panel').forEach(p=>p.hidden=p.id!==key);document.querySelectorAll('[data-group]').forEach(b=>b.classList.toggle('active',b.dataset.group===key));}
function refreshImages(box,page){const query=box.querySelector('input').value.toLowerCase();const rows=[...box.querySelectorAll('tbody tr')];const found=rows.filter(r=>r.textContent.toLowerCase().includes(query));const pages=Math.max(1,Math.ceil(found.length/50));page=Math.max(0,Math.min(pages-1,page));box.dataset.page=page;rows.forEach(r=>r.hidden=true);found.slice(page*50,(page+1)*50).forEach(r=>r.hidden=false);box.querySelector('.page-info').textContent=`${page+1} / ${pages} 頁，共 ${found.length} 筆`;}
function stepImages(box,offset){refreshImages(box,Number(box.dataset.page||0)+offset);}
document.querySelectorAll('.image-list').forEach(b=>refreshImages(b,0));if(document.querySelector('.tab-panel'))selectGroup('train');
</script>"""


def write_page(path, content, config):
    basis = BASES[config["mode"]]
    notice = (
        "以每個損傷類別的一對其餘像素計算 TP/FP/FN/TN；整體 Accuracy、Precision、Recall、Dice、IoU、FPR、FNR 採像素 micro，"
        "mIoU、Mean Dice 採類別 macro，不含背景類別，也不平均預測與標記都不存在的類別。單一類別的 mIoU、Mean Dice 不適用，保留欄位並填 NA，避免重複整體分數。"
        "Accuracy 包含大量負像素，需與全部預測背景的基準比較；沒有標記及預測的類別也可能有 100% Accuracy，這不代表偵測成功。"
        "NA 表示缺少有效輸入或分母為零，原因保留在 CSV。合併組先加總像素計數再計算，不直接平均 train 與 val。"
    )
    restriction = (
        "此報表將預測框與 XML 框填滿後比較；它不驗證實際塗色輪廓，不能作為分割精度。Precision／Recall 也不是物件一對一匹配的數值。"
        if config["mode"] == "boxes"
        else "此報表比較原始預測遮罩與現有分割標記；標記完整性仍會影響結果。"
    )
    path.write_text(
        f'<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>九個指標逐筆報表</title><style>{CSS}input{{padding:10px;border:1px solid #cad2df;border-radius:6px;max-width:280px}}tr[hidden]{{display:none}}</style></head><body><main><h1>九個指標逐筆報表</h1><p>{escape(basis)} ／ {escape(config["created_at"])}</p><div class="note"><strong>{escape(restriction)}</strong><p>{escape(notice)}</p></div>{content}</main>{SCRIPT}</body></html>',
        encoding="utf-8",
    )


def generate_reports(records, output, config):
    output.mkdir(parents=True, exist_ok=False)
    panels = []
    for group, splits in GROUPS.items():
        folder = output / group
        folder.mkdir()
        selected = [record for record in records if record["split"] in splits]
        grouped = defaultdict(list)
        for record in selected:
            grouped[(record["model"], record["version"])].append(record)
        summaries, class_rows, image_rows, image_class_rows = [], [], [], []
        for (model, version), subset in sorted(grouped.items()):
            overall, by_class = pooled_metrics(subset)
            meta = {
                "model": model,
                "version": version,
                "group": group,
                "evaluation_basis": config["mode"],
                "image_count": len(subset),
                "evaluated_images": sum(record["pixel_status"] == "ok" for record in subset),
            }
            summaries.append({**meta, **overall, **metric_context(overall, "dataset_micro")})
            for name, values in by_class.items():
                eligible = [record["pixel_counts"][name] for record in subset if record["pixel_counts"] is not None]
                support = metric_context(
                    values,
                    "dataset_class",
                    sum(c["TP"] + c["FN"] > 0 for c in eligible),
                    sum(c["TP"] + c["FP"] > 0 for c in eligible),
                )
                class_rows.append({**meta, "class": name, **values, **support})
        for record in selected:
            meta = {
                field: record[field]
                for field in (
                    "model",
                    "version",
                    "split",
                    "source",
                    "width",
                    "height",
                    "pixel_status",
                    "pixel_reason",
                    "pixel_truth",
                )
            }
            _, tail = split_relative(Path(record["source"]))
            image_id = record["split"] + "/" + (tail.as_posix() if tail else Path(record["source"]).name)
            meta.update(image_name=Path(record["source"]).name, image_id=image_id, evaluation_basis=config["mode"])
            image_rows.append(
                {**meta, **record["pixel_overall"], **metric_context(record["pixel_overall"], "image_micro")}
            )
            for name, values in record["pixel_classes"].items():
                valid = values["TP"] is not None
                support = metric_context(
                    values,
                    "image_class",
                    int(values["TP"] + values["FN"] > 0) if valid else None,
                    int(values["TP"] + values["FP"] > 0) if valid else None,
                )
                image_class_rows.append({**meta, "class": name, **values, **support})
        not_evaluated = [row for row in image_rows if row["pixel_status"] != "ok"]
        formats = {
            "summary.csv": (
                summaries,
                ["model", "version", "group", "evaluation_basis", "image_count", "evaluated_images"],
            ),
            "class_metrics.csv": (
                class_rows,
                ["model", "version", "group", "evaluation_basis", "image_count", "evaluated_images", "class"],
            ),
            "image_metrics.csv": (
                image_rows,
                [
                    "model",
                    "version",
                    "split",
                    "source",
                    "image_name",
                    "image_id",
                    "width",
                    "height",
                    "evaluation_basis",
                    "pixel_status",
                    "pixel_reason",
                    "pixel_truth",
                ],
            ),
            "image_class_metrics.csv": (
                image_class_rows,
                [
                    "model",
                    "version",
                    "split",
                    "source",
                    "image_name",
                    "image_id",
                    "width",
                    "height",
                    "evaluation_basis",
                    "pixel_status",
                    "pixel_reason",
                    "pixel_truth",
                    "class",
                ],
            ),
            "not_evaluated.csv": (
                not_evaluated,
                [
                    "model",
                    "version",
                    "split",
                    "source",
                    "image_name",
                    "image_id",
                    "width",
                    "height",
                    "evaluation_basis",
                    "pixel_status",
                    "pixel_reason",
                    "pixel_truth",
                ],
            ),
        }
        for filename, (rows, metadata) in formats.items():
            write_metric_csv(folder / filename, rows, [*metadata, *METRIC_COLUMNS, *COUNT_COLUMNS, *CONTEXT_COLUMNS])
        (folder / "metrics.json").write_text(
            json.dumps({"summary": summaries, "classes": class_rows}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_page(folder / "report.html", report_panel(group, summaries, class_rows, image_rows), config)
        panels.append(
            f'<section class="tab-panel" id="{group}">{report_panel(group, summaries, class_rows, image_rows, group + "/")}</section>'
        )
        print(
            f"{group}: {len(image_rows)} image rows; {len(image_class_rows)} image/class rows; evaluated {len(image_rows) - len(not_evaluated)}",
            flush=True,
        )
    tabs = (
        '<nav class="tabs">'
        + "".join(
            f'<button data-group="{group}" onclick="selectGroup(\'{group}\')">{TITLES[group]}</button>'
            for group in GROUPS
        )
        + "</nav>"
    )
    write_page(output / "index.html", tabs + "".join(panels), config)
    (output / "settings.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode", choices=("boxes", "masks"), required=True, help="必須明確選擇：boxes 是框區域，masks 才是真實分割"
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--annotations", type=Path, nargs="+", default=[DEFAULT_ANNOTATIONS], help="boxes 模式使用的 XML"
    )
    parser.add_argument(
        "--mask-annotations", type=Path, nargs="+", default=[], help="masks 模式的 NPZ 遮罩或 YOLO 分割 TXT"
    )
    parser.add_argument("--versions", nargs="+")
    parser.add_argument("--eval-conf", type=float, default=0.0)
    parser.add_argument("--xml-origin", type=int, choices=(0, 1), default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.eval_conf <= 1:
        parser.error("Confidence must be in [0,1]")
    if args.mode == "masks" and not args.mask_annotations:
        parser.error("masks mode requires --mask-annotations")
    results = args.results.expanduser().resolve()
    if not results.is_dir():
        parser.error(f"Missing results folder: {results}")
    annotations = AnnotationIndex(args.annotations if args.mode == "boxes" else [])
    records, _issues = load_records(results, annotations, args.versions, args.xml_origin, args.eval_conf)
    if not records:
        parser.error("No valid train/val inference records")
    evaluate_images(records, args.mode, args.mask_annotations, args.eval_conf)
    created = datetime.now(timezone(timedelta(hours=8)))
    output = args.output.expanduser().resolve() / created.strftime("report_%Y%m%d_%H%M%S_%f")
    config = {
        "created_at": created.isoformat(),
        "mode": args.mode,
        "results": str(results),
        "annotations": [str(path.resolve()) for path in args.annotations],
        "mask_annotations": [str(path.resolve()) for path in args.mask_annotations],
        "eval_conf": args.eval_conf,
        "xml_origin": args.xml_origin,
        "versions": args.versions,
        "metric_units": "0_to_1; HTML percent",
        "absent_class_policy": "exclude_from_macro; undefined_ratio_NA",
        "class_policy": "independent_foreground_one_vs_rest; background_excluded_from_macro",
        "single_class_macro_policy": "not_applicable_NA",
        "report_schema": 2,
    }
    generate_reports(records, output, config)
    print(f"Report: {output / 'index.html'}", flush=True)
    if args.open:
        webbrowser.open((output / "index.html").as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
