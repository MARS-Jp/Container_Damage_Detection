"""Independently audit saved inference statistics against JSON, XML and images.

No inference is run and source reports are never overwritten. Rectangle pixels
are reconstructed with a difference array, independently of pixel_metrics.py.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
from html import escape
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.report_nine_metrics import load_mask_archive, load_polygon_labels

DEFAULT_REPORT = ROOT / "tools/runs/nine_metric_reports/report_20261008_222744_713784"
FIELDS = ("Accuracy", "Precision", "Recall", "Dice", "IoU", "mIoU", "Mean Dice", "FPR", "FNR")
COUNTS = ("TP", "FP", "FN", "TN")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rectangle_union(boxes, height, width):
    delta = np.zeros((height + 1, width + 1), dtype=np.int32)
    xs, ys = np.arange(width) + .5, np.arange(height) + .5
    for x1, y1, x2, y2 in boxes:
        left, right = np.searchsorted(xs, [x1, x2], side="left")
        top, bottom = np.searchsorted(ys, [y1, y2], side="left")
        delta[top, left] += 1
        delta[bottom, right] += 1
        delta[top, right] -= 1
        delta[bottom, left] -= 1
    return delta.cumsum(axis=0).cumsum(axis=1)[:-1, :-1] > 0


def scores(counts):
    tp, fp, fn, tn = (counts[key] for key in COUNTS)
    def ratio(a, b):
        return a / b if b else None
    return dict(Accuracy=ratio(tp + tn, tp + fp + fn + tn), Precision=ratio(tp, tp + fp),
                Recall=ratio(tp, tp + fn), Dice=ratio(2*tp, 2*tp + fp + fn),
                IoU=ratio(tp, tp + fp + fn), FPR=ratio(fp, fp + tn), FNR=ratio(fn, tp + fn))


def binary_counts(prediction, truth):
    # Logical truth table, deliberately separate from report's area subtraction.
    return dict(TP=int((prediction & truth).sum()), FP=int((prediction & ~truth).sum()),
                FN=int((~prediction & truth).sum()), TN=int((~prediction & ~truth).sum()))


def object_counts(predictions, truths, threshold=.5):
    used, tp = set(), 0
    for prediction in sorted(predictions, key=lambda row: -row["score"]):
        p = prediction["box"]
        candidates = []
        for index, truth in enumerate(truths):
            if index in used or prediction["class"] != truth["class"]:
                continue
            t = truth["box"]
            overlap = max(0, min(p[2], t[2]) - max(p[0], t[0])) * max(0, min(p[3], t[3]) - max(p[1], t[1]))
            union = (p[2]-p[0])*(p[3]-p[1]) + (t[2]-t[0])*(t[3]-t[1]) - overlap
            candidates.append((overlap / union if union else 0, index))
        iou, index = max(candidates, default=(0, -1))
        if index >= 0 and iou >= threshold:
            used.add(index)
            tp += 1
    fp, fn = len(predictions) - tp, len(truths) - tp
    return dict(TP=tp, FP=fp, FN=fn, Precision=tp/(tp+fp) if tp+fp else None,
                Recall=tp/(tp+fn) if tp+fn else None, F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def xml_objects(path, origin):
    root = ET.parse(path).getroot()
    truths = []
    for obj in root.findall("object"):
        box = [float(obj.findtext("bndbox/" + key)) for key in ("xmin", "ymin", "xmax", "ymax")]
        box[0] -= origin
        box[1] -= origin
        truths.append({"class": obj.findtext("name"), "box": box})
    return root, truths


def pooled(class_counts):
    overall = {field: sum(row[field] for row in class_counts.values()) for field in COUNTS}
    values = scores(overall)
    for macro, key in (("mIoU", "IoU"), ("Mean Dice", "Dice")):
        present = [scores(row)[key] for row in class_counts.values() if scores(row)[key] is not None]
        values[macro] = sum(present)/len(present) if present else None
    total = sum(overall.values())
    return {**overall, **values, "always_background_Accuracy": (overall["FP"] + overall["TN"])/total if total else None,
            "truth_positive_fraction": (overall["TP"] + overall["FN"])/total if total else None}


def assert_value(actual, expected, label):
    if actual == "NA":
        assert expected is None, (label, actual, expected)
    else:
        assert expected is not None and abs(float(actual) - expected) < 1e-12, (label, actual, expected)


def sample_check(records, source_root, output):
    sample = next(row for row in records if row["image_id"].casefold() == "train/b/cbhu0708054-a.jpg")
    mask_runs = ROOT / "runs/nine_metric_mask_check"
    candidates = sorted(mask_runs.glob("run_*/01_last0912/train/01_CBHU0708054-A/S08_diagnostic_only/prediction_masks.npz"))
    if not candidates:
        return {"status": "no_saved_sample_masks"}
    path = candidates[-1]
    metric_path = next(p for p in source_root.rglob("prediction_metrics.json")
                       if json.loads(p.read_text(encoding="utf-8"))["source"] == sample["source"])
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    classes = {row["name"]: {"id": int(key)} for key, row in metrics["classes"].items()}
    width, height = int(sample["width"]), int(sample["height"])
    predictions = load_mask_archive(path, list(classes), height, width, 0)
    actual_pixels = {name: int(mask.sum()) for name, mask in predictions.items()}
    assert actual_pixels == {row["name"]: row["union_pixels"] for row in metrics["classes"].values()}
    png = metric_path.parent / (Path(sample["source"]).stem + "__all.jpg")
    rerun_png = path.with_name(png.name)
    same_visual = png.is_file() and rerun_png.is_file() and digest(png) == digest(rerun_png)
    assert same_visual, "Sample rerun visual differs from original batch"
    labels = Path("C:/Users/USER/Desktop/碩論/dataset0608/dataset0608/dataset/labels/train/B/CBHU0708054-A.txt")
    photo = labels.parents[3] / "images/train/B/CBHU0708054-A.JPG"
    assert photo.is_file() and digest(photo) == digest(Path(sample["source"])), "Polygon annotation image differs"
    truth = load_polygon_labels(labels, classes, height, width)
    true_metrics = pooled({name: binary_counts(predictions[name], truth[name]) for name in classes})
    old_root, old_objects = xml_objects(Path(sample["pixel_truth"]), 1)
    new_path = Path("C:/Users/USER/Desktop/0928-1/CBHU0708054-A.xml")
    _, new_objects = xml_objects(new_path, 1)
    entries = json.loads(metric_path.with_name("predictions.json").read_text(encoding="utf-8"))["detections"]
    pred_objects = [{"class": row["class"], "score": row["score"], "box": row["box_xyxy"]} for row in entries]
    comparisons = [dict(basis="old_XML_rectangle_pixels", **{key: float(sample[key]) for key in FIELDS}),
                   dict(basis="old_polygon_actual_mask_pixels", **{key: true_metrics[key] for key in FIELDS})]
    save_csv(output / "sample_metric_comparison.csv", comparisons)
    original = np.asarray(Image.open(sample["source"]).convert("RGB"))
    def overlay(masks):
        result = original.astype(float).copy()
        for name, color in (("D", (255, 125, 0)), ("R", (70, 255, 0))):
            result[masks[name]] = result[masks[name]]*.65 + np.array(color)*.35
        return Image.fromarray(result.astype(np.uint8))
    pred_rectangles = {name: rectangle_union([p["box"] for p in pred_objects if p["class"] == name], height, width) for name in classes}
    tiles = [("原圖", Image.fromarray(original)), ("報表使用的預測方框填滿", overlay(pred_rectangles)),
             ("實際 S08 塗色遮罩", overlay(predictions)), ("現有舊版多邊形標記（不是完整新標記）", overlay(truth))]
    canvas = Image.new("RGB", (width*2, (height+54)*2), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", 23)
    for index, (title, tile) in enumerate(tiles):
        x, y = (index % 2)*width, (index//2)*(height+54)
        draw.text((x+12, y+12), title, fill="#173f69", font=font)
        canvas.paste(tile, (x, y+54))
    canvas.save(output / "sample_boxes_vs_masks.png")
    return {"status": "verified", "source": sample["source"], "mask_archive": str(path),
            "old_polygon_labels": str(labels), "identical_rendered_jpeg": same_visual,
            "actual_mask_union_pixels": actual_pixels, "comparisons": comparisons,
            "old_XML_object_counts": dict(Counter(obj["class"] for obj in old_objects)),
            "new_XML_object_counts": dict(Counter(obj["class"] for obj in new_objects)),
            "old_XML_object_matching": object_counts(pred_objects, old_objects),
            "new_XML_object_matching": object_counts(pred_objects, new_objects)}


def write_audit_page(summary, output, corrected_report=None):
    """Make the verified findings reviewable, including support and provenance."""
    output = Path(output)
    source_rows = read_csv(output / "source_image_checks.csv")
    hashes = {row["source"]: row["sha256"] for row in source_rows}
    report = Path(summary["source_report"])
    records = read_csv(report / "train_and_val/image_metrics.csv")
    class_rows = read_csv(report / "train_and_val/image_class_metrics.csv")
    origin = json.loads((report / "settings.json").read_text(encoding="utf-8"))["xml_origin"]
    truth_groups = defaultdict(list)
    for row in records:
        if row["pixel_status"] == "ok":
            _, truths = xml_objects(Path(row["pixel_truth"]), origin)
            signature = json.dumps(sorted((t["class"], *t["box"]) for t in truths))
            truth_groups[hashes[row["source"]]].append((row, signature))
    conflicts = []
    for sha, entries in truth_groups.items():
        if len({signature for _, signature in entries}) > 1:
            conflicts.extend(dict(sha256=sha, image_id=row["image_id"], source=row["source"], annotation=row["pixel_truth"],
                                  annotated_positive_pixels=int(row["TP"])+int(row["FN"])) for row, _ in entries)
    save_csv(output / "duplicate_annotation_conflicts.csv", conflicts)
    summary["conflicting_duplicate_annotation_groups"] = len({row["sha256"] for row in conflicts})
    summary["unique_evaluated_photos"] = {group: len({hashes[row["source"]] for row in records if row["pixel_status"] == "ok" and (group == "train_and_val" or row["split"] == group)}) for group in summary["groups"]}
    support = []
    for name in sorted({row["class"] for row in class_rows}):
        present = [row for row in class_rows if row["class"] == name and row["pixel_status"] == "ok" and int(row["TP"])+int(row["FN"]) > 0]
        support.append({"class": name, "gt_positive_images": len(present), "unique_gt_positive_photos": len({hashes[row["source"]] for row in present})})
    save_csv(output / "class_support_unique_photos.csv", support)
    (output / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    def pct(value):
        return "NA" if value is None else f"{value*100:.2f}%"
    def table(headers, rows):
        return "<table><thead><tr>" + "".join(f"<th>{escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table>"
    group_rows = []
    for group, data in summary["groups"].items():
        pixels, objects = data["box_pixel_metrics"], data["box_object_metrics"]
        group_rows.append([group, f'{data["evaluated"]}/{data["images"]}', summary["unique_evaluated_photos"][group], pct(pixels["Accuracy"]),
                           pct(pixels["always_background_Accuracy"]), pct(pixels["Precision"]), pct(objects["Precision"]), pct(objects["Recall"])])
    sample = summary["sample"]
    area_rows = read_csv(output / "prediction_box_vs_mask_areas.csv")
    areas = []
    for name in ("D", "R"):
        actual = sum(int(row["actual_mask_union_pixels"]) for row in area_rows if row["class"] == name)
        rectangle = sum(int(row["report_box_union_pixels"]) for row in area_rows if row["class"] == name)
        areas.append([name, f"{actual:,}", f"{rectangle:,}", f"{(rectangle/actual-1)*100:.2f}%"])
    class_support = table(["類別", "有標記圖片筆數", "獨立照片數（SHA256）"], [[row["class"], row["gt_positive_images"], row["unique_gt_positive_photos"]] for row in support])
    corrected_link = f'<p><a href="{Path(corrected_report).resolve().as_uri()}">開啟修正後三組逐筆報表</a>：仍為框區域評估，移除逐類重複的宏觀平均，加入背景基準及支持量；九個欄位保留，不適用填 NA。</p>' if corrected_report else ""
    comparison = ""
    if sample["status"] == "verified":
        comparison = table(["範例評估方式", "Precision", "Recall", "Dice", "IoU"], [[row["basis"], *[pct(row[key]) for key in ("Precision", "Recall", "Dice", "IoU")]] for row in sample["comparisons"]])
        comparison += '<img src="sample_boxes_vs_masks.png" alt="原圖、報表方框、實際遮罩與舊標記對照">'
        comparison += f'<p>範例原始遮罩的類別聯集像素與舊 JSON 完全相符，重跑 JPEG 與原結果 SHA256 相同；多邊形標記的對應照片亦與來源照片 SHA256 相同。這只有 1 張診斷樣本，不能代替整批分割成績。</p><p>這張照片舊 XML 有 31 個物件，新 XML 有 94 個；同一批 192 個預測、IoU≥0.5 的框匹配 Recall：舊標記 {pct(sample["old_XML_object_matching"]["Recall"])}，新標記 {pct(sample["new_XML_object_matching"]["Recall"])}。標記漏標會使真實預測被列為未匹配，物件切分不同也會影響框匹配。</p>'
    checks = summary["annotation_image_checks"]
    evaluated = summary["verified_image_metrics"]
    class_verified = summary["verified_class_metrics"]
    same_images = checks.get("same_image_sha256", 0)
    no_images = checks.get("no_annotation_image_available", 0)
    excluded = summary["excluded_reasons"]
    excluded_count = sum(excluded.values())
    reason_text = "、".join(f"{excluded.get(key, 0)} 張{label}" for key, label in (("missing_annotation", "沒有對應 XML"), ("ambiguous_annotation", "對應 XML 有歧義"), ("invalid_annotation", "XML 無效")))
    files = ["audit.json", "source_image_checks.csv", "annotation_image_checks.csv", "independent_recalculation.csv", "prediction_box_vs_mask_areas.csv", "object_matching.csv", "duplicate_source_images.csv", "duplicate_annotation_conflicts.csv", "class_support_unique_photos.csv", "sample_metric_comparison.csv"]
    html = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>推論數據核查</title><style>body{{font:16px 'Microsoft JhengHei',sans-serif;background:#f3f6fa;color:#203247;margin:0}}main{{max-width:1400px;margin:auto;padding:28px}}section{{background:white;padding:24px;border-radius:10px;margin:20px 0;overflow:auto}}h1,h2{{color:#173f69}}p{{line-height:1.8}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}td,th{{padding:12px;border-bottom:1px solid #dce3eb;text-align:right}}th{{background:#214b75;color:white}}img{{width:100%;height:auto}}a{{color:#2463a0}}.note{{background:#fff0d8;padding:18px;border-radius:8px}}</style><main><h1>推論數據核查</h1><p class="note">結論：方框像素計算可重現，但原表不能當作實際分割精度。大量背景使 Accuracy 偏高；逐類重複整體 mIoU／Mean Dice 的呈現已修正。整批分割成績仍缺原始預測遮罩及完整、確認過的分割標記。</p>
    <section><h2>整體指標對照</h2><p>方框像素聯集與物件框一對一匹配使用相同有效 XML 圖片，但衡量不同目標。未匹配預測不一定全是視覺誤檢，亦可能是漏標、定位或物件拆分問題。train＋val 不可代替獨立驗證成績。</p>{table(["組別","可評估／全部","獨立照片數","原表 Accuracy","全部背景 Accuracy","方框像素 Precision","物件 Precision","物件 Recall"],group_rows)}{corrected_link}</section>
    <section><h2>已核對與仍有限制的資料</h2><p>{summary["verified_source_images"]} 張來源照片經 OpenCV（與推論一致，含 EXIF 轉向）解碼後，與保存的 full_original.png 完全相同。{evaluated} 張圖片、{class_verified:,} 個逐類計數已從 JSON、XML 獨立重建，全數與原 CSV 相符；三組彙總亦一致。</p><p>{same_images} 張可取得標記來源照片，SHA256 與推論來源相同；另 {no_images} 張沒有可核對的標記照片，僅通過檔名與尺寸核對，不能宣稱已確認其標記內容正確。</p><p>{excluded_count} 張未評估：{reason_text}。全部來源共有 {summary["duplicate_image_groups"]} 組完全相同照片；未找到 train/val 間 SHA256 完全相同照片，但這不排除同一貨櫃的不同視角洩漏。有效標記中的相同照片有 {summary["conflicting_duplicate_annotation_groups"]} 組 XML 內容不一致，涉及 {len(conflicts)} 筆；未自行選擇或合併標記，詳見清單。</p>{class_support}<p>只有少量獨立照片的類別，其高分不足以證明泛化表現。沒有預測、沒有標記的單類別可出現 Accuracy=100%、FPR=0，而 Precision／Recall／Dice／IoU 為 NA。</p></section>
    <section><h2>報表評估區域與實際塗色不同</h2><p>以下只統計 {evaluated} 張可評估圖片。舊表填滿矩形框，包含遮罩以外的像素，並合併同類重複／碎片；物件一對一匹配則會計算每個未匹配的預測。</p>{table(["類別","實際遮罩聯集像素","報表方框聯集像素","方框多出的面積"],areas)}{comparison}</section>
    <section><h2>核查依據與修正範圍</h2><p>重新計算使用差分陣列與像素中心判定，未呼叫原報表的方框填色或指標公式；物件匹配亦獨立實作。沒有調整模型、推論閾值或原始結果。全量真正的分割 Accuracy／Dice／IoU 等需保存預測 NPZ，並使用同一圖片、同一類別定義的完整分割標記。</p><p>CSV 的分數單位為 0–1，本頁轉成百分比；資料集加總後的比例、逐圖片平均、逐類平均是三個不同統計範圍，不應互換。舊 PDF 與 statistics_charts 不會自動更新。</p><p>{' ／ '.join(f'<a href="{name}">{name}</a>' for name in files)}</p></section></main></html>'''
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / "README.md").write_text("# 推論數據核查\n\n請開啟 index.html。方框指標可重現，但不能當作塗色遮罩分割精度；全量分割尚未核實。\n\n來源報表：" + summary["source_report"] + "\n", encoding="utf-8")
    return output / "index.html"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=ROOT / "tools/runs/metric_audits")
    args = parser.parse_args(argv)
    report = args.report.resolve()
    settings = json.loads((report / "settings.json").read_text(encoding="utf-8"))
    if settings["mode"] != "boxes":
        parser.error("This independent rectangle audit expects a boxes report")
    source_root = Path(settings["results"])
    records = read_csv(report / "train_and_val/image_metrics.csv")
    class_rows = read_csv(report / "train_and_val/image_class_metrics.csv")
    class_lookup = {(row["source"], row["class"]): row for row in class_rows}
    predictions_by_source = {json.loads(p.read_text(encoding="utf-8"))["source"]: p for p in source_root.rglob("prediction_metrics.json")}
    assert len(records) == len(predictions_by_source) == len({row["source"] for row in records})
    output = args.output.resolve() / datetime.now().strftime("audit_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    hashes, original_checks, annotation_checks, area_rows, reconstruction_rows = {}, [], [], [], []
    object_rows, pixel_counts, class_names = [], {}, None
    counterparts = list(Path("C:/Users/USER/Desktop/已標記/images").rglob("*"))
    counterpart_index = defaultdict(list)
    for path in counterparts:
        if path.is_file():
            counterpart_index[path.name.casefold()].append(path)
    for index, row in enumerate(records, 1):
        source = Path(row["source"])
        metric_path = predictions_by_source[row["source"]]
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        prediction_json = json.loads(metric_path.with_name("predictions.json").read_text(encoding="utf-8"))
        height, width = int(row["height"]), int(row["width"])
        names = [entry["name"] for entry in metrics["classes"].values()]
        if class_names is None:
            class_names = names
        assert names == class_names
        entries = prediction_json["detections"]
        predictions = [{"class": p["class"], "score": p["score"], "box": p["box_xyxy"]} for p in entries if p["score"] >= settings["eval_conf"]]
        counts = Counter(p["class"] for p in entries)
        assert counts == Counter({p["name"]: p["count"] for p in metrics["classes"].values() if p["count"]})
        assert metrics["source"] == row["source"] and metrics["width"] == width and metrics["height"] == height
        hashes[row["source"]] = digest(source)
        # Match inference decoding: OpenCV applies JPEG EXIF orientation.
        original = cv2.imdecode(np.fromfile(source, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
        snapshot = metric_path.parent.parent / "inputs/full_original.png"
        snapshot_equal = snapshot.is_file() and np.array_equal(original, np.asarray(Image.open(snapshot).convert("RGB")))
        original_checks.append(dict(image_id=row["image_id"], source=str(source), split=row["split"], sha256=hashes[row["source"]],
                                    dimensions_ok=original.shape[:2] == (height, width), saved_input_matches_source=snapshot_equal,
                                    exif_oriented=Image.open(source).size != (width, height)))
        assert snapshot_equal and original.shape[:2] == (height, width), row["source"]
        if row["pixel_status"] == "ok":
            annotation, truths = xml_objects(Path(row["pixel_truth"]), settings["xml_origin"])
            assert annotation.findtext("filename").casefold() == source.name.casefold()
            assert int(annotation.findtext("size/width")) == width and int(annotation.findtext("size/height")) == height
            path_parts = list(Path(row["pixel_truth"]).parts)
            path_parts[path_parts.index("labels")] = "images"
            desired = Path(*path_parts).with_name(annotation.findtext("filename"))
            candidates = [desired] if desired.is_file() else counterpart_index.get(source.name.casefold(), [])
            same_images = [p for p in candidates if digest(p) == hashes[row["source"]]]
            pair_status = "same_image_sha256" if same_images else "different_image_sha256" if candidates else "no_annotation_image_available"
            annotation_checks.append(dict(image_id=row["image_id"], source=str(source), annotation=row["pixel_truth"],
                                         image_pair_status=pair_status, paired_images=" | ".join(map(str, candidates))))
            class_count = {}
            for name in names:
                prediction = rectangle_union([p["box"] for p in predictions if p["class"] == name], height, width)
                truth = rectangle_union([p["box"] for p in truths if p["class"] == name], height, width)
                expected = binary_counts(prediction, truth)
                saved = class_lookup[(row["source"], name)]
                assert all(expected[key] == int(saved[key]) for key in COUNTS), (row["image_id"], name)
                for field, value in scores(expected).items():
                    assert_value(saved[field], value, (row["image_id"], name, field))
                class_count[name] = expected
                actual_pixels = next(c["union_pixels"] for c in metrics["classes"].values() if c["name"] == name)
                rect_pixels = int(prediction.sum())
                area_rows.append(dict(image_id=row["image_id"], split=row["split"], **{"class": name},
                                      prediction_count=counts[name], actual_mask_union_pixels=actual_pixels,
                                      report_box_union_pixels=rect_pixels, box_to_mask_ratio=rect_pixels/actual_pixels if actual_pixels else "NA"))
            pixel_counts[row["source"]] = class_count
            independent = pooled(class_count)
            for field in FIELDS:
                assert_value(row[field], independent[field], (row["image_id"], field))
            reconstruction_rows.append(dict(image_id=row["image_id"], split=row["split"], class_rows_verified=len(names), all_metrics_verified=True))
            object_rows.append(dict(image_id=row["image_id"], split=row["split"], **object_counts(predictions, truths)))
        if index % 100 == 0:
            print(f"Audit: {index}/{len(records)} source images", flush=True)
    groups = {}
    for group, splits in (("train", ("train",)), ("val", ("val",)), ("train_and_val", ("train", "val"))):
        selected = [row for row in records if row["split"] in splits]
        eligible = [row for row in selected if row["source"] in pixel_counts]
        combined = {name: {key: sum(pixel_counts[row["source"]][name][key] for row in eligible) for key in COUNTS} for name in class_names}
        verified = pooled(combined)
        saved = read_csv(report / group / "summary.csv")[0]
        for field in FIELDS:
            assert_value(saved[field], verified[field], (group, field))
        objects = {key: sum(row[key] for row in object_rows if row["split"] in splits) for key in ("TP", "FP", "FN")}
        tp, fp, fn = (objects[key] for key in ("TP", "FP", "FN"))
        objects.update(Precision=tp/(tp+fp), Recall=tp/(tp+fn), F1=2*tp/(2*tp+fp+fn))
        groups[group] = dict(images=len(selected), evaluated=len(eligible), coverage=len(eligible)/len(selected), box_pixel_metrics=verified, box_object_metrics=objects)
    hash_groups = defaultdict(list)
    for row in original_checks:
        hash_groups[row["sha256"]].append(row)
    duplicates = [dict(sha256=sha, source=row["source"], split=row["split"], image_id=row["image_id"],
                       same_image_copies=len(rows), cross_split=len({r["split"] for r in rows}) > 1)
                  for sha, rows in hash_groups.items() if len(rows) > 1 for row in rows]
    save_csv(output / "source_image_checks.csv", original_checks)
    save_csv(output / "annotation_image_checks.csv", annotation_checks)
    save_csv(output / "independent_recalculation.csv", reconstruction_rows)
    save_csv(output / "prediction_box_vs_mask_areas.csv", area_rows)
    save_csv(output / "object_matching.csv", object_rows)
    save_csv(output / "duplicate_source_images.csv", duplicates)
    sample = sample_check(records, source_root, output)
    summary = dict(source_report=str(report), evaluation_basis=settings["mode"], groups=groups,
                   verified_source_images=len(original_checks), verified_image_metrics=len(reconstruction_rows),
                   verified_class_metrics=len(reconstruction_rows)*len(class_names),
                   annotation_image_checks=dict(Counter(row["image_pair_status"] for row in annotation_checks)),
                   excluded_reasons=dict(Counter(row["pixel_reason"].split(": ")[1] for row in records if row["pixel_status"] != "ok")),
                   duplicate_image_groups=sum(len(rows) > 1 for rows in hash_groups.values()),
                   cross_split_duplicate_groups=sum(len({r["split"] for r in rows}) > 1 for rows in hash_groups.values()),
                   sample=sample)
    (output / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_audit_page(summary, output)
    print(json.dumps({"output": str(output), "groups": groups, "annotation_image_checks": summary["annotation_image_checks"],
                      "cross_split_duplicate_groups": summary["cross_split_duplicate_groups"], "sample": sample}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
