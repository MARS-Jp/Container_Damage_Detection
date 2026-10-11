"""Export a source-labelled research comparison from the corrected local report.

The external nine-metric rows are a transcription of the user's image, not
verified paper results. Public benchmark values are kept in a separate table.
"""
from __future__ import annotations

import argparse
import csv
from html import escape
import hashlib
import json
import math
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "tools/runs/nine_metric_reports/report_20261008_235420_507363"
AUDIT = ROOT / "tools/runs/metric_audits/audit_20261008_235145_226565/audit.json"
RESEARCH = ROOT / "research/model_comparison_20261009"
METRICS = ("Accuracy", "Precision", "Recall", "Dice", "IoU", "mIoU", "Mean Dice", "FPR", "FNR")
GROUPS = ("train", "val", "train_and_val")
LABELS = {"train": "train", "val": "val", "train_and_val": "train + val"}
INK = colors.HexColor("#21364a")
NAVY = colors.HexColor("#244d72")
MUTED = colors.HexColor("#5c6e80")
PALE = colors.HexColor("#f1f5f8")


def csv_read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12), (a, b)


def load_verified_local(report, audit_path):
    settings = json.loads((report / "settings.json").read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert settings["mode"] == "boxes"
    assert settings["class_policy"] == "independent_foreground_one_vs_rest; background_excluded_from_macro"
    assert settings["single_class_macro_policy"] == "not_applicable_NA"
    summaries = {}
    checks = []
    for group in GROUPS:
        records = csv_read(report / group / "summary.csv")
        assert len(records) == 1
        row = records[0]
        assert row["evaluation_basis"] == "boxes"
        assert row["version"] == "S08_diagnostic_only"
        classes = csv_read(report / group / "class_metrics.csv")
        assert len(classes) == 8
        assert all(c["mIoU"] == c["Mean Dice"] == "NA" for c in classes)
        counts = {key: sum(int(c[key]) for c in classes) for key in ("TP", "FP", "FN", "TN")}
        assert all(counts[k] == int(row[k]) for k in counts)
        tp, fp, fn, tn = (counts[k] for k in ("TP", "FP", "FN", "TN"))
        formulas = {
            "Accuracy": (tp + tn) / (tp + fp + fn + tn),
            "Precision": tp / (tp + fp), "Recall": tp / (tp + fn),
            "Dice": 2*tp / (2*tp + fp + fn), "IoU": tp / (tp + fp + fn),
            "FPR": fp / (fp + tn), "FNR": fn / (tp + fn),
        }
        active = [c for c in classes if int(c["TP"]) + int(c["FP"]) + int(c["FN"]) > 0]
        formulas["mIoU"] = sum(int(c["TP"])/(int(c["TP"])+int(c["FP"])+int(c["FN"])) for c in active)/len(active)
        formulas["Mean Dice"] = sum(2*int(c["TP"])/(2*int(c["TP"])+int(c["FP"])+int(c["FN"])) for c in active)/len(active)
        for key in METRICS:
            assert_close(row[key], formulas[key])
            assert_close(row[key], audit["groups"][group]["box_pixel_metrics"][key])
            row[key] = float(row[key])
        assert int(row["evaluated_images"]) == audit["groups"][group]["evaluated"]
        assert int(row["image_count"]) == audit["groups"][group]["images"]
        row["unique_evaluated_photos"] = audit["unique_evaluated_photos"][group]
        row["active_macro_classes"] = len(active)
        summaries[group] = row
        checks.append({"group": group, "nine_metrics_recomputed": True,
                       "audit_matches": True, "active_macro_classes": len(active)})
    for key in ("TP", "FP", "FN", "TN"):
        assert int(summaries["train_and_val"][key]) == int(summaries["train"][key]) + int(summaries["val"][key])
    return summaries, audit, settings, checks


class ComparisonPDF:
    def __init__(self, path, date):
        pdfmetrics.registerFont(TTFont("CJK", "C:/Windows/Fonts/msjh.ttc", subfontIndex=0))
        pdfmetrics.registerFont(TTFont("CJKBold", "C:/Windows/Fonts/msjhbd.ttc", subfontIndex=0))
        pdfmetrics.registerFont(TTFont("Latin", "C:/Windows/Fonts/arial.ttf"))
        pdfmetrics.registerFont(TTFont("LatinBold", "C:/Windows/Fonts/arialbd.ttf"))
        pdfmetrics.registerFontFamily("CJK", normal="CJK", bold="CJKBold")
        pdfmetrics.registerFontFamily("Latin", normal="Latin", bold="LatinBold")
        self.width, self.height = landscape(A4)
        self.margin = 35
        self.usable = self.width - 2*self.margin
        self.canvas = canvas.Canvas(str(path), pagesize=(self.width, self.height), pageCompression=1)
        self.canvas.setTitle("本研究與既有模型的數據並列 - last0912 / S08")
        self.canvas.setAuthor("")
        self.canvas.setSubject("來源分別標示的九指標表、三組本機結果及論文公開基準；評估條件未統一")
        self.page = 0
        self.date = date
        self.styles = {
            "body": ParagraphStyle("body", fontName="CJK", fontSize=11, leading=17, textColor=INK, wordWrap="CJK"),
            "small": ParagraphStyle("small", fontName="CJK", fontSize=9.5, leading=14.5, textColor=INK, wordWrap="CJK"),
            "section": ParagraphStyle("section", fontName="CJKBold", fontSize=12, leading=18, textColor=NAVY, wordWrap="CJK"),
        }
        self.layout = []

    def begin(self, title, subtitle):
        if self.page:
            self.canvas.showPage()
        self.page += 1
        c = self.canvas
        c.bookmarkPage(f"p{self.page}")
        c.addOutlineEntry(title, f"p{self.page}", 0)
        c.setFillColor(NAVY)
        c.setFont("CJKBold", 21)
        c.drawString(self.margin, self.height-45, title)
        c.setFillColor(MUTED)
        c.setFont("CJK", 10)
        c.drawString(self.margin, self.height-67, subtitle)
        c.setStrokeColor(colors.HexColor("#d5dfe7"))
        c.line(self.margin, 30, self.width-self.margin, 30)
        c.setFont("CJK", 8.5)
        c.drawString(self.margin, 16, f"{self.date} | 分數來源與評估方式分別註明；無同條件排名")
        c.drawRightString(self.width-self.margin, 16, str(self.page))
        return self.height-91

    def paragraph(self, text, y, style="body", gap=9):
        p = Paragraph(text, self.styles[style])
        _, h = p.wrap(self.usable, self.height)
        assert y-h >= 43, (self.page, "paragraph overflow", y, h, text)
        p.drawOn(self.canvas, self.margin, y-h)
        self.layout.append({"page": self.page, "kind": "paragraph", "top": y, "bottom": y-h})
        return y-h-gap

    def table(self, rows, y, widths, size=9.5, pad=7, highlight=None, first_left=True):
        assert math.isclose(sum(widths), self.usable, abs_tol=1e-6)
        prepared = []
        for i, row in enumerate(rows):
            cells = []
            for j, value in enumerate(row):
                style = ParagraphStyle("cell", fontName="CJKBold" if i == 0 else "CJK", fontSize=size,
                                       leading=size*1.35, textColor=colors.white if i == 0 else INK,
                                       alignment=0 if first_left and j == 0 else 1, wordWrap="CJK")
                cells.append(Paragraph(str(value), style))
            prepared.append(cells)
        t = Table(prepared, colWidths=widths, repeatRows=1)
        instructions = [
            ("BACKGROUND", (0,0), (-1,0), NAVY),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, PALE]),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), pad), ("BOTTOMPADDING", (0,0), (-1,-1), pad),
            ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ("LINEBELOW", (0,0), (-1,0), .6, NAVY),
            ("LINEBELOW", (0,1), (-1,-1), .4, colors.HexColor("#d5dfe7")),
        ]
        if highlight is not None:
            instructions.append(("BACKGROUND", (0,highlight), (-1,highlight), colors.HexColor("#e3eef7")))
        t.setStyle(TableStyle(instructions))
        _, h = t.wrap(self.usable, self.height)
        assert y-h >= 43, (self.page, "table overflow", y, h)
        t.drawOn(self.canvas, self.margin, y-h)
        self.layout.append({"page": self.page, "kind": "table", "top": y, "bottom": y-h})
        return y-h-12


def metric_header():
    labels = []
    for name in METRICS:
        text = name.replace("Mean Dice", "Mean<br/>Dice")
        arrow = "↓" if name in ("FPR", "FNR") else "↑" if name in ("Dice", "IoU", "mIoU", "Mean Dice") else ""
        labels.append(text + (f' <font name="Latin">{arrow}</font>' if arrow else ""))
    return labels


def link(url, label):
    return f'<link href="{escape(url, quote=True)}" color="#245f9a"><u>{escape(label)}</u></link>'


def build_pdf(output, ref, ours, audit):
    p = ComparisonPDF(output, ref["retrieval_cutoff"])
    y = p.begin("本研究與既有模型的數據並列", "九項指標 | 本研究使用 val | 其他模型分數依使用者圖片轉錄")
    y = p.paragraph("<b>評估條件未統一。</b>本研究是方框區域像素評估；圖片未提供其他模型的資料集、規模或評估方式。此表可呈現各來源數字，尚不能據此判定模型優劣。", y, gap=12)
    rows = [["Model", "Year", "Venue", *metric_header()]]
    rows.append(["Ours (val) [L]<br/><font size=8>last0912 / S08</font>", "2026*", "-", *[f'{ours["val"][m]:.4f}' for m in METRICS]])
    for row in ref["reference_rows"]:
        rows.append([f'{row["Model"]} [U]', row["Year"], row["Venue"], *[f'{row[m]:.4f}' for m in METRICS]])
    widths = [102, 39, 62, *[(p.usable-203)/9]*9]
    y = p.table(rows, y, widths, size=9.3, pad=7, highlight=1)
    y = p.paragraph("數值範圍 0-1，顯示四位小數。箭頭表示指標方向。* 2026 為本次評估整理年份，非論文發表年；本研究 Venue 未填。圖片中紅線遮住的舊 Ours 列未採用。", y, "small")
    y = p.paragraph("分數來源與適用範圍", y, "section", gap=5)
    rows = [["標示", "資料與來源", "可以確認的內容 / 尚缺資訊"],
            ["[L] 本機核查", "貨櫃 8 類；val 160/174 張<br/>核查後 summary.csv / audit.json", "九項數值可由 TP、FP、FN、TN 重算。<br/>矩形區域代理評估，未代表真實塗色遮罩精度。"],
            ["[U] 圖片轉錄", "SegFormer、LRFormer、SegMAN<br/>1791524499680.jpg 的可見三列", "數字與圖片一致；未能在已查閱原始論文核實實驗來源。<br/>缺資料集、切分、模型規模、GT 與計算方式。"]]
    y = p.table(rows, y, [100, 255, p.usable-355], size=9.2, pad=6)
    p.paragraph("LRFormer 的 Venue 依作者官方資料更正為 <b>IEEE TPAMI（2025）</b>，圖中的 TIP 誤植未沿用。三篇論文可查證的公開基準分數另列於第 3 頁；身分查證不代表圖片九項分數已獲論文證實。", y, "small", gap=0)

    y = p.begin("本研究：train、val 與合併結果", "同一 checkpoint / S08 設定 | 七項比例採 micro，mIoU 與 Mean Dice 採逐類 macro")
    rows = [["Metric", "train", "val（主列）", "train + val"]]
    rows += [[m, *[f'{ours[g][m]:.4f}' for g in GROUPS]] for m in METRICS]
    rows.append(["全部預測背景 Accuracy", *[f'{float(ours[g]["always_background_accuracy"]):.4f}' for g in GROUPS]])
    y = p.table(rows, y, [220, *[(p.usable-220)/3]*3], size=10, pad=5)
    rows = [["資料組", "推論張數", "有效評估筆數", "獨立可評估照片", "macro 納入類別"]]
    rows += [[LABELS[g], ours[g]["image_count"], ours[g]["evaluated_images"], ours[g]["unique_evaluated_photos"], f'{ours[g]["active_macro_classes"]}/8'] for g in GROUPS]
    y = p.table(rows, y, [170, 135, 155, 170, p.usable-630], size=9.5, pad=5)
    y = p.paragraph("填滿預測框與 XML 框後，按類別取聯集並計算像素計數；mIoU 與 Mean Dice 不納入預測與 GT 都不存在的類別。合併分數由 train 與 val 的計數相加後重算，不是兩組分數的平均。", y, "small", gap=6)
    p.paragraph("train 與合併組包含訓練資料，不能代替 val。整批 1,050 張中有 434 張因缺少、歧義或無效 XML 未評估；616 筆有效輸出對應 524 張獨立照片，且有 37 組相同照片存在不同標記。高 Accuracy 也受到大量負像素影響。", y, "small", gap=0)

    y = p.begin("原始研究的公開基準與來源", "僅列原文可回溯的 mIoU（%） | 公開語意分割資料集，與本研究貨櫃資料不同")
    rows = [["Model / source", "ADE20K", "Cityscapes", "COCO-Stuff", "選用欄位與證據位置"]]
    notes = ["Single-scale；Table 1(a), p.7", "原文報告；Tables 3-5, pp.6-7", "原文報告；Table 2, arXiv v2"]
    for i, row in enumerate(ref["published_benchmarks"]):
        rows.append([f'{row["Model"]} [{i+1}]', *[f'{row[k]:.1f}' for k in ("ADE20K_mIoU_percent", "Cityscapes_mIoU_percent", "COCO_Stuff_mIoU_percent")], notes[i]])
    y = p.table(rows, y, [164, 108, 108, 108, p.usable-488], size=10, pad=8)
    y = p.paragraph("三列選用 B5、B、B 的指定版本。這裡沒有統一模型大小、訓練配方或完整測試流程，也未重跑這些模型；只能作公開基準參考。它們不是第 1 頁圖片九指標的原始出處，不能拿來填補或改寫未知成績。", y, "small")
    y = p.paragraph("原始來源（核查截至 2026-10-09）", y, "section", gap=6)
    for paper in ref["papers"]:
        names = paper["authors"].split(";")[0] + " et al." if paper["id"] != "3" else "Yunxiang Fu, Meng Lou, Yizhou Yu"
        y = p.paragraph(f'[{paper["id"]}] {escape(names)} {escape(paper["title"])}. <b>{escape(paper["venue"])}</b>.<br/>{link(paper["url"], "原文 / 版本化作者稿")}；{link(paper["metadata_url"], "正式頁面 / 作者官方程式庫")}。證據位置：{escape(paper["location"])}。', y, "small", gap=9)
    p.paragraph("要形成可作優劣判斷的比較，須讓三個模型在相同貨櫃 val 圖片、相同經確認的 GT 與同一評估程式上運行。真正的分割比較需要逐像素／多邊形 GT 及完整預測遮罩；現有批次矩形代理分數不足以完成此比較。", y, "small", gap=0)
    p.canvas.save()
    return p.layout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--reference-data", type=Path, default=RESEARCH / "reference_data.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/research_comparison_20261009")
    parser.add_argument("--pdf", type=Path, default=ROOT / "output/pdf/model_research_comparison_20261009.pdf")
    args = parser.parse_args(argv)
    ref = json.loads(args.reference_data.read_text(encoding="utf-8"))
    ours, audit, settings, checks = load_verified_local(args.report, args.audit)
    ledger = csv_read(RESEARCH / "research/source-ledger.csv")
    claims = csv_read(RESEARCH / "research/claim-evidence-matrix.csv")
    ids = {r["source_id"] for r in ledger}
    assert len(ids) == len(ledger)
    for claim in claims:
        assert set(claim["source_ids"].split(";")) <= ids
        assert all(claim[key] for key in ("claim", "evidence_location", "status"))
    for row in ref["reference_rows"]:
        assert row["metric_provenance"] == "user_image_unverified_experiment"
        assert row["evaluation_basis"] == "unknown"
        assert all(0 <= row[m] <= 1 for m in METRICS)
    assert next(r for r in ref["reference_rows"] if r["Model"] == "LRFormer")["Venue"] == "TPAMI"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    local = {"Model": "Ours (val): last0912 / S08", "Year": 2026, "Venue": "-",
             "year_basis": "evaluation_year_not_publication", "dataset": "container_damage_8_classes",
             "variant": "last0912.pt / S08_diagnostic_only", "split": "val", "evaluation_basis": "boxes",
             "evaluated_images": int(ours["val"]["evaluated_images"]),
             "total_images": int(ours["val"]["image_count"]), "identity_source": "S07;S08",
             "metric_source": "S07;S08", "metric_provenance": "local_recomputed_box_region_proxy",
             "comparable_head_to_head": False, **{m: ours["val"][m] for m in METRICS}}
    main_rows = [local, *[{**r, "year_basis": "publication_year", "split": "unknown",
                            "evaluated_images": "NA", "total_images": "NA", "comparable_head_to_head": False}
                           for r in ref["reference_rows"]]]
    fields = ["Model", "Year", "Venue", *METRICS, "year_basis", "dataset", "variant", "split",
              "evaluation_basis", "evaluated_images", "total_images", "metric_provenance",
              "metric_source", "identity_source", "comparable_head_to_head", "venue_correction"]
    csv_write(args.output_dir / "comparison.csv", main_rows, fields)
    split_rows = [ours[g] for g in GROUPS]
    csv_write(args.output_dir / "ours_splits.csv", split_rows, list(split_rows[0]))
    csv_write(args.output_dir / "published_benchmarks.csv", ref["published_benchmarks"], list(ref["published_benchmarks"][0]))
    layout = build_pdf(args.pdf, ref, ours, audit)
    inputs = [args.audit, args.reference_data, args.report / "settings.json", Path(ref["user_image"])]
    inputs += [args.report / g / name for g in GROUPS for name in ("summary.csv", "class_metrics.csv")]
    manifest = {"date": ref["retrieval_cutoff"], "pdf": str(args.pdf.resolve()), "pages": 3,
                "comparison_kind": "source_labelled_descriptive_not_head_to_head",
                "main_local_group": "val", "external_nine_metrics_original_experiment_verified": False,
                "local_formula_checks": checks, "ledger_rows": len(ledger), "claims": len(claims),
                "ledger_references_valid": True, "layout": layout,
                "inputs": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in inputs],
                "records": main_rows, "published_benchmarks": ref["published_benchmarks"]}
    (args.output_dir / "comparison_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = """# 本研究與其他研究比較表

主表使用核查後的 val（160/174 張）；train 與合併資料另列。
本研究九項數值為方框區域像素代理指標，不能視為完整遮罩分割精度。
其他模型的九項值轉錄自使用者提供圖片；尚未核實資料集、規模、切分和評估方法。
模型身分另依原始論文／作者資料核對：LRFormer 為 IEEE TPAMI 2025，圖片 TIP 已修正。
因此主表沒有同條件排名、最佳值加粗或提升百分比。

- comparison.csv：主表及每列資料來源狀態，保留本機分數的原始精度。
- ours_splits.csv：train、val、train_and_val，含有效筆數與獨立照片數。
- published_benchmarks.csv：原始研究指定版本的 ADE20K / Cityscapes / COCO-Stuff mIoU（%）。
- comparison_table.png：PDF 第 1 頁的完整可分享圖片，附來源與評估限制。
- comparison_manifest.json：輸入 SHA-256、算式驗證、完整原始精度及來源狀態。

來源與證據對應保存在 research/model_comparison_20261009/research/。
這次只整理既有結果，沒有重新推論或訓練任何模型。
重新匯出：python tools/compare_research_metrics.py
PDF 使用 ReportLab；中文字型為 Windows Microsoft JhengHei。
PNG 由 Poppler 渲染 PDF 第 1 頁，應在 PDF 更新後重新產生。
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"pdf": str(args.pdf.resolve()), "output_dir": str(args.output_dir.resolve()),
                      "pages": 3, "numeric_checks": "passed", "ledger_checks": "passed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
