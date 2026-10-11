"""Export the verified inference audit and corrected split tables as a PDF."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from html import escape
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "tools/runs/metric_audits/audit_20261008_235145_226565"
REPORT = ROOT / "tools/runs/nine_metric_reports/report_20261008_235420_507363"
METRICS = ("Accuracy", "Precision", "Recall", "Dice", "IoU", "mIoU", "Mean Dice", "FPR", "FNR")
GROUPS = (("train", "只有 train"), ("val", "只有 val"), ("train_and_val", "綜合 train + val"))
NAVY = colors.HexColor("#173f69")
INK = colors.HexColor("#203247")
MUTED = colors.HexColor("#596a7d")
PALE = colors.HexColor("#f0f5fa")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def percent(value):
    return "NA" if value is None or value == "NA" else f"{float(value) * 100:.2f}%"


class AuditPDF:
    def __init__(self, output):
        pdfmetrics.registerFont(TTFont("CJK", "C:/Windows/Fonts/msjh.ttc", subfontIndex=0))
        pdfmetrics.registerFont(TTFont("CJKBold", "C:/Windows/Fonts/msjhbd.ttc", subfontIndex=0))
        self.width, self.height = landscape(A4)
        self.margin = 36
        self.usable = self.width - self.margin * 2
        self.canvas = canvas.Canvas(str(output), pagesize=(self.width, self.height), pageCompression=1)
        self.canvas.setTitle("推論數據核查報告 - last0912 / S08")
        self.canvas.setAuthor("")
        self.canvas.setSubject("方框區域評估、實際遮罩對照與 XML 標記一致性核查")
        self.page = 0
        self.date = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")
        self.body = ParagraphStyle(
            "body", fontName="CJK", fontSize=11, leading=18, textColor=INK, wordWrap="CJK", splitLongWords=True
        )
        self.small = ParagraphStyle("small", parent=self.body, fontSize=9.5, leading=15)

    def begin(self, title, subtitle):
        if self.page:
            self.canvas.showPage()
        self.page += 1
        c = self.canvas
        c.bookmarkPage(f"p{self.page}")
        c.addOutlineEntry(title, f"p{self.page}", level=0)
        c.setFillColor(NAVY)
        c.setFont("CJKBold", 22)
        c.drawString(self.margin, self.height - 48, title)
        c.setFillColor(MUTED)
        c.setFont("CJK", 10)
        c.drawString(self.margin, self.height - 70, subtitle)
        c.setStrokeColor(colors.HexColor("#dce3eb"))
        c.line(self.margin, 31, self.width - self.margin, 31)
        c.setFont("CJK", 9)
        c.drawString(self.margin, 17, f"{self.date}  |  01_last0912 / S08_diagnostic_only")
        c.drawRightString(self.width - self.margin, 17, str(self.page))
        return self.height - 95

    def paragraph(self, text, y, small=False, bold=False):
        text = f"<b>{text}</b>" if bold else text
        obj = Paragraph(text, self.small if small else self.body)
        _, height = obj.wrap(self.usable, self.height)
        assert y - height >= 45, (self.page, y, height, text[:100])
        obj.drawOn(self.canvas, self.margin, y - height)
        return y - height - 10

    def table(self, rows, y, widths=None, size=10, padding=8):
        widths = widths or [self.usable / len(rows[0])] * len(rows[0])
        normal = ParagraphStyle(
            "cell", fontName="CJK", fontSize=size, leading=size * 1.35, textColor=INK, wordWrap="CJK", alignment=2
        )
        head = ParagraphStyle("head", parent=normal, fontName="CJKBold", textColor=colors.white)
        prepared = [
            [Paragraph(escape(str(value)), head if i == 0 else normal) for value in row] for i, row in enumerate(rows)
        ]
        table = Table(prepared, colWidths=widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), padding),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, NAVY),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#dce3eb")),
                ]
            )
        )
        _, height = table.wrap(self.usable, self.height)
        assert y - height >= 45, (self.page, y, height)
        table.drawOn(self.canvas, self.margin, y - height)
        return y - height - 16

    def image(self, image, y):
        width = self.usable
        height = width * image.height / image.width
        assert y - height >= 45, (self.page, y, height)
        self.canvas.drawImage(ImageReader(image), self.margin, y - height, width=width, height=height)
        return y - height - 10


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/inference_metric_audit_20261009.pdf")
    args = parser.parse_args(argv)
    audit, report = args.audit.resolve(), args.report.resolve()
    data = json.loads((audit / "audit.json").read_text(encoding="utf-8"))
    groups = data["groups"]
    # Export only the corrected report; never reuse the superseded PDF tables.
    settings = json.loads((report / "settings.json").read_text(encoding="utf-8"))
    assert settings["mode"] == "boxes" and settings["single_class_macro_policy"] == "not_applicable_NA"
    classes = {group: read_csv(report / group / "class_metrics.csv") for group, _ in GROUPS}
    for rows in classes.values():
        assert all(row["mIoU"] == row["Mean Dice"] == "NA" for row in rows)
    for group, _ in GROUPS:
        row = read_csv(report / group / "summary.csv")[0]
        assert int(row["evaluated_images"]) == groups[group]["evaluated"]
        assert all(abs(float(row[key]) - groups[group]["box_pixel_metrics"][key]) < 1e-12 for key in METRICS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdf = AuditPDF(args.output)
    y = pdf.begin("推論數據核查報告", f"核查來源：2026-10-08；PDF 整理日期：{pdf.date}")
    y = pdf.paragraph("結論：原方框像素分數可重現，但不能代表實際塗色遮罩的分割精度。", y, bold=True)
    y = pdf.paragraph(
        "已核對 1,050 張原圖與推論輸入快照，並獨立重算 616 張有效圖片的 4,928 個逐類計數。算式與原報表一致；異常感主要來自大量背景、評估區域不同，以及標記版本與重複照片的衝突。",
        y,
    )
    rows = [["資料組", "推論圖片", "可評估", "未評估", "評估涵蓋率", "獨立可評估照片"]]
    for group, label in GROUPS:
        item = groups[group]
        rows.append(
            [
                label,
                item["images"],
                item["evaluated"],
                item["images"] - item["evaluated"],
                percent(item["coverage"]),
                data["unique_evaluated_photos"][group],
            ]
        )
    y = pdf.table(rows, y, size=10)
    y = pdf.paragraph("1. 合併 Accuracy 為 97.78%，全部預測背景也可得到 97.02%，只高 0.76 個百分點。", y)
    y = pdf.paragraph(
        "2. 原表填滿預測框與 XML 框後比較，沒有衡量實際遮罩輪廓；同類碎片與重複框取聯集後，也不會逐一受罰。", y
    )
    y = pdf.paragraph(
        "3. 依現有 XML、同類且 IoU &gt;= 0.5 做一對一框匹配，val Recall 僅 6.74%；高 Accuracy 不能代表損傷皆被找到。", y
    )
    y = pdf.paragraph(
        "4. 找到 37 組相同照片、不同 XML 標記，涉及 88 筆；範例圖的舊 XML 有 31 個物件，新 XML 有 94 個。", y
    )
    pdf.paragraph(
        "整批真正的分割精度尚未驗證：仍需全部原始預測遮罩與經確認、對應原圖的完整分割標記。本報告未修改模型、推論結果或標記內容。",
        y,
        small=True,
    )

    y = pdf.begin("三組九項指標", "評估基礎：方框區域像素；數值為百分比，非實際分割遮罩精度")
    rows = [["指標", "train（456 張）", "val（160 張）", "train + val（616 張）"]]
    for metric in METRICS:
        rows.append([metric, *[percent(groups[group]["box_pixel_metrics"][metric]) for group, _ in GROUPS]])
    rows.append(
        [
            "全部背景 Accuracy",
            *[percent(groups[group]["box_pixel_metrics"]["always_background_Accuracy"]) for group, _ in GROUPS],
        ]
    )
    rows.append(
        [
            "高於背景基準（百分點）",
            *[
                f"{(groups[group]['box_pixel_metrics']['Accuracy'] - groups[group]['box_pixel_metrics']['always_background_Accuracy']) * 100:.4f}"
                for group, _ in GROUPS
            ],
        ]
    )
    y = pdf.table(rows, y, widths=[205, *[(pdf.usable - 205) / 3] * 3], size=10, padding=5)
    y = pdf.paragraph(
        "七項比例先加總各類別的 TP、FP、FN、TN 再計算（micro）；mIoU、Mean Dice 為加總後逐類 IoU、Dice 的等權平均（macro），排除預測與標記都不存在的類別。背景不加入 macro，但各類別的負像素仍計入 Accuracy。",
        y,
        small=True,
    )
    pdf.paragraph(
        "合併組直接加總 train 與 val 的計數，沒有平均兩組分數。缺輸入或分母為零時為 NA，不補零；逐類 mIoU／Mean Dice 不適用，已改為 NA。",
        y,
        small=True,
    )

    y = pdf.begin("物件命中率與區域面積對照", "物件匹配：同類、依信心分數排序、IoU >= 0.5、每個物件最多匹配一次")
    rows = [["資料組", "TP", "未匹配預測", "FN", "Precision", "Recall", "F1"]]
    for group, label in GROUPS:
        item = groups[group]["box_object_metrics"]
        rows.append(
            [label, item["TP"], item["FP"], item["FN"], *[percent(item[key]) for key in ("Precision", "Recall", "F1")]]
        )
    y = pdf.table(rows, y)
    y = pdf.paragraph(
        "合併方框像素 Precision 為 66.25%，物件框 Precision 為 28.23%。兩者衡量目標與分母不同；像素聯集比例不能代替每個預測物件的命中率。未匹配預測也可能來自漏標、定位偏差或物件拆分差異，不能全數判為視覺誤檢。",
        y,
    )
    area = read_csv(audit / "prediction_box_vs_mask_areas.csv")
    rows = [["類別", "實際遮罩聯集像素", "報表方框聯集像素", "方框多出的面積"]]
    for name in ("D", "R"):
        actual = sum(int(row["actual_mask_union_pixels"]) for row in area if row["class"] == name)
        box = sum(int(row["report_box_union_pixels"]) for row in area if row["class"] == name)
        rows.append([name, f"{actual:,}", f"{box:,}", percent(box / actual - 1)])
    y = pdf.table(rows, y)
    y = pdf.paragraph(
        "面積比較限於同一批 616 張有效圖片。實際遮罩面積來自推論當時保存的 JSON；矩形聯集則由預測框重新重建。面積總數不能還原遮罩的形狀與空間交集。",
        y,
        small=True,
    )
    pdf.paragraph(
        "S08 是低分診斷設定：原圖 imgsz=640、NMS IoU=0.7；全圖保留 conf&gt;=0.15，固定左右區域追加低至 0.01 的 D/R。它可增加候選物件，不能保證定位或分割更準。",
        y,
        small=True,
    )

    sample = data["sample"]
    picture = Image.open(audit / "sample_boxes_vs_masks.png").convert("RGB")
    half = picture.height // 2
    y = pdf.begin("方框填滿與原圖對照", "診斷樣本：CBHU0708054-A；方框像素評估區域與實際遮罩不同")
    y = pdf.image(picture.crop((0, 0, picture.width, half)), y)
    rows = [["範例評估方式", "Precision", "Recall", "Dice", "IoU"]]
    for row, label in zip(sample["comparisons"], ("預測框 / 舊 XML 框", "預測遮罩 / 舊多邊形")):
        rows.append([label, *[percent(row[key]) for key in ("Precision", "Recall", "Dice", "IoU")]])
    y = pdf.table(rows, y, widths=[210, *[(pdf.usable - 210) / 4] * 4], size=9.5, padding=5)
    pdf.paragraph(
        "兩列使用不同幾何型態的舊標記；只能作本圖對照，不能代替全量分割成績。實際遮罩與舊多邊形填色見下一頁。",
        y,
        small=True,
    )

    y = pdf.begin("實際遮罩與舊分割標記對照", "橘色 D＝凹陷；綠色 R＝鏽蝕。右圖為現有舊版多邊形，不是完整新標記")
    y = pdf.image(picture.crop((0, half, picture.width, picture.height)), y)
    rows = [["標記版本", "凹陷 D", "鏽蝕 R", "總物件", "一對一框 Recall"]]
    for version, label in (("old", "舊 XML"), ("new", "新 XML（0928-1）")):
        counts = sample[f"{version}_XML_object_counts"]
        rows.append(
            [
                label,
                counts["D"],
                counts["R"],
                sum(counts.values()),
                percent(sample[f"{version}_XML_object_matching"]["Recall"]),
            ]
        )
    y = pdf.table(rows, y, size=9.5, padding=5)
    pdf.paragraph(
        "原始遮罩的聯集像素與舊 JSON 相同；重跑 JPEG 與原結果 SHA256 相同；舊多邊形對應照片亦與來源照片 SHA256 相同。新 XML 只有框，仍不能作真實分割邊界評估。",
        y,
        small=True,
    )

    for group, label in GROUPS:
        y = pdf.begin(f"逐類九項指標：{label}", "方框區域像素評估；N 標記＝該類別有真值正像素的圖片筆數")
        rows = [["類別", "N 標記", *METRICS]]
        for row in classes[group]:
            rows.append([row["class"], row["gt_positive_images"], *[percent(row[key]) for key in METRICS]])
        widths = [32, 54, 74, 72, 64, 64, 64, 64, 82, 63, 63]
        widths = [value * pdf.usable / sum(widths) for value in widths]
        y = pdf.table(rows, y, widths=widths, size=9, padding=8)
        y = pdf.paragraph(
            "mIoU、Mean Dice 是跨類別平均，對單一類別不適用。原表把整體分數重複填到每個類別，容易誤讀；此版保留九欄，但這兩欄改成 NA。",
            y,
        )
        y = pdf.paragraph(
            "Accuracy 與 FPR 使用該類別的全部正負像素。沒有預測、沒有標記時，Accuracy 可為 100%、FPR 可為 0；Precision、Recall、Dice、IoU 等仍為 NA，不能視為偵測成功。",
            y,
        )
        if group == "train_and_val":
            pdf.paragraph(
                "U 類別只有 4 張有標記的獨立照片。少量資料的高分不足以證明泛化能力；有重複照片的類別也不能將圖片筆數視為獨立樣本數。",
                y,
                small=True,
            )
        else:
            pdf.paragraph(
                "資料集先加總該類別的像素計數再計算，不是每張圖片分數的直接平均。類別代碼沿用模型；已知 D 為凹陷、R 為鏽蝕。",
                y,
                small=True,
            )

    y = pdf.begin("標記一致性與資料核對", "已確認數值可重現，但尚未確認所有標記內容完整且正確")
    y = pdf.paragraph(
        "1,050 張原圖與保存的輸入 PNG，經與推論相同的 OpenCV 解碼後全數一致，其中 15 張有 EXIF 轉向。292 張可取得標記來源照片，SHA256 全部相同；另 324 張缺少對應標記照片，僅核對檔名與尺寸。",
        y,
    )
    y = pdf.paragraph(
        "有效的 616 筆圖片資料只涵蓋 524 張獨立照片。全部來源有 172 組重複照片；有效 XML 有 37 組同照片標記內容不同，涉及 88 筆。以下列出三組實例，未自行挑選或合併真值。",
        y,
    )
    conflicts = read_csv(audit / "duplicate_annotation_conflicts.csv")
    selected = list(dict.fromkeys(row["sha256"] for row in conflicts))[:3]
    rows = [["相同照片 SHA256 前 12 碼", "來源圖片識別", "標記正像素"]]
    for row in conflicts:
        if row["sha256"] in selected:
            rows.append([row["sha256"][:12], row["image_id"], f"{int(row['annotated_positive_pixels']):,}"])
    y = pdf.table(rows, y, widths=[195, pdf.usable - 345, 150], size=9.5, padding=6)
    y = pdf.paragraph(
        "未評估 434 張：缺 XML 360 張、XML 配對有歧義 73 張、XML 無效 1 張。未評估圖片仍保留在逐張 CSV，不補成背景或零分。",
        y,
    )
    pdf.paragraph(
        "沒有找到 train/val 間 SHA256 完全相同照片；這不排除同一貨櫃不同視角的洩漏。標記衝突與缺漏尚未修復，因此本報告不能作為模型最終驗收成績。",
        y,
        small=True,
    )

    y = pdf.begin("計算方式、修正內容與來源", "計數單位為像素或物件，需依評估方式區分；本 PDF 的分數以百分比顯示")
    rows = [["指標", "公式或範圍"]]
    for name, formula in (
        ("Accuracy", "(TP + TN) / (TP + FP + FN + TN)"),
        ("Precision", "TP / (TP + FP)"),
        ("Recall", "TP / (TP + FN)"),
        ("Dice", "2TP / (2TP + FP + FN)"),
        ("IoU", "TP / (TP + FP + FN)"),
        ("FPR", "FP / (FP + TN)"),
        ("FNR", "FN / (TP + FN)"),
        ("mIoU / Mean Dice", "逐類 IoU / Dice 的等權平均；單一類別不適用；排除分母為零的類別"),
    ):
        rows.append([name, formula])
    y = pdf.table(rows, y, widths=[170, pdf.usable - 170], size=9.5, padding=5)
    y = pdf.paragraph(
        "修正內容：移除逐類重複的宏觀分數；加入 metric_scope、標記正像素／圖片支持量、全部背景 Accuracy 及差值。原推論不變。19 項測試通過；獨立矩形重建、逐張與逐類計數及三組彙總核對一致。",
        y,
        small=True,
    )
    y = pdf.paragraph(
        "核查來源（相對專案根目錄）：tools/runs/metric_audits/"
        + escape(audit.name)
        + "/audit.json；同目錄保存逐筆來源核對、重算、物件匹配、標記衝突清單及比較圖。",
        y,
        small=True,
    )
    y = pdf.paragraph(
        "修正報表：tools/runs/nine_metric_reports/"
        + escape(report.name)
        + "/；包含 train、val、train_and_val 的 summary.csv、class_metrics.csv、image_metrics.csv、image_class_metrics.csv。逐筆 CSV 保留所有推論圖片；本 PDF 為彙總與核查說明。",
        y,
        small=True,
    )
    pdf.paragraph(
        "舊版 nine_metrics_report_20261008.pdf 未覆寫。本文件使用核查與修正後數據，仍屬方框區域評估；全量真正的分割九指標需重新保存預測 NPZ，並使用經核對的完整分割標記。",
        y,
        small=True,
    )
    pdf.canvas.save()
    print(args.output.resolve())
    print(f"Pages: {pdf.page}")


if __name__ == "__main__":
    main()
