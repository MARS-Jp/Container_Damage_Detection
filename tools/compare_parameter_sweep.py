"""Export scientific comparison figures from saved parameter-sweep predictions.

    python tools/compare_parameter_sweep.py --run-dir PATH --reference-run EARLIER_PATH

Predictions are displayed unchanged in aligned panels. Captions sit outside the
photographs; all crops use the same normalized coordinates. No mask is inferred
from JPEG colors. Counts and areas come from prediction_metrics.json, if available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib import pyplot as plt
from PIL import Image, ImageOps

DETAIL_CROP = (0.25, 0.35, 0.85, 0.95)


def read_records(run_dir: Path, versions: list[str] | None = None, reference: bool = False) -> list[dict]:
    """Include completed jobs only, associating metrics with the exact source image."""
    jobs = json.loads((run_dir / "parameters.json").read_text(encoding="utf-8"))
    with (run_dir / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        successful = {row["output"] for row in csv.DictReader(handle) if row["status"] == "ok"}
    version_names = list(dict.fromkeys(job["version"] for job in jobs))
    records = []
    for job in jobs:
        if job["output"] not in successful or (versions and job["version"] not in versions):
            continue
        for image_path in sorted(Path(job["output"]).rglob("*__all.jpg")):
            metrics_path = image_path.parent / "prediction_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
            source = metrics.get("source", job["source"])
            # Older runs with a directory as source lack an unambiguous per-image identity.
            if not metrics and Path(source).is_dir():
                continue
            records.append(
                {
                    "job": job,
                    "image": image_path,
                    "metrics": metrics,
                    "source": source,
                    "key": (str(Path(job["weight"]).resolve()).casefold(), str(Path(source).resolve()).casefold()),
                    "label": f"{'前次 ' if reference else ''}V{version_names.index(job['version']) + 1:02d}",
                    "reference": reference,
                }
            )
    return records


def get_class_metric(record: dict, name: str, field: str):
    for metric in record["metrics"].get("classes", {}).values():
        if metric["name"].casefold() == name.casefold():
            return metric[field]
    return ""


def panel_caption(record: dict) -> str:
    if record.get("caption"):
        return record["caption"]
    job = record["job"]
    thresholds = job.get("class_conf", {})
    dent = thresholds.get("D", job["conf"])
    rust = thresholds.get("R", job["conf"])
    text = f"{record['label']} | {job['imgsz']} | 門檻 D:{dent:g}, R:{rust:g} | IoU {job['iou']:g}"
    if record["metrics"]:
        text += f"\n預測數量：D {get_class_metric(record, 'D', 'count')} / R {get_class_metric(record, 'R', 'count')}"
    else:
        text += "\n前次結果參照"
    return text


def find_panel_image(record: dict, kind: str) -> Path | None:
    if kind == "all":
        return record["image"]
    stem = record["image"].name.removesuffix("__all.jpg")
    matches = list(record["image"].parent.glob(f"{stem}__class_*_{kind}.jpg"))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous class image: {record['image'].parent}, class {kind}")
    return matches[0] if matches else None


def draw_figure(
    records: list[dict],
    output: Path,
    kind: str,
    crop: tuple | None = None,
    crop_label: str = "中下部同位置放大",
) -> None:
    """Place exact result images in a labeled grid, including the unannotated input."""
    panels = [(Path(records[0]["source"]), "原圖\n無預測塗色")]
    panels.extend((find_panel_image(record, kind), panel_caption(record)) for record in records)
    columns = 3
    rows = math.ceil(len(panels) / columns)
    figure = plt.figure(figsize=(18, rows * 5.8))
    figure.patch.set_facecolor("#f3f5f7")
    # Give captions their own rows so multi-line titles cannot cover photographs.
    grid = figure.add_gridspec(
        rows * 2,
        columns,
        height_ratios=[0.55, 4.5] * rows,
        left=0.02,
        right=0.985,
        bottom=0.055,
        top=0.91,
        hspace=0.10,
        wspace=0.055,
    )
    for index, (image_path, caption) in enumerate(panels):
        row, column = divmod(index, columns)
        caption_axis = figure.add_subplot(grid[row * 2, column])
        caption_axis.text(0, 0.15, caption, fontsize=10.5, ha="left", va="bottom")
        caption_axis.axis("off")
        axis = figure.add_subplot(grid[row * 2 + 1, column])
        if image_path and image_path.is_file():
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                if crop:
                    left, top, right, bottom = crop
                    image = image.crop(
                        (
                            round(left * image.width),
                            round(top * image.height),
                            round(right * image.width),
                            round(bottom * image.height),
                        )
                    )
                axis.imshow(image)
        else:
            axis.text(0.5, 0.5, "此類別沒有輸出圖片", ha="center", va="center")
        axis.axis("off")
    category = {"all": "全部類別", "D": "D 凹陷", "R": "R 鏽蝕"}[kind]
    detail = f"｜{crop_label}" if crop else ""
    figure.suptitle(f"{Path(records[0]['source']).stem}｜{category}參數比較{detail}", fontsize=19, y=0.995)
    figure.text(
        0.02, 0.007, "D：橘色　R：綠色｜顏色與透明度固定；參數和數量放在照片外。預測數量不代表準確率。", fontsize=10
    )
    figure.savefig(output, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def create_comparisons(
    run_dir: Path,
    reference_run: Path | None = None,
    reference_versions: list[str] | None = None,
) -> list[Path]:
    """Export all/D/R panels, identical-location details, and an auditable summary."""
    font_path = Path("C:/Windows/Fonts/msjh.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    current = read_records(run_dir)
    references = read_records(reference_run, reference_versions, True) if reference_run else []
    groups = {}
    for record in current:
        groups.setdefault(record["key"], []).append(record)
    if not groups:
        raise ValueError(f"No successful image outputs in {run_dir}")
    comparison_dir = run_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    rows = []
    report = [
        "# 第二輪推論參數比較",
        "",
        "所有照片維持原本的類別顏色與 0.35 透明度；照片內不顯示文字、信心分數或框線。",
        "比較圖的參數與數量只放在照片外，裁切圖使用同一位置。",
        "",
        "D/R 採不同門檻時，先用最低門檻產生候選，再對完整預測（框與遮罩一起）套用各類別門檻。",
        "未指定的類別沿用全域門檻，NMS 維持依類別處理。",
        "",
        "數量與覆蓋率描述預測，沒有真實標註時不能據此宣稱準確率或選出最佳模型。",
        "",
    ]
    for index, (key, records) in enumerate(groups.items(), 1):
        records = records + [record for record in references if record["key"] == key]
        group_dir = comparison_dir / f"{index:02d}_{Path(records[0]['source']).stem}"
        group_dir.mkdir(parents=True, exist_ok=True)
        for filename, kind, crop in (
            ("all_variants.png", "all", None),
            ("dent_variants.png", "D", None),
            ("rust_variants.png", "R", None),
            ("detail_all.png", "all", DETAIL_CROP),
            ("detail_dent.png", "D", DETAIL_CROP),
        ):
            output = group_dir / filename
            draw_figure(records, output, kind, crop)
            generated.append(output)
        report.extend(
            [
                f"## {Path(records[0]['source']).name}",
                "",
                f"權重：`{records[0]['job']['weight']}`",
                "",
                (
                    f"[全圖比較]({group_dir.name}/all_variants.png) · [D 比較]({group_dir.name}/dent_variants.png) · "
                    f"[R 比較]({group_dir.name}/rust_variants.png) · [中下部放大]({group_dir.name}/detail_all.png)"
                ),
                "",
                "| 版本 | imgsz | D 門檻 | R 門檻 | IoU | D 數量 | R 數量 | D 覆蓋率 % | R 覆蓋率 % |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for record in records:
            job = record["job"]
            thresholds = job.get("class_conf", {})
            row = {
                "label": record["label"],
                "version": job["version"],
                "weight": job["weight"],
                "source": record["source"],
                "imgsz": job["imgsz"],
                "D_conf": thresholds.get("D", job["conf"]),
                "R_conf": thresholds.get("R", job["conf"]),
                "iou": job["iou"],
                "D_count": get_class_metric(record, "D", "count"),
                "R_count": get_class_metric(record, "R", "count"),
                "D_coverage_percent": get_class_metric(record, "D", "coverage_percent"),
                "R_coverage_percent": get_class_metric(record, "R", "coverage_percent"),
                "output": job["output"],
            }
            rows.append(row)
            columns = (
                "label",
                "imgsz",
                "D_conf",
                "R_conf",
                "iou",
                "D_count",
                "R_count",
                "D_coverage_percent",
                "R_coverage_percent",
            )
            report.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
        report.append("")
    with (comparison_dir / "comparison_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (comparison_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Saved {len(generated)} comparison figures to: {comparison_dir}")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--reference-versions", nargs="+", default=["imgsz960"])
    args = parser.parse_args()
    create_comparisons(args.run_dir, args.reference_run, args.reference_versions)


if __name__ == "__main__":
    main()
