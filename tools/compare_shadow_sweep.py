"""Create separate and cross-group comparisons from fill-only inference results.

    python tools/compare_shadow_sweep.py --run-dir PATH

Reads only completed jobs. Source identity comes from each predictions.json, so
directories, repeated filenames and multiple models remain correctly separated.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.compare_parameter_sweep import DETAIL_CROP, draw_figure, font_manager, get_class_metric, plt


GROUP_LABELS = {"shadow": "1 陰影增強", "general": "2 一般參數", "combined": "3 陰影＋一般"}
GROUP_DIRS = {"shadow": "01_shadow", "general": "02_general", "combined": "03_combined"}
# Fixed comparison choices, not a ranking inferred from prediction counts.
OVERVIEW = {
    ("shadow", "baseline"), ("shadow", "shadow_default"),
    *((group, version) for group in ("general", "combined")
      for version in ("conf010", "iou050", "imgsz960", "imgsz1280")),
    ("combined", "size960_conf010_iou050"), ("combined", "size960_gamma045_shadow128"),
}


def read_records(run_dir: Path) -> list[dict]:
    """Match each successful job to its per-image predictions, masks and parameters."""
    jobs = json.loads((run_dir / "parameters.json").read_text(encoding="utf-8"))
    with (run_dir / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        successful = {row["output"]: row for row in csv.DictReader(handle) if row["status"] == "ok"}
    records = []
    for job in jobs:
        if job["output"] not in successful:
            continue
        for report_path in sorted(Path(job["output"]).rglob("predictions.json")):
            report = json.loads(report_path.read_text(encoding="utf-8"))
            images = list(report_path.parent.glob("*__all.jpg"))
            if len(images) != 1:
                raise ValueError(f"Expected one combined image beside {report_path}, found {len(images)}")
            record = {
                "job": job, "image": images[0], "source": report["source"],
                "metrics": report.get("metrics", {}), "report": report,
                "seconds": successful[job["output"]]["seconds"],
                "key": (str(Path(job["weight"]).resolve()).casefold(), str(Path(report["source"]).resolve()).casefold()),
                "label": f"{GROUP_LABELS[job['group']]} / {job['version']}",
            }
            record["caption"] = panel_caption(record)
            records.append(record)
    return records


def panel_caption(record: dict) -> str:
    """Keep all explanatory text outside the photographs."""
    job = record["job"]
    views = "+".join(job["views"])
    enhancement = "原圖推論"
    if len(job["views"]) > 1:
        active = []
        if "gamma" in job["views"]:
            active.append(f"Gamma {job['gamma']:g}")
        if "clahe" in job["views"]:
            active.append(f"CLAHE {job['clahe_clip']:g}")
        enhancement = " / ".join(active) + f" / 暗部 {job['shadow_threshold']} / 合併 {job['merge_iou']:g}"
    return (
        f"{record['label']}\n"
        f"imgsz {job['imgsz']} / conf {job['conf']:g} / NMS IoU {job['iou']:g}\n"
        f"{views} | {enhancement}\n"
        f"預測數：{record['report']['merged_count']} / D {get_class_metric(record, 'D', 'count')}"
        f" / R {get_class_metric(record, 'R', 'count')}"
    )


def create_comparisons(run_dir: Path) -> list[Path]:
    """Export three group galleries, a fixed cross-group overview, and measured counts."""
    font_path = Path("C:/Windows/Fonts/msjh.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    records = read_records(run_dir)
    if not records:
        raise ValueError(f"No successful predictions to compare in {run_dir}")
    by_source = {}
    for record in records:
        by_source.setdefault(record["key"], []).append(record)
    output_root = run_dir / "comparisons"
    output_root.mkdir(parents=True, exist_ok=True)
    generated, rows = [], []
    report_lines = [
        "# 三組參數比較", "",
        "所有結果都使用相同的分割塗色；圖片內沒有信心分數、類別標籤、偵測框或輪廓線。",
        "三組各自保留原圖基準；baseline 使用 predict-2.py 的一般參數。",
        "一般組只推論原圖；陰影組固定一般參數；混合組搭配陰影與一般參數。", "",
        "有套用背景排除區時，設定保存在本次 run 的 exclusion_regions.json；所有版本使用同一份設定。",
        "各張 predictions.json 記錄實際匹配的 excluded_regions。這是人工指定偵測範圍，不是模型辨識能力改善。", "",
        "跨組總覽使用預先指定版本，不以預測數或覆蓋率挑選最佳結果。",
        "覆蓋率來自原始二值遮罩的同類聯集；沒有標註時無法判定哪版最準確。",
        "comparison_summary.csv 的 job_seconds 包含模型載入、該來源下所有圖片的推論與存檔；",
        "來源是資料夾時，同一工作耗時會出現在每張圖片的資料列，不能當成單張延遲。", "",
    ]
    for index, source_records in enumerate(by_source.values(), 1):
        # Prefix protects identically named images/models from overwriting one another.
        image_dir = output_root / f"{index:03d}_{Path(source_records[0]['source']).stem}"
        image_dir.mkdir(parents=True, exist_ok=True)
        report_lines.extend([
            f"## {Path(source_records[0]['source']).name}", "",
            f"模型：`{source_records[0]['job']['weight']}`", "",
        ])
        for group in GROUP_LABELS:
            selected = [r for r in source_records if r["job"]["group"] == group]
            if not selected:
                continue
            group_dir = image_dir / GROUP_DIRS[group]
            group_dir.mkdir(parents=True, exist_ok=True)
            for filename, kind, crop in (
                ("all_variants.png", "all", None), ("dent_variants.png", "D", None),
                ("rust_variants.png", "R", None), ("detail_all.png", "all", DETAIL_CROP),
                ("detail_dent.png", "D", DETAIL_CROP),
            ):
                output = group_dir / filename
                draw_figure(selected, output, kind, crop)
                generated.append(output)
            report_lines.append(f"- [{GROUP_LABELS[group]}：完整比較]({image_dir.name}/{group_dir.name}/all_variants.png)")
        overview = [r for r in source_records if (r["job"]["group"], r["job"]["version"]) in OVERVIEW]
        if not overview:
            overview = source_records
        for filename, crop in (("three_groups_overview.png", None), ("three_groups_detail.png", DETAIL_CROP)):
            output = image_dir / filename
            draw_figure(overview, output, "all", crop)
            generated.append(output)
        report_lines.extend([
            f"- [跨組總覽]({image_dir.name}/three_groups_overview.png)",
            f"- [跨組局部放大]({image_dir.name}/three_groups_detail.png)", "",
            "| 組別 | 版本 | 尺寸 | conf | IoU | 原圖預測數 | 合併數 | D 數量 | R 數量 | D 覆蓋率 % | R 覆蓋率 % |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for record in source_records:
            job, prediction = record["job"], record["report"]
            row = {key: job[key] for key in (
                "group", "version", "weight", "imgsz", "conf", "iou", "gamma", "clahe_clip",
                "shadow_threshold", "merge_iou", "max_det", "alpha",
            )}
            row.update(source=record["source"], views="+".join(job["views"]),
                       original_count=prediction["view_counts"]["original"],
                       gamma_count=prediction["view_counts"].get("gamma", ""),
                       clahe_count=prediction["view_counts"].get("clahe", ""),
                       merged_count=prediction["merged_count"],
                       D_count=get_class_metric(record, "D", "count"), R_count=get_class_metric(record, "R", "count"),
                       D_coverage_percent=get_class_metric(record, "D", "coverage_percent"),
                       R_coverage_percent=get_class_metric(record, "R", "coverage_percent"),
                       job_seconds=record["seconds"], output=str(record["image"].parent))
            rows.append(row)
            table_fields = ("group", "version", "imgsz", "conf", "iou", "original_count", "merged_count",
                            "D_count", "R_count", "D_coverage_percent", "R_coverage_percent")
            report_lines.append("| " + " | ".join(str(row[key]) for key in table_fields) + " |")
        report_lines.append("")
    with (output_root / "comparison_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "README.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Saved {len(generated)} comparison figures to: {output_root}", flush=True)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    create_comparisons(args.run_dir.resolve())


if __name__ == "__main__":
    main()
