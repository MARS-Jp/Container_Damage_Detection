"""批次測試 predict_shadow.py 的不同參數組合。.

使用方式：
    python tools/predict_parameter_sweep.py

例如只測指定參數：
    python tools/predict_parameter_sweep.py ^
        --conf-list 0.10 0.15 0.20 ^
        --gamma-list 0.4 0.5 0.6 ^
        --imgsz-list 640 960

輸出：
    runs/segment_shadow_sweep/sweep_時間/
        001_imgsz640_conf0p10_.../
        002_imgsz640_conf0p10_.../
        ...
        parameter_results.csv   # 每張圖片、每組參數的結果
        parameter_summary.csv   # 每組參數的彙整結果
        parameter_summary.json

注意：
- 權重檔不會被修改。
- 參數採 Cartesian product（笛卡兒積）產生所有組合。
- 不想掃描的參數只保留一個值即可。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path
from statistics import mean

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import predict_shadow as base

# ============================================================
# 直接改這裡就可以控制要測試的參數。
# 每個 list 會彼此排列組合。
#
# 目前預設：
#   2 imgsz × 3 conf × 1 iou × 2 gamma × 1 clahe
#   × 1 shadow_threshold × 1 merge_iou = 12 組
# ============================================================
PARAMETER_GRID = {
    "imgsz": [640, 960],
    "conf": [0.10, 0.15, 0.20],
    "iou": [0.70],
    "gamma": [0.40, 0.50],
    "clahe_clip": [1.0],
    "shadow_threshold": [255],
    "merge_iou": [0.90],
}

OUTPUT_DIR = REPO_ROOT / "runs" / "segment_shadow_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="對 predict_shadow.py 執行多組參數組合並彙整結果。")

    parser.add_argument("--weights", nargs="+", type=Path, default=base.WEIGHT_SOURCES)
    parser.add_argument("--source", nargs="+", type=Path, default=base.DATASET_SOURCES)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default=base.DEVICE)

    parser.add_argument(
        "--views",
        nargs="+",
        choices=("original", "gamma", "clahe"),
        default=["original", "gamma", "clahe"],
    )
    parser.add_argument("--save-views", action="store_true")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.35)

    # 可以不改程式，直接由命令列覆寫 sweep 清單。
    parser.add_argument("--imgsz-list", nargs="+", type=int, default=PARAMETER_GRID["imgsz"])
    parser.add_argument("--conf-list", nargs="+", type=float, default=PARAMETER_GRID["conf"])
    parser.add_argument("--iou-list", nargs="+", type=float, default=PARAMETER_GRID["iou"])
    parser.add_argument("--gamma-list", nargs="+", type=float, default=PARAMETER_GRID["gamma"])
    parser.add_argument(
        "--clahe-clip-list",
        nargs="+",
        type=float,
        default=PARAMETER_GRID["clahe_clip"],
    )
    parser.add_argument(
        "--shadow-threshold-list",
        nargs="+",
        type=int,
        default=PARAMETER_GRID["shadow_threshold"],
    )
    parser.add_argument(
        "--merge-iou-list",
        nargs="+",
        type=float,
        default=PARAMETER_GRID["merge_iou"],
    )

    exclusions = parser.add_mutually_exclusive_group()
    exclusions.add_argument(
        "--exclude-regions",
        type=Path,
        default=base.EXCLUSION_REGIONS,
    )
    exclusions.add_argument(
        "--no-exclusions",
        dest="exclude_regions",
        action="store_const",
        const=None,
    )

    args = parser.parse_args()
    validate_args(parser, args)
    args.views = list(dict.fromkeys(["original", *args.views]))
    return args


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.imgsz_list:
        parser.error("--imgsz-list 至少要有一個值")

    for imgsz in args.imgsz_list:
        if imgsz <= 0 or imgsz % 32:
            parser.error("所有 imgsz 都必須是正整數且為 32 的倍數")

    for name in ("conf_list", "iou_list", "merge_iou_list"):
        for value in getattr(args, name):
            if not 0 <= value <= 1:
                parser.error(f"{name} 的值必須介於 0 和 1")

    for value in args.gamma_list:
        if not 0 < value <= 1:
            parser.error("gamma 必須 > 0 且 <= 1")

    for value in args.clahe_clip_list:
        if value <= 0:
            parser.error("clahe-clip 必須 > 0")

    for value in args.shadow_threshold_list:
        if not 1 <= value <= 255:
            parser.error("shadow-threshold 必須介於 1..255")

    if args.max_det < 1:
        parser.error("--max-det 必須 >= 1")

    if not 0 <= args.alpha <= 1:
        parser.error("--alpha 必須介於 0 和 1")


def build_combinations(args: argparse.Namespace) -> list[dict]:
    keys = (
        "imgsz",
        "conf",
        "iou",
        "gamma",
        "clahe_clip",
        "shadow_threshold",
        "merge_iou",
    )

    values = (
        args.imgsz_list,
        args.conf_list,
        args.iou_list,
        args.gamma_list,
        args.clahe_clip_list,
        args.shadow_threshold_list,
        args.merge_iou_list,
    )

    return [dict(zip(keys, combo)) for combo in product(*values)]


def number_text(value) -> str:
    """讓浮點數安全地放進資料夾名稱，例如 0.15 -> 0p15。."""
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p")


def combo_name(index: int, params: dict) -> str:
    return (
        f"{index:03d}"
        f"_imgsz{params['imgsz']}"
        f"_conf{number_text(params['conf'])}"
        f"_iou{number_text(params['iou'])}"
        f"_gamma{number_text(params['gamma'])}"
        f"_clahe{number_text(params['clahe_clip'])}"
        f"_shadow{params['shadow_threshold']}"
        f"_merge{number_text(params['merge_iou'])}"
    )


def make_predict_args(
    cli_args: argparse.Namespace,
    params: dict,
    exclusion_rules: list[dict],
) -> argparse.Namespace:
    """建立 predict_shadow.predict_image() 所需要的參數物件。."""
    return argparse.Namespace(
        imgsz=params["imgsz"],
        conf=params["conf"],
        iou=params["iou"],
        device=cli_args.device,
        gamma=params["gamma"],
        clahe_clip=params["clahe_clip"],
        shadow_threshold=params["shadow_threshold"],
        merge_iou=params["merge_iou"],
        max_det=cli_args.max_det,
        views=cli_args.views,
        save_views=cli_args.save_views,
        alpha=cli_args.alpha,
        exclusion_rules=exclusion_rules,
    )


def summarize_report(report: dict) -> dict:
    detections = report.get("detections", [])

    confidences = [float(item["confidence"]) for item in detections if "confidence" in item]

    class_counts = Counter(item.get("class_name", str(item.get("class_id", "?"))) for item in detections)

    view_counts = Counter(item.get("view", "unknown") for item in detections)

    return {
        "merged_count": int(report.get("merged_count", len(detections))),
        "mean_confidence": mean(confidences) if confidences else 0.0,
        "max_confidence": max(confidences) if confidences else 0.0,
        "class_counts": dict(class_counts),
        "merged_view_counts": dict(view_counts),
        "raw_view_counts": report.get("raw_view_counts", {}),
        "filtered_view_counts": report.get("view_counts", {}),
        "metrics": report.get("metrics", {}),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 126)
    print("參數組合結果")
    print("=" * 126)

    header = (
        f"{'ID':>3} "
        f"{'imgsz':>5} "
        f"{'conf':>6} "
        f"{'iou':>5} "
        f"{'gamma':>6} "
        f"{'clahe':>6} "
        f"{'shadow':>6} "
        f"{'merge':>6} "
        f"{'images':>6} "
        f"{'detections':>10} "
        f"{'avg_conf':>9}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['combo_id']:>3} "
            f"{row['imgsz']:>5} "
            f"{row['conf']:>6.2f} "
            f"{row['iou']:>5.2f} "
            f"{row['gamma']:>6.2f} "
            f"{row['clahe_clip']:>6.2f} "
            f"{row['shadow_threshold']:>6} "
            f"{row['merge_iou']:>6.2f} "
            f"{row['images']:>6} "
            f"{row['total_detections']:>10} "
            f"{row['mean_confidence']:>9.4f}"
        )


def main() -> int:
    args = parse_args()

    weights = list(dict.fromkeys(base.resolve_path(p) for p in args.weights))
    sources = [base.resolve_path(p) for p in args.source]
    images = base.collect_images(sources)

    if not weights:
        raise ValueError("至少需要一個模型權重")
    if not images:
        raise ValueError("至少需要一張圖片")

    for weight in weights:
        if not weight.is_file():
            raise FileNotFoundError(f"找不到模型：{weight}")

    if args.exclude_regions:
        exclusion_rules = base.load_exclusion_rules(base.resolve_path(args.exclude_regions))
    else:
        exclusion_rules = []

    import torch

    from ultralytics import YOLO

    if args.device == "auto":
        args.device = "0" if torch.cuda.is_available() else "cpu"

    combinations = build_combinations(args)

    root = base.resolve_path(args.output) / datetime.now().strftime("sweep_%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True, exist_ok=False)

    sweep_settings = {
        "weights": [str(p) for p in weights],
        "source": [str(p) for p in images],
        "device": args.device,
        "views": args.views,
        "save_views": args.save_views,
        "max_det": args.max_det,
        "alpha": args.alpha,
        "exclude_regions": str(args.exclude_regions) if args.exclude_regions else None,
        "number_of_combinations": len(combinations),
        "combinations": combinations,
    }
    (root / "sweep_settings.json").write_text(
        json.dumps(sweep_settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"裝置：{args.device}")
    print(f"模型數：{len(weights)}")
    print(f"圖片數：{len(images)}")
    print(f"參數組合數：{len(combinations)}")
    print(f"每張圖片每組參數會推論 {len(args.views)} 個視圖")
    print(f"總模型推論次數約：{len(weights) * len(images) * len(combinations) * len(args.views)}")
    print(f"輸出位置：{root}")

    # 每張圖片一列
    detail_rows: list[dict] = []

    # 每組參數彙整
    combo_aggregate = defaultdict(
        lambda: {
            "images": 0,
            "total_detections": 0,
            "confidences": [],
            "class_counts": Counter(),
        }
    )

    # 先建立每組參數的資料夾與設定檔。
    combo_dirs = {}
    for combo_index, params in enumerate(combinations, 1):
        folder = root / combo_name(combo_index, params)
        folder.mkdir(parents=True, exist_ok=False)
        combo_dirs[combo_index] = folder

        (folder / "parameters.json").write_text(
            json.dumps(
                {"combo_id": combo_index, **params},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # 模型只載入一次，再依序跑所有參數組合。
    for model_index, weight in enumerate(weights, 1):
        print(f"\n載入模型 [{model_index}/{len(weights)}]：{weight.name}")
        model = YOLO(str(weight))

        if model.task != "segment":
            raise ValueError(f"需要 segmentation 模型，{weight} 的 task 為 {model.task!r}")

        for combo_index, params in enumerate(combinations, 1):
            folder = combo_dirs[combo_index]
            predict_args = make_predict_args(args, params, exclusion_rules)

            model_output = folder / f"{model_index:02d}_{base.safe_name(weight.stem)}"

            print(
                f"\n[參數 {combo_index}/{len(combinations)}] "
                f"imgsz={params['imgsz']}, "
                f"conf={params['conf']}, "
                f"iou={params['iou']}, "
                f"gamma={params['gamma']}, "
                f"clahe={params['clahe_clip']}, "
                f"shadow={params['shadow_threshold']}, "
                f"merge_iou={params['merge_iou']}"
            )

            for image_index, source in enumerate(images, 1):
                image_output = model_output / f"{image_index:06d}_{base.safe_name(source.stem)}"

                print(
                    f"  [{image_index}/{len(images)}] {source.name}",
                    flush=True,
                )

                report = base.predict_image(
                    model,
                    source,
                    image_output,
                    predict_args,
                )

                summary = summarize_report(report)

                row = {
                    "combo_id": combo_index,
                    "combo_folder": folder.name,
                    **params,
                    "model": weight.name,
                    "image": source.name,
                    "image_path": str(source),
                    "merged_count": summary["merged_count"],
                    "mean_confidence": round(summary["mean_confidence"], 6),
                    "max_confidence": round(summary["max_confidence"], 6),
                    "class_counts": json.dumps(
                        summary["class_counts"],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "merged_view_counts": json.dumps(
                        summary["merged_view_counts"],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "raw_view_counts": json.dumps(
                        summary["raw_view_counts"],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "filtered_view_counts": json.dumps(
                        summary["filtered_view_counts"],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "metrics": json.dumps(
                        summary["metrics"],
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
                    "result_folder": str(image_output),
                }
                detail_rows.append(row)

                agg = combo_aggregate[combo_index]
                agg["images"] += 1
                agg["total_detections"] += summary["merged_count"]
                agg["class_counts"].update(summary["class_counts"])

                for det in report.get("detections", []):
                    if "confidence" in det:
                        agg["confidences"].append(float(det["confidence"]))

        del model

    # -----------------------------
    # 每張圖片的詳細結果
    # -----------------------------
    detail_fields = [
        "combo_id",
        "combo_folder",
        "imgsz",
        "conf",
        "iou",
        "gamma",
        "clahe_clip",
        "shadow_threshold",
        "merge_iou",
        "model",
        "image",
        "image_path",
        "merged_count",
        "mean_confidence",
        "max_confidence",
        "class_counts",
        "merged_view_counts",
        "raw_view_counts",
        "filtered_view_counts",
        "metrics",
        "result_folder",
    ]

    write_csv(
        root / "parameter_results.csv",
        detail_rows,
        detail_fields,
    )

    # -----------------------------
    # 每組參數的總結果
    # -----------------------------
    summary_rows = []

    for combo_index, params in enumerate(combinations, 1):
        agg = combo_aggregate[combo_index]
        confidences = agg["confidences"]

        summary_rows.append(
            {
                "combo_id": combo_index,
                "combo_folder": combo_dirs[combo_index].name,
                **params,
                "images": agg["images"],
                "total_detections": agg["total_detections"],
                "detections_per_image": round(agg["total_detections"] / agg["images"], 6) if agg["images"] else 0.0,
                "mean_confidence": round(mean(confidences), 6) if confidences else 0.0,
                "max_confidence": round(max(confidences), 6) if confidences else 0.0,
                "class_counts": json.dumps(
                    dict(agg["class_counts"]),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )

    summary_fields = [
        "combo_id",
        "combo_folder",
        "imgsz",
        "conf",
        "iou",
        "gamma",
        "clahe_clip",
        "shadow_threshold",
        "merge_iou",
        "images",
        "total_detections",
        "detections_per_image",
        "mean_confidence",
        "max_confidence",
        "class_counts",
    ]

    write_csv(
        root / "parameter_summary.csv",
        summary_rows,
        summary_fields,
    )

    (root / "parameter_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_summary(summary_rows)

    print("\n完成。")
    print(f"各組影像結果：{root}")
    print(f"詳細結果 CSV：{root / 'parameter_results.csv'}")
    print(f"參數總表 CSV：{root / 'parameter_summary.csv'}")
    print(f"參數總表 JSON：{root / 'parameter_summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
