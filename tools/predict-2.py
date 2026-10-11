"""Run ``predict_seg_by_class.py`` with multiple weights and image sources.

Edit ``WEIGHT_SOURCES`` and ``DATASET_SOURCES`` below, then run:

    python tools/predict-2.py

Every model is applied to every source, one job at a time.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ======================== Edit these defaults as needed ========================
# Model paths
WEIGHT_SOURCES = [
    # r"C:\Users\USER\Downloads\best0907-10csar.pt",
    # r"C:\Users\USER\Downloads\best0907-p2.pt"
    # r"C:\Users\USER\Downloads\best0902-2.pt",
    r"C:\Users\USER\Downloads\best0929-3.pt",  # Active baseline restored on 2026-09-18.
]

# Image or image-directory paths
DATASET_SOURCES = [
    r"C:\Users\USER\Desktop\碩論\貨櫃-20221012T025654Z-001\貨櫃\Container_damage Dataset\train\B\CBHU0708054-A.JPG",
    # r"C:\Users\USER\Desktop\碩論\貨櫃-20221012T025654Z-001\貨櫃\Container_damage Dataset\train\B\CBHU0708054-B.JPG",
    # r"C:\Users\USER\Desktop\碩論\貨櫃-20221012T025654Z-001\貨櫃\Container_damage Dataset\train\B\IMG_4155.JPG",
    # r"C:\Users\USER\Desktop\碩論\貨櫃-20221012T025654Z-001\貨櫃\Container_damage Dataset\val 驗證\IMAG0749.jpg",
    # r"C:\Users\USER\Desktop\碩論\貨櫃-20221012T025654Z-001\貨櫃\Container_damage Dataset\val 驗證\P5021976.JPG",
    # r"C:\Users\USER\Desktop\碩論\貨櫃-20221012T025654Z-001\貨櫃\Container_damage Dataset\val 驗證\P6232556.JPG",
    # r"C:\Users\USER\Desktop\碩論\貨櫃-20221012T025654Z-001\貨櫃\Container_damage Dataset\val 驗證\P8293423.JPG",
]

# Inference settings
OUTPUT_DIR = "./runs/segment_by_class"
IMAGE_SIZE = 640
CONFIDENCE = 0.15
IOU_THRESHOLD = 0.7
DEVICE = "0"  # Examples: "0", "cpu", "0,1"
RETINA_MASKS = True
# ==============================================================================


SCRIPT_DIR = Path(__file__).resolve().parent
PREDICT_SCRIPT = SCRIPT_DIR / "predict_seg_by_class.py"


def resolve_path(value: str | Path) -> Path:
    """Resolve relative paths from the directory containing this script."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def parse_args() -> argparse.Namespace:
    """Parse command-line overrides for the editable defaults above."""
    parser = argparse.ArgumentParser(description="Run class-colored segmentation for multiple models and sources.")
    parser.add_argument(
        "--weights",
        nargs="+",
        default=WEIGHT_SOURCES,
        help="One or more .pt model files.",
    )
    parser.add_argument(
        "--source",
        nargs="+",
        default=DATASET_SOURCES,
        help="One or more images or image directories.",
    )
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--conf", type=float, default=CONFIDENCE)
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument(
        "--retina-masks",
        action=argparse.BooleanOptionalAction,
        default=RETINA_MASKS,
    )
    return parser.parse_args()


def model_run_name(weight: Path) -> str:
    """Build a unique output name such as ``train-MultiStateCSAR_best``."""
    if weight.parent.name == "weights":
        return f"{weight.parent.parent.name}_{weight.stem}"
    return weight.stem


def build_command(
    args: argparse.Namespace,
    weight: Path,
    source: Path,
    output: Path,
) -> list[str]:
    """Build one child-process command for a weight/source pair."""
    command = [
        sys.executable,
        str(PREDICT_SCRIPT),
        "--model",
        str(weight),
        "--source",
        str(source),
        "--output",
        str(output),
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(args.conf),
        "--device",
        str(args.device),
        "--iou",
        str(args.iou),
    ]
    if args.retina_masks:
        command.append("--retina-masks")
    return command


def main() -> int:
    """Run every weight/source combination sequentially."""
    args = parse_args()

    if not PREDICT_SCRIPT.is_file():
        print(f"找不到推論程式：{PREDICT_SCRIPT}", file=sys.stderr)
        return 1

    weights = [resolve_path(value) for value in args.weights]
    sources = [resolve_path(value) for value in args.source]
    output_root = resolve_path(args.output)

    missing_weights = [path for path in weights if not path.is_file()]
    missing_sources = [path for path in sources if not path.exists()]
    if missing_weights:
        for path in missing_weights:
            print(f"找不到模型：{path}", file=sys.stderr)
    if missing_sources:
        for path in missing_sources:
            print(f"找不到圖片或資料夾：{path}", file=sys.stderr)
    if missing_weights or missing_sources:
        return 1

    total = len(weights) * len(sources)
    if total == 0:
        print("WEIGHT_SOURCES 或 DATASET_SOURCES 沒有設定內容。")
        return 0

    failures: list[str] = []
    job_number = 0

    for weight in weights:
        model_output = output_root / model_run_name(weight)

        for source in sources:
            job_number += 1
            print(f"\n{'=' * 70}")
            print(f"[{job_number}/{total}] 模型：{weight}")
            print(f"[{job_number}/{total}] 來源：{source}")
            print(f"[{job_number}/{total}] 輸出：{model_output}")
            print(f"{'=' * 70}", flush=True)

            try:
                result = subprocess.run(
                    build_command(args, weight, source, model_output),
                    cwd=SCRIPT_DIR,
                    check=False,
                )
            except OSError as error:
                failures.append(f"{weight.name} × {source.name}: {error}")
                continue

            if result.returncode != 0:
                failures.append(f"{weight.name} × {source.name}: exit {result.returncode}")

    print(f"\n全部 {total} 組工作已執行完畢。")
    if failures:
        print("以下工作執行失敗：", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
