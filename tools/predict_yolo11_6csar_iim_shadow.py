"""Run shadow-Dent-focused segmentation inference with one or more weights and image sources.

Default inference emphasizes recall for very small dents in dark corrugated regions:
    imgsz=1280, conf=0.10, iou=0.50, retina_masks=True

Examples:
    python tools/predict_yolo11_6csar_iim_shadow.py
    python tools/predict_yolo11_6csar_iim_shadow.py --diagnostic
    python tools/predict_yolo11_6csar_iim_shadow.py --weights /path/to/best.pt --source /path/to/images

``--diagnostic`` lowers confidence to 0.05. It is intended to reveal whether the target Dent exists as a low-score
candidate; it is not a recommended production threshold.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICT_SCRIPT = REPO_ROOT / "tools/predict_seg_by_class.py"

# Edit these lists if you prefer running the script without command-line paths.
# 模型路徑
DEFAULT_WEIGHTS = [
    # Path(r"C:\Users\USER\Downloads\best (1).pt"),
    Path(r"C:\Users\USER\Downloads\last0912.pt"),
    # Path(r"C:\Users\USER\Desktop\碩論\權重檔\jammy_storage_project_0824\yolo11-6csar-iim-2\weights\best.pt"),
]
# 圖片路徑
DEFAULT_SOURCES = [
    Path(r"C:\Users\USER\Desktop\碩論\dataset0608\dataset0608\dataset\images\train\B\CBHU0708054-A.JPG"),
]


def parse_args() -> argparse.Namespace:
    """Parse paths and inference settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", type=Path, default=DEFAULT_WEIGHTS, help="One or more .pt weights.")
    parser.add_argument("--source", nargs="+", type=Path, default=DEFAULT_SOURCES, help="Images or image directories.")
    parser.add_argument("--output", type=Path, default=Path("runs/segment_by_class_shadow"), help="Output root.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size; 1280 is recommended here.")
    parser.add_argument("--conf", type=float, default=0.10, help="Global confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold.")
    parser.add_argument("--device", default="0", help="Inference device, for example 0 or cpu.")
    parser.add_argument(
        "--retina-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render masks at original-image resolution (enabled by default).",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Use conf=0.05 to expose low-score Dent candidates in the shadow region.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running inference.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve a user path from the current working directory."""
    return path.expanduser().resolve()


def model_run_name(weight: Path, diagnostic: bool) -> str:
    """Create a stable output directory name for one checkpoint."""
    run_name = f"{weight.parent.parent.name}_{weight.stem}" if weight.parent.name == "weights" else weight.stem
    return f"{run_name}_diagnostic-conf005" if diagnostic else f"{run_name}_shadow-conf010"


def build_command(
    args: argparse.Namespace,
    weight: Path,
    source: Path,
    output: Path,
    confidence: float,
) -> list[str]:
    """Build one predict_seg_by_class.py command."""
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
        str(confidence),
        "--iou",
        str(args.iou),
        "--device",
        str(args.device),
    ]
    if args.retina_masks:
        command.append("--retina-masks")
    return command


def main() -> int:
    """Validate inputs and run every weight/source pair sequentially."""
    args = parse_args()
    if args.imgsz < 640 or args.imgsz % 32:
        raise ValueError("--imgsz must be at least 640 and divisible by 32.")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        raise ValueError("--conf and --iou must be between 0 and 1.")
    if not PREDICT_SCRIPT.is_file():
        raise FileNotFoundError(f"Prediction renderer not found: {PREDICT_SCRIPT}")

    weights = [resolve_path(path) for path in args.weights]
    sources = [resolve_path(path) for path in args.source]
    output_root = resolve_path(args.output)
    missing_weights = [path for path in weights if not path.is_file()]
    missing_sources = [path for path in sources if not path.exists()]
    if missing_weights or missing_sources:
        details = [*(f"Weight not found: {path}" for path in missing_weights)]
        details.extend(f"Source not found: {path}" for path in missing_sources)
        raise FileNotFoundError("\n".join(details))

    confidence = 0.05 if args.diagnostic else args.conf
    failures = []
    total = len(weights) * len(sources)
    job = 0
    for weight in weights:
        model_output = output_root / model_run_name(weight, args.diagnostic)
        for source in sources:
            job += 1
            command = build_command(args, weight, source, model_output, confidence)
            print(f"\n[{job}/{total}] {' '.join(command)}", flush=True)
            if args.dry_run:
                continue
            result = subprocess.run(command, cwd=REPO_ROOT, check=False)
            if result.returncode:
                failures.append(f"{weight.name} × {source.name}: exit {result.returncode}")

    if failures:
        print("Inference failures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"\nCompleted {total} inference job(s). Output: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
