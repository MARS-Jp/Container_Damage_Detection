"""Run parameter comparisons using the current settings in ``predict-2.py``.

    python tools/predict_parameter_sweep.py
    python tools/predict_parameter_sweep.py --dry-run
    python tools/predict_parameter_sweep.py --versions baseline conf005 imgsz960
    python tools/predict_parameter_sweep.py --suite focused --compare
    python tools/predict_parameter_sweep.py --suite all --compare

The initial suite changes one parameter per version; the focused suite also tests
class-specific thresholds and selected combinations. Models, active sources, device,
and retina-mask settings are inherited from predict-2.py on every invocation.
The shadow/general/combined suites use the same fill-only predict_shadow.py renderer.
Select all to run all three groups. These suites require original-resolution retina masks.
Results go to a new timestamped directory, with separate version/model/source
folders, parameters.json, per-job inference.log files, and a summary.csv.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
BASE = runpy.run_path(str(SCRIPT_DIR / "predict-2.py"))

# Edit or add versions here. Omitted parameters retain the baseline value.
PARAMETER_TESTS = {
    "baseline": {},
    "conf005": {"conf": 0.05},
    "conf010": {"conf": 0.10},
    "conf025": {"conf": 0.25},
    "iou030": {"iou": 0.30},
    "iou050": {"iou": 0.50},
    "imgsz960": {"imgsz": 960},
    "imgsz1280": {"imgsz": 1280},
}
FOCUSED_TESTS = {
    "baseline": {},
    "size576": {"imgsz": 576},
    "size704": {"imgsz": 704},
    "size768": {"imgsz": 768},
    "nms085": {"iou": 0.85},
    "d008_r020": {"class_conf": {"D": 0.08, "R": 0.20}},
    "d010_r025": {"class_conf": {"D": 0.10, "R": 0.25}},
    "d010_r020_nms050": {"class_conf": {"D": 0.10, "R": 0.20}, "iou": 0.50},
    "size704_d010_r020_nms050": {"imgsz": 704, "class_conf": {"D": 0.10, "R": 0.20}, "iou": 0.50},
    "conservative025_nms030": {"conf": 0.25, "iou": 0.30},
}
# 陰影版本：除 baseline 外，未列出的項目皆沿用 shadow_default。
# imgsz/conf/iou/device 與模型、圖片取自 predict-2.py，增強參數在此集中設定。
SHADOW_DEFAULTS = {
    "views": ["original", "gamma", "clahe"],
    "gamma": 0.65,
    "clahe_clip": 2.0,
    "shadow_threshold": 255,
    "merge_iou": 0.60,
    "max_det": 300,
    "alpha": 0.35,
}
SHADOW_TESTS = {
    "baseline": {"views": ["original"]},
    "shadow_default": {},
    "gamma_only": {"views": ["original", "gamma"]},
    "clahe_only": {"views": ["original", "clahe"]},
    "gamma045": {"gamma": 0.45},
    "gamma085": {"gamma": 0.85},
    "clahe40": {"clahe_clip": 4.0},
    "shadow128": {"shadow_threshold": 128},
    "merge040": {"merge_iou": 0.40},
}
GENERAL_TESTS = {name: {**settings, "views": ["original"]} for name, settings in PARAMETER_TESTS.items()}
COMBINED_TESTS = {
    "baseline": {"views": ["original"]},
    "shadow_default": {},
    "conf010": {"conf": 0.10},
    "iou050": {"iou": 0.50},
    "imgsz960": {"imgsz": 960},
    "imgsz1280": {"imgsz": 1280},
    "size960_conf010_iou050": {"imgsz": 960, "conf": 0.10, "iou": 0.50},
    "size960_gamma045_shadow128": {"imgsz": 960, "gamma": 0.45, "shadow_threshold": 128},
}
GROUP_TESTS = {"shadow": SHADOW_TESTS, "general": GENERAL_TESTS, "combined": COMBINED_TESTS}
GROUP_DIRS = {"shadow": "01_shadow", "general": "02_general", "combined": "03_combined"}
PARAMETER_KEYS = ("imgsz", "conf", "iou", "device", "retina_masks")
SUMMARY_FIELDS = (
    "group", "version", "weight", "source", *PARAMETER_KEYS, "class_conf", *SHADOW_DEFAULTS,
    "exclude_regions", "status", "returncode", "seconds", "output", "log",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Use the existing driver defaults, with optional command-line overrides."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", nargs="+", default=BASE["WEIGHT_SOURCES"])
    parser.add_argument("--source", nargs="+", default=BASE["DATASET_SOURCES"])
    parser.add_argument(
        "--output", default=None,
        help="Parent output directory; relative paths resolve from tools/, as in predict-2.py.",
    )
    parser.add_argument("--imgsz", type=int, default=BASE["IMAGE_SIZE"], help="Baseline image size.")
    parser.add_argument("--conf", type=float, default=BASE["CONFIDENCE"], help="Baseline confidence threshold.")
    parser.add_argument("--iou", type=float, default=BASE["IOU_THRESHOLD"], help="Baseline NMS IoU threshold.")
    parser.add_argument("--device", default=BASE["DEVICE"])
    parser.add_argument("--retina-masks", action=argparse.BooleanOptionalAction, default=BASE["RETINA_MASKS"])
    parser.add_argument("--suite", choices=("initial", "focused", *GROUP_TESTS, "all"), default="initial")
    parser.add_argument("--versions", nargs="+", help="Version names within the selected suite(s).")
    parser.add_argument("--save-metrics", action="store_true", help="Save class counts and mask coverage.")
    parser.add_argument("--compare", action="store_true", help="Automatically create full-image and D/R comparison figures.")
    parser.add_argument("--reference-run", type=Path, help="Optional earlier sweep to include as a labeled reference.")
    parser.add_argument("--reference-versions", nargs="+", default=["imgsz960"])
    exclusions = parser.add_mutually_exclusive_group()
    exclusions.add_argument("--exclude-regions", type=Path, default=SCRIPT_DIR / "shadow_exclusion_regions.json",
                            help="Background exclusion JSON for shadow/general/combined suites; snapshotted once per run.")
    exclusions.add_argument("--no-exclusions", dest="exclude_regions", action="store_const", const=None)
    parser.add_argument("--dry-run", action="store_true", help="Print all commands without running inference or writing files.")
    args = parser.parse_args(argv)
    args.groups = list(GROUP_TESTS) if args.suite == "all" else [args.suite] if args.suite in GROUP_TESTS else []
    available = ({name: preset for group in args.groups for name, preset in GROUP_TESTS[group].items()}
                 if args.groups else dict(PARAMETER_TESTS, **FOCUSED_TESTS))
    if args.versions is None:
        args.versions = list(available if args.groups else FOCUSED_TESTS if args.suite == "focused" else PARAMETER_TESTS)
    if any(version not in available for version in args.versions):
        parser.error(f"Selected versions do not belong to suite {args.suite!r}; available: {', '.join(available)}")
    if args.output is None:
        folder = "parameter_tests_round2" if args.suite == "focused" else "parameter_tests"
        args.output = str(
            SCRIPT_DIR.parent / "runs/segment_shadow/parameter_tests"
            if args.groups else Path(BASE["OUTPUT_DIR"]) / folder
        )
    if args.imgsz <= 0 or not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error("imgsz must be positive; conf and iou must be between 0 and 1.")
    if len(set(args.versions)) != len(args.versions):
        parser.error("Each version may be selected only once.")
    if args.groups:
        if args.imgsz % 32:
            parser.error("The comparison suites require imgsz to be a multiple of 32.")
        if not args.retina_masks:
            parser.error("The comparison suites require retina masks; enable RETINA_MASKS in predict-2.py or pass --retina-masks.")
        if args.reference_run:
            parser.error("--reference-run is supported only for the initial/focused suites.")
    return args


def safe_component(value: str) -> str:
    """Keep readable Unicode names while making directory components safe."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .") or "unnamed"


def build_jobs(args: argparse.Namespace, run_dir: Path) -> list[dict]:
    """Plan isolated output paths, including for weights or sources with identical names."""
    jobs = []
    if args.groups:
        versions = [
            (group, index, version, GROUP_TESTS[group][version])
            for group in args.groups
            for index, version in enumerate((v for v in args.versions if v in GROUP_TESTS[group]), 1)
        ]
    else:
        versions = [(args.suite, index, version, dict(PARAMETER_TESTS, **FOCUSED_TESTS)[version])
                    for index, version in enumerate(args.versions, 1)]
    for group, version_index, version, overrides in versions:
        settings = {key: getattr(args, key) for key in PARAMETER_KEYS}
        settings["class_conf"] = {}
        if args.groups:
            settings.update(SHADOW_DEFAULTS)
        settings.update(overrides)
        version_dir = (
            f"{version_index:02d}_{version}_img{settings['imgsz']}"
            f"_conf{settings['conf']:g}_iou{settings['iou']:g}"
        )
        if args.suite == "focused" or args.groups:
            version_dir = f"{version_index:02d}_{version}"
        group_dir = run_dir / GROUP_DIRS[group] if args.groups else run_dir
        for model_index, weight_value in enumerate(args.weights, 1):
            weight = BASE["resolve_path"](weight_value)
            model_name = safe_component(BASE["model_run_name"](weight))
            for source_index, source_value in enumerate(args.source, 1):
                source = BASE["resolve_path"](source_value)
                output = group_dir / version_dir / f"{model_index:02d}_{model_name}" / (
                    f"{source_index:02d}_{safe_component(source.stem)}"
                )
                if args.groups:
                    command = [sys.executable, str(SCRIPT_DIR / "predict_shadow.py"),
                               "--weights", str(weight), "--source", str(source), "--output", str(output)]
                    for key in ("imgsz", "conf", "iou", "device", *SHADOW_DEFAULTS):
                        if key != "views":
                            command.extend([f"--{key.replace('_', '-')}", str(settings[key])])
                    command.extend(["--views", *settings["views"], "--save-views"])
                    if args.exclude_regions:
                        command.extend(["--exclude-regions", str(BASE["resolve_path"](args.exclude_regions))])
                    else:
                        command.append("--no-exclusions")
                else:
                    command = BASE["build_command"](argparse.Namespace(**settings), weight, source, output)
                    command.append("--hide-labels")
                    if settings["class_conf"]:
                        command.extend(["--class-conf", *[f"{name}={value:g}" for name, value in settings["class_conf"].items()]])
                    if args.save_metrics or args.compare:
                        command.append("--save-metrics")
                jobs.append({
                    "group": group,
                    "version": version,
                    "weight": str(weight),
                    "source": str(source),
                    **settings,
                    "exclude_regions": str(BASE["resolve_path"](args.exclude_regions)) if args.groups and args.exclude_regions else None,
                    "output": str(output),
                    "log": str(output / "inference.log"),
                    "command": command,
                })
    return jobs


def run_job(job: dict) -> tuple[int, float]:
    """Stream child output to the console and UTF-8 log, preserving errors for review."""
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "parameters.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    started = time.perf_counter()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    with Path(job["log"]).open("w", encoding="utf-8") as log:
        try:
            with subprocess.Popen(
                job["command"], cwd=SCRIPT_DIR, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            ) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
        except OSError as error:
            message = f"Failed to start inference: {error}\n"
            print(message, end="", file=sys.stderr)
            log.write(message)
            returncode = -1
    return returncode, round(time.perf_counter() - started, 3)


def main(argv: list[str] | None = None) -> int:
    """Run every selected version/model/source combination, continuing after job failures."""
    args = parse_args(argv)
    if not args.weights or not args.source:
        print("No models or sources configured in predict-2.py.", file=sys.stderr)
        return 1
    predict_script = SCRIPT_DIR / "predict_shadow.py" if args.groups else BASE["PREDICT_SCRIPT"]
    if not predict_script.is_file():
        print(f"Missing inference script: {predict_script}", file=sys.stderr)
        return 1
    if not args.dry_run:
        missing = [
            BASE["resolve_path"](value) for value in args.weights
            if not BASE["resolve_path"](value).is_file()
        ] + [
            BASE["resolve_path"](value) for value in args.source
            if not BASE["resolve_path"](value).exists()
        ]
        if missing:
            for path in missing:
                print(f"Missing input: {path}", file=sys.stderr)
            return 1
        if args.groups and args.exclude_regions and not BASE["resolve_path"](args.exclude_regions).is_file():
            print(f"Missing exclusion configuration: {args.exclude_regions}", file=sys.stderr)
            return 1

    run_dir = BASE["resolve_path"](args.output) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    jobs = build_jobs(args, run_dir)
    version_count = len({(job["group"], job["version"]) for job in jobs})
    print(f"{version_count} versions x {len(args.weights)} models x {len(args.source)} sources = {len(jobs)} jobs")
    print(f"Output: {run_dir}", flush=True)
    if args.dry_run:
        for index, job in enumerate(jobs, 1):
            print(f"[{index}/{len(jobs)}] {job['group']}/{job['version']}: {subprocess.list2cmdline(job['command'])}")
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    if args.groups and args.exclude_regions:
        snapshot = run_dir / "exclusion_regions.json"
        snapshot.write_bytes(BASE["resolve_path"](args.exclude_regions).read_bytes())
        args.exclude_regions = snapshot
        jobs = build_jobs(args, run_dir)
    (run_dir / "parameters.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = 0
    with (run_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as summary:
        writer = csv.DictWriter(summary, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        summary.flush()
        for index, job in enumerate(jobs, 1):
            print(
                f"\n[{index}/{len(jobs)}] {job['group']}/{job['version']} | imgsz={job['imgsz']}"
                f" conf={job['conf']} class_conf={job['class_conf']} iou={job['iou']} | {Path(job['weight']).name}"
                f" | {Path(job['source']).name}\nOutput: {job['output']}", flush=True,
            )
            returncode, seconds = run_job(job)
            failed += returncode != 0
            row = {key: job[key] for key in SUMMARY_FIELDS if key in job}
            row["class_conf"] = json.dumps(job["class_conf"], ensure_ascii=False)
            if "views" in job:
                row["views"] = json.dumps(job["views"])
            row.update(status="ok" if returncode == 0 else "failed", returncode=returncode, seconds=seconds)
            writer.writerow(row)
            summary.flush()
            print(f"[{index}/{len(jobs)}] {row['status']} ({seconds:.1f}s)", flush=True)

    print(f"\nFinished: {len(jobs) - failed} succeeded, {failed} failed.\nResults: {run_dir}")
    if args.compare:
        if str(SCRIPT_DIR.parent) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR.parent))
        if args.groups:
            from tools.compare_shadow_sweep import create_comparisons

            create_comparisons(run_dir)
        else:
            from tools.compare_parameter_sweep import create_comparisons

            create_comparisons(run_dir, args.reference_run, args.reference_versions)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
