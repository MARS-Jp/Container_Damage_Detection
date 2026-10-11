# 透過網路模型將4點矩型的segment格式變成不規則形狀
"""Build a YOLO segmentation dataset with pseudo masks from rectangular labels.

This is a wrapper around generate_pseudo_seg_from_bbox.py. It treats existing
rectangular YOLO segment labels as boxes, generates pseudo polygon labels, and
writes a complete dataset folder with a new data.yaml.

Example:
    python tools/build_pseudo_seg_dataset.py ^
        --source-data C:/Users/sile7/Downloads/dataset0608/dataset0608/data.yaml ^
        --out-root C:/Users/sile7/Downloads/dataset0608/dataset0608/dataset_pseudo_grabcut ^
        --debug-dir C:/Users/sile7/Downloads/dataset0608/dataset0608/runs/pseudo_masks ^
        --seed-mode hybrid
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from generate_pseudo_seg_from_bbox import iter_images, label_path_for, process_image, save_debug_panel

DEFAULT_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a pseudo YOLO segmentation dataset from rectangular labels.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--source-data", type=Path, help="Original Ultralytics data.yaml.")
    src.add_argument("--source-root", type=Path, help="Original dataset root with images/ and labels/.")
    parser.add_argument("--out-root", required=True, type=Path, help="Output pseudo dataset root.")
    parser.add_argument(
        "--splits", nargs="*", default=None, help="Splits to process. Defaults to train/val/test found."
    )
    parser.add_argument(
        "--image-mode",
        choices=("hardlink", "copy", "none"),
        default="hardlink",
        help="How to place images in the pseudo dataset. Hardlink falls back to copy if needed.",
    )
    parser.add_argument("--debug-dir", type=Path, help="Optional directory for debug panels.")
    parser.add_argument("--classes", nargs="*", help="Optional class ids to pseudo-mask. Others stay rectangular.")
    parser.add_argument("--seed-mode", choices=("center", "rust", "hybrid"), default="hybrid")
    parser.add_argument("--fg-thres", type=float, default=0.62)
    parser.add_argument("--bg-thres", type=float, default=0.24)
    parser.add_argument("--pad", type=float, default=0.15)
    parser.add_argument("--grabcut-iters", type=int, default=3)
    parser.add_argument("--min-area", type=float, default=0.00002)
    parser.add_argument("--max-contours", type=int, default=3)
    parser.add_argument("--approx-frac", type=float, default=0.003)
    parser.add_argument("--empty-policy", choices=("bbox", "skip"), default="bbox")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files.")
    return parser.parse_args()


def as_posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return load_simple_data_yaml(path)


def load_simple_data_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    names: dict[int, str] = {}
    in_names = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names and line.startswith(" "):
            key, _, value = line.strip().partition(":")
            if key.isdigit():
                names[int(key)] = value.strip().strip("'\"")
            continue
        in_names = False
        key, _, value = line.partition(":")
        if key and value:
            data[key.strip()] = value.strip().strip("'\"")
    if names:
        data["names"] = names
    if "nc" in data:
        try:
            data["nc"] = int(data["nc"])
        except ValueError:
            pass
    return data


def dataset_root_from_yaml(data_yaml: Path, data: dict[str, Any]) -> Path:
    root_value = data.get("path", ".")
    root = Path(str(root_value))
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def normalize_entry(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return [str(value)]


def labels_rel_for_images_rel(images_rel: Path) -> Path:
    parts = list(images_rel.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts)
    return Path("labels") / images_rel.name


def split_entries_from_yaml(
    data: dict[str, Any], source_root: Path, requested_splits: list[str] | None
) -> tuple[list[tuple[str, Path, Path, Path, Path]], dict[str, Any]]:
    entries: list[tuple[str, Path, Path, Path, Path]] = []
    split_names = requested_splits or [s for s in DEFAULT_SPLITS if s in data]
    for split in split_names:
        for item in normalize_entry(data.get(split)):
            images_rel = Path(item)
            images_dir = images_rel if images_rel.is_absolute() else source_root / images_rel
            if not images_dir.is_dir():
                print(f"warning: skip {split}, image directory not found: {images_dir}")
                continue
            labels_rel = labels_rel_for_images_rel(images_rel)
            labels_dir = labels_rel if labels_rel.is_absolute() else source_root / labels_rel
            entries.append((split, images_dir, labels_dir, images_rel, labels_rel))
    return entries, data


def split_entries_from_root(
    source_root: Path, requested_splits: list[str] | None
) -> tuple[list[tuple[str, Path, Path, Path, Path]], dict[str, Any]]:
    entries: list[tuple[str, Path, Path, Path, Path]] = []
    split_names = requested_splits or list(DEFAULT_SPLITS)
    for split in split_names:
        images_rel = Path("images") / split
        labels_rel = Path("labels") / split
        images_dir = source_root / images_rel
        labels_dir = source_root / labels_rel
        if images_dir.is_dir():
            entries.append((split, images_dir, labels_dir, images_rel, labels_rel))
    data = {"train": "images/train", "val": "images/val"}
    return entries, data


def infer_names(entries: list[tuple[str, Path, Path, Path, Path]]) -> dict[int, str]:
    max_cls = -1
    for _, _, labels_dir, _, _ in entries:
        if not labels_dir.exists():
            continue
        for label_path in labels_dir.rglob("*.txt"):
            for raw in label_path.read_text(encoding="utf-8").splitlines():
                parts = raw.strip().split()
                if parts and parts[0].isdigit():
                    max_cls = max(max_cls, int(parts[0]))
    return {i: f"class_{i}" for i in range(max_cls + 1)}


def write_data_yaml(out_root: Path, data: dict[str, Any], entries: list[tuple[str, Path, Path, Path, Path]]) -> None:
    names = data.get("names")
    if isinstance(names, list):
        names = {i: str(name) for i, name in enumerate(names)}
    if not isinstance(names, dict):
        names = infer_names(entries)
    names = {int(k): str(v) for k, v in names.items()}
    nc = int(data.get("nc", len(names)))

    split_lines: dict[str, str] = {}
    for split, _, _, images_rel, _ in entries:
        split_lines.setdefault(split, as_posix(images_rel))
    if "val" not in split_lines and "train" in split_lines:
        split_lines["val"] = split_lines["train"]

    lines = [f"path: {as_posix(out_root.resolve())}"]
    for split in DEFAULT_SPLITS:
        if split in split_lines:
            lines.append(f"{split}: {split_lines[split]}")
    lines.extend([f"nc: {nc}", "", "names:"])
    for i in range(nc):
        lines.append(f"  {i}: {names.get(i, f'class_{i}')}")
    (out_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def place_image(src: Path, dst: Path, mode: str, dry_run: bool) -> None:
    if mode == "none" or dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def pseudo_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        classes=args.classes,
        seed_mode=args.seed_mode,
        fg_thres=args.fg_thres,
        bg_thres=args.bg_thres,
        pad=args.pad,
        grabcut_iters=args.grabcut_iters,
        min_area=args.min_area,
        max_contours=args.max_contours,
        approx_frac=args.approx_frac,
        empty_policy=args.empty_policy,
    )


def process_entry(
    args: argparse.Namespace,
    entry: tuple[str, Path, Path, Path, Path],
    seen: set[tuple[Path, Path, Path]],
) -> tuple[int, int]:
    split, images_dir, labels_dir, images_rel, labels_rel = entry
    out_images_dir = args.out_root / images_rel
    out_labels_dir = args.out_root / labels_rel
    key = (images_dir.resolve(), out_images_dir.resolve(), out_labels_dir.resolve())
    if key in seen:
        return 0, 0
    seen.add(key)

    image_paths = iter_images(images_dir)
    converted = 0
    skipped = 0
    p_args = pseudo_args(args)
    for image_path in image_paths:
        rel = image_path.relative_to(images_dir)
        label_path = label_path_for(image_path, images_dir, labels_dir)
        result = process_image(image_path, label_path, p_args)
        if result is None:
            skipped += 1
            continue

        out_image = out_images_dir / rel
        out_label = out_labels_dir / rel.with_suffix(".txt")
        place_image(image_path, out_image, args.image_mode, args.dry_run)
        if not args.dry_run:
            out_label.parent.mkdir(parents=True, exist_ok=True)
            out_label.write_text("\n".join(result.lines) + ("\n" if result.lines else ""), encoding="utf-8")

        if args.debug_dir:
            debug_path = args.debug_dir / split / rel.with_suffix(".jpg")
            if not args.dry_run:
                save_debug_panel(image_path, result, debug_path)
        converted += 1
    return converted, skipped


def main() -> int:
    args = parse_args()
    args.out_root = args.out_root.resolve()
    if args.source_data:
        data = load_yaml(args.source_data)
        source_root = dataset_root_from_yaml(args.source_data, data)
        entries, data = split_entries_from_yaml(data, source_root, args.splits)
    else:
        source_root = args.source_root.resolve()
        entries, data = split_entries_from_root(source_root, args.splits)

    if not entries:
        print("No valid image splits found.")
        return 1

    total_converted = 0
    total_skipped = 0
    seen: set[tuple[Path, Path, Path]] = set()
    for entry in entries:
        converted, skipped = process_entry(args, entry, seen)
        total_converted += converted
        total_skipped += skipped

    if not args.dry_run:
        args.out_root.mkdir(parents=True, exist_ok=True)
        write_data_yaml(args.out_root, data, entries)

    print(f"Pseudo dataset: {args.out_root}")
    print(f"Converted {total_converted} images. Skipped {total_skipped}.")
    print(f"Data YAML: {args.out_root / 'data.yaml'}")
    if args.debug_dir:
        print(f"Debug panels: {args.debug_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
