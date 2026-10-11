"""Visualize YOLO segmentation labels over images.

This tool overlays YOLO segment polygon labels on their matching images and
writes individual previews plus a contact sheet for quick dataset QA.

Example:
    python tools/visualize_yolo_seg_labels.py ^
        --dataset-root C:/Users/sile7/Downloads/dataset0608/dataset0608/dataset_pseudo_grabcut ^
        --split train ^
        --limit 24 ^
        --out-dir runs/pseudo_label_preview
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay YOLO segmentation labels on images.")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Dataset root containing images/ and labels/.")
    parser.add_argument("--split", default="train", help="Dataset split under images/ and labels/, e.g. train or val.")
    parser.add_argument("--images", type=Path, help="Optional explicit image directory.")
    parser.add_argument("--labels", type=Path, help="Optional explicit label directory.")
    parser.add_argument("--out-dir", default=Path("runs/pseudo_label_preview"), type=Path, help="Output directory.")
    parser.add_argument("--limit", type=int, default=40, help="Maximum number of images to visualize.")
    parser.add_argument("--start", type=int, default=0, help="Start index after sorting images.")
    parser.add_argument("--cols", type=int, default=4, help="Number of columns in the contact sheet.")
    parser.add_argument("--thumb-width", type=int, default=520, help="Thumbnail width in the contact sheet.")
    parser.add_argument("--alpha", type=float, default=0.38, help="Polygon fill opacity.")
    parser.add_argument("--line-width", type=int, default=2, help="Polygon/bbox line width.")
    parser.add_argument("--draw-fill", action="store_true", help="Fill polygons with transparent color.")
    parser.add_argument(
        "--only-with-labels", action="store_true", help="Skip images with no label file or empty labels."
    )
    parser.add_argument("--save-individual", action="store_true", help="Save every preview image in addition to sheet.")
    return parser.parse_args()


def iter_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def simple_yaml_names(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
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
        if in_names and not line.startswith(" "):
            break
    return names


def load_names(dataset_root: Path) -> dict[int, str]:
    data_yaml = dataset_root / "data.yaml"
    try:
        import yaml

        data: Any = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) if data_yaml.exists() else {}
        raw_names = data.get("names", {}) if isinstance(data, dict) else {}
        if isinstance(raw_names, list):
            return {i: str(name) for i, name in enumerate(raw_names)}
        if isinstance(raw_names, dict):
            return {int(k): str(v) for k, v in raw_names.items()}
    except Exception:
        pass
    return simple_yaml_names(data_yaml)


def label_path_for(image_path: Path, images_root: Path, labels_root: Path) -> Path:
    return labels_root / image_path.relative_to(images_root).with_suffix(".txt")


def color_for_class(cls_id: int) -> tuple[int, int, int]:
    palette = (
        (255, 56, 56),
        (255, 157, 151),
        (255, 112, 31),
        (255, 178, 29),
        (207, 210, 49),
        (72, 249, 10),
        (146, 204, 23),
        (61, 219, 134),
        (26, 147, 52),
        (0, 212, 187),
        (44, 153, 168),
        (0, 194, 255),
        (52, 69, 147),
        (100, 115, 255),
        (0, 24, 236),
        (132, 56, 255),
    )
    r, g, b = palette[cls_id % len(palette)]
    return (b, g, r)


def parse_label_line(line: str, shape: tuple[int, int]) -> tuple[int, np.ndarray, bool] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls_id = int(float(parts[0]))
        values = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
    except ValueError:
        return None

    h, w = shape
    if len(values) == 4:
        xc, yc, bw, bh = values
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h
        pts = np.asarray([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], dtype=np.float32)
        return cls_id, pts, True

    if len(values) >= 6 and len(values) % 2 == 0:
        pts = values.reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        return cls_id, pts, False
    return None


def draw_label(
    image: np.ndarray,
    cls_id: int,
    pts: np.ndarray,
    is_box: bool,
    names: dict[int, str],
    alpha: float,
    line_width: int,
    draw_fill: bool,
) -> None:
    color = color_for_class(cls_id)
    pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    if draw_fill and not is_box:
        overlay = image.copy()
        cv2.fillPoly(overlay, [pts_i], color)
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, dst=image)
    cv2.polylines(image, [pts_i], True, color, line_width, lineType=cv2.LINE_AA)

    x, y = pts_i.reshape(-1, 2).min(axis=0)
    label = names.get(cls_id, str(cls_id))
    if is_box:
        label += " box"
    font_scale = 0.5
    thickness = max(1, line_width - 1)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    y1 = max(0, int(y) - th - baseline - 4)
    x1 = max(0, int(x))
    cv2.rectangle(image, (x1, y1), (x1 + tw + 6, y1 + th + baseline + 4), color, -1)
    cv2.putText(
        image,
        label,
        (x1 + 3, y1 + th + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def visualize_one(
    image_path: Path,
    label_path: Path,
    names: dict[int, str],
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"warning: failed to read image: {image_path}")
        return None, 0
    h, w = image.shape[:2]
    labels = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
    count = 0
    for line in labels:
        parsed = parse_label_line(line, (h, w))
        if parsed is None:
            continue
        cls_id, pts, is_box = parsed
        draw_label(image, cls_id, pts, is_box, names, args.alpha, args.line_width, args.draw_fill)
        count += 1
    return image, count


def resize_thumb(image: np.ndarray, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    height = max(1, round(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_contact_sheet(previews: list[tuple[np.ndarray, str]], cols: int, thumb_width: int) -> np.ndarray:
    thumbs = [add_title(resize_thumb(image, thumb_width), title) for image, title in previews]
    if not thumbs:
        raise ValueError("No previews to make a contact sheet.")
    max_h = max(t.shape[0] for t in thumbs)
    padded = []
    for thumb in thumbs:
        if thumb.shape[0] < max_h:
            pad = np.full((max_h - thumb.shape[0], thumb.shape[1], 3), 32, dtype=np.uint8)
            thumb = np.vstack([thumb, pad])
        padded.append(thumb)

    rows = []
    for i in range(0, len(padded), cols):
        row = padded[i : i + cols]
        if len(row) < cols:
            blank = np.full_like(row[0], 32)
            row.extend([blank.copy() for _ in range(cols - len(row))])
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    images_root = (args.images or (dataset_root / "images" / args.split)).resolve()
    labels_root = (args.labels or (dataset_root / "labels" / args.split)).resolve()
    out_dir = args.out_dir.resolve()

    image_paths = iter_images(images_root)
    if args.start:
        image_paths = image_paths[args.start :]
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        print(f"No images found under {images_root}")
        return 1

    names = load_names(dataset_root)
    previews: list[tuple[np.ndarray, str]] = []
    saved = 0
    for image_path in image_paths:
        label_path = label_path_for(image_path, images_root, labels_root)
        if args.only_with_labels and (not label_path.exists() or not label_path.read_text(encoding="utf-8").strip()):
            continue
        preview, count = visualize_one(image_path, label_path, names, args)
        if preview is None:
            continue
        title = f"{image_path.relative_to(images_root)} ({count})"
        previews.append((preview, title))
        if args.save_individual:
            rel = image_path.relative_to(images_root)
            out_path = out_dir / "images" / rel.with_suffix(".jpg")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), preview)
            saved += 1

    if not previews:
        print("No previews generated.")
        return 1

    sheet = make_contact_sheet(previews, max(1, args.cols), max(120, args.thumb_width))
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = out_dir / f"{args.split}_label_preview_sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet)

    rows = math.ceil(len(previews) / max(1, args.cols))
    print(f"Images root: {images_root}")
    print(f"Labels root: {labels_root}")
    print(f"Previewed {len(previews)} images in {rows} rows.")
    print(f"Contact sheet: {sheet_path}")
    if args.save_individual:
        print(f"Individual previews: {saved} images under {out_dir / 'images'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
