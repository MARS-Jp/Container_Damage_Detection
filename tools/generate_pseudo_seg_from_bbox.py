# 將bbox格式轉換成Segment，仍然是4點矩形
"""Generate YOLO segmentation pseudo-labels from YOLO bbox labels.

This script is intended for weakly supervised experiments where only bounding
boxes are available. It builds a rust-oriented heatmap inside each box, converts
the heatmap to foreground/background seeds, optionally refines them with
GrabCut, then writes YOLO segmentation polygon labels.

Example:
    python tools/generate_pseudo_seg_from_bbox.py ^
        --images dataset/images/train ^
        --labels dataset/labels/train ^
        --out-labels dataset/labels_pseudo_seg/train ^
        --debug-dir runs/pseudo_seg_debug/train
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass
class BoxLabel:
    cls: str
    xyxy: tuple[int, int, int, int]


@dataclass
class PseudoResult:
    lines: list[str]
    heatmap: np.ndarray
    seedmap: np.ndarray
    mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert bbox or rectangular segment labels into pseudo YOLO segmentation labels."
    )
    parser.add_argument("--images", required=True, type=Path, help="Image directory, processed recursively.")
    parser.add_argument("--labels", required=True, type=Path, help="YOLO bbox/segment label directory.")
    parser.add_argument("--out-labels", required=True, type=Path, help="Output directory for YOLO segment labels.")
    parser.add_argument("--debug-dir", type=Path, help="Optional directory for image/heatmap/seed/mask debug panels.")
    parser.add_argument(
        "--classes",
        nargs="*",
        help="Optional class ids to process, e.g. --classes 0 2. Other classes are copied as bbox rectangles.",
    )
    parser.add_argument(
        "--seed-mode",
        choices=("center", "rust", "hybrid"),
        default="hybrid",
        help=(
            "Seed heatmap source. 'center' is class-agnostic and works for all rectangular labels, "
            "'rust' favors reddish-brown regions, and 'hybrid' combines both."
        ),
    )
    parser.add_argument("--fg-thres", type=float, default=0.62, help="Heatmap threshold for sure foreground seeds.")
    parser.add_argument("--bg-thres", type=float, default=0.24, help="Heatmap threshold for probable background.")
    parser.add_argument(
        "--pad", type=float, default=0.15, help="Context padding around each bbox, as box-size fraction."
    )
    parser.add_argument("--grabcut-iters", type=int, default=3, help="GrabCut refinement iterations. Use 0 to disable.")
    parser.add_argument("--min-area", type=float, default=0.00002, help="Minimum contour area as image-area fraction.")
    parser.add_argument("--max-contours", type=int, default=3, help="Maximum contours emitted per bbox.")
    parser.add_argument(
        "--approx-frac", type=float, default=0.003, help="Polygon simplification fraction of perimeter."
    )
    parser.add_argument(
        "--empty-policy",
        choices=("bbox", "skip"),
        default="bbox",
        help="What to do when no pseudo mask is found for a bbox.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run conversion without writing labels.")
    return parser.parse_args()


def iter_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def label_path_for(image_path: Path, images_root: Path, labels_root: Path) -> Path:
    return labels_root / image_path.relative_to(images_root).with_suffix(".txt")


def read_labels(label_path: Path, image_shape: tuple[int, int]) -> list[BoxLabel]:
    h, w = image_shape
    labels: list[BoxLabel] = []
    if not label_path.exists():
        return labels

    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        cls, values = parts[0], [float(x) for x in parts[1:]]
        if len(values) == 4:
            xc, yc, bw, bh = values
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h
        elif len(values) >= 6 and len(values) % 2 == 0:
            pts = np.asarray(values, dtype=np.float32).reshape(-1, 2)
            x1, y1 = pts.min(axis=0) * (w, h)
            x2, y2 = pts.max(axis=0) * (w, h)
        else:
            continue

        x1i = int(np.clip(np.floor(x1), 0, w - 1))
        y1i = int(np.clip(np.floor(y1), 0, h - 1))
        x2i = int(np.clip(np.ceil(x2), x1i + 1, w))
        y2i = int(np.clip(np.ceil(y2), y1i + 1, h))
        labels.append(BoxLabel(cls=cls, xyxy=(x1i, y1i, x2i, y2i)))
    return labels


def rust_heatmap(crop_bgr: np.ndarray) -> np.ndarray:
    """Return a 0..1 heatmap for reddish-brown rust-like pixels."""
    crop = crop_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(crop)

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue = hsv[..., 0]
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0

    orange_red = np.maximum(0.0, 1.0 - np.abs(hue - 13.0) / 28.0)
    deep_red = np.maximum(0.0, 1.0 - np.minimum(hue, 180.0 - hue) / 18.0)
    hue_score = np.maximum(orange_red, deep_red)

    red_over_blue = np.clip((r - b) * 2.2, 0.0, 1.0)
    warm_balance = np.clip((r + 0.35 * g - 0.85 * b), 0.0, 1.0)
    chroma = np.clip((sat - 0.12) / 0.55, 0.0, 1.0)
    brightness = np.clip((val - 0.05) / 0.35, 0.0, 1.0) * np.clip((1.0 - val + 0.25) / 0.6, 0.0, 1.0)

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    texture = cv2.normalize(cv2.magnitude(gx, gy), None, 0.0, 1.0, cv2.NORM_MINMAX)

    score = 0.45 * hue_score + 0.25 * red_over_blue + 0.18 * warm_balance + 0.12 * texture
    score *= np.maximum(chroma, 0.35) * np.maximum(brightness, 0.35)
    return cv2.GaussianBlur(np.clip(score, 0.0, 1.0), (0, 0), 1.2)


def center_heatmap(shape: tuple[int, int], rel_box: tuple[int, int, int, int]) -> np.ndarray:
    """Return a class-agnostic center prior inside a relative xyxy box."""
    h, w = shape
    x1, y1, x2, y2 = rel_box
    inside = np.zeros((h, w), dtype=np.uint8)
    inside[y1:y2, x1:x2] = 1
    if not inside.any():
        return np.zeros((h, w), dtype=np.float32)

    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 3).astype(np.float32)
    if float(dist.max()) > 0:
        dist /= float(dist.max())

    yy, xx = np.ogrid[:h, :w]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    sigma_x = max(1.0, (x2 - x1) * 0.35)
    sigma_y = max(1.0, (y2 - y1) * 0.35)
    gaussian = np.exp(-(((xx - cx) ** 2) / (2 * sigma_x**2) + ((yy - cy) ** 2) / (2 * sigma_y**2)))
    prior = np.maximum(dist, gaussian.astype(np.float32) * inside)
    return cv2.GaussianBlur(np.clip(prior, 0.0, 1.0), (0, 0), 0.8)


def padded_box(box: tuple[int, int, int, int], shape: tuple[int, int], pad_frac: float) -> tuple[int, int, int, int]:
    h, w = shape
    x1, y1, x2, y2 = box
    pad_x = round((x2 - x1) * pad_frac)
    pad_y = round((y2 - y1) * pad_frac)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    )


def fallback_rectangle(cls: str, box: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    h, w = shape
    x1, y1, x2, y2 = box
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return polygon_line(cls, np.asarray(pts, dtype=np.float32), (h, w))


def polygon_line(cls: str, contour: np.ndarray, shape: tuple[int, int]) -> str:
    h, w = shape
    pts = contour.reshape(-1, 2).astype(np.float32)
    pts[:, 0] = np.clip(pts[:, 0] / w, 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / h, 0.0, 1.0)
    coords = " ".join(f"{v:.6f}" for v in pts.reshape(-1))
    return f"{cls} {coords}"


def build_mask_for_box(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    px1, py1, px2, py2 = padded_box(box, (h, w), args.pad)
    crop = image[py1:py2, px1:px2]

    rel_x1, rel_y1 = x1 - px1, y1 - py1
    rel_x2, rel_y2 = x2 - px1, y2 - py1
    rust = rust_heatmap(crop)
    center = center_heatmap(crop.shape[:2], (rel_x1, rel_y1, rel_x2, rel_y2))
    if args.seed_mode == "rust":
        heat = rust
    elif args.seed_mode == "center":
        heat = center
    else:
        heat = np.maximum(center * 0.85, rust)

    inside = np.zeros(heat.shape, dtype=bool)
    inside[rel_y1:rel_y2, rel_x1:rel_x2] = True

    sure_fg = (heat >= args.fg_thres) & inside
    probable_bg = ((heat <= args.bg_thres) & inside) | ~inside

    if sure_fg.sum() < max(4, inside.sum() * 0.003):
        values = heat[inside]
        if values.size:
            adaptive = max(args.fg_thres * 0.75, float(np.quantile(values, 0.88)))
            sure_fg = (heat >= adaptive) & inside

    seed = np.full(heat.shape, 128, dtype=np.uint8)
    seed[probable_bg] = 0
    seed[sure_fg] = 255

    if args.grabcut_iters > 0 and sure_fg.any() and probable_bg.any():
        gc_mask = np.full(heat.shape, cv2.GC_PR_BGD, dtype=np.uint8)
        gc_mask[inside] = cv2.GC_PR_FGD
        gc_mask[probable_bg] = cv2.GC_BGD
        gc_mask[sure_fg] = cv2.GC_FGD
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(crop, gc_mask, None, bgd_model, fgd_model, args.grabcut_iters, cv2.GC_INIT_WITH_MASK)
            refined = np.isin(gc_mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
            mask = refined & inside & (heat >= args.bg_thres)
            mask |= sure_fg
        except cv2.error:
            mask = sure_fg
    else:
        mask = (heat >= max(args.bg_thres, args.fg_thres * 0.65)) & inside

    mask_u8 = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)

    full_heat = np.zeros((h, w), dtype=np.float32)
    full_seed = np.zeros((h, w), dtype=np.uint8)
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_heat[py1:py2, px1:px2] = np.maximum(full_heat[py1:py2, px1:px2], heat)
    full_seed[py1:py2, px1:px2] = np.maximum(full_seed[py1:py2, px1:px2], seed)
    full_mask[py1:py2, px1:px2] = np.maximum(full_mask[py1:py2, px1:px2], mask_u8)
    return full_heat, full_seed, full_mask


def contours_to_lines(
    cls: str,
    mask: np.ndarray,
    shape: tuple[int, int],
    args: argparse.Namespace,
) -> list[str]:
    image_area = shape[0] * shape[1]
    min_area_px = max(4.0, args.min_area * image_area)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    lines: list[str] = []
    for contour in contours[: args.max_contours]:
        if cv2.contourArea(contour) < min_area_px:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, max(1.0, args.approx_frac * peri), True)
        if len(approx) >= 3:
            lines.append(polygon_line(cls, approx, shape))
    return lines


def process_image(image_path: Path, label_path: Path, args: argparse.Namespace) -> PseudoResult | None:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"warning: failed to read image: {image_path}")
        return None

    h, w = image.shape[:2]
    labels = read_labels(label_path, (h, w))
    classes = set(args.classes) if args.classes else None

    out_lines: list[str] = []
    heat_all = np.zeros((h, w), dtype=np.float32)
    seed_all = np.zeros((h, w), dtype=np.uint8)
    mask_all = np.zeros((h, w), dtype=np.uint8)

    for label in labels:
        if classes is not None and label.cls not in classes:
            out_lines.append(fallback_rectangle(label.cls, label.xyxy, (h, w)))
            continue

        heat, seed, mask = build_mask_for_box(image, label.xyxy, args)
        heat_all = np.maximum(heat_all, heat)
        seed_all = np.maximum(seed_all, seed)
        mask_all = np.maximum(mask_all, mask)

        lines = contours_to_lines(label.cls, mask, (h, w), args)
        if lines:
            out_lines.extend(lines)
        elif args.empty_policy == "bbox":
            out_lines.append(fallback_rectangle(label.cls, label.xyxy, (h, w)))

    return PseudoResult(out_lines, heat_all, seed_all, mask_all)


def save_debug_panel(image_path: Path, result: PseudoResult, out_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    heat = cv2.applyColorMap((np.clip(result.heatmap, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    seed = cv2.cvtColor(result.seedmap, cv2.COLOR_GRAY2BGR)
    mask = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)
    panel = np.concatenate([image, heat, seed, mask], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def main() -> int:
    args = parse_args()
    image_paths = iter_images(args.images)
    if not image_paths:
        print(f"No images found under {args.images}")
        return 1

    converted = 0
    skipped = 0
    for image_path in image_paths:
        rel = image_path.relative_to(args.images)
        label_path = label_path_for(image_path, args.images, args.labels)
        out_label = args.out_labels / rel.with_suffix(".txt")
        result = process_image(image_path, label_path, args)
        if result is None:
            skipped += 1
            continue

        if not args.dry_run:
            out_label.parent.mkdir(parents=True, exist_ok=True)
            out_label.write_text("\n".join(result.lines) + ("\n" if result.lines else ""), encoding="utf-8")

        if args.debug_dir:
            save_debug_panel(image_path, result, args.debug_dir / rel.with_suffix(".jpg"))

        converted += 1

    print(f"Converted {converted} images. Skipped {skipped}. Output labels: {args.out_labels}")
    if args.debug_dir:
        print(f"Debug panels: {args.debug_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
