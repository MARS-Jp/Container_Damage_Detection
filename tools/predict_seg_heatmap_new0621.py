# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Run YOLO segmentation inference and visualize mask-derived heatmaps.

This tool does not require training a heatmap head. It uses an existing YOLO
segmentation model, converts predicted instance masks into:

- a Gaussian heatmap centered on the morphological thickness center of each predicted mask
- a seedmap from each mask's distance transform
- the regular YOLO segmentation overlay
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

IMG_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="YOLO segment predict with post-processed heatmap/seedmap overlays.")
    parser.add_argument("--model", required=True, help="Path to an existing YOLO segmentation .pt model.")
    parser.add_argument("--source", required=True, help="Image file, image directory, glob, or video source.")
    parser.add_argument("--project", default="runs/segment_heatmap", help="Output root directory.")
    parser.add_argument("--name", default="predict", help="Output run name.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0, or cpu. Defaults to Ultralytics auto.")
    parser.add_argument("--classes", type=int, nargs="+", default=None, help="Optional class IDs to keep.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity for heatmap and seedmap.")
    parser.add_argument(
        "--overlay-thres", type=float, default=0.05, help="Hide heatmap/seedmap values below this normalized threshold."
    )
    parser.add_argument(
        "--sigma-scale", type=float, default=1.0, help="Gaussian sigma scaler based on max object thickness."
    )
    parser.add_argument(
        "--heatmap-mode",
        choices=("mask", "center", "box"),
        default="mask",
        help=(
            "Heatmap generation mode: 'mask' keeps the Gaussian clipped by the predicted mask, "
            "'center' draws an unclipped Gaussian at the distance-transform center, "
            "and 'box' uses the detection box center."
        ),
    )
    parser.add_argument("--score-weight", action="store_true", help="Weight heatmap/seedmap strength by confidence.")
    parser.add_argument("--retina-masks", action="store_true", help="Use high-resolution YOLO masks.")
    parser.add_argument("--segment-fill", action="store_true", help="Fill segmentation masks in the Segment panel.")
    parser.add_argument("--hide-labels", action="store_true", help="Hide class/conf labels in the Segment panel.")
    parser.add_argument("--line-width", type=int, default=3, help="Segment outline and box line width.")
    parser.add_argument("--save-components", action="store_true", help="Also save individual overlay images.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse output directory if it exists.")
    return parser.parse_args()


def increment_path(path: Path, exist_ok: bool = False) -> Path:
    """Return a path that does not already exist unless exist_ok is True."""
    if exist_ok or not path.exists():
        return path
    for i in range(2, 10000):
        candidate = path.with_name(f"{path.name}{i}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not increment output path: {path}")


def safe_stem(path: str | Path, index: int) -> str:
    """Build a stable output stem for file, stream, or array result paths."""
    p = Path(str(path))
    stem = p.stem if p.stem else f"frame_{index:06d}"
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)


def masks_to_numpy(result) -> np.ndarray:
    """Return result masks as a boolean array shaped [N, H, W] in original image size."""
    if result.masks is None or result.masks.data is None:
        h, w = result.orig_img.shape[:2]
        return np.zeros((0, h, w), dtype=bool)

    masks = result.masks.data.detach().cpu().numpy()
    h, w = result.orig_img.shape[:2]
    resized = []
    for mask in masks:
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        resized.append(mask > 0.5)
    return np.stack(resized, axis=0) if resized else np.zeros((0, h, w), dtype=bool)


def get_scores(result, n: int) -> np.ndarray:
    """Return per-mask confidence scores, or ones when scores are unavailable."""
    if result.boxes is None or result.boxes.conf is None or len(result.boxes.conf) != n:
        return np.ones(n, dtype=np.float32)
    return result.boxes.conf.detach().cpu().numpy().astype(np.float32)


def get_boxes(result, n: int) -> np.ndarray | None:
    """Return per-mask boxes as xyxy pixels when available."""
    if result.boxes is None or result.boxes.xyxy is None or len(result.boxes.xyxy) != n:
        return None
    return result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)


def class_color(class_id: int, name: str = "") -> tuple[int, int, int]:
    """Return a stable BGR color for a class."""
    key = name.strip().lower()
    if key.startswith("r") or "rust" in key:
        return (220, 40, 255)
    if key.startswith("d") or "dent" in key or "damage" in key:
        return (245, 245, 245)
    palette = (
        (220, 40, 255),
        (245, 245, 245),
        (45, 180, 255),
        (80, 220, 80),
        (255, 160, 40),
        (180, 80, 255),
    )
    return palette[class_id % len(palette)]


def draw_segment_outline(
    image: np.ndarray, masks: np.ndarray, result, line_width: int = 3, hide_labels: bool = False
) -> np.ndarray:
    """Draw segmentation contours and boxes without filling mask interiors."""
    out = image.copy()
    if result.boxes is None or len(masks) == 0:
        return out

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    scores = get_scores(result, len(masks))
    names = result.names or {}

    for mask, box, class_id, score in zip(masks, boxes, classes, scores):
        name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
        color = class_color(class_id, name)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(out, contours, -1, color, line_width, lineType=cv2.LINE_AA)

        x1, y1, x2, y2 = box.round().astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_width, lineType=cv2.LINE_AA)
        if hide_labels:
            continue

        label = f"{name} {score:.2f}"
        font_scale = max(0.45, min(out.shape[:2]) / 900)
        thickness = max(1, line_width - 1)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_y1 = max(0, y1 - th - baseline - 4)
        label_y2 = label_y1 + th + baseline + 4
        cv2.rectangle(out, (x1, label_y1), (x1 + tw + 6, label_y2), color, -1)
        text_color = (8, 20, 55) if sum(color) > 560 else (255, 255, 255)
        cv2.putText(
            out,
            label,
            (x1 + 3, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    return out


def build_heatmap_and_seedmap(
    masks: np.ndarray,
    scores: np.ndarray,
    sigma_scale: float,
    score_weight: bool,
    heatmap_mode: str = "mask",
    boxes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create mask-derived Gaussian heatmap and distance-transform seedmap."""
    if masks.shape[0] == 0:
        h, w = masks.shape[1:]
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

    h, w = masks.shape[1:]
    yy, xx = np.ogrid[:h, :w]
    heatmap = np.zeros((h, w), dtype=np.float32)
    seedmap = np.zeros((h, w), dtype=np.float32)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        if not mask.any():
            continue
        weight = float(score) if score_weight else 1.0
        mask_u8 = mask.astype(np.uint8)

        # 1. 計算 Distance Transform (找尋生鏽區最厚/最深的地方)
        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
        max_dist = float(dist.max())

        # 2. 找出距離變換最大值的位置 (即特徵中心 cx, cy)
        if max_dist > 0:
            _, max_val, _, max_loc = cv2.minMaxLoc(dist)
            if heatmap_mode == "box" and boxes is not None:
                x1, y1, x2, y2 = boxes[i]
                cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                sigma = max(2.0, min(x2 - x1 + 1, y2 - y1 + 1) * sigma_scale * 0.5)
            else:
                cx, cy = max_loc

            # 3. 設定動態 Sigma，大小跟隨該點的厚度，避免長條形狀被壓扁
            # 乘上 2.5 是一個視覺平滑基準，再疊加使用者的 sigma_scale 微調
            if heatmap_mode != "box" or boxes is None:
                sigma = max(2.0, max_val * sigma_scale * 2.5)

            # 4. 畫出高斯分佈並套用 Mask 切割邊緣
            gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)).astype(np.float32)
            if heatmap_mode == "mask":
                gaussian *= mask_u8
            heatmap = np.maximum(heatmap, gaussian * weight)

            # 5. 生成正規化的 Seedmap
            seedmap = np.maximum(seedmap, (dist / max_dist).astype(np.float32) * weight)

    return normalize01(heatmap), normalize01(seedmap)


def normalize01(x: np.ndarray) -> np.ndarray:
    """Normalize a float image to [0, 1]."""
    max_value = float(x.max())
    if max_value <= 0:
        return x.astype(np.float32)
    return (x / max_value).astype(np.float32)


def overlay_colormap(
    image: np.ndarray, values: np.ndarray, alpha: float, colormap: int, min_value: float = 0.05
) -> np.ndarray:
    """Overlay a normalized heat image on a BGR image."""
    normalized = normalize01(values)
    active = normalized >= min_value
    out = image.copy()
    if not active.any():
        return out

    color = cv2.applyColorMap((normalized * 255).astype(np.uint8), colormap)
    strength = np.clip((normalized - min_value) / max(1e-6, 1.0 - min_value), 0.0, 1.0)
    alpha_map = (alpha * strength)[..., None]
    blended = image.astype(np.float32) * (1.0 - alpha_map) + color.astype(np.float32) * alpha_map
    out[active] = np.clip(blended[active], 0, 255).astype(np.uint8)
    return out


def make_panel(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    """Create a single-row labeled panel."""
    h = max(im.shape[0] for im in images)
    resized = []
    for image in images:
        if image.shape[0] != h:
            scale = h / image.shape[0]
            image = cv2.resize(image, (round(image.shape[1] * scale), h), interpolation=cv2.INTER_AREA)
        image = image.copy()
        resized.append(image)

    panel = np.concatenate(resized, axis=1)
    x = 0
    for image, label in zip(resized, labels):
        cv2.rectangle(panel, (x, 0), (x + image.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(panel, label, (x + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        x += image.shape[1]
    return panel


def save_result(result, save_dir: Path, index: int, args: argparse.Namespace) -> None:
    """Save segmentation, heatmap, seedmap, and panel visualizations for one result."""
    image = result.orig_img.copy()
    masks = masks_to_numpy(result)
    scores = get_scores(result, len(masks))
    boxes = get_boxes(result, len(masks))
    heatmap, seedmap = build_heatmap_and_seedmap(
        masks, scores, args.sigma_scale, args.score_weight, args.heatmap_mode, boxes
    )

    segment = (
        result.plot()
        if args.segment_fill
        else draw_segment_outline(image, masks, result, args.line_width, args.hide_labels)
    )
    heat_overlay = overlay_colormap(image, heatmap, args.alpha, cv2.COLORMAP_JET, args.overlay_thres)
    seed_overlay = overlay_colormap(image, seedmap, args.alpha, cv2.COLORMAP_TURBO, args.overlay_thres)
    panel = make_panel([image, segment, heat_overlay, seed_overlay], ["Image", "Segment", "Heatmap", "Seedmap"])

    stem = safe_stem(result.path, index)
    cv2.imwrite(str(save_dir / f"{stem}_panel.jpg"), panel)
    if args.save_components:
        cv2.imwrite(str(save_dir / f"{stem}_segment.jpg"), segment)
        cv2.imwrite(str(save_dir / f"{stem}_heatmap.jpg"), heat_overlay)
        cv2.imwrite(str(save_dir / f"{stem}_seedmap.jpg"), seed_overlay)
        cv2.imwrite(str(save_dir / f"{stem}_heatmap_raw.png"), (heatmap * 255).astype(np.uint8))
        cv2.imwrite(str(save_dir / f"{stem}_seedmap_raw.png"), (seedmap * 255).astype(np.uint8))


def main() -> None:
    """Run prediction and save visualizations."""
    args = parse_args()
    save_dir = increment_path(Path(args.project) / args.name, args.exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        classes=args.classes,
        retina_masks=args.retina_masks,
        stream=True,
        verbose=True,
    )
    for i, result in enumerate(results):
        save_result(result, save_dir, i, args)

    print(f"Saved segmentation heatmap visualizations to {save_dir}")


if __name__ == "__main__":
    main()
