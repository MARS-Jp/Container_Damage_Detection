"""以原圖、暗部 Gamma 提亮及 CLAHE 三種視圖進行 YOLO 分割推論。.

執行：python tools/predict_shadow.py
覆寫：python tools/predict_shadow.py --conf 0.20 --imgsz 960 --device cpu
比較：python tools/predict_shadow.py --views original
檢查：python tools/predict_shadow.py --save-views
三組比較：python tools/predict_parameter_sweep.py --suite all --compare
圖片只保留分割塗色，不顯示類別文字、信心分數、偵測框或輪廓線。
背景排除設定：tools/shadow_exclusion_regions.json；--no-exclusions 可停用。
目前只對指定圖片的右側橘色柱子排除 R 類別，未修改模型或 conf 門檻。

輸出在專案 runs/segment_shadow/run_時間/模型/圖片/，每次執行使用新資料夾：
  *__original.jpg：相同參數下只使用原圖的預測（不是 predict-2.py 的舊設定）。
  *__all.jpg、*__class_*.jpg：合併結果及各類別的分割圖。
  predictions.json：每個預測的類別、分數、來源視圖；settings.json 記錄本次參數。
--save-views 另外儲存增強輸入與各次預測。預設每張圖片推論三次，耗時會增加。

增強只改變亮度，不改變圖片座標。跨視圖以同類別的遮罩 IoU 去重，
保留分數較高的完整預測，不把不同遮罩強制聯集。結果全部畫回原圖。
此方法用於嘗試改善陰影處漏檢；新增預測可能是誤檢，需用標註驗證。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.predict_seg_by_class import class_color, normalize_names, prediction_metrics, safe_name, write_image

# ============================ 使用者設定區 ============================
# 可像 predict-2.py 一樣直接修改這兩個清單；每個模型都會推論所有圖片。
WEIGHT_SOURCES = [r"C:\Users\USER\Downloads\last0912.pt"]
DATASET_ROOT = Path(r"C:\Users\USER\Desktop\碩論\貨櫃-20221012T025654Z-001\貨櫃\Container_damage Dataset")
DATASET_SOURCES = [
    DATASET_ROOT / "train/B/CBHU0708054-A.JPG",
    # DATASET_ROOT / "train/B/CBHU0708054-B.JPG",
    # DATASET_ROOT / "train/B/IMG_4155.JPG",
    # DATASET_ROOT / "val 驗證/IMAG0749.jpg",
    # DATASET_ROOT / "val 驗證/P5021976.JPG",
    # DATASET_ROOT / "val 驗證/P6232556.JPG",
    # DATASET_ROOT / "val 驗證/P8293423.JPG",
]
OUTPUT_DIR = REPO_ROOT / "runs/segment_shadow"
IMAGE_SIZE = 640  # 細小損傷可用 1280；顯存不足時改成 960 或 640。
CONFIDENCE = 0.15  # 漏檢多可試 0.10；誤檢多可試 0.25。
IOU_THRESHOLD = 0.7  # 各次推論內的框 NMS 門檻。
DEVICE = "auto"  # 有 CUDA 使用 GPU 0，否則使用 CPU；也可指定 "0"。
GAMMA = 0.5  # 0 < gamma < 1；越小提亮越強，1 表示不提亮。
CLAHE_CLIP = 1.0  # 局部對比增強強度；過大可能放大紋理與雜訊。
SHADOW_THRESHOLD = 255  # 0..255 的亮度門檻；亮度 >= 此值的像素保持原樣。
MERGE_IOU = 0.90  # 不同視圖、同類別的遮罩 IoU 超過此值才去重。
EXCLUSION_REGIONS = SCRIPT_DIR / "shadow_exclusion_regions.json"
# =====================================================================

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read editable defaults and optional command-line overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", type=Path, default=WEIGHT_SOURCES)
    parser.add_argument("--source", nargs="+", type=Path, default=DATASET_SOURCES)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--conf", type=float, default=CONFIDENCE)
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--clahe-clip", type=float, default=CLAHE_CLIP)
    parser.add_argument("--shadow-threshold", type=int, default=SHADOW_THRESHOLD)
    parser.add_argument("--merge-iou", type=float, default=MERGE_IOU)
    parser.add_argument("--max-det", type=int, default=300, help="每個視圖的預測數上限")
    parser.add_argument(
        "--views",
        nargs="+",
        choices=("original", "gamma", "clahe"),
        default=["original", "gamma", "clahe"],
        help="原圖始終會執行一次",
    )
    parser.add_argument("--save-views", action="store_true", help="額外儲存增強圖與各視圖的預測")
    parser.add_argument("--alpha", type=float, default=0.35, help="遮罩填色透明度")
    exclusions = parser.add_mutually_exclusive_group()
    exclusions.add_argument(
        "--exclude-regions", type=Path, default=EXCLUSION_REGIONS, help="依原始圖片路徑匹配的背景排除區 JSON"
    )
    exclusions.add_argument(
        "--no-exclusions",
        dest="exclude_regions",
        action="store_const",
        const=None,
        help="停用背景排除，保留模型的完整預測",
    )
    args = parser.parse_args(argv)
    if args.imgsz <= 0 or args.imgsz % 32:
        parser.error("--imgsz 必須是正整數且為 32 的倍數")
    for key in ("conf", "iou", "merge_iou", "alpha"):
        if not 0 <= getattr(args, key) <= 1:
            parser.error(f"--{key.replace('_', '-')} 必須介於 0 和 1")
    if not 0 < args.gamma <= 1 or not 0 < args.clahe_clip < float("inf"):
        parser.error("--gamma 必須 > 0 且 <= 1；--clahe-clip 必須為有限正數")
    if not 1 <= args.shadow_threshold <= 255 or args.max_det < 1:
        parser.error("--shadow-threshold 必須為 1..255；--max-det 必須 >= 1")
    args.views = list(dict.fromkeys(["original", *args.views]))
    try:
        args.exclusion_rules = load_exclusion_rules(resolve_path(args.exclude_regions)) if args.exclude_regions else []
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(f"無法讀取背景排除區：{error}")
    return args


def resolve_path(value: str | Path) -> Path:
    """Resolve relative paths from tools/, matching predict-2.py."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else SCRIPT_DIR / path).resolve()


def collect_images(sources: list[Path]) -> list[Path]:
    """Expand image directories recursively, de-duplicating overlapping sources."""
    images = []
    for source in sources:
        if source.is_dir():
            found = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        elif source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
            found = [source]
        else:
            raise FileNotFoundError(f"找不到圖片或不支援此來源：{source}")
        if not found:
            raise FileNotFoundError(f"資料夾內沒有可推論的圖片：{source}")
        images.extend(found)
    return list(dict.fromkeys(images))


def load_exclusion_rules(path: Path) -> list[dict]:
    """Load explicit image-specific polygons, without generalizing them to other images."""
    rules = json.loads(path.read_text(encoding="utf-8"))["regions"]
    if not isinstance(rules, list):
        raise ValueError("regions 必須是清單")
    for rule in rules:
        points = np.asarray(rule["polygon_normalized"], dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 3
            or not np.isfinite(points).all()
            or np.any((points < 0) | (points > 1))
        ):
            raise ValueError("polygon_normalized 至少需要三個 0..1 座標點")
        if not isinstance(rule["source"], str) or not rule["source"]:
            raise ValueError("每個排除區必須指定圖片 source")
        if (
            not isinstance(rule["classes"], list)
            or not rule["classes"]
            or not all(isinstance(name, str) and name for name in rule["classes"])
        ):
            raise ValueError("每個排除區必須指定非空 classes 清單")
    return rules


def exclusion_masks(source: Path, shape: tuple[int, int], names: dict, rules: list[dict]) -> tuple[dict, list]:
    """Rasterize only this exact source's exclusions, using original-image coordinates."""
    matched = [rule for rule in rules if resolve_path(rule["source"]) == source.resolve()]
    masks = {}
    height, width = shape
    for rule in matched:
        region = np.zeros(shape, dtype=np.uint8)
        points = np.rint(np.asarray(rule["polygon_normalized"]) * [width - 1, height - 1]).astype(np.int32)
        cv2.fillPoly(region, [points], 1)
        for name in rule["classes"]:
            ids = [class_id for class_id, class_name in names.items() if class_name.casefold() == name.casefold()]
            if len(ids) != 1:
                raise ValueError(f"背景排除類別 {name!r} 無法對應模型類別 {names}")
            class_id = ids[0]
            masks[class_id] = masks.get(class_id, np.zeros(shape, dtype=bool)) | region.astype(bool)
    return masks, matched


def enhance_shadow(image: np.ndarray, view: str, args: argparse.Namespace) -> np.ndarray:
    """Lift dark luminance only, smoothly fading enhancement toward bright areas."""
    if view == "original":
        return image
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    luminance = ycrcb[:, :, 0].copy()
    if view == "gamma":
        table = np.rint(255 * (np.arange(256, dtype=np.float32) / 255) ** args.gamma).astype(np.uint8)
        target = cv2.LUT(luminance, table)
    elif view == "clahe":
        target = cv2.createCLAHE(clipLimit=args.clahe_clip, tileGridSize=(8, 8)).apply(luminance)
    else:
        raise ValueError(f"Unknown view: {view}")
    # Gamma=1 and already-bright pixels remain byte-for-byte identical to the input.
    strength = np.clip((args.shadow_threshold - luminance.astype(np.float32)) / args.shadow_threshold, 0, 1)
    lifted = luminance + strength * np.maximum(target.astype(np.float32) - luminance, 0)
    ycrcb[:, :, 0] = np.rint(lifted).clip(0, 255).astype(np.uint8)
    enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    unchanged = ycrcb[:, :, 0] == luminance
    enhanced[unchanged] = image[unchanged]
    return enhanced


@dataclass
class Detection:
    """One unchanged model prediction, with a binary mask in original-image coordinates."""

    box: np.ndarray  # xyxy, confidence, class_id
    mask: np.ndarray
    view: str


def exclude_background(detections: list[Detection], masks: dict[int, np.ndarray]) -> list[Detection]:
    """Clip only the requested class pixels; discard predictions with no remaining pixels."""
    kept = []
    for detection in detections:
        excluded = masks.get(int(detection.box[5]))
        if excluded is None or not np.any(detection.mask & excluded):
            kept.append(detection)
            continue
        mask = detection.mask & ~excluded
        if mask.any():
            kept.append(Detection(detection.box.copy(), mask, detection.view))
    return kept


def extract_detections(result, view: str, shape: tuple[int, int]) -> list[Detection]:
    """Require retina masks so no stretching or padding offsets enter the fusion."""
    if result.boxes is None or len(result.boxes) == 0:
        return []
    if result.masks is None:
        raise RuntimeError("模型有框但沒有分割遮罩，請確認使用 segmentation 權重")
    masks = result.masks.data.detach().cpu().numpy().astype(bool)
    boxes = result.boxes.data.detach().cpu().numpy()
    if masks.shape != (len(boxes), *shape):
        raise RuntimeError(f"遮罩座標不符：{masks.shape}，預期 {(len(boxes), *shape)}")
    return [Detection(box.copy(), mask, view) for box, mask in zip(boxes, masks) if mask.any()]


def merge_detections(detections: list[Detection], threshold: float) -> list[Detection]:
    """Suppress cross-view duplicates of the same class using exact binary-mask IoU.

    Different classes and predictions from the same view never suppress each other. Scores select a complete mask;
    neither scores nor mask pixels are averaged/unioned.
    """
    kept = []
    areas = [int(d.mask.sum()) for d in detections]
    bounds = [cv2.boundingRect(d.mask.astype(np.uint8)) for d in detections]
    for index in sorted(range(len(detections)), key=lambda i: -float(detections[i].box[4])):
        candidate = detections[index]
        duplicate = False
        for previous in kept:
            other = detections[previous]
            if candidate.view == other.view or candidate.box[5] != other.box[5]:
                continue
            x, y, w, h = bounds[index]
            ox, oy, ow, oh = bounds[previous]
            left, top, right, bottom = max(x, ox), max(y, oy), min(x + w, ox + ow), min(y + h, oy + oh)
            if right <= left or bottom <= top:
                continue
            intersection = np.count_nonzero(candidate.mask[top:bottom, left:right] & other.mask[top:bottom, left:right])
            union = areas[index] + areas[previous] - intersection
            if union and intersection / union > threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(index)
    return [detections[i] for i in kept]


def render(image: np.ndarray, detections: list[Detection], names: dict, args: argparse.Namespace) -> np.ndarray:
    """Only blend actual mask pixels; preserve holes and omit text, boxes and outlines."""
    output = image.copy()
    for detection in detections:
        class_id = int(detection.box[5])
        color = class_color(class_id)
        mask = detection.mask
        output[mask] = np.rint(output[mask] * (1 - args.alpha) + np.array(color) * args.alpha).astype(np.uint8)
    return output


def predict_image(model, source: Path, output: Path, args: argparse.Namespace) -> dict:
    """Run all photometric views sequentially, save comparisons and prediction provenance."""
    image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"圖片無法解碼：{source}")
    names = normalize_names(model.names)
    excluded_masks, matched_rules = exclusion_masks(source, image.shape[:2], names, args.exclusion_rules)
    stem = safe_name(source.stem)
    candidates = []
    counts = {}
    raw_counts = {}
    for view in args.views:
        enhanced = enhance_shadow(image, view, args)
        result = model.predict(
            source=enhanced,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            retina_masks=True,
            agnostic_nms=False,
            max_det=args.max_det,
            save=False,
            verbose=False,
        )[0]
        detections = extract_detections(result, view, image.shape[:2])
        raw_counts[view] = len(detections)
        detections = exclude_background(detections, excluded_masks)
        candidates.extend(detections)
        counts[view] = len(detections)
        if view == "original":
            write_image(output / f"{stem}__original.jpg", render(image, detections, names, args))
        elif args.save_views:
            write_image(output / "views" / f"{stem}__{view}_input.png", enhanced)
            write_image(output / "views" / f"{stem}__{view}_prediction.jpg", render(image, detections, names, args))
        del result  # Do not retain the three full-resolution mask batches on the GPU.
    merged = merge_detections(candidates, args.merge_iou)
    write_image(output / f"{stem}__all.jpg", render(image, merged, names, args))
    for class_id, name in names.items():
        selected = [d for d in merged if int(d.box[5]) == class_id]
        filename = f"{stem}__class_{class_id:03d}_{safe_name(name)}.jpg"
        write_image(output / filename, render(image, selected, names, args))
    report = {
        "source": str(source),
        "view_counts": counts,
        "merged_count": len(merged),
        "raw_view_counts": raw_counts,
        "excluded_regions": matched_rules,
        "exclusion_region_pixels_by_class": {str(key): int(mask.sum()) for key, mask in excluded_masks.items()},
        "metrics": prediction_metrics(
            np.stack([d.mask for d in merged]) if merged else np.zeros((0, *image.shape[:2]), dtype=bool),
            np.array([int(d.box[5]) for d in merged], dtype=int),
            names,
            image.shape[:2],
        ),
        "detections": [
            {
                "class_id": int(d.box[5]),
                "class_name": names[int(d.box[5])],
                "confidence": float(d.box[4]),
                "view": d.view,
                "box_xyxy": d.box[:4].tolist(),
                "mask_pixels": int(d.mask.sum()),
            }
            for d in merged
        ],
    }
    (output / "predictions.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    """Load each model once and process every image with isolated output directories."""
    args = parse_args()
    weights = list(dict.fromkeys(resolve_path(p) for p in args.weights))
    if not weights or not args.source:
        raise ValueError("WEIGHT_SOURCES 和 DATASET_SOURCES 都至少需要一個路徑")
    for weight in weights:
        if not weight.is_file():
            raise FileNotFoundError(f"找不到模型：{weight}")
    images = collect_images([resolve_path(p) for p in args.source])
    import torch

    from ultralytics import YOLO

    if args.device == "auto":
        args.device = "0" if torch.cuda.is_available() else "cpu"
    output = resolve_path(args.output) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    settings = {**vars(args), "weights": weights, "source": images, "retina_masks": True}
    (output / "settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"裝置：{args.device}；每張圖推論 {len(args.views)} 次；輸出：{output}", flush=True)
    for model_index, weight in enumerate(weights, 1):
        model = YOLO(str(weight))
        if model.task != "segment":
            raise ValueError(f"需要分割模型，{weight} 的 task 為 {model.task!r}")
        model_output = output / f"{model_index:02d}_{safe_name(weight.stem)}"
        for image_index, source in enumerate(images, 1):
            image_output = model_output / f"{image_index:06d}_{safe_name(source.stem)}"
            print(f"[{model_index}/{len(weights)}][{image_index}/{len(images)}] {source.name}", flush=True)
            report = predict_image(model, source, image_output, args)
            print(f"  各視圖：{report['view_counts']}；合併後：{report['merged_count']}", flush=True)
        del model
    print(f"完成。結果位置：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
