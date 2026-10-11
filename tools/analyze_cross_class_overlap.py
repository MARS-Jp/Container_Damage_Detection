"""Analyze DENT/RUST mask overlap and class-filtered inference changes."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulate_feature_suppression import extract_prediction_dict, letterbox, roi_raw_scores

DEFAULT_SOURCES = [
    r"C:\Users\USER\Desktop\碩論\貨櫃-20221012T025654Z-001\貨櫃\Container_damage Dataset\train\B\CBHU0708054-A.JPG",
    r"C:\Users\USER\Desktop\碩論\貨櫃-20221012T025654Z-001\貨櫃\Container_damage Dataset\train\B\CBHU0708054-B.JPG",
    r"C:\Users\USER\Desktop\碩論\貨櫃-20221012T025654Z-001\貨櫃\Container_damage Dataset\train\B\IMG_4155.JPG",
    r"C:\Users\USER\Desktop\碩論\貨櫃-20221012T025654Z-001\貨櫃\Container_damage Dataset\val 驗證\IMAG0749.jpg",
    r"C:\Users\USER\Desktop\碩論\貨櫃-20221012T025654Z-001\貨櫃\Container_damage Dataset\val 驗證\P5021976.JPG",
    r"C:\Users\USER\Desktop\碩論\貨櫃-20221012T025654Z-001\貨櫃\Container_damage Dataset\val 驗證\P6232556.JPG",
    r"C:\Users\USER\Desktop\碩論\貨櫃-20221012T025654Z-001\貨櫃\Container_damage Dataset\val 驗證\P8293423.JPG",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path(r"C:\Users\USER\Downloads\last0918-1.pt"))
    parser.add_argument("--source", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--output", type=Path, default=Path("runs/cross_class_overlap_last0918-1"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def masks_from_result(result: Any) -> list[np.ndarray]:
    h, w = result.orig_img.shape[:2]
    if result.masks is None:
        return []
    data = result.masks.data.detach().cpu().numpy()
    output = []
    for mask in data:
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        output.append(mask > 0.5)
    return output


def collect(result: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    masks = masks_from_result(result)
    if result.boxes is None:
        return []
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    scores = result.boxes.conf.detach().cpu().numpy()
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    return [
        {"class_id": int(cls), "class_name": names[int(cls)], "confidence": float(score), "mask": mask, "box": box}
        for cls, score, mask, box in zip(classes, scores, masks, boxes)
    ]


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return intersection / union if union else 0.0


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return float(intersection / union) if union else 0.0


def match_detections(
    full: list[dict[str, Any]], filtered: list[dict[str, Any]], class_id: int
) -> dict[int, dict[str, Any]]:
    full_ids = [i for i, row in enumerate(full) if row["class_id"] == class_id]
    filtered_ids = [i for i, row in enumerate(filtered) if row["class_id"] == class_id]
    available = set(full_ids)
    matches: dict[int, dict[str, Any]] = {}
    for filtered_index in filtered_ids:
        candidates = [(mask_iou(filtered[filtered_index]["mask"], full[i]["mask"]), i) for i in available]
        if not candidates:
            continue
        overlap, full_index = max(candidates)
        if overlap >= 0.5:
            matches[full_index] = {"filtered": filtered[filtered_index], "mask_iou": overlap}
            available.remove(full_index)
    return matches


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("no_rows\n", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    names_raw = model.names
    names = (
        {int(k): str(v) for k, v in names_raw.items()}
        if isinstance(names_raw, dict)
        else dict(enumerate(map(str, names_raw)))
    )
    wanted: dict[str, int] = {}
    for target in ("dent", "rust"):
        candidates = [class_id for class_id, name in names.items() if name.strip().casefold() == target]
        if not candidates:
            candidates = [class_id for class_id, name in names.items() if target in name.strip().casefold()]
        if not candidates:
            abbreviation = {"dent": "d", "rust": "r"}[target]
            candidates = [class_id for class_id, name in names.items() if name.strip().casefold() == abbreviation]
        if not candidates:
            raise ValueError(f"Could not identify {target.upper()} in model names: {names}")
        wanted[target] = candidates[0]

    args.output.mkdir(parents=True, exist_ok=True)
    overlap_rows: list[dict[str, Any]] = []
    filtered_pair_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    raw_roi_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for source_value in args.source:
        source = Path(source_value)
        full_result = next(
            iter(
                model.predict(
                    source=str(source),
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    retina_masks=True,
                    stream=True,
                    save=False,
                    verbose=False,
                )
            )
        )
        full = collect(full_result, names)
        filtered_by_name: dict[str, list[dict[str, Any]]] = {}
        matches_by_name: dict[str, dict[int, dict[str, Any]]] = {}
        for class_name, class_id in wanted.items():
            solo_result = next(
                iter(
                    model.predict(
                        source=str(source),
                        imgsz=args.imgsz,
                        conf=args.conf,
                        iou=args.iou,
                        device=args.device,
                        retina_masks=True,
                        classes=[class_id],
                        stream=True,
                        save=False,
                        verbose=False,
                    )
                )
            )
            filtered = collect(solo_result, names)
            filtered_by_name[class_name] = filtered
            matches_by_name[class_name] = match_detections(full, filtered, class_id)

        for class_name, competitor_name in (("dent", "rust"), ("rust", "dent")):
            class_id = wanted[class_name]
            competitor_id = wanted[competitor_name]
            solo = filtered_by_name[class_name]
            competitor_solo = filtered_by_name[competitor_name]
            matched_solo = list(matches_by_name[class_name].values())
            for solo_index, detection in enumerate(solo):
                is_in_all = any(mask_iou(item["filtered"]["mask"], detection["mask"]) >= 0.99 for item in matched_solo)
                if is_in_all:
                    continue
                competitors = [row for row in competitor_solo if row["class_id"] == competitor_id]
                best = max(competitors, key=lambda row: box_iou(detection["box"], row["box"]), default=None)
                best_box_iou = box_iou(detection["box"], best["box"]) if best else 0.0
                best_mask_pixels = int(np.count_nonzero(detection["mask"] & best["mask"])) if best else 0
                unmatched_rows.append(
                    {
                        "image": source.name,
                        "class": class_name.upper(),
                        "solo_index": solo_index,
                        "confidence_only": detection["confidence"],
                        "present_in_all": False,
                        "best_other_class": competitor_name.upper() if best else "",
                        "best_other_confidence_only": best["confidence"] if best else "",
                        "box_iou_with_best_other": best_box_iou,
                        "mask_intersection_pixels_with_best_other": best_mask_pixels,
                        "mask_iou_with_best_other": mask_iou(detection["mask"], best["mask"]) if best else 0.0,
                    }
                )

        dent_id, rust_id = wanted["dent"], wanted["rust"]
        dent_indices = [i for i, row in enumerate(full) if row["class_id"] == dent_id]
        rust_indices = [i for i, row in enumerate(full) if row["class_id"] == rust_id]
        image = full_result.orig_img.copy()
        dent_union = np.zeros(image.shape[:2], dtype=bool)
        rust_union = np.zeros(image.shape[:2], dtype=bool)
        for i in dent_indices:
            dent_union |= full[i]["mask"]
        for i in rust_indices:
            rust_union |= full[i]["mask"]
        intersection = dent_union & rust_union
        intersection_pixels = int(np.count_nonzero(intersection))
        union_pixels = int(np.count_nonzero(dent_union | rust_union))

        if intersection_pixels:
            transformed, ratio, left, top = letterbox(image, args.imgsz)
            height, width = image.shape[:2]
            new_width, new_height = round(width * ratio), round(height * ratio)

            def map_mask(mask: np.ndarray) -> np.ndarray:
                resized = cv2.resize(mask.astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                mapped = np.zeros((args.imgsz, args.imgsz), dtype=np.uint8)
                mapped[top : top + new_height, left : left + new_width] = resized
                return mapped

            rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().unsqueeze(0) / 255.0
            with torch.inference_mode():
                raw_predictions = extract_prediction_dict(model.model(tensor.to(next(model.model.parameters()).device)))
            overlap_scores = roi_raw_scores(raw_predictions, map_mask(intersection), (dent_id, rust_id))
            dent_only_scores = (
                roi_raw_scores(raw_predictions, map_mask(dent_union & ~rust_union), (dent_id, rust_id))
                if np.any(dent_union & ~rust_union)
                else {}
            )
            rust_only_scores = (
                roi_raw_scores(raw_predictions, map_mask(rust_union & ~dent_union), (dent_id, rust_id))
                if np.any(rust_union & ~dent_union)
                else {}
            )
            raw_row: dict[str, Any] = {
                "image": source.name,
                "overlap_pixels": intersection_pixels,
                "dent_head_topk_overlap": overlap_scores.get("head_dent_topk", ""),
                "dent_head_max_overlap": overlap_scores.get("head_dent_max", ""),
                "rust_head_topk_overlap": overlap_scores.get("head_rust_topk", ""),
                "rust_head_max_overlap": overlap_scores.get("head_rust_max", ""),
                "dent_head_topk_dent_only_pixels": dent_only_scores.get("head_dent_topk", ""),
                "rust_head_topk_rust_only_pixels": rust_only_scores.get("head_rust_topk", ""),
            }
            if dent_only_scores.get("head_dent_topk"):
                raw_row["dent_overlap_to_dent_only_ratio"] = (
                    overlap_scores["head_dent_topk"] / dent_only_scores["head_dent_topk"]
                )
            if rust_only_scores.get("head_rust_topk"):
                raw_row["rust_overlap_to_rust_only_ratio"] = (
                    overlap_scores["head_rust_topk"] / rust_only_scores["head_rust_topk"]
                )
            raw_roi_rows.append(raw_row)
        pair_count = 0
        for dent_index in dent_indices:
            dent = full[dent_index]
            for rust_index in rust_indices:
                rust = full[rust_index]
                both = dent["mask"] & rust["mask"]
                pixels = int(np.count_nonzero(both))
                if not pixels:
                    continue
                pair_count += 1
                dmatch = matches_by_name["dent"].get(dent_index)
                rmatch = matches_by_name["rust"].get(rust_index)
                overlap_rows.append(
                    {
                        "image": source.name,
                        "dent_full_index": dent_index,
                        "dent_conf_all": dent["confidence"],
                        "dent_conf_only": dmatch["filtered"]["confidence"] if dmatch else "",
                        "dent_conf_delta_only_minus_all": dmatch["filtered"]["confidence"] - dent["confidence"]
                        if dmatch
                        else "",
                        "dent_match_mask_iou": dmatch["mask_iou"] if dmatch else "",
                        "rust_full_index": rust_index,
                        "rust_conf_all": rust["confidence"],
                        "rust_conf_only": rmatch["filtered"]["confidence"] if rmatch else "",
                        "rust_conf_delta_only_minus_all": rmatch["filtered"]["confidence"] - rust["confidence"]
                        if rmatch
                        else "",
                        "rust_match_mask_iou": rmatch["mask_iou"] if rmatch else "",
                        "intersection_pixels": pixels,
                        "dent_coverage": pixels / max(1, int(np.count_nonzero(dent["mask"]))),
                        "rust_coverage": pixels / max(1, int(np.count_nonzero(rust["mask"]))),
                        "pair_mask_iou": mask_iou(dent["mask"], rust["mask"]),
                    }
                )

        # Also inspect candidate overlaps when each class is run alone. This
        # catches a candidate that may have been removed from the joint result.
        dent_solo = filtered_by_name["dent"]
        rust_solo = filtered_by_name["rust"]
        for di, dent in enumerate(dent_solo):
            dent_full_match = next(
                (v for k, v in matches_by_name["dent"].items() if mask_iou(full[k]["mask"], dent["mask"]) >= 0.5), None
            )
            for ri, rust in enumerate(rust_solo):
                pixels = int(np.count_nonzero(dent["mask"] & rust["mask"]))
                if not pixels:
                    continue
                rust_full_match = next(
                    (v for k, v in matches_by_name["rust"].items() if mask_iou(full[k]["mask"], rust["mask"]) >= 0.5),
                    None,
                )
                filtered_pair_rows.append(
                    {
                        "image": source.name,
                        "dent_solo_index": di,
                        "dent_conf_only": dent["confidence"],
                        "dent_present_in_all": dent_full_match is not None,
                        "dent_conf_all": dent_full_match["filtered"]["confidence"] if dent_full_match else "",
                        "rust_solo_index": ri,
                        "rust_conf_only": rust["confidence"],
                        "rust_present_in_all": rust_full_match is not None,
                        "rust_conf_all": rust_full_match["filtered"]["confidence"] if rust_full_match else "",
                        "intersection_pixels": pixels,
                        "dent_coverage": pixels / max(1, int(np.count_nonzero(dent["mask"]))),
                        "rust_coverage": pixels / max(1, int(np.count_nonzero(rust["mask"]))),
                        "pair_mask_iou": mask_iou(dent["mask"], rust["mask"]),
                    }
                )

        d_deltas = [v["filtered"]["confidence"] - full[k]["confidence"] for k, v in matches_by_name["dent"].items()]
        r_deltas = [v["filtered"]["confidence"] - full[k]["confidence"] for k, v in matches_by_name["rust"].items()]
        summary_rows.append(
            {
                "image": source.name,
                "dent_count_all": len(dent_indices),
                "rust_count_all": len(rust_indices),
                "dent_count_only": len(filtered_by_name["dent"]),
                "rust_count_only": len(filtered_by_name["rust"]),
                "overlapping_instance_pairs": pair_count,
                "union_intersection_pixels": intersection_pixels,
                "union_intersection_over_damage_union": intersection_pixels / max(1, union_pixels),
                "dent_matched_all_to_only": len(matches_by_name["dent"]),
                "rust_matched_all_to_only": len(matches_by_name["rust"]),
                "dent_mean_confidence_delta_only_minus_all": float(np.mean(d_deltas)) if d_deltas else "",
                "rust_mean_confidence_delta_only_minus_all": float(np.mean(r_deltas)) if r_deltas else "",
            }
        )

        overlay = image.copy()
        tint = np.zeros_like(image)
        tint[dent_union] = (255, 80, 20)  # blue/cyan in BGR
        tint[rust_union] = (20, 80, 255)  # orange/red in BGR
        tint[intersection] = (255, 0, 255)
        overlay = cv2.addWeighted(overlay, 0.66, tint, 0.34, 0)
        cv2.putText(
            overlay,
            f"DENT={len(dent_indices)} RUST={len(rust_indices)} overlap={intersection_pixels}px pairs={pair_count}",
            (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(args.output / f"{source.stem}__overlap.png"), overlay)
        print(
            f"{source.name}: DENT {len(dent_indices)}, RUST {len(rust_indices)}, overlap pairs {pair_count}, pixels {intersection_pixels}"
        )

    write_csv(args.output / "overlap_pairs.csv", overlap_rows)
    write_csv(args.output / "class_only_overlap_pairs.csv", filtered_pair_rows)
    write_csv(args.output / "class_only_unmatched.csv", unmatched_rows)
    write_csv(args.output / "overlap_roi_raw_scores.csv", raw_roi_rows)
    write_csv(args.output / "image_summary.csv", summary_rows)
    report = [
        "# DENT/RUST cross-class overlap analysis",
        "",
        f"- Weights: `{args.weights}`",
        f"- Inference: imgsz={args.imgsz}, conf={args.conf}, iou={args.iou}, device={args.device}, retina_masks=True",
        f"- Classes: DENT={wanted['dent']} ({names[wanted['dent']]}), RUST={wanted['rust']} ({names[wanted['rust']]})",
        "- Confidence deltas compare matched masks (mask IoU >= 0.50) from all-class inference against the same class-filtered inference.",
        "- Positive delta means the class-only run has higher confidence. Since class filtering affects post-processing, this comparison screens inference-time suppression; it does not establish learned-feature causality.",
        "",
        "## Per-image summary",
        "",
        "| Image | DENT | RUST | Overlap pairs | Intersect px | DENT Δconf | RUST Δconf |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        d = (
            f"{row['dent_mean_confidence_delta_only_minus_all']:.4f}"
            if row["dent_mean_confidence_delta_only_minus_all"] != ""
            else "N/A"
        )
        r = (
            f"{row['rust_mean_confidence_delta_only_minus_all']:.4f}"
            if row["rust_mean_confidence_delta_only_minus_all"] != ""
            else "N/A"
        )
        report.append(
            f"| {row['image']} | {row['dent_count_all']} | {row['rust_count_all']} | {row['overlapping_instance_pairs']} | {row['union_intersection_pixels']} | {d} | {r} |"
        )
    report.extend(
        [
            "",
            "## Raw class-head scores in overlap ROIs",
            "",
            "| Image | ROI px | DENT overlap | DENT non-overlap | DENT ratio | RUST overlap | RUST non-overlap | RUST ratio |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in raw_roi_rows:
        vals = [
            row.get("dent_head_topk_overlap"),
            row.get("dent_head_topk_dent_only_pixels"),
            row.get("dent_overlap_to_dent_only_ratio"),
            row.get("rust_head_topk_overlap"),
            row.get("rust_head_topk_rust_only_pixels"),
            row.get("rust_overlap_to_rust_only_ratio"),
        ]
        formatted = [f"{value:.4f}" if isinstance(value, (float, int)) else "N/A" for value in vals]
        report.append(f"| {row['image']} | {row['overlap_pixels']} | " + " | ".join(formatted) + " |")
    report.extend(
        [
            "",
            "These are mean top-k sigmoid values from the pre-NMS class head over feature-grid cells touching each ROI. The non-overlap columns use the same image's predicted class mask union with the opposite class removed. Ratios below 1 indicate a lower class-head response in the overlap ROI; a single image cannot isolate causation from local appearance, size, or annotation/prediction error.",
            "",
            "Pair-level instance confidence changes in the joint result are in `overlap_pairs.csv`.",
            "Raw overlap and non-overlap values are also saved in `overlap_roi_raw_scores.csv`.",
            "Pairs that overlap only when DENT and RUST are each run separately are in `class_only_overlap_pairs.csv`; `*_present_in_all=false` means the candidate did not match a joint-result mask.",
            "Class-only candidates absent from the joint result, with their highest other-class box overlap, are in `class_only_unmatched.csv`.",
            "Pink pixels in each overlay show the union intersection in the joint result.",
            "",
        ]
    )
    (args.output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote analysis to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
