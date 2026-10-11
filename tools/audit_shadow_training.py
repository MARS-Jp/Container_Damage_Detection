"""Read-only audit of local segmentation labels and checkpoint training history.

XML boxes are review references only. This tool neither creates training labels
from boxes nor changes datasets, weights, inference defaults, or training state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import runpy
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.evaluate_shadow_annotations import REGIONS, box_iou, in_roi, read_annotation


def digest(path: Path) -> str:
    with path.open("rb") as file:
        sha = hashlib.sha256()
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def save_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_polygons(text: str, names: dict) -> tuple[list[dict], list[dict]]:
    polygons, issues, seen = [], [], set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields, issue = line.split(), None
        try:
            values = np.asarray(fields, dtype=np.float64)
            if len(values) == 5:
                issue = "box_row_not_segmentation"
            elif len(values) < 7 or len(values) % 2 == 0:
                issue = "invalid_polygon_columns"
            elif not np.isfinite(values).all():
                issue = "non_finite"
            elif values[0] != int(values[0]) or int(values[0]) not in names:
                issue = "unknown_class"
            elif np.any((values[1:] < 0) | (values[1:] > 1)):
                issue = "outside_image"
            else:
                points = values[1:].reshape(-1, 2)
                if len(np.unique(points, axis=0)) < 3 or cv2.contourArea(points.astype(np.float32)) <= 1e-12:
                    issue = "degenerate_polygon"
                else:
                    class_id = int(values[0])
                    box = np.r_[points.min(0), points.max(0)].tolist()
                    polygons.append(
                        {
                            "line": line_number,
                            "class_id": class_id,
                            "class_name": names[class_id],
                            "points": points.tolist(),
                            "box_normalized": box,
                        }
                    )
                    # The current loader de-duplicates labels by (class, bbox),
                    # even when the actual polygon contours differ.
                    key = (class_id, *box)
                    if key in seen:
                        issue = "duplicate_class_bbox"
                    seen.add(key)
        except ValueError:
            issue = "non_numeric"
        if issue:
            issues.append({"line": line_number, "issue": issue})
    return polygons, issues


def rasterize(polygons: list[dict], width: int, height: int) -> dict[str, np.ndarray]:
    masks = {}
    for polygon in polygons:
        name = polygon["class_name"]
        mask = masks.setdefault(name, np.zeros((height, width), dtype=np.uint8))
        points = np.rint(np.asarray(polygon["points"]) * [width, height]).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
    return masks


def compare_annotations(annotation: dict, polygons: list[dict]) -> list[dict]:
    width, height = annotation["width"], annotation["height"]
    masks = rasterize(polygons, width, height)
    rows = []
    for obj in annotation["objects"]:
        box, name = obj["box_xyxy"], obj["class_name"]
        candidates = [
            (box_iou(box, np.asarray(old["box_normalized"]) * [width, height, width, height]), old["line"])
            for old in polygons
            if old["class_name"] == name
        ]
        iou, line = max(candidates, default=(0, None), key=lambda pair: pair[0])
        iou = float(iou)
        x1, y1 = np.floor(box[:2]).astype(int)
        x2, y2 = np.ceil(box[2:]).astype(int)
        pixels = int(masks[name][y1:y2, x1:x2].sum()) if name in masks else 0
        region = next((key for key, roi in REGIONS.items() if in_roi(box, roi, width, height)), "other")
        rows.append(
            {
                "xml_id": obj["id"],
                "class": name,
                "region": region,
                "xmin_0based": box[0],
                "ymin_0based": box[1],
                "xmax": box[2],
                "ymax": box[3],
                "max_same_class_bbox_iou": round(iou, 6),
                "best_old_label_line": line if iou > 0 else "",
                "old_mask_pixels_in_xml_box": pixels,
                "old_mask_fraction_of_xml_box": round(pixels / ((x2 - x1) * (y2 - y1)), 6),
                "review_reason": "no_same_class_bbox_overlap"
                if iou == 0
                else "no_same_class_mask_pixels"
                if pixels == 0
                else "review_extent_and_instance_split",
            }
        )
    return rows


def audit_dataset(dataset: Path, names: dict, output: Path) -> dict:
    rows, problems, class_counts = [], [], Counter()
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    by_hash, by_group = defaultdict(list), defaultdict(list)
    labels_by_image = {}
    image_keys = set()
    for image in sorted((dataset / "images").rglob("*")):
        if not image.is_file() or image.suffix.lower() not in extensions:
            continue
        relative = image.relative_to(dataset / "images")
        image_keys.add(str(relative.with_suffix("")).casefold())
        label = dataset / "labels" / relative.with_suffix(".txt")
        polygons, issues = parse_polygons(label.read_text(encoding="utf-8-sig"), names) if label.exists() else ([], [])
        if not label.exists():
            issues.append({"line": 0, "issue": "missing_label"})
        elif not label.read_text(encoding="utf-8-sig").strip():
            issues.append({"line": 0, "issue": "empty_label_review_background"})
        class_counts.update(p["class_name"] for p in polygons)
        for issue in issues:
            problems.append({"image": str(image), "label": str(label), **issue})
        group = re.sub(r"-[A-Za-z]$", "", image.stem).upper()
        hash_value = digest(image)
        by_hash[hash_value].append(str(relative))
        by_group[group].append(str(relative))
        labels_by_image[str(relative)] = polygons
        rows.append(
            {
                "image": str(image),
                "label": str(label),
                "split_directory": relative.parts[0],
                "container_name_group": group,
                "image_sha256": hash_value,
                "label_sha256": digest(label) if label.exists() else "",
                "polygon_count": len(polygons),
                "issues": ";".join(x["issue"] for x in issues),
            }
        )
    orphans = [
        str(p)
        for p in sorted((dataset / "labels").rglob("*.txt"))
        if str(p.relative_to(dataset / "labels").with_suffix("")).casefold() not in image_keys
    ]
    save_csv(output / "dataset_inventory.csv", rows)
    save_csv(output / "dataset_issues.csv", problems)
    conflicts = []
    for members in by_hash.values():
        if len(members) < 2:
            continue
        image = cv2.imdecode(np.fromfile(dataset / "images" / members[0], np.uint8), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        masks = [rasterize(labels_by_image[m], width, height) for m in members]
        for index in range(1, len(members)):
            classes = sorted(masks[0].keys() | masks[index].keys())
            differences = {
                name: int(
                    np.count_nonzero(
                        masks[0].get(name, np.zeros((height, width), np.uint8))
                        != masks[index].get(name, np.zeros((height, width), np.uint8))
                    )
                )
                for name in classes
            }
            if any(differences.values()):
                conflicts.append(
                    {
                        "reference_image": members[0],
                        "identical_image": members[index],
                        "reference_classes": ",".join(sorted(masks[0])),
                        "other_classes": ",".join(sorted(masks[index])),
                        "class_mask_difference_pixels": json.dumps(differences),
                    }
                )
    save_csv(output / "duplicate_label_conflicts.csv", conflicts)
    summary = {
        "root": str(dataset),
        "image_count": len(rows),
        "images_by_directory_split": dict(Counter(r["split_directory"] for r in rows)),
        "valid_polygon_counts": dict(class_counts),
        "issue_counts": dict(Counter(p["issue"] for p in problems)),
        "orphan_label_files": orphans,
        "filename_container_groups": len(by_group),
        "byte_identical_image_groups": [v for v in by_hash.values() if len(v) > 1],
        "byte_identical_different_mask_pairs": conflicts,
        "cross_directory_container_groups": {
            k: v for k, v in by_group.items() if len({Path(p).parts[0] for p in v}) > 1
        },
        "scope": "Local image/label syntax and byte identity only; excludes perceptual duplicates, "
        "visual completeness, and unconfirmed Linux training data.",
    }
    save_json(output / "dataset_audit.json", summary)
    return summary


def audit_checkpoint(path: Path, output: Path) -> dict:
    import torch

    from ultralytics import YOLO

    model = YOLO(str(path))
    ckpt, head = model.ckpt, model.model.model[-1]
    history = ckpt.get("train_results") or {}
    save_json(output / "training_history.json", history)
    if history:
        save_csv(output / "training_history.csv", [dict(zip(history, values)) for values in zip(*history.values())])
    metrics = {}
    for key, values in history.items():
        if key.startswith("metrics/") and values:
            best = int(np.nanargmax(values))
            metrics[key] = {
                "first": values[0],
                "last": values[-1],
                "peak": values[best],
                "peak_epoch": history["epoch"][best],
            }
    commit = (ckpt.get("git") or {}).get("commit")
    source_checks = []
    for name in (
        "ultralytics/nn/modules/head.py",
        "ultralytics/nn/modules/block.py",
        "ultralytics/nn/modules/iim.py",
        "ultralytics/models/yolo/segment/predict.py",
        "ultralytics/data/augment.py",
        "ultralytics/data/dataset.py",
        "ultralytics/data/utils.py",
    ):
        local = ROOT / name
        if not commit or not local.exists():
            continue
        prior = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=ROOT, capture_output=True)
        source_checks.append(
            {
                "file": name,
                "commit_available": prior.returncode == 0,
                "matches_committed_training_version": prior.returncode == 0
                and prior.stdout.replace(b"\r\n", b"\n") == local.read_bytes().replace(b"\r\n", b"\n"),
            }
        )
    floating = [p for p in model.model.state_dict().values() if torch.is_floating_point(p)]
    result = {
        "path": str(path),
        "sha256": digest(path),
        "task": model.task,
        "names": model.names,
        "checkpoint_date": ckpt.get("date"),
        "ultralytics_version": ckpt.get("version"),
        "git": ckpt.get("git"),
        "saved_epoch": ckpt.get("epoch"),
        "history_epochs": len(history.get("epoch", [])),
        "head": type(head).__name__,
        "head_strides": head.stride.tolist(),
        "model_yaml": model.model.yaml,
        "train_args": ckpt.get("train_args"),
        "stored_validation_metrics": ckpt.get("train_metrics"),
        "history_metrics": metrics,
        "source_checks": source_checks,
        "nonfinite_state_tensor_count": sum(not torch.isfinite(p).all().item() for p in floating),
        "provenance_limit": "The checkpoint records data.yaml but no resolved data paths or label-content hashes. "
        "Saved metrics are historical records, not a newly verified independent evaluation.",
    }
    save_json(output / "checkpoint_audit.json", result)
    if history:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        epochs = history["epoch"]
        for suffix, label in (("B", "Boxes"), ("M", "Masks")):
            axes[0].plot(epochs, np.asarray(history[f"metrics/mAP50({suffix})"]) * 100, label=label)
        axes[0].set(title="Stored validation mAP50", xlabel="Epoch", ylabel="mAP50 (%)")
        for key in ("train/seg_loss", "val/seg_loss"):
            axes[1].plot(epochs, history[key], label=key)
        axes[1].set(title="Stored segmentation loss", xlabel="Epoch", ylabel="Loss")
        for axis in axes:
            axis.legend()
            axis.grid(alpha=0.2)
        fig.savefig(output / "training_history.png", dpi=150)
        plt.close(fig)
    return result


def review_page(image: np.ndarray, annotation: dict, polygons: list[dict], rows: list[dict], output: Path):
    from PIL import Image, ImageDraw, ImageFont

    from tools.predict_seg_by_class import class_color

    crop_dir = output / "review_crops"
    crop_dir.mkdir(exist_ok=True)
    masks = rasterize(polygons, annotation["width"], annotation["height"])
    ids = {p["class_name"]: p["class_id"] for p in polygons}
    ordered = sorted(
        rows,
        key=lambda r: (
            {"right_shadow": 0, "left_MSC": 1, "other": 2}[r["region"]],
            r["old_mask_pixels_in_xml_box"] > 0,
            r["xml_id"],
        ),
    )
    cards, preview = [], []
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    for row in ordered:
        name, number = row["class"], row["xml_id"]
        x1, y1, x2, y2 = (row[k] for k in ("xmin_0based", "ymin_0based", "xmax", "ymax"))
        bounds = (
            max(0, int(x1) - 22),
            max(0, int(y1) - 22),
            min(image.shape[1], int(x2) + 22),
            min(image.shape[0], int(y2) + 22),
        )
        left, top, right, bottom = bounds
        original = image[top:bottom, left:right].copy()
        filled = original.copy()
        if name in masks:
            selected = masks[name][top:bottom, left:right].astype(bool)
            filled[selected] = np.rint(filled[selected] * 0.65 + np.asarray(class_color(ids[name])) * 0.35).astype(
                np.uint8
            )
        for label, crop in (("original", original), ("old_label", filled)):
            cv2.imencode(".png", crop)[1].tofile(crop_dir / f"{number:03d}_{label}.png")
        cards.append(
            f'<article data-region="{row["region"]}"><h3>#{number} {name} · {row["region"]}</h3>'
            f"<p>框交集 IoU {row['max_same_class_bbox_iou']:.3f}；框內舊同類遮罩 {row['old_mask_pixels_in_xml_box']} 像素</p>"
            '<div class="pair">'
            + "".join(
                f'<figure><img loading="lazy" src="review_crops/{number:03d}_{label}.png"><figcaption>{caption}</figcaption></figure>'
                for label, caption in [("original", "原圖局部"), ("old_label", "舊分割標註填色")]
            )
            + "</div></article>"
        )
        if number in (84, 85, 30, 86, 62, 64):
            panel = Image.new("RGB", (660, 290), "white")
            draw = ImageDraw.Draw(panel)
            draw.text(
                (10, 8), f"XML #{number} {name}: original | existing label (not prediction)", font=font, fill="black"
            )
            for offset, crop in ((0, original), (330, filled)):
                thumb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                scale = min(310 / thumb.width, 250 / thumb.height)
                thumb = thumb.resize(
                    (round(thumb.width * scale), round(thumb.height * scale)), Image.Resampling.NEAREST
                )
                panel.paste(thumb, (offset + (330 - thumb.width) // 2, 30))
            preview.append(panel)
    page = """<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>訓練標註核對</title>
<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;background:#f3f5f7;color:#18252f}
article{background:white;padding:20px;margin:18px 0;border-radius:12px}.pair{display:flex;gap:20px}figure{margin:0;width:48%}
img{width:100%;height:280px;object-fit:contain;image-rendering:auto}figcaption{margin-top:10px;color:#52616d}
select{font:inherit;padding:8px}h3{margin-top:0}</style>
<h1>補標核對頁：CBHU0708054-A</h1><p>新 XML 有框，舊 TXT 有多邊形。右欄僅顯示既有標註，並非模型預測，也沒有自動補畫。
每張局部圖包含 XML 框外 22 像素背景；照片內不畫框、不放文字。先檢查右下陰影，再檢查左側 MSC 凹陷。</p>
<p>框交集和框內舊遮罩像素用來找出需人工核對的項目，不能當成分割準確率；物件切分方式也需要統一。</p>
<label>區域 <select id="filter"><option value="all">全部 94 項</option><option>right_shadow</option><option>left_MSC</option><option>other</option></select></label>
"""
    # Keep filtering entirely local; the review page has no uploads or write operations.
    page += (
        "".join(cards)
        + """<script>document.querySelector('#filter').addEventListener('change',e=>
document.querySelectorAll('article').forEach(a=>a.hidden=e.target.value!=='all'&&a.dataset.region!==e.target.value));</script></html>"""
    )
    (output / "review.html").write_text(page, encoding="utf-8")
    if preview:
        montage = Image.new("RGB", (1320, ((len(preview) + 1) // 2) * 290), "#eef1f4")
        for index, panel in enumerate(preview):
            montage.paste(panel, ((index % 2) * 660, (index // 2) * 290))
        montage.save(output / "priority_review.png")


def main():
    base = runpy.run_path(str(ROOT / "tools/predict-2.py"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=base["WEIGHT_SOURCES"][0])
    parser.add_argument("--source", type=Path, default=base["DATASET_SOURCES"][0])
    parser.add_argument("--xml", type=Path, default="C:/Users/USER/Desktop/0928-1/CBHU0708054-A.xml")
    parser.add_argument("--old-xml", type=Path, default="C:/Users/USER/Desktop/已標記/labels/train/B/CBHU0708054-A.xml")
    parser.add_argument("--data", type=Path, default="C:/Users/USER/Desktop/碩論/dataset0608/dataset0608/data.yaml")
    parser.add_argument(
        "--dataset-root", type=Path, default="C:/Users/USER/Desktop/碩論/dataset0608/dataset0608/dataset"
    )
    parser.add_argument("--image-relative", type=Path, default="train/B/CBHU0708054-A.JPG")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/shadow_training_audit")
    args = parser.parse_args()
    output = args.output.resolve() / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    print(f"OUTPUT={output}", flush=True)
    config = yaml.safe_load(args.data.read_text(encoding="utf-8-sig"))
    names = (
        dict(enumerate(config["names"]))
        if isinstance(config["names"], list)
        else {int(k): v for k, v in config["names"].items()}
    )
    label = args.dataset_root / "labels" / args.image_relative.with_suffix(".txt")
    local_image = args.dataset_root / "images" / args.image_relative
    annotation = read_annotation(args.xml)
    old_xml = read_annotation(args.old_xml)
    image = cv2.imdecode(np.fromfile(args.source, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (annotation["height"], annotation["width"]):
        raise ValueError("Source image size does not match XML")
    if (
        annotation["filename"].casefold() != args.source.name.casefold()
        or args.image_relative.name.casefold() != args.source.name.casefold()
    ):
        raise ValueError("Source filename does not match XML / local label source")
    if digest(args.source) != digest(local_image):
        raise ValueError("Source and local dataset images differ; do not compare annotations across them")
    polygons, issues = parse_polygons(label.read_text(encoding="utf-8-sig"), names)
    if issues:
        raise ValueError(f"Target segmentation annotation has issues: {issues}")
    if old_xml["filename"].casefold() != annotation["filename"].casefold() or (old_xml["width"], old_xml["height"]) != (
        annotation["width"],
        annotation["height"],
    ):
        raise ValueError("Old XML does not match source metadata")
    snapshots = output / "sources"
    snapshots.mkdir()
    source_hashes = {}
    for key, path in {
        "new.xml": args.xml,
        "old.xml": args.old_xml,
        "old_segmentation.txt": label,
        "data.yaml": args.data,
    }.items():
        source_hashes[str(path.resolve())] = digest(path)
        shutil.copy2(path, snapshots / key)
    source_hashes[str(args.source.resolve())] = digest(args.source)
    source_hashes[str(local_image.resolve())] = digest(local_image)
    rows = compare_annotations(annotation, polygons)
    save_csv(output / "annotation_review.csv", rows)
    region_rows = []
    for region in ("full", "left_MSC", "right_shadow", "other"):
        for name in sorted({r["class"] for r in rows}):
            selected = [r for r in rows if r["class"] == name and (region == "full" or r["region"] == region)]
            region_rows.append(
                {
                    "region": region,
                    "class": name,
                    "new_xml_count": len(selected),
                    "no_same_class_bbox_overlap": sum(r["max_same_class_bbox_iou"] == 0 for r in selected),
                    "no_same_class_mask_pixels": sum(r["old_mask_pixels_in_xml_box"] == 0 for r in selected),
                }
            )
    save_csv(output / "region_review.csv", region_rows)
    dataset = audit_dataset(args.dataset_root.resolve(), names, output)
    print(f"DATASET_IMAGES={dataset['image_count']}; ISSUES={dataset['issue_counts']}", flush=True)
    checkpoint = audit_checkpoint(args.model.resolve(), output)
    checkpoint_names = {int(k): v for k, v in checkpoint["names"].items()}
    if checkpoint_names != names:
        raise ValueError("Class mapping mismatch between checkpoint and dataset: examine checkpoint_audit.json")
    summary = {
        "source_hashes": source_hashes,
        "model_sha256": checkpoint["sha256"],
        "local_data_yaml": config,
        "local_dataset_root_explicit_override": str(args.dataset_root.resolve()),
        "yaml_train_equals_val": config.get("train") == config.get("val"),
        "class_mapping_matches_checkpoint": checkpoint_names == names,
        "old_xml_counts": dict(Counter(x["class_name"] for x in old_xml["objects"])),
        "old_segmentation_counts": dict(Counter(x["class_name"] for x in polygons)),
        "new_xml_counts": dict(Counter(x["class_name"] for x in annotation["objects"])),
        "old_DR_overlap_pixels": int(
            np.count_nonzero(
                rasterize(polygons, annotation["width"], annotation["height"]).get("D", 0)
                & rasterize(polygons, annotation["width"], annotation["height"]).get("R", 0)
            )
        ),
        "regions": region_rows,
        "dataset": dataset,
        "limitations": [
            "Linux data.yaml has not been provided; this local dataset is not proven to be the training source.",
            "XML boxes are not segmentation ground truth; no new masks or training labels were generated.",
            "Mask pixel counts are rasterized old annotations inside XML boxes, not a mask IoU score.",
            "No inference or retraining is performed by this audit.",
        ],
    }
    save_json(output / "audit_summary.json", summary)
    review_page(image, annotation, polygons, rows, output)
    # Verify audit inputs remained byte-identical throughout the run.
    if any(digest(Path(path)) != old_hash for path, old_hash in source_hashes.items()):
        raise RuntimeError("An input changed during the audit; rerun with stable inputs")
    print(json.dumps({"counts": summary["new_xml_counts"], "regions": region_rows}, ensure_ascii=False), flush=True)
    print(f"COMPLETE={output}", flush=True)


if __name__ == "__main__":
    main()
