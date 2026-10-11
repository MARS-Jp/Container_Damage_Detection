"""Draw YOLO bounding-box labels on images for visual inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

DEFAULT_IMAGE_DIR = Path(r"C:\Users\sile7\Downloads\dataset0621\dataset0608\dataset0608\dataset\images\train\B")
DEFAULT_LABEL_DIR = Path(r"C:\Users\sile7\Downloads\dataset0621\dataset0608\dataset0608\dataset\labels\train_bbox\B")
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\sile7\Downloads\dataset0621\dataset0608\dataset0608\dataset\visualized\train_bbox\B"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COLORS = [
    (0, 255, 0),
    (255, 128, 0),
    (0, 165, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 0, 255),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="將 YOLO bbox 標籤畫到圖片上。")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR, help="圖片資料夾")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABEL_DIR, help="YOLO 標籤資料夾")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="標註圖片輸出資料夾")
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help='可選的類別名稱，例如：--names "person" "car"',
    )
    parser.add_argument("--show", action="store_true", help="逐張顯示；按任意鍵下一張，按 q 結束")
    return parser.parse_args()


def draw_labels(image, label_path: Path, names: list[str] | None):
    height, width = image.shape[:2]
    if not label_path.exists():
        return image, 0

    count = 0
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        fields = line.strip().split()
        if not fields:
            continue
        if len(fields) < 5:
            print(f"警告：略過格式錯誤的標籤 {label_path}:{line_number}")
            continue
        try:
            class_id = int(float(fields[0]))
            x_center, y_center, box_width, box_height = map(float, fields[1:5])
        except ValueError:
            print(f"警告：略過無法解析的標籤 {label_path}:{line_number}")
            continue

        x1 = max(0, min(width - 1, round((x_center - box_width / 2) * width)))
        y1 = max(0, min(height - 1, round((y_center - box_height / 2) * height)))
        x2 = max(0, min(width - 1, round((x_center + box_width / 2) * width)))
        y2 = max(0, min(height - 1, round((y_center + box_height / 2) * height)))
        color = COLORS[class_id % len(COLORS)]
        class_name = names[class_id] if names and 0 <= class_id < len(names) else f"class {class_id}"
        caption = class_name
        if len(fields) >= 6:
            caption += f" {fields[5]}"

        thickness = max(2, round(min(width, height) / 500))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        (text_width, text_height), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y = max(text_height + baseline, y1)
        cv2.rectangle(image, (x1, text_y - text_height - baseline), (x1 + text_width, text_y), color, -1)
        cv2.putText(image, caption, (x1, text_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        count += 1
    return image, count


def main() -> None:
    args = parse_args()
    if not args.images.is_dir():
        raise FileNotFoundError(f"找不到圖片資料夾：{args.images}")
    if not args.labels.is_dir():
        raise FileNotFoundError(f"找不到標籤資料夾：{args.labels}")

    image_paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise FileNotFoundError(f"圖片資料夾中沒有支援的圖片：{args.images}")
    args.output.mkdir(parents=True, exist_ok=True)

    saved = total_boxes = missing_labels = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"警告：無法讀取圖片，略過：{image_path}")
            continue
        label_path = args.labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
        image, box_count = draw_labels(image, label_path, args.names)
        total_boxes += box_count
        output_path = args.output / image_path.name
        if not cv2.imwrite(str(output_path), image):
            print(f"警告：無法寫入：{output_path}")
            continue
        saved += 1

        if args.show:
            cv2.imshow("YOLO labels (q: quit)", image)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print(f"完成：輸出 {saved} 張圖片、繪製 {total_boxes} 個框。")
    print(f"輸出位置：{args.output}")
    if missing_labels:
        print(f"注意：有 {missing_labels} 張圖片找不到同名 .txt 標籤（仍輸出原圖）。")


if __name__ == "__main__":
    main()
