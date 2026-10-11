"""將多個模型對同一張圖片的推論結果排版後並列比較。

一般使用時只需修改下方「使用者設定區」中的 IMAGE_NAME 與
COMPARE_FOLDERS，然後執行：

    python tools/compare_inference_images.py

程式會遞迴搜尋每個比較資料夾，因此圖片放在編號子資料夾內也能找到。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


# ============================ 使用者設定區 ============================
# 要比較的圖片檔名（包含副檔名）。
IMAGE_NAME = "CBHU0708054-A__all.jpg"

#IMAGE_NAME = "IMAG0749__all.jpg"

# 所有模型推論結果的共同根目錄。
RESULT_ROOT = Path(r"C:\Users\USER\Documents\推論結果")

# 要比較的資料夾，數量不限。路徑相對於 RESULT_ROOT；順序會對應到各顯示位置。
COMPARE_FOLDERS = [
    #Path(r"0905-10csar\predict-2"),
    Path(r"0926推論結果\predict-2"),
    Path(r"0929-1推論結果\predict-2"),
]

# 是否在建立完成後用系統圖片檢視器顯示比較圖。
SHOW_WINDOW = True

# 每一格的最大尺寸。原圖比例不同時會置中，不會拉伸變形。
MAX_CELL_WIDTH = 960
MAX_CELL_HEIGHT = 720

# 合成圖存放處。
OUTPUT_DIR = Path(__file__).resolve().parent / "runs" / "image_comparisons"
# =====================================================================


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
BACKGROUND_COLOR = (28, 28, 28)
LABEL_BACKGROUND = (0, 0, 0, 190)
LABEL_COLOR = (255, 255, 255, 255)


def parse_args() -> argparse.Namespace:
    """提供少量命令列覆寫；直接修改上方變數仍是主要使用方式。"""
    parser = argparse.ArgumentParser(description="並列比較多個資料夾內的同名推論圖片。")
    parser.add_argument("--no-show", action="store_true", help="只儲存合成圖，不開啟顯示視窗")
    parser.add_argument("--output", type=Path, default=None, help="自訂合成圖輸出路徑")
    return parser.parse_args()


def layout_for(image_count: int) -> tuple[int, int]:
    """回傳 (列數, 欄數)，2 張橫排、3 張三等份、4 張 2x2，其餘近似方形。"""
    if image_count <= 0:
        raise ValueError("至少需要一個比較資料夾。")
    if image_count <= 3:
        return 1, image_count
    if image_count == 4:
        return 2, 2

    columns = math.ceil(math.sqrt(image_count))
    rows = math.ceil(image_count / columns)
    return rows, columns


def grid_position(index: int, image_count: int, columns: int) -> tuple[int, int]:
    """取得陣列項目的格子座標；4 張依左上、左下、右上、右下排列。"""
    if image_count == 4:
        return index % 2, index // 2
    return divmod(index, columns)


def find_image(folder: Path, image_name: str) -> Path:
    """先找資料夾根層，再遞迴尋找唯一一張同名圖片。"""
    if not folder.is_dir():
        raise FileNotFoundError(f"找不到比較資料夾：{folder}")

    direct_path = folder / image_name
    if direct_path.is_file():
        return direct_path

    matches = sorted(path for path in folder.rglob(image_name) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"在資料夾中找不到 {image_name}：{folder}")
    if len(matches) > 1:
        match_list = "\n".join(f"  - {path}" for path in matches)
        raise RuntimeError(
            f"資料夾內有 {len(matches)} 張同名圖片，無法判斷要使用哪一張：{folder}\n"
            f"請將 COMPARE_FOLDERS 的路徑寫得更精確。\n{match_list}"
        )
    return matches[0]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """優先使用 Windows 繁中文字型，其他系統則使用常見的 DejaVu Sans。"""
    font_candidates = (
        Path(r"C:\Windows\Fonts\msjhbd.ttc"),
        Path(r"C:\Windows\Fonts\msjh.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    )
    for font_path in font_candidates:
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def fit_label(
    label: str, draw: ImageDraw.ImageDraw, max_width: int, start_size: int
) -> tuple[str, ImageFont.ImageFont]:
    """縮小或截短過長標籤，確保左上角的資料夾名稱不超出圖片。"""
    horizontal_padding = 24
    available_width = max(1, max_width - horizontal_padding * 2)

    for font_size in range(start_size, 15, -2):
        font = load_font(font_size)
        if draw.textbbox((0, 0), label, font=font)[2] <= available_width:
            return label, font

    font = load_font(16)
    shortened = label
    while shortened and draw.textbbox((0, 0), f"{shortened}…", font=font)[2] > available_width:
        shortened = shortened[:-1]
    return f"{shortened}…", font


def add_folder_label(image: Image.Image, label: str) -> Image.Image:
    """在圖片左上角繪製半透明的資料夾名稱。"""
    labeled = image.convert("RGBA")
    overlay = Image.new("RGBA", labeled.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_size = max(22, min(42, round(min(image.size) * 0.055)))
    fitted_label, font = fit_label(label, draw, image.width, font_size)
    left, top, right, bottom = draw.textbbox((0, 0), fitted_label, font=font)
    text_width = right - left
    text_height = bottom - top
    padding_x = max(12, font_size // 2)
    padding_y = max(8, font_size // 4)
    box_width = text_width + padding_x * 2
    box_height = text_height + padding_y * 2

    draw.rounded_rectangle((0, 0, box_width, box_height), radius=8, fill=LABEL_BACKGROUND)
    draw.text((padding_x, padding_y - top), fitted_label, font=font, fill=LABEL_COLOR)
    return Image.alpha_composite(labeled, overlay).convert("RGB")


def open_rgb(path: Path) -> Image.Image:
    """讀圖並套用 EXIF 方向資訊，避免手機照片旋轉錯誤。"""
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def build_comparison(
    folder_entries: list[Path], image_name: str
) -> tuple[Image.Image, list[Path]]:
    """載入各資料夾的同名圖片並建立等分比較圖。"""
    if not folder_entries:
        raise ValueError("COMPARE_FOLDERS 至少要填入一個資料夾。")
    if Path(image_name).suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"不支援的圖片副檔名：{image_name}（支援：{supported}）")

    resolved_entries: list[tuple[Path, Path]] = []
    for entry in folder_entries:
        folder = entry if entry.is_absolute() else RESULT_ROOT / entry
        resolved_entries.append((entry, find_image(folder, image_name)))

    source_images = [open_rgb(image_path) for _, image_path in resolved_entries]
    resampling = getattr(Image, "Resampling", Image)
    resized_images = [
        ImageOps.contain(image, (MAX_CELL_WIDTH, MAX_CELL_HEIGHT), resampling.LANCZOS)
        for image in source_images
    ]
    # 依縮放後的實際尺寸決定格子大小，避免直式圖片左右產生大量空白。
    cell_width = max(image.width for image in resized_images)
    cell_height = max(image.height for image in resized_images)
    rows, columns = layout_for(len(source_images))
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), BACKGROUND_COLOR)

    for index, ((folder_entry, _), resized) in enumerate(zip(resolved_entries, resized_images)):
        # 相對路徑只顯示最外層的模型資料夾名稱，不顯示 predict-2 等子資料夾。
        display_name = folder_entry.parts[0] if not folder_entry.is_absolute() else folder_entry.name
        labeled = add_folder_label(resized, display_name)
        row, column = grid_position(index, len(source_images), columns)
        x = column * cell_width + (cell_width - labeled.width) // 2
        y = row * cell_height + (cell_height - labeled.height) // 2
        canvas.paste(labeled, (x, y))

    return canvas, [image_path for _, image_path in resolved_entries]


def show_comparison(comparison: Image.Image) -> None:
    """使用作業系統預設的圖片檢視器顯示結果。"""
    comparison.show(title=f"Inference comparison - {IMAGE_NAME}")


def main() -> None:
    args = parse_args()
    comparison, image_paths = build_comparison(COMPARE_FOLDERS, IMAGE_NAME)

    output_path = args.output or OUTPUT_DIR / f"{Path(IMAGE_NAME).stem}_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(output_path)

    print(f"完成：已比較 {len(image_paths)} 張同名推論結果。")
    for image_path in image_paths:
        print(f"  - {image_path}")
    print(f"合成圖：{output_path.resolve()}")

    if SHOW_WINDOW and not args.no_show:
        show_comparison(comparison)


if __name__ == "__main__":
    main()
