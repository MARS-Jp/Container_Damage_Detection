r"""Browse inference results with A (previous), D (next), and Esc/Q (quit).

Run from the repository:
    C:\Users\USER\anaconda3\python.exe tool/picture/view_inference_images.py

The default folder is the requested last0912 run. Only *__all.jpg result images
are shown by default; --all-images includes per-class images and saved inputs.
Use --root to choose another folder, --pattern for a different filename filter,
or --start 300 to begin at the 300th image. Image files are never modified.

Requires Tkinter and Pillow. The base Anaconda Python above has both installed.
"""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk


# 可直接修改這裡，選擇要查看的推論結果資料夾。
DEFAULT_RESULTS_DIR = Path(
    r"C:\Users\USER\Documents\iim_0905\Yolov11_AttnRes\tools\runs\segment_by_class"
    r"\shadow_diagnostics\run_20261002_232129_326448\01_last0912"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def natural_key(path: Path):
    """Keep folder 2 before folder 10, including runs with >1,000 images."""
    return [(0, int(part)) if part.isdigit() else (1, part.casefold())
            for part in re.split(r"(\d+)", path.as_posix())]


def collect_images(folder: Path, pattern="*__all.jpg", all_images=False):
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"找不到資料夾：{folder}")
    paths = [path for path in folder.rglob("*") if path.is_file()
             and path.suffix.casefold() in IMAGE_SUFFIXES
             and (all_images or fnmatch.fnmatchcase(path.name.casefold(), pattern.casefold()))]
    paths.sort(key=lambda path: natural_key(path.relative_to(folder)))
    if not paths:
        selected = "支援的圖片" if all_images else pattern
        raise ValueError(f"資料夾內找不到 {selected}：{folder}")
    return paths


class ImageViewer:
    def __init__(self, window, folder, pattern="*__all.jpg", all_images=False, start=1):
        self.window = window
        self.pattern = pattern
        self.all_images = all_images
        self.original = None
        self.photo = None
        self.resize_job = None
        self.fullscreen = False
        window.title("推論圖片檢視器")
        window.geometry("1280x900")
        window.minsize(640, 480)
        window.option_add("*Font", ("Microsoft JhengHei UI", 10))

        toolbar = ttk.Frame(window, padding=(10, 8))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="上一張 (A)", command=lambda: self.move(-1)).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="下一張 (D)", command=lambda: self.move(1)).pack(side="left", padx=(0, 12))
        ttk.Button(toolbar, text="選擇資料夾", command=self.choose_folder).pack(side="left")
        self.counter = tk.StringVar()
        ttk.Label(toolbar, textvariable=self.counter).pack(side="right")

        self.filename = tk.StringVar()
        self.path_label = ttk.Label(window, textvariable=self.filename, padding=(10, 4), anchor="w", wraplength=1200)
        self.path_label.pack(fill="x")
        self.canvas = tk.Canvas(window, background="#15191f", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.image_item = self.canvas.create_image(0, 0, anchor="center")
        self.status = tk.StringVar(value="A 上一張｜D 下一張（循環）｜← / → 也可切換｜F11 全螢幕｜Esc / Q 離開")
        ttk.Label(window, textvariable=self.status, padding=(10, 6), anchor="w").pack(fill="x")
        self.canvas.bind("<Configure>", self.schedule_resize)
        window.bind("<KeyPress>", self.on_key)
        self.set_folder(folder, start)

    def set_folder(self, folder, start=1):
        folder = Path(folder).expanduser().resolve()
        paths = collect_images(folder, self.pattern, self.all_images)
        if not 1 <= start <= len(paths):
            raise ValueError(f"起始張數需介於 1 和 {len(paths)}，收到 {start}")
        self.folder, self.paths, self.index = folder, paths, start - 1
        self.show_image()

    def choose_folder(self):
        selected = filedialog.askdirectory(parent=self.window, title="選擇推論結果資料夾", initialdir=str(self.folder))
        if selected:
            try:
                self.set_folder(Path(selected))
            except (OSError, ValueError) as error:
                messagebox.showerror("無法開啟資料夾", str(error), parent=self.window)

    def move(self, offset):
        self.index = (self.index + offset) % len(self.paths)
        self.show_image()

    def on_key(self, event):
        key = event.keysym.lower()
        if key in {"d", "right"}:
            self.move(1)
        elif key in {"a", "left"}:
            self.move(-1)
        elif key in {"escape", "q"}:
            self.window.destroy()
        elif key == "f11":
            self.fullscreen = not self.fullscreen
            self.window.attributes("-fullscreen", self.fullscreen)

    def show_image(self):
        if self.resize_job is not None:
            self.window.after_cancel(self.resize_job)
            self.resize_job = None
        path = self.paths[self.index]
        self.counter.set(f"{self.index + 1} / {len(self.paths)}")
        self.filename.set(str(path.relative_to(self.folder)))
        self.window.title(f"[{self.index + 1}/{len(self.paths)}] {path.name} — 推論圖片檢視器")
        self.canvas.delete("error")
        try:
            with Image.open(path) as source:
                self.original = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, ValueError) as error:
            self.original = None
            self.canvas.itemconfigure(self.image_item, image="")
            self.photo = None
            self.canvas.create_text(20, 20, anchor="nw", fill="#f6b4b4", width=600, tags="error",
                                    text=f"無法讀取：{path.name}\n{error}\n\n可按 D / A 切換其他圖片。")
            self.status.set("圖片讀取失敗｜D 下一張｜A 上一張｜Esc / Q 離開")
            return
        self.render_image()

    def schedule_resize(self, event):
        self.path_label.configure(wraplength=max(200, self.window.winfo_width() - 20))
        if self.resize_job is not None:
            self.window.after_cancel(self.resize_job)
        self.resize_job = self.window.after(60, self.render_image)

    def render_image(self):
        self.resize_job = None
        if self.original is None:
            return
        width, height = max(1, self.canvas.winfo_width() - 20), max(1, self.canvas.winfo_height() - 20)
        scale = min(width / self.original.width, height / self.original.height)
        size = (max(1, round(self.original.width * scale)), max(1, round(self.original.height * scale)))
        resized = self.original.resize(size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized, master=self.window)
        self.canvas.itemconfigure(self.image_item, image=self.photo)
        self.canvas.coords(self.image_item, self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)
        self.status.set(f"原圖 {self.original.width} × {self.original.height}｜顯示 {scale:.0%}｜"
                        "A 上一張｜D 下一張（循環）｜← / → 切換｜F11 全螢幕｜Esc / Q 離開")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_RESULTS_DIR, help="推論結果資料夾；遞迴搜尋子資料夾")
    parser.add_argument("--pattern", default="*__all.jpg", help="圖片檔名篩選，例如 *__class_002_D.jpg")
    parser.add_argument("--all-images", action="store_true", help="顯示所有影像，包括逐類結果及 inputs 原圖")
    parser.add_argument("--start", type=int, default=1, help="從第幾張開始，預設第 1 張")
    args = parser.parse_args(argv)
    window = tk.Tk()
    window.withdraw()
    try:
        ImageViewer(window, args.root, args.pattern, args.all_images, args.start)
    except (OSError, ValueError) as error:
        messagebox.showerror("無法開啟推論圖片", str(error), parent=window)
        window.destroy()
        return 1
    window.deiconify()
    window.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
