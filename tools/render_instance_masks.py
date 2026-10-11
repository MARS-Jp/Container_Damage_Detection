"""Render YOLO segmentation as an original image beside colored masks on black."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true', help='Skip complete output sets')
    args = parser.parse_args()
    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    files = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in extensions) if source.is_dir() else [source]
    if not files:
        raise ValueError(f'No images found: {source}')
    model = YOLO(args.model)
    failures = []
    completed = skipped = 0
    for number, path in enumerate(files, 1):
        destination = output / path.relative_to(source).parent / path.name if source.is_dir() else output
        if args.resume and all((destination / name).is_file() for name in ('comparison.png', 'masks_black.png', 'predictions.json')):
            skipped += 1
            continue
        try:
            result = model.predict(source=str(path), conf=args.conf,
                                   imgsz=args.imgsz, retina_masks=True,
                                   device=args.device, save=False, verbose=False)[0]
            render_result(result, destination)
            completed += 1
        except Exception as error:
            failures.append({'source': str(path), 'error': str(error)})
            print(f'ERROR {path}: {error}', flush=True)
        if number % 25 == 0 or number == len(files):
            print(f'Progress {number}/{len(files)}; completed={completed}; skipped={skipped}; failed={len(failures)}', flush=True)
    summary = {'total': len(files), 'completed': completed, 'skipped': skipped, 'failures': failures,
               'model': args.model, 'source': args.source, 'conf': args.conf, 'imgsz': args.imgsz}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    if failures:
        raise SystemExit(1)


def render_result(result, output):
    output.mkdir(parents=True, exist_ok=True)
    original = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
    height, width = original.shape[:2]
    colored = np.zeros_like(original)
    palette = [(255, 50, 50), (50, 220, 90), (40, 160, 255),
               (255, 210, 40), (200, 60, 255), (255, 120, 30),
               (30, 230, 220), (255, 100, 180)]
    instances = []
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        # Paint less confident objects first so stronger predictions win overlaps.
        for index in np.argsort(result.boxes.conf.cpu().numpy()):
            mask = masks[index]
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            color = palette[int(index) % len(palette)]
            colored[mask > 0.5] = color
            cls = int(result.boxes.cls[index])
            instances.append({'instance': int(index), 'class': result.names[cls],
                              'confidence': float(result.boxes.conf[index]),
                              'color_rgb': color, 'pixels': int((mask > 0.5).sum())})
    border = 8
    comparison = Image.new('RGB', (width * 2 + border * 3, height + border * 2), 'white')
    comparison.paste(Image.fromarray(original), (border, border))
    comparison.paste(Image.fromarray(colored), (width + border * 2, border))
    comparison.save(output / 'comparison.png')
    Image.fromarray(colored).save(output / 'masks_black.png')
    (output / 'predictions.json').write_text(json.dumps(instances, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
