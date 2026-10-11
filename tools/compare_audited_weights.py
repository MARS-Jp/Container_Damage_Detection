"""Evaluate one saved multi-weight prediction run and render fill-only comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.audit_shadow_training import digest, save_csv, save_json
from tools.evaluate_shadow_annotations import REGIONS, evaluate, read_annotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--xml', type=Path, default='C:/Users/USER/Desktop/0928-1/CBHU0708054-A.xml')
    args = parser.parse_args()
    run = args.run.resolve()
    annotation = read_annotation(args.xml)
    settings = json.loads((run / 'settings.json').read_text(encoding='utf-8'))
    reports = sorted(run.glob('*/*/predictions.json'))
    if not reports:
        raise ValueError('No multi-weight prediction reports found')
    output = run / 'comparison'
    output.mkdir(exist_ok=True)
    rows, missed, images, identities = [], [], [], []
    source = None
    for path in reports:
        report = json.loads(path.read_text(encoding='utf-8'))
        current_source = Path(report['source']).resolve()
        if source is not None and source != current_source:
            raise ValueError('This comparison requires one identical source for every weight')
        source = current_source
        with Image.open(source) as photo:
            if photo.size != (annotation['width'], annotation['height']) or source.name.casefold() != annotation['filename'].casefold():
                raise ValueError('XML and source mismatch')
        model_folder = path.parent.parent.name
        index = int(model_folder.split('_', 1)[0]) - 1
        weight = Path(settings['weights'][index])
        identity = {'model': weight.stem, 'weight': str(weight), 'weight_sha256': digest(weight)}
        identities.append(identity)
        for iou in (.3, .5):
            scores, misses = evaluate(annotation, report, iou)
            rows.extend({'model': weight.stem, 'box_iou_threshold': iou, **r} for r in scores)
            missed.extend({'model': weight.stem, 'box_iou_threshold': iou, **r} for r in misses)
        images.append((weight.stem, next(path.parent.glob('*__all.jpg'))))
    save_csv(output / 'metrics.csv', rows)
    save_csv(output / 'missed_annotations.csv', missed)
    save_json(output / 'evaluation.json', {'xml': str(args.xml.resolve()), 'xml_sha256': digest(args.xml),
        'models': identities, 'source': str(source), 'source_sha256': digest(source),
        'settings': settings, 'metrics': rows,
        'scope': 'Single-image same-class one-to-one bounding-box matching; not mask IoU or independent validation.'})
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 22)
    panels = [('Original', source), *images]
    width, height = annotation['width'], annotation['height']
    canvas = Image.new('RGB', (400*len(panels), 350*3), '#f2f4f6')
    draw = ImageDraw.Draw(canvas)
    for row_index, (region, roi) in enumerate({'full': (0, 0, 1, 1), **REGIONS}.items()):
        bounds = tuple(round(value * (width if index % 2 == 0 else height)) for index,value in enumerate(roi))
        for column, (name, path) in enumerate(panels):
            with Image.open(path) as photo:
                crop = photo.convert('RGB').crop(bounds)
                crop.thumbnail((390, 300), Image.Resampling.LANCZOS)
                x, y = column*400, row_index*350
                draw.text((x+8, y+5), f'{name} | {region}', font=font, fill='#152937')
                canvas.paste(crop, (x+(400-crop.width)//2, y+40))
    canvas.save(output / 'models_comparison.png')
    print(json.dumps([r for r in rows if r['box_iou_threshold'] == .5 and r['class'] in ('D', 'R')], ensure_ascii=False), flush=True)
    print(f'COMPLETE={output}')


if __name__ == '__main__':
    main()
