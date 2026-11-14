"""Rasterize LangSplat annotations using its fillPoly and per-category union rules."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--labels', type=Path, required=True)
args = parser.parse_args()
total = 0
for source in sorted(args.labels.glob('frame_*.json')):
    data = json.loads(source.read_text())
    shape = (data['info']['height'], data['info']['width'])
    masks = {}
    for obj in data['objects']:
        label = obj['category']
        if Path(label).name != label:
            raise ValueError(f'Invalid category: {label}')
        mask = masks.setdefault(label, np.zeros(shape, dtype=np.uint8))
        cv2.fillPoly(mask, [np.asarray(obj['segmentation'], dtype=np.int32)], 255)
    output = args.labels / 'gt' / source.stem
    output.mkdir(parents=True, exist_ok=True)
    for label, mask in masks.items():
        if not cv2.imwrite(str(output / f'{label}.jpg'), mask):
            raise OSError(f'Cannot write {output / label}')
    total += len(masks)
print(f'Wrote {total} per-frame category masks')
