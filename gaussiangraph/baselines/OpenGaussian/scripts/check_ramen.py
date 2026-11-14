"""Fail early when the complete ramen baseline inputs/runtime are unavailable."""
import argparse
import importlib
import numpy as np
from PIL import Image
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--gt', type=Path, required=True)
parser.add_argument('--training-only', action='store_true', help='Defer optional rendering postprocess dependency check')
args = parser.parse_args()
errors = []
modules = ['torch', 'torchvision', 'numpy', 'scipy', 'plyfile', 'bitarray',
           'ashawkey_diff_gaussian_rasterization']
if not args.training_only:
    modules.append('pytorch3d.ops')
for name in modules:
    try:
        module = importlib.import_module(name)
        print(f'OK import {name}')
        if name == 'torch':
            if not module.cuda.is_available():
                errors.append('CUDA unavailable')
            else:
                print(module.cuda.get_device_name(0), module.__version__)
    except Exception as exc:
        errors.append(f'{name}: {exc}')

images = sorted(p for p in (args.data / 'images').glob('*')
                if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
if not images:
    errors.append(f'No images in {args.data / "images"}')
for image in images:
    for suffix in ('_s.npy', '_f.npy'):
        path = args.data / 'language_features' / (image.stem + suffix)
        if not path.is_file():
            errors.append(f'Missing {path}')
    seg_path = args.data / 'language_features' / (image.stem + '_s.npy')
    feat_path = args.data / 'language_features' / (image.stem + '_f.npy')
    if seg_path.is_file() and feat_path.is_file():
        seg = np.load(seg_path, mmap_mode='r')
        feat = np.load(feat_path, mmap_mode='r')
        with Image.open(image) as rgb:
            expected = (4, rgb.height, rgb.width)
        if seg.shape != expected or feat.ndim != 2 or feat.shape[1] != 512:
            errors.append(f'Unexpected mask/feature shape: {image.stem}: {seg.shape}, {feat.shape}')
        elif seg.min() < -1 or seg.max() >= feat.shape[0]:
            errors.append(f'Mask ID exceeds feature table: {image.stem}')
if not (args.data / 'sparse' / '0' / 'cameras.bin').is_file() and not (
        args.data / 'sparse' / '0' / 'cameras.txt').is_file():
    errors.append('Missing COLMAP sparse/0/cameras.bin or cameras.txt')
frames = ('00006', '00024', '00060', '00065', '00081', '00119', '00128')
for frame in frames:
    if not list((args.gt / f'frame_{frame}').glob('*.jpg')):
        errors.append(f'Missing GT masks for frame_{frame} in {args.gt}')
if not Path('assets/text_features.json').is_file():
    errors.append('Missing assets/text_features.json; extract assets/text_features.zip')
print(f'Images: {len(images)}; missing/runtime errors: {len(errors)}')
for error in errors[:20]:
    print('ERROR:', error)
raise SystemExit(bool(errors))
