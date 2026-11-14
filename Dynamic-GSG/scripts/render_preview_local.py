"""Render saved RGB and instance-colored Gaussians without a GUI."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizer
from utils.recon_helpers import setup_camera
from utils.slam_helpers import transform_to_frame, transformed_params2rendervar

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('result_dir', type=Path)
parser.add_argument('--frames', type=int, nargs='+', default=[0])
group = parser.add_mutually_exclusive_group()
group.add_argument('--highlight-ids', type=int, nargs='+')
group.add_argument('--query-result', type=Path, help='Highlight IDs from a saved query JSON')
parser.add_argument('--output', type=Path)
args = parser.parse_args()
highlight_ids = args.highlight_ids
if args.query_result:
    highlight_ids = [int(n.get('id', n.get('idx'))) for n in json.loads(args.query_result.read_text())['results']]
    if not highlight_ids:
        parser.error('Query returned no objects; nothing to highlight')
data = np.load(args.result_dir / 'params_with_idx.npz')
params = {k: torch.as_tensor(data[k], device='cuda', dtype=torch.float32) for k in ['means3D', 'rgb_colors', 'unnorm_rotations', 'logit_opacities', 'log_scales', 'cam_unnorm_rots', 'cam_trans']}
width, height = int(data['org_width']), int(data['org_height'])
cam = setup_camera(width, height, data['intrinsics'], data['w2c'])
ids = data['object_idx'].reshape(-1).astype(int)
colors = np.random.default_rng(0).uniform(.15, 1, (int(ids.max())+1, 3))
colors[0] = [.5, .5, .5]
if highlight_ids is not None:
    unknown = set(highlight_ids)-set(ids.tolist())
    if unknown:
        parser.error(f'Unknown instance IDs: {sorted(unknown)}')
    colors[:] = [.35, .35, .35]
    colors[highlight_ids] = [1., .12, .12]
instance_colors = torch.as_tensor(colors[ids], dtype=torch.float32, device='cuda')
fig, axes = plt.subplots(len(args.frames), 2, figsize=(12, len(args.frames)*3.6), squeeze=False)
with torch.no_grad():
    for row, frame in enumerate(args.frames):
        transformed = transform_to_frame(params, frame, False, False)
        variables = transformed_params2rendervar(params, transformed)
        rgb, _, _ = GaussianRasterizer(raster_settings=cam)(**variables)
        variables['colors_precomp'] = instance_colors
        instances, _, _ = GaussianRasterizer(raster_settings=cam)(**variables)
        for col, (render, title) in enumerate([(rgb, 'RGB Gaussian reconstruction'), (instances, 'Query matches (red)' if highlight_ids is not None else 'Instance IDs (gray = background)')]):
            axes[row, col].imshow(render.clamp(0, 1).permute(1, 2, 0).cpu())
            axes[row, col].set_title(f'Frame {frame}: {title}')
            axes[row, col].axis('off')
fig.tight_layout()
output = args.output or args.result_dir / ('retrieval_highlight.png' if highlight_ids is not None else 'reconstruction_instances.png')
fig.savefig(output, dpi=150)
print(output)
