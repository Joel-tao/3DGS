"""Query saved instance CLIP features without a display or external LLM service."""
import argparse
import gzip
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault('HF_HOME', '/media/joe/Data/Dynamic-GSG/cache/huggingface')
import numpy as np
import open_clip
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('result_dir', type=Path)
parser.add_argument('query')
parser.add_argument('--top-k', type=int, default=5)
parser.add_argument('--output', type=Path, help='Also save ranked objects as JSON')
args = parser.parse_args()
if args.top_k < 1:
    parser.error('--top-k must be positive')
with gzip.open(args.result_dir / 'objects.pkl.gz', 'rb') as f:
    objects = pickle.load(f)
map_file = args.result_dir / 'params_with_idx.npz'
if map_file.exists():
    with np.load(map_file) as saved_map:
        live_ids = set(saved_map['object_idx'].reshape(-1).astype(int).tolist())
    objects = [obj for obj in objects if int(obj['idx']) in live_ids]
if not objects:
    raise SystemExit('No saved objects.')
from runpy import run_path
resolved = args.result_dir / 'config.resolved.json'
cfg = (json.loads(resolved.read_text()) if resolved.exists() else run_path(str(args.result_dir / 'config.py'))['config'])['lang']
model_name = cfg.get('clip_model_name', 'ViT-H-14')
model, _, _ = open_clip.create_model_and_transforms(model_name, cfg.get('clip_pretrained', 'laion2b_s32b_b79k'))
model.eval()
with torch.no_grad():
    text = model.encode_text(open_clip.get_tokenizer(model_name)([args.query])).float()
    features = torch.from_numpy(np.stack([obj['clip_ft'] for obj in objects])).float()
    scores = torch.nn.functional.normalize(features, dim=-1) @ torch.nn.functional.normalize(text, dim=-1).T
values, indices = scores.flatten().topk(min(args.top_k, len(objects)))
result = [dict(idx=int(objects[i]['idx']), category=objects[i].get('category'), score=float(score)) for score, i in zip(values, indices)]
print(json.dumps(result, indent=2))
if args.output:
    args.output.write_text(json.dumps(dict(query=args.query, results=result), indent=2))
