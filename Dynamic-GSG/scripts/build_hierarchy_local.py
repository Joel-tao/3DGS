"""Build a reproducible local hierarchy from a saved Replica object Gaussian map."""
import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils.scene_graph_local import build_hierarchy
from utils.model_classification import model_overrides

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('result_dir', type=Path)
parser.add_argument('--overrides', type=Path, help='JSON keyed by instance ID: category and/or object_type')
parser.add_argument('--xy-margin', type=float, default=.08)
parser.add_argument('--vertical-tolerance', type=float, default=.15)
parser.add_argument('--model-classifications',type=Path)
parser.add_argument('--output-stem',default='scene_graph_hierarchy_local')
args = parser.parse_args()
cfg = json.loads((args.result_dir/'config.resolved.json').read_text())
if 'replica' not in cfg['data']['gradslam_data_cfg'].lower():
    raise SystemExit('Only Replica with its Z-up absolute trajectory is supported. Do not infer gravity from an arbitrary camera frame.')
trajectory = Path(cfg['data']['basedir'])/cfg['data']['sequence']/'traj.txt'
first_pose = np.loadtxt(trajectory)[cfg['data'].get('start', 0)].reshape(4, 4)
# Dataset-relative map coordinates are first-camera coordinates. Restore the
# Replica world frame before comparing heights; camera Y/Z are not gravity.
with np.load(args.result_dir/'params_with_idx.npz') as data:
    xyz = data['means3D'] @ first_pose[:3, :3].T + first_pose[:3, 3]
    ids = data['object_idx'].ravel()
with gzip.open(args.result_dir/'objects.pkl.gz', 'rb') as f:
    objects = pickle.load(f)
nodes = []
excluded = []
for obj in objects:
    points = xyz[ids == obj['idx']]
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        excluded.append(dict(id=int(obj['idx']), reason='no_finite_gaussians'))
        continue
    # Quantiles reduce isolated splat outliers; not an exact object mesh.
    lo, hi = np.quantile(points, [.05, .95], axis=0)
    nodes.append(dict(id=int(obj['idx']), category=obj['category'], gaussian_count=len(points), center_m=np.median(points, axis=0).tolist(), bounds_min_m=lo.tolist(), bounds_max_m=hi.tolist()))
overrides = json.loads(args.overrides.read_text()) if args.overrides else {}
annotations = None
if args.model_classifications:
    if args.overrides:
        parser.error('Model classifications and manual overrides must be used separately')
    annotations=json.loads(args.model_classifications.read_text())
    overrides=model_overrides(nodes,annotations)
unknown_ids = set(overrides)-{str(n['id']) for n in nodes}
if unknown_ids:
    raise SystemExit(f'Override IDs absent from the map: {sorted(unknown_ids)}')
graph = build_hierarchy(nodes, xy_margin=args.xy_margin, vertical_tolerance=args.vertical_tolerance, overrides=overrides)
graph.update(coordinate_frame='Replica world, Z-up, meters', map_to_world=first_pose.tolist(), bounds_quantiles=[.05,.95], excluded_objects=excluded, classification_overrides=overrides, source_config=cfg, limitations=['Category rules replace LVLM classification.', 'Attached-to edges are geometric hypotheses, not verified support/on/inside relations.', 'Input instances may contain segmentation errors or duplicate physical objects.'])
if annotations is not None:
    graph['method']='Local language-model category classification + geometric attachment; DAM disabled'
    graph['classification_model']=annotations['model']
    graph['dam_used']=False
    graph['limitations']=['Text-only local small model replaces the paper model; DAM disabled.', 'Attachment edges are geometric hypotheses, not verified support.', 'Upstream segmentation and association errors remain.']
    for n in graph['nodes'][1:]:
        n['type_source']='local_text_model'
        n['classification']=annotations['objects'][str(n['id'])]
output = args.result_dir/(args.output_stem+'.json')
output.write_text(json.dumps(graph, indent=2, allow_nan=False))
counts = {t:sum(n['object_type']==t for n in graph['nodes']) for t in ['Asset','Ordinary','Standalone']}
summary = dict(objects=len(nodes), types=counts, attached=sum(e['relation']=='attached_to' for e in graph['edges']), unresolved=sum(n.get('attachment_status')=='unresolved' for n in graph['nodes']))
print(json.dumps(summary, indent=2))
lines = ['# Local scene hierarchy', '', graph['method'], '', '| ID | Category | Type | Parent | Status |', '|---|---|---|---|---|']
for n in graph['nodes'][1:]:
    lines.append(f"| {n['id']} | {n['category']} | {n['object_type']} | {n['parent']} | {n['attachment_status']} |")
(args.result_dir/(args.output_stem+'.md')).write_text('\n'.join(lines)+'\n')
