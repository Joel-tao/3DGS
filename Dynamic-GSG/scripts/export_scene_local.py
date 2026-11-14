"""Export Gaussian centers and a geometric proximity graph (not the paper's LLM hierarchy)."""
import argparse
import gzip
import json
import pickle
from pathlib import Path
import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('result_dir', type=Path)
parser.add_argument('--near-distance', type=float, default=1.0, help='Center distance threshold in meters')
args = parser.parse_args()
if (args.result_dir / 'config.resolved.json').exists():
    (args.result_dir / 'config_replay.py').write_text('import json\nfrom pathlib import Path\nconfig = json.loads(Path(__file__).with_name("config.resolved.json").read_text())\n')
data = np.load(args.result_dir / 'params_with_idx.npz')
points = data['means3D']
colors = np.clip(data['rgb_colors'], 0, 1)
ids = data['object_idx'].reshape(-1)
with gzip.open(args.result_dir / 'objects.pkl.gz', 'rb') as f:
    objects = pickle.load(f)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(str(args.result_dir / 'gaussian_centers_rgb.ply'), pcd)
instance_colors = np.random.default_rng(0).uniform(.15, 1, (int(ids.max()) + 1, 3))
instance_colors[0] = [.5, .5, .5]
pcd.colors = o3d.utility.Vector3dVector(instance_colors[ids.astype(int)])
o3d.io.write_point_cloud(str(args.result_dir / 'gaussian_centers_instances.ply'), pcd)
nodes = []
for obj in objects:
    xyz = points[ids == obj['idx']]
    if not len(xyz):
        continue
    nodes.append(dict(id=int(obj['idx']), category=obj['category'], gaussian_count=len(xyz), center_m=np.median(xyz, axis=0).tolist(), aabb_min_m=xyz.min(axis=0).tolist(), aabb_max_m=xyz.max(axis=0).tolist()))
edges = []
for i, a in enumerate(nodes):
    for b in nodes[i+1:]:
        distance = float(np.linalg.norm(np.array(a['center_m']) - b['center_m']))
        if distance <= args.near_distance:
            edges.append(dict(source=a['id'], target=b['id'], relation='near', center_distance_m=distance))
graph = dict(directed=False, method='Geometric center proximity only; not the paper LLM hierarchy', near_distance_m=args.near_distance, nodes=nodes, edges=edges)
(args.result_dir / 'scene_graph_geometric.json').write_text(json.dumps(graph, indent=2))
print(json.dumps(dict(gaussians=len(points), objects=len(nodes), edges=len(edges)), indent=2))
