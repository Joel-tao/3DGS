"""Query local hierarchy using category/parent filters; no external API."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.scene_graph_local import query_hierarchy
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('result_dir', type=Path)
parser.add_argument('--category', help='e.g. pillow, lamp, chair')
parser.add_argument('--parent', help='e.g. couch; filters inferred attached-to relationships')
parser.add_argument('--id', type=int)
parser.add_argument('--parent-id', type=int, help='Filter by a specific parent instance')
parser.add_argument('--output', type=Path)
parser.add_argument('--graph-stem',default='scene_graph_hierarchy_local')
args = parser.parse_args()
graph = json.loads((args.result_dir/(args.graph_stem+'.json')).read_text())
result = dict(query=dict(category=args.category,parent=args.parent,parent_id=args.parent_id,id=args.id), relation_notice='Geometric attachment candidates, not verified on/inside relations', results=query_hierarchy(graph,args.category,args.parent,args.id,args.parent_id))
print(json.dumps(result, indent=2))
if args.output:
    args.output.write_text(json.dumps(result, indent=2))
