"""Draw the saved hierarchy, with unverified attachment edges clearly marked."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('result_dir', type=Path)
parser.add_argument('--graph-stem',default='scene_graph_hierarchy_local')
args = parser.parse_args()
graph = json.loads((args.result_dir/(args.graph_stem+'.json')).read_text())
nodes = {n['id']:n for n in graph['nodes']}
children = {k:[] for k in nodes}
for edge in graph['edges']:
    children[edge['target']].append(edge['source'])
for key in children:
    children[key].sort(key=lambda i:(nodes[i]['object_type'],nodes[i]['category'],i))
positions = {}
leaf = [0]
def place(node_id, depth):
    if children[node_id]:
        ys = [place(c,depth+1) for c in children[node_id]]
        y = sum(ys)/len(ys)
    else:
        y = leaf[0]
        leaf[0] += 1
    positions[node_id] = (depth, y)
    return y
place('room',0)
colors = dict(Room='#465564',Asset='#19738a',Ordinary='#aa6418',Standalone='#6960a0')
fig, ax = plt.subplots(figsize=(15,max(7,leaf[0]*.45)))
for e in graph['edges']:
    x,y = positions[e['source']]; px,py = positions[e['target']]
    ax.plot([px,x],[py,y], color='#9b9b9b',lw=1.1,linestyle='--' if e['relation']=='attached_to' else '-')
for key,node in nodes.items():
    x,y = positions[key]
    unresolved = node.get('attachment_status')=='unresolved'
    label = 'room0' if key=='room' else f"#{key} {node['category']}"+(' [unresolved]' if unresolved else '')
    ax.text(x,y,label,ha='center',va='center',fontsize=10,color='white',bbox=dict(boxstyle='round,pad=.4',facecolor=colors[node['object_type']],edgecolor='#ca3030' if unresolved else 'white',linewidth=2))
ax.invert_yaxis(); ax.set_xlim(-.45,2.55); ax.axis('off')
ax.set_title(('Local model classification' if 'classification_model' in graph else 'Category rules')+' + geometric attachment candidates\nSolid = room membership; dashed = unverified attachment',fontsize=14,pad=24)
ax.legend(handles=[Line2D([0],[0],marker='s',ls='',color=v,label=k,markersize=10) for k,v in colors.items()],loc='lower left')
fig.tight_layout()
for ext in ['png','svg']:
    path = args.result_dir/f'{args.graph_stem}.{ext}'
    fig.savefig(path,dpi=150,bbox_inches='tight')
    print(path)
