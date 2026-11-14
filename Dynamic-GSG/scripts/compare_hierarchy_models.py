"""Report differences between local rule and local model hierarchies, not accuracy."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('result_dir',type=Path);a=p.parse_args()
rule=json.loads((a.result_dir/'scene_graph_hierarchy_local.json').read_text())
model=json.loads((a.result_dir/'scene_graph_hierarchy_model.json').read_text())
old={n['id']:n for n in rule['nodes']};new={n['id']:n for n in model['nodes']}
if old.keys()!=new.keys():raise SystemExit('Cannot compare graphs with different instance sets')
changed_types=[dict(id=i,category=new[i]['category'],rule=old[i]['object_type'],model=new[i]['object_type']) for i in new if old[i]['object_type']!=new[i]['object_type']]
changed_parents=[dict(id=i,category=new[i]['category'],rule=old[i]['parent'],model=new[i]['parent']) for i in new if old[i]['parent']!=new[i]['parent']]
summary=dict(model=model['classification_model'],dam_used=False,instance_count=len(new)-1,type_counts={t:sum(n['object_type']==t for n in model['nodes']) for t in ['Asset','Ordinary','Standalone']},attachment_candidates=sum(e['relation']=='attached_to' for e in model['edges']),unresolved=sum(n.get('attachment_status')=='unresolved' for n in model['nodes']),changed_types=changed_types,changed_parents=changed_parents,notice='Differences only; no ground-truth relation accuracy evaluation.')
(a.result_dir/'hierarchy_model_comparison.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
