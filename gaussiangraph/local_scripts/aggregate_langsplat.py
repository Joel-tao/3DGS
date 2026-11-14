import json, shutil, sys
from pathlib import Path
sys.path.insert(0,'/home/joe/gaussiangraph/baselines/OpenGaussian')
from scripts.compute_lerf_iou import evalute
b=Path('/media/joe/Data/gaussiangraph'); roots={i:b/f'experiments/langsplat_3d_level{i}' for i in (1,2,3)}
scores={i:json.loads((p/'level_scores.json').read_text()) for i,p in roots.items()}
out=b/'experiments/langsplat_3d_multilevel_v1';out.mkdir(exist_ok=True)
selected={}
# Fixed before inspecting aggregate GT: highest visible-Gaussian text cosine,
# ties resolved by lower level. GT used only after all masks are selected.
for frame, queries in scores[1].items():
 selected[frame]={}
 for query in queries:
  level=max((1,2,3),key=lambda i:scores[i][frame][query])
  selected[frame][query]=level
  name=frame+'_'+query+'.png';shutil.copyfile(roots[level]/name,out/name)
(out/'selected_levels.json').write_text(json.dumps(selected,indent=2))
m=evalute(str(b/'data/lerf_ovs/label/ramen/gt'),str(out),'ramen')
m.update(method='langsplat',protocol='table1_3d_classification_v1',level_selection='maximum visible-Gaussian text cosine; ties lower level; no GT',decoded_in_3d=True)
(out/'metrics.json').write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2))
