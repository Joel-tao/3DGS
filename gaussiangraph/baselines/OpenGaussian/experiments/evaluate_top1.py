"""Common, predeclared top-1-cluster protocol for both baseline and CF.

Uses existing text features, occurrence cutoff 5, scale/KNN filtering and alpha .7.
Writes explicit zero masks for empty renders; never chooses settings using test GT.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from arguments import ModelParams,PipelineParams
from scene import Scene,GaussianModel
from gaussian_renderer import render
from scripts.compute_lerf_iou import evalute
p=ArgumentParser();m=ModelParams(p);pipe=PipelineParams(p)
p.add_argument('--text-features',type=Path,default=Path('assets/text_features.json'));p.add_argument('--output',type=Path,required=True);p.add_argument('--iteration',type=int,default=70000);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
if (a.output/'metrics.json').exists():raise FileExistsError('Evaluation output already contains metrics')
frames=['frame_00006','frame_00024','frame_00060','frame_00065','frame_00081','frame_00119','frame_00128']
queries=['nori','sake cup','kamaboko','corn','spoon','egg','onion segments','plate','napkin','bowl','glass of water','hand','chopsticks','wavy noodles']
with torch.no_grad():
 g=GaussianModel(a.sh_degree);scene=Scene(m.extract(a),g,load_iteration=a.iteration,shuffle=False)
 views={v.image_name:v for v in scene.getTrainCameras()+scene.getTestCameras() if v.image_name in frames}
 if set(views)!=set(frames):raise ValueError('Evaluation view missing')
 data=np.load(Path(a.model_path)/'cluster_lang.npz');features=torch.from_numpy(data['leaf_feat']).cuda();count=torch.from_numpy(data['occu_count']).cuda();features[count<5]=0;features=F.normalize(features,dim=1)
 labels=torch.from_numpy(data['leaf_ind']).cuda();text=json.loads(a.text_features.read_text());bg=torch.ones(3,device='cuda')
 records=[]
 for query in queries:
  sim=F.normalize(torch.tensor(text[query],device='cuda',dtype=torch.float32)[None],dim=1)@features.T
  selected=sim.argmax(-1)
  for frame in frames:
   v=views[frame];r=render(v,g,pipe.extract(a),bg,a.iteration,rescale=False,leaf_cluster_idx=labels,selected_leaf_id=selected,render_feat_map=False,better_vis=True,seg_rgb=True,post_process=True,root_num=1,leaf_num=len(features))
   absent=len(r['leaf_clusters_imgs'])==0
   pred=np.zeros((v.image_height,v.image_width),dtype='uint8') if absent else (r['leaf_cluster_silhouettes'][0]>.7).cpu().numpy().astype('uint8')
   Image.fromarray(pred*255).save(a.output/(frame+'_'+query+'.png'))
   records.append(dict(query=query,frame=frame,cluster=selected.item(),score=sim.max().item(),no_render=absent,foreground_pixels=int(pred.sum())))
   del r
  print('Completed',query,flush=True)
 gt=Path(a.source_path).parent/'label/ramen/gt';metrics=evalute(str(gt),str(a.output),'ramen')
 metrics['protocol']='top1_fixed_v1';metrics['model_path']=a.model_path
 metrics['empty_gt_predictions']=sum(r['foreground_pixels']==0 for r in records if (gt/r['frame']/(r['query']+'.jpg')).exists())
 (a.output/'metrics.json').write_text(json.dumps(metrics,indent=2));(a.output/'queries.json').write_text(json.dumps(records,indent=2))
 print(json.dumps(metrics,indent=2),flush=True)
