"""Evaluate predeclared postprocessing ablations, keeping the original baseline intact."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,csv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from arguments import ModelParams,PipelineParams
from scene import Scene,GaussianModel
from gaussian_renderer import render
from utils.opengs_utlis import load_code_book
p=ArgumentParser();m=ModelParams(p);pipe=PipelineParams(p)
p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
frames=['frame_00006','frame_00024','frame_00060','frame_00065','frame_00081','frame_00119','frame_00128']
queries=['nori','sake cup','kamaboko','corn','spoon','egg','onion segments','plate','napkin','bowl','glass of water','hand','chopsticks','wavy noodles']
# Predeclared comparisons: never select a configuration by its test GT score.
variants={'upstream':(True,True),'no_knn':(True,False),'no_scale_filter':(False,True),'no_scale_or_knn':(False,False)}
with torch.no_grad():
 g=GaussianModel(a.sh_degree);s=Scene(m.extract(a),g,load_iteration=70000,shuffle=False)
 views={v.image_name:v for v in s.getTrainCameras()+s.getTestCameras() if v.image_name in frames}
 data=np.load(Path(a.model_path)/'cluster_lang.npz');lang=torch.from_numpy(data['leaf_feat']).cuda();counts=torch.from_numpy(data['occu_count']).cuda();lang[counts<5]=0
 labels=torch.from_numpy(data['leaf_ind']).cuda();lang=F.normalize(lang,dim=1)
 code,_=load_code_book(str(Path(a.model_path)/'point_cloud/iteration_70000/leaf_code_book'));centers=code['ins_feat'].cuda()
 text=json.loads(Path('assets/text_features.json').read_text());bg=torch.ones(3,device='cuda')
 rows=[];selection=[]
 for query in queries:
  sim=F.normalize(torch.tensor(text[query],device='cuda',dtype=torch.float32).reshape(1,-1),dim=1)@lang.T
  best=sim.argmax(-1);selected=best
  for candidate in sim.topk(10).indices[0][1:]:
   if candidate-best<10 and torch.norm(centers[best]-centers[candidate],dim=1)<.9:selected=torch.cat((selected,candidate.reshape(1)))
  selection.append({'query':query,'top_cluster':best.item(),'selected':selected.tolist(),'selected_roots':(selected//10).tolist(),'score':sim.max().item(),'occurrences':counts[best].item()})
  for frame in frames:
   gt_path=Path(a.source_path).parent/'label/ramen/gt'/frame/(query+'.jpg')
   gt=np.array(Image.open(gt_path))>10 if gt_path.exists() else None
   v=views[frame]
   for name,(scale,knn) in variants.items():
    r=render(v,g,pipe.extract(a),bg,70000,rescale=False,leaf_cluster_idx=labels,selected_leaf_id=selected,render_feat_map=False,better_vis=scale,seg_rgb=True,post_process=knn,root_num=64,leaf_num=10)
    masks=r['leaf_cluster_silhouettes'];missing=len(r['leaf_clusters_imgs'])==0
    pred=np.zeros((v.image_height,v.image_width),bool) if missing else (masks[0]>.7).cpu().numpy()
    visible=(torch.isin(labels,selected)&r['visibility_filter']);small=visible&(g.get_scaling<.1).all(1)
    row={'variant':name,'frame':frame,'query':query,'is_gt':gt is not None,'missing_output':missing,'selected_points':torch.isin(labels,selected).sum().item(),'visible_points':visible.sum().item(),'small_visible_points':small.sum().item(),'pred_pixels':int(pred.sum()),'iou':float(np.logical_and(pred,gt).sum()/max(np.logical_or(pred,gt).sum(),1)) if gt is not None else None}
    rows.append(row)
    dest=a.output/name/frame;dest.mkdir(parents=True,exist_ok=True);Image.fromarray(pred.astype('uint8')*255).save(dest/(query+'.png'))
    del r,masks
  print('Completed query:',query,flush=True)
 summary={}
 for name in variants:
  subset=[r for r in rows if r['variant']==name and r['is_gt']];values=np.array([r['iou'] for r in subset]);summary[name]={'count':len(subset),'missing_predictions':sum(r['missing_output'] for r in subset),'mIoU':values.mean().item(),'Acc@0.25':(values>.25).mean().item(),'Acc@0.5':(values>.5).mean().item()}
 with (a.output/'items.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 (a.output/'selection.json').write_text(json.dumps(selection,indent=2));(a.output/'metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
