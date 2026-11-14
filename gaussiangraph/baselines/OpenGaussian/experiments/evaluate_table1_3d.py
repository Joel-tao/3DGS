"""Explicit 3D decoding/classification BEFORE rasterization, common to all methods.

This is a registered independent Table-1-style protocol, not the unpublished author
script: 14 competing text classes, alpha>.7, no scale/KNN cleanup, zero empty masks.
"""
import sys,json,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from arguments import ModelParams,PipelineParams
from scene import Scene,GaussianModel
from gaussian_renderer import render
from scripts.compute_lerf_iou import evalute

def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

p=ArgumentParser();m=ModelParams(p);pipeline=PipelineParams(p)
p.add_argument('--method',choices=['langsplat','legaussian','cluster'],required=True)
p.add_argument('--le-decoding',choices=['argmax','softmax'],default='argmax')
p.add_argument('--min-association-views',type=int,default=5)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--decoder',type=Path)
p.add_argument('--codebook',type=Path);p.add_argument('--mapping',type=Path)
p.add_argument('--output',type=Path,required=True);p.add_argument('--model-source',type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
if (a.output/'metrics.json').exists():raise FileExistsError('Use a new evaluation directory')
queries=['nori','sake cup','kamaboko','corn','spoon','egg','onion segments','plate','napkin','bowl','glass of water','hand','chopsticks','wavy noodles']
frames=['frame_00006','frame_00024','frame_00060','frame_00065','frame_00081','frame_00119','frame_00128']
with torch.no_grad():
 g=GaussianModel(a.sh_degree);scene=Scene(m.extract(a),g,load_iteration=30000,shuffle=False)
 params,iteration=torch.load(a.checkpoint,map_location='cpu');g.active_sh_degree=params[0]
 if a.method=='legaussian':
  geometry=[params[i] for i in [1,2,3,5,6,7]];latent=params[4]
  Decoder=module(a.model_source/'scene/index_decoder.py','le_index').IndexDecoder
  state=torch.load(a.decoder,map_location='cpu');decoder=Decoder(latent.shape[1],state['conv3.bias'].numel()).cuda();decoder.load_state_dict(state);decoder.eval()
  codebook=torch.load(a.codebook,map_location='cpu')['embedding.weight'][:,:512].cuda()
 else:
  geometry=list(params[1:7]);latent=params[7] if a.method=='langsplat' else None
  if a.method=='langsplat':
   Autoencoder=module(a.model_source/'autoencoder/model.py','lang_ae').Autoencoder
   decoder=Autoencoder([256,128,64,32,3],[16,32,64,128,256,256,512]).cuda();decoder.load_state_dict(torch.load(a.decoder,map_location='cpu'));decoder.eval()
 del params
 for name,value in zip(('_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity'),geometry):setattr(g,name,torch.nn.Parameter(value.cuda(),requires_grad=False))
 n=len(g._xyz);g._ins_feat=torch.zeros(n,6,device='cuda');g._ins_feat_q=torch.empty(0,device='cuda')
 text=json.loads(Path('assets/text_features.json').read_text());text=F.normalize(torch.tensor([text[q] for q in queries],device='cuda',dtype=torch.float32),dim=1)
 scores=torch.empty(n,len(queries),device='cuda');valid=torch.ones(n,dtype=torch.bool,device='cuda')
 if a.method=='cluster':
  data=np.load(a.mapping);features=torch.from_numpy(data['leaf_feat']).cuda();ids=torch.from_numpy(data['leaf_ind']).cuda();count=torch.from_numpy(data['occu_count']).cuda();table_valid=(features.norm(dim=1)>0)&(count>=a.min_association_views)
  valid=(ids>=0)&(ids<len(features));safe=ids.clamp(0,len(features)-1);valid &=table_valid[safe]
 for start in range(0,n,8192):
  if a.method=='langsplat':decoded=decoder.decode(F.normalize(latent[start:start+8192].cuda(),dim=1))
  elif a.method=='legaussian':
   local=F.normalize(latent[start:start+8192].cuda(),dim=1);logits=decoder(local[:,:,None,None]).squeeze(-1).squeeze(-1);decoded=codebook[logits.argmax(1)] if a.le_decoding=="argmax" else logits.softmax(1)@codebook
  else:decoded=features[safe[start:start+8192]]
  scores[start:start+8192]=F.normalize(decoded,dim=1)@text.T
 scores[~valid]=-torch.inf;labels=scores.argmax(1);labels[~valid]=-1
 print('Decoded/classified',n,'Gaussians before rendering',flush=True)
 np.save(a.output/'gaussian_classes.npy',labels.cpu().numpy())
 views={v.image_name:v for v in scene.getTrainCameras()+scene.getTestCameras() if v.image_name in frames};metadata={}
 for frame,view in views.items():
  maxima=None
  for index,query in enumerate(queries):
   r=render(view,g,pipeline.extract(a),torch.ones(3,device='cuda'),iteration,rescale=False,leaf_cluster_idx=labels,selected_leaf_id=torch.tensor([index],device='cuda'),render_feat_map=False,better_vis=False,seg_rgb=True,post_process=False,root_num=1,leaf_num=len(queries))
   if maxima is None:maxima=scores[r['visibility_filter']].max(0).values.cpu().tolist()
   masks=r['leaf_cluster_silhouettes'];empty=not r['leaf_clusters_imgs']
   pred=np.zeros((view.image_height,view.image_width),np.uint8) if empty else (masks[0]>.7).cpu().numpy().astype('uint8')
   Image.fromarray(pred*255).save(a.output/(frame+'_'+query+'.png'));del r,masks
  metadata[frame]=dict(zip(queries,maxima));print('Rendered',frame,flush=True)
 metrics=evalute(str(Path(a.source_path).parent/'label/ramen/gt'),str(a.output),'ramen');metrics.update(method=a.method,protocol='table1_3d_classification_v1',checkpoint=str(a.checkpoint),decoded_in_3d=True,min_association_views=a.min_association_views,le_decoding=a.le_decoding if a.method=="legaussian" else None)
 (a.output/'metrics.json').write_text(json.dumps(metrics,indent=2));(a.output/'level_scores.json').write_text(json.dumps(metadata,indent=2))
 print(json.dumps(metrics,indent=2),flush=True)
