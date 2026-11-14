"""Associate arbitrary compact CF labels using memory-bounded batches of 8 clusters."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
from arguments import ModelParams,PipelineParams
from scene import Scene,GaussianModel
from gaussian_renderer import render
from train import construct_pseudo_ins_feat
p=ArgumentParser();m=ModelParams(p);pipe=PipelineParams(p)
p.add_argument('--iteration',type=int,default=70000);p.add_argument('--association-score',choices=['opengs','iou'],default='opengs');p.add_argument('--output',type=Path)
p.add_argument('--no-pseudo-filter',action='store_true')
a=p.parse_args()
with torch.no_grad():
    g=GaussianModel(a.sh_degree);scene=Scene(m.extract(a),g,load_iteration=a.iteration,shuffle=False)
    state=torch.load(Path(a.model_path)/f'cf_checkpoint{a.iteration}.pth',map_location='cpu')
    g._ins_feat=torch.nn.Parameter(state['features'].cuda(),requires_grad=False)
    labels=state['clustering']['labels'].cuda();del state
    if labels.shape!=(len(g._xyz),) or (labels<0).any():raise ValueError('Incomplete or mismatched CF labels')
    k=int(labels.max())+1
    singleton_fraction=float((torch.bincount(labels)==1).sum())/len(labels)
    if k>max(10000,len(labels)//20) or singleton_fraction>.5:
        raise RuntimeError(f'Fragmented clustering: K={k}, singleton point fraction={singleton_fraction:.3f}; association stopped before expensive rendering')
    if a.output:
        a.output.mkdir(parents=True,exist_ok=True)
        if (a.output/'cluster_lang.npz').exists():raise FileExistsError('Refusing to overwrite association')
        scene.model_path=str(a.output)
    batch=8;roots=(k+batch-1)//batch
    print('Associating',k,'clusters in',roots,'render batches',flush=True)
    bg=torch.tensor([1.,1.,1.] if a.white_background else [0.,0.,0.],device='cuda')
    construct_pseudo_ins_feat(scene,render,(pipe.extract(a),bg,a.iteration),cluster_indices=labels,mode='lang',root_num=roots,leaf_num=batch,sam_level=3,save_memory=True,association_score=a.association_score,filter=not a.no_pseudo_filter,association_debug=False)
    print('CF association complete.',flush=True)
