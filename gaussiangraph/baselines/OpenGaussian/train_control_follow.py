"""Train an independent Control-Follow variant from the frozen 30k reconstruction.

Uses the SAME third-party masks and OpenGaussian losses to isolate clustering.
This is not the full SAM2/confidence-weighted GaussianGraph reproduction.
"""
from argparse import ArgumentParser
from pathlib import Path
import json,random,time
import numpy as np
import torch
from arguments import ModelParams,PipelineParams
from scene import Scene,GaussianModel
from scene.control_follow import CFConfig,ControlFollow,GradientHistory
from gaussian_renderer import render
from utils.opengs_utlis import get_SAM_mask_and_feat,mask_feature_mean
from train import separation_loss
from utils.confidence_loss import confidence_losses

def smooth_loss(feature,masks,means):
    losses=[]
    for start in range(0,len(masks),2):
        mask=masks[start:start+2]
        distance=(feature[None]-means[start:start+2,:,None,None]).norm(dim=1)
        losses.append((distance*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1))
    return torch.cat(losses).mean()

p=ArgumentParser();model=ModelParams(p);pipeline=PipelineParams(p)
p.add_argument('--base-model',type=Path,required=True)
p.add_argument('--control-start',type=int,default=40000);p.add_argument('--follow-start',type=int,default=50000)
p.add_argument('--refine-start',type=int,default=68000);p.add_argument('--iterations',type=int,default=70000)
p.add_argument('--history-window',type=int,default=1000);p.add_argument('--stable-epsilon',type=float,default=0,help='0: training-gradient median; positive: fixed paper epsilon')
p.add_argument('--resume',type=Path)
p.add_argument('--confidence-dir',type=Path,help='Aligned SAM2 *_c.npy files; enables confidence Eq3/4 losses')
p.add_argument('--cluster-space',choices=['raw','unit'],default='raw')
p.add_argument('--position-scale',type=float,default=.5)
p.add_argument('--control-points',type=int,default=5000);p.add_argument('--voxel-size',type=float,default=.02)
p.add_argument('--control-threshold',type=float,default=0);p.add_argument('--follow-threshold',type=float,default=0)
p.add_argument('--refine-space',choices=['raw','unit'],default='unit');p.add_argument('--refine-min-points',type=int,default=32)
p.add_argument('--refine-threshold',type=float,default=.2);p.add_argument('--refresh-control',type=int,default=200)
p.add_argument('--seed',type=int,default=0);p.add_argument('--initial-iteration',type=int,default=30000)
a=p.parse_args()
if not(a.initial_iteration<a.control_start<a.follow_start<a.refine_start<=a.iterations):p.error('Invalid stage schedule')
out=Path(a.model_path);out.mkdir(parents=True,exist_ok=True)
if (out/'run.json').exists() and not a.resume:raise FileExistsError('Use a fresh output directory or explicit --resume')
if not a.resume:(out/'run.json').write_text(json.dumps(vars(a),default=str,indent=2))
random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
config=CFConfig(cluster_space=a.cluster_space,position_scale=a.position_scale,control_points=a.control_points,voxel_size=a.voxel_size,control_threshold=a.control_threshold,follow_threshold=a.follow_threshold,refine_threshold=a.refine_threshold,refine_space=a.refine_space,refine_min_points=a.refine_min_points)
cf=ControlFollow(config)
dataset=model.extract(a);dataset.model_path=str(a.base_model)
g=GaussianModel(a.sh_degree);scene=Scene(dataset,g,load_iteration=a.initial_iteration,shuffle=False);scene.model_path=str(out)
params,loaded=torch.load(a.base_model/f'chkpnt{a.initial_iteration}.pth',map_location='cpu')
assert loaded==a.initial_iteration and len(params)==14
for name,value in zip(('_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity','_ins_feat'),params[1:8]):
 setattr(g,name,torch.nn.Parameter(value.detach().cuda(),requires_grad=name=='_ins_feat'))
g.active_sh_degree=params[0];g._ins_feat_q=torch.empty(0,device='cuda');del params
optimizer=torch.optim.Adam([g._ins_feat],lr=.001,eps=1e-15)
history=GradientHistory(len(g._xyz),g._xyz.device);stack=[];first=a.initial_iteration
if a.resume:
    state=torch.load(a.resume,map_location='cpu');first=state['iteration'];g._ins_feat.data.copy_(state['features'].cuda());optimizer.load_state_dict(state['optimizer'])
    saved=state['clustering'];cf=ControlFollow(CFConfig(**saved['config']))
    if a.cluster_space != cf.config.cluster_space or a.position_scale != cf.config.position_scale:
        raise ValueError('Resume must preserve clustering space and position scale; use a fresh run for this comparison')
    for key in ('controls','control_labels','labels'):setattr(cf,key,saved[key].cuda() if saved[key] is not None else None)
    cf.config.refine_space=a.refine_space;cf.config.refine_min_points=a.refine_min_points;cf.config.refine_threshold=a.refine_threshold
    cf.threshold=saved['threshold'];cf.phase=saved['phase'];history.total=state['gradient_total'].cuda();history.seen=state['gradient_seen'].cuda();history.steps=state['gradient_steps']
    by_name={v.image_name:v for v in scene.getTrainCameras()};stack=[by_name[n] for n in state.get('remaining_views',[])]
    random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);del state
bg=torch.zeros(3,device='cuda');pipe=pipeline.extract(a)
def save(iteration):
    d=out/'point_cloud'/f'iteration_{iteration}';d.mkdir(parents=True,exist_ok=True)
    g.save_ply(str(d/'point_cloud.ply'),['ins_feat'] if cf.labels is not None else [])
    torch.save(dict(iteration=iteration,features=g._ins_feat.detach().cpu(),optimizer=optimizer.state_dict(),clustering=cf.state_dict(),gradient_total=history.total.cpu(),gradient_seen=history.seen.cpu(),gradient_steps=history.steps,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),remaining_views=[v.image_name for v in stack]),out/f'cf_checkpoint{iteration}.pth')
    if cf.labels is not None:
        np.savez(out/'clusters.npz',labels=cf.labels.cpu().numpy(),control_indices=cf.controls.cpu().numpy())
with (out/'progress.jsonl').open('a' if a.resume else 'w',buffering=1) as log:
    started=time.monotonic()
    for iteration in range(first+1,a.iterations+1):
        if iteration==a.control_start:
            averages=history.total/max(history.steps,1);observed=history.seen>0
            epsilon=a.stable_epsilon or float(averages[observed].median())+1e-15
            (out/'stability.json').write_text(json.dumps(dict(epsilon=epsilon,selection='fixed_epsilon' if a.stable_epsilon else 'training_median',steps=history.steps,observed=int(observed.sum()),mean_gradient=float(averages[observed].mean())),indent=2))
            stable=history.stable(epsilon);cf.select_controls(g._xyz,stable);cf.control(g._xyz,g._ins_feat)
            print('Control stage:',int(stable.sum()),'stable,',len(cf.controls),'controls',flush=True)
        elif a.control_start<iteration<a.follow_start and (iteration-a.control_start)%a.refresh_control==0:
            cf.control(g._xyz,g._ins_feat)
        if iteration==a.follow_start:
            print('Starting exact online Follow assignment',flush=True);cf.follow(g._xyz,g._ins_feat)
            print('Follow clusters:',int(cf.labels.max())+1,flush=True)
        if iteration==a.refine_start:
            cf.refine(g._ins_feat);print('Refined clusters:',int(cf.labels.max())+1,flush=True)
        g._ins_feat_q=cf.quantize(g._ins_feat) if cf.labels is not None else torch.empty(0,device='cuda')
        if not stack:stack=scene.getTrainCameras().copy();random.shuffle(stack)
        view=stack.pop();view.to_gpu();optimizer.zero_grad(set_to_none=True)
        pkg=render(view,g,pipe,bg,iteration,rescale=iteration>=a.control_start)
        _,masks,_=get_SAM_mask_and_feat(view.original_sam_mask,level=3)
        means=mask_feature_mean(pkg['ins_feat'],masks,image_mask=(pkg['silhouette']>.7).float())
        if a.confidence_dir:
            confidence=torch.from_numpy(np.load(a.confidence_dir/(view.image_name+'_c.npy'))).to(means.device)
            start_id=int(view.original_sam_mask[2].max().item())+1
            end_id=int(view.original_sam_mask[3].max().item())+1
            smooth,contrast=confidence_losses(pkg['ins_feat'],masks,means,confidence[start_id:end_id])
            loss=.01*smooth+contrast
        else:
            loss=.01*smooth_loss(pkg['ins_feat'],masks,means)+separation_loss(means,iteration)
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at {iteration}')
        loss.backward()
        if a.control_start-a.history_window<=iteration<a.control_start:
            history.update(g._ins_feat.grad,pkg['visibility_filter'])
        optimizer.step()
        view.to_cpu()
        if iteration%100==0 or iteration==a.iterations:
            record=dict(iteration=iteration,phase=cf.phase,loss=loss.item(),clusters=int(cf.labels.max())+1 if cf.labels is not None else 0,seconds=time.monotonic()-started)
            log.write(json.dumps(record)+'\n');print(record,flush=True)
        if iteration in (a.control_start-1,a.follow_start-1,a.refine_start-1,a.iterations):
            # Refresh quantized values after the final optimizer update before export.
            g._ins_feat_q=cf.quantize(g._ins_feat).detach() if cf.labels is not None else torch.empty(0,device='cuda');save(iteration)
        del pkg,loss,means,masks
    (out/'TRAINING_COMPLETE').write_text('Control-Follow training complete; association/evaluation remain.\n')
