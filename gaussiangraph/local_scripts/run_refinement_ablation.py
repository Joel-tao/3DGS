from pathlib import Path
import sys,json,subprocess,datetime,traceback,hashlib
import numpy as np,torch
B=Path('/media/joe/Data/gaussiangraph');R=Path('/home/joe/gaussiangraph/baselines/OpenGaussian');sys.path.insert(0,str(R))
from scene.control_follow import ControlFollow,CFConfig
root=B/'experiments/refinement_ablation_v1';root.mkdir(parents=True,exist_ok=True);source=B/'outputs/ramen_sam2_confidence_v1';P=B/'envs/opengs/bin/python'
def status(state,**kw):(B/'logs/refinement-ablation-status.json').write_text(json.dumps(dict(time=datetime.datetime.now().isoformat(),state=state,**kw),indent=2))
def run(args,log):
 with (B/'logs'/log).open('a') as f:subprocess.run(list(map(str,[P,'-u']+args)),cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
def stats(labels):
 a=labels.cpu().numpy();counts=np.unique(a[a>=0],return_counts=True)[1]
 return dict(clusters=len(counts),assigned=int((a>=0).sum()),size_quantiles=np.quantile(counts,[0,.25,.5,.75,1]).tolist(),clusters_under32=int((counts<32).sum()),points_under32=int(counts[counts<32].sum()))
try:
 status('preparing')
 state=torch.load(source/'cf_checkpoint67999.pth',map_location='cpu');f=state['features'];saved=state['clustering'];parent=saved['labels'].cpu().clone();assert state['iteration']==67999 and saved['phase']=='follow'
 cf=ControlFollow(CFConfig(**saved['config']));cf.labels=parent.clone();cf.refine(f);child=cf.labels.cpu();assert len(parent)==len(child) and (child>=0).all()
 controls=torch.load(source/'cf_checkpoint49999.pth',map_location='cpu')['clustering']['labels']
 stage={'control_at49999':stats(controls),'follow_at67999':stats(parent),'refine_same67999_features':stats(child)}
 (root/'stage_stats.json').write_text(json.dumps(stage,indent=2))
 parents=[]
 for label in torch.unique(parent).tolist():
  ids=(parent==label).nonzero().flatten();kids,counts=torch.unique(child[ids],return_counts=True)
  parents.append(dict(parent=label,points=len(ids),children=len(kids),child_ids=kids.tolist(),child_sizes=counts.tolist()))
 parents.sort(key=lambda r:r['children'],reverse=True);(root/'parent_children.json').write_text(json.dumps(parents,indent=2))
 feature_hash=hashlib.sha256(f.numpy().tobytes()).hexdigest();results=[]
 for name,labels in [('no_refine',parent),('refine',child)]:
  model=root/name;model.mkdir(exist_ok=True);pc=model/'point_cloud';pc.mkdir(exist_ok=True);link=pc/'iteration_67999'
  if not link.exists():link.symlink_to(source/'point_cloud/iteration_67999',target_is_directory=True)
  torch.save(dict(iteration=67999,features=f,clustering=dict(labels=labels)),model/'cf_checkpoint67999.pth')
  (model/'config.json').write_text(json.dumps(dict(source_checkpoint=str(source/'cf_checkpoint67999.pth'),feature_sha256=feature_hash,geometry=str(link.resolve()),refine=name=='refine',refinement_config=saved['config'],no_training_updates=True),indent=2))
  out=model/'association';status('associating',variant=name)
  if not (out/'cluster_lang.npz').exists():run(['associate_control_follow.py','-s',B/'data/lerf_ovs_sam2/ramen','-m',model,'--iteration','67999','--eval','--data_device','cpu','--association-score','iou','--no-pseudo-filter','--output',out],f'refinement-{name}-association.log')
  a=np.load(out/'cluster_lang.npz');ids=a['leaf_ind'].astype(int);valid=np.linalg.norm(a['leaf_feat'],axis=1)>1e-8
  for views in [1,5]:
   ev=model/f'eval_occ{views}';status('evaluating',variant=name,min_views=views)
   if not (ev/'metrics.json').exists():run(['experiments/evaluate_table1_3d.py','-s',B/'data/lerf_ovs/ramen','-m',B/'outputs/ramen_baseline','--eval','--data_device','cpu','--method','cluster','--checkpoint',B/'outputs/common/ramen_3dgs30000.pth','--mapping',out/'cluster_lang.npz','--min-association-views',str(views),'--output',ev],f'refinement-{name}-occ{views}-eval.log')
   results.append(dict(variant=name,clusters=stats(labels)['clusters'],nonzero_clip_coverage=float(valid[ids].mean()),classified_coverage=float((valid&(a['occu_count']>=views))[ids].mean()),**json.loads((ev/'metrics.json').read_text())))
   (root/'summary.json').write_text(json.dumps(results,indent=2))
 (root/'RESULTS.md').write_text('# Same-checkpoint refinement ablation\n\nBoth variants use identical 67999-step features and geometry; only labels change. No optimizer updates.\n\n| Variant | Min views | Clusters | Nonzero CLIP | Classified | mIoU | Acc@.25 | Acc@.5 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'+''.join(f"| {r['variant']} | {r['min_association_views']} | {r['clusters']} | {100*r['nonzero_clip_coverage']:.2f}% | {100*r['classified_coverage']:.2f}% | {100*r['mIoU']:.2f}% | {100*r['Acc@0.25']:.2f}% | {100*r['Acc@0.5']:.2f}% |\n" for r in results))
 status('complete',summary=str(root/'RESULTS.md'))
except Exception:
 status('failed',traceback=traceback.format_exc());raise
