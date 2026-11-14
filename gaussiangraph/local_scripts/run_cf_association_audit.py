from pathlib import Path
import subprocess,json,datetime,traceback,numpy as np
B=Path('/media/joe/Data/gaussiangraph');R=Path('/home/joe/gaussiangraph/baselines/OpenGaussian');base=B/'experiments/association_audit_v1';base.mkdir(exist_ok=True)
status=B/'logs/cf-association-audit-status.json'
def update(state,**kw):status.write_text(json.dumps(dict(time=datetime.datetime.now().isoformat(),state=state,**kw),indent=2))
def run(cmd,log):
 with (B/'logs'/log).open('a') as f:subprocess.run(list(map(str,cmd)),cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
results=[]
try:
 for model,source in [('cf','ramen_control_follow_v2'),('sam2_cf','ramen_sam2_confidence_v1')]:
  data=B/('data/lerf_ovs_sam2/ramen' if model=='sam2_cf' else 'data/lerf_ovs/ramen')
  for filtered in [True,False]:
   name=model+('_iou_filtered' if filtered else '_iou_unfiltered');out=base/name
   update('associating',experiment=name)
   if not (out/'cluster_lang.npz').exists():
    cmd=[B/'envs/opengs/bin/python','-u','associate_control_follow.py','-s',data,'-m',B/'outputs'/source,'--eval','--data_device','cpu','--association-score','iou','--output',out]
    if not filtered:cmd+=['--no-pseudo-filter']
    run(cmd,name+'-association.log')
   config=dict(source_model=source,mask_source=str(data),association_score='iou',pseudo_filter=filtered,alpha_threshold=.8,match_threshold=.2,geometry_and_labels_frozen=True)
   (out/'config.json').write_text(json.dumps(config,indent=2))
   a=np.load(out/'cluster_lang.npz');ids=a['leaf_ind'].astype(int);nonzero=np.linalg.norm(a['leaf_feat'],axis=1)>1e-8
   coverage={str(v):float((nonzero&(a['occu_count']>=v))[ids].mean()) for v in [1,5]}
   for views in [1,5]:
    ev=out/f'eval_occ{views}';update('evaluating',experiment=name,min_views=views)
    if not (ev/'metrics.json').exists():
     run([B/'envs/opengs/bin/python','-u','experiments/evaluate_table1_3d.py','-s',B/'data/lerf_ovs/ramen','-m',B/'outputs/ramen_baseline','--eval','--data_device','cpu','--method','cluster','--checkpoint',B/'outputs/common/ramen_3dgs30000.pth','--mapping',out/'cluster_lang.npz','--min-association-views',str(views),'--output',ev],name+f'-occ{views}-eval.log')
    results.append(dict(experiment=name,min_views=views,coverage=coverage[str(views)],**json.loads((ev/'metrics.json').read_text())))
   (base/'summary.json').write_text(json.dumps(results,indent=2))
 (base/'RESULTS.md').write_text('# Frozen CF association audit\n\nOnly IoU association, pseudo filtering and view gates vary; no training or GT selection.\n\n| Experiment | Min views | Coverage | mIoU | Acc@.25 | Acc@.5 |\n|---|---:|---:|---:|---:|---:|\n'+''.join(f"| {r['experiment']} | {r['min_views']} | {100*r['coverage']:.2f}% | {100*r['mIoU']:.2f}% | {100*r['Acc@0.25']:.2f}% | {100*r['Acc@0.5']:.2f}% |\n" for r in results))
 update('complete',summary=str(base/'RESULTS.md'))
except Exception:
 update('failed',traceback=traceback.format_exc());raise
