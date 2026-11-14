"""Sequential GPU queue: prepare SAM2, evaluate repaired CF, train SAM2+confidence."""
import sys,json,subprocess,time,tarfile
from pathlib import Path
repo=Path(__file__).resolve().parents[1];root=Path('/media/joe/Data/gaussiangraph');python=sys.executable
status_file=root/'logs/ramen-sam2-status.json';base=root/'outputs/ramen_baseline';v2=root/'outputs/ramen_control_follow_v2';source=root/'data/lerf_ovs_sam2/ramen';out=root/'outputs/ramen_sam2_confidence_v1'
common=lambda model,data:['-s',str(data),'-m',str(model),'--eval','--data_device','cpu']
commands=[
 ('sam2_extraction',[python,'-u','experiments/extract_sam2.py','--source',str(root/'data/lerf_ovs/ramen'),'--output',str(source)]),
 ('confidence_smoke',[python,'-u','train_control_follow.py',*common(root/'outputs/ramen_sam2_confidence_smoke',source),'--base-model',str(base),'--confidence-dir',str(source/'language_features'),'--control-start','30008','--follow-start','30012','--refine-start','30016','--iterations','30018','--history-window','4','--control-points','200']),
 ('cf_v2_association',[python,'-u','associate_control_follow.py',*common(v2,root/'data/lerf_ovs/ramen')]),
 ('cf_v2_evaluation',[python,'-u','experiments/evaluate_top1.py',*common(v2,root/'data/lerf_ovs/ramen'),'--output',str(root/'experiments/control_follow_v2_top1')]),
 ('sam2_confidence_training',[python,'-u','train_control_follow.py',*common(out,source),'--base-model',str(base),'--confidence-dir',str(source/'language_features')]),
 ('sam2_confidence_association',[python,'-u','associate_control_follow.py',*common(out,source),'--association-score','iou']),
 ('sam2_confidence_evaluation',[python,'-u','experiments/evaluate_top1.py',*common(out,source),'--output',str(root/'experiments/sam2_confidence_top1')])]
def update(phase,**kwargs):
 r=dict(phase=phase,time_unix=time.time(),**kwargs);tmp=status_file.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2));tmp.replace(status_file);print(r,flush=True)
manifest=root/'experiments/SAM2_CONFIDENCE_PIPELINE.json';manifest.write_text(json.dumps(commands,indent=2))
with tarfile.open(root/'experiments/sam2_confidence_source.tar.gz','w:gz') as arc:
 for name in subprocess.check_output(['git','ls-files','--modified','--others','--exclude-standard'],cwd=repo,text=True).splitlines():arc.add(repo/name,arcname=name)
try:
 for phase,cmd in commands:
  update(phase)
  with (root/f'logs/ramen-{phase}.log').open('w') as log:
   child=subprocess.Popen(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT);update(phase,pid=child.pid)
   if child.wait()!=0:raise RuntimeError(f'{phase} failed; see ramen-{phase}.log')
  if phase=='sam2_extraction':
   import numpy as np
   frames=list((source/'images').glob('*.jpg'))
   if len(frames)!=131:raise ValueError('Unexpected image count')
   for f in frames:
    confidence=np.load(source/'language_features'/(f.stem+'_c.npy'));features=np.load(source/'language_features'/(f.stem+'_f.npy'))
    if confidence.shape!=(len(features),) or not np.isfinite(confidence).all() or (confidence<0).any() or (confidence>1).any():raise ValueError('SAM2 confidence alignment failed')
  if phase=='cf_v2_evaluation':
   result=json.loads((root/'experiments/control_follow_v2_top1/metrics.json').read_text());(root/'logs/ramen-cf-v2-status.json').write_text(json.dumps({'phase':'complete','metrics':result},indent=2))
 result={name:json.loads((root/path).read_text()) for name,path in [('baseline','experiments/baseline_top1/metrics.json'),('cf_v2','experiments/control_follow_v2_top1/metrics.json'),('sam2_confidence','experiments/sam2_confidence_top1/metrics.json')]}
 (root/'experiments/SAM2_CF_COMPARISON.json').write_text(json.dumps(result,indent=2));update('complete',metrics=result['sam2_confidence'])
except Exception as e:
 update('failed',error=str(e));raise
