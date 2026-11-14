"""Run the isolated CF experiment to completion; persist phase and failures."""
import json,subprocess,sys,time,tarfile
from pathlib import Path
repo=Path(__file__).resolve().parents[1];root=Path('/media/joe/Data/gaussiangraph');out=root/'outputs/ramen_control_follow_v1';out.mkdir(parents=True,exist_ok=True)
python=sys.executable
common=['-s',str(root/'data/lerf_ovs/ramen'),'-m',str(out),'--eval','--data_device','cpu']
commands=[('training',[python,'-u','train_control_follow.py',*common,'--base-model',str(root/'outputs/ramen_baseline')]),('association',[python,'-u','associate_control_follow.py',*common,'--association-score','opengs']),('evaluation',[python,'-u','experiments/evaluate_top1.py',*common,'--output',str(root/'experiments/control_follow_top1')])]
status_file=root/'logs/ramen-cf-status.json'
def status(phase,**extra):
 record=dict(phase=phase,time_unix=time.time(),output=str(out),**extra);temporary=status_file.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2));temporary.replace(status_file);print(record,flush=True)
(out/'pipeline_commands.json').write_text(json.dumps(commands,indent=2))
subprocess.run([python,'-m','pip','freeze'],stdout=(out/'environment.txt').open('w'),check=True)
(out/'upstream_commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True))
(out/'changes.patch').write_text(subprocess.check_output(['git','diff'],cwd=repo,text=True))
files=subprocess.check_output(['git','ls-files','--modified','--others','--exclude-standard'],cwd=repo,text=True).splitlines()
with tarfile.open(out/'source.tar.gz','w:gz') as arc:
 for f in files:arc.add(repo/f,arcname=f)
try:
 for phase,cmd in commands:
  status(phase)
  with (root/f'logs/ramen-cf-{phase}.log').open('w') as log:
   child=subprocess.Popen(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT);status(phase,pid=child.pid)
   if child.wait()!=0:raise RuntimeError(f'{phase} failed; inspect ramen-cf-{phase}.log')
 result=json.loads((root/'experiments/control_follow_top1/metrics.json').read_text())
 baseline=json.loads((root/'experiments/baseline_top1/metrics.json').read_text())
 comparison={'protocol':'top1_fixed_v1','baseline':baseline,'control_follow':result,'note':'Independent CF implementation, same third-party SAM/CLIP; no SAM2/confidence weighting yet'}
 (root/'experiments/CF_COMPARISON.json').write_text(json.dumps(comparison,indent=2));status('complete',metrics=result)
except Exception as error:
 status('failed',error=str(error));raise
