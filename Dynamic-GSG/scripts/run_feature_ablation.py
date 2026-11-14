"""Run matched, serial local feature-loss ablations on three Replica scenes."""
import argparse,json,os,pprint,re,subprocess,time,hashlib
from pathlib import Path
from runpy import run_path
from copy import deepcopy
import psutil

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path('/media/joe/Data/Dynamic-GSG/experiments/feature_ablation_20261017'))
parser.add_argument('--frames',type=int,default=200)
args=parser.parse_args()
args.root.mkdir(parents=True,exist_ok=True)
base=run_path(str(ROOT/'configs/replica/dgsg_local.py'))['config']
base['record_run_metrics']=True
base['save_checkpoints']=False
base['data']['num_frames']=args.frames
base['workdir']=str(args.root)
# Fixed protocol; no language model or DAM is used in any run.
plan=[]
for scene in ['room0','room1','office0']:
    for variant,weight in [('feature_on',.5),('feature_off',0.)]:
        cfg=deepcopy(base);cfg['data']['sequence']=scene
        cfg['run_name']=f'{scene}_{variant}'
        cfg['mapping']['loss_weights']['features']=weight
        for key,name in [('variables_path','variables.npz'),('keyframe_list_path','keyframelist.pkl.gz')]:cfg['viz'][key]=str(args.root/cfg['run_name']/name)
        result=args.root/cfg['run_name']
        if result.exists():raise SystemExit(f'Refusing to overwrite experiment {result}')
        path=args.root/f'{cfg["run_name"]}.py';path.write_text('config = '+pprint.pformat(cfg,sort_dicts=False)+'\n')
        plan.append((cfg,path))
for i in range(0,len(plan),2):
    a,b=deepcopy(plan[i][0]),deepcopy(plan[i+1][0])
    for cfg in [a,b]:
        cfg.pop('run_name');cfg.pop('viz');cfg['mapping']['loss_weights'].pop('features')
    assert a==b,'Unmatched ablation settings'
manifest=dict(protocol='3 scenes x feature weight .5 / 0; single seed 0; 200 sampled frames per scene; no ground-truth semantic or relation metrics',configs=[c for c,_ in plan],script_sha256=hashlib.sha256((ROOT/'scripts/dynamic_gsg_real_ssim.py').read_bytes()).hexdigest(),note='Zero feature weight still computes the auxiliary feature render, isolating objective changes rather than optimizing runtime.')
(args.root/'manifest.json').write_text(json.dumps(manifest,indent=2))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HOME='/media/joe/Data/Dynamic-GSG/cache/huggingface',TORCH_HOME='/media/joe/Data/Dynamic-GSG/cache/torch',MPLBACKEND='Agg',TOKENIZERS_PARALLELISM='false')
records=[]
for cfg,path in plan:
    name=cfg['run_name'];log=args.root/f'{name}.log'
    print('START',name,flush=True);start=time.perf_counter();peak_rss=0
    with log.open('w') as output:
        proc=subprocess.Popen(['/home/joe/anaconda3/envs/dgsg/bin/python','-u',str(ROOT/'scripts/dynamic_gsg_real_ssim.py'),str(path)],cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            try:peak_rss=max(peak_rss,psutil.Process(proc.pid).memory_info().rss)
            except psutil.NoSuchProcess:pass
            text=log.read_text(errors='replace');frames=re.findall(r'frame (\d+) num of objects: (\d+)',text)
            status=dict(run=name,pid=proc.pid,elapsed_seconds=time.perf_counter()-start,latest_frame=int(frames[-1][0]) if frames else None,completed=records)
            tmp=args.root/'status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(args.root/'status.json')
            time.sleep(2)
    record=dict(run=name,scene=cfg['data']['sequence'],feature_weight=cfg['mapping']['loss_weights']['features'],returncode=proc.returncode,end_to_end_seconds=time.perf_counter()-start,peak_rss_bytes_sampled=peak_rss)
    records.append(record);(args.root/'batch_results.json').write_text(json.dumps(records,indent=2))
    print('END',json.dumps(record),flush=True)
(args.root/'status.json').write_text(json.dumps(dict(finished=True,completed=records),indent=2))
if any(x['returncode'] for x in records):raise SystemExit('Some experiments failed; inspect per-run logs.')
