import subprocess, pathlib, json, datetime, traceback
B=pathlib.Path('/media/joe/Data/gaussiangraph')
status=B/'logs/langsplat-ramen-status.json'
def update(**kw):
    status.write_text(json.dumps(dict(time=datetime.datetime.now().isoformat(),**kw),indent=2))
def run(cmd,cwd,log):
    with (B/'logs'/log).open('a') as f:
        subprocess.run([str(x) for x in cmd],cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
    for level in [1,2,3]:
        update(state='training',level=level)
        ckpt=B/f'outputs/langsplat_ramen_{level}/chkpnt30000.pth'
        if not ckpt.exists():
            run([B/'envs/langsplat/bin/python','-u','train.py','-s',B/'data/langsplat/ramen','-m',B/'outputs/langsplat_ramen','--start_checkpoint',B/'outputs/common/ramen_3dgs30000.pth','--feature_level',str(level),'--iterations','30000','--save_iterations','30000','--checkpoint_iterations','30000','--test_iterations','30000','--data_device','cpu','--eval','--port','6021'],B/'build/LangSplat',f'langsplat-ramen-level{level}-train.log')
        update(state='evaluating',level=level)
        out=B/f'experiments/langsplat_3d_level{level}'
        if not (out/'metrics.json').exists():
            run([B/'envs/opengs/bin/python','-u','experiments/evaluate_table1_3d.py','-s',B/'data/lerf_ovs/ramen','-m',B/'outputs/ramen_baseline','--eval','--data_device','cpu','--method','langsplat','--checkpoint',ckpt,'--decoder',B/'outputs/langsplat_ae/ramen/best_ckpt.pth','--model-source',B/'build/LangSplat','--output',out],'/home/joe/gaussiangraph/baselines/OpenGaussian',f'langsplat-ramen-level{level}-eval.log')
    update(state='levels_complete',levels=[1,2,3])
except Exception:
    update(state='failed',traceback=traceback.format_exc())
    raise
