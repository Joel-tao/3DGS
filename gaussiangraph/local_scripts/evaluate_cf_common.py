import subprocess,pathlib
b=pathlib.Path('/media/joe/Data/gaussiangraph')
for model,name in [('ramen_control_follow_v2','cf_3d_v1'),('ramen_sam2_confidence_v1','sam2_confidence_cf_3d_v1')]:
 cmd=[b/'envs/opengs/bin/python','-u','experiments/evaluate_table1_3d.py','-s',b/'data/lerf_ovs/ramen','-m',b/'outputs/ramen_baseline','--eval','--data_device','cpu','--method','cluster','--checkpoint',b/'outputs/common/ramen_3dgs30000.pth','--mapping',b/f'outputs/{model}/cluster_lang.npz','--output',b/f'experiments/{name}']
 with (b/f'logs/{name}-eval.log').open('w') as f:subprocess.run(list(map(str,cmd)),cwd='/home/joe/gaussiangraph/baselines/OpenGaussian',stdout=f,stderr=subprocess.STDOUT,check=True)
