from pathlib import Path
import subprocess
b=Path('/media/joe/Data/gaussiangraph')
for model in ['ramen_baseline','ramen_control_follow_v2','ramen_sam2_confidence_v1']:
 ckpt=b/'outputs'/model/'chkpnt70000.pth' if model=='ramen_baseline' else b/'outputs/common/ramen_3dgs30000.pth'
 cmd=[b/'envs/opengs/bin/python','-u','experiments/evaluate_table1_3d.py','-s',b/'data/lerf_ovs/ramen','-m',b/'outputs/ramen_baseline','--eval','--data_device','cpu','--method','cluster','--checkpoint',ckpt,'--mapping',b/'outputs'/model/'cluster_lang.npz','--min-association-views','1','--output',b/'experiments/alignment_audit'/f'{model}_occ1']
 with (b/'logs'/f'audit-{model}-occ1.log').open('w') as f:subprocess.run(list(map(str,cmd)),cwd='/home/joe/gaussiangraph/baselines/OpenGaussian',stdout=f,stderr=subprocess.STDOUT,check=True)
