"""Render paired ablation views and plot metrics without requiring a display."""
import argparse,json,sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizer
from utils.recon_helpers import setup_camera
from utils.slam_helpers import transform_to_frame,transformed_params2rendervar

p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
for scene in ['room0','room1','office0']:
    images={}
    for variant in ['feature_on','feature_off']:
        path=a.root/f'{scene}_{variant}'/'params_with_idx.npz'
        if not path.exists():continue
        with np.load(path) as d:
            params={k:torch.as_tensor(d[k],device='cuda',dtype=torch.float32) for k in ['means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales','cam_unnorm_rots','cam_trans']}
            cam=setup_camera(int(d['org_width']),int(d['org_height']),d['intrinsics'],d['w2c'])
            ids=d['object_idx'].ravel().astype(int)
        colors=np.random.default_rng(0).uniform(.15,1,(int(ids.max())+1,3));colors[0]=[.5,.5,.5]
        colors=torch.as_tensor(colors[ids],dtype=torch.float32,device='cuda')
        n=params['cam_trans'].shape[-1];frames=[0,n//2,n-1]
        with torch.no_grad():
            for frame in frames:
                transformed=transform_to_frame(params,frame,False,False)
                variables=transformed_params2rendervar(params,transformed)
                for mode in ['RGB','instances']:
                    variables['colors_precomp']=params['rgb_colors'] if mode=='RGB' else colors
                    image,_,_=GaussianRasterizer(raster_settings=cam)(**variables)
                    images[variant,frame,mode]=image.clamp(0,1).permute(1,2,0).cpu().numpy()
        del params,cam,colors,variables,transformed,image;torch.cuda.empty_cache()
    if len(images)!=12:continue
    fig,axes=plt.subplots(3,4,figsize=(18,9))
    for row,frame in enumerate(frames):
        for col,(variant,mode) in enumerate([('feature_on','RGB'),('feature_off','RGB'),('feature_on','instances'),('feature_off','instances')]):
            axes[row,col].imshow(images[variant,frame,mode]);axes[row,col].axis('off')
            axes[row,col].set_title(f'{variant} / {mode} / frame {frame}',fontsize=10)
    fig.suptitle(f'{scene}: feature loss ablation (instance colors are per-run IDs, not matched identities)',fontsize=13)
    fig.tight_layout();fig.savefig(a.root/f'{scene}_paired_views.png',dpi=150);plt.close(fig)
rows=json.loads((a.root/'summary.json').read_text())
fig,axes=plt.subplots(2,3,figsize=(15,8))
for ax,(metric,label) in zip(axes.flat,[('psnr','PSNR (dB), higher better'),('lpips','LPIPS, lower better'),('l1','Depth L1 (m), lower better'),('live_instances','Predicted instance count (not accuracy)'),('end_to_end_seconds','Total wall time (s)'),('peak_reserved_gib','PyTorch peak reserved memory (GiB)')]):
    for offset,(weight,color,name) in enumerate([(.5,'#187b94','feature on'),(0.,'#c27b33','feature off')]):
        vals=[next((r.get(metric,np.nan) for r in rows if r['scene']==scene and r['feature_weight']==weight and r['status']=='complete'),np.nan) for scene in ['room0','room1','office0']]
        ax.bar(np.arange(3)+(offset-.5)*.35,vals,width=.35,color=color,label=name)
    ax.set_xticks(np.arange(3),['room0','room1','office0']);ax.set_title(label,fontsize=10);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Local single-seed ablation; sampled training-view metrics; not the paper benchmark')
fig.tight_layout();fig.savefig(a.root/'comparison_metrics.png',dpi=160)
print('Saved paired scene renders and comparison_metrics.png')
