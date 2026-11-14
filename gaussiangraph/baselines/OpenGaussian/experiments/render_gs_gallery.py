import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from arguments import ModelParams,PipelineParams
from scene import Scene,GaussianModel
from gaussian_renderer import render
p=ArgumentParser();m=ModelParams(p);pipe=PipelineParams(p);a=p.parse_args();out=Path('/media/joe/Data/gaussiangraph/experiments/gs_gallery');out.mkdir(exist_ok=True)
with torch.no_grad():
 g=GaussianModel(a.sh_degree);s=Scene(m.extract(a),g,load_iteration=30000,shuffle=False);views=s.getTestCameras();chosen=[views[i] for i in [0,len(views)//2,len(views)-1]]
 fig,axes=plt.subplots(3,3,figsize=(13,10));records=[]
 for row,v in enumerate(chosen):
  r=render(v,g,pipe.extract(a),torch.zeros(3,device='cuda'),30000,rescale=False,render_feat_map=False)
  rgb=r['render'].clamp(0,1).permute(1,2,0).cpu().numpy();gt=v.original_image[:3].permute(1,2,0).cpu().numpy();alpha=r['alpha'].squeeze().cpu().numpy();depth=r['depth'].squeeze().cpu().numpy();valid=(alpha>.7)&np.isfinite(depth);d=np.ma.masked_where(~valid,depth);lo,hi=np.percentile(depth[valid],[2,98])
  Image.fromarray((rgb*255).astype('uint8')).save(out/(v.image_name+'_3dgs.png'));np.save(out/(v.image_name+'_depth_raw.npy'),depth)
  axes[row,0].imshow(gt);axes[row,1].imshow(rgb);axes[row,2].imshow(d,cmap='magma',vmin=lo,vmax=hi)
  for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
  axes[row,0].set_ylabel(v.image_name)
  records.append(dict(frame=v.image_name,split='held-out test',checkpoint=30000))
  del r
 for ax,title in zip(axes[0],['Input photograph (held-out)','Our 3DGS RGB reconstruction','Rendered depth (relative colours)']):ax.set_title(title)
 fig.suptitle('Local ramen 3D Gaussian Splatting | 714,346 Gaussians',fontsize=16);fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(out/'ramen_3dgs.png',dpi=160);fig.savefig(out/'ramen_3dgs.pdf');plt.close(fig)
 (out/'sources.json').write_text(json.dumps(records,indent=2));print('Gallery complete')
