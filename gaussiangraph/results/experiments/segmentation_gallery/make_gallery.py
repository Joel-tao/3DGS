from pathlib import Path
import json,csv,sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0,'/home/joe/gaussiangraph/baselines/OpenGaussian')
from scripts.compute_lerf_iou import load_image_as_binary,calculate_iou
B=Path('/media/joe/Data/gaussiangraph');E=B/'experiments';out=E/'segmentation_gallery';gtroot=B/'data/lerf_ovs/label/ramen/gt'
methods=[('LangSplat',E/'langsplat_3d_multilevel_v1'),('LEGaussian',E/'legaussian_3d_v1'),('OpenGaussian',E/'opengs_3d_v1'),('SAM2+CF (70k)',E/'association_audit_v1/sam2_cf_iou_unfiltered/eval_occ5'),('No refine (67999)',E/'refinement_ablation_v1/no_refine/eval_occ5')]
rows=[]
def draw(frame,queries):
 image=np.asarray(Image.open(B/'data/lerf_ovs/ramen/images'/(frame+'.jpg')).convert('RGB'))
 fig,axes=plt.subplots(len(queries),7,figsize=(19.6,2.4*len(queries)+.9),squeeze=False)
 for i,q in enumerate(queries):
  gt=load_image_as_binary(str(gtroot/frame/(q+'.jpg'))).astype(bool)
  for j,(name,folder) in enumerate([('RGB',None),('GT',None)]+methods):
   ax=axes[i,j];ax.imshow(image);ax.set_xticks([]);ax.set_yticks([])
   if j:
    mask=gt if j==1 else load_image_as_binary(str(folder/(frame+'_'+q+'.png')),is_png=True).astype(bool)
    assert mask.shape==image.shape[:2]
    overlay=np.zeros((*mask.shape,4));overlay[mask]=[.05,.85,.45,.55];ax.imshow(overlay)
    if j>1:
     v=calculate_iou(gt,mask);ax.set_xlabel(f'IoU {100*v:.1f}%',fontsize=10)
   if i==0:ax.set_title(name,fontsize=11)
   if j==0:ax.set_ylabel(q,fontsize=11)
 fig.suptitle(f'LOCAL ramen segmentation | {frame} | green = selected region',fontsize=15)
 fig.text(.5,.008,'Actual saved masks. CF: pure IoU, no pseudo filter, >=5 views. Checkpoints differ; see sources.json. Empty prediction = no green.',ha='center',fontsize=9)
 fig.tight_layout(rect=[0,.03,1,.96]);return fig
frame='frame_00006';queries=['bowl','chopsticks','egg','nori']
fig=draw(frame,queries);fig.savefig(out/'preview.png',dpi=140);plt.close(fig)
with PdfPages(out/'all_queries.pdf') as pdf:
 for d in sorted(gtroot.iterdir()):
  if not d.is_dir():continue
  queries=sorted(p.stem for p in d.glob('*.jpg'))
  for start in range(0,len(queries),4):
   fig=draw(d.name,queries[start:start+4]);pdf.savefig(fig);plt.close(fig)
  for q in queries:
   gt=load_image_as_binary(str(d/(q+'.jpg')))
   for name,folder in methods:
    path=folder/(d.name+'_'+q+'.png');mask=load_image_as_binary(str(path),is_png=True);rows.append(dict(frame=d.name,query=q,method=name,IoU=calculate_iou(gt,mask),prediction=str(path)))
with (out/'per_query_iou.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
(out/'sources.json').write_text(json.dumps(dict(gt=str(gtroot),methods={n:str(p) for n,p in methods},preview_selection='fixed frame_00006, bowl/chopsticks/egg/nori; no sorting by performance',all_queries=len(rows)//len(methods)),indent=2))
print('Completed',len(rows)//len(methods),'GT query-view cases',flush=True)
