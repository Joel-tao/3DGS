from pathlib import Path
import numpy as np,torch,json
from PIL import Image
import open_clip
B=Path('/media/joe/Data/gaussiangraph');src=B/'data/lerf_ovs_sam2/ramen';dst=B/'data/lerf_ovs_sam2_aligned/ramen';dst.mkdir(parents=True,exist_ok=True);lf=dst/'language_features';lf.mkdir(exist_ok=True)
for name in ['images','sparse']:
 p=dst/name
 if not p.exists():p.symlink_to(src/name,target_is_directory=True)
model,_,preprocess=open_clip.create_model_and_transforms('ViT-B-16',pretrained='laion2b_s34b_b88k',device='cuda',precision='fp16',cache_dir=str(B/'checkpoints/clip'));model.eval()
with torch.inference_mode():
 for file in sorted((src/'language_features').glob('*_s.npy')):
  name=file.name[:-6]
  for suffix in ['_s.npy','_c.npy']:
   p=lf/(name+suffix)
   if not p.exists():p.symlink_to(src/'language_features'/(name+suffix))
  out=lf/(name+'_f.npy')
  if out.exists():continue
  seg=np.load(file,mmap_mode='r')[3];img=np.asarray(Image.open(src/'images'/(name+'.jpg')).convert('RGB'));n=len(np.load(src/'language_features'/(name+'_c.npy')));features=[]
  for start in range(0,n,16):
   tiles=[]
   for i in range(start,min(start+16,n)):
    mask=seg==i;y,x=np.where(mask)
    if len(x)==0:raise ValueError(f'Empty mask {name}:{i}')
    x0,x1,y0,y1=x.min(),x.max()+1,y.min(),y.max()+1
    crop=img[y0:y1,x0:x1].copy();crop[~mask[y0:y1,x0:x1]]=0;h,w=crop.shape[:2];side=max(h,w);square=np.zeros((side,side,3),np.uint8);square[(side-h)//2:(side-h)//2+h,(side-w)//2:(side-w)//2+w]=crop
    tiles.append(preprocess(Image.fromarray(square)))
   f=torch.nn.functional.normalize(model.encode_image(torch.stack(tiles).cuda().half()).float(),dim=1);features.append(f.cpu().numpy())
  result=np.concatenate(features);assert result.shape==(n,512) and np.isfinite(result).all();np.save(out,result);print(name,'aligned',n,flush=True)
(dst/'feature_source.json').write_text(json.dumps(dict(source=str(src),change='CLIP crop uses exactly the assigned level-3 training region; same mask IDs, confidence, images and geometry',clip='ViT-B-16 laion2b_s34b_b88k'),indent=2))
