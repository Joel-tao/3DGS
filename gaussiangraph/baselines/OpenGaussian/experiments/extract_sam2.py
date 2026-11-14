"""SAM2 Hiera-L automatic masks, quality metadata, and matched OpenCLIP features.

Stage masks and CLIP separately to bound VRAM on an 8GB GPU. Resume is per frame.
This uses original SAM2, not SAM2.1; confidence is an explicit independent choice.
"""
import sys,json,hashlib,gc,subprocess
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import torch
from PIL import Image
p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--sam2-source',type=Path,default=Path('/media/joe/Data/gaussiangraph/build/sam2'))
p.add_argument('--checkpoint',type=Path,default=Path('/media/joe/Data/gaussiangraph/checkpoints/sam2_hiera_large.pt'))
p.add_argument('--limit',type=int,default=0);p.add_argument('--points-per-side',type=int,default=32);p.add_argument('--points-per-batch',type=int,default=8)
p.add_argument('--cache-dir',type=Path,default=Path('/media/joe/Data/gaussiangraph/checkpoints/clip'))
a=p.parse_args();sys.path.insert(0,str(a.sam2_source))
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
files=sorted(f for f in (a.source/'images').iterdir() if f.suffix.lower() in ('.jpg','.jpeg','.png'))
if a.limit:files=files[:a.limit]
a.output.mkdir(parents=True,exist_ok=True);dest=a.output/'language_features';dest.mkdir(exist_ok=True)
for name in ('images','sparse'):
 target=a.output/name
 if not target.exists():target.symlink_to((a.source/name).resolve(),target_is_directory=True)
config={'model':'SAM2 Hiera-L (July 2024)','sam2_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.sam2_source,text=True).strip(),'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'points_per_side':a.points_per_side,'points_per_batch':a.points_per_batch,'pred_iou_thresh':.8,'stability_score_thresh':.95,'crop_n_layers':0,'clip':'ViT-B-16','clip_pretrained':'laion2b_s34b_b88k','confidence':'clip(predicted_iou,0,1)*clip(stability_score,0,1)','layout':'only compatibility level 3 active; levels 0-2 are -1'}
manifest=a.output/'feature_source.json'
if manifest.exists() and json.loads(manifest.read_text())!=config:raise ValueError('Preprocessing configuration changed; use a new output directory')
manifest.write_text(json.dumps(config,indent=2))
if any(not (dest/(f.stem+'_sam2.npz')).exists() for f in files):
 model=build_sam2('sam2_hiera_l.yaml',str(a.checkpoint),apply_postprocessing=False)
 generator=SAM2AutomaticMaskGenerator(model,points_per_side=a.points_per_side,points_per_batch=a.points_per_batch,pred_iou_thresh=.8,stability_score_thresh=.95,crop_n_layers=0,min_mask_region_area=0)
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for f in files:
   path=dest/(f.stem+'_sam2.npz')
   if path.exists():continue
   image=np.asarray(Image.open(f).convert('RGB')).copy();masks=generator.generate(image)
   masks=sorted((m for m in masks if m['area']>0),key=lambda m:-m['area'])
   if not masks:raise ValueError(f'No SAM2 masks for {f.name}')
   raw=np.stack([m['segmentation'] for m in masks]);n,h,w=raw.shape
   iou=np.array([m['predicted_iou'] for m in masks],np.float32);stability=np.array([m['stability_score'] for m in masks],np.float32)
   confidence=np.clip(iou,0,1)*np.clip(stability,0,1)
   np.savez_compressed(path,masks=np.packbits(raw.reshape(n,-1),axis=1),shape=np.array([n,h,w]),predicted_iou=iou,stability_score=stability,bbox=np.array([m['bbox'] for m in masks]))
   seg=np.full((4,h,w),-1,np.int32)
   for index,mask in enumerate(raw):seg[3,mask]=index
   np.save(dest/(f.stem+'_s.npy'),seg);np.save(dest/(f.stem+'_c.npy'),confidence)
   print('SAM2',f.name,'masks',n,'mean confidence',float(confidence.mean()),flush=True)
 del generator,model;gc.collect();torch.cuda.empty_cache()
import open_clip
clip,_,preprocess=open_clip.create_model_and_transforms('ViT-B-16',pretrained='laion2b_s34b_b88k',device='cuda',precision='fp16',cache_dir=str(a.cache_dir));clip.eval()
with torch.inference_mode():
 for f in files:
  path=dest/(f.stem+'_f.npy')
  if path.exists():continue
  image=np.asarray(Image.open(f).convert('RGB'));data=np.load(dest/(f.stem+'_sam2.npz'));n,h,w=data['shape'];raw=np.unpackbits(data['masks'],axis=1,count=int(h*w)).reshape(n,h,w).astype(bool)
  features=[]
  for start in range(0,n,16):
   tiles=[]
   for index in range(start,min(start+16,n)):
    x,y,bw,bh=data['bbox'][index].astype(int);crop=image[y:y+bh,x:x+bw].copy();crop[~raw[index,y:y+bh,x:x+bw]]=0
    length=max(bw,bh);square=np.zeros((length,length,3),np.uint8);square[(length-bh)//2:(length-bh)//2+bh,(length-bw)//2:(length-bw)//2+bw]=crop;tiles.append(preprocess(Image.fromarray(square)))
   embedded=clip.encode_image(torch.stack(tiles).cuda().half()).float();embedded=torch.nn.functional.normalize(embedded,dim=1);features.append(embedded.cpu().numpy())
  features=np.concatenate(features);assert features.shape==(n,512) and np.isfinite(features).all();np.save(path,features)
  print('CLIP',f.name,'features',features.shape,flush=True)
 queries=['nori','sake cup','kamaboko','corn','spoon','egg','onion segments','plate','napkin','bowl','glass of water','hand','chopsticks','wavy noodles']
 token=open_clip.get_tokenizer('ViT-B-16')(queries).cuda();encoded=torch.nn.functional.normalize(clip.encode_text(token).float(),dim=1).cpu().numpy()
 (a.output/'text_features.json').write_text(json.dumps(dict(zip(queries,encoded.tolist()))))
(a.output/'EXTRACTION_STATUS.json').write_text(json.dumps({'frames_requested':len(files),'frames_with_features':sum((dest/(f.stem+'_f.npy')).exists() for f in files),'limited_smoke':bool(a.limit)},indent=2))
print('Extraction complete',len(files),'frames',flush=True)
