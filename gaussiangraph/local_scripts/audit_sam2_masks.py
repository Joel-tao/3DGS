from pathlib import Path
import numpy as np,json
b=Path('/media/joe/Data/gaussiangraph');root=b/'data/lerf_ovs_sam2/ramen/language_features';rows=[]
for p in sorted(root.glob('*_sam2.npz')):
 name=p.name.removesuffix('_sam2.npz');a=np.load(p);n,h,w=map(int,a['shape']);seg=np.load(root/(name+'_s.npy'),mmap_mode='r')[3];counts=np.bincount(seg[seg>=0],minlength=n);raw=np.unpackbits(a['masks'],axis=1,count=h*w).sum(1);ratio=counts/np.maximum(raw,1)
 f=np.load(root/(name+'_f.npy'));c=np.load(root/(name+'_c.npy'))
 assert len(f)==n==len(c) and seg.max()<n
 rows.append(dict(frame=name,masks=n,empty=int((counts==0).sum()),retained_under_half=int((ratio<.5).sum()),retained_fraction_median=float(np.median(ratio)),uncovered_fraction=float((seg<0).mean()),confidence_mean=float(c.mean()),clip_norm_mean=float(np.linalg.norm(f,axis=1).mean())))
summary=dict(frames=len(rows),masks=sum(r['masks'] for r in rows),empty=sum(r['empty'] for r in rows),retained_under_half=sum(r['retained_under_half'] for r in rows),mean_uncovered_fraction=float(np.mean([r['uncovered_fraction'] for r in rows])),mean_confidence=float(np.mean([r['confidence_mean'] for r in rows])))
out=b/'experiments/alignment_audit/sam2_mask_layout.json';out.write_text(json.dumps(dict(summary=summary,frames=rows),indent=2));print(json.dumps(summary,indent=2))
