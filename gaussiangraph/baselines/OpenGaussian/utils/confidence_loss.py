"""Confidence-weighted GaussianGraph Eq.3/4 with explicit numerical conventions.

Conf is uniform per SAM2 mask. Eq.3 is divided by image pixel count to keep its
scale independent of resolution; Eq.4 uses the paper's m(m+1) denominator.
Squared distances are floored at 1e-6. These implementation choices are recorded.
"""
import torch

def confidence_losses(feature,masks,means,confidence,eps=1e-6):
    if confidence.ndim!=1 or len(confidence)!=len(masks):raise ValueError('Confidence/mask ID alignment mismatch')
    if not torch.isfinite(confidence).all() or (confidence<0).any() or (confidence>1).any():raise ValueError('Confidence must be finite and within [0,1]')
    valid=masks.flatten(1).any(1);masks=masks[valid];means=means[valid];confidence=confidence[valid]
    m=len(masks)
    if m==0:return feature.sum()*0,feature.sum()*0
    smooth=feature.sum()*0
    for start in range(0,m,2):
        delta=feature[None]-means[start:start+2,:,None,None]
        per_mask=(delta.square().sum(1)*masks[start:start+2]).sum((1,2))
        smooth=smooth+(per_mask*confidence[start:start+2]).sum()
    smooth=smooth/(feature.shape[-1]*feature.shape[-2])
    if m<2:return smooth,feature.sum()*0
    distance=(means[:,None,:]-means[None,:,:]).square().sum(2).clamp_min(eps)
    weight=(confidence[:,None]+confidence[None,:])/2
    off_diagonal=~torch.eye(m,device=feature.device,dtype=torch.bool)
    contrast=(weight[off_diagonal]/distance[off_diagonal]).sum()/(m*(m+1))
    return smooth,contrast
