"""Independent implementation of GaussianGraph III-B, not authors' complete code.

Unpublished choices (thresholds, refinement and update cadence) are explicit config.
Clustering uses detached features; quantization has a straight-through gradient.
"""
from dataclasses import dataclass,asdict
import numpy as np
import torch
from sklearn.cluster import Birch

@dataclass
class CFConfig:
    control_points: int = 5000
    voxel_size: float = .02
    radius: float = .1
    max_nn: int = 50
    cluster_space: str = "raw"
    feature_scale: float = 1.
    position_scale: float = .5
    branching_factor: int = 100
    control_threshold: float = 0.  # 0: distribution heuristic in public snippet
    follow_threshold: float = 0.   # 0: use control threshold
    refine_space: str = "raw"
    refine_min_points: int = 1
    refine_threshold: float = .2   # feature-only BIRCH; unspecified in paper

class GradientHistory:
    def __init__(self,n,device):
        self.total=torch.zeros(n,device=device);self.seen=torch.zeros(n,device=device,dtype=torch.long);self.steps=0
    @torch.no_grad()
    def update(self,gradient,visible):
        if gradient.shape[0]!=len(self.total):raise ValueError('Gaussian topology changed: remap gradient history first')
        self.total+=gradient.detach().norm(dim=1);self.seen+=visible.long();self.steps+=1
    def stable(self,epsilon):
        if self.steps==0:raise ValueError('No gradient history; cannot claim stable-point selection')
        return (self.total/self.steps<epsilon)&(self.seen>0)
    def remap(self,old_indices):
        """Map new order to old indices; -1 marks a new/split Gaussian without history."""
        valid=old_indices>=0
        if (old_indices[valid]>=len(self.total)).any():raise ValueError('Invalid topology mapping')
        total=self.total.new_zeros(len(old_indices));seen=self.seen.new_zeros(len(old_indices))
        total[valid]=self.total[old_indices[valid]];seen[valid]=self.seen[old_indices[valid]]
        self.total,self.seen=total,seen

class ControlFollow:
    def __init__(self,config=None):
        self.config=config or CFConfig();self.controls=None;self.labels=None;self.control_labels=None;self.threshold=None;self.phase='continuous'
    def vectors(self,xyz,features):
        values=features.detach()
        if self.config.cluster_space == "unit":
            values=torch.nn.functional.normalize(values,dim=1)
        elif self.config.cluster_space != "raw":
            raise ValueError("Unknown clustering feature space")
        return torch.cat((values*self.config.feature_scale,xyz.detach()*self.config.position_scale),1)
    def select_controls(self,xyz,stable):
        import open3d as o3d
        pool=torch.where(stable)[0].cpu().numpy()
        if len(pool)<2:raise ValueError('Insufficient observed stable points; inspect epsilon/window before proceeding')
        points=xyz.detach().cpu().numpy()[pool]
        cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
        voxel=self.config.voxel_size
        down,_,trace=cloud.voxel_down_sample_and_trace(voxel,cloud.get_min_bound()-voxel/2,cloud.get_max_bound()+voxel/2)
        down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.config.radius,max_nn=self.config.max_nn))
        fpfh=o3d.pipelines.registration.compute_fpfh_feature(down,o3d.geometry.KDTreeSearchParamHybrid(radius=self.config.radius,max_nn=self.config.max_nn))
        # Preserve the public snippet's FPFH-sum ranking. It is a heuristic,
        # not a separately specified keypoint detector in the paper.
        score=np.asarray(fpfh.data).sum(0);order=np.argsort(-score,kind='stable')[:self.config.control_points]
        centers=np.asarray(down.points);chosen=[]
        for index in order:
            original=np.asarray(trace[index],dtype=np.int64)
            local=original[np.argmin(((points[original]-centers[index])**2).sum(1))]
            chosen.append(pool[local])
        self.controls=torch.tensor(chosen,device=xyz.device,dtype=torch.long)
        return self.controls
    @torch.no_grad()
    def control(self,xyz,features):
        if self.controls is None:raise ValueError('Select stable controls first')
        vec=self.vectors(xyz,features)[self.controls].cpu().numpy()
        threshold=self.config.control_threshold or max(float(vec[:,:-3].std(0).mean()+vec[:,-3:].std(0).mean())/2,1e-6)
        result=Birch(threshold=threshold,branching_factor=self.config.branching_factor,n_clusters=None).fit_predict(vec)
        self.threshold=threshold;self.control_labels=torch.tensor(result,device=xyz.device,dtype=torch.long)
        self.labels=torch.full((len(xyz),),-1,device=xyz.device,dtype=torch.long);self.labels[self.controls]=self.control_labels;self.phase='control'
    @torch.no_grad()
    def follow(self,xyz,features):
        """Exact sequential assignment; update means after EACH point, create on demand."""
        vec=self.vectors(xyz,features).cpu().numpy().astype(np.float64)
        controls=self.controls.cpu().numpy();control_ids=self.control_labels.cpu().numpy();k=int(control_ids.max())+1
        centers=np.zeros((k,vec.shape[1]));count=np.bincount(control_ids,minlength=k).astype(np.int64)
        np.add.at(centers,control_ids,vec[controls]);centers/=count[:,None]
        labels=np.full(len(vec),-1,dtype=np.int64);labels[controls]=control_ids
        threshold=self.config.follow_threshold or self.threshold
        # Fixed input-index order makes this order-sensitive online process reproducible.
        for index in np.flatnonzero(labels<0):
            delta=centers-vec[index];distance=np.einsum('ij,ij->i',delta,delta);nearest=int(distance.argmin())
            if distance[nearest]>threshold**2:
                labels[index]=len(centers);centers=np.concatenate((centers,vec[index:index+1]));count=np.append(count,1)
            else:
                labels[index]=nearest;count[nearest]+=1;centers[nearest]+=(vec[index]-centers[nearest])/count[nearest]
        self.labels=torch.tensor(labels,device=xyz.device);self.phase='follow'
    @torch.no_grad()
    def refine(self,features):
        """Feature-only within-cluster BIRCH, with no predetermined object count."""
        if self.labels is None or (self.labels<0).any():raise ValueError('Follow must assign every point before refinement')
        values=features.detach().cpu().numpy()
        if self.config.refine_space == 'unit':
            values=values/np.maximum(np.linalg.norm(values,axis=1,keepdims=True),1e-12)
        elif self.config.refine_space != 'raw':raise ValueError('Unknown refinement feature space')
        old=self.labels.cpu().numpy();new=np.empty(len(old),np.int64);offset=0
        for group in np.unique(old):
            inds=np.flatnonzero(old==group);v=values[inds]
            if len(inds)<2*self.config.refine_min_points:
                new[inds]=offset;offset+=1;continue
            sub=Birch(threshold=self.config.refine_threshold,branching_factor=self.config.branching_factor,n_clusters=None).fit_predict(v)
            ids,counts=np.unique(sub,return_counts=True);supported=ids[counts>=self.config.refine_min_points]
            if len(supported)<2:
                new[inds]=offset;offset+=1;continue
            centers=np.stack([v[sub==label].mean(0) for label in supported])
            # Preserve supported child memberships. Attach small fragments to the
            # nearest supported child; do not create singleton semantic objects.
            mapped=np.full(len(inds),-1,np.int64)
            for child,label in enumerate(supported):mapped[sub==label]=child
            small=np.flatnonzero(mapped<0)
            for start in range(0,len(small),2048):
                take=small[start:start+2048];distance=((v[take,None,:]-centers[None,:,:])**2).sum(2);mapped[take]=distance.argmin(1)
            new[inds]=offset+mapped;offset+=len(supported)
        self.labels=torch.tensor(new,device=features.device);self.phase='refined'
    def quantize(self,features):
        if self.labels is None:return features
        if len(features)!=len(self.labels):raise ValueError('Gaussian topology changed; rebuild clustering before quantizing')
        with torch.no_grad():
            valid=self.labels>=0;k=int(self.labels[valid].max())+1
            centers=features.new_zeros((k,features.shape[1]));counts=features.new_zeros(k)
            centers.index_add_(0,self.labels[valid],features.detach()[valid]);counts.index_add_(0,self.labels[valid],torch.ones_like(self.labels[valid],dtype=features.dtype));centers/=counts.clamp_min(1)[:,None]
            quantized=features.detach().clone();quantized[valid]=centers[self.labels[valid]]
        return features+(quantized-features).detach()
    def state_dict(self):
        return dict(config=asdict(self.config),controls=self.controls,control_labels=self.control_labels,labels=self.labels,threshold=self.threshold,phase=self.phase)
