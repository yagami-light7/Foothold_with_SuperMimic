"""Static terrain queries for privileged dynamics; no reference/Actor dependency.

The provider is queried from each predicted state, rather than cached at rollout
start. Heights are relative to a valid surface under a current foot (fallback
base surface). Invalid samples have zero *storage* height and valid_mask=0.
Discrete support membership/argmax is piecewise constant; edge distances and
bilinear height maps retain gradients in predicted XY. Flat height has zero XY
gradient by definition. No gradient is invented through a discontinuous edge.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .rotations import yaw_mat


@dataclass
class TerrainFeatures:
    relative_heights: torch.Tensor  # [B, anchors, P*P], metres
    valid_mask: torch.Tensor        # same shape, bool
    edge_features: torch.Tensor     # same shape, signed distance/patch width, clipped

    def flatten(self):
        return torch.cat((self.relative_heights.flatten(1),
                          self.valid_mask.to(self.relative_heights.dtype).flatten(1),
                          self.edge_features.flatten(1)), -1)


class TerrainFeatureProvider:
    def __init__(self, kin, *, patch_points=5, patch_width=.4):
        if not isinstance(patch_points,int) or patch_points<1 or patch_width<=0:
            raise ValueError('Invalid terrain patch geometry')
        self.kin, self.patch_points, self.patch_width = kin, patch_points, patch_width
        from .roban import FOOT_BODY_NAMES, foot_geometry
        self.foot_ids = [kin.tracked_names.index(n) for n in FOOT_BODY_NAMES]
        self.offsets = kin.q_lo.new_tensor([f['sole'] for f in foot_geometry()])
        x=torch.linspace(-patch_width/2,patch_width/2,patch_points,device=kin.q_lo.device,dtype=kin.q_lo.dtype)
        grid=torch.meshgrid(x,x,indexing='ij')
        self.grid=torch.stack((grid[0].flatten(),grid[1].flatten(),torch.zeros_like(grid[0]).flatten()),-1)
        self.dim = 3 * 3 * patch_points**2

    def surface(self, xy):
        raise NotImplementedError

    def features(self, S):
        bp,bR,_,_=self.kin.forward(*S)
        feet=bp[:,self.foot_ids]+(bR[:,self.foot_ids]@self.offsets[...,None]).squeeze(-1)
        centers=torch.cat((feet,S[0][:,None]),1)
        H=yaw_mat(S[1])
        grid=(H[:,None]@self.grid[None,...,None]).squeeze(-1)
        points=centers[:,:,None]+grid[:,None]
        h,valid,edges=self.surface(points[...,:2])
        ch,cv,_=self.surface(centers[...,:2])
        # Select the foot closest to a valid surface (the current support), then
        # base. A projected swing foot above a taller column is not the support.
        distance=torch.where(cv[:,:2],(centers[:,:2,2]-ch[:,:2]).abs(),torch.inf)
        first=torch.where(cv[:,:2].any(-1),distance.argmin(-1),torch.full_like(cv[:,0],2,dtype=torch.long))
        support=ch.gather(1,first[:,None])
        support=torch.where(cv.any(-1,keepdim=True),support,torch.zeros_like(support))
        relative=torch.where(valid,h-support[...,None],torch.zeros_like(h))
        return TerrainFeatures(relative,valid,(edges/self.patch_width).clamp(-1,1))

    def __call__(self, S):
        return self.features(S).flatten()

    def contract(self):
        return dict(provider=type(self).__name__,patch_points=self.patch_points,patch_width=self.patch_width,
                    anchors=['left_sole','right_sole','base'],dim=self.dim,
                    gradients='piecewise XY; validity/support selection discrete; no straight-through edge gradients')


class FlatTerrainProvider(TerrainFeatureProvider):
    def surface(self, xy):
        h=xy[...,0]*0  # Keeps a valid (zero) derivative with respect to predicted position.
        return h,torch.ones_like(h,dtype=torch.bool),torch.full_like(h,self.patch_width)


class PillarTerrainProvider(TerrainFeatureProvider):
    """Axis-aligned static tops [cx,cy,hx,hy,height], optional physical base plane."""
    def __init__(self,kin,pillars,*,base_height=None,**kwargs):
        super().__init__(kin,**kwargs)
        self.pillars=kin.q_lo.new_tensor(pillars)
        if self.pillars.ndim!=2 or self.pillars.shape[1]!=5 or len(self.pillars)==0 or (self.pillars[:,2:4]<=0).any():
            raise ValueError('Pillar tops need [N,5] positive half widths')
        self.base_height=base_height

    def surface(self,xy):
        margin=self.pillars[:,2:4]-(xy[...,None,:]-self.pillars[:,:2]).abs()
        distance=margin.min(-1).values
        inside=distance>=0
        heights=torch.where(inside,self.pillars[:,4],-torch.inf)
        height,index=heights.max(-1)
        valid=inside.any(-1)
        edge=distance.max(-1).values
        if self.base_height is not None:
            height=torch.maximum(height,torch.full_like(height,self.base_height))
            valid=torch.ones_like(valid)
        return torch.where(valid,height,torch.zeros_like(height)),valid,edge

    def contract(self):
        return dict(**super().contract(),pillars=self.pillars.cpu().tolist(),base_height=self.base_height)


class HeightMapTerrainProvider(TerrainFeatureProvider):
    """Stair/mesh adapter: bilinear sampled world XY grid; mask explicitly marks void.

    grid shape [ny,nx], spacing in metres, origin is the first cell center.
    Values across a stair riser are interpolated; use a fine grid and account for
    this smoothing when matching mesh contacts. No moving surfaces are modeled.
    """
    def __init__(self,kin,heights,valid_mask,origin,spacing,**kwargs):
        super().__init__(kin,**kwargs)
        self.heights=kin.q_lo.new_tensor(heights)
        self.valid=kin.q_lo.new_tensor(valid_mask)
        self.origin=kin.q_lo.new_tensor(origin)
        self.spacing=float(spacing)
        if self.heights.ndim!=2 or min(self.heights.shape)<2 or self.valid.shape!=self.heights.shape or spacing<=0 or not torch.isfinite(self.heights).all() or not ((self.valid==0)|(self.valid==1)).all():
            raise ValueError('Invalid terrain height map/mask')

    def surface(self,xy):
        scale=xy.new_tensor([self.heights.shape[1]-1,self.heights.shape[0]-1])*self.spacing
        grid=2*(xy-self.origin)/scale-1
        shape=xy.shape[:-1]
        samples=grid.reshape(1,1,-1,2)
        get=lambda x,mode: F.grid_sample(x[None,None],samples,mode=mode,padding_mode='zeros',align_corners=True).reshape(shape)
        valid=(get(self.valid,'bilinear')>1-1e-6)&(grid.abs()<=1).all(-1)
        h=get(self.heights*self.valid,'bilinear')
        edge=(1-grid.abs()).min(-1).values*scale.min()/2
        return h,valid,edge

    def contract(self):
        return dict(**super().contract(),heights=self.heights.cpu().tolist(),valid_mask=self.valid.cpu().tolist(),
                    origin=self.origin.cpu().tolist(),spacing=self.spacing)


def make_terrain(kin, cfg):
    kind=cfg.get('terrain_provider','flat')
    options=dict(patch_points=cfg.get('terrain_patch_points',5),patch_width=cfg.get('terrain_patch_width',.4))
    if kind=='flat': return FlatTerrainProvider(kin,**options)
    if kind=='pillar': return PillarTerrainProvider(kin,cfg['pillars'],base_height=cfg.get('base_height'),**options)
    if kind in ('stair','heightmap'):
        return HeightMapTerrainProvider(kin,cfg['heights'],cfg['valid_mask'],cfg['origin'],cfg['spacing'],**options)
    raise ValueError(f'Unknown terrain provider {kind}')
