"""Isolated, reference-only Stage 1.5 contracts and reproducible train-set norms."""
from pathlib import Path
import hashlib, json, random
import numpy as np
import torch
from .joint_runtime import project, digest, profile_hash
from .motion_prior_runtime import reconstruct
from .networks import Policy, WorldModel
from .goals import GoalSpec
from .reference_foot import ReferenceFootGoalSource, PairedMotionMDP
from .foot_trajectory import AlternatingConfig
from .foot_mdp import FootTrackingMDP, resolve_foot_task

VERSION='lafan1_stage15_v1'

class CausalHeadingGoals(ReferenceFootGoalSource):
    """Reconstruct virtual heading channels ONLY from the same 230D FootGoal.

    Actual root yaw cancels from relative orientation. Tilt is available from
    body gravity; the fixed desired tilt/bias are training-set calibration.
    No Motion root orientation is queried by this observation path.
    """
    def __init__(self,*args,heading_bias=0.,reference_tilt=(0.,0.),**kwargs):
        super().__init__(*args,**kwargs)
        self.heading_bias=heading_bias;self.reference_tilt=reference_tilt

    def obs(self,mdp,S,a_prev,xi,f,command=None):
        from .rotations import yaw_mat,so3_exp
        from .reference_foot import ReferenceFootCommand
        if command is None:raise ValueError('Foot-only heading requires immutable command')
        shape=f.shape
        if f.ndim==2:
            B,H=shape
            cmd=ReferenceFootCommand(**{k:v.repeat_interleave(H,0) for k,v in command.tensors().items()})
            flat=tuple(x.flatten(0,1) for x in S);frames=f.flatten()
        else:flat=S;frames=f;cmd=command
        step=frames-cmd.start_frame
        g=self.goal(flat,frames,step,cmd).view(-1,10,23)
        angle=torch.atan2(torch.stack((g[:,:,5],g[:,:,16]),-1),torch.stack((g[:,:,3],g[:,:,14]),-1))
        contact=torch.stack((g[:,:,9],g[:,:,20]),-1)
        weight=.25+.75*contact
        yaw=torch.atan2((angle.sin()*weight).sum((1,2)),(angle.cos()*weight).sum((1,2)))-self.heading_bias
        e=torch.zeros((len(frames),3),device=frames.device);e[:,2]=yaw
        roll=torch.zeros_like(e);roll[:,0]=self.reference_tilt[0]
        pitch=torch.zeros_like(e);pitch[:,1]=self.reference_tilt[1]
        desired=yaw_mat(flat[1])@so3_exp(e)@so3_exp(pitch)@so3_exp(roll)
        if f.ndim==2:desired=desired.view(*shape,3,3)
        return mdp.obs(S,a_prev,xi,f,reference_R=desired)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def atomic_save(value,path):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        torch.save(value,f);f.flush();__import__('os').fsync(f.fileno())
    tmp.replace(path)

def split_frames(total,lookahead=10,exclude_eval=True):
    """All fit statistics exclude three predeclared 10s evaluation intervals."""
    f=torch.arange(total-lookahead-1)
    keep=torch.ones_like(f,dtype=torch.bool)
    if exclude_eval:
        for start in (0,3000,6000):keep &= ~((f>=max(0,start-lookahead))&(f<=start+500+lookahead))
    valid=f[keep];val=valid[(valid//50)%10==9];train=valid[(valid//50)%10!=9]
    return train,val

def goal_floor():
    # 2 feet: p3, rotation6, contact, swing phase; final time in seconds.
    return torch.tensor(([.05]*3+[.15]*6+[.25,.25])*2+[.10]).repeat(10)

@torch.no_grad()
def fit_goal_norm(norm,train_goals):
    floor=goal_floor().to(train_goals)
    norm.mean.copy_(train_goals.mean(0))
    norm.var.copy_((train_goals.var(0,unbiased=False).maximum(floor.square())-norm.eps**2).clamp_min(0))
    norm.count.fill_(len(train_goals));norm.max_count=float(len(train_goals))
    return dict(count=len(train_goals),minimum_effective_std=float(norm.std().min()),
                floors_per_frame=goal_floor()[:23].tolist(),sha256=tensor_hash(norm.state_dict()))

def build(teacher_path,cfg,device,stage_checkpoint=None):
    teacher_path=Path(teacher_path).resolve();sha=digest(teacher_path)
    if sha!=cfg['teacher_sha256']:raise ValueError('Teacher checksum mismatch')
    source=torch.load(teacher_path,map_location='cpu',weights_only=False)
    if source['iteration']!=14484 or source['joint_contract']['profile_sha256']!=profile_hash(source['profile']):
        raise ValueError('Teacher iteration/profile differs')
    kin,motion,mg,mdp,mp=reconstruct(project(source,'motion'),device)
    heading=cfg.get('causal_heading',False)
    fp=Policy(99,230,hist=cfg['hist'],obs_adapter='identity' if heading else 'mask_reference_orientation').to(device)
    fp.dec=mp.dec
    fp.obs_norm.load_state_dict(mp.obs_norm.state_dict(),strict=True)
    fp.obs_norm.max_count=float(fp.obs_norm.count)
    # Fresh E_f; do not import saturated Quintic E_f or its incompatible norms.
    with torch.no_grad():fp.enc[-1].weight.mul_(.01);fp.enc[-1].bias.zero_()
    goals=(CausalHeadingGoals(motion,kin,GoalSpec('reference_foot',10,.02),AlternatingConfig(),
            heading_bias=cfg['heading_bias_rad'],reference_tilt=cfg['reference_tilt_roll_pitch_rad']) if heading else
           ReferenceFootGoalSource(motion,kin,GoalSpec('reference_foot',10,.02),AlternatingConfig()))
    paired=PairedMotionMDP(kin,motion,mdp.default_q,goals,mdp.cfg)
    _,_,_,guard,_=reconstruct(project(source,'foot'),device)
    wm=WorldModel().to(device);wm.load_state_dict(source['wm'],strict=True)
    contract=dict(kind=VERSION,teacher_sha256=sha,teacher_iteration=14484,hist=cfg['hist'],
        goal=goals.spec.contract(),observation=('Teacher ObsNorm frozen; heading from 230D FootGoal and body gravity only' if heading else
                                             'Teacher ObsNorm copied and frozen; normalized channels 93:99 zeroed'),
        physical=source['joint_contract']['physical'],data=source['joint_contract']['data'],
        reward=mdp.contract(),actor_inputs=('230D original FK FootGoal + 10x93 causal proprioception; FootGoal-derived heading slots' if heading else
                                           '230D original FK FootGoal + 10x93 causal proprioception; six masked slots'))
    if heading:contract['heading_calibration']=dict(bias_rad=cfg['heading_bias_rad'],tilt=cfg['reference_tilt_roll_pitch_rad'])
    if stage_checkpoint is not None:
        if stage_checkpoint['contract']!=contract:raise ValueError('Stage15 checkpoint contract mismatch')
        mp.load_state_dict(stage_checkpoint['motion'],strict=True)
        fp.load_state_dict(stage_checkpoint['foot'],strict=True)
        wm.load_state_dict(stage_checkpoint['wm'],strict=True)
        # Loading Foot cannot change frozen D or Teacher M weights.
        if tensor_hash(mp.state_dict())!=tensor_hash(source['policy']['motion']):
            raise ValueError('Stage15 changed frozen Teacher Motion/Decoder')
    mp.eval().requires_grad_(False);fp.requires_grad_(False);fp.enc.requires_grad_(True)
    return dict(source=source,kin=kin,motion=motion,mg=mg,mdp=mdp,mp=mp,fp=fp,goals=goals,
                paired=paired,guard=guard,wm=wm,contract=contract)

@torch.no_grad()
def dataset(parts,cfg):
    m=parts['motion'];g=parts['goals'];mp=parts['mp'];device=m.root_p.device
    excluded=cfg.get('exclude_eval_windows',True)
    tr,va=split_frames(m.num_frames,exclude_eval=excluded);tr=tr.to(device);va=va.to(device)
    def encode(frames,augment):
        S=m.ref_state(frames);command=g.new_command(S,frames=frames)
        target=mp.encode(m.goal(frames,10)).detach()
        goals=[g.goal(S,frames,torch.zeros_like(frames),command)]
        if augment:
            # Reproducible causal root state variations; goals transform exactly.
            from .rotations import so3_exp
            gen=torch.Generator(device=device).manual_seed(cfg['seed']+15)
            for xy,yaw in cfg.get('root_augmentation_sigmas',[[.03,.08],[.03,.08]]):
                p=S[0].clone();p[:,:2]+=torch.randn((len(frames),2),device=device,generator=gen)*xy
                w=torch.zeros((len(frames),3),device=device);w[:,2]=torch.randn(len(frames),device=device,generator=gen)*yaw
                modified=(p,so3_exp(w)@S[1],*S[2:])
                goals.append(g.goal(modified,frames,torch.zeros_like(frames),command))
        return torch.cat(goals),target.repeat(len(goals),1)
    X,Y=encode(tr,True);V,Z=encode(va,False)
    meta=fit_goal_norm(parts['fp'].goal_norm,X)
    meta.update(train_frames=tr.cpu().tolist(),validation_frames=va.cpu().tolist(),
        excluded_eval_windows_s=[[0,10],[60,70],[120,130]] if excluded else [],seed=cfg['seed'],
        augmentation_sigmas=cfg.get('root_augmentation_sigmas',[[.03,.08],[.03,.08]]),
        augmentation='Fixed-seed root XY/yaw perturbations; Teacher labels unchanged; no reference joint input to E_f',
        train_max_abs_normalized=float(parts['fp'].goal_norm(X).abs().max()),
        validation_max_abs_normalized=float(parts['fp'].goal_norm(V).abs().max()))
    meta.update(teacher_train_latent_saturation=float((Y.abs()>.95).float().mean()),
                teacher_validation_latent_saturation=float((Z.abs()>.95).float().mean()),
                target='tanh Teacher Motion Encoder output, not its pre-activation')
    return X,Y,V,Z,meta

def checkpoint(parts,cfg,phase,iteration,optimizer=None,**extra):
    return dict(contract=parts['contract'],config=cfg,phase=phase,iteration=iteration,
        motion=parts['mp'].state_dict(),foot=parts['fp'].state_dict(),wm=parts['wm'].state_dict(),
        optimizer=optimizer.state_dict() if optimizer else None,
        rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate()),
        **extra)
