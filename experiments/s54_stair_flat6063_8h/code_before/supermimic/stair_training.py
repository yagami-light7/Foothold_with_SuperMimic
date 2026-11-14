"""T1.5 data/scene hooks for the existing SuperMimic training algorithm."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .m34 import dataset_contract, motion_library, sha256
from .motion import MotionLib, BIN_FRAMES
from .stair_data import load_stairs
from .stair_runtime import make_stair_env
from .s54 import FOOT_BODY_NAMES, assert_checkpoint


def read_design(path, require_gates=True):
    cfg=json.loads(Path(path).read_text())
    if cfg.get('training_enabled',True) is False:
        raise ValueError('GPU training is on hold at the user request; viewer/evaluation data remain available')
    if cfg['train_stair_ids']!=['stair_up_4'] or cfg['excluded_stair_ids']!=['stair_up_2']:
        raise ValueError('This experiment is explicitly authorized for four-step reference only')
    if cfg['assignment']!='single_motion_finetuning; stair_up_4 is training, not heldout':
        raise ValueError('Explicit training designation required; upstream test split must not be relabeled silently')
    if require_gates:
        for name,path in cfg['gates'].items():
            report=json.loads(Path(path).read_text())
            passed=(report.get('passed') is True if name=='contact' else
                    report.get('status') in ['PASS','completed'])
            if not passed:raise ValueError(f'Unpassed prerequisite {name}: {path}')
        kin=json.loads(Path(cfg['gates']['kinematic']).read_text())
        if kin['rows'][0]['frames_saved']!=436 or kin['newton_fk']['status']!='PASS':
            raise ValueError('Full-frame Newton kinematics not validated')
    return cfg


def contract(cfg):
    source=Path(cfg['bundle'])
    return dict(format='s54_stair_t15_mixed_train_v1',flat=dataset_contract(cfg['flat_dataset']),
        stair_ids=cfg['train_stair_ids'],excluded_ids=cfg['excluded_stair_ids'],
        stair_npz_sha256=sha256(source/'test/stair_up_4.npz'),
        terrain_mesh_sha256=sha256(Path(cfg['terrain'])/'terrain_mesh.npz'),
        source_metadata_sha256=sha256(source/'metadata.json'),
        explicit_assignment=cfg['assignment'],flat_world_translation_m=[20.,0.,0.],
        stair_world_transform='identity',split='experiment_train')


def library(cfg,kin,device):
    flat=motion_library(cfg['flat_dataset'],kin,device,'train')
    stair=load_stairs(cfg['bundle'],kin,device,ids=cfg['train_stair_ids'])
    # Only one immutable paired translation per FLAT clip; stairs remain in
    # authoritative world coordinates. No per-frame/robot-dependent alignment.
    translation=flat.root_p.new_tensor([20.,0.,0.])
    if flat.body_p[...,:2].abs().max()>11:
        raise ValueError('Flat motion could leave its finite isolated ground; resize explicitly')
    flat.root_p+=translation;flat.body_p+=translation
    feet=[kin.tracked_names.index(n) for n in FOOT_BODY_NAMES]
    sole=(flat.body_p[:,feet]+(flat.body_R[:,feet] @ translation.new_tensor([.051,0.,-.062])[:,None])[...,0])
    flat.contacts=(sole[...,2]<.02)&(flat.body_v[:,feet,:2].norm(dim=-1)<.08)
    m=MotionLib.__new__(MotionLib);m.device=device;m.split='experiment_train'
    m.names=flat.names+stair.names;m.modes=flat.modes+stair.modes;m.rows=flat.rows+stair.rows
    m.validation=flat.validation+stair.validation
    for name in ['root_p','root_R','root_v','root_w','q','qd','body_p','body_R','body_v','body_w',
                 'root_v_local','root_w_local','root_gz','contacts']:
        setattr(m,name,torch.cat((getattr(flat,name),getattr(stair,name))))
    m.clip_len=torch.cat((flat.clip_len,stair.clip_len))
    m.clip_start=torch.cumsum(torch.cat((m.clip_len.new_zeros(1),m.clip_len[:-1])),0)
    m.clip_end=m.clip_start+m.clip_len-1;m.num_frames=int(m.clip_len.sum())
    m.clip_id=torch.repeat_interleave(torch.arange(len(m.names),device=device),m.clip_len)
    bins=[(c,s) for c in range(len(m.names)) for s in range(int(m.clip_start[c]),int(m.clip_end[c]),BIN_FRAMES)]
    m.bin_clip=torch.tensor([c for c,s in bins],device=device);m.bin_start=torch.tensor([s for c,s in bins],device=device)
    m.num_bins=len(bins);m.frame_bin=torch.zeros(m.num_frames,dtype=torch.long,device=device)
    for b,(c,s) in enumerate(bins):m.frame_bin[s:min(s+BIN_FRAMES,int(m.clip_end[c])+1)]=b
    return m


def sample_frames(env,n,cfg):
    m=env.motion;iteration=getattr(env,'stair_iteration',0)
    stages=cfg['sampling_stages'];stage=next((s for s in stages if iteration<s['until_iteration']),stages[-1])
    stair_fraction=stage['stair_fraction']
    weights=torch.zeros_like(env.bin_G)
    for is_stair,mass in [(False,1-stair_fraction),(True,stair_fraction)]:
        clips=[i for i,mode in enumerate(m.modes) if (mode=='stair_up')==is_stair]
        for c in clips:
            mask=m.bin_clip==c
            adaptive=torch.softmax(-25*env.bin_G[mask]/env.G_max,0)
            weights[mask]=mass/len(clips)*(.5/int(mask.sum())+.5*adaptive)
    bins=torch.multinomial(weights,n,replacement=True)
    starts=m.bin_start[bins];ends=torch.minimum(starts+BIN_FRAMES,m.clip_end[m.bin_clip[bins]])
    return starts+(torch.rand(n,device=env.device)*(ends-starts)).long(),bins


def environment(sim_cfg,n,motion,mdp,cfg,**kwargs):
    # Install the sampler before the constructor performs the first reset.
    # This derived environment uses the SAME step/PD/state code as TrackingSim.
    from .env import TrackingSim
    from .stair_runtime import make_stair_env
    # make_stair_env accepts an explicit class to avoid process-global monkeypatches.
    class MixedTrackingSim(TrackingSim):
        def sample_frames(self,count):
            return sample_frames(self,count,cfg)
    return make_stair_env(sim_cfg,n,motion,mdp,cfg['terrain'],eval_mode=False,
        reset_noise=True,mixed_flat=True,environment_class=MixedTrackingSim,**kwargs)


def initialize_flat(checkpoint,wm,pi,physics,goal_spec,hist):
    assert_checkpoint(checkpoint);goal_spec.assert_checkpoint(checkpoint)
    if checkpoint['args']['hist']!=hist or checkpoint['policy_contract']['obs_dim']!=103:
        raise ValueError('Flat observation/history contract differs')
    source=checkpoint['physical_contract'];expected={**physics}
    # Only these explicitly audited transfer differences are permitted. Other
    # actuator/clock/gravity/schema differences remain fatal.
    permitted=['backend_versions','terrain_task']
    a={k:v for k,v in source.items() if k not in permitted}
    b={k:v for k,v in expected.items() if k not in permitted}
    a={**a,'solver':{**a['solver'],'use_mujoco_contacts':False}}
    b={**b,'solver':{**b['solver'],'use_mujoco_contacts':False}}
    if a!=b:raise ValueError('Unapproved flat -> stair physical/actuator transfer difference')
    wm.load_state_dict(checkpoint['wm']);pi.load_state_dict(checkpoint['pi'])
    # Preserve initial forward values (mean/variance), but let real TRAIN-only
    # transitions adapt statistics instead of inheriting millions of flat counts.
    counts={}
    for label,normalizer in [('policy_obs',pi.obs_norm),('motion_goal',pi.goal_norm),
                             ('wm_input',wm.in_norm),('wm_output',wm.out_norm)]:
        before=float(normalizer.count)
        normalizer.count.clamp_(max=10000)
        counts[label]=dict(source=before,finetuning=float(normalizer.count),mean_variance_preserved=True)
    return dict(type='flat_to_stair_explicit_finetune',source_iteration=checkpoint['iteration'],
        source_sim_steps=checkpoint['sim_steps'],source_physics=source,target_physics=physics,
        copied=['WorldModel','MotionEncoder','SharedDecoder','all normalizer means/variances'],
        normalizer_effective_count=counts,
        fresh=['Critic (fixed-world/XYZ/slip reward changed)','optimizers','replay','sampling statistics'],
        reason='Preserve source policy interface; recalibrate WM from real terrain transitions; no reference transitions as WM labels')
