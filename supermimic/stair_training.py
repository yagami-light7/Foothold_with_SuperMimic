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


def read_design(path, require_gates=True, *, require_smoke=True):
    cfg=json.loads(Path(path).read_text())
    if cfg.get('training_enabled',True) is False:
        raise ValueError('GPU training is on hold at the user request; viewer/evaluation data remain available')
    if cfg['train_stair_ids']!=['stair_up_4'] or cfg['excluded_stair_ids']!=['stair_up_2']:
        raise ValueError('This experiment is explicitly authorized for four-step reference only')
    if cfg['assignment']!='single_motion_finetuning; stair_up_4 is training, not heldout':
        raise ValueError('Explicit training designation required; upstream test split must not be relabeled silently')
    if cfg.get('normalizer_policy', 'frozen_source') not in ('frozen_source','source_online'):
        raise ValueError('Preserve full source statistics at initialization; count recapping is disabled')
    stages=cfg['sampling_stages']
    if (not stages or any(not 0<s['stair_fraction']<1 for s in stages)
            or any(a['until_iteration']>=b['until_iteration'] for a,b in zip(stages,stages[1:]))):
        raise ValueError('Invalid flat/stair sampling stages')
    if require_gates:
        for name,path in cfg['gates'].items():
            report=json.loads(Path(path).read_text())
            passed=(report.get('passed') is True if name=='contact' else
                    report.get('status') in ['PASS','completed'])
            if not passed:raise ValueError(f'Unpassed prerequisite {name}: {path}')
        kin=json.loads(Path(cfg['gates']['kinematic']).read_text())
        if kin['rows'][0]['frames_saved']!=436 or kin['newton_fk']['status']!='PASS':
            raise ValueError('Full-frame Newton kinematics not validated')
        if require_smoke and cfg.get('required_smoke_gate'):
            gate_path=Path(cfg['required_smoke_gate'])
            if not gate_path.exists():
                raise ValueError('New GPU Smoke is not approved yet: run tools/run_s54_stair.py gate after Smoke completes')
            gate=json.loads(gate_path.read_text())
            if gate.get('passed') is not True or gate.get('data_contract')!=contract(cfg):
                raise ValueError('A successful Smoke gate for this exact data/initialization contract is required')
    expected=cfg.get('source_checkpoint_sha256')
    if expected and sha256(cfg['source_checkpoint'])!=expected:
        raise ValueError('Pinned Flat checkpoint changed')
    return cfg


def contract(cfg):
    source=Path(cfg['bundle'])
    return dict(format='s54_stair_t15_mixed_train_v2',flat=dataset_contract(cfg['flat_dataset']),
        stair_ids=cfg['train_stair_ids'],excluded_ids=cfg['excluded_stair_ids'],
        stair_npz_sha256=sha256(cfg.get('stair_npz',source/'test/stair_up_4.npz')),
        terrain_mesh_sha256=sha256(Path(cfg['terrain'])/'terrain_mesh.npz'),
        source_metadata_sha256=sha256(source/'metadata.json'),
        explicit_assignment=cfg['assignment'],flat_world_translation_m=[20.,0.,0.],
        stair_world_transform='identity',split='experiment_train',
        source_checkpoint_sha256=cfg.get('source_checkpoint_sha256'),
        normalizer_policy=cfg.get('normalizer_policy','frozen_source'),
        reset_noise=cfg.get('reset_noise',False),flat_sampling='equal mode then equal clip',
        sampling_stages=cfg['sampling_stages'],reset_feasibility=cfg.get('reset_feasibility'),
        flat_ground_collider=cfg.get('flat_ground_collider','box'))


def library(cfg,kin,device):
    flat=motion_library(cfg['flat_dataset'],kin,device,'train')
    stair=load_stairs(cfg['bundle'],kin,device,ids=cfg['train_stair_ids'],npz_path=cfg.get('stair_npz'))
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
        modes={m.modes[c] for c in clips}
        for c in clips:
            mask=m.bin_clip==c
            adaptive=torch.softmax(-25*env.bin_G[mask]/env.G_max,0)
            same_mode=sum(m.modes[j]==m.modes[c] for j in clips)
            weights[mask]=mass/len(modes)/same_mode*(.5/int(mask.sum())+.5*adaptive)
    bins=torch.multinomial(weights,n,replacement=True)
    starts=m.bin_start[bins];ends=torch.minimum(starts+BIN_FRAMES,m.clip_end[m.bin_clip[bins]])
    frames=starts+(torch.rand(n,device=env.device)*(ends-starts)).long()
    if hasattr(m,'reset_feasible'):
        invalid=~m.reset_feasible[frames]
        while invalid.any():
            frames[invalid]=starts[invalid]+(torch.rand(int(invalid.sum()),device=env.device)*(ends[invalid]-starts[invalid])).long()
            invalid=~m.reset_feasible[frames]
    return frames,bins


def environment(sim_cfg,n,motion,mdp,cfg,**kwargs):
    # Install the sampler before the constructor performs the first reset.
    # This derived environment uses the SAME step/PD/state code as TrackingSim.
    from .env import TrackingSim
    from .stair_runtime import make_stair_env
    # make_stair_env accepts an explicit class to avoid process-global monkeypatches.
    class MixedTrackingSim(TrackingSim):
        def ensure_reset_validator(self):
            if hasattr(self,'reset_validator') or not cfg.get('reset_feasibility'):return
            from .stair_reset import TerrainResetValidator
            self.reset_validator=TerrainResetValidator(self.mdp.kin,cfg['terrain'],
                clearance=cfg['reset_feasibility']['clearance_m'],
                max_attempts=cfg['reset_feasibility']['max_attempts'])
            m=self.motion;mask=torch.empty(m.num_frames,dtype=torch.bool,device=self.device)
            for start in range(0,m.num_frames,256):
                f=torch.arange(start,min(start+256,m.num_frames),device=self.device)
                mask[f]=self.reset_validator.valid(m.ref_state(f))
            # Every nonempty sampling bin must still contain a feasible reset.
            for b in range(m.num_bins):
                end=min(int(m.bin_start[b])+BIN_FRAMES,int(m.clip_end[m.bin_clip[b]]))
                if not mask[int(m.bin_start[b]):end].any():raise ValueError('Motion bin has no geometry-feasible reset state')
            m.reset_feasible=mask
            self.reset_geometry_audit=dict(total_frames=m.num_frames,
                excluded_reset_frames=(~mask).nonzero().flatten().tolist(),
                clearance_m=self.reset_validator.clearance,
                scope='Reset sampling only; all original frames retained as motion goals')
        def sample_frames(self,count):
            self.ensure_reset_validator()
            return sample_frames(self,count,cfg)
        def prepare_motion_reset(self,state,frames):
            if not cfg.get('reset_feasibility') or not self.reset_noise:return state
            self.ensure_reset_validator()
            return self.reset_validator.condition(state,self.motion.ref_state(frames))
    return make_stair_env(sim_cfg,n,motion,mdp,cfg['terrain'],eval_mode=False,
        reset_noise=cfg.get('reset_noise',False),mixed_flat=True,environment_class=MixedTrackingSim,
        flat_collider=cfg.get('flat_ground_collider','box'),**kwargs)


def initialize_flat(checkpoint,wm,pi,physics,goal_spec,hist,*,normalizer_policy='frozen_source'):
    if normalizer_policy not in ('frozen_source','source_online'):
        raise ValueError('Unknown transfer normalization policy')
    assert_checkpoint(checkpoint);goal_spec.assert_checkpoint(checkpoint)
    if checkpoint['args']['hist']!=hist or checkpoint['policy_contract']['obs_dim']!=103:
        raise ValueError('Flat observation/history contract differs')
    source=checkpoint['physical_contract'];expected={**physics}
    # Only these explicitly audited transfer differences are permitted. Other
    # actuator/clock/gravity/schema differences remain fatal.
    permitted=['backend_versions','terrain_task','collision_decimation']
    a={k:v for k,v in source.items() if k not in permitted}
    b={k:v for k,v in expected.items() if k not in permitted}
    a={**a,'solver':{**a['solver'],'use_mujoco_contacts':False}}
    b={**b,'solver':{**b['solver'],'use_mujoco_contacts':False}}
    if a!=b:raise ValueError('Unapproved flat -> stair physical/actuator transfer difference')
    wm.load_state_dict(checkpoint['wm']);pi.load_state_dict(checkpoint['pi'])
    # Preserve the complete source statistics. The previous exploratory Smoke
    # recapped 32M samples to 10K, moving both policy inputs and WM output scales
    # during collection. That experiment was not accepted. The experiment may
    # freeze statistics or use the original TRAIN-only online update, always
    # retaining the full source count instead of recapping it.
    counts={}
    for label,normalizer in [('policy_obs',pi.obs_norm),('motion_goal',pi.goal_norm),
                             ('wm_input',wm.in_norm),('wm_output',wm.out_norm)]:
        before=float(normalizer.count)
        counts[label]=dict(source=before,finetuning=float(normalizer.count),mean_variance_preserved=True)
    return dict(type='flat_to_stair_explicit_finetune',source_iteration=checkpoint['iteration'],
        source_sim_steps=checkpoint['sim_steps'],source_physics=source,target_physics=physics,
        copied=['WorldModel','MotionEncoder','SharedDecoder','all normalizer means/variances'],
        normalizer_effective_count=counts,
        normalizer_policy=normalizer_policy,
        fresh=['Critic (fixed-world/XYZ/slip reward changed)','optimizers','replay','sampling statistics'],
        reason='Preserve source policy interface; recalibrate WM from real terrain transitions; no reference transitions as WM labels')


_REWARD_DIAGNOSTIC_EVENTS = {}


def validate_transition(out, mdp, cfg, log_path):
    """Reject extreme physical transitions before they enter WM replay/statistics.

    Falls remain valid training data. These bounds diagnose numerical/contact
    explosions; they neither clip state/reward nor turn failures into success.
    """
    from .mdp import unflatten_state
    limits=cfg.get('transition_limits',{})
    state=unflatten_state(out['s1'])
    values=dict(root_speed_m_s=state[2].norm(dim=-1),root_angvel_rad_s=state[3].norm(dim=-1),
                joint_speed_rad_s=state[5].abs().amax(-1),normalized_reward=out['r']/mdp.max_step_reward)
    bad=~torch.isfinite(out['r'])
    for key in ('s','s1','a','a_prev','xi'):
        if key in out:bad |= ~torch.isfinite(out[key]).reshape(len(bad),-1).all(-1)
    for key,limit in limits.items():
        if key!='normalized_reward':bad |= values[key]>limit

    def snapshot(ids):
        payload={key:value[ids].detach().cpu() for key,value in out.items()
                 if isinstance(value,torch.Tensor) and value.ndim and value.shape[0]==len(bad)}
        payload['diagnostic_values']={k:v[ids].detach().cpu() for k,v in values.items()}
        payload['reward_terms']={k:v[ids].detach().cpu() for k,v in out.get('terms',{}).items()}
        return payload

    if bad.any():
        ids=bad.nonzero().flatten()
        torch.save(snapshot(ids),Path(log_path)/'failure_physics_transition.pt')
        raise FloatingPointError(f'Extreme Newton transition at envs {ids.tolist()}; saved failure_physics_transition.pt')

    # Tracking penalties are unbounded below. A finite negative reward is useful
    # learning feedback, not evidence of a numerical/contact explosion. Keep it
    # intact in replay and save a bounded diagnostic sample instead of aborting.
    threshold=cfg.get('tracking_reward_warning_below',limits.get('normalized_reward',-20.))
    low=values['normalized_reward']<threshold
    if low.any():
        path=Path(log_path);event=_REWARD_DIAGNOSTIC_EVENTS.get(str(path),0)+1
        _REWARD_DIAGNOSTIC_EVENTS[str(path)]=event
        if event<=16 or event%1000==0:
            ids=low.nonzero().flatten();selected=ids[:5]
            with (path/'tracking_reward_diagnostics.jsonl').open('a') as sink:
                sink.write(json.dumps(dict(kind='low_finite_tracking_reward',event=event,
                    count=int(low.sum()),min_normalized_reward=float(values['normalized_reward'].min()),
                    env_ids=selected.tolist(),frames=out['f'][selected].tolist(),
                    action='retain unchanged in replay; no physics threshold exceeded'))+'\n')
            sample_path=path/'tracking_low_reward_sample.pt'
            if event==1 and not sample_path.exists():torch.save(snapshot(selected),sample_path)
