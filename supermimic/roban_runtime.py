"""Small adapters consumed by the existing Trainer, not another training loop."""
from dataclasses import asdict
import json
from pathlib import Path


def select_roban_files(directory, clips=None):
    import fnmatch
    root=Path(directory)
    files=sorted([*root.glob('*.csv'),*root.glob('*.npz')])
    if not files: raise ValueError(f'No Roban CSV/NPZ files in {root}')
    if clips:
        selected=set()
        for pattern in clips:
            matched=[f for f in files if fnmatch.fnmatchcase(f.stem,pattern)]
            if not matched: raise ValueError(f'No Roban reference matches {pattern!r}')
            selected.update(matched)
        files=sorted(selected)
    return [str(f) for f in files]


def configure_robot(argv):
    """Must run before any robot-specific module import."""
    import argparse
    import os
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--robot',choices=['s54','roban'])
    p.add_argument('--roban_config')
    args,_=p.parse_known_args(argv)
    kind=args.robot or ('roban' if args.roban_config else 's54')
    if args.roban_config and kind!='roban':
        raise ValueError('--roban_config requires --robot roban')
    os.environ['SUPERMIMIC_ROBOT']=kind
    return kind


def add_roban_args(parser,argv):
    parser.add_argument('--robot',choices=['s54','roban'],default='s54')
    parser.add_argument('--roban_config',help='Trainer-consumed Roban experiment JSON')
    parser.add_argument('--reference_fps',type=float,help='Explicit source FPS when registry/NPZ has no provenance')
    parser.add_argument('--critic_use_body_reference',action=__import__('argparse').BooleanOptionalAction,default=False)
    parser.add_argument('--body_prior_weight',type=float,default=0.0)
    parser.add_argument('--wm_use_terrain',action=__import__('argparse').BooleanOptionalAction,default=False)
    parser.add_argument('--terrain_provider',choices=['flat'],default='flat',help='Only flat physics is enabled in this stage')
    parser.add_argument('--terrain_patch_points',type=int,default=5)
    parser.add_argument('--terrain_patch_width',type=float,default=.4)
    parser.add_argument('--roban_reward_config',type=json.loads,default={})
    parser.add_argument('--roban_gradient_audit',action='store_true',help='Measure isolated primary-foot Actor gradients')
    p=__import__('argparse').ArgumentParser(add_help=False)
    p.add_argument('--roban_config');known,_=p.parse_known_args(argv)
    if known.roban_config:
        cfg=json.loads(Path(known.roban_config).read_text())
        allowed={a.dest for a in parser._actions}
        if set(cfg)-allowed:
            raise ValueError(f'Unknown Trainer config keys: {set(cfg)-allowed}')
        parser.set_defaults(**cfg)
    from .robot_contract import ROBOT_KIND
    if ROBOT_KIND=='roban':
        if not known.roban_config:
            parser.set_defaults(motion_dir='library/motion_reference/roban/style100_roban_50hz/Neutral',clips=['Neutral_FW'])
        parser.set_defaults(robot='roban',goal_mode='roban_foot',motion_obs_mode='roban_proprio_v1',
                            log_dir=str(Path(__file__).resolve().parents[1]/'logs/roban_smoke'))


def build_task(args,kin,files,device):
    import torch
    from .motion_reference import MotionReferenceProvider
    from .roban import default_joint_pos_list
    from .goals import GoalSpec
    from .roban_task import RobanFootGoals,RobanFootMDP,FootRewardConfig
    from .terrain import make_terrain
    motion=MotionReferenceProvider(files,kin,device,args.reference_fps)
    print(f"Roban reference: {motion.num_frames} control frames; trusted contact fraction="
          f"{motion.validation['trusted_contact_fraction']:.3f}; prior quality mean="
          f"{motion.validation['body_prior_quality_mean']:.3f}",flush=True)
    if args.body_prior_weight and motion.validation['body_prior_quality_mean']==0:
        print('Roban Body Prior is fully masked: source has no accepted IK quality frames.',flush=True)
    spec=GoalSpec('roban_foot',args.goal_frames,args.goal_dt)
    goals=RobanFootGoals(motion,kin,spec)
    default_q=torch.tensor(default_joint_pos_list(),device=device)
    rewards=dict(args.roban_reward_config)
    rewards['body_prior_weight']=args.body_prior_weight
    cfg=FootRewardConfig(**rewards)
    # Terrain always describes the real flat scene. WM/Critic may omit it in
    # E0/E1/E2; the foot reward still checks the real support plane analytically.
    terrain=make_terrain(kin,vars(args)) if args.wm_use_terrain else None
    mdp=RobanFootMDP(kin,motion,default_q,goals,cfg,
                     critic_use_body_reference=args.critic_use_body_reference,terrain=terrain)
    args.reward_config=asdict(cfg)
    args.motion_obs_mode='roban_proprio_v1'
    return motion,goals,mdp,terrain


def initialize_compatible(ck,wm,pi,physical,*,hist,goal_contract=None):
    """Explicit partial reuse, with a complete tensor report and no optimizer copy.

    S54 is rejected by the robot contract. Same-Roban flat WM can initialize a
    terrain WM by copying base columns and zeroing new terrain columns. A motion
    goal encoder is never copied into a foot encoder, even if shapes coincide.
    """
    import torch
    from .roban import assert_checkpoint
    assert_checkpoint(ck)
    saved=ck.get('physical_contract')
    if saved != physical:
        raise ValueError('Partial transfer requires matching Roban physics/actuator/state contract')
    if ck.get('args',{}).get('hist') != hist:
        raise ValueError('Partial transfer history mismatch')
    operations=[];report=[]
    for name,module in (('wm',wm),('pi',pi)):
        state=module.state_dict();source=ck.get(name,{})
        same_goal=(ck.get('goal_contract')==goal_contract if goal_contract is not None else
                   ck.get('goal_contract',{}).get('mode')=='roban_foot')
        for key,target in state.items():
            value=source.get(key)
            reuse=value is not None and value.shape==target.shape
            if name=='pi' and key.startswith(('enc.','goal_norm.')) and not same_goal:
                reuse=False
            if name=='pi' and key.startswith(('dec.','obs_norm.')):
                reuse=reuse and ck.get('policy_contract',{}).get('observation_mode')=='roban_proprio_v1'
            status='copied' if reuse else 'new'
            if reuse:
                if not torch.isfinite(value).all(): raise ValueError(f'Nonfinite transfer tensor {name}.{key}')
                operations.append((target,value))
            elif name=='wm' and value is not None and wm.terrain_provider is not None:
                base=wm.in_dim-wm.terrain_provider.dim
                if key=='net.0.weight' and value.shape==(target.shape[0],base):
                    expanded=torch.zeros_like(target);expanded[:,:base]=value
                    operations.append((target,expanded));status='expanded_zero_terrain_columns'
                elif key in ('in_norm.mean','in_norm.var') and value.shape==(base,):
                    expanded=target.clone();expanded[:base]=value
                    operations.append((target,expanded));status='expanded_terrain_normalization'
            report.append(dict(module=name,key=key,status=status,shape=list(target.shape),
                               source_shape=list(value.shape) if value is not None else None))
            if name=='wm' and status=='new':
                raise ValueError(f'Unexpected WM transfer mismatch: {key}; use matching network architecture')
            if name=='pi' and key.startswith(('dec.','obs_norm.')) and status!='copied':
                raise ValueError(f'Incompatible decoder/proprioception tensor: {key}; start fresh explicitly')
        extra=set(source)-set(state)
        report.extend(dict(module=name,key=k,status='source_only') for k in sorted(extra))
    if not any(r['module']=='pi' and r['key'].startswith('dec.') and r['status']=='copied' for r in report):
        raise ValueError('No compatible Roban decoder; use a fresh run explicitly')
    with torch.no_grad():
        for target,value in operations: target.copy_(value)
    return dict(version=1,source_iteration=ck.get('iteration'),mapping=report,
                optimizers='fresh',critic='fresh',goal_encoder='copied only for the same foot schema')
