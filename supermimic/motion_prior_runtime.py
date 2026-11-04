"""Checkpoint reconstruction for bounded evaluations; no teacher in direct Foot deployment."""
import torch

from .kinematics import S54Kinematics
from .motion import MotionLib
from .mdp import MDP,RewardCfg
from .goals import GoalSource,GoalSpec
from .foot_mdp import FootTrackingMDP,resolve_foot_task,assert_task_checkpoint
from .foot_trajectory import AlternatingConfig
from .networks import Policy,DualEncoderPolicy
from .motion_prior_data import load_manifest
from .reference_selection import select_reference_files
from .s54 import default_joint_pos_list,assert_checkpoint
from .transfer import physical_contract,assert_policy_checkpoint


def reconstruct(checkpoint,device,manifest=None,split='train'):
    assert_checkpoint(checkpoint)
    if checkpoint.get('physical_contract') is not None and checkpoint['physical_contract']!=physical_contract():
        raise ValueError('Evaluation physics differs from checkpoint')
    cfg=checkpoint['args'];kin=S54Kinematics(device)
    spec=GoalSpec(cfg.get('goal_mode','motion'),cfg.get('goal_frames',10),cfg.get('goal_dt'))
    foot_cfg=AlternatingConfig(**(cfg.get('foot_config') or {}))
    motion=None
    if spec.mode!='foot_trajectory':
        path=manifest or cfg.get('reference_manifest')
        files=load_manifest(path,split)[0] if path else select_reference_files(cfg['motion_dir'],cfg.get('clips'))
        motion=MotionLib(files,kin,device)
    task=resolve_foot_task(cfg.get('foot_task_config')) if spec.mode=='foot_trajectory' else None
    goals=GoalSource(motion,kin,spec,foot_cfg,command_mode=task.command_mode if task else 'alternating')
    default=torch.tensor(default_joint_pos_list(),device=device)
    mdp=FootTrackingMDP(kin,default,goals,task) if task else MDP(kin,motion,default,RewardCfg(**cfg.get('reward_config',{})))
    if spec.mode=='reference_foot':
        from .reference_foot import ReferenceFootGoalSource,PairedMotionMDP
        goals=ReferenceFootGoalSource(motion,kin,spec,foot_cfg)
        mdp=PairedMotionMDP(kin,motion,default,goals,RewardCfg(**cfg['reward_config']))
    if cfg.get('prior_config'):
        from .upper_prior import PriorFootMDP,resolve_prior
        mdp=PriorFootMDP(kin,default,goals,task,resolve_prior(cfg['prior_config']))
    assert_task_checkpoint(checkpoint,mdp);spec.assert_checkpoint(checkpoint,foot_cfg)
    assert_policy_checkpoint(checkpoint,cfg['hist'],cfg.get('obs_adapter','identity'),not cfg.get('freeze_decoder',False))
    pi=(DualEncoderPolicy(hist=cfg['hist']) if spec.mode=='reference_foot' else
        Policy(99,spec.dim,hist=cfg['hist'],obs_adapter=cfg.get('obs_adapter','identity'))).to(device)
    pi.load_state_dict(checkpoint['pi'],strict=True);pi.eval()
    return kin,motion,goals,mdp,pi
