"""Appendix B Motion/Foot interfaces with one actual Decoder parameter set."""
from contextlib import contextmanager
from dataclasses import asdict
import torch
from torch import nn
from .networks import Policy, WorldModel, Critic
from .goals import GoalSpec
from .s54 import assert_checkpoint
from .transfer import physical_contract, _validate_motion_task, _validate_shapes


class JointPolicy(nn.Module):
    def __init__(self, hist=10):
        super().__init__()
        self.motion = Policy(99, 730, hist=hist)
        self.foot = Policy(99, 230, hist=hist, obs_adapter='mask_reference_orientation')
        self.foot.dec = self.motion.dec
        assert self.motion.dec is self.foot.dec
        assert self.motion.obs_norm is not self.foot.obs_norm

    def canonical_state(self):
        return dict(motion=self.motion.state_dict(),
                    foot={k:v for k,v in self.foot.state_dict().items() if not k.startswith('dec.')})

    def load_canonical(self, state):
        self.motion.load_state_dict(state['motion'], strict=True)
        expected={k for k in self.foot.state_dict() if not k.startswith('dec.')}
        if set(state['foot']) != expected:
            raise ValueError('Foot Encoder/normalization contract differs')
        self.foot.load_state_dict(dict(state['foot'], **{'dec.'+k:v for k,v in self.motion.dec.state_dict().items()}), strict=True)
        assert self.motion.dec is self.foot.dec


def initialize_teacher(teacher, policy, wm, critics, data_contract):
    assert_checkpoint(teacher)
    if teacher.get('physical_contract') != physical_contract():
        raise ValueError('Teacher physics/actuators/backend contract mismatch')
    if teacher.get('goal_contract') != GoalSpec('motion',10,.02).contract():
        raise ValueError('Teacher is not the original Motion 730D interface')
    _validate_motion_task(teacher)
    contract=teacher.get('policy_contract',{})
    if contract.get('hist') != policy.motion.hist or contract.get('obs_adapter') != 'identity':
        raise ValueError('Teacher history/observation semantics differ')
    if teacher.get('data_contract') != data_contract:
        raise ValueError('Teacher data checksum/split mismatch')
    for module,weights,label in [(policy.motion,teacher['pi'],'Motion Actor'),(wm,teacher['wm'],'WM'),
                                  (critics['motion'],teacher['critic'],'Motion Critic')]:
        _validate_shapes(module,weights,label)
    policy.motion.load_state_dict(teacher['pi'],strict=True)
    wm.load_state_dict(teacher['wm'],strict=True)
    critics['motion'].load_state_dict(teacher['critic'],strict=True)
    mapping=[]
    for source,target,state in [('pi','policy.motion',teacher['pi']),('wm','wm',teacher['wm']),
                               ('critic','critics.motion',teacher['critic'])]:
        mapping += [dict(source=source+'.'+k,target=target+'.'+k,shape=list(v.shape),status='loaded') for k,v in state.items()]
    return dict(version=1,loaded=['WM + both WM norms','Motion Encoder + Motion Goal/Obs norms',
                                  'unique Shared Decoder','compatible Motion Critic + norm'],
                new=['Foot Encoder','Foot GoalNorm/ObsNorm','Foot Critic + norm','all five optimizers','both Replay buffers'],
                source_iteration=teacher['iteration'],source_sim_steps=teacher['sim_steps'],
                source_wall_seconds=teacher['wall_time'],mapping=mapping,
                foot_adapter='mask normalized observation channels 93:99; independent norms',
                shared_decoder_identity=policy.motion.dec is policy.foot.dec)


@contextmanager
def frozen(*modules):
    """Freeze parameters only. Never stop the differentiable input/state path."""
    previous=[(p,p.requires_grad) for module in modules for p in module.parameters()]
    try:
        for p,_ in previous:p.requires_grad_(False)
        yield
    finally:
        for p,flag in previous:p.requires_grad_(flag)


def optimizers(policy, wm, critics, cfg):
    result={
        'wm':torch.optim.Adam(wm.parameters(),lr=cfg['lr_wm']),
        'motion':torch.optim.Adam([dict(params=policy.motion.enc.parameters(),lr=cfg['lr_motion_encoder']),
                                  dict(params=policy.motion.dec.parameters(),lr=cfg['lr_decoder'])],betas=(.9,.95)),
        'foot':torch.optim.Adam(policy.foot.enc.parameters(),lr=cfg['lr_foot_encoder'],betas=(.9,.95)),
        **{'critic_'+name:torch.optim.Adam(critic.parameters(),lr=cfg['lr_critic']) for name,critic in critics.items()}}
    owners={}
    for name,opt in result.items():
        for group in opt.param_groups:
            group['initial_lr']=group['lr']
            for p in group['params']:
                if id(p) in owners:raise ValueError('An optimizer parameter is owned twice')
                owners[id(p)]=name
    return result
