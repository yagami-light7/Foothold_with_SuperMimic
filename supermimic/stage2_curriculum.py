"""Isolated Stage2 transfer, Quintic goals and five-stage curriculum.

No MotionLib, CSV, Motion Encoder or MotionReward is loaded by build().
"""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import math,copy
import torch
from torch import nn
from .curriculum import Curriculum,Command,TASKS,feasibility
from .goals import GoalSource,GoalSpec
from .foot_trajectory import FootstepCommand,AlternatingConfig
from .rotations import yaw_mat,so3_exp
from .foot_mdp import FootTrackingMDP,resolve_foot_task
from .networks import Policy,WorldModel,Critic
from .kinematics import S54Kinematics
from .s54 import default_joint_pos_list
from .transfer import physical_contract
from .stage15 import tensor_hash
from .joint_runtime import digest
from .joint_updates import JointUpdates
from .joint_replay import JointReplay

STAGES=('C0_balance_short','C1_longer_walk','C2_turn_in_place','C3_walk_turn','C4_mixed')

class Stage2Curriculum(Curriculum):
    def __init__(self,cfg):
        self.profile=copy.deepcopy(cfg)
        # The existing immutable SE(2) plan builder is reused unchanged.
        super().__init__(dict(seed=cfg['seed']+200,weights=cfg['stage_weights']+[cfg['stage_weights'][-1]],
            stage_hours=cfg['stage_hours']+[0.],yaw_limits=cfg['stage_yaw_max']+[cfg['stage_yaw_max'][-1]],
            switch_probability=0.,segment_seconds=4.,clearance=cfg['stage_clearance'][0],
            max_speed=.45,max_swing_speed=1.8,max_swing_accel=12.,min_swing=.32,min_support=.12))
        self.evaluations=[];self.last_improvement=0.;self.best_survival=0.;self.regressions=0
    def sample(self):
        stage=self.stage;weights=list(self.profile['stage_weights'][stage])
        cap=self.profile['stage_length_max'][stage]*(.6+.4*self.progress)
        if cap<.16:
            weights[2]+=weights[3];weights[3]=0.
        for i in (2,3,4,5):weights[i]*=.3+.7*self.success.get(TASKS[i],.5)
        task=self.rng.choices(TASKS,weights)[0]
        for _ in range(200):
            if task=='standing':cmd=Command()
            else:
                if task=='short':lo,hi=.02,min(.08,cap)
                elif task=='medium':lo,hi=.08,min(.16,cap)
                elif task=='long':lo,hi=.16,max(.16,cap)
                else:lo,hi=.02,min(.10,cap)
                length=0. if task=='turn' else self.rng.uniform(min(lo,hi),hi)
                if task=='short' and stage==0 and self.rng.random()<.1:length=0. # weight-shift/stepping, not forward walking
                flo,fhi=self.profile['stage_frequency'][stage]
                if task=='long':fhi=min(fhi,1.15)
                # Separate lower/upper strata; success feedback reduces hard bins.
                mid=(flo+fhi)/2
                hard=self.rng.random()<.35*(.3+.7*self.bucket_success.get(task+'_high',.5))
                frequency=self.rng.uniform(mid if hard else flo,fhi if hard else mid)
                limit=self.profile['stage_yaw_max'][stage]*(.4+.6*self.progress)
                yaw=self.rng.choice((-1,1))*self.rng.uniform(.04,max(.04,limit)) if task in ('turn','curve') else 0.
                cmd=Command(task,length,frequency,yaw)
            reason=feasibility(cmd,self.cfg)
            if reason is None:break
            self.rejected[reason]+=1
        else:raise RuntimeError('No feasible Stage2 command: '+task)
        self.counts[task]+=1;self.hist[cmd.bucket]+=1
        return cmd
    def apply_stage(self):
        self.cfg['clearance']=self.profile['stage_clearance'][self.stage]
        self.cfg['switch_probability']=.1 if self.stage>=4 else 0.
    def load_state_dict(self,state):
        if state['profile']!=self.profile:raise ValueError('Stage2 curriculum profile mismatch')
        for k,v in copy.deepcopy(state).items():
            if k!='rng':setattr(self,k,v)
        self.rng.setstate(state['rng']);self.apply_stage()
    def update_evaluation(self,report,elapsed,full=False):
        if not report['complete']:return False
        rows=report['rows'];required={c.task for _,c in evaluation_cases(self.stage)}
        for task in required:
            rr=[r for r in rows if r['command']['task']==task]
            if rr:self.success[task]=sum(r['success_rate'] for r in rr)/len(rr)
        mid=sum(self.profile['stage_frequency'][self.stage])/2
        for row in rows:
            cmd=row['command'];key=cmd['task']+('_high' if cmd['frequency']>=mid else '_low')
            self.bucket_success[key]=.7*self.bucket_success.get(key,.5)+.3*row['success_rate']
        passed=all(r['success_rate']>=self.profile['success_threshold'] and r['survival_rate']>=self.profile['survival_threshold'] for r in rows)
        self.passes=self.passes+1 if passed else 0;self.failures=0 if passed else self.failures+1
        self.progress=min(1.,self.progress+.2) if passed else max(0.,self.progress-.1)
        old=self.stage
        # A fast pass requests a 10s confirmation; it never promotes by itself.
        if full and passed and self.passes>=self.profile['consecutive_passes'] and elapsed-self.stage_started>=self.profile['min_stage_seconds']:
            self.stage=min(4,self.stage+1);self.passes=0;self.progress=0.
        if self.stage>0 and self.failures>=3 and self.success.get('standing',0)<.8:
            self.stage-=1;self.passes=0;self.progress=0.;self.failures=0
        if old!=self.stage:self.stage_started=elapsed
        self.evaluations.append(dict(elapsed=elapsed,stage=old,full=full,passed=passed,rows=rows))
        self.apply_stage();return old!=self.stage


def evaluation_cases(stage):
    rows=[('standing',Command()),('weight_shift',Command('short',0.,.9)),('short_walk',Command('short',.04,.95))]
    if stage>=1:rows += [('medium_walk',Command('medium',.12,1.)),('long_walk',Command('long',.18,.85)),('high_frequency_short',Command('short',.04,1.4))]
    if stage>=2:rows += [(f'turn_{sign:+d}',Command('turn',0.,1.,sign*.12)) for sign in (-1,1)]
    if stage>=3:rows += [(f'curve_{sign:+d}',Command('curve',.07,1.,sign*.12)) for sign in (-1,1)]
    return rows

class QuinticHeadingGoals(GoalSource):
    def __init__(self,kin,calibration,curriculum):
        super().__init__(None,kin,GoalSpec('foot_trajectory',10,.02),AlternatingConfig(),command_mode='curriculum')
        self.curriculum=curriculum;self.command_duration=12.;self.calibration=calibration
    def obs(self,mdp,S,a_prev,xi,f,command=None):
        if type(command) is not FootstepCommand:raise ValueError('Only immutable generated Foot commands allowed')
        shape=f.shape
        if f.ndim==2:
            _,H=shape;cmd=FootstepCommand(**{k:v.repeat_interleave(H,0) for k,v in command.tensors().items()})
            flat=tuple(x.flatten(0,1) for x in S);ticks=f.flatten()
        else:flat=S;ticks=f;cmd=command
        g=self.goal(flat,ticks,ticks,cmd).view(-1,10,23)
        angle=torch.atan2(torch.stack((g[:,:,5],g[:,:,16]),-1),torch.stack((g[:,:,3],g[:,:,14]),-1))
        weight=.25+.75*torch.stack((g[:,:,9],g[:,:,20]),-1)
        yaw=torch.atan2((angle.sin()*weight).sum((1,2)),(angle.cos()*weight).sum((1,2)))-self.calibration['bias_rad']
        e=torch.zeros((len(ticks),3),device=ticks.device);e[:,2]=yaw
        roll=torch.zeros_like(e);roll[:,0]=self.calibration['tilt'][0]
        pitch=torch.zeros_like(e);pitch[:,1]=self.calibration['tilt'][1]
        desired=yaw_mat(flat[1])@so3_exp(e)@so3_exp(pitch)@so3_exp(roll)
        if f.ndim==2:desired=desired.view(*shape,3,3)
        return mdp.obs(S,a_prev,xi,f,reference_R=desired)


def build(cfg,device,resume=None):
    path=Path(cfg['phase_a_checkpoint']).resolve();source=torch.load(path,map_location='cpu',weights_only=False)
    if source.get('phase')!='A' or source['iteration']!=625:raise ValueError('Expected verified Phase A best distillation checkpoint, never Phase B')
    contract=source['contract'];expected=GoalSpec('reference_foot',10,.02).contract()
    if contract['goal']!=expected or contract['physical']!=physical_contract() or not source['config']['causal_heading']:
        raise ValueError('Phase A time/physics/heading contract mismatch')
    state=source['foot'];decoder={k[4:]:v for k,v in state.items() if k.startswith('dec.')}
    original={k[4:]:v for k,v in source['motion'].items() if k.startswith('dec.')}
    if tensor_hash(decoder)!=tensor_hash(original):raise ValueError('Phase A Decoder differs from Teacher')
    kin=S54Kinematics(device);curriculum=Stage2Curriculum(cfg)
    goals=QuinticHeadingGoals(kin,contract['heading_calibration'],curriculum)
    mdp=FootTrackingMDP(kin,torch.tensor(default_joint_pos_list(),device=device),goals,resolve_foot_task(cfg['task']))
    policy=Policy(99,230,hist=cfg['hist'],obs_adapter='identity').to(device);policy.load_state_dict(state,strict=True)
    policy.requires_grad_(False);policy.enc.requires_grad_(True)
    wm=WorldModel().to(device);wm.load_state_dict(source['wm'],strict=True)
    critic=Critic(mdp.critic_dim+230).to(device)
    result=dict(kind='s54_stage2_curriculum_v1',source_sha256=digest(path),source_iteration=625,
        source_teacher_sha256=contract['teacher_sha256'],goal=goals.spec.contract(),heading=goals.calibration,
        physical=physical_contract(),task=mdp.contract(),hist=10,goal_norm='Copied Phase A; frozen',obs_norm='Copied Teacher/Phase A; frozen',
        norm_hash=tensor_hash({k:v for k,v in state.items() if k.startswith(('goal_norm.','obs_norm.'))}))
    if resume:
        if resume['stage2_contract']!=result:raise ValueError('Stage2 resume contract differs')
        policy.load_state_dict(resume['policy'],strict=True);wm.load_state_dict(resume['wm'],strict=True);critic.load_state_dict(resume['critic'],strict=True)
        curriculum.load_state_dict(resume['curriculum']);curriculum.apply_stage()
        if tensor_hash(policy.dec.state_dict())!=tensor_hash(decoder):raise ValueError('Resume changed frozen Decoder')
    return dict(kin=kin,curriculum=curriculum,goals=goals,mdp=mdp,policy=policy,wm=wm,critic=critic,contract=result,
        transfer=dict(source=str(path),source_sha256=digest(path),source_phase='A',decoder_frozen=True,motion_reference_loaded=False,
            loaded=['Foot Encoder','Teacher shared Decoder','Teacher WM and WM norms','Phase A GoalNorm','Teacher ObsNorm'],
            fresh=['Foot Critic','all optimizers','Replay'],normalizers_unchanged=True))

class Stage2Updates(JointUpdates):
    def __init__(self,parts,optimizers,buffer,cfg):
        options=dict(cfg,pi_batch_foot=cfg['pi_batch'])
        super().__init__(SimpleNamespace(foot=parts['policy'],motion=SimpleNamespace(enc=nn.Identity())),parts['wm'],
            {'foot':parts['critic']},optimizers,{'foot':parts['mdp']},{'foot':parts['goals']},{'foot':buffer},options)
    def wm_update(self):
        # Same single-buffer supervised multistep objective as Stage15.
        from .stage15_updates import Stage15Updates
        self.buffers['motion']=self.buffers['foot']
        try:return Stage15Updates.wm_update(self)
        finally:self.buffers.pop('motion')
