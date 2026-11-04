"""S54 command curriculum. Frequency = alternating landings/s (not gait cycles/s).

All plans are immutable episode schedules. Changes are sampled ahead of time at
completed double support boundaries; replay and BPTT see the very same future.
The envelope is a conservative trajectory filter, NOT a dynamics certificate.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import math
import random

import torch
from .foot_trajectory import FootstepCommand, command_time

TASKS = ('standing', 'short', 'medium', 'long', 'turn', 'curve')
DEFAULTS = dict(
    version=1, seed=5401, budget_seconds=28800, checkpoint_seconds=900,
    eval_seconds=900, eval_fraction=.05, stage_hours=[.8,1.6,2.,1.2,1.6,.8],
    weights=[[1,0,0,0,0,0],[.2,.8,0,0,0,0],[.1,.50,.32,.08,0,0],
             [.1,.18,.10,.02,.6,0],[.1,.22,.15,.03,.2,.3],[.1,.20,.15,.05,.2,.3]],
    length_bins=[[.02,.08],[.08,.20],[.20,.40]],
    frequency_bins=[[.25,.8],[.8,1.3],[1.3,2.]],
    yaw_limits=[0,0,0,.25,.40,.50], segment_seconds=4., switch_probability=.35,
    max_speed=.45, max_swing_speed=1.8, max_swing_accel=12., max_separation=.65,
    max_yaw_step=.25, swing_fraction=.72, min_swing=.32, min_support=.12,
    clearance=.04, success_threshold=.85, standing_threshold=.95, rmse_threshold=.045,
    consecutive_passes=2, min_stage_seconds=600, keep_checkpoints=3)


@dataclass(frozen=True)
class Command:
    task: str = 'standing'
    length: float = 0.
    frequency: float = 0.
    yaw_rate: float = 0.

    @property
    def bucket(self):
        f = 'low' if self.frequency < .8 else 'mid' if self.frequency < 1.3 else 'high'
        return f'{self.task}_{f}'


def feasibility(c, cfg=DEFAULTS, width=.24):
    values = (c.length,c.frequency,c.yaw_rate)
    if not all(math.isfinite(x) for x in values): return 'nonfinite'
    if not 0 <= c.length <= .4 or not 0 <= c.frequency <= 2: return 'range'
    if c.frequency == 0:
        return None if c.length == 0 and c.yaw_rate == 0 else 'zero_frequency_requires_standing'
    period = 1/c.frequency
    swing = cfg['swing_fraction']*period
    if swing < cfg['min_swing'] or period-swing < cfg['min_support']: return 'timing'
    delta = abs(c.yaw_rate)/c.frequency
    if delta > cfg['max_yaw_step']: return 'yaw_per_step'
    if c.length*c.frequency > cfg['max_speed']: return 'speed'
    displacement = 2*c.length + width*delta
    if math.hypot(c.length,width) > cfg['max_separation']: return 'separation'
    if 1.875*displacement/swing > cfg['max_swing_speed']: return 'swing_speed'
    # max quintic acceleration = 10/sqrt(3); vertical bump uses two half swings.
    if max(5.774*displacement,23.095*cfg['clearance'])/swing**2 > cfg['max_swing_accel']:
        return 'swing_acceleration'
    return None


class Curriculum:
    def __init__(self, config=None):
        self.cfg = {**DEFAULTS, **(config or {})}
        if len(self.cfg['weights']) != 6 or any(len(w)!=6 or min(w)<0 or sum(w)<=0 for w in self.cfg['weights']):
            raise ValueError('Expected six nonnegative six-task distributions')
        if not 0 < self.cfg['budget_seconds'] <= 28800: raise ValueError('Budget must be <=8 h')
        if not 0 < self.cfg['swing_fraction'] < 1 or self.cfg['segment_seconds'] <= 0:
            raise ValueError('Invalid timing')
        for key in ('stage_hours','yaw_limits'):
            if len(self.cfg[key])!=6: raise ValueError(f'{key} must contain six stage entries')
        for key in ('length_bins','frequency_bins'):
            if len(self.cfg[key])!=3 or any(len(r)!=2 or not all(math.isfinite(x) for x in r)
                or r[0]<0 or r[1]<=r[0] for r in self.cfg[key]): raise ValueError(f'Invalid {key}')
        if not 0<=self.cfg['switch_probability']<=1 or not 0<self.cfg['eval_fraction']<=.05:
            raise ValueError('switch_probability in [0,1]; eval_fraction in (0,.05]')
        self.rng = random.Random(self.cfg['seed'])
        self.stage = 0; self.progress = 0.; self.stage_started = 0.; self.passes = 0; self.failures = 0
        self.success = {}; self.bucket_success = {}; self.counts = Counter(); self.rejected = Counter(); self.hist = Counter()
        self.executed = Counter(); self.best = {}; self.last_eval = 0.; self.last_save = 0.; self.eval_spent = 0.

    def state_dict(self):
        return {k:v for k,v in self.__dict__.items() if k!='rng'} | {'rng':self.rng.getstate()}

    def load_state_dict(self, state):
        if state['cfg'] != self.cfg: raise ValueError('Curriculum config mismatch on resume')
        for k,v in state.items():
            if k!='rng': setattr(self,k,v)
        self.rng.setstate(state['rng'])

    def sample(self):
        weights = list(self.cfg['weights'][self.stage])
        # Keep an easy-task floor; only reduce hard buckets when independent eval fails.
        for i in (2,3,4,5): weights[i] *= .25 + .75*self.success.get(TASKS[i],.5)
        task = self.rng.choices(TASKS,weights)[0]
        for _ in range(100):
            if task == 'standing': c = Command()
            else:
                b = {'short':0,'medium':1,'long':2}.get(task,0)
                cap = min(1., .25+.75*self.progress) if self.stage==2 else 1.
                lo,hi = self.cfg['length_bins'][b]
                length = 0. if task=='turn' or (task=='short' and self.stage>=2 and self.rng.random()<.05) else self.rng.uniform(lo,lo+(hi-lo)*cap)
                if self.stage==1: freq = self.rng.uniform(.5,.8)
                else:
                    # Stratified frequency bins; joint feasibility is rejection sampled.
                    fw=[w*(.25+.75*self.bucket_success.get(f'{task}_{label}',.5))
                        for w,label in zip((.4,.4,.2),('low','mid','high'))]
                    fbin = self.rng.choices(range(3),fw)[0]
                    flo,fhi = self.cfg['frequency_bins'][fbin]
                    freq = self.rng.uniform(flo,fhi)
                limit = self.cfg['yaw_limits'][self.stage]*(.4+.6*self.progress)
                yaw = self.rng.choice([-1,1])*self.rng.uniform(.04,max(.04,limit)) if task in ('turn','curve') else 0.
                c = Command(task,length,freq,yaw)
            why = feasibility(c,self.cfg)
            if why is None: break
            self.rejected[why] += 1
        else:
            raise RuntimeError(f'No feasible command for {task}; check curriculum bounds')
        self.counts[c.task] += 1
        self.hist[f'length_{int(c.length/.05):02d}'] += 1
        self.hist[f'frequency_{int(c.frequency/.25):02d}'] += 1
        self.hist[f'yaw_{math.floor(c.yaw_rate/.1):+03d}'] += 1
        self.hist[c.bucket] += 1
        return c

    def update(self, metrics, elapsed, allow_promotion=True, bucket_metrics=None):
        """Only complete, physical evaluations may call this. Never WM loss."""
        for bucket,m in (bucket_metrics or {}).items(): self.bucket_success[bucket]=m['success_rate']
        for task, m in metrics.items():
            self.success[task] = m['success_rate']
        required = [t for t,w in zip(TASKS,self.cfg['weights'][self.stage]) if w>0]
        passed = all(t in metrics and metrics[t]['success_rate'] >=
                     (self.cfg['standing_threshold'] if t=='standing' else self.cfg['success_threshold'])
                     and metrics[t]['foot_rmse'] <= self.cfg['rmse_threshold'] for t in required)
        self.passes = self.passes+1 if passed else 0
        self.failures = 0 if passed else self.failures+1
        self.progress = min(1.,self.progress+.2) if passed else max(0.,self.progress-.1)
        old = self.stage
        if self.stage>0 and self.failures>=3 and self.success.get('standing',0)<.8:
            self.stage-=1; self.progress=0.; self.failures=0
        elif allow_promotion and self.passes>=self.cfg['consecutive_passes'] and elapsed-self.stage_started>=self.cfg['min_stage_seconds']:
            self.stage=min(5,self.stage+1); self.passes=0; self.progress=0.
        if old!=self.stage: self.stage_started=elapsed
        return old!=self.stage

    def scalars(self):
        n=max(1,sum(self.counts.values()))
        return {'Curriculum/Stage':self.stage,'Curriculum/Progress':self.progress,
                **{f'Commands/Fraction/{k}':self.counts[k]/n for k in TASKS},
                **{f'Commands/ScheduledHistogram/{k}':v for k,v in self.hist.items()},
                **{f'Commands/Executed/{k}':v for k,v in self.executed.items()},
                **{f'Commands/Rejected/{k}':v for k,v in self.rejected.items()},
                **{f'Curriculum/Success/{k}':v for k,v in self.success.items()},
                **{f'Curriculum/BucketSuccess/{k}':v for k,v in self.bucket_success.items()}}

    def observe(self, command, steps, dt=.02):
        if command.labels is None: return
        table=command.labels
        selected=(table[...,0] <= command_time(steps,dt,table.dtype)[:,None]).sum(-1).sub(1).clamp_min(0)
        rows=table[torch.arange(len(steps),device=steps.device),selected].detach().cpu().tolist()
        for _,length,freq,yaw,task in rows:
            self.executed['task_'+TASKS[int(task)]]+=1
            self.executed[f'length_{int(length/.05):02d}']+=1
            self.executed[f'frequency_{int(freq/.25):02d}']+=1
            self.executed[f'yaw_{math.floor(yaw/.1):+03d}']+=1

    def plan(self, initial_pos, initial_quat, heading, duration=12., fixed=None):
        """Each environment gets a separate plan; no mutation of active landing targets.

        A segment ends only after a complete left/right pair. This prevents
        stranded feet when switching to Standing. Goal lookahead sees the switch.
        """
        B=initial_pos.shape[0]; M=64
        plans=[]
        for b in range(B):
            p=initial_pos[b].detach().cpu().clone(); q=initial_quat[b].detach().cpu().clone()
            center=p[:,:2].mean(0); offsets=p[:,:2]-center
            yaw0=2*math.atan2(float(heading[b,3]),float(heading[b,0])); yaw=yaw0
            # offsets are stored in the starting heading frame.
            rot=torch.tensor([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
            offsets=offsets@rot
            t=.2; events=[]; c=fixed or self.sample(); segment_end=t+self.cfg['segment_seconds']; side=self.rng.randrange(2)
            def safe_switch(candidate):
                if candidate.frequency==0: return True
                cc=center.clone(); yy=yaw; ff=side; prev=p.clone()
                period=1/candidate.frequency; swing=self.cfg['swing_fraction']*period
                for _ in range(2):
                    delta=candidate.yaw_rate*period; mid=yy+.5*delta
                    cc=cc+cc.new_tensor([math.cos(mid),math.sin(mid)])*candidate.length; yy+=delta
                    rr=cc.new_tensor([[math.cos(yy),-math.sin(yy)],[math.sin(yy),math.cos(yy)]])
                    endpoint=cc+rr@offsets[ff]
                    distance=float((endpoint-prev[ff,:2]).norm())
                    if (1.875*distance/swing>self.cfg['max_swing_speed'] or
                        5.774*distance/swing**2>self.cfg['max_swing_accel'] or
                        float((endpoint-prev[1-ff,:2]).norm())>self.cfg['max_separation']): return False
                    prev[ff,:2]=endpoint; ff=1-ff
                return True
            labels=[(0.,c.length,c.frequency,c.yaw_rate,TASKS.index(c.task))]
            pair=0
            while t < duration:
                if c.frequency==0:
                    t=segment_end
                    if fixed is not None: break
                else:
                    period=1/c.frequency; swing=self.cfg['swing_fraction']*period
                    dyaw=c.yaw_rate*period
                    # SE(2) exponential integration avoids a fixed straight travel axis.
                    mid=yaw+.5*dyaw
                    center=center+center.new_tensor([math.cos(mid),math.sin(mid)])*c.length
                    yaw+=dyaw
                    rot=center.new_tensor([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
                    target=torch.cat((center+rot@offsets[side],p[side,2:3]))
                    tq=q.new_tensor([math.cos(yaw/2),0,0,math.sin(yaw/2)])
                    events.append((side,target,tq,t,t+swing,self.cfg['clearance']))
                    p[side]=target; q[side]=tq
                    side=1-side; pair+=1; t+=period
                if t>=segment_end and pair%2==0 and fixed is None:
                    if self.rng.random()<self.cfg['switch_probability']:
                        candidate=self.sample()
                        if safe_switch(candidate): c=candidate
                        else: self.rejected['transition_envelope']+=1
                        labels.append((t,c.length,c.frequency,c.yaw_rate,TASKS.index(c.task)))
                    segment_end=t+self.cfg['segment_seconds']
                if len(events)>=M: raise ValueError('Episode plan exceeds padded event capacity')
            def zeros(*shape): return initial_pos.new_zeros(*shape)
            foot=torch.zeros(1,M,dtype=torch.long,device=initial_pos.device)
            pos=zeros(1,M,3); quat=zeros(1,M,4); quat[...,0]=1
            lift=zeros(1,M); touch=zeros(1,M); height=zeros(1,M)
            for i,(s,pp,qq,st,en,h) in enumerate(events):
                foot[0,i]=s; pos[0,i]=pp.to(pos); quat[0,i]=qq.to(quat)
                lift[0,i]=st; touch[0,i]=en; height[0,i]=h
            label_table=zeros(1,M,5); label_table[...,0]=1e9
            label_table[0,:len(labels)]=initial_pos.new_tensor(labels)
            plans.append(FootstepCommand(initial_pos[b:b+1].clone(),initial_quat[b:b+1].clone(),foot,pos,quat,
                         lift,touch,height,torch.tensor([len(events)],device=foot.device),heading[b:b+1].clone(),label_table))
        return FootstepCommand.cat(plans).validate()


def evaluation_commands(stage, full=False):
    """Fixed independent grid. Rejected cells are reported, never silently clamped."""
    out=[Command()]
    if stage>=1: out += [Command('short',.04,.65)]
    if full and stage>=1: out += [Command('short',.07,.75)]
    if stage>=2: out += [Command('medium',.12,.9),Command('long',.24,.6)]
    if stage>=3: out += [Command('turn',0,.8,s*.12) for s in (-1,1)]
    if stage>=4: out += [Command('curve',.06,.8,s*.12) for s in (-1,1)]
    if full and stage>=2:
        out += [Command('short' if l<.08 else 'medium' if l<.2 else 'long',l,f)
                for l in (0.,.05,.15,.3,.4) for f in (.5,1.,1.5,2.)]
    if full and stage>=3: out += [Command('turn',0,1.,s*.35) for s in (-1,1)]
    if full and stage>=4: out += [Command('curve',.12,1.,s*.25) for s in (-1,1)]
    return out
