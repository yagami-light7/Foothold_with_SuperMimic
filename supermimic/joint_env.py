"""One Newton scene, disjoint Motion/Foot environment partitions and clocks.

Physics/control/reset implementations reuse TrackingSim. Only task routing and
bookkeeping differ; no teleporting of a transition across interface resets.
"""
from contextlib import contextmanager
import torch
from .env import TrackingSim
from .mdp import flatten_state
from .motion import CONTROL_DT
from .s54 import NUM_J,JOINT_OFFSET,COM_OFFSET


class JointScene:
    def __init__(self, sim_cfg, counts, motion, mdps, goals, gamma=.99):
        self.counts,self.mdps,self.goals,self.motion=counts,mdps,goals,motion
        self.device=sim_cfg.device; self.gamma=gamma
        nm,nf=counts['motion'],counts['foot']; self.N=nm+nf
        self.ids={'motion':torch.arange(nm,device=self.device),
                  'foot':torch.arange(nm,nm+nf,device=self.device)}
        # One and only one SimulationContext/Scene/solver. Initial construction
        # is nonrandomized, then the original Motion domain distribution is set.
        self.base=TrackingSim(sim_cfg,self.N,motion,mdps['motion'],gamma=gamma,eval_mode=True,goals=goals['motion'])
        self.base.eval_mode=False
        self.base._setup_friction()  # ground mu=.3; Foot's own mu remains 1
        self.default_com=self.base.robot.data.body_com_pose_b.torch[:,self.base.torso_id,:3].clone()
        self.training_xi=self.base.xi.clone(); self.training_xi[:,0]=1.
        self.training_xi[:nm,0].uniform_(.3,1.2)
        self.training_xi[:nm,COM_OFFSET]=torch.stack([torch.empty(nm,device=self.device).uniform_(-v,v) for v in (.025,.05,.05)],-1)
        self.set_context(self.training_xi)
        self.base.command=goals['foot'].new_command(mdps['foot'].nominal_state(self.N))
        self.evaluation=False
        self.reset_all()

    def set_context(self, xi):
        self.base.xi.copy_(xi)
        com=self.default_com+self.base.xi[:,COM_OFFSET]
        self.base.robot.set_coms_index(coms=com.unsqueeze(1),body_ids=[self.base.torso_id])
        self.base._apply_friction()

    def state(self,name=None):
        state=self.base.state()
        return tuple(x[self.ids[name]] for x in state) if name else state

    def command(self,name):
        return self.base.command.index(self.ids[name]).clone() if name=='foot' else None

    def reset(self,name,local_ids=None,frames=None,noise=True):
        b=self.base; ids=self.ids[name] if local_ids is None else self.ids[name][local_ids]
        previous=(b.mdp,b.goals,b.reset_noise,b.randomize)
        try:
            b.mdp,b.goals=self.mdps[name],self.goals[name]
            b.reset_noise=noise; b.randomize=name=='motion' and not self.evaluation
            b.reset(ids,frames)
        finally:b.mdp,b.goals,b.reset_noise,b.randomize=previous

    def reset_all(self):
        for name in self.ids:self.reset(name)

    def step(self, actions):
        b=self.base; S=b.state(); f=b.f.clone(); ticks=b.ep_step.clone()
        previous=b.a_prev.clone(); xi=b.xi.clone(); command=self.command('foot')
        cids=b.command_id[self.ids['foot']].clone()
        a=torch.cat((actions['motion'],actions['foot']))
        target=b.pd_target(a,xi)
        b.robot.set_joint_position_target_index(target=target[:,b.sim_to_canonical])
        b.scene.write_data_to_sim(); b.sim.step(render=False); b.scene.update(CONTROL_DT)
        S1=b.state(); b.f=f+1; b.ep_step=ticks+1; b.a_prev=a.clone()
        out={}
        for name,ids in self.ids.items():
            mdp=self.mdps[name]; s=tuple(x[ids] for x in S); s1=tuple(x[ids] for x in S1)
            cmd=command if name=='foot' else None
            r,terms=mdp.reward(s1,f[ids]+1,a[ids],previous[ids],terms=True,command=cmd,ep_step=ticks[ids]+1)
            term=mdp.termination(s1,f[ids]+1,command=cmd,ep_step=ticks[ids]+1)
            if not all(torch.isfinite(x).all() for x in s1):raise FloatingPointError('Nonfinite Newton state: '+name)
            timeout=ticks[ids]+1>=500
            clip_end=torch.zeros_like(timeout)
            if name=='motion':
                clip_end=f[ids]+1>=self.motion.clip_end[self.motion.clip_id[f[ids]]]
                timeout=timeout|clip_end
            done=term|timeout
            b.ep_disc_ret[ids]+=(self.gamma**ticks[ids].float())*r; b.ep_ret[ids]+=r
            tr=dict(s=flatten_state(s),s1=flatten_state(s1),a=a[ids].clone(),a_prev=previous[ids],xi=xi[ids],
                    f=f[ids],ep_step=ticks[ids],r=r,term=term,timeout=timeout,done=done,
                    clip_end=clip_end,terms=terms,pd_target=target[ids],state1=s1)
            if name=='foot':tr.update(command=cmd,command_id=cids)
            completed=done.nonzero().flatten()
            if len(completed):
                glob=ids[completed]; T=b.ep_step[glob].float()
                tr.update(ep_len=T.clone(),ep_ret_norm=(b.ep_ret[glob]/(T*mdp.max_step_reward)).clone())
                if name=='motion' and not self.evaluation:
                    boot=torch.where(term[completed],0.,(self.gamma**T)*(b.ep_ret[glob]/T)/(1-self.gamma))
                    ret=b.ep_disc_ret[glob]+boot; bins=b.start_bin[glob]
                    b.bin_G.index_put_((bins,),.8*b.bin_G[bins]+.2*ret)
                self.reset(name,completed,noise=not self.evaluation or name=='foot')
            out[name]=tr
        return out

    def checkpoint(self):
        b=self.base
        return dict(state=b.state(),f=b.f,ep_step=b.ep_step,a_prev=b.a_prev,xi=b.xi,
                    command=b.command.tensors(),command_id=b.command_id,next_command_id=b._next_command_id,
                    ep_ret=b.ep_ret,ep_disc_ret=b.ep_disc_ret,start_bin=b.start_bin,bin_G=b.bin_G,bin_w_avg=b.bin_w_avg)

    def restore(self, state):
        from .rotations import mat_to_quat
        from .foot_trajectory import FootstepCommand
        b=self.base; S=[x.to(self.device) for x in state['state']]
        p,R,v,w,q,qd=S; quat=mat_to_quat(R)
        ids=torch.arange(self.N,device=self.device)
        b.robot.write_root_link_pose_to_sim_index(root_pose=torch.cat((p+b.scene.env_origins,quat[:,1:],quat[:,:1]),-1),env_ids=ids)
        b.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v,w),-1),env_ids=ids)
        b.robot.write_joint_position_to_sim_index(position=q[:,b.sim_to_canonical],env_ids=ids)
        b.robot.write_joint_velocity_to_sim_index(velocity=qd[:,b.sim_to_canonical],env_ids=ids)
        self.set_context(state['xi'].to(self.device))
        for key in ['f','ep_step','a_prev','command_id','ep_ret','ep_disc_ret','start_bin','bin_G','bin_w_avg']:
            getattr(b,key).copy_(state[key].to(self.device))
        b.command=FootstepCommand(**{k:v.to(self.device) for k,v in state['command'].items()})
        b._next_command_id=state['next_command_id']; b.refresh()
        b.robot.set_joint_position_target_index(target=b.pd_target(b.a_prev)[:,b.sim_to_canonical])
