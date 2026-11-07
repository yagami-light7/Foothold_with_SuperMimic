"""Stage15 routing of original Motion-reward BPTT/GAE to E_f only.

Actor/critic math is from joint_updates.py; no latent regularizer or FootReward.
Frozen D and WM still differentiate through their inputs.
"""
import math
from types import SimpleNamespace
import torch
from torch import nn
from .mdp import unflatten_state,normalize_xi
from .s54 import NUM_J
from .control import bound_action,bound_bootstrap,ACTION_LIMIT
from .joint_models import frozen
from .joint_updates import JointUpdates,versions,snapshots,delta,gradnorm
from .buffer import ReplayBuffer
from .reference_foot import ReferenceFootCommand

class Stage15Replay(ReplayBuffer):
    def __init__(self,T,N,device):super().__init__(T,N,device,foot_commands=True)
    def add(self,tr):
        if type(tr.get('command')) is not ReferenceFootCommand:raise ValueError('Only original LAFAN reference commands')
        if not torch.equal(tr['f'],tr['command'].start_frame+tr['ep_step'].long()):raise ValueError('Reference/tick misalignment')
        super().add(tr)
    def sample_windows(self,B,k):
        t=torch.randint(self.size-k+1,(B,),device=self.device);env=torch.randint(self.N,(B,),device=self.device)
        rows=self._row(t[:,None]+torch.arange(k,device=self.device));ee=env[:,None].expand_as(rows)
        out={key:self.data[key][rows,ee] for key in ('s','s1','a','xi','done')}
        f=self.data['f'][rows,ee];tick=self.data['ep_step'][rows,ee];cid=self.data['command_id'][rows,ee]
        ok=(~out['done'][:,:-1])&(f[:,1:]==f[:,:-1]+1)&(tick[:,1:]==tick[:,:-1]+1)&(cid[:,1:]==cid[:,:-1])
        out['mask']=torch.cumprod(torch.cat((torch.ones(B,1,device=self.device),ok.float()),1),1)
        return out

class Stage15Updates(JointUpdates):
    def __init__(self,parts,critic,optimizers,buffer,cfg):
        options=dict(cfg,pi_batch_motion=cfg['pi_batch'])
        super().__init__(SimpleNamespace(motion=parts['fp'],foot=parts['mp']),parts['wm'],
            {'motion':critic},optimizers,{'motion':parts['paired']},{'motion':parts['goals']},{'motion':buffer},options)
    def valid_starts(self,name):
        # Existing sampler includes in-episode history. Exclude future clamping.
        old=self.mdps[name].motion.clip_end
        self.mdps[name].motion.clip_end=old-10
        try:return super().valid_starts(name)
        finally:self.mdps[name].motion.clip_end=old
    def wm_update(self):
        cfg=self.cfg;k=cfg['wm_window'];w=self.buffers['motion'].sample_windows(cfg['wm_batch'],k)
        S=unflatten_state(w['s'][:,0]);pred=[]
        for j in range(k):
            S=self.wm.step(S,w['a'][:,j],normalize_xi(w['xi'][:,j]));pred.append(S)
        P=tuple(torch.stack([s[i] for s in pred],1) for i in range(6))
        error=(self.wm.state_error(P,unflatten_state(w['s1']))/torch.maximum(self.wm.out_norm.std(),self.err_floor)).square().mean(-1)
        loss=(error*w['mask']).sum()/cfg['wm_batch']
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite WM')
        opt=self.optimizers['wm'];opt.zero_grad(set_to_none=True);loss.backward()
        gn=nn.utils.clip_grad_norm_(self.wm.parameters(),1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True)
        return dict(wm_loss=float(loss.detach()/k),wm_err8=float((error[:,-1].detach()*w['mask'][:,-1]).sum()/w['mask'][:,-1].sum().clamp_min(1)),wm_gradnorm=float(gn))

    def actor_critic(self,name='motion',audit=False):
        cfg=self.cfg;policy=getattr(self.policy,name);other=getattr(self.policy,'motion' if name=='foot' else 'foot')
        mdp=self.mdps[name];goals=self.goals[name];critic=self.critics[name];H=cfg['horizon'];B=cfg['pi_batch_'+name]
        opt=self.optimizers[name];buffer=self.buffers[name]
        s,h=self.valid_starts(name);S=unflatten_state(s['s']);S0=S;f=s['f'];step=s['ep_step'];a_prev=s['a_prev']
        command=buffer.commands(s['command_id'])
        xi=s['xi'];xi_n=normalize_xi(xi)
        with torch.no_grad():history=mdp.add_noise(goals.obs(mdp,unflatten_state(h['s']),h['a_prev'],h['xi'],h['f'],command))
        goal=goals.goal(S,f,step,command);g0=goal;states=[];frames=[];ticks=[];goals_all=[];actions=[];previous=[];clipped=[]
        wm_version=versions(self.wm);other_version=versions(other.enc);decoder_version=versions(policy.dec)
        D_before=snapshots(policy.dec) if audit else None;E_before=snapshots(policy.enc) if audit else None
        for o in self.optimizers.values():o.zero_grad(set_to_none=True)
        frozen_modules=[self.wm,*self.critics.values(),other.enc]
        frozen_modules.append(policy.dec)
        params=tuple(policy.enc.parameters())
        upper=None
        gradient_stats={}
        with frozen(*frozen_modules):
            first_z=None
            for t in range(H):
                z=policy.encode(goal)
                if t==0:z.retain_grad();first_z=z
                raw=policy.decode(history,z);action=bound_action(raw)
                previous.append(a_prev);a_prev=action;clipped.append((raw.abs()>ACTION_LIMIT).float().mean())
                S=self.wm.step(S,action,xi_n);f=f+1;step=step+1
                o=mdp.add_noise(goals.obs(mdp,S,action,xi,f,command))
                history=torch.cat((history[:,1:],o[:,None]),1);goal=goals.goal(S,f,step,command)
                states.append(S);frames.append(f);ticks.append(step);goals_all.append(goal);actions.append(action)
            flat=lambda xs:torch.stack(xs).flatten(0,1)
            S_all=tuple(flat([st[i] for st in states]) for i in range(6));f_all=flat(frames);steps_all=flat(ticks)
            cmd_all=command.repeat(H) if command is not None else None
            reward,terms=mdp.reward(S_all,f_all,flat(actions),flat(previous),True,command=cmd_all,ep_step=steps_all)
            done=mdp.soft_termination(S_all,f_all,command=cmd_all,ep_step=steps_all)
            reward,done=reward.view(H,B),done.view(H,B)
            alive=torch.cumprod(1-done,0);c_t=torch.cat((torch.ones(1,B,device=self.device),alive[:-1]))
            disc=cfg['gamma']**torch.arange(H,device=self.device)[:,None]
            reward_J=(disc*c_t*reward).sum(0)
            x=torch.cat((mdp.critic_features(S,f,xi_n,command=command,ep_step=step),goal),-1)
            raw_value=critic(x);bootstrap=cfg['gamma']**H*alive[-1]*bound_bootstrap(raw_value,upper)
            loss=-(reward_J+bootstrap).mean()
            if audit:
                rg=torch.autograd.grad(-reward_J.mean(),params,retain_graph=True,allow_unused=True)
                bg=torch.autograd.grad(-bootstrap.mean(),params,retain_graph=True,allow_unused=True)
                rn,bn=gradnorm(rg,self.device),gradnorm(bg,self.device)
                dot=sum(((x.detach()*y.detach()).sum() for x,y in zip(rg,bg) if x is not None and y is not None),torch.zeros((),device=self.device))
                gradient_stats.update(reward_gradnorm=float(rn),bootstrap_gradnorm=float(bn),grad_cosine=float(dot/(rn*bn).clamp_min(1e-20)))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite '+name+' Actor loss')
            loss.backward()
            z_grad=float(first_z.grad.detach().norm()) if first_z.grad is not None else 0.
            if not math.isfinite(z_grad) or z_grad==0:raise RuntimeError(name+' latent gradient is missing/nonfinite/zero')
            gn=nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);nn.utils.clip_grad_value_(params,.1);opt.step()
            if versions(self.wm)!=wm_version or any(p.grad is not None for p in self.wm.parameters()):raise RuntimeError('Actor mutated WM')
            if versions(other.enc)!=other_version or any(p.grad is not None for p in other.enc.parameters()):raise RuntimeError('Actor crossed Encoder boundary')
            if (versions(policy.dec)!=decoder_version or any(p.grad is not None for p in policy.dec.parameters())):raise RuntimeError('Foot changed Shared Decoder')
        stats=dict(actor_loss=float(loss.detach()),imagined_reward=float((c_t.detach()*reward.detach()).sum(0).mean()/H/mdp.max_step_reward),
                   imagined_survival=float(alive[-1].detach().mean()),actor_gradnorm=float(gn),latent_gradnorm=z_grad,
                   action_rms=float(flat(actions).detach().square().mean().sqrt()),action_clip_fraction=float(torch.stack(clipped).mean()),
                   actor_bootstrap=float(bootstrap.detach().mean()),**gradient_stats)
        if audit:
            stats.update(delta_decoder=delta(policy.dec,D_before),delta_encoder=delta(policy.enc,E_before),delta_wm_actor=0.)
            if stats['delta_decoder']!=0.:raise RuntimeError('Foot Decoder checksum drift')
        opt.zero_grad(set_to_none=True)
        # Preserve original imagined lambda-return Critic algorithm and reward units.
        with torch.no_grad():
            states_c=tuple(torch.cat((S0[i][None],torch.stack([st[i].detach() for st in states]))).flatten(0,1) for i in range(6))
            frames_c=torch.cat((s['f'][None],torch.stack(frames))).flatten()
            steps_c=torch.cat((s['ep_step'][None],torch.stack(ticks))).flatten()
            goals_c=torch.cat((g0[None],torch.stack(goals_all))).flatten(0,1)
            context=xi_n[None].expand(H+1,-1,-1).flatten(0,1)
            cmd_c=command.repeat(H+1) if command is not None else None
            X=torch.cat((mdp.critic_features(states_c,frames_c,context,command=cmd_c,ep_step=steps_c),goals_c),-1).view(H+1,B,-1)
        critic.norm.update(X)
        with torch.no_grad():
            V=bound_bootstrap(critic(X),upper);ret=V[-1];targets=[]
            for t in range(H-1,-1,-1):
                ret=reward[t].detach()+cfg['gamma']*(1-done[t].detach())*((1-cfg['lam'])*V[t+1]+cfg['lam']*ret)
                targets.append(ret)
            Y=torch.stack(targets[::-1]).flatten();X=X[:-1].flatten(0,1)
        cv=self.optimizers['critic_'+name];mb=len(Y)//cfg['critic_minibatches'];losses=[]
        for _ in range(cfg['critic_epochs']):
            perm=torch.randperm(len(Y),device=self.device)
            for j in range(cfg['critic_minibatches']):
                ids=perm[j*mb:(j+1)*mb];v_loss=(critic(X[ids])-Y[ids]).square().mean()
                if not torch.isfinite(v_loss):raise FloatingPointError('Nonfinite '+name+' Critic')
                cv.zero_grad(set_to_none=True);v_loss.backward();nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);cv.step()
                losses.append(float(v_loss.detach()))
        cv.zero_grad(set_to_none=True);stats['critic_loss']=sum(losses)/len(losses)
        return stats
