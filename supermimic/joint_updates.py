"""Original SuperMimic WM/BPTT/GAE operations, routed by interface.

No goal enters WM. A Foot optimizer owns only E_f; D remains differentiable
with respect to its input while its parameters are frozen in that backward.
"""
import math
import torch
from torch import nn
from .mdp import unflatten_state,normalize_xi
from .s54 import NUM_J
from .control import bound_action,bound_bootstrap,return_upper_bound,ACTION_LIMIT
from .joint_models import frozen


def versions(module):return tuple(p._version for p in module.parameters())
def snapshots(module):return tuple(p.detach().clone() for p in module.parameters())
def delta(module,before):return float(sum((p.detach()-q).square().sum() for p,q in zip(module.parameters(),before)).sqrt())
def gradnorm(gs,device):
    return sum((g.detach().float().square().sum() for g in gs if g is not None),torch.zeros((),device=device)).sqrt()


class JointUpdates:
    def __init__(self,policy,wm,critics,optimizers,mdps,goals,buffers,cfg):
        self.policy,self.wm,self.critics,self.optimizers=policy,wm,critics,optimizers
        self.mdps,self.goals,self.buffers,self.cfg=mdps,goals,buffers,cfg
        self.device=next(wm.parameters()).device
        self.err_floor=torch.cat((torch.full((3,),3e-3),torch.full((3,),5e-3),torch.full((3,),2e-2),
            torch.full((3,),5e-2),torch.full((NUM_J,),3e-3),torch.full((NUM_J,),5e-2))).to(self.device)

    def wm_update(self):
        cfg=self.cfg; half=cfg['wm_batch']//2;k=cfg['wm_window']
        windows={name:b.sample_windows(half,k) for name,b in self.buffers.items()}
        w={key:torch.cat([windows[name][key] for name in ('motion','foot')]) for key in ['s','s1','a','xi','mask']}
        S=unflatten_state(w['s'][:,0]);predictions=[]
        for j in range(k):
            S=self.wm.step(S,w['a'][:,j],normalize_xi(w['xi'][:,j]));predictions.append(S)
        predicted=tuple(torch.stack([s[i] for s in predictions],1) for i in range(6))
        std=torch.maximum(self.wm.out_norm.std(),self.err_floor)
        error=(self.wm.state_error(predicted,unflatten_state(w['s1']))/std).square().mean(-1)
        loss=(error*w['mask']).sum()/cfg['wm_batch']
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite mixed WM loss')
        opt=self.optimizers['wm'];opt.zero_grad(set_to_none=True);loss.backward()
        gn=nn.utils.clip_grad_norm_(self.wm.parameters(),1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True)
        result={'wm_gradnorm':float(gn)}
        for i,name in enumerate(('motion','foot')):
            e=error[i*half:(i+1)*half].detach();mask=w['mask'][i*half:(i+1)*half]
            result[name]=dict(wm_loss=float((e*mask).sum()/half/k),wm_err8=float((e[:,-1]*mask[:,-1]).sum()/mask[:,-1].sum().clamp_min(1)),
                              sampled_windows=half,valid_prediction_steps=int(mask.sum()))
        return result

    def valid_starts(self,name):
        B=self.cfg['pi_batch_'+name];hist=self.cfg['hist'];H=self.cfg['horizon']
        buffer=self.buffers[name];starts=[];histories=[];n=0
        for _ in range(1000):
            s,h=buffer.sample_starts(2*B,hist)
            if name=='foot':ok=s['ep_step']+H<=self.mdps[name].cfg.episode_steps
            else:
                motion=self.goals[name].motion
                ok=s['f']+H<=motion.clip_end[motion.clip_id[s['f']]]
            idx=ok.nonzero().flatten()[:B-n]
            if len(idx):
                starts.append({k:v[idx] for k,v in s.items()});histories.append({k:v[idx] for k,v in h.items()});n+=len(idx)
            if n==B:break
        if n!=B:raise RuntimeError('Insufficient valid '+name+' Replay starts')
        cat=lambda rows:{k:torch.cat([r[k] for r in rows]) for k in rows[0]}
        return cat(starts),cat(histories)

    def actor_critic(self,name,audit=False):
        cfg=self.cfg;policy=getattr(self.policy,name);other=getattr(self.policy,'motion' if name=='foot' else 'foot')
        mdp=self.mdps[name];goals=self.goals[name];critic=self.critics[name];H=cfg['horizon'];B=cfg['pi_batch_'+name]
        opt=self.optimizers[name];buffer=self.buffers[name]
        s,h=self.valid_starts(name);S=unflatten_state(s['s']);S0=S;f=s['f'];step=s['ep_step'];a_prev=s['a_prev']
        command=buffer.commands(s['command_id']) if name=='foot' else None
        xi=s['xi'];xi_n=normalize_xi(xi)
        with torch.no_grad():history=mdp.add_noise(goals.obs(mdp,unflatten_state(h['s']),h['a_prev'],h['xi'],h['f'],command))
        goal=goals.goal(S,f,step,command);g0=goal;states=[];frames=[];ticks=[];goals_all=[];actions=[];previous=[];clipped=[]
        wm_version=versions(self.wm);other_version=versions(other.enc);decoder_version=versions(policy.dec)
        D_before=snapshots(policy.dec) if audit else None;E_before=snapshots(policy.enc) if audit else None
        for o in self.optimizers.values():o.zero_grad(set_to_none=True)
        frozen_modules=[self.wm,*self.critics.values(),other.enc]
        if name=='foot':frozen_modules.append(policy.dec)
        params=tuple(policy.enc.parameters())+(tuple(policy.dec.parameters()) if name=='motion' else ())
        upper=return_upper_bound(mdp.max_step_reward,cfg['gamma']) if name=='foot' else None
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
            if name=='foot':
                foot_loss=-(disc*c_t.detach()*terms['foot_objective'].view(H,B)).sum(0).mean()
                fg=torch.autograd.grad(foot_loss,params,retain_graph=True,allow_unused=True)
                gradient_stats['pure_foot_gradnorm']=float(gradnorm(fg,self.device))
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
            if name=='foot' and (versions(policy.dec)!=decoder_version or any(p.grad is not None for p in policy.dec.parameters())):raise RuntimeError('Foot changed Shared Decoder')
            if name=='motion' and versions(policy.dec)==decoder_version:raise RuntimeError('Motion failed to update Shared Decoder')
        stats=dict(actor_loss=float(loss.detach()),imagined_reward=float((c_t.detach()*reward.detach()).sum(0).mean()/H/mdp.max_step_reward),
                   imagined_survival=float(alive[-1].detach().mean()),actor_gradnorm=float(gn),latent_gradnorm=z_grad,
                   action_rms=float(flat(actions).detach().square().mean().sqrt()),action_clip_fraction=float(torch.stack(clipped).mean()),
                   actor_bootstrap=float(bootstrap.detach().mean()),**gradient_stats)
        if audit:
            stats.update(delta_decoder=delta(policy.dec,D_before),delta_encoder=delta(policy.enc,E_before),delta_wm_actor=0.)
            if name=='foot' and stats['delta_decoder']!=0.:raise RuntimeError('Foot Decoder checksum drift')
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
