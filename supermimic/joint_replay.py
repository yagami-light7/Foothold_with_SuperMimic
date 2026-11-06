"""Interface-specific replay schemas for joint training; executed action is a."""
import torch
from .buffer import ReplayBuffer
from .foot_trajectory import FootstepCommand


class JointReplay(ReplayBuffer):
    def __init__(self,T,N,device,interface):
        if interface not in ('motion','foot'):raise ValueError('Unknown replay interface')
        super().__init__(T,N,device,foot_commands=interface=='foot')
        self.interface=interface;self.fields['start_frame']=((),torch.long)
        self.data['start_frame']=torch.zeros(T,N,dtype=torch.long,device=device)

    def add(self,tr):
        if self.interface=='motion' and tr.get('command') is not None:raise ValueError('Foot command leaked into Motion Replay')
        if self.interface=='foot' and (type(tr.get('command')) is not FootstepCommand or 'command_id' not in tr):raise ValueError('Foot Replay requires immutable Quintic commands')
        start=(tr['f']-tr['ep_step'].long()) if self.interface=='motion' else torch.full_like(tr['f'],-1)
        super().add(dict(tr,start_frame=start))

    def sample_windows(self,B,k):
        t0=torch.randint(0,self.size-k+1,(B,),device=self.device)
        env=torch.randint(0,self.N,(B,),device=self.device)
        ages=t0[:,None]+torch.arange(k,device=self.device);rows=self._row(ages);e=env[:,None].expand_as(rows)
        out={key:self.data[key][rows,e] for key in ['s','s1','a','xi','done']}
        ticks=self.data['ep_step'][rows,e];frames=self.data['f'][rows,e];starts=self.data['start_frame'][rows,e]
        valid=(~out['done'][:,:-1])&(ticks[:,1:]==ticks[:,:-1]+1)&(frames[:,1:]==frames[:,:-1]+1)&(starts[:,1:]==starts[:,:-1])
        if self.interface=='foot':
            cid=self.data['command_id'][rows,e];valid&=cid[:,1:]==cid[:,:-1]
        out['mask']=torch.cumprod(torch.cat((torch.ones(B,1,device=self.device),valid.float()),-1),-1)
        return out

    def contract(self):
        return dict(interface=self.interface,state_dim=71,action_executed='a: after original action clamp',
                    goal_dim=730 if self.interface=='motion' else 230,start_frame='f-ep_step' if self.interface=='motion' else 'not applicable (-1)',
                    independent_clock=True,immutable_commands=self.interface=='foot',fields=list(self.fields))
