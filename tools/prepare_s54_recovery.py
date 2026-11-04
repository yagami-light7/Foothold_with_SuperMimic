"""Build explicit rollback weights + fresh critic/optimizers without resetting the budget."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from supermimic.curriculum_runtime import atomic_save
from supermimic.mdp import OBS_DIM
from supermimic.networks import Critic,Policy,WorldModel
from supermimic.s54 import assert_checkpoint


def prepare(progress_path,weights_path,out):
    out=Path(out)
    if out.exists(): raise ValueError('Recovery output already exists; refusing overwrite')
    progress=torch.load(progress_path,map_location='cpu',weights_only=False)
    source=torch.load(weights_path,map_location='cpu',weights_only=False)
    assert_checkpoint(progress);assert_checkpoint(source)
    for key in ('robot_contract','goal_contract','task_contract'):
        if progress[key]!=source[key]: raise ValueError(f'Cannot migrate incompatible {key}')
    if progress['args']['hist']!=source['args']['hist']: raise ValueError('Actor history mismatch')
    if progress['curriculum']['cfg']!=source['curriculum']['cfg']: raise ValueError('Curriculum config mismatch')
    a=progress['args'];torch.manual_seed(a['seed'])
    wm=WorldModel();wm.load_state_dict(source['wm'])
    pi=Policy(OBS_DIM,progress['goal_contract']['dim'],hist=a['hist']);pi.load_state_dict(source['pi'])
    critic=Critic(progress['critic']['norm.mean'].numel())
    progress.update(wm=wm.state_dict(),pi=pi.state_dict(),critic=critic.state_dict(),
        opt_wm=torch.optim.Adam(wm.parameters(),lr=a['lr']).state_dict(),
        opt_pi=torch.optim.Adam(pi.parameters(),lr=a['lr'],betas=(.9,.95)).state_dict(),
        opt_v=torch.optim.Adam(critic.parameters(),lr=a['lr']).state_dict())
    progress['recovery']=dict(progress_source=str(progress_path),weights_source=str(weights_path),
        weights_iteration=source['iteration'],preserved_iteration=progress['iteration'],
        preserved_wall_seconds=progress['wall_time'],reset='critic + all optimizer states; WM/Actor norms from weight source',
        physical_mastery_claimed=False)
    atomic_save(progress,out)
    return progress['recovery']

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--progress',required=True)
    p.add_argument('--weights',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(prepare(a.progress,a.weights,a.out),indent=2))
