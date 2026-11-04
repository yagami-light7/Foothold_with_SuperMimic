"""CPU geometric preflight. Joint-limited FK fitting is necessary, never sufficient.

Tests representative swing/stance poses with the pelvis over the foot midpoint;
failed fits are reported as unverified, not proof of an impossible robot motion.
"""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

from pathlib import Path
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from supermimic.curriculum import Curriculum,Command,feasibility,evaluation_commands
from supermimic.goals import GoalSource,GoalSpec
from supermimic.foot_mdp import FootTrackingMDP,FootTrackingCfg
from supermimic.foot_trajectory import query
from supermimic.kinematics import S54Kinematics
from supermimic.rotations import quat_to_mat
from supermimic.s54 import default_joint_pos_list

torch.set_num_threads(4)
kin=S54Kinematics('cpu');goals=GoalSource(None,kin,GoalSpec('foot_trajectory'),command_mode='curriculum')
mdp=FootTrackingMDP(kin,torch.tensor(default_joint_pos_list()),goals,FootTrackingCfg(command_mode='curriculum',episode_steps=500))
c=Curriculum();goals.curriculum=c;goals.command_duration=12.
S=mdp.nominal_state(1); feet=kin.forward(*S)[0][:,mdp.foot_indices];feet[...,2]=torch.tensor(goals.ankle_height)
quat=torch.tensor([[[1.,0,0,0],[1.,0,0,0]]]);heading=quat[:,0]
rows=[]
for cmd in evaluation_commands(5,True):
    why=feasibility(cmd,c.cfg,width=float((feet[0,0,:2]-feet[0,1,:2]).norm()))
    if why:
        rows.append(dict(command=vars(cmd),status='envelope_rejected',reason=why));continue
    plan=c.plan(feet,quat,heading,fixed=cmd)
    times=torch.linspace(.2,6.,13)[None];ref=query(plan,times)
    target=ref.position[0]; targetR=quat_to_mat(ref.quaternion[0]); B=target.shape[0]
    p=target.mean(1).detach();p[:,2]=mdp.nominal_height
    R=targetR[:,0].detach();v=torch.zeros_like(p)
    q=torch.nn.Parameter(S[4].repeat(B,1)); dz=torch.nn.Parameter(torch.zeros(B))
    opt=torch.optim.Adam([q,dz],lr=.025)
    for _ in range(150):
        base=torch.cat((p[:,:2],(p[:,2]+dz)[:,None]),-1)
        bp,bR,_,_=kin.forward(base,R,v,v,q,torch.zeros_like(q))
        pos=bp[:,mdp.foot_indices];ori=bR[:,mdp.foot_indices]
        loss=((pos-target)**2).sum()+.01*((ori-targetR)**2).sum()+1e-5*((q-S[4])**2).sum()
        opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():q.clamp_(kin.q_lo,kin.q_hi);dz.clamp_(-.08,.02)
    err=(pos-target).norm(dim=-1).detach();orierr=(ori-targetR).square().sum((-1,-2)).sqrt().detach()
    rows.append(dict(command=vars(cmd),status='geometric_fit' if float(err.max())<.03 and float(orierr.max())<.35 else 'geometric_fit_unverified',
                     max_foot_error_m=float(err.max()),max_rotation_matrix_error=float(orierr.max()),
                     note='No collision, torque, balance, or dynamic certification'))
    print(vars(cmd),rows[-1]['status'],rows[-1]['max_foot_error_m'],flush=True)
Path('results/s54/curriculum_reach.json').write_text(json.dumps(rows,indent=2)+'\n')
