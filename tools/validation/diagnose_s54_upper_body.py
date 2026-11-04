"""CPU-only counterfactual audit of upper-body coverage in the saved foot reward.

This is a static FK/reward comparison, not a Newton rollout or a stability test.
No live viewer, checkpoint, training configuration, or policy action is changed.
"""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from supermimic.curriculum import Command,Curriculum
from supermimic.foot_mdp import FootTrackingMDP,resolve_foot_task,assert_task_checkpoint
from supermimic.foot_trajectory import AlternatingConfig
from supermimic.goals import GoalSource,GoalSpec
from supermimic.kinematics import S54Kinematics
from supermimic.s54 import COM_BODY,JOINT_NAMES,default_joint_pos_list,assert_checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert_checkpoint(ck);cfg=ck['args']
    kin=S54Kinematics('cpu')
    task=resolve_foot_task(cfg['foot_task_config'])
    goals=GoalSource(None,kin,GoalSpec(cfg['goal_mode'],cfg['goal_frames'],cfg['goal_dt']),
                     AlternatingConfig(**cfg['foot_config']),task.command_mode)
    goals.spec.assert_checkpoint(ck,goals.foot_config)
    curriculum=Curriculum(ck['curriculum']['cfg']);curriculum.load_state_dict(ck['curriculum'])
    goals.curriculum=curriculum;goals.fixed_command=Command();goals.command_duration=12.
    mdp=FootTrackingMDP(kin,torch.tensor(default_joint_pos_list()),goals,task)
    assert_task_checkpoint(ck,mdp)
    S=mdp.nominal_state(1);command=goals.new_command(S)
    zero=torch.zeros(1,len(JOINT_NAMES));ticks=torch.zeros(1,dtype=torch.long)
    bp0,bR0,_,_=kin.forward(*S)
    torso_index=kin.tracked_names.index(COM_BODY)
    baseline,base_terms=mdp.reward(S,None,zero,zero,terms=True,command=command,ep_step=ticks)
    rows=[]
    for joint in ['waist_yaw_joint','zarm_l1_joint','zarm_r1_joint','zhead_1_joint']:
        index=JOINT_NAMES.index(joint)
        for degrees in [0.,15.,30.,45.,60.]:
            state=tuple(v.clone() for v in S)
            value=state[4][0,index]+degrees*torch.pi/180
            if not kin.q_lo[index]<=value<=kin.q_hi[index]:continue
            state[4][0,index]=value
            bp,bR,_,_=kin.forward(*state)
            reward,terms=mdp.reward(state,None,zero,zero,terms=True,command=command,ep_step=ticks)
            delta_p=(bp[:,mdp.foot_indices]-bp0[:,mdp.foot_indices]).abs().max()
            delta_R=(bR[:,mdp.foot_indices]-bR0[:,mdp.foot_indices]).abs().max()
            assert float(delta_p)<1e-7 and float(delta_R)<1e-7
            assert torch.allclose(terms['foot_objective'],base_terms['foot_objective'])
            assert torch.allclose(terms['base_upright'],base_terms['base_upright'])
            rows.append(dict(joint=joint,offset_degrees=degrees,inside_soft_limits=bool(mdp.soft_lo[index]<=value<=mdp.soft_hi[index]),
                reward=float(reward[0]),reward_drop=float((baseline-reward)[0]),
                reward_drop_percent_of_neutral=float(((baseline-reward)/baseline)[0]*100),
                posture_penalty_per_step=float(task.w_posture*terms['posture'][0]*.02),
                soft_limit_penalty_per_step=float(task.w_joint_limit*terms['joint_limit'][0]*.02),
                foot_position_change_max_m=float(delta_p),foot_rotation_matrix_change_max=float(delta_R),
                upright_score=float(terms['base_upright'][0]),
                root_yaw_degrees=float(torch.atan2(state[1][0,1,0],state[1][0,0,0])*180/torch.pi),
                torso_yaw_degrees=float(torch.atan2(bR[0,torso_index,1,0],bR[0,torso_index,0,0])*180/torch.pi)))
    report=dict(checkpoint=str(Path(a.checkpoint).resolve()),iteration=ck['iteration'],kind='static_cpu_fk_reward_counterfactual',
        neutral_reward_per_step=float(baseline[0]),w_posture=task.w_posture,
        interpretation='Feet and base stay identical. This does not establish dynamic feasibility or episode success.',rows=rows)
    target=Path(a.out);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
