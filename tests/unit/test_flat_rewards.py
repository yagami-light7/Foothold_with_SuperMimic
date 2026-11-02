"""对照站立、正确摆动、滑动和腾空的奖励，防止只看总分造成步态漏洞。"""

import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import copy
import unittest
from typing import Any
import torch
from configs.s54_flat_reward import FlatWalkRewardCfg
from configs.walk_training import apply_task_profile,validate_profile,adaptive_gate
from tasks.foot_placement.rewards import (FootRewardCfg,compute_foot_placement_reward,
                                         reward_config_from_dict,reward_reference_upper)
from tasks.foot_placement.foot_frames import get_sole_velocity,SOLE_LOCAL_POS


class FlatRewardTests(unittest.TestCase):
    def fixture(self,phase=.25) -> dict[str,Any]:
        quat=torch.tensor([[0.,0.,0.,1.]])
        left=torch.tensor([[.15,.095,.03]])
        right=torch.tensor([[0.,-.095,0.]])
        target=left.clone();target[:,2]=0
        zero=torch.zeros(1,3)
        return dict(left_pos_w=left,right_pos_w=right,left_target_w=target,right_target_w=right.clone(),
                    left_force_w=zero.clone(),right_force_w=torch.tensor([[0.,0.,660.]]),
                    phase=torch.tensor([phase]),left_is_swing=torch.tensor([True]),
                    projected_gravity_b=torch.tensor([[0.,0.,-1.]]),actions=torch.zeros(1,27),
                    prev_actions=torch.zeros(1,27),dt=.02,cfg=FlatWalkRewardCfg(),
                    root_quat_w=quat,reference_heading_w=torch.zeros(1),root_yaw_rate=torch.zeros(1),
                    flat_state=dict(left_quat=quat,right_quat=quat,left_target_quat=quat,right_target_quat=quat,
                                    left_velocity=zero.clone(),right_velocity=zero.clone(),
                                    root_height=torch.tensor([.98]),nominal_height=torch.tensor([.98]),
                                    root_vz=torch.zeros(1),upper_joint_offsets=torch.zeros(1,15)))

    def test_phase_lifting_outscores_standing_and_wrong_foot(self):
        proper=self.fixture();normal,terms=compute_foot_placement_reward(**proper)
        stand=copy.deepcopy(proper);stand['left_pos_w'][:,[0,2]]=0;stand['left_force_w'][:,2]=330
        standing,standing_terms=compute_foot_placement_reward(**stand)
        self.assertGreater(normal.item(),standing.item())
        self.assertEqual(terms['swing_phase'].item(),4.)
        self.assertEqual(standing_terms['swing_phase'].item(),0.)
        wrong=copy.deepcopy(proper);wrong['right_pos_w'][:,2]=.03
        wrong['right_force_w'].zero_();wrong['left_force_w'][:,2]=660
        reward,wrong_terms=compute_foot_placement_reward(**wrong)
        self.assertLess(reward.item(),normal.item())
        self.assertEqual(wrong_terms['swing_phase'].item(),0.)
        for phase in (.05,.45):
            _,outside=compute_foot_placement_reward(**self.fixture(phase))
            self.assertEqual(outside['swing_phase'].item(),0.)

    def test_hopping_cannot_keep_tracking_or_phase_rewards(self):
        data=self.fixture();baseline,_=compute_foot_placement_reward(**data)
        data['right_pos_w'][:,2]=.08;data['right_force_w'].zero_()
        data['flat_state']['right_velocity'][:,2]=.5
        reward,terms=compute_foot_placement_reward(**data)
        self.assertEqual(terms['tracking'].item(),0.)
        self.assertEqual(terms['swing_phase'].item(),0.)
        self.assertLess(terms['stance_hop'].item(),-20.)
        self.assertLess(reward.item(),baseline.item())

    def test_normal_foot_roll_has_height_tolerance(self):
        data=self.fixture();_,reference=compute_foot_placement_reward(**data)
        data['right_pos_w'][:,2]=.02  # 接地时脚掌滚动，参考点升高而不是整脚腾空。
        _,rolled=compute_foot_placement_reward(**data)
        torch.testing.assert_close(rolled['tracking'],reference['tracking'])
        self.assertEqual(rolled['stance_hop'].item(),0.)

    def test_contact_slip_and_stance_vertical_motion_are_penalized(self):
        data=self.fixture();data['flat_state']['right_velocity'][:,0]=.2
        _,terms=compute_foot_placement_reward(**data)
        self.assertAlmostEqual(terms['stance_slip'].item(),-4*.2**2,places=6)
        data['flat_state']['right_velocity'].zero_();data['flat_state']['left_velocity'][:,0]=.2
        _,swing=compute_foot_placement_reward(**data)
        self.assertEqual(swing['stance_slip'].item(),0.)
        data['flat_state']['right_velocity'][:,2]=.3
        _,vertical=compute_foot_placement_reward(**data)
        self.assertLess(vertical['stance_hop'].item(),0.)

    def test_body_posture_and_action_regularization(self):
        data=self.fixture();_,reference=compute_foot_placement_reward(**data)
        data['flat_state']['root_height']+=.1;data['flat_state']['root_vz']+=.5
        data['flat_state']['upper_joint_offsets']+=.1;data['actions']+=.2
        _,terms=compute_foot_placement_reward(**data)
        self.assertAlmostEqual(terms['body_motion'].item(),-.225,places=6)
        self.assertLess(terms['posture'].item(),reference['posture'].item())
        self.assertAlmostEqual(terms['action_rate'].item(),-.05*.2**2,places=6)

    def test_left_right_mirror_and_reference_upper_bound(self):
        left=self.fixture();r1,t1=compute_foot_placement_reward(**left)
        right=copy.deepcopy(left)
        for a,b in [('left_pos_w','right_pos_w'),('left_target_w','right_target_w'),('left_force_w','right_force_w')]:
            right[a],right[b]=right[b],right[a]
        right['phase']=torch.tensor([.75]);right['left_is_swing']=torch.tensor([False])
        r2,t2=compute_foot_placement_reward(**right)
        torch.testing.assert_close(r1,r2)
        for name in t1:
            torch.testing.assert_close(t1[name],t2[name])
        for phase in torch.linspace(0,.999,200):
            data=self.fixture(float(phase));data['left_is_swing']=data['phase']<.5
            r,_=compute_foot_placement_reward(**data)
            upper=reward_reference_upper(data['phase'],data['left_is_swing'],data['dt'],data['cfg'])
            self.assertTrue((r<=upper+1e-6).all())

    def test_sole_velocity_contains_rotational_offset(self):
        quat=torch.tensor([[0.,0.,0.,1.]])
        velocity=get_sole_velocity(quat,torch.zeros(1,3),torch.tensor([[0.,0.,1.]]))
        torch.testing.assert_close(velocity,torch.tensor([[0.,SOLE_LOCAL_POS[0],0.]]))

    def test_old_and_new_configuration_are_explicit(self):
        from tasks.foot_placement.foot_placement_env_cfg import S54FootPlacementEnvCfg
        self.assertIsInstance(reward_config_from_dict({}),FootRewardCfg)
        cfg=S54FootPlacementEnvCfg()
        apply_task_profile(cfg,{'reward_cfg':{'profile':'flat_walk_v2'}})
        self.assertIsInstance(cfg.reward_cfg,FlatWalkRewardCfg)
        for override in ({'hop_height_scale':0.},{'swing_window_half_width':.3},{'stance_slip_w':-1.}):
            with self.assertRaises(ValueError):
                validate_profile({'reward_cfg':{'profile':'flat_walk_v2',**override}})

    def test_new_gate_rejects_hopping_and_sliding_even_with_good_tracking(self):
        result=dict(survival_fraction=1.,half_cycle_success_left_right=[.9,.9],
                    consecutive_8_success_fraction=.9,late_xy_error_m=.01,mean_upright=.99,
                    max_abs_heading_error_rad=.05,mean_reference_forward_velocity=.5,
                    command_nominal_forward_speed_m_s=.6,reward_profile='flat_walk_v2',
                    both_air_fraction=0.,stance_hop_fraction=0.,contact_foot_slip_speed_m_s=.03)
        self.assertTrue(adaptive_gate(result)[0])
        for key,bad in [('both_air_fraction',.03),('stance_hop_fraction',.03),('contact_foot_slip_speed_m_s',.11)]:
            self.assertFalse(adaptive_gate({**result,key:bad})[0])


if __name__=='__main__':
    unittest.main()
