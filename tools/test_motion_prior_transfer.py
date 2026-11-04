"""CPU acceptance: data isolation, exact migration, frozen-path BPTT, priors and contracts."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from supermimic.kinematics import S54Kinematics
from supermimic.motion import MotionLib
from supermimic.mdp import MDP,RewardCfg,OBS_DIM,normalize_xi,flatten_state
from supermimic.foot_mdp import FootTrackingMDP,FootTrackingCfg,assert_task_checkpoint
from supermimic.goals import GoalSource,GoalSpec
from supermimic.reference_foot import ReferenceFootGoalSource,PairedMotionMDP,ReferenceFootCommand
from supermimic.foot_trajectory import AlternatingConfig,command_time
from supermimic.motion_prior_data import load_manifest,ContactConfig,estimate_contacts
from supermimic.networks import Policy,WorldModel,DualEncoderPolicy
from supermimic.transfer import initialize_from_motion,initialize_from_bridge,physical_contract,policy_contract,assert_policy_checkpoint
from supermimic.s54 import default_joint_pos_list,checkpoint_contract,XI_DIM,JOINT_NAMES
from supermimic.buffer import ReplayBuffer
from supermimic.upper_prior import PriorConfig,PriorFootMDP,command_features
from supermimic.curriculum import Curriculum
from supermimic.rotations import so3_exp,mat_to_quat

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'results/s54/motion_prior_stage0_verified/manifest.json'


class MotionPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);torch.manual_seed(5404)
        cls.kin=S54Kinematics('cpu');cls.default=torch.tensor(default_joint_pos_list())
        cls.files,cls.data_contract=load_manifest(MANIFEST)
        cls.motion=MotionLib(cls.files,cls.kin,'cpu')
        cls.motion_mdp=MDP(cls.kin,cls.motion,cls.default)
        cls.goals=GoalSource(None,cls.kin,GoalSpec('foot_trajectory'),command_mode='standing')
        cls.mdp=FootTrackingMDP(cls.kin,cls.default,cls.goals,FootTrackingCfg(command_mode='standing'))
        cls.paired=ReferenceFootGoalSource(cls.motion,cls.kin,GoalSpec('reference_foot'),AlternatingConfig())
        cls.bridge=PairedMotionMDP(cls.kin,cls.motion,cls.default,cls.paired,RewardCfg())
        cls.source_pi=Policy(99,730,hist=10);cls.source_wm=WorldModel()
        cls.source_wm.in_norm.mean.fill_(.2);cls.source_wm.out_norm.mean.fill_(.001)
        cls.ck=dict(robot_contract=checkpoint_contract(),physical_contract=physical_contract(),
            goal_contract=GoalSpec('motion').contract(),task_contract=cls.motion_mdp.contract(),
            policy_contract=policy_contract(10),args=dict(hist=10,goal_mode='motion'),
            wm=cls.source_wm.state_dict(),pi=cls.source_pi.state_dict(),iteration=5,sim_steps=123,data_contract=cls.data_contract)

    def test_01_data_contract_and_leakage(self):
        train,contract=load_manifest(MANIFEST);heldout,_=load_manifest(MANIFEST,'heldout')
        self.assertEqual(len(train),1);self.assertEqual(len(heldout),1);self.assertFalse(set(train)&set(heldout))
        data=json.loads(MANIFEST.read_text());data['clips'][1]['source_identity']=data['clips'][0]['source_identity']
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'manifest.json';path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'leaks'):load_manifest(path)
            data['clips'][1]['source_identity']='distinct';data['clips'][0]['csv_sha256']='changed'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'changed'):load_manifest(path)

    def test_02_contact_debounce_and_hysteresis(self):
        cfg=ContactConfig(min_duration=.06)
        h=np.zeros((10,2));speed=np.zeros_like(h)
        h[5:7]=.03;h[7:]=.1
        c=estimate_contacts(h,speed,cfg)
        self.assertFalse(c[:2].any());self.assertTrue(c[2:9].all());self.assertFalse(c[9].any())

    def test_03_dims_and_stage_contracts(self):
        self.assertEqual(GoalSpec().dim,730);self.assertEqual(GoalSpec('reference_foot').dim,230)
        self.assertEqual(self.motion_mdp.critic_dim+730,914);self.assertEqual(self.mdp.critic_dim+230,365)
        self.assertEqual(self.bridge.critic_dim+230,365)
        self.assertEqual(self.source_wm.in_dim,133);self.assertEqual(self.source_wm.net[-1].out_features,70)
        self.assertEqual(self.source_pi(torch.zeros(2,10,99),torch.zeros(2,730)).shape,(2,29))

    def test_04_exact_selective_migration_and_fresh_norms(self):
        pi=Policy(99,230,hist=10,obs_adapter='mask_reference_orientation');wm=WorldModel()
        original=pi.enc[0].weight.clone();report=initialize_from_motion(self.ck,wm,pi,hist=10)
        torch.testing.assert_close(pi.enc[0].weight,original,rtol=0,atol=0)
        for k,v in pi.dec.state_dict().items():torch.testing.assert_close(v,self.source_pi.dec.state_dict()[k],rtol=0,atol=0)
        for k,v in wm.state_dict().items():torch.testing.assert_close(v,self.ck['wm'][k],rtol=0,atol=0)
        self.assertEqual(float(pi.obs_norm.count),0);self.assertEqual(float(pi.goal_norm.count),0)
        self.assertFalse(report['teacher_required_by_stage2']);self.assertIn('critic',report['new'])

    def test_05_fail_fast_robot_hist_shape_physics_task(self):
        pi=Policy(99,230,hist=10,obs_adapter='mask_reference_orientation');wm=WorldModel()
        snapshot=wm.net[0].weight.clone()
        cases=[]
        bad=dict(self.ck,robot_contract={'robot':'g1'});cases.append(bad)
        bad=dict(self.ck,args=dict(hist=2));cases.append(bad)
        bad=dict(self.ck,physical_contract={'dt':.04});cases.append(bad)
        bad=dict(self.ck,task_contract=self.mdp.contract());cases.append(bad)
        bad=dict(self.ck,task_contract=dict(self.ck['task_contract'],obs_dim=98));cases.append(bad)
        bad=dict(self.ck,policy_contract=policy_contract(2));cases.append(bad)
        bad=dict(self.ck,wm=dict(self.ck['wm']));bad['wm']['net.0.weight']=torch.zeros(1,1);cases.append(bad)
        for ck in cases:
            with self.assertRaises(ValueError):initialize_from_motion(ck,wm,pi,hist=10)
            torch.testing.assert_close(wm.net[0].weight,snapshot,rtol=0,atol=0)

    def test_06_wm_only_allows_hist_change_without_decoder(self):
        pi=Policy(99,230,hist=2);wm=WorldModel();before=pi.dec[0].weight.clone()
        report=initialize_from_motion(self.ck,wm,pi,hist=2,components='wm')
        torch.testing.assert_close(pi.dec[0].weight,before);self.assertEqual(report['loaded'],['wm'])

    def test_07_heading_adapter_masks_only_reference_channels(self):
        pi=Policy(99,230,hist=10,obs_adapter='mask_reference_orientation')
        obs=torch.randn(2,10,99);other=obs.clone();other[...,93:]+=3
        g=torch.randn(2,230)
        torch.testing.assert_close(pi(obs,g),pi(other,g),rtol=0,atol=0)
        other=obs.clone();other[...,61:64]+=.2
        self.assertGreater(float((pi(obs,g)-pi(other,g)).abs().max()),0)
        # Historical Motion has its original heading semantics.
        self.assertEqual(self.source_pi.obs_adapter,'identity')

    def test_08_same_task_resume_vs_cross_task_rejection(self):
        assert_task_checkpoint({'task_contract':self.mdp.contract()},self.mdp)
        assert_task_checkpoint({'task_contract':self.motion_mdp.contract()},self.motion_mdp)
        with self.assertRaises(ValueError):assert_task_checkpoint(self.ck,self.mdp)
        with self.assertRaises(ValueError):GoalSpec('foot_trajectory').assert_checkpoint(self.ck)
        assert_policy_checkpoint({'policy_contract':policy_contract(10,'mask_reference_orientation',False)},10,'mask_reference_orientation',False)
        with self.assertRaises(ValueError):assert_policy_checkpoint({},10,'mask_reference_orientation',False)

    def test_09_paired_time_fields_and_clip_end(self):
        f=torch.tensor([100,400]);S=self.motion.ref_state(f);c=self.paired.new_command(S,frames=f)
        goal=self.paired.goal(S,f,torch.zeros(2,dtype=torch.long),c).reshape(2,10,23)
        ref=self.paired.query(c,torch.tensor([[.1,.2],[.1,.2]],dtype=torch.float64))
        torch.testing.assert_close(ref.position[:,0],self.paired.position[f+5])
        self.assertEqual(goal.shape,(2,10,23));torch.testing.assert_close(goal[:,:,22],torch.arange(1,11).float()[None].expand(2,-1)*.1)
        self.assertEqual(self.motion.goal(f,10).shape,(2,730))
        end=self.motion.clip_end[:1];endc=self.paired.new_command(self.motion.ref_state(end),frames=end)
        held=self.paired.query(endc,torch.tensor([[0.,10.]],dtype=torch.float64))
        torch.testing.assert_close(held.position[:,0],held.position[:,1])

    def test_10_paired_reward_and_privileged_critic(self):
        f=torch.tensor([100,101]);S=self.motion.ref_state(f);c=self.paired.new_command(S,frames=f)
        a=torch.zeros(2,29);step=torch.zeros(2,dtype=torch.long);xi=torch.zeros(2,33)
        torch.testing.assert_close(self.bridge.reward(S,f,a,a),self.motion_mdp.reward(S,f,a,a))
        torch.testing.assert_close(self.bridge.termination(S,f),self.motion_mdp.termination(S,f))
        self.assertEqual(self.bridge.critic_features(S,f,xi,command=c,ep_step=step).shape,(2,135))
        # Correct observers differ in heading, rather than silently sharing the wrong reference.
        S1=list(S);S1[1]=so3_exp(torch.tensor([[.1,.03,0],[.1,.03,0]]))@S1[1]
        motion_obs=self.motion_mdp.obs(tuple(S1),a,xi,f)
        foot_obs=self.paired.obs(self.bridge,tuple(S1),a,xi,f,c)
        torch.testing.assert_close(motion_obs[:,:93],foot_obs[:,:93])
        self.assertGreater(float((motion_obs[:,93:]-foot_obs[:,93:]).abs().max()),1e-5)

    def test_11_reference_command_replay_ring_and_history(self):
        buffer=ReplayBuffer(4,1,'cpu',foot_commands=True);a=torch.zeros(1,29);f=torch.tensor([100]);S=self.motion.ref_state(f)
        for tick in range(7):
            start=f+10*(tick//3);c=self.paired.new_command(self.motion.ref_state(start),frames=start)
            local=torch.tensor([tick%3]);state=self.motion.ref_state(start+local)
            tr=dict(s=flatten_state(state),s1=flatten_state(state),a=a,a_prev=a,xi=torch.zeros(1,33),f=start+local,
                    ep_step=local,r=torch.zeros(1),done=local==2,term=local==2,timeout=local==2,
                    command=c,command_id=torch.tensor([tick//3]))
            buffer.add(tr)
            restored=buffer.commands(tr['command_id']);self.assertIsInstance(restored,ReferenceFootCommand)
            torch.testing.assert_close(self.paired.goal(state,start+local,local,restored),self.paired.goal(state,start+local,local,c))
        self.assertNotIn(0,buffer.command_store)
        sample,hist=buffer.sample_starts(8,10)
        cmds=buffer.commands(sample['command_id']);self.assertEqual(cmds.start_frame.shape,(8,))

    def test_12_frozen_decoder_wm_bptt_updates_only_encoder(self):
        pi=Policy(99,230,hist=10,obs_adapter='mask_reference_orientation')
        wm=WorldModel(hidden=(64,64))
        # Synthetic action-sensitive dynamics verifies the chain; not a trained-model accuracy claim.
        torch.nn.init.normal_(wm.net[-1].weight,std=.002)
        for p in wm.parameters():p.requires_grad_(False)
        for p in pi.dec.parameters():p.requires_grad_(False)
        S=self.mdp.nominal_state(2);S=list(S);S[0]=S[0]+torch.tensor([.03,0,0]);S=tuple(S)
        c=self.goals.new_command(S);xi=torch.zeros(2,33);xi[:,0]=1;prev=torch.zeros(2,29)
        obs=self.goals.obs(self.mdp,S,prev,xi,None,c);hist=obs[:,None].repeat(1,10,1);objective=0
        for tick in range(3):
            step=torch.full((2,),tick);a=pi(hist,self.goals.goal(S,None,step,c));S=wm.step(S,a,normalize_xi(xi))
            _,heads=self.mdp.reward(S,None,a,prev,True,command=c,ep_step=step+1)
            objective=objective+heads['foot_objective'].mean();prev=a
        (-objective).backward()
        norm=torch.stack([p.grad.square().sum() for p in pi.enc.parameters()]).sum().sqrt()
        self.assertTrue(torch.isfinite(norm));self.assertGreater(float(norm),0)
        self.assertTrue(all(p.grad is None for p in pi.dec.parameters()));self.assertTrue(all(p.grad is None for p in wm.parameters()))

    def test_13_dual_encoder_keeps_shared_decoder_and_original_motion(self):
        policy=DualEncoderPolicy(hist=10)
        policy.dec.load_state_dict(self.source_pi.dec.state_dict());policy.motion_enc.load_state_dict(self.source_pi.enc.state_dict())
        o=torch.randn(2,10,99);g=torch.randn(2,730)
        torch.testing.assert_close(policy.forward_motion(o,g),self.source_pi(o,g))
        self.assertEqual(policy(o,torch.zeros(2,230)).shape,(2,29))

    def test_14_real_eval_and_world_model_both_gate_curriculum(self):
        cur=Curriculum(dict(min_stage_seconds=0,consecutive_passes=1,wm_gate={'base_height_rmse_m':.01}))
        good={'standing':dict(success_rate=1.,foot_rmse=.001)}
        self.assertFalse(cur.update(good,100,wm_metrics=None));self.assertEqual(cur.stage,0)
        self.assertFalse(cur.update(good,100,wm_metrics={'base_height_rmse_m':.03}));self.assertEqual(cur.stage,0)
        self.assertTrue(cur.update(good,100,wm_metrics={'base_height_rmse_m':.005}));self.assertEqual(cur.stage,1)

    def test_20_posture_gate_blocks_high_reward_but_tilted_standing(self):
        cur=Curriculum(dict(min_stage_seconds=0,consecutive_passes=1,
                            posture_gate={'root_pitch_max_abs_deg':10.}))
        metrics={'standing':dict(success_rate=1.,foot_rmse=.001,root_pitch_max_abs_deg=25.)}
        self.assertFalse(cur.update(metrics,100));self.assertEqual(cur.stage,0)
        metrics['standing']['root_pitch_max_abs_deg']=2.
        self.assertTrue(cur.update(metrics,100));self.assertEqual(cur.stage,1)

    def _prior(self,folder):
        S=self.mdp.nominal_state(2)
        walking=GoalSource(None,self.kin,GoalSpec('foot_trajectory'),command_mode='alternating').new_command(S)
        speed=command_features(walking,torch.tensor([20,20]))[0].mean()
        bank=Path(folder)/'bank.npz';P=16
        quat=mat_to_quat(so3_exp(torch.tensor([[.07,0.,0.]]))).numpy()
        np.savez(bank,upper_joint_names=np.array([n for n in JOINT_NAMES if not n.startswith('leg_')]),
                 upper_q=np.full((1,P,17),.1,dtype=np.float32),upper_qd=np.full((1,P,17),.2,dtype=np.float32),
                 torso_quat=np.repeat(quat[:,None],P,1),speed=np.array([speed],dtype=np.float32),modes=np.array([1]),
                 contract=json.dumps({'version':1,'data':'synthetic gradient fixture'}))
        cfg=PriorConfig(bank=str(bank),w_upper_pose=.03,w_torso=.02,w_upper_velocity=.01,speed_tolerance=2.)
        return PriorFootMDP(self.kin,self.default,self.goals,self.mdp.cfg,cfg),S,walking

    def test_15_standing_missing_template_and_off_match_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            mdp,S,walking=self._prior(tmp);command=self.goals.new_command(S);a=torch.zeros(2,29);tick=torch.tensor([20,20])
            baseline=self.mdp.reward(S,None,a,a,command=command,ep_step=tick)
            r,terms=mdp.reward(S,None,a,a,True,command=command,ep_step=tick)
            torch.testing.assert_close(r,baseline,rtol=0,atol=0);self.assertFalse(terms['prior_matched'].any())
            self.assertEqual(mdp.max_step_reward,self.mdp.max_step_reward)
            r,terms=mdp.reward(S,None,a,a,True,command=walking,ep_step=tick)
            self.assertTrue(terms['prior_matched'].all());self.assertGreater(float(terms['prior_upper_pose_contribution'].mean()),0)
            walking.target_pos[:,:,1]+=10
            _,terms=mdp.reward(S,None,a,a,True,command=walking,ep_step=tick)
            self.assertFalse(terms['prior_matched'].any())
        off=PriorFootMDP(self.kin,self.default,self.goals,self.mdp.cfg,PriorConfig())
        self.assertEqual(off.contract(),self.mdp.contract())

    def test_16_prior_finite_difference_and_state_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            mdp,S,c=self._prior(tmp);S=list(S);S[4]=S[4].clone().requires_grad_()
            tick=torch.tensor([20,20]);a=torch.zeros(2,29)
            _,heads=mdp.reward(tuple(S),None,a,a,True,command=c,ep_step=tick)
            value=heads['prior_upper_pose_contribution'].sum();grad=torch.autograd.grad(value,S[4])[0]
            j=JOINT_NAMES.index('zarm_l1_joint');eps=1e-3;values=[]
            for sign in (-1,1):
                copyS=list(S);q=S[4].detach().clone();q[:,j]+=sign*eps;copyS[4]=q
                _,t=mdp.reward(tuple(copyS),None,a,a,True,command=c,ep_step=tick);values.append(t['prior_upper_pose_contribution'].sum())
            self.assertAlmostEqual(float(grad[:,j].sum()),float((values[1]-values[0])/(2*eps)),delta=2e-6)
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.norm()),0)

    def test_17_all_prior_heads_reach_encoder_through_frozen_wm(self):
        with tempfile.TemporaryDirectory() as tmp:
            mdp,S,command=self._prior(tmp)
            pi=Policy(99,230,hist=10,obs_adapter='mask_reference_orientation');wm=WorldModel(hidden=(64,64))
            torch.nn.init.normal_(wm.net[-1].weight,std=.002)
            for p in list(wm.parameters())+list(pi.dec.parameters()):p.requires_grad_(False)
            step=torch.tensor([20,20]);a=pi(torch.zeros(2,10,99),self.goals.goal(S,None,step,command))
            predicted=wm.step(S,a,torch.zeros(2,33))
            _,heads=mdp.reward(predicted,None,a,torch.zeros_like(a),True,command=command,ep_step=step+1)
            params=tuple(pi.enc.parameters())
            for key in ('prior_upper_pose_contribution','prior_torso_contribution','prior_upper_velocity_contribution'):
                grads=torch.autograd.grad(heads[key].sum(),params,retain_graph=True)
                norm=torch.stack([g.square().sum() for g in grads]).sum().sqrt()
                self.assertTrue(torch.isfinite(norm));self.assertGreater(float(norm),0)
            self.assertTrue(all(p.grad is None for p in wm.parameters()))

    def test_18_bridge_motion_reward_bptt_is_not_action_mse(self):
        pi=DualEncoderPolicy(hist=10);wm=WorldModel(hidden=(64,64));torch.nn.init.normal_(wm.net[-1].weight,std=.002)
        for p in list(wm.parameters())+list(pi.dec.parameters())+list(pi.motion_enc.parameters()):p.requires_grad_(False)
        f=torch.tensor([100,120]);S=self.motion.ref_state(f);command=self.paired.new_command(S,frames=f)
        a=pi(torch.zeros(2,10,99),self.paired.goal(S,f,torch.zeros(2,dtype=torch.long),command))
        prediction=wm.step(S,a,torch.zeros(2,33));prev=a.detach()
        # Only state tracking. Disallow direct action-rate gradients in this check.
        reward,heads=self.bridge.reward(prediction,f+1,a,prev,True)
        grads=torch.autograd.grad(heads['track'].sum(),tuple(pi.enc.parameters()))
        norm=torch.stack([g.square().sum() for g in grads]).sum().sqrt()
        self.assertGreater(float(norm),0);self.assertTrue(torch.isfinite(norm))
        self.assertTrue(all(p.grad is None for p in pi.dec.parameters()))

    def test_19_dual_and_single_foot_encoder_have_same_seed_initialization(self):
        torch.manual_seed(54);single=Policy(99,230,hist=10,obs_adapter='mask_reference_orientation')
        torch.manual_seed(54);dual=DualEncoderPolicy(hist=10)
        for k,v in single.enc.state_dict().items():torch.testing.assert_close(v,dual.enc.state_dict()[k],rtol=0,atol=0)

    def test_21_bridge_transfer_report_excludes_discarded_teacher_weights(self):
        source=DualEncoderPolicy(hist=10);wm=WorldModel()
        initialize_from_motion(self.ck,wm,source,hist=10)
        ck=dict(self.ck,wm=wm.state_dict(),pi=source.state_dict(),
            goal_contract=GoalSpec('reference_foot').contract(),task_contract=self.bridge.contract(),
            policy_contract=policy_contract(10,'mask_reference_orientation',False))
        target=Policy(99,230,hist=10,obs_adapter='mask_reference_orientation')
        report=initialize_from_bridge(ck,wm,target,hist=10)
        for key,value in source.enc.state_dict().items():torch.testing.assert_close(value,target.enc.state_dict()[key])
        self.assertEqual(float(target.goal_norm.count),0)
        self.assertTrue(all('motion_' not in row['target'] for row in report['mapping']))
        self.assertTrue(all('reason' in row for row in report['mapping']))


if __name__=='__main__':unittest.main(verbosity=2)
