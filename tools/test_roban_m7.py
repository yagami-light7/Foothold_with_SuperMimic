"""Independent M7 contract, frame/clock/anchor and fail-closed event tests."""
import os
os.environ['SUPERMIMIC_ROBOT']='roban'
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from supermimic.integration.pmt_dataset_loader import load_catalog,load_sample,PMTMotionReferenceProvider,sha256,rotation
from supermimic.integration.pillar_event_controller import PillarEventController,PillarEventBlocked
from supermimic.integration.pillar_scene import PMTTerrainProvider
from supermimic.integration.contact_diagnostics import RobotGeometry,ContactSnapshot,contacted_pillars
from supermimic.kinematics import RobotKinematics
from supermimic.roban import foot_geometry,JOINT_NAMES,TRACKED_BODIES
from supermimic.roban_task import RobanFootGoals,RobanFootMDP,FootRewardConfig
from supermimic.goals import GoalSpec
from supermimic.rotations import yaw_mat

ROOT=Path(os.environ.get('PMT_DATASET_ROOT','/mnt/sage_data/workspace/Foothold/PMT/TCRS/outputs/roban_pillar_dataset'))
torch.set_num_threads(2)


class M7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples=load_catalog(ROOT,seed=44)
        cls.dynamic=next(s for s in cls.samples if s.dynamic)
        cls.static=next(s for s in cls.samples if not s.dynamic)
        cls.kin=RobotKinematics()
        cls.motion=PMTMotionReferenceProvider(cls.samples,cls.kin)

    def edited(self,filename,mutate):
        """Copy source files (never hard-link or mutate the original sample)."""
        context=tempfile.TemporaryDirectory(prefix='m7_contract_')
        self.addCleanup(context.cleanup)
        path=Path(context.name)/'sample'
        shutil.copytree(self.static.directory,path)
        with np.load(path/filename,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
        mutate(values)
        np.savez_compressed(path/filename,**values)
        m=json.loads((path/'manifest.json').read_text())
        m['files'][filename]=sha256(path/filename)
        (path/'manifest.json').write_text(json.dumps(m))
        return path

    def test_01_catalog_discovery_hash_and_unique_names(self):
        self.assertEqual(len(self.samples),2)
        self.assertEqual(len(set(self.motion.names)),2)
        self.assertEqual(self.motion.num_frames,576)
        self.assertTrue(all(s.alignment['verified_files']>=27 for s in self.samples))

    def test_02_joint_and_body_order_rejected(self):
        for key in ('joint_names','body_names'):
            with self.subTest(key=key):
                path=self.edited('body_reference.npz',lambda z:z.__setitem__(key,z[key][::-1]))
                with self.assertRaisesRegex(ValueError,'names/order'):load_sample(path)

    def test_03_quaternion_convention_rejected(self):
        path=self.edited('body_reference.npz',lambda z:z.__setitem__('quaternion_order',np.array('xyzw')))
        with self.assertRaisesRegex(ValueError,'quaternion'):load_sample(path)

    def test_04_ankle_to_sole_and_independent_fk(self):
        s=self.dynamic
        expected=s.foot['foot_pos_w']+np.einsum('tfij,fj->tfi',rotation(s.foot['foot_quat_w']),np.array([g['sole'] for g in foot_geometry()]))
        np.testing.assert_allclose(s.planned_sole,expected,atol=1e-12)
        self.assertLess(s.alignment['planned_ankle_fk_max_error_m'],.005)
        self.assertLess(s.alignment['body_fk_max_m'],1e-7)
        # Detect a root/foot frame disagreement even with a correctly rehashed file.
        path=self.edited('foot_trajectory.npz',lambda z:z['foot_pos_w'].__iadd__([1.,0,0]))
        with self.assertRaisesRegex(ValueError,'Planned/FK'):load_sample(path)

    def test_05_time_alignment_and_fps_rejected(self):
        path=self.edited('foot_trajectory.npz',lambda z:z['time_s'].__iadd__(.02))
        with self.assertRaisesRegex(ValueError,'timeline'):load_sample(path)
        path=self.edited('body_reference.npz',lambda z:z.__setitem__('fps',np.array(100.)))
        with self.assertRaisesRegex(ValueError,'FPS'):load_sample(path)

    def test_06_planned_contact_never_measured_truth(self):
        self.assertFalse(self.motion.contact_valid.any())
        self.assertGreater(int(self.motion.contact.sum()),0)
        path=self.edited('contact_schedule.npz',lambda z:z['contact_valid'].__setitem__((0,0),True))
        with self.assertRaisesRegex(ValueError,'measured truth'):load_sample(path)

    def test_07_phase_semantics_explicit(self):
        for s in self.samples:
            np.testing.assert_array_equal(s.foot['phase'],s.contact['phase'])
            np.testing.assert_array_equal(s.body['swing_phase'],np.where(s.body['contact'],0.,s.foot['phase']))
        self.assertGreater(int((self.motion.phase[self.motion.contact]>0).sum()),0)

    def test_08_goal_reward_share_planned_source_and_23d(self):
        m=self.motion;f=torch.tensor([50,350]);S=m.ref_state(f);anchor=m.anchor(f)
        goals=RobanFootGoals(m,self.kin,GoalSpec('roban_foot',3,.1))
        g=goals.goal(S,f,torch.zeros_like(f),anchor).reshape(2,3,23)
        ref=m.sample(m.indices(f,3,.1),anchor)
        Ht=yaw_mat(S[1]).transpose(-1,-2)[:,None,None]
        expected=(Ht@(ref['foot_p']-S[0][:,None,None])[...,None]).squeeze(-1)
        torch.testing.assert_close(g[...,:3],expected[:,:,0])
        torch.testing.assert_close(g[...,11:14],expected[:,:,1])
        task=RobanFootMDP(self.kin,m,torch.zeros(21),goals)
        current=task._reference(f,anchor)
        torch.testing.assert_close(current['foot_p'][:,0],m.target_foot_p[f])
        self.assertGreater(float((m.target_foot_p-m.foot_p).abs().max()),1e-5)

    def test_09_clip_boundary_no_cross_scene(self):
        m=self.motion
        frames=m.clip_end-1
        indices=m.indices(frames,10,.1)
        torch.testing.assert_close(indices,m.clip_end[:,None].expand(-1,10))
        self.assertTrue((m.clip_id[indices]==torch.arange(2)[:,None]).all())
        goals=RobanFootGoals(m,self.kin,GoalSpec('roban_foot',10,.1))
        goal=goals.goal(m.ref_state(frames),frames,frames*0,m.anchor(frames)).reshape(2,10,23)
        torch.testing.assert_close(goal[:,:,-1],torch.full((2,10),.02))

    def test_10_episode_anchor_includes_pillars(self):
        m=self.motion;f=torch.tensor([0]);theta=.7
        A=torch.tensor([[[np.cos(theta),-np.sin(theta),0.],[np.sin(theta),np.cos(theta),0.],[0.,0.,1.]]],dtype=torch.float32)
        t=torch.tensor([[1.,-2.,.3]])
        anchored=m.anchor(f,A,t);identity=m.anchor(f)
        a,b=m.sample(f[:,None],anchored),m.sample(f[:,None],identity)
        torch.testing.assert_close(a['root_p'],(A[:,None]@b['root_p'][...,None]).squeeze(-1)+t[:,None])
        torch.testing.assert_close(a['foot_p'],(A[:,None,None]@b['foot_p'][...,None]).squeeze(-1)+t[:,None,None])
        p,_,_=self.samples[0].pillar_layout(0,A[0].numpy(),t[0].numpy())
        np.testing.assert_allclose(p[:,:3],self.samples[0].schedule['pose_xyzyaw'][0,:,:3]@A[0].numpy().T+t[0].numpy(),atol=1e-7)

    def test_11_model_and_mesh_tamper_rejected(self):
        path=self.edited('body_reference.npz',lambda z:z.__setitem__('model_sha256',np.array('0'*64)))
        with self.assertRaisesRegex(ValueError,'model/frame'):load_sample(path)
        path=self.edited('terrain_mesh.npz',lambda z:z['vertices_local'].__iadd__(.1))
        with self.assertRaisesRegex(ValueError,'mesh hash'):load_sample(path)

    def test_12_dynamic_events_share_clock(self):
        s=self.dynamic
        for e in s.events:
            lo,hi=[round(e[k]*50) for k in ('timestamp','end_time_s')]
            self.assertFalse(s.schedule['active'][lo:hi,e['pillar_id']].any())
            self.assertTrue(s.schedule['active'][hi,e['pillar_id']])
            self.assertLess(hi,e['target_liftoff_frame'])
        provider=PMTTerrainProvider(self.kin,s)
        with self.assertRaisesRegex(ValueError,'explicit reference frame'):provider.surface(torch.zeros(1,2))
        p=s.schedule['pose_xyzyaw'][3,2]
        h,v,_=provider.at(3).surface(torch.tensor(p[:2],dtype=torch.float32)[None])
        self.assertTrue(v.item());self.assertAlmostEqual(h.item(),p[2]+s.sizes[2,2]/2,places=6)

    def test_13_sample_reload_reproducible(self):
        reloaded=load_catalog(ROOT,seed=44)
        for old,new in zip(self.samples,reloaded):
            self.assertEqual(old.alignment,new.alignment)
            for name in ('mesh','foot','body','schedule'):
                for k,v in getattr(old,name).items():np.testing.assert_array_equal(v,getattr(new,name)[k])

    def test_14_body_prior_requires_explicit_quality_trust(self):
        self.assertEqual(float(self.motion.quality.sum()),0.)
        trusted=PMTMotionReferenceProvider(self.samples,self.kin,trust_offline_body_prior=True)
        self.assertEqual(float(trusted.quality.sum()),576.)
        np.testing.assert_allclose(trusted.qd[:288],self.samples[0].body['joint_vel'],atol=2e-6)
        np.testing.assert_allclose(trusted.body_v[:288],self.samples[0].body['body_lin_vel_w'],atol=2e-5)
        f=torch.tensor([0]);goals=RobanFootGoals(trusted,self.kin,GoalSpec('roban_foot',2,.1))
        task=RobanFootMDP(self.kin,trusted,torch.zeros(21),goals,FootRewardConfig(body_prior_weight=.03))
        _,terms=task.reward(trusted.ref_state(f),f,torch.zeros(1,21),torch.zeros(1,21),True,command=trusted.anchor(f))
        torch.testing.assert_close(terms['body_prior'],torch.ones(1))
        torch.testing.assert_close(terms['body_prior_contribution'],torch.full((1,),.0006))

    def test_15_actual_contact_blocks_without_mutation(self):
        updates=[]
        class Clear:
            def distances(self,*args):return np.ones(3)*10
        c=PillarEventController(self.dynamic,lambda *x:updates.append(x))
        c.advance(0,set(),Clear())
        with self.assertRaises(PillarEventBlocked):c.advance(1,{2},Clear())
        self.assertEqual(updates,[]);self.assertTrue(c.blocked)
        self.assertEqual(c.log[-1]['reason'],'actual_robot_contact_on_recycled_pillar')

    def test_16_safe_dwell_overlap_deadline_and_skip(self):
        class Geometry:
            def __init__(self,dist=10.):self.dist=dist
            def distances(self,*a):return np.ones(3)*self.dist
        c=PillarEventController(self.dynamic,lambda *a:None,safe_frames=3)
        c.advance(0,set(),Geometry())
        with self.assertRaisesRegex(PillarEventBlocked,'safe_frames'):c.advance(1,set(),Geometry())
        c=PillarEventController(self.dynamic,lambda *a:None)
        c.advance(0,set(),Geometry(0.))
        with self.assertRaisesRegex(PillarEventBlocked,'not_clear'):c.advance(1,set(),Geometry(0.))
        c=PillarEventController(self.dynamic,lambda *a:None)
        with self.assertRaisesRegex(ValueError,'consecutively'):c.advance(2,set(),Geometry())
        with self.assertRaisesRegex(ValueError,'planned masks'):c.advance(0,np.array([1,1]),Geometry())

    def test_18_target_overlap_rejected(self):
        class TargetBlocked:
            def distances(self,poses,active):
                dist=np.ones(3)*10.
                if np.linalg.norm(poses[2,:2]-np.array(selftarget)[:2])<1e-8:dist[2]=-.01
                return dist
        selftarget=self.dynamic.events[0]['pose_after']
        updates=[];c=PillarEventController(self.dynamic,lambda *x:updates.append(x))
        c.advance(0,set(),TargetBlocked())
        with self.assertRaisesRegex(PillarEventBlocked,'new_target_intersects_robot'):c.advance(1,set(),TargetBlocked())
        self.assertFalse(updates)

    def test_19_contact_guard_uses_pairs_even_at_zero_force(self):
        snapshot=ContactSnapshot(rows=[dict(body0='/World/TestCube',body1='/World/Pillars/pillar_2',force_on_body0=[0.,0.,0.])],
            foot_force=np.zeros((2,3)),pillar_force=np.zeros((3,3)),foot_pillar_force=np.zeros((2,3,3)),
            robot_contact_pillars=set(),contact_positions=np.zeros((1,3)),cube_force=np.zeros(3))
        self.assertEqual(contacted_pillars(snapshot,'TestCube'),{2})
        self.assertEqual(contacted_pillars(snapshot,'Robot'),set())

    def test_17_all_events_replay_exactly_when_safe(self):
        class Clear:
            def distances(self,*a):return np.ones(3)*10
        poses=self.dynamic.schedule['pose_xyzyaw'][0].copy();active=self.dynamic.schedule['active'][0].copy()
        def apply(pid,p,on):poses[pid]=p;active[pid]=on
        c=PillarEventController(self.dynamic,apply)
        for f in range(self.dynamic.frames):
            c.advance(f,set(),Clear())
            np.testing.assert_array_equal(poses,self.dynamic.schedule['pose_xyzyaw'][f])
            np.testing.assert_array_equal(active,self.dynamic.schedule['active'][f])
        self.assertEqual(sum(e['status']=='GUARD_PASS' for e in c.log),4)


if __name__=='__main__':unittest.main(verbosity=2)
