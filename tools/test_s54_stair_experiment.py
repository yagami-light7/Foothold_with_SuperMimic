"""CPU regressions for relocation, physical failure capture and frozen evaluation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import torch
from supermimic.kinematics import S54Kinematics
from supermimic.m34 import sha256
from supermimic.mdp import flatten_state
from supermimic.stair_data import load_stairs
from supermimic.stair_training import validate_transition,read_design,contract
from supermimic.stair_evaluation import evaluate_checkpoint
from sync_swanlab import evaluation_metrics
from t15_launch import training_command
from supermimic.stair_reset import TerrainResetValidator,segment_box_distance_squared


class StairExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg=json.loads((ROOT/'experiments/s54_stair_flat6063_8h/design.json').read_text())

    def test_standalone_reference_is_byte_exact_and_validated(self):
        motion=load_stairs(self.cfg['bundle'],S54Kinematics('cpu'),'cpu',
            ids=['stair_up_4'],npz_path=self.cfg['stair_npz'])
        self.assertEqual(motion.names,['stair_up_4']);self.assertEqual(motion.num_frames,436)
        self.assertEqual(sha256(self.cfg['stair_npz']),sha256(Path(self.cfg['bundle'])/'test/stair_up_4.npz'))
        with TemporaryDirectory() as directory:
            bad=Path(directory)/'bad.npz';bad.write_bytes(b'wrong file')
            with self.assertRaisesRegex(ValueError,'checksum'):
                load_stairs(self.cfg['bundle'],S54Kinematics('cpu'),'cpu',ids=['stair_up_4'],npz_path=bad)

    def test_physical_explosion_is_preserved_and_rejected(self):
        p=torch.tensor([[0.,0.,1.]])
        state=(p,torch.eye(3)[None],p*0,p*0,torch.zeros(1,29),torch.zeros(1,29))
        out={'s1':flatten_state(state),'r':torch.tensor([-.13]),'f':torch.tensor([0])}
        with TemporaryDirectory() as directory:
            validate_transition(out,SimpleNamespace(max_step_reward=.13),self.cfg,directory)
            bad=list(state);bad[2]=torch.tensor([[100.,0.,0.]])
            out['s1']=flatten_state(tuple(bad))
            with self.assertRaisesRegex(FloatingPointError,'Extreme Newton'):
                validate_transition(out,SimpleNamespace(max_step_reward=.13),self.cfg,directory)
            saved=torch.load(Path(directory)/'failure_physics_transition.pt',weights_only=False)
            self.assertEqual(float(saved['diagnostic_values']['root_speed_m_s'][0]),100.)

    def test_finite_negative_tracking_return_is_not_a_physics_failure(self):
        p=torch.tensor([[0.,0.,1.]])
        state=(p,torch.eye(3)[None],torch.tensor([[1.33,0.,0.]]),p*0,
               torch.zeros(1,29),torch.full((1,29),20.97))
        out={'s1':flatten_state(state),'r':torch.tensor([-5.3408728]),'f':torch.tensor([13958])}
        original=out['r'].clone()
        with TemporaryDirectory() as directory:
            validate_transition(out,SimpleNamespace(max_step_reward=.13),self.cfg,directory)
            self.assertTrue(torch.equal(out['r'],original))
            self.assertFalse((Path(directory)/'failure_physics_transition.pt').exists())
            row=json.loads((Path(directory)/'tracking_reward_diagnostics.jsonl').read_text())
            self.assertEqual(row['kind'],'low_finite_tracking_reward')
            self.assertLess(row['min_normalized_reward'],-41.)

    def test_nonfinite_state_still_stops(self):
        p=torch.tensor([[0.,0.,1.]])
        state=(p,torch.eye(3)[None],p*0,p*0,torch.zeros(1,29),torch.zeros(1,29))
        out={'s1':flatten_state(state),'r':torch.tensor([0.]),'f':torch.tensor([0])}
        out['s1'][0,0]=float('nan')
        with TemporaryDirectory() as directory:
            with self.assertRaises(FloatingPointError):
                validate_transition(out,SimpleNamespace(max_step_reward=.13),self.cfg,directory)

    def test_formal_profile_matches_previous_flat_experiment(self):
        base=ROOT/'experiments/s54_stair_flat6063_8h'
        profile=json.loads((base/'flat_training_profile.json').read_text())
        cfg=json.loads((base/'train.json').read_text())
        previous=json.loads((ROOT/profile['source']).read_text())
        for key,expected in profile['matched_fields'].items():
            with self.subTest(parameter=key):
                self.assertEqual(cfg[key],previous[key]);self.assertEqual(cfg[key],expected)
        self.assertEqual(cfg['wall_seconds'],28800)
        self.assertEqual(cfg['m34_eval_every'],1000)
        smoke=json.loads((base/'smoke.json').read_text())
        for key in ['num_envs','buffer_steps','wm_batch','wm_batches','pi_batch','horizon','lr','actor_lr','warmup_steps']:
            self.assertEqual(smoke[key],cfg[key])

    def test_segment_box_distance_handles_crossing_parallel_and_point(self):
        start=torch.tensor([[-2.,2.,0.],[-2.,0.,0.],[2.,2.,2.]])
        end=torch.tensor([[2.,2.,0.],[2.,0.,0.],[2.,2.,2.]])
        distance=segment_box_distance_squared(start,end,torch.full_like(start,-1.),torch.ones_like(start))
        self.assertTrue(torch.allclose(distance,torch.tensor([1.,0.,3.])))

    def test_saved_reset_explosions_are_rejected_before_physics(self):
        capture=ROOT/'logs/s54_stair_flat6063_smoke_v3/failure_physics_transition.pt'
        if not capture.exists():self.skipTest('Local v3 diagnostic capture unavailable')
        from supermimic.stair_training import library
        from supermimic.mdp import unflatten_state
        kin=S54Kinematics('cpu');motion=library(self.cfg,kin,'cpu')
        validator=TerrainResetValidator(kin,self.cfg['terrain'],clearance=-.001,max_attempts=64)
        data=torch.load(capture,map_location='cpu',weights_only=False)
        state=unflatten_state(data['s']);original=data['s'].clone()
        self.assertFalse(validator.valid(state).any())
        torch.manual_seed(54)
        accepted=validator.condition(state,motion.ref_state(data['f']))
        self.assertTrue(validator.valid(accepted).all())
        self.assertTrue(torch.equal(data['s'],original))

    def test_all_stair_nominal_frames_meet_original_geometry_tolerance(self):
        kin=S54Kinematics('cpu')
        motion=load_stairs(self.cfg['bundle'],kin,'cpu',ids=['stair_up_4'],npz_path=self.cfg['stair_npz'])
        validator=TerrainResetValidator(kin,self.cfg['terrain'],clearance=-.001)
        self.assertTrue(validator.valid(motion.ref_state(torch.arange(436))).all())

    def test_mesh_flat_ground_preserves_original_box_surfaces(self):
        import numpy as np
        from supermimic.stair_runtime import flat_zone_mesh
        from supermimic.stair_data import geometry_tools
        mesh=flat_zone_mesh()
        self.assertTrue(np.array_equal(mesh.bounds,np.array([[8.,-12.,-.1],[32.,12.,0.]])))
        rng=np.random.default_rng(54)
        points=rng.uniform([8.,-12.],[32.,12.],(1024,2))
        self.assertTrue(np.all(geometry_tools().top_surface(mesh,points)==0.))
        self.assertTrue(mesh.is_watertight)
        self.assertAlmostEqual(mesh.volume,24*24*.1)

    def test_newton_contact_refresh_preserves_clock_and_substeps(self):
        from supermimic.stair_runtime import make_stair_sim
        cfg=make_stair_sim('cuda:0',SimpleNamespace())
        self.assertEqual(cfg.dt,.02)
        self.assertEqual(cfg.physics.num_substeps,4)
        self.assertEqual(cfg.physics.collision_decimation,1)
        self.assertFalse(cfg.physics.solver_cfg.use_mujoco_contacts)

    def test_seed_list_becomes_separate_cli_arguments(self):
        command=training_command({'m34_eval_seeds':[0,1,2],'no_compile':True},'python')
        self.assertEqual(command[command.index('--m34_eval_seeds')+1:][:3],['0','1','2'])
        self.assertIn('--no_compile',command)

    def test_missing_or_failed_smoke_never_unlocks_training(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);cfg={**self.cfg,'required_smoke_gate':str(root/'gate.json')}
            path=root/'design.json';path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'not approved'):
                read_design(path)
            (root/'gate.json').write_text(json.dumps(dict(passed=False,data_contract=contract(cfg))))
            with self.assertRaisesRegex(ValueError,'successful Smoke'):
                read_design(path)

    def test_evaluation_is_frozen_cached_and_scalar_only(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);ck=root/'checkpoint.pt';ck.write_bytes(b'frozen checkpoint')
            output=root/'eval';calls=[]
            def child(command,**kwargs):
                calls.append(command)
                self.assertIn('--no-video',command)
                target=Path(command[command.index('--out')+1]);target.mkdir()
                row=dict(seed=0,normalized_return=.95,raw_reward_sum=10.,survived=True,
                    completion_rate=1.,episode_s=8.7,yaw_rmse_rad=.02,root_xyz_rmse_m=[.01,.02,.03],
                    left_right_foot_position_rmse_m=[.02,.03])
                (target/'summary.json').write_text(json.dumps(dict(status='completed',
                    checkpoint_sha256=sha256(ck),rows=[row])))
                return subprocess.CompletedProcess(command,0)
            with patch('supermimic.stair_evaluation.subprocess.run',side_effect=child):
                report=evaluate_checkpoint(ck,self.cfg,output,seeds=[0])
                again=evaluate_checkpoint(ck,self.cfg,output,seeds=[0])
            self.assertEqual(report,again);self.assertEqual(len(calls),4)
            self.assertTrue(report['flat_retention_passed'])
            self.assertEqual(evaluation_metrics(report),report['scalars'])
            self.assertTrue(all(isinstance(v,(float,int)) for v in evaluation_metrics(report).values()))
            ck.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'different checkpoint'):
                evaluate_checkpoint(ck,self.cfg,output,seeds=[0])

    def test_failed_evaluation_keeps_diagnostics(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);ck=root/'checkpoint.pt';ck.write_bytes(b'weights')
            output=root/'eval'
            with patch('supermimic.stair_evaluation.subprocess.run',side_effect=subprocess.TimeoutExpired('newton',1)):
                with self.assertRaises(subprocess.TimeoutExpired):
                    evaluate_checkpoint(ck,self.cfg,output,seeds=[0])
            self.assertEqual(json.loads((output/'status.json').read_text())['status'],'failed')
            self.assertFalse((output/'summary.json').exists())
            self.assertTrue((output/'attempt_001/StairTrain.log').exists())


if __name__=='__main__':unittest.main(verbosity=2)
