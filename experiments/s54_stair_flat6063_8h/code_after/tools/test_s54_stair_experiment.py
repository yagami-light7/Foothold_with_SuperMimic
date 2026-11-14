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
