"""Metadata/replay and complete checkpoint state restoration regression tests."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from supermimic.curriculum import Curriculum
from supermimic.curriculum_runtime import atomic_save
from supermimic.buffer import ReplayBuffer,FIELDS
from tools.test_curriculum import CurriculumTests

class ReplayTests(unittest.TestCase):
    def test_immutable_command_labels_and_times(self):
        c=Curriculum();c.stage=4
        plan=c.plan(*CurriculumTests().inputs())
        b=ReplayBuffer(4,2,'cpu',foot_commands=True)
        tr={k:torch.zeros((2,)+((shape,) if isinstance(shape,int) else shape),dtype=dtype) for k,(shape,dtype) in FIELDS.items()}
        tr.update(command=plan,command_id=torch.tensor([10,11]));b.add(tr)
        original=plan.clone();plan.target_pos.add_(1);plan.labels.add_(2)
        restored=b.commands(torch.tensor([11,10]))
        torch.testing.assert_close(restored.labels,original.labels.flip(0))
        torch.testing.assert_close(restored.target_pos,original.target_pos.flip(0))
        c.observe(restored,torch.tensor([25,50]));self.assertEqual(sum(v for k,v in c.executed.items() if k.startswith('task_')),2)

    def test_stage_fixture_checkpoint_restore(self):
        c=Curriculum({'min_stage_seconds':0});m={'standing':dict(success_rate=1.,foot_rmse=.01)}
        c.update(m,1);c.update(m,2,allow_promotion=False);self.assertEqual(c.stage,0)
        self.assertTrue(c.update(m,3));self.assertEqual(c.stage,1)
        with tempfile.TemporaryDirectory() as d:
            atomic_save(c.state_dict(),Path(d)/'state.pt')
            other=Curriculum({'min_stage_seconds':0})
            other.load_state_dict(torch.load(Path(d)/'state.pt',weights_only=False))
            self.assertEqual(other.stage,1); self.assertEqual(other.stage_started,3)
            self.assertEqual([c.sample() for _ in range(30)],[other.sample() for _ in range(30)])

if __name__=='__main__':unittest.main()
