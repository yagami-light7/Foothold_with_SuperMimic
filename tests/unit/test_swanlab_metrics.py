"""精简云端曲线，完整诊断仍在本地；不连接SwanLab。"""

import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import tempfile
import unittest
from argparse import Namespace
from unittest.mock import Mock
from scripts.swanlab_metrics import compact_metrics,REWARD_KEYS
from scripts.train_4060 import Supervisor


class CompactMetricsTests(unittest.TestCase):
    def test_cloud_drops_joint_and_case_explosion_without_mutating_raw_data(self):
        raw: dict[str,float] = {f'straight/Control/position_error_rad/joint{i}':float(i) for i in range(100)}
        raw.update({'straight/Episode/normalized_return':.8,'straight/Reward/tracking':.03,
                    'straight/Loss/learning_rate':1e-4,'Eval/left_success_min':.9,
                    'Eval/adaptive/step400_L2000_R2000/reward':.1})
        snapshot=raw.copy();result=compact_metrics(raw)
        self.assertEqual(result,{'Train/return_0to1':.8,'Reward/tracking':.03,
                                 'PPO/learning_rate':1e-4,'Eval/left_success_min':.9})
        self.assertEqual(raw,snapshot)
        self.assertEqual(len(REWARD_KEYS),8)

    def test_supervisor_filters_only_at_sdk_boundary_and_supports_full_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            args=Namespace(max_hours=1,swanlab_detail='compact')
            supervisor=Supervisor(args,Path(temp),{})
            supervisor.sdk=Mock()
            raw={'straight/Reward/total':.1,'straight/Control/joint':.5}
            supervisor.log(raw,1)
            supervisor.sdk.log.assert_called_once_with({'Train/reward_per_step':.1},step=1)
            args.swanlab_detail='full';supervisor.log(raw,2)
            supervisor.sdk.log.assert_called_with(raw,step=2)


if __name__=='__main__':
    unittest.main()
