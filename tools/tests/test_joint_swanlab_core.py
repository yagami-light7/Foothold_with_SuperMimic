"""CPU-only dashboard contract tests; no SDK, network, or training required."""
import unittest

from tools.ops.sync_s54_joint_ab1 import CORE_KEYS, values, verification_targets, iteration_rows


class CoreMetrics(unittest.TestCase):
    def test_closed_allowlist_drops_diagnostics_and_nonfinite_values(self):
        result=values({'values':{'motion/wm_loss':.1,'foot/actor_gradnorm':2.,
            'motion/delta_decoder':3.,'Counters/real_foot':99,'unannounced/metric':7,
            'motion/sim_rew':float('nan'),'foot/sim_rew':float('inf'),
            'Eval/Foot/10s/SurvivalRate':True}})
        self.assertEqual(result,{'Motion/WorldModel/Loss':.1,'Foot/Actor/GradNorm':2.})
        self.assertEqual(len(CORE_KEYS),32)

    def test_fixed_probe_means_and_full_short_protocols_are_separate(self):
        result=values({'values':{
            'Eval/Motion/10s/NormalizedReturn/offset0':.8,
            'Eval/Motion/10s/NormalizedReturn/offset60':.9,
            'Eval/Motion/Full/NormalizedReturn/offset0':.7,
            'Eval/Foot/10s/NormalizedReturn/seed54':.1,
            'Eval/Foot/10s/NormalizedReturn/seed55':.3,
            'Eval/Foot/10s/foot_position_rmse_m/seed54':1.,
            'Eval/Foot/10s/foot_position_rmse_m/seed55':3.,
            'Eval/Foot/10s/actual_forward_speed_m_s/seed54':-1.,
            'Eval/Foot/10s/actual_forward_speed_m_s/seed55':-2.,
            'Eval/Foot/10s/WM32/feet_position_rmse_m/seed54':.1,
            'Eval/Foot/5s/NormalizedReturn/seed54':.2,
            'Eval/Foot/10s/SurvivalRate':0.}})
        self.assertAlmostEqual(result['Eval/Motion/10s/NormalizedReturn'],.85)
        self.assertEqual(result['Eval/Motion/Full/NormalizedReturn'],.7)
        self.assertAlmostEqual(result['Eval/Foot/10s/NormalizedReturn'],.2)
        self.assertEqual(result['Eval/Foot/10s/foot_position_rmse_m'],2.)
        self.assertEqual(result['Eval/Foot/10s/actual_forward_speed_m_s'],-1.5)
        self.assertEqual(result['Eval/Foot/10s/SurvivalRate'],0.)
        self.assertTrue(set(result)<=CORE_KEYS)
        self.assertEqual(len(result),6)

    def test_final_verification_uses_latest_core_values_without_uploading_counters(self):
        events=[{'event_step':1,'values':{'motion/wm_loss':.2,'foot/wm_loss':.3,
            'Progress/Iteration':20,'Progress/WallHours':.1}},
            {'event_step':2,'values':{'Counters/real_motion':999,
            'Progress/Iteration':21,'Progress/WallHours':.2}}]
        targets=verification_targets(events)
        self.assertEqual(targets['Motion/WorldModel/Loss'],(1,.2))
        self.assertEqual(targets['Progress/Iteration'],(2,21))
        self.assertNotIn('Counters/real_motion',values(events[-1]))

    def test_export_uses_learner_iteration_and_merges_eval_without_losing_train(self):
        events=[{'event_step':1,'iteration':20,'values':{'foot/wm_loss':.1}},
            {'event_step':2,'iteration':20,'values':{'Eval/Foot/10s/SurvivalRate':0.}},
            {'event_step':3,'iteration':21,'values':{'Progress/WallHours':.2}}]
        exported=iteration_rows(events)
        self.assertEqual([x['event_step'] for x in exported],[20,21])
        self.assertEqual(exported[0]['values'],{'Foot/WorldModel/Loss':.1,
            'Eval/Foot/10s/SurvivalRate':0.,'Progress/Iteration':20})


if __name__=='__main__':unittest.main()
