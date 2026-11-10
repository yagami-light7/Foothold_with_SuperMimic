"""Check the evaluation aggregation protocol before uploading cloud charts."""
import unittest

from sync_s54_m34_results import aggregate, evaluation_metrics


def row(name, value, steps, noise=False, seed=0):
    return dict(id=name, mode='straight', split='test', normalized_return=value,
                reference_steps=steps, reset_noise=noise, seed=seed, survived=True,
                root_position_rmse_m=.1, root_orientation_rmse_rad=.2,
                joint_rmse_rad=.3, foot_position_rmse_m=.4, episode_s=steps*.02)


class EvaluationProtocolTests(unittest.TestCase):
    def test_reference_mean_does_not_favor_long_clips_or_extra_seeds(self):
        # A has three seeds and a longer reference; B has one seed.
        result = aggregate([row('A', .2, 100), row('A', .4, 100, True, 1),
                            row('A', .6, 100, True, 2), row('B', 1., 10)])
        self.assertAlmostEqual(result['NormalizedReturn'], .7)
        self.assertAlmostEqual(result['ReferenceStepWeightedReturn'], 50/110)

    def test_nominal_and_perturbed_test_results_remain_distinct(self):
        summary = dict(status='completed', checkpoint_sha256='same', splits={
            'test': dict(status='completed', checkpoint_sha256='same',
                         rows=[row('forward_unseen', .9, 100),
                               row('forward_unseen', .3, 100, True, 1)])})
        values = evaluation_metrics(summary)
        self.assertEqual(values['Eval/Test/Nominal/NormalizedReturn'], .9)
        self.assertEqual(values['Eval/Test/Perturbed/NormalizedReturn'], .3)

    def test_wrong_split_or_incomplete_result_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluation_metrics(dict(status='failed'))
        summary = dict(status='completed', checkpoint_sha256='same', splits={
            'train': dict(status='completed', checkpoint_sha256='same',
                          rows=[row('forward_unseen', .9, 100)])})
        with self.assertRaisesRegex(ValueError, 'Mixed train/test'):
            evaluation_metrics(summary)


if __name__ == '__main__':
    unittest.main()
