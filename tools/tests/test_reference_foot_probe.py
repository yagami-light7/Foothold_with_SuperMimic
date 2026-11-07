"""Goal transport and future-window guards for the inference-only probe."""
import copy
import unittest

from supermimic.goals import GoalSpec
from tools.validation.evaluate_s54_reference_foot_probe import encoding_contract,reference_window


class ReferenceFootProbe(unittest.TestCase):
    def test_source_changes_explicitly_but_actor_encoding_is_identical(self):
        saved=GoalSpec('foot_trajectory',10,.1).contract()
        ref=GoalSpec('reference_foot',10,.1).contract()
        result=encoding_contract(saved,ref)
        self.assertEqual(result['changed_fields'],['mode','version'])
        self.assertEqual(saved['dim'],ref['dim'])

    def test_dimension_coordinates_rotation_and_spacing_mismatch_rejected(self):
        saved=GoalSpec('foot_trajectory',10,.1).contract()
        ref=GoalSpec('reference_foot',10,.1).contract()
        for key,value in [('dim',730),('coordinates','world'),('rotation','euler'),('dt',.02),('frames',5)]:
            bad=copy.deepcopy(ref);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):encoding_contract(saved,bad)

    def test_no_reference_end_clamping_in_future_goals(self):
        self.assertEqual(reference_window(60.,10.,0,13065),(3000,500))
        self.assertEqual(reference_window(250.,10.,0,13065),(12500,500))
        for offset,seconds in [(251.,10.),(260.,2.),(-1.,10.),(0.,.011),(60.011,10.)]:
            with self.subTest(offset=offset,seconds=seconds),self.assertRaises(ValueError):
                reference_window(offset,seconds,0,13065)


if __name__=='__main__':unittest.main()
