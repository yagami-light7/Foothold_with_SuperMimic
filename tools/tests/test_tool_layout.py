"""Layout moves preserve configuration meaning and resolvable tool entrypoints."""
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from supermimic.config_paths import resolve_config_path
from supermimic.foot_mdp import resolve_foot_task


class LayoutTests(unittest.TestCase):
    def test_all_mapped_targets_exist(self):
        mapping=json.loads((ROOT/'tools/layout_migration.json').read_text())
        for section in ('scripts','configs'):
            for old,new in mapping[section].items():
                with self.subTest(old=old):self.assertTrue((ROOT/new).is_file(),new)

    def test_legacy_relative_and_foreign_absolute_config_paths(self):
        mapping=json.loads((ROOT/'tools/layout_migration.json').read_text())['configs']
        for old,new in mapping.items():
            with self.subTest(old=old):
                self.assertEqual(resolve_config_path(old).resolve(),(ROOT/new).resolve())
                self.assertEqual(resolve_config_path('/old/checkout/'+old).resolve(),(ROOT/new).resolve())

    def test_legacy_standing_contract_unchanged(self):
        old=resolve_foot_task('configs/s54_standing_balance.json')
        new=resolve_foot_task('configs/s54/standing/s54_standing_balance.json')
        self.assertEqual(old,new)

    def test_stage1_training_and_validation_are_separate(self):
        self.assertTrue((ROOT/'tools/train/run_s54_stage1_walk.sh').exists())
        self.assertTrue((ROOT/'tools/validation/test_env_s54.py').exists())
        self.assertFalse((ROOT/'tools/tests/test_env_s54.py').exists())
        self.assertFalse(list((ROOT/'tools').glob('test_*.py')))


if __name__=='__main__':unittest.main(verbosity=2)
