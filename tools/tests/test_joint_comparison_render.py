"""Recording selection and physical end boundaries; no renderer/GPU required."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.visualization.render_s54_joint_comparison import select_evaluation, select_row, playback_index


class ComparisonReplay(unittest.TestCase):
    def pair(self, directory, iteration, other_checkpoint=False, complete=True):
        for name in ('motion', 'foot'):
            if name == 'foot' and not complete:
                continue
            path = directory/name/'summary.json'; path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(interface=name, iteration=iteration, full=False,
                checkpoint='other.pt' if other_checkpoint and name=='foot' else 'last.pt', rows=[])))

    def test_latest_complete_pair_never_mixes_checkpoints_or_partial_evaluations(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.pair(run/'evaluation/iteration_000010', 10)
            self.pair(run/'evaluation/iteration_000020', 20)
            self.pair(run/'evaluation/iteration_000030', 30, other_checkpoint=True)
            self.pair(run/'evaluation/iteration_000040', 40, complete=False)
            directory, motion, foot = select_evaluation(run)
            self.assertEqual(directory.name, 'iteration_000020')
            self.assertEqual(motion['iteration'], foot['iteration'])
            with self.assertRaises(ValueError):
                select_evaluation(run, run/'evaluation/iteration_000030')

    def test_selects_requested_duration_and_offset_without_new_simulation(self):
        summary = dict(interface='foot', rows=[dict(seed=54,duration_s=t) for t in (2.,5.,10.)])
        self.assertEqual(select_row(summary,seed=54,duration=10.)['duration_s'],10.)
        with self.assertRaises(ValueError):
            select_row(summary,seed=55,duration=10.)
        summary = dict(interface='motion', rows=[dict(seed=54,offset_s=t) for t in (0.,60.,120.)])
        self.assertEqual(select_row(summary,seed=54,offset=60.)['offset_s'],60.)

    def test_holds_terminal_state_only_after_actual_physical_end(self):
        self.assertEqual(playback_index(0.,149,.02),(0,False))
        self.assertEqual(playback_index(2.96,149,.02),(148,False))
        self.assertEqual(playback_index(2.98,149,.02),(149,True))
        self.assertEqual(playback_index(10.,149,.02),(149,True))
        with self.assertRaises(ValueError):
            playback_index(-1.,149,.02)


if __name__ == '__main__':
    unittest.main()
