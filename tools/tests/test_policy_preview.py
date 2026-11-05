"""Duration/offset boundaries and shared-GPU isolation for synchronized previews."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.visualization import policy_preview as preview


class PreviewTests(unittest.TestCase):
    def test_requested_duration_clamps_to_reference_end(self):
        self.assertEqual(preview.reference_window(13066,300),(0,13065))
        self.assertEqual(preview.reference_window(13066,30),(0,1500))
        self.assertEqual(preview.reference_window(13066,300,260),(13000,65))

    def test_substep_duration_never_reads_past_reference(self):
        self.assertEqual(preview.reference_window(13066,.039),(0,1))
        self.assertEqual(preview.reference_window(13066,300,261.28),(13064,1))

    def test_invalid_and_empty_intervals_are_rejected(self):
        for seconds,start in [(0,0),(-1,0),(float('nan'),0),(float('inf'),0),
                              (1,-1),(1,.011),(1,261.3),(1,300),(.001,0)]:
            with self.subTest(seconds=seconds,start=start),self.assertRaises(ValueError):
                preview.reference_window(13066,seconds,start)

    def gpu_case(self,processes,free='20000\n'):
        with tempfile.TemporaryDirectory(prefix='s54_preview_gpu_',dir='/tmp') as tmp:
            with patch.object(preview,'ROOT',Path(tmp)),patch.object(preview.subprocess,'check_output',
                    side_effect=[processes,free]):
                with preview.gpu_slot():pass

    def test_desktop_can_remain_but_compute_jobs_are_blocked(self):
        self.gpu_case('807229, /opt/todesk/bin/ToDesk_Session\n')
        with self.assertRaisesRegex(RuntimeError,'compute job'):
            self.gpu_case('123, /usr/bin/python3\n')
        with self.assertRaisesRegex(RuntimeError,'4096'):
            self.gpu_case('',free='1000\n')


if __name__=='__main__':unittest.main(verbosity=2)
