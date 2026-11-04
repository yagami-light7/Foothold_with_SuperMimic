"""Execute the trainer's actual logging branch without starting Isaac Lab.

Cadence=1 smoke runs cannot expose references to log-only variables on skipped
iterations. Exercise both cadence=20 paths and GPU telemetry failure here.
"""
import ast
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from supermimic.curriculum import Curriculum


class CurriculumLoggingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=Path(__file__).resolve().parents[1]/'supermimic/train.py'
        tree=ast.parse(source.read_text())
        loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For)
                  and isinstance(n.target,ast.Name) and n.target.id=='it')
        branch=next(n for n in loop.body if isinstance(n,ast.If)
                    and isinstance(n.test,ast.Name) and n.test.id=='curriculum')
        cls.code=compile(ast.Module(body=[branch],type_ignores=[]),str(source),'exec')

    def namespace(self,path,step,failure=False):
        probe=Mock(return_value=SimpleNamespace(stdout='0, 172\n'))
        if failure: probe.side_effect=OSError('GPU telemetry fixture unavailable')
        return dict(curriculum=Curriculum(),current_it=step,args=SimpleNamespace(log_every=20),
                    log_path=path,elapsed=lambda:1.,STOP_REQUESTED=False,diag_agg={},
                    opt_pi=SimpleNamespace(param_groups=[{'lr':1e-5}]),
                    opt_v=SimpleNamespace(param_groups=[{'lr':1e-5}]),
                    subprocess=SimpleNamespace(run=probe,SubprocessError=subprocess.SubprocessError),
                    torch=SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda:1024)),
                    json=json,os=__import__('os'),emit=Mock(),collection_frames=128,collection_seconds=1.)

    def test_first_nonlogging_iteration(self):
        with tempfile.TemporaryDirectory() as path:
            ns=self.namespace(Path(path),1)
            exec(self.code,ns)
            ns['subprocess'].run.assert_not_called();ns['emit'].assert_not_called()
            self.assertFalse((Path(path)/'training_status.json').exists())
            self.assertNotIn('status_temp',ns);self.assertNotIn('gpu_stats',ns)

    def test_logging_iteration_and_following_skip(self):
        with tempfile.TemporaryDirectory() as path:
            ns=self.namespace(Path(path),20)
            exec(self.code,ns)
            saved=json.loads((Path(path)/'training_status.json').read_text())
            self.assertEqual(saved['iteration'],20);self.assertEqual(saved['status'],'running')
            self.assertFalse((Path(path)/'training_status.json.tmp').exists())
            ns['emit'].assert_called_once()
            ns['current_it']=21;exec(self.code,ns)
            self.assertEqual(ns['emit'].call_count,1)
            self.assertEqual(json.loads((Path(path)/'training_status.json').read_text())['iteration'],20)

    def test_memory_limit_requests_safe_stop(self):
        with tempfile.TemporaryDirectory() as path:
            ns=self.namespace(Path(path),20)
            ns['args'].max_device_memory_mib=100.
            with contextlib.redirect_stdout(io.StringIO()) as output: exec(self.code,ns)
            self.assertTrue(ns['STOP_REQUESTED'])
            self.assertIn('safe save/stop requested',output.getvalue())

    def test_telemetry_failure_still_writes_status(self):
        with tempfile.TemporaryDirectory() as path:
            ns=self.namespace(Path(path),20,failure=True)
            with contextlib.redirect_stdout(io.StringIO()) as output: exec(self.code,ns)
            self.assertIn('GPU telemetry unavailable',output.getvalue())
            self.assertEqual(json.loads((Path(path)/'training_status.json').read_text())['iteration'],20)
            ns['emit'].assert_called_once()

if __name__=='__main__': unittest.main()
