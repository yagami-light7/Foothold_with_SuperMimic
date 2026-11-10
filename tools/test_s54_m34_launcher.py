"""Host-side M3.4 launch tests with Docker and SwanLab replaced by recorders."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'tools'))
from sync_swanlab import load_swanlab


class HostLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='s54 m34 host ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'tools').mkdir(); (self.root/'configs').mkdir(); (self.root/'bin').mkdir()
        shutil.copy2(REPO/'tools/run_s54_m34.sh', self.root/'tools/run_s54_m34.sh')
        (self.root/'configs/train.json').write_text('{}')
        self.calls = self.root/'calls.jsonl'
        common = ('#!/usr/bin/env python3\nimport json, os, sys\nfrom pathlib import Path\n'
                  f'calls=Path({str(self.calls)!r})\n'
                  'def record(kind):\n'
                  ' with calls.open("a") as f: f.write(json.dumps([kind,sys.argv[1:]])+"\\n")\n')
        docker = self.root/'bin/docker'
        docker.write_text(common+'record("docker")\na=sys.argv[1:]\n'
            'if a[0]=="inspect" and "--format" in a:\n'
            ' fmt=a[a.index("--format")+1]\n'
            f' if "Mounts" in fmt: print(os.environ.get("FAKE_MOUNT",{str(self.root)!r}))\n'
            ' elif "State.Running" in fmt: print("true")\n'
            'elif a[0]=="exec" and "tools/check_s54_cuda.py" in a: raise SystemExit(int(os.environ.get("FAKE_CUDA_EXIT","0")))\n'
            'elif a[0]=="exec" and "nvidia-smi" in a: print(os.environ.get("FAKE_BUSY",""))\n')
        sdk = self.root/'bin/swan-python'
        sdk.write_text(common+'record("sdk")\n'
                       'if "--check" in sys.argv: raise SystemExit(int(os.environ.get("FAKE_AUTH_EXIT","0")))\n')
        git = self.root/'bin/git'; git.write_text('#!/bin/sh\nprintf "test-source-sha\\n"\n')
        for p in (docker, sdk, git): p.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.root/'bin')+os.pathsep+os.environ['PATH'],
                        SWANLAB_ENABLED='false', SWANLAB_PYTHON=str(sdk), CONTAINER_NAME='s54_m34_test')

    def run_launcher(self, mode='train', checkpoint=None, **env):
        command = ['bash', 'tools/run_s54_m34.sh', mode, 's54_host_test', 'configs/train.json']
        if checkpoint is not None:
            command.append(str(checkpoint))
        result = subprocess.run(command,
                                cwd=self.root, env=self.env | env, capture_output=True, text=True)
        return result

    def records(self):
        return [json.loads(s) for s in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def test_disabled_swanlab_needs_no_sdk_and_host_owns_run(self):
        result = self.run_launcher(SWANLAB_PYTHON='/missing/python')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('SwanLab: disabled', result.stdout)
        self.assertFalse(any(kind == 'sdk' for kind, _ in self.records()))
        self.assertEqual((self.root/'logs/s54_host_test').stat().st_uid, os.getuid())

    def test_enabled_swanlab_checks_auth_then_detaches_core_only_uploader(self):
        result = self.run_launcher(SWANLAB_ENABLED='true', SWANLAB_PROJECT='S54 Project',
                                   SWANLAB_WORKSPACE='my-team', SWANLAB_EXPERIMENT='run one')
        self.assertEqual(result.returncode, 0, result.stderr)
        deadline = time.monotonic()+3
        while len([1 for kind, _ in self.records() if kind == 'sdk']) < 2 and time.monotonic() < deadline:
            time.sleep(.01)
        records = self.records()
        self.assertEqual(records[0][0], 'sdk'); self.assertIn('--check', records[0][1])
        upload = next(args for kind, args in records if kind == 'sdk' and '--follow' in args)
        self.assertIn('--core_only', upload); self.assertIn('--completion_marker', upload)
        for flag, expected in [('--project', 'S54 Project'), ('--workspace', 'my-team'),
                               ('--experiment_name', 'run one'), ('--startup_timeout', '3600')]:
            self.assertEqual(upload[upload.index(flag)+1], expected)
        self.assertTrue((self.root/'logs/s54_host_test/swanlab_sync.log').exists())

    def test_failed_auth_prevents_gpu_launch(self):
        result = self.run_launcher(SWANLAB_ENABLED='true', FAKE_AUTH_EXIT='7')
        self.assertEqual(result.returncode, 7)
        self.assertFalse(any(kind == 'docker' for kind, _ in self.records()))
        self.assertFalse((self.root/'logs/s54_host_test').exists())

    def test_wrong_mount_and_busy_gpu_cannot_launch_training(self):
        for environment in ({'FAKE_MOUNT': '/another/repo'}, {'FAKE_BUSY': '1234'}):
            with self.subTest(environment=environment):
                result = self.run_launcher(**environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root/'logs/s54_host_test').exists())
                self.assertFalse(any('tmux' in args for kind, args in self.records() if kind == 'docker'))

    def test_cuda_execution_failure_prevents_job_and_run_creation(self):
        result = self.run_launcher(FAKE_CUDA_EXIT='17')
        self.assertEqual(result.returncode, 1)
        self.assertIn('CUDA preflight failed', result.stderr)
        self.assertFalse((self.root/'logs/s54_host_test').exists())
        self.assertFalse(any('tmux' in args for kind, args in self.records() if kind == 'docker'))
        self.assertFalse(any(kind == 'sdk' and '--follow' in args for kind, args in self.records()))

    def test_resume_passes_checkpoint_and_reports_skipped_preparation(self):
        checkpoint = Path('logs/s54_old/last.pt')
        (self.root/checkpoint).parent.mkdir(parents=True)
        (self.root/checkpoint).write_bytes(b'checkpoint path fixture')
        result = self.run_launcher(mode='resume', checkpoint=checkpoint)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PD/fresh smoke skipped', result.stdout)
        job = next(args for kind, args in self.records() if kind == 'docker' and 'tmux' in args)
        self.assertIn('--resume logs/s54_old/last.pt', job[-1])
        self.assertEqual((self.root/checkpoint).read_bytes(), b'checkpoint path fixture')

    def test_missing_resume_checkpoint_fails_before_auth_or_docker(self):
        result = self.run_launcher(mode='resume', checkpoint='missing.pt', SWANLAB_ENABLED='true')
        self.assertEqual(result.returncode, 2)
        self.assertIn('Resume checkpoint required', result.stdout)
        self.assertEqual(self.records(), [])

    def test_core_sdk_disables_environment_probes_and_uses_supported_login(self):
        captured = []
        class Settings:
            def __init__(self, **kwargs): captured.append(kwargs)
            @staticmethod
            def Probe(**kwargs): return kwargs
        logins = []
        sdk = SimpleNamespace(Settings=Settings, merge_settings=lambda settings: None,
                              login=lambda: logins.append(True))
        with patch.dict(sys.modules, swanlab=sdk):
            load_swanlab('online', authenticate=True)
        self.assertEqual(logins, [True])
        self.assertFalse(captured[0]['interactive'])
        self.assertEqual(captured[0]['probe'], dict(hardware=False, runtime=False, requirements=False,
            conda=False, git=False, swanlab=False, monitor=False))


if __name__ == '__main__': unittest.main(verbosity=2)
