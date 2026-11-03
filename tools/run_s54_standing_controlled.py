"""Print, or explicitly execute, ONE finite RTX 3090 Standing pilot.

The existing SuperMimic training algorithm and CSV logger are reused unchanged.
Default is a dry run. One seed per invocation; no automatic resume/extension.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


def training_command(profile, python, seed, run, phase='calibration', resume=None):
    if not re.fullmatch(r's54_[A-Za-z0-9_.-]+', run):
        raise ValueError('Run name must be a fresh s54_* name')
    train = deepcopy(profile['train'])
    if phase not in ('calibration', 'pilot'):
        raise ValueError('Unknown controlled phase')
    if phase == 'calibration':
        if resume:
            raise ValueError('Calibration starts fresh')
        train['iterations'] = train['wm_only_iters']
    elif not resume:
        raise ValueError('Pilot requires a reviewed calibration last.pt (--resume)')
    if not 1 <= train['iterations'] <= 600 or not 0 < profile['max_wall_minutes'] <= 30:
        raise ValueError('Controlled pilot is limited to 600 iterations and 30 minutes')
    if not 1 <= train['num_envs'] <= 128 or train.get('resume'):
        raise ValueError('Use <=128 environments and a fresh run')
    cmd = [python, '-u', 'supermimic/train.py']
    for key, value in train.items():
        if isinstance(value, bool):
            if value: cmd.append('--' + key)
        else: cmd.extend(['--' + key, str(value)])
    cmd += ['--seed', str(seed), '--log_dir', 'logs/' + run]
    if resume:
        cmd += ['--resume', str(resume)]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/s54_standing_3090.json')
    p.add_argument('--seed', type=int, default=54)
    p.add_argument('--phase', choices=['calibration', 'pilot'], default='calibration')
    p.add_argument('--resume', help='Reviewed calibration last.pt for the bounded pilot phase')
    p.add_argument('--run')
    p.add_argument('--python', default=os.environ.get('S54_PYTHON', '/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python'))
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    a.run = a.run or f's54_standing_3090_{a.phase}_seed{a.seed}'
    profile = json.loads((REPO/a.config).read_text())
    try:
        cmd = training_command(profile, a.python, a.seed, a.run, a.phase, a.resume)
    except ValueError as exc:
        p.error(str(exc))
    print(shlex.join(cmd), flush=True)
    print(f"Phase {a.phase}; total iteration limit {cmd[cmd.index('--iterations')+1]} / "
          f"{profile['max_wall_minutes']} minutes; seed {a.seed}", flush=True)
    if not a.execute: return
    if a.resume and not (REPO/a.resume).is_file():
        p.error('Resume checkpoint does not exist')
    path = REPO/'logs'/a.run
    if path.exists(): p.error('Existing run is protected; choose a fresh name')
    path.mkdir(parents=True)
    (path/'controlled_profile.json').write_text(json.dumps(dict(profile=profile, command=cmd, phase=a.phase), indent=2)+'\n')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4',
               WARP_CACHE_PATH=str(REPO/'.cache/warp'), TORCHINDUCTOR_CACHE_DIR=str(REPO/'.cache/inductor'))
    with (path/'stdout.log').open('w') as stdout:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=profile['max_wall_minutes']*60)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            code = 124
    (path/'controlled_exit.json').write_text(json.dumps(dict(exit_code=code, wall_limit_minutes=profile['max_wall_minutes'])))
    print(f'Exit {code}; CSV/checkpoints/stdout: {path}', flush=True)
    sys.exit(code)


if __name__ == '__main__': main()
