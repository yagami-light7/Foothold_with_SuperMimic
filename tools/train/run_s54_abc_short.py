"""Serial, tmux-protected A/B/C short runs from one immutable Teacher snapshot."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.train.run_s54_motion_prior import command_for, default_python
from tools.ops.swanlab_jobs import tmux_proxy_options


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def plans(profile_path, profile, batch, python, teacher):
    result = []
    for group in 'ABC':
        args = SimpleNamespace(stage='stage2', stage1_train=False, stage2_short=True,
            profile=str(profile_path), group=group, task=profile['task'], seed=profile['seed'],
            source=None if group == 'A' else str(teacher), resume=None, wm_source=None,
            run=f'{batch}_{group}', manifest='results/s54/motion_prior_stage0_verified/manifest.json',
            python=python)
        training_command, _ = command_for(args)
        launcher = [python, '-u', 'tools/train/run_s54_motion_prior.py', '--stage', 'stage2',
            '--stage2-short', '--group', group, '--task', args.task, '--seed', str(args.seed),
            '--profile', str(profile_path), '--run', args.run, '--experiment-name',
            f'{group}_{batch.removeprefix("s54_abc_short_")}_seed{args.seed}_{args.task}']
        if args.source:
            launcher += ['--source', args.source]
        result.append(dict(group=group, run=args.run, launcher=launcher, training_command=training_command))
    return result


def eval_bundle(run, report, profile):
    directory = ROOT/'logs'/run
    # A network connection is not required to prepare the durable Eval bundle.
    run_id = (directory/'swanlab_run_id').read_text().strip()
    name = (directory/'swanlab_experiment_name').read_text().strip()
    metrics = {'Eval/RealTransitions': report['eval_real_steps']}
    for duration in profile['evaluation']['durations_s']:
        rows = [r for r in report['rows'] if r['duration_s'] == duration]
        assert sorted(r['seed'] for r in rows) == sorted(profile['evaluation']['seeds'])
        prefix = f'Eval/Foot{duration:g}s/'
        metrics.update({prefix+'NormalizedReturn': sum(r['norm_return'] for r in rows)/len(rows),
                        prefix+'PhysicalTerminations': sum(r['n_failed'] for r in rows),
                        prefix+'Timeouts': sum(r['n_timeout'] for r in rows),
                        prefix+'Episodes': len(rows),
                        prefix+'MeanEpisodeSteps': sum(r['mean_episode_length'] for r in rows)/len(rows)})
        for key in ('foot_position_rmse_m', 'foot_velocity_rmse_m_s', 'foot_orientation_rmse_deg',
                    'actual_loaded_stance_slip_m_s', 'contact_state_accuracy',
                    'scheduled_touchdown_rmse_m', 'upper_velocity_rms_rad_s'):
            values = [r['metrics'].get(key) for r in rows]
            if all(isinstance(v, (float, int)) and math.isfinite(v) for v in values):
                metrics[prefix+key] = sum(values)/len(values)
        for key in ('root_pitch_max_abs_deg', 'root_roll_max_abs_deg', 'root_height_deviation_max_m'):
            metrics[prefix+'Worst/'+key] = max(r['metrics'][key] for r in rows)
        wm8 = [r['wm_paired'].get('horizons', {}).get('8') for r in rows]
        metrics[prefix+'WM8/AvailableEpisodes'] = sum(v is not None for v in wm8)
        if all(v is not None for v in wm8):
            for key in ('base_height_rmse_m', 'root_pitch_rmse_deg', 'feet_position_rmse_m', 'paired_reward_mae'):
                metrics[prefix+'WM8/Worst/'+key] = max(v[key] for v in wm8)
    return dict(target=dict(project=profile['swanlab']['project'], workspace=profile['swanlab']['workspace'],
                            run_id=run_id, name=name),
                training_args_path=str(directory/'args.json'),
                rows=[dict(step=profile['train']['iterations'], metrics=metrics)],
                protocol=profile['evaluation'])


def audit_training(directory, profile, teacher):
    """Check actual saved updates/samples and Decoder immutability, not exit=0 alone."""
    import torch
    torch.set_num_threads(1)
    final = torch.load(directory/'last.pt', map_location='cpu', weights_only=False)
    warm = torch.load(directory/'wm_calibrated.pt', map_location='cpu', weights_only=False)
    assert final['iteration'] == profile['train']['iterations']
    assert final['sim_steps'] == profile['comparison']['same_stage2_real_transitions_target']
    assert final['stage_stats']['actor_updates'] == profile['comparison']['same_actor_updates_target']
    assert not final['policy_contract']['decoder_trainable']
    assert warm['iteration'] == profile['train']['wm_only_iters']
    for name in ('wm', 'pi', 'critic'):
        assert all(torch.isfinite(v).all() for v in final[name].values() if v.is_floating_point())
    decoder = {k: v for k, v in final['pi'].items() if k.startswith('dec.')}
    assert decoder and all(torch.equal(v, warm['pi'][k]) for k, v in decoder.items())
    encoder = {k: v for k, v in warm['pi'].items() if k.startswith('enc.')}
    assert any(not torch.equal(v, final['pi'][k]) for k, v in encoder.items()), 'Foot Encoder did not update'
    if directory.name.endswith('_C'):
        source = torch.load(teacher, map_location='cpu', weights_only=False)
        assert all(torch.equal(v, source['pi'][k]) for k, v in decoder.items())
    def tensor_hash(values):
        value = hashlib.sha256()
        for k, v in sorted(values.items()):
            value.update(k.encode()); value.update(v.contiguous().numpy().tobytes())
        return value.hexdigest()
    result = dict(iteration=final['iteration'], real_transitions=final['sim_steps'],
        actor_updates=final['stage_stats']['actor_updates'], decoder_unchanged=True,
        decoder_hash=tensor_hash(decoder), initial_foot_encoder_hash=tensor_hash(encoder),
        foot_encoder_updated=True)
    audit_path = directory/'training_audit.json'
    if audit_path.exists():
        assert json.loads(audit_path.read_text()) == result, 'Saved training audit changed'
    else:
        write_json(audit_path, result)
    return result


def completed_training(run_dir, profile, entry):
    """Only skip a finished, equal-budget run; never silently resume partial RL."""
    status = json.loads((run_dir/'training_status.json').read_text())
    launch = json.loads((run_dir/'launch.json').read_text())
    if (status['status'] != 'completed' or status['iteration'] != profile['train']['iterations']
            or (run_dir/'train.exitcode').read_text().strip() != '0'
            or launch['profile'] != profile or launch['group'] != entry['group']):
        raise RuntimeError('Existing training is incomplete or its protocol differs; refusing to reuse')
    # Infrastructure fixes may differ; the actual learning implementation must
    # remain identical to the completed control run.
    snapshot = json.loads((run_dir/'source_snapshot/manifest.json').read_text())
    for path, expected in snapshot['files'].items():
        if path.startswith('supermimic/') and digest(ROOT/path) != expected:
            raise RuntimeError(f'Training implementation changed since completed run: {path}')


def completed_evaluation(run_dir, profile):
    report = json.loads((run_dir/'evaluation/summary.json').read_text())
    evaluation = profile['evaluation']
    expected = sorted((duration, seed) for duration in evaluation['durations_s'] for seed in evaluation['seeds'])
    observed = sorted((row['duration_s'], row['seed']) for row in report['rows'])
    if (observed != expected or report['reset_noise'] != evaluation['reset_noise']
            or Path(report['checkpoint']).resolve() != (run_dir/'last.pt').resolve()
            or any(row['controller'] != 'policy' or row['task'] != 'alternating' for row in report['rows'])):
        raise RuntimeError('Existing evaluation does not match the fixed comparison protocol')
    return report


def queue_evaluation_upload(run_dir, report, profile):
    """Persist before launching; cloud availability must never abort B/C."""
    bundle_path = run_dir/'evaluation/eval_bundle.json'
    payload = eval_bundle(run_dir.name, report, profile)
    if bundle_path.exists():
        assert json.loads(bundle_path.read_text()) == payload, 'Existing Eval bundle changed'
    else:
        write_json(bundle_path, payload)
    session = run_dir.name+'_eval_upload'
    verified = run_dir/'evaluation/upload_status.json'
    if verified.exists() and json.loads(verified.read_text()).get('status') == 'verified':
        return
    try:
        if subprocess.run(['tmux', 'has-session', '-t', '='+session], capture_output=True).returncode == 0:
            return
        python = os.environ.get('SWANLAB_PYTHON', str(ROOT/'.venv-swanlab/bin/python'))
        command = shlex.join([python, '-u', 'tools/ops/sync_eval_swanlab.py', '--bundle', str(bundle_path)])
        body = command+' >> '+shlex.quote(str(run_dir/'evaluation/swanlab_sync.log'))+' 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(ROOT),
                        *tmux_proxy_options(), body], check=True, timeout=10)
        print('Independent Eval upload queued: '+session, flush=True)
    except (OSError, subprocess.SubprocessError) as error:
        write_json(run_dir/'evaluation/queue_status.json', dict(status='launch_failed', error=str(error)))
        print('WARNING: Eval uploader launch failed; durable bundle retained for retry: '+str(error), flush=True)


def upload_states(entries):
    result = {}
    for entry in entries:
        path = ROOT/'logs'/entry['run']/'evaluation/upload_status.json'
        queue = path.with_name('queue_status.json')
        result[entry['group']] = (json.loads(path.read_text())['status'] if path.exists()
                                 else json.loads(queue.read_text())['status'] if queue.exists() else 'pending')
    return result


def run_worker(args, directory, profile, entries):
    from tools.visualization.policy_preview import gpu_slot
    state = dict(status='starting', task=profile['task'], training_seed=profile['seed'], completed_groups=[])
    stopping = False
    child = None

    def record(**values):
        state.update(values, updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        write_json(directory/'batch_status.json', state)

    def stop(*_):
        nonlocal stopping
        stopping = True
        (directory/'STOP').touch()
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def execute(command):
        nonlocal child
        print(shlex.join(command), flush=True)
        child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, start_new_session=True)
        record(active_pid=child.pid)
        for line in child.stdout:
            print(line, end='', flush=True)
        code = child.wait()
        child = None
        record(active_pid=None)
        if code:
            raise RuntimeError(f'Command exited {code}; see batch.log and per-run stdout.log')

    code = 1
    audits = []
    try:
        with gpu_slot():
            record(status='running', teacher_sha256=digest(directory/'teacher_v1.pt'))
            for entry in entries:
                if stopping or (directory/'STOP').exists():
                    raise InterruptedError('Safe batch stop requested; no next group will start')
                assert digest(directory/'teacher_v1.pt') == profile['teacher']['sha256']
                inputs = json.loads((directory/'shared_inputs.json').read_text())
                assert all(digest(ROOT/path) == expected for path, expected in inputs.items()), 'Shared task/goal config changed during batch'
                run_dir = ROOT/'logs'/entry['run']
                record(group=entry['group'], run=entry['run'], phase='training')
                if run_dir.exists():
                    if not args.reuse_completed:
                        raise FileExistsError('Existing run protected; use resume to reuse completed groups')
                    completed_training(run_dir, profile, entry)
                    print(f'Reusing completed {entry["group"]} training; no optimizer or simulator steps repeated.', flush=True)
                else:
                    execute(entry['launcher'] + ['--execute', '--stream-log', '--swanlab-mode', 'online'])
                    completed_training(run_dir, profile, entry)
                audits.append(audit_training(run_dir, profile, directory/'teacher_v1.pt'))
                assert all(a['initial_foot_encoder_hash'] == audits[0]['initial_foot_encoder_hash'] for a in audits)
                if entry['group'] == 'B':
                    assert audits[1]['decoder_hash'] == audits[0]['decoder_hash'], 'A/B random Decoder initialization differed'
                if stopping or (directory/'STOP').exists():
                    raise InterruptedError('Stopped after checkpoint save')
                record(phase='evaluation')
                evaluation = profile['evaluation']
                command = [args.python, '-u', 'tools/validation/evaluate_s54_motion_prior.py',
                    '--checkpoint', str(run_dir/'last.pt'), '--out', str(run_dir/'evaluation'),
                    '--durations', *map(str, evaluation['durations_s']), '--seeds', *map(str, evaluation['seeds']),
                    '--viz', 'none']
                if evaluation['reset_noise']:
                    command.append('--reset_noise')
                if (run_dir/'evaluation/summary.json').exists() and args.reuse_completed:
                    print(f'Reusing completed {entry["group"]} Newton evaluation.', flush=True)
                else:
                    execute(['timeout', '--signal=TERM', '--kill-after=15s', str(evaluation['process_budget_seconds'])+'s', *command])
                report = completed_evaluation(run_dir, profile)
                record(phase='queueing_evaluation_upload')
                queue_evaluation_upload(run_dir, report, profile)
                state['completed_groups'].append(entry['group'])
                record(phase='group_completed', uploads=upload_states(entries))
            assert digest(directory/'teacher_v1.pt') == profile['teacher']['sha256']
            record(status='completed', phase='done', uploads=upload_states(entries))
            code = 0
    except BaseException as error:
        record(status='stopped' if stopping or isinstance(error, InterruptedError) else 'failed', error=str(error))
        raise
    finally:
        (directory/'batch.exitcode').write_text(str(code)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['start', 'resume', 'dry-run', 'status', 'attach', 'stop', '_worker'], nargs='?', default='start')
    parser.add_argument('--batch')
    parser.add_argument('--profile', type=Path, default=Path('configs/motion_prior/stage2/abc_short_3090.json'))
    parser.add_argument('--python', default=default_python())
    parser.add_argument('--reuse-completed', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if not args.batch and args.mode in ('resume', 'status', 'attach', 'stop', '_worker'):
        parser.error('--batch is required for this mode')
    args.batch = args.batch or 's54_abc_short_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    if not re.fullmatch(r's54_[A-Za-z0-9_.-]+', args.batch):
        parser.error('Use a batch name starting s54_')
    directory = ROOT/'logs'/args.batch
    if args.mode == 'attach':
        os.execvp('tmux', ['tmux', 'attach', '-t', '='+args.batch])
    if args.mode == 'status':
        state = json.loads((directory/'batch_status.json').read_text())
        state['uploads'] = upload_states([dict(group=g, run=args.batch+'_'+g) for g in 'ABC'])
        print(json.dumps(state, indent=2))
        return
    if args.mode == 'stop':
        state = json.loads((directory/'batch_status.json').read_text())
        pid = int((directory/'batch.pid').read_text())
        command = subprocess.check_output(['ps', '-p', str(pid), '-o', 'args='], text=True)
        if 'run_s54_abc_short.py' not in command or args.batch not in command:
            raise RuntimeError('Stale PID; refusing to signal an unrelated process')
        (directory/'STOP').touch()
        os.kill(pid, signal.SIGTERM)
        print('Safe save/stop requested for', state.get('run', args.batch))
        return
    recovering = args.mode == 'resume'
    profile_path = directory/'profile.json' if args.mode in ('resume', '_worker') else args.profile.resolve()
    profile = json.loads(profile_path.read_text())
    teacher = directory/'teacher_v1.pt' if args.mode in ('resume', '_worker') else ROOT/profile['teacher']['checkpoint']
    if digest(teacher) != profile['teacher']['sha256']:
        raise ValueError('Teacher bytes do not match the fixed v1 SHA256')
    train = profile['train']
    assert 0 < train['wm_only_iters'] < train['iterations']
    assert train['warmup_steps'] >= max(train['hist'], train['wm_window'])
    expected = train['num_envs']*(train['warmup_steps']+int(train['iterations']/train['utd']))
    assert expected == profile['comparison']['same_stage2_real_transitions_target']
    assert train['iterations']-train['wm_only_iters'] == profile['comparison']['same_actor_updates_target']
    assert profile['swanlab']['project'] == 'Foothold_with_SuperMimic'
    entries = plans(profile_path, profile, args.batch, args.python, teacher)
    if args.mode == 'dry-run':
        for entry in entries:
            print(entry['group'], shlex.join(entry['training_command']))
        print(f'DRY RUN ONLY: {expected:,} real training transitions per group; no GPU simulation started.')
        return
    if args.mode == '_worker':
        with (directory/'batch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (directory/'batch.pid').write_text(str(os.getpid())+'\n')
            run_worker(args, directory, profile, entries)
        return
    if directory.exists() and not recovering:
        raise FileExistsError('Existing batch protected; choose a fresh --batch name')
    if recovering:
        # Preserve the original failure receipts and log. Incomplete training
        # is deliberately not resumed with an empty replay and extra samples.
        for entry in entries:
            run_dir = ROOT/'logs'/entry['run']
            if run_dir.exists():
                completed_training(run_dir, profile, entry)
                if (run_dir/'evaluation/summary.json').exists():
                    completed_evaluation(run_dir, profile)
        with (directory/'batch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from tools.visualization.policy_preview import gpu_slot
    with gpu_slot():
        free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()[0])
        if free < 8192:
            raise RuntimeError(f'Initial {train["num_envs"]}-env profile requires at least 8 GiB free; actual peak is to be measured')
    subprocess.run(['tmux', '-V'], check=True, stdout=subprocess.DEVNULL)
    session_exists = subprocess.run(['tmux', 'has-session', '-t', '='+args.batch], capture_output=True).returncode == 0
    if session_exists:
        dead = subprocess.check_output(['tmux', 'list-panes', '-s', '-t', '='+args.batch,
                                        '-F', '#{pane_dead}'], text=True).splitlines()
        if not recovering or not dead or any(v != '1' for v in dead):
            raise RuntimeError('Batch tmux session has a live pane; refusing duplicate start')
    if recovering:
        history = directory/'recovery'/str(time.time_ns())
        history.mkdir(parents=True)
        for name in ('batch_status.json', 'batch.exitcode', 'batch.pid'):
            if (directory/name).exists():
                shutil.copyfile(directory/name, history/name)
        (directory/'batch.exitcode').unlink(missing_ok=True)
        (directory/'STOP').unlink(missing_ok=True)
    else:
        directory.mkdir()
        shutil.copyfile(profile_path, directory/'profile.json')
        shutil.copyfile(teacher, directory/'teacher_v1.pt')
        (directory/'teacher_v1.pt').chmod(0o444)
        assert digest(directory/'teacher_v1.pt') == profile['teacher']['sha256']
        shared = [f'configs/motion_prior/stage2/stage2_{profile["task"]}.json',
                  'configs/motion_prior/stage2/conservative_footsteps.json']
        write_json(directory/'shared_inputs.json', {path: digest(ROOT/path) for path in shared})
    write_json(directory/'batch_status.json', dict(status='queued', task=profile['task'], groups=['A', 'B', 'C'],
                                                  recovery=recovering))
    job = shlex.join(['env', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=4', 'PYTHONDONTWRITEBYTECODE=1',
        'SWANLAB_PYTHON='+os.environ.get('SWANLAB_PYTHON', str(ROOT/'.venv-swanlab/bin/python')),
        args.python, '-u', 'tools/train/run_s54_abc_short.py', '_worker', '--batch', args.batch, '--python', args.python,
        *(['--reuse-completed'] if recovering else [])])
    body = 'set -o pipefail\n'+job+' 2>&1 | tee -a '+shlex.quote(str(directory/'batch.log'))
    tmux_command = (['tmux', 'new-window', '-d', '-P', '-F', '#{window_id}', '-t', '='+args.batch, '-n', 'recovered']
                    if session_exists else ['tmux', 'new-session', '-d', '-P', '-F', '#{window_id}', '-s', args.batch])
    window = subprocess.check_output([*tmux_command, '-c', str(ROOT), *tmux_proxy_options(),
                                     'bash -lc '+shlex.quote(body)], text=True).strip()
    subprocess.run(['tmux', 'set-option', '-w', '-t', window, 'remain-on-exit', 'on'], check=True)
    subprocess.run(['tmux', 'select-window', '-t', window], check=True)
    print(f'Started {args.batch}; serial A -> B -> C, {train["iterations"]} iterations per group, with independent eval and SwanLab upload.')
    if profile['wall_seconds'] is None:
        print('Training wall-clock limit: disabled; stop at the iteration target or an explicit safe-stop request.')
    print('Attach: tmux attach -t '+args.batch)
    print('Log: tail -F '+str(directory/'batch.log'))


if __name__ == '__main__':
    main()
