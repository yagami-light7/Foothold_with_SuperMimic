"""Upload one S54 run's CSV metrics and configuration from the host to SwanLab."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
from statistics import mean
import subprocess
import time
import uuid


METRICS = {
    'wm_loss': 'WorldModel/Loss',
    'wm_err8': 'WorldModel/Error8',
    'pi_loss': 'Policy/Loss',
    'v_loss': 'Critic/Loss',
    'imag_rew': 'Reward/Imagined',
    'sim_rew': 'Reward/Simulator',
    'ep_ret_norm': 'Reward/EpisodeReturn',
    'ep_len': 'Episode/Length',
    'term_frac': 'Episode/TerminationFraction',
    'imag_survival': 'Policy/Survival',
    'pi_gradnorm': 'Policy/GradNorm',
    'it_per_s': 'Performance/IterationsPerSecond',
    'wall_h': 'Performance/WallHours',
    'sim_steps': 'Performance/SimulatedSteps',
    'lr': 'Optimization/LearningRate',
}

OPTIONAL_METRICS = {
    'head_rms_deg': 'Head/RMSErrorDeg',
    'head_penalty_norm': 'Head/NormalizedPenalty',
}

# Core-only uploads omit paths, clip inventories, unrelated task dictionaries,
# head diagnostics and arbitrary curriculum/gradient telemetry. Motion reward
# weights and observation mode identify this experiment and are retained.
CORE_CONFIG_KEYS = {
    'goal_mode', 'num_envs', 'iterations', 'buffer_steps', 'warmup_steps', 'utd',
    'wm_window', 'wm_batch', 'wm_batches', 'wm_only_iters', 'pi_batch', 'horizon',
    'hist', 'goal_frames', 'goal_dt', 'lr', 'actor_lr', 'critic_lr', 'gamma', 'lam',
    'action_noise', 'm34_straight_iters', 'm34_eval_every', 'amp', 'seed',
    'motion_obs_mode', 'obs_dim', 'motion_reward_config',
}
CORE_METRICS = {key: METRICS[key] for key in (
    'wm_loss', 'pi_loss', 'v_loss', 'sim_rew', 'ep_ret_norm', 'ep_len',
    'term_frac', 'pi_gradnorm', 'it_per_s', 'sim_steps',
)}


def read_latest_run(csv_path):
    """Ignore a partially written last line and any older appended CSV runs."""
    if not csv_path.exists():
        return None, []
    text = csv_path.read_text()
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith('\n'):
        lines.pop()
    headers = [i for i, line in enumerate(lines)
               if line.startswith('iteration,wall_h,sim_steps,')]
    if not headers:
        return None, []
    start = headers[-1]
    return start, list(csv.DictReader(lines[start:]))


def aggregate(rows):
    """Equal reference weighting, averaging seeds within each reference first."""
    if not rows:
        raise ValueError('No evaluation rows')
    clips = defaultdict(list)
    for row in rows:
        clips[row['id']].append(row)
    metrics = {
        'NormalizedReturn': 'normalized_return', 'SurvivalRate': 'survived',
        'RootPositionRMSE_m': 'root_position_rmse_m',
        'RootOrientationRMSE_rad': 'root_orientation_rmse_rad',
        'JointRMSE_rad': 'joint_rmse_rad', 'FootPositionRMSE_m': 'foot_position_rmse_m',
        'EpisodeSeconds': 'episode_s',
    }
    result = {label: mean(mean(float(r[key]) for r in group) for group in clips.values())
              for label, key in metrics.items()}
    result['ReferenceStepWeightedReturn'] = (
        sum(mean(r['normalized_return'] for r in group) * group[0]['reference_steps']
            for group in clips.values())
        / sum(group[0]['reference_steps'] for group in clips.values()))
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('Nonfinite evaluation metrics')
    return result


def evaluation_metrics(summary):
    """Scalar metrics only; train/test and nominal/perturbed protocols stay separate."""
    if summary['status'] != 'completed':
        raise ValueError('Refusing to upload an incomplete evaluation as completed')
    values = {}
    for split, data in summary['splits'].items():
        if split not in ('train', 'test'):
            raise ValueError('Unexpected evaluation split')
        if data['status'] != 'completed' or data['checkpoint_sha256'] != summary['checkpoint_sha256']:
            raise ValueError('Incomplete or inconsistent frozen evaluation')
        groups = defaultdict(list)
        for row in data['rows']:
            if row['split'] != split:
                raise ValueError('Mixed train/test rows')
            protocol = 'Perturbed' if row['reset_noise'] else 'Nominal'
            groups[protocol].append(row)
        for protocol, rows in groups.items():
            prefix = f'Eval/{split.title()}/{protocol}'
            values.update({f'{prefix}/{key}': value for key, value in aggregate(rows).items()})
            field = 'id' if split == 'test' else 'mode'
            for name in sorted({r[field] for r in rows}):
                selected = aggregate([r for r in rows if r[field] == name])
                selected_keys = ('NormalizedReturn', 'SurvivalRate', 'RootPositionRMSE_m')
                values.update({f'{prefix}/{name}/{key}': selected[key] for key in selected_keys})
    return values


def training_exit_code(run_dir, container=None, session=None):
    marker = run_dir / 'train.exitcode'
    if marker.exists():
        return int(marker.read_text().strip())
    if container:
        result = subprocess.run(
            ['docker', 'exec', container, 'tmux', 'has-session', '-t', f'={session}'],
            capture_output=True, timeout=15,
        )
        if result.returncode:
            # Also covers a stopped container or forcibly killed tmux session.
            return 1
    return None


def load_swanlab(mode, authenticate=False, core_only=True):
    # The launcher passes --project explicitly. Recent SDKs reserve this env
    # name for a structured Settings.Project value, not a plain project name.
    # Remove it in this uploader process before importing/initializing the SDK.
    os.environ.pop('SWANLAB_PROJECT', None)
    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError('Install the host SDK: .venv-swanlab/bin/python -m pip install swanlab') from exc
    probe = dict(monitor=False)
    if core_only:
        probe.update(hardware=False, runtime=False, requirements=False, conda=False, git=False, swanlab=False)
    settings = swanlab.Settings(interactive=False, probe=swanlab.Settings.Probe(**probe))
    swanlab.merge_settings(settings)
    if mode == 'online' and authenticate:
        # Uses SWANLAB_API_KEY or the cached host login; never prompts in a job.
        swanlab.login()
    return swanlab


def sync_run(args, swanlab):
    run_dir = args.run_dir.resolve()
    (run_dir / 'swanlab_sync.pid').write_text(f'{os.getpid()}\n')
    config_path = run_dir / 'args.json'
    csv_path = run_dir / 'train_log.csv'
    deadline = time.monotonic() + args.startup_timeout
    while True:
        try:
            config = json.loads(config_path.read_text())
            break
        except (FileNotFoundError, json.JSONDecodeError):
            if training_exit_code(run_dir, args.container, args.session) is not None:
                raise RuntimeError('Training exited before writing args.json')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Timed out waiting for {config_path}')
            time.sleep(args.poll_seconds)

    last_step, active_header = -1, None
    uploaded_evaluations = {}
    previous_status = run_dir/'swanlab_sync_status.json'
    if previous_status.exists():
        previous=json.loads(previous_status.read_text())
        if previous.get('mode')==args.mode and previous.get('project')==args.project:
            last_step=previous.get('last_step',-1)
            uploaded_evaluations=previous.get('evaluations', {})
    finish_state = 'crashed'
    result = {'mode': args.mode, 'project': args.project, 'experiment_name': args.experiment_name or run_dir.name,
              'clips': config.get('clips'), 'last_step': last_step,
              'evaluations': uploaded_evaluations}
    id_path=run_dir/'swanlab_run_id'
    if not id_path.exists():
        id_path.write_text(os.environ.get('SWANLAB_RUN_ID') or uuid.uuid4().hex[:16])
    run_id = id_path.read_text().strip()
    resume_mode = os.environ.get('SWANLAB_RESUME', 'allow' if run_id else None)
    if run_id:
        result.update(run_id=run_id, resume=resume_mode)

    def status(state, error=None):
        result.update(status=state, last_step=last_step,
                      updated_at=datetime.now(timezone.utc).isoformat())
        if error:
            result['error'] = error
        path = run_dir / 'swanlab_sync_status.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(path)

    initialized = False
    core_only = getattr(args, 'core_only', True)
    try:
        uploaded_config = {k: v for k, v in config.items() if k in CORE_CONFIG_KEYS and v is not None} if core_only else config
        kwargs = dict(project=args.project, name=result['experiment_name'], config=uploaded_config,
                      log_dir=str(run_dir / 'swanlab'), mode=args.mode)
        if args.workspace:
            kwargs['workspace'] = args.workspace
        if run_id:
            kwargs.update(id=run_id, resume=resume_mode)
        swanlab.init(**kwargs)
        initialized = True
        status('running')
        while True:
            # Read completion first, then drain the final CSV after the writer exited.
            code = training_exit_code(run_dir, args.container, args.session)
            header, rows = read_latest_run(csv_path)
            if active_header is None:
                active_header = header
            elif header is not None and header != active_header:
                raise RuntimeError('A different CSV run was appended; refusing to mix experiments')
            extra={}
            extra_path=run_dir/'curriculum_metrics.jsonl'
            if not core_only and extra_path.exists():
                for line in extra_path.read_text().splitlines(keepends=True):
                    if not line.endswith('\n'): continue
                    event=json.loads(line)
                    extra.setdefault(event['iteration'],{}).update(event['values'])
            max_step=max((int(r['iteration']) for r in rows if r.get('iteration','').isdigit()),default=-1)
            for row in rows:
                try:
                    step = int(row['iteration'])
                    if not core_only and extra_path.exists() and code is None and step == max_step: continue
                    selected_metrics = CORE_METRICS if core_only else METRICS
                    values = {name: float(row[key]) for key, name in selected_metrics.items()}
                    if not core_only:
                        values.update({name: float(row[key]) for key, name in OPTIONAL_METRICS.items()
                                       if row.get(key) not in (None, '')})
                except (KeyError, TypeError, ValueError):
                    continue
                if step <= last_step or not all(math.isfinite(v) for v in values.values()):
                    continue
                values.update(extra.get(step,{}))
                swanlab.log(values, step=step)
                last_step = step
                print(f'Recorded iteration {step} ({args.mode})', flush=True)
                status('running')
            # Evaluations finish after the training row at the same iteration.
            # SwanLab tracks steps per metric, so these distinct Eval/* series
            # can be logged later without delaying the live training curves.
            for path in sorted((run_dir/'evals').glob('*/summary.json')):
                name = path.parent.name
                if not (name.startswith('iter_') or name.startswith('final_')):
                    continue
                evaluation_step = int(name.split('_', 1)[1])
                summary = json.loads(path.read_text())
                if summary.get('status') != 'completed':
                    continue
                key = str(path.relative_to(run_dir))
                digest = summary['checkpoint_sha256']
                if key in uploaded_evaluations:
                    if uploaded_evaluations[key] != digest:
                        raise ValueError('A previously uploaded frozen evaluation changed checkpoint')
                    continue
                swanlab.log(evaluation_metrics(summary), step=evaluation_step)
                uploaded_evaluations[key] = digest
                print(f'Recorded frozen evaluation {evaluation_step} ({args.mode})', flush=True)
                status('running')
            if not args.follow or code is not None:
                if code not in (None, 0):
                    raise RuntimeError(f'Training exited with status {code}; final available metrics were recorded')
                finish_state = 'success'
                break
            # Manual follow without a launcher completion marker can use the target.
            if not args.container and not args.completion_marker and last_step >= int(config.get('iterations', 0)) > 0:
                finish_state = 'success'
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        finish_state = 'aborted'
        status('aborted')
        raise
    except Exception as exc:
        status('failed', str(exc))
        raise
    finally:
        if initialized:
            try:
                swanlab.finish(state=finish_state)
            except Exception as exc:
                finish_state = 'crashed'
                status('failed', f'SwanLab finish failed: {exc}')
                raise
        if finish_state == 'success':
            status('finished')
        elif finish_state == 'crashed' and result.get('status')=='training_failed':
            status('finished_training_failed')
    print(f'SwanLab sync finished at iteration {last_step}', flush=True)


def stop_sync(*_):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path)
    parser.add_argument('--project', default='SuperMimic')
    parser.add_argument('--workspace')
    parser.add_argument('--experiment_name')
    parser.add_argument('--mode', choices=['online', 'offline'], default='online')
    parser.add_argument('--core_only', action=argparse.BooleanOptionalAction, default=True,
                        help='Upload only core hyperparameters/metrics (default); --no-core_only opts into detailed telemetry')
    parser.add_argument('--follow', action='store_true', help='Follow new metrics instead of uploading one snapshot')
    parser.add_argument('--completion_marker', action='store_true', help='Wait for train.exitcode, even after the last metric')
    parser.add_argument('--container', help='Also detect when a Docker training session disappears')
    parser.add_argument('--session')
    parser.add_argument('--poll_seconds', type=float, default=5)
    parser.add_argument('--startup_timeout', type=float, default=300)
    parser.add_argument('--check', action='store_true', help='Check SDK/login without creating a cloud experiment')
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.startup_timeout <= 0:
        parser.error('Polling interval and startup timeout must be positive')
    if bool(args.container) != bool(args.session):
        parser.error('--container and --session must be used together')
    if not args.check and not args.run_dir:
        parser.error('--run_dir is required')
    signal.signal(signal.SIGTERM, stop_sync)
    swanlab = load_swanlab(args.mode, authenticate=args.check, core_only=args.core_only)
    if args.check:
        print(f'PASS: SwanLab SDK ready ({args.mode})' + ('; cloud authentication checked' if args.mode == 'online' else ''))
        return
    sync_run(args, swanlab)


if __name__ == '__main__':
    main()

