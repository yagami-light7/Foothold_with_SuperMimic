"""Archive core local training/evaluation data in SwanLab, without any media.

This uploads existing results only. It neither resumes training nor selects a
checkpoint using evaluation performance. Nominal and perturbed protocols stay
on distinct curves, and each reference has equal weight in reported means.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import uuid

from sync_swanlab import (CORE_CONFIG_KEYS, CORE_METRICS, aggregate, evaluation_metrics,
                          load_swanlab, read_latest_run)


def collect_snapshot(run_dir, eval_dir):
    events = defaultdict(dict)
    _, rows = read_latest_run(run_dir / 'train_log.csv')
    if not rows:
        raise ValueError('Training CSV is empty')
    for row in rows:
        step = int(row['iteration'])
        for key, label in CORE_METRICS.items():
            value = float(row[key])
            if math.isfinite(value):
                events[step][label] = value
    evaluations = []
    for path in sorted((run_dir / 'evals').glob('iter_*/summary.json')):
        summary = json.loads(path.read_text())
        if summary['status'] != 'completed':
            continue
        step = int(path.parent.name.removeprefix('iter_'))
        events[step].update(evaluation_metrics(summary))
        evaluations.append(dict(iteration=step, checkpoint_sha256=summary['checkpoint_sha256'],
                                seeds=summary['seeds'], source=str(path)))
    request = json.loads((eval_dir / 'request.json').read_text())
    latest = json.loads((eval_dir / 'evaluation/summary.json').read_text())
    step = request['iteration']
    events[step].update(evaluation_metrics(latest))
    evaluations.append(dict(iteration=step, checkpoint_sha256=latest['checkpoint_sha256'],
                            seeds=latest['seeds'], source=str(eval_dir / 'evaluation/summary.json')))
    return dict(events=dict(sorted(events.items())), training_rows=len(rows),
                last_training_iteration=max(int(r['iteration']) for r in rows),
                latest_iteration=step, evaluations=evaluations,
                checkpoint_sha256=latest['checkpoint_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--eval-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New upload receipt/log directory')
    parser.add_argument('--project', default='SuperMimic-S54')
    parser.add_argument('--workspace')
    parser.add_argument('--name')
    parser.add_argument('--mode', choices=('online', 'offline'), default='online')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume-upload', action='store_true', help='Continue a failed upload using its saved run ID')
    args = parser.parse_args()
    snapshot = collect_snapshot(args.run_dir, args.eval_dir)
    if args.dry_run:
        print(json.dumps({key: value for key, value in snapshot.items() if key != 'events'}, indent=2))
        print('Steps:', list(snapshot['events']))
        return
    previous = None
    if args.resume_upload:
        previous = json.loads((args.out / 'swanlab_upload_status.json').read_text())
        if (previous['status'] not in ('failed', 'uploading', 'uploaded') or previous['project'] != args.project
                or previous['checkpoint_sha256'] != snapshot['checkpoint_sha256']):
            raise ValueError('Resume requires the same failed upload, project and checkpoint')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        (args.out / f'swanlab_upload_status.failed_{stamp}.json').write_text(json.dumps(previous, indent=2) + '\n')
    else:
        args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'upload_snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    config = json.loads((args.run_dir / 'args.json').read_text())
    config = {key: value for key, value in config.items() if key in CORE_CONFIG_KEYS and value is not None}
    exit_path = args.run_dir / 'train.exitcode'
    training_exitcode = int(exit_path.read_text()) if exit_path.exists() else None
    config.update(training_interrupted=training_exitcode not in (None, 0),
                  last_logged_iteration=snapshot['last_training_iteration'],
                  latest_checkpoint_iteration=snapshot['latest_iteration'],
                  latest_eval_seeds=[0], latest_eval_reset_noise=False,
                  reward_normalization='sum_reward / (full_reference_steps * max_step_reward)',
                  backward_evaluation='cross-motion-mode extrapolation')
    receipt = dict(status='initializing', project=args.project, name=args.name or args.run_dir.name + '_local_archive',
                   run_id=uuid.uuid4().hex[:16], training_exitcode=training_exitcode,
                   training_rows=snapshot['training_rows'], evaluations=snapshot['evaluations'],
                   upload_scope='core_training_and_evaluation_data_only',
                   media_enqueued=False, checkpoint_sha256=snapshot['checkpoint_sha256'])
    resume_after = -1
    if previous:
        receipt.update(run_id=previous['run_id'], name=previous['name'])
        resume_after = previous.get('last_enqueued_step', -1)
        receipt['last_enqueued_step'] = resume_after

    def save_status(state, **updates):
        receipt.update(status=state, updated_at=datetime.now(timezone.utc).isoformat(), **updates)
        (args.out / 'swanlab_upload_status.json').write_text(json.dumps(receipt, indent=2) + '\n')

    save_status('initializing')
    sdk = load_swanlab(args.mode, authenticate=True, core_only=True)
    run = None
    try:
        kwargs = dict(project=args.project, name=receipt['name'], id=receipt['run_id'],
                      resume='allow', config=config, mode=args.mode, on_online_error='raise',
                      log_dir=str(args.out / 'swanlab'), public=False,
                      job_type='local-results-archive',
                      description='Archive of interrupted local training and frozen Newton evaluations. '
                                  'Latest checkpoint chosen by iteration, not test performance. '
                                  'Archive completion does not mean training converged.')
        if args.workspace:
            kwargs['workspace'] = args.workspace
        run = sdk.init(**kwargs)
        if args.mode == 'online':
            save_status('uploading', url=run.url, path=run.path)
        for step, values in snapshot['events'].items():
            if step <= resume_after:
                continue
            sdk.log(values, step=step)
            save_status('uploading', last_enqueued_step=step)
        saved_files = []
        for name in ('normalized_return_summary.json',):
            saved_files.extend(run.save(str((args.eval_dir / name).resolve()),
                                        base_path=args.eval_dir.resolve(), policy='now'))
        core_export = args.out / 'core_training_and_eval_metrics.json'
        core_export.write_text(json.dumps(snapshot['events'], indent=2) + '\n')
        saved_files.extend(run.save(str(core_export.resolve()), base_path=args.out.resolve(), policy='now'))
        sdk.finish(state='success')
        run = None
        save_status('uploaded', last_enqueued_step=snapshot['last_training_iteration'],
                    media_enqueued=False, saved_files=saved_files)
        print(json.dumps(receipt, indent=2), flush=True)
    except BaseException as exc:
        save_status('failed', error=repr(exc))
        if run is not None:
            sdk.finish(state='crashed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
