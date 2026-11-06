"""Durable, CPU-only training/Eval upload queue; never blocks GPU experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.ops.upload_eval_swanlab import validate, write_json


def training_synced(run_dir):
    status = run_dir/'swanlab_sync_status.json'
    final = run_dir/'final_sync.exitcode'
    return (status.exists() and json.loads(status.read_text()).get('status') == 'finished'
            and final.exists() and final.read_text().strip() == '0')


def execute(command, timeout=180):
    """Bound one network attempt, including its children; leave durable data."""
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
        if code:
            raise RuntimeError(f'Upload subprocess exited {code}')
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def sync_once(run_dir, bundle_path, bundle, record):
    """Caller owns uploader.lock, shared with the live training uploader."""
    target = bundle['target']
    if not training_synced(run_dir):
        record('syncing_training')
        execute([sys.executable, '-u', 'tools/ops/sync_swanlab.py', '--run_dir', str(run_dir),
                 '--project', target['project'], '--workspace', target['workspace'],
                 '--experiment_name', target['name'], '--mode', 'online', '--follow', '--completion_marker'])
        (run_dir/'upload.exitcode').write_text('0\n')
        # The cursor can advance before a network failure. Drain the SDK's
        # durable cache before permitting Eval to write to this same run ID.
        for cache in sorted((run_dir/'swanlab').glob('run-*')):
            if cache.is_dir():
                if not any(cache.glob('run-*.swanlab')):
                    print(f'Skipping uninitialized SDK directory (no record file): {cache.name}', flush=True)
                    continue
                execute([sys.executable, '-m', 'swanlab', 'sync', str(cache),
                         '--project', target['project'], '--workspace', target['workspace'],
                         '--id', target['run_id']])
        (run_dir/'final_sync.exitcode').write_text('0\n')
    record('syncing_evaluation')
    execute([sys.executable, '-u', 'tools/ops/upload_eval_swanlab.py',
             '--bundle', str(bundle_path), '--execute'])
    verified = json.loads((bundle_path.parent/'upload_status.json').read_text())
    if verified['status'] != 'verified':
        raise RuntimeError('Eval upload has no verified cloud read-back')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    bundle_path = args.bundle.resolve()
    bundle = json.loads(bundle_path.read_text())
    validate(bundle)
    run_dir = Path(bundle['training_args_path']).resolve().parent
    if (run_dir/'swanlab_run_id').read_text().strip() != bundle['target']['run_id']:
        raise ValueError('Bundle must target the original training run ID')
    if (run_dir/'train.exitcode').read_text().strip() != '0':
        raise ValueError('This Eval queue requires completed training')
    state = dict(attempt=0, run_id=bundle['target']['run_id'])

    def record(status, **values):
        state.update(status=status, updated_at=datetime.now(timezone.utc).isoformat(), **values)
        write_json(bundle_path.parent/'queue_status.json', state)
        print(f'Eval upload queue: {status}; attempt={state["attempt"]}', flush=True)

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (run_dir/'evaluation_queue.lock').open('a') as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        delay = 5
        while True:
            try:
                with (run_dir/'uploader.lock').open('a') as upload_lock:
                    try:
                        fcntl.flock(upload_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        record('waiting_for_training_uploader')
                    else:
                        state['attempt'] += 1
                        sync_once(run_dir, bundle_path, bundle, record)
                        record('verified', error=None)
                        return
            except KeyboardInterrupt:
                record('stopped')
                raise
            except Exception as error:
                record('retry_pending', error=str(error), retry_seconds=delay)
                print(f'Upload failed: {type(error).__name__}: {error}; local data retained', flush=True)
            time.sleep(delay)
            delay = min(300, delay*2)


if __name__ == '__main__':
    main()
