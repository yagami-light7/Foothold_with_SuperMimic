"""Independent, synchronous Newton evaluation of immutable M3.4 checkpoints.

The trainer waits for child processes, retaining its replay, RNG, normalization
and optimizer states. Children never return model weights or sampler feedback.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def evaluate_checkpoint(checkpoint, dataset, output, seeds=(0, 1, 2)):
    """Evaluate train and test without exposing test feedback to optimization.

    Retry incomplete attempts in fresh directories; completed results are reused
    only for the exact same checkpoint hash, dataset path and seed protocol.
    """
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    dataset = str(Path(dataset).resolve())
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Evaluation requires distinct nonnegative integer seeds')
    output.mkdir(parents=True, exist_ok=True)
    contract = dict(checkpoint_sha256=_sha256(checkpoint), dataset=dataset, seeds=seeds)
    summary = output/'summary.json'
    if summary.exists():
        previous = json.loads(summary.read_text())
        if any(previous.get(k) != v for k, v in contract.items()):
            raise ValueError('Refusing to reuse evaluation from another checkpoint/protocol')
        return previous['splits']
    attempt = output/f'attempt_{len(list(output.glob("attempt_*")))+1:03d}'
    attempt.mkdir()
    state = dict(status='running', checkpoint=str(checkpoint), attempt=str(attempt), **contract)
    _write(output/'status.json', state)
    results = {}
    base = [sys.executable, '-u', 'tools/m34_rollout.py', '--dataset', dataset,
            '--checkpoint', str(checkpoint)]
    try:
        for split in ('train', 'test'):
            combined = dict(status='completed', rows=[], checkpoint_sha256=contract['checkpoint_sha256'])
            protocols = []
            if 0 in seeds: protocols.append(('nominal', [0], False))
            perturbed = [s for s in seeds if s != 0]
            if perturbed: protocols.append(('perturbed', perturbed, True))
            for name, selected, noisy in protocols:
                target = attempt/f'{split}_{name}'
                command = [*base, '--split', split, '--out', str(target), '--seeds', *map(str, selected)]
                if noisy: command += ['--reset_noise']
                print('EVAL:', ' '.join(command), flush=True)
                # Keep detailed simulator output local, separate from train CSV.
                with (attempt/f'{split}_{name}.log').open('w') as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                result = json.loads((target/'results.json').read_text())
                if result['status'] != 'completed' or result['checkpoint_sha256'] != contract['checkpoint_sha256']:
                    raise RuntimeError('Incomplete evaluation or changed checkpoint')
                combined['rows'].extend(result['rows'])
            results[split] = combined
        if _sha256(checkpoint) != contract['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed during frozen evaluation')
        _write(summary, dict(status='completed', splits=results, **contract))
        state['status'] = 'completed'
        return results
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        _write(output/'status.json', state)
