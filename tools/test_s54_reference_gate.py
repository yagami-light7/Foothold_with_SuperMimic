"""Integration test: an unrepairable candidate must never become trainable."""
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from prepare_s54_motion import prepare


def main():
    source = REPO / 'library/motion_reference/kuavo_s54/retarget_data/reference_grounded'
    clip, stem = 'dance1_subject1', 'known_collision'
    cache = REPO / '.cache'
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='s54_collision_gate_', dir=cache) as folder:
        root = Path(folder)
        src, out = root / 'source', root / 'accepted'
        for directory in ('raw', 'baseline', 'reference', f'diagnostics/{stem}'):
            (src / directory).mkdir(parents=True)
        x = np.loadtxt(source / f'{clip}.csv', delimiter=',')[942:945]
        raw = src / 'raw' / f'{stem}.csv'
        np.savetxt(raw, x[:, :34], delimiter=',', fmt='%.17g')
        prepare(raw, src / f'{stem}.csv', 3, src / 'raw')
        for directory in ('baseline', 'reference'):
            with (source / directory / f'{clip}_30fps.pkl').open('rb') as f:
                motion = pickle.load(f)
            for key in ('root_pos', 'root_rot', 'dof_pos'):
                motion[key] = motion[key][942:945]
            with (src / directory / f'{stem}_30fps.pkl').open('wb') as f:
                pickle.dump(motion, f)
        with np.load(source / 'reference' / f'{clip}_human.npz') as h:
            np.savez_compressed(src / 'reference' / f'{stem}_human.npz',
                                **{k: h[k] if k == 'names' else h[k][942:945] for k in h.files})
        upper = json.loads((source / 'diagnostics' / clip / 'upper_body_metrics.json').read_text())
        (src / 'diagnostics' / stem / 'upper_body_metrics.json').write_text(json.dumps(upper))
        cfg = json.loads((REPO / 'configs/s54_self_collision.json').read_text())
        # Deliberately disable optimization to exercise a real, known failed
        # collision certificate through the complete quarantine path.
        cfg.update(frame_iterations=0, trajectory_passes=0)
        config = root / 'reject_config.json'
        config.write_text(json.dumps(cfg))
        result = subprocess.run([sys.executable, str(REPO / 'tools/repair_s54_collisions.py'),
                                 '--input_dir', str(src), '--out_dir', str(out), '--all',
                                 '--collision_config', str(config), '--continue_on_error'],
                                capture_output=True, text=True)
        assert result.returncode != 0, result.stdout
        assert not list(out.glob('*.csv')), 'Rejected clip became visible to training'
        rejected = list((out / 'rejected').glob('*/diagnostics/*/self_collision_metrics.json'))
        assert len(rejected) == 1, result.stdout + result.stderr
        report = json.loads(rejected[0].read_text())
        assert report['certificate']['status'] == 'FAIL'
        assert report['certificate']['first_failure']['distance_m'] < 0
        assert (src / f'{stem}.csv').exists(), 'Source data was changed'
        evidence = dict(status='PASS', failing_candidate_quarantined=True,
                        accepted_csv_count=0, source_preserved=True,
                        first_failure=report['certificate']['first_failure'])
        path = REPO / 'results/s54/reference_gate_regression.json'
        path.write_text(json.dumps(evidence, indent=2))
        print(json.dumps(evidence, indent=2), flush=True)


if __name__ == '__main__':
    main()
