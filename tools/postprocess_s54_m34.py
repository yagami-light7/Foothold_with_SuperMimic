"""Optional same-source raw PD comparisons and baseline videos after a run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_dir', required=True, type=Path)
    p.add_argument('--wait_done', action='store_true', help='Wait up to one hour for the tmux pipeline completion marker')
    a = p.parse_args()
    if a.wait_done:
        deadline = time.monotonic()+3600
        while not (a.run_dir/'train.exitcode').exists():
            if time.monotonic() > deadline: raise TimeoutError('Pipeline completion marker did not arrive')
            time.sleep(5)
    config = json.loads((a.run_dir/'pipeline_config.json').read_text())
    clean = json.loads((a.run_dir/'pd/results.json').read_text())
    for mode, clip in [('straight', 'straight_flat_20'), ('turn', 'turn_flat_11'), ('spin', 'spin_flat_02')]:
        out = a.run_dir/('pd_raw_'+clip)
        if not out.exists():
            subprocess.run([sys.executable, '-u', 'tools/m34_rollout.py', '--dataset', config['dataset'],
                            '--out', str(out), '--raw_comparison', clip], check=False)
        raw = json.loads((out/'results.json').read_text())
        if not raw['rows']: continue
        clean_row = next(r for r in clean['rows'] if r['id'] == clip)
        video = a.run_dir/f'pd_{mode}.mp4'
        if not video.exists():
            subprocess.run([sys.executable, 'tools/render_s54_motion_prior.py',
                '--records', clean_row['record'], raw['rows'][0]['record'],
                '--labels', f'TCRS-Clean {clip}', f'GMR-Raw {clip}',
                '--out', str(video), '--seconds', '3'], check=True)
    subprocess.run([sys.executable, 'tools/report_s54_m34_pd.py', '--run_dir', str(a.run_dir)], check=True)
    evaluations = {}
    for split in ('train', 'test'):
        nominal = a.run_dir/f'eval_{split}/results.json'
        perturbed = a.run_dir/f'eval_{split}_perturbed/results.json'
        if not nominal.exists() or not perturbed.exists(): break
        result = json.loads(nominal.read_text()); noisy = json.loads(perturbed.read_text())
        if result['status'] != 'completed' or noisy['status'] != 'completed': break
        if result['checkpoint_sha256'] != noisy['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed across seeds')
        result['rows'] += noisy['rows']; evaluations[split] = result
    if len(evaluations) == 2:
        from run_s54_m34 import report
        if evaluations['train']['checkpoint_sha256'] != evaluations['test']['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed across train/test evaluation')
        report(a.run_dir, config, a.run_dir/'last.pt', evaluations)


if __name__ == '__main__': main()
