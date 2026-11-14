"""Repair existing S54 references into a separate, certified training directory."""
import argparse
from datetime import datetime, timezone
import fnmatch
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.s54 import REPO
from s54_reference_publish import finish_candidate, accepted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input_dir', type=Path, default=REPO / 'library/motion_reference/kuavo_s54/retarget_data/reference_grounded')
    p.add_argument('--out_dir', type=Path, default=REPO / 'library/motion_reference/kuavo_s54/retarget_data/reference_safe')
    p.add_argument('--clips', nargs='+')
    p.add_argument('--all', action='store_true')
    p.add_argument('--skip_existing', action='store_true')
    p.add_argument('--continue_on_error', action='store_true')
    p.add_argument('--resume_candidates', action='store_true', help='Resume an interrupted private candidate whose source CSV still matches')
    p.add_argument('--collision_config', type=Path, default=REPO / 'configs/s54_self_collision.json')
    a = p.parse_args()
    src, out = a.input_dir.resolve(), a.out_dir.resolve()
    if not out.is_relative_to(REPO) or out == src or out in (REPO, REPO / 'library', REPO / 'library/motion_reference/kuavo_s54/retarget_data'):
        p.error('Use a separate output directory inside the S54 workspace')
    if not a.clips and not a.all:
        p.error('Select --clips or --all')
    files = [v for v in sorted(src.glob('*.csv')) if a.all or any(fnmatch.fnmatch(v.stem, s) for s in a.clips)]
    if not files:
        p.error('No matching source clips')
    cfg = json.loads(a.collision_config.read_text())
    ground = json.loads((REPO / 'configs/s54_grounding.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    records = []
    manifest = out / 'diagnostics' / ('repair_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    manifest.parent.mkdir(parents=True, exist_ok=True)
    for source in files:
        stem = source.stem
        if (out / source.name).exists():
            if a.skip_existing and accepted(out / source.name):
                records.append(dict(clip=stem, status='SKIP_CERTIFIED'))
                continue
            raise FileExistsError(f'Choose a new output directory: {out / source.name}')
        stage = None
        if a.resume_candidates:
            from prepare_s54_motion import sha
            candidates = [v for v in (out / '.staging').glob(stem + '_*')
                          if (v / 'diagnostics' / stem / 'collision_progress.npz').exists()
                          and (v / source.name).exists() and sha(v / source.name) == sha(source)]
            if candidates:
                stage = max(candidates, key=lambda v: (v / 'diagnostics' / stem / 'collision_progress.npz').stat().st_mtime)
                print(f'Resuming private candidate {stage}', flush=True)
        resume = stage is not None
        stage = stage or out / '.staging' / (stem + '_' + uuid.uuid4().hex[:10])
        stage.mkdir(parents=True, exist_ok=resume)
        try:
            if not resume:
                for file in (source, source.with_suffix('.json')):
                    shutil.copy2(file, stage / file.name)
                for directory in ('raw', 'baseline', 'reference'):
                    (stage / directory).mkdir()
                    for file in (src / directory).glob(f'{stem}*'):
                        shutil.copy2(file, stage / directory / file.name)
                shutil.copytree(src / 'diagnostics' / stem, stage / 'diagnostics' / stem)
            result = finish_candidate(stage, out, stem, cfg, ground)
            records.append(dict(clip=stem, status='PASS', frames=result['source_frames']))
        except Exception as exc:
            rejected = out / 'rejected' / stage.name
            rejected.parent.mkdir(exist_ok=True)
            stage.rename(rejected)
            records.append(dict(clip=stem, status='FAIL', error=str(exc), diagnostic=str(rejected)))
            manifest.write_text(json.dumps(records, indent=2))
            if not a.continue_on_error:
                raise
        manifest.write_text(json.dumps(records, indent=2))
    print(f'Repair manifest: {manifest}', flush=True)
    if any(r['status'] == 'FAIL' for r in records):
        raise SystemExit('Some clips were rejected; see manifest. They are absent from the training CSV directory.')


if __name__ == '__main__':
    main()
