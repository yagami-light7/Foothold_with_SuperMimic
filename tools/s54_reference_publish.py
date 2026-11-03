"""Private candidates become training-visible only after all acceptance checks."""
import json
from pathlib import Path
import shutil
import numpy as np

from supermimic.s54 import REPO
from supermimic.reference_quality import validate_root_motion
from s54_self_collision import repair_export
from s54_grounding import ground_export
from validate_s54_grounding import validate as validate_ground
from validate_s54_self_collision import validate as validate_self


def finish_candidate(stage, out, stem, collision_cfg, ground_cfg):
    stage, out = Path(stage), Path(out)
    diagnostic = stage / 'diagnostics' / stem
    # A repair of an existing dataset retains its old grounding evidence, then
    # recomputes grounding for the changed leg trajectory.
    for name in ('grounding_metrics.json', 'grounding_validation.json', 'before_ground_guard.npz'):
        path = diagnostic / name
        if path.exists():
            path.rename(path.with_name('before_self_collision_' + name))
    repair_export(stage, stem, collision_cfg)
    ground_export(stage, stem, ground_cfg)
    validate_ground(stage / f'{stem}.csv')
    result = validate_self(stage / f'{stem}.csv')
    publish(stage, out, stem)
    return result


def publish(stage, out, stem):
    if (out / f'{stem}.csv').exists():
        raise FileExistsError(f'Refusing to replace accepted clip {stem}')
    for directory in ('raw', 'baseline', 'reference'):
        for source in (stage / directory).glob(f'{stem}*'):
            destination = out / directory / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    target = out / 'diagnostics' / stem
    shutil.copytree(stage / 'diagnostics' / stem, target, dirs_exist_ok=True)
    def remap(value):
        if isinstance(value, str):
            return value.replace(str(stage) + '/', str(out) + '/')
        if isinstance(value, list):
            return [remap(v) for v in value]
        if isinstance(value, dict):
            return {k: remap(v) for k, v in value.items()}
        return value
    for report in target.glob('*.json'):
        value = remap(json.loads(report.read_text()))
        if report.name == 'upper_body_metrics.json':
            value['baseline_pkl'] = str(out / 'baseline' / f'{stem}_30fps.pkl')
        report.write_text(json.dumps(value, indent=2))
    path = stage / f'{stem}.csv'
    meta = remap(json.loads(path.with_suffix('.json').read_text()))
    meta['raw_csv'] = str((out / 'raw' / path.name).relative_to(REPO))
    # JSON is written before the CSV. The top-level training glob never sees a
    # partial or unverified candidate, including interrupted batch runs.
    (out / f'{stem}.json').write_text(json.dumps(meta, indent=2))
    temporary = out / f'.{stem}.accepted.tmp'
    shutil.copy2(path, temporary)
    temporary.replace(out / path.name)
    shutil.rmtree(stage)


def accepted(csv):
    from prepare_s54_motion import sha
    csv = Path(csv)
    try:
        validate_root_motion(np.loadtxt(csv, delimiter=',', ndmin=2), label=str(csv))
    except ValueError:
        return False
    metadata = json.loads(csv.with_suffix('.json').read_text())
    certificate = metadata.get('self_collision_validation', {})
    required = ('continuous_interpolation', 'control_50hz', 'broadphase_no_missed_contacts',
                'recorded_adjustments_match', 'geometry_matches_repair', 'all_joint_velocity_limits',
                'motion_quality', 'heads_remain_neutral', 'finite_motionlib', 'foot_grounding')
    return (certificate.get('version') == 1 and certificate.get('status') == 'PASS'
            and certificate.get('csv_sha256') == metadata.get('csv_sha256') == sha(csv)
            and all(certificate.get('checks', {}).get(k) is True for k in required)
            and certificate.get('continuous_certificate', {}).get('continuous_joint_interpolation') is True)
