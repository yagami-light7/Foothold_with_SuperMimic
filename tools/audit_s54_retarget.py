"""Inventory interrupted exports and audit actual S54 root/joint trajectories.

Read-only for datasets. Collision/foot checks verify the CSV-bound certificate;
this tool does not rerun expensive mesh collision certification for every clip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.reference_quality import root_motion_report
from supermimic.s54 import JOINT_NAMES, REPO, actuator_parameters, checkpoint_contract, model_contract


CERTIFICATE_CHECKS = (
    'continuous_interpolation', 'control_50hz', 'broadphase_no_missed_contacts',
    'recorded_adjustments_match', 'geometry_matches_repair', 'all_joint_velocity_limits',
    'motion_quality', 'heads_remain_neutral', 'finite_motionlib', 'foot_grounding',
)


def bvh_frames(path):
    with path.open() as stream:
        for line in stream:
            match = re.fullmatch(r'\s*Frames:\s*(\d+)\s*', line)
            if match:
                return int(match[1])
    raise ValueError(f'No BVH frame count: {path}')


def audit_clip(path, source_frames, limits, speeds, contract):
    record = dict(clip=path.stem, csv=str(path), status='FAIL', checks={})
    try:
        meta = json.loads(path.with_suffix('.json').read_text())
        x = np.loadtxt(path, delimiter=',', ndmin=2)
        if x.shape[1] != 7 + len(JOINT_NAMES) or len(x) < 2 or not np.isfinite(x).all():
            raise ValueError(f'Expected finite (N,36), got {x.shape}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        root = root_motion_report(x)
        q = x[:, 7:]
        source_speed = abs(np.diff(q, axis=0)) * 30.
        count = int(np.floor((len(x) - 1) / 30. / .02)) + 1
        t = np.arange(count) * .02 * 30.
        left = np.minimum(np.floor(t).astype(int), len(x) - 2)
        fraction = np.clip(t - left, 0., 1.)[:, None]
        control = x[left] * (1 - fraction) + x[left + 1] * fraction
        # Joint/root position interpolation is exactly the MotionLib rule.
        # Quaternion columns are not used here; source root rotation was
        # checked above with sign-invariant quaternion interval angles.
        control_speed = abs(np.diff(control[:, 7:], axis=0)) / .02
        cert = meta.get('self_collision_validation', {})
        checks = dict(
            complete_source_clip=source_frames is not None and len(x) == source_frames,
            robot_contract=all(meta.get(k) == v for k, v in contract.items()),
            metadata_units_fps=meta.get('fps') == 30. and meta.get('quat') == 'xyzw' and meta.get('units') == 'radian',
            metadata_frame_count=meta.get('frames') == len(x),
            csv_hash=meta.get('csv_sha256') == digest,
            root_motion=root['status'] == 'PASS',
            physical_joint_limits=bool(np.all(q >= limits[:, 0] - 1e-6) and np.all(q <= limits[:, 1] + 1e-6)),
            source_joint_velocity=bool(np.all(source_speed <= speeds + 1e-5)),
            control_joint_velocity=bool(np.all(control_speed <= speeds + 1e-5)),
            neutral_head_reference=bool(abs(q[:, -2:]).max() <= 1e-8),
            csv_bound_collision_foot_certificate=(cert.get('version') == 1 and cert.get('status') == 'PASS'
                and cert.get('csv_sha256') == digest
                and all(cert.get('checks', {}).get(k) is True for k in CERTIFICATE_CHECKS)
                and cert.get('continuous_certificate', {}).get('continuous_joint_interpolation') is True),
        )
        record.update(checks=checks, status='PASS' if all(checks.values()) else 'FAIL',
                      source_frames=len(x), expected_source_frames=source_frames, control_frames=count,
                      root_motion=root, control_max_root_speed_m_s=float(np.linalg.norm(np.diff(control[:, :3], axis=0), axis=1).max() / .02),
                      source_joint_speed_above_urdf=int(np.sum(source_speed > speeds + 1e-5)),
                      control_joint_speed_above_urdf=int(np.sum(control_speed > speeds + 1e-5)),
                      max_source_joint_velocity_ratio=float((source_speed / speeds).max()),
                      max_control_joint_velocity_ratio=float((control_speed / speeds).max()),
                      csv_sha256=digest)
        record['failed_checks'] = [k for k, passed in checks.items() if not passed]
    except (ValueError, OSError, KeyError, IndexError) as exc:
        record['error'] = str(exc)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw_dir', type=Path, default=REPO/'library/motion_reference/kuavo_s54/retarget_data/raw')
    parser.add_argument('--motion_dirs', type=Path, nargs='+', default=[
        REPO/'library/motion_reference/kuavo_s54/retarget_data/reference_safe', REPO/'library/motion_reference/kuavo_s54/retarget_data/reference_startup_fixed'])
    parser.add_argument('--out', type=Path, default=REPO/'results/s54/retarget_inventory')
    parser.add_argument('--required_clips', nargs='*', default=[])
    args = parser.parse_args()
    source = {p.stem: bvh_frames(p) for p in sorted(args.raw_dir.glob('*.bvh'))}
    if not source:
        parser.error('No source BVHs')
    _, rows = model_contract()
    limits = np.array([r['range'] for r in rows])
    parameters = actuator_parameters()
    speeds = np.array([parameters[n]['velocity'] for n in JOINT_NAMES])
    contract = checkpoint_contract()
    records, directories = [], []
    exported, safe = set(), set()
    for directory in args.motion_dirs:
        csvs = sorted(directory.glob('*.csv'))
        rows = [audit_clip(path, source.get(path.stem), limits, speeds, contract) for path in csvs]
        records += rows
        names = {p.stem for p in csvs}
        exported |= names
        accepted = {r['clip'] for r in rows if r['status'] == 'PASS'}
        safe |= accepted
        manifests = []
        for path in sorted((directory/'diagnostics').glob('generation_*.json')):
            entries = json.loads(path.read_text())
            manifests += [dict(manifest=str(path), **r) for r in entries if r['status'] == 'FAIL']
        directories.append(dict(directory=str(directory), exported_count=len(csvs),
                                safe_count=len(accepted), unsafe_count=len(csvs)-len(accepted),
                                missing_exports=sorted(set(source)-names),
                                unsafe_exports=[r['clip'] for r in rows if r['status'] != 'PASS'],
                                unpublished_staging=[str(p) for p in sorted((directory/'.staging').glob('*')) if p.is_dir()],
                                recorded_generation_failures=manifests))
        print(f'{directory.name}: {len(csvs)} exported, {len(accepted)} pass audit, '
              f'{len(csvs)-len(accepted)} fail, {len(set(source)-names)} not exported', flush=True)
    report = dict(raw_count=len(source), unique_exported_count=len(exported & set(source)),
                  missing_exports=sorted(set(source)-exported), safe_unique_count=len(safe & set(source)),
                  needs_valid_export=sorted(set(source)-safe), directories=directories, clips=records,
                  scope='Actual full CSV/hash/root/joint checks at 30/50 Hz; collision/foot certificate integrity, not a new all-mesh proof or dynamic feasibility guarantee.')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'inventory.json').write_text(json.dumps(report, indent=2))
    for key in ('missing_exports', 'needs_valid_export'):
        (args.out/f'{key}.txt').write_text(''.join(f'{name}\n' for name in report[key]))
    print(f'Total: {len(source)} BVHs, {report["unique_exported_count"]} unique exported, '
          f'{len(report["missing_exports"])} not exported; {report["safe_unique_count"]} have a valid audited export')
    print(f'Report: {args.out / "inventory.json"}')
    absent = set(args.required_clips)-safe
    if absent:
        raise SystemExit(f'Required training clips lack a safe full export: {sorted(absent)}')


if __name__ == '__main__':
    main()
