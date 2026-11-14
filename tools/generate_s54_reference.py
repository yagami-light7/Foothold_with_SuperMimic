"""LAFAN1 BVH -> ground-aware GMR -> corrected S54 PKL/36-column CSV.

CPU-only. Reuses the installed Leju-GMR loader, solver and model; it never edits
Leju-GMR. Optional --view opens one accepted clip. Multiple accepted clips share
one MotionLib directory; grounding also checks 50 Hz control interpolation.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fnmatch
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys
import uuid

# GMR and its sparse QPs are CPU jobs. Avoid OpenMP oversubscription when
# producing many clips; callers can explicitly choose another thread count.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

from retarget_s54_upper_body import REPO, load_bvh_frames
from supermimic.s54 import JOINT_NAMES, model_contract
from supermimic.reference_quality import validate_root_motion


def settle_first_frame(solver, frame, max_calls=300):
    """Keep IK solver convergence outside the exported motion timeline."""
    previous = None
    stable = 0
    for call in range(1, max_calls + 1):
        pose = solver.retarget(deepcopy(frame), offset_to_ground=False).copy()
        if not np.isfinite(pose).all():
            raise ValueError('Nonfinite first-frame IK warmup')
        if previous is not None:
            position = float(np.linalg.norm(pose[:3] - previous[:3]))
            rotation = float(2 * np.arccos(np.clip(abs(pose[3:7] @ previous[3:7]), 0, 1)))
            # The upper-body stage reconstructs both arms. Their weak GMR
            # null-space drift must not prevent root/legs/waist from settling.
            joints = float(abs(pose[7:20] - previous[7:20]).max())
            if call in (2, 3, 5, 10, 20, 50, 100, 200, 300):
                print(f'First-frame IK call {call}: root_delta={position:.6g} m, '
                      f'rotation_delta={rotation:.6g} rad, joint_delta={joints:.6g} rad', flush=True)
            stable = stable + 1 if position < 1e-5 and rotation < 1e-4 and joints < 1e-4 else 0
            if stable >= 2:
                report = dict(status='PASS', calls=call, last_root_position_change_m=position,
                              last_root_rotation_change_rad=rotation, last_lower_body_joint_change_rad=joints)
                print(f'First-frame IK settled before export: {report}', flush=True)
                return pose, report
        previous = pose
    raise ValueError(f'First-frame IK did not settle in {max_calls} calls '
                     f'(position={position}, rotation={rotation}, joints={joints}); refusing solver-transient reference')


def baseline(gmr, bvh, dst, max_frames, ground_cfg=None):
    frames, height = load_bvh_frames(gmr, bvh)
    count = min(max_frames or len(frames), len(frames))
    header = bvh.read_text().split('MOTION', 1)[1][:150]
    fps = 1 / float(re.search(r'Frame Time:\s*([\d.]+)', header)[1])
    if abs(fps - 30) > .001:
        raise ValueError(f'Expected LAFAN1 30 FPS: {bvh}, got {fps}')
    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
    solver = GeneralMotionRetargeting('bvh_lafan1', 'kuavo_s54', actual_human_height=height, verbose=False)
    expected, joint_rows = model_contract()
    for field in ('body_parentid', 'body_pos', 'body_quat', 'jnt_type', 'jnt_axis', 'jnt_pos', 'jnt_range', 'qpos0'):
        if not np.array_equal(getattr(expected, field), getattr(solver.model, field)):
            raise ValueError(f'GMR model differs from the SuperMimic S54 contract: {field}')
    source_module = sys.modules['s54_upper_bvh_loader']
    source_module.apply_initial_knee_flexion_bvh_script(solver, .1)
    grounder = None
    if ground_cfg:
        from s54_grounding import GroundedTargets
        grounder = GroundedTargets(solver, frames, ground_cfg)
    addresses = [solver.model.jnt_qposadr[solver.model.joint(n).id] for n in JOINT_NAMES[:27]]
    lower = np.array([r['range'][0] for r in joint_rows[:27]])
    upper = np.array([r['range'][1] for r in joint_rows[:27]])
    poses = []
    warmup = None
    snapped_intervals, max_roundoff = 0, 0.
    for i, frame in enumerate(frames[:count]):
        if grounder:
            grounder.index = i
        if i == 0:
            pose, warmup = settle_first_frame(solver, frame)
            if grounder:
                grounder.targets[:] = grounder.targets[-1:]
        else:
            pose = solver.retarget(dict(frame), offset_to_ground=False).copy()
        if pose.shape != (solver.model.nq,) or not np.isfinite(pose).all():
            raise ValueError(f'Invalid GMR frame {i}')
        excess = np.maximum(lower - pose[addresses], pose[addresses] - upper)
        if excess.max() > 1e-6:
            raise ValueError(f'GMR frame {i} violates a physical joint bound by {excess.max()} rad')
        snapped_intervals += int(np.sum(excess > 0))
        max_roundoff = max(max_roundoff, float(excess.max()))
        # Mink can exceed a bound by solver roundoff (~1e-7 rad). This is
        # recorded, limited to 1e-6 rad, and distinct from trajectory clipping.
        pose[addresses] = np.clip(pose[addresses], lower, upper)
        poses.append(pose)
        if i == 0 or (i + 1) % 300 == 0 or i + 1 == count:
            print(f'{bvh.stem}: headless baseline {i + 1}/{count}', flush=True)
    poses = np.asarray(poses)
    validate_root_motion(np.c_[poses[:, :3], poses[:, [4, 5, 6, 3]]], label=f'{bvh.stem} baseline')
    result = dict(fps=30., root_pos=poses[:, :3], root_rot=poses[:, [4, 5, 6, 3]],
                  dof_pos=poses[:, addresses], robot='kuavo_s54',
                  dof_layout=dict(leg_indices=list(range(12)), waist_indices=[12], arm_indices=list(range(13, 27))),
                  local_body_pos=None, link_body_list=None, startup_warmup=warmup,
                  ik_bound_roundoff=dict(snapped_values=snapped_intervals, max_excess_rad=max_roundoff,
                                         allowed_roundoff_rad=1e-6))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open('wb') as f:
        pickle.dump(result, f)
    if grounder:
        dst.with_suffix('.ground_ik.json').write_text(json.dumps(grounder.report(), indent=2))
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gmr', type=Path, default=Path('/mnt/sage_data/workspace/Leju-GMR'))
    p.add_argument('--bvh_dir', type=Path)
    select = p.add_mutually_exclusive_group(required=True)
    select.add_argument('--clips', nargs='+', help='Clip stems or quoted glob patterns, e.g. walk1_subject1 or "walk*"')
    select.add_argument('--all', action='store_true', help='Generate all local LAFAN1 BVHs sequentially')
    p.add_argument('--exclude', nargs='*', default=[], help='Clip stems or glob patterns to exclude')
    p.add_argument('--out_dir', type=Path, default=REPO / 'library/motion_reference/kuavo_s54/retarget_data/reference_startup_fixed')
    p.add_argument('--config', type=Path, default=REPO / 'configs/s54_upper_body_retarget.json')
    p.add_argument('--ground_config', type=Path, default=REPO / 'configs/s54_grounding.json')
    p.add_argument('--collision_config', type=Path, default=REPO / 'configs/s54_self_collision.json')
    p.add_argument('--no_grounding', action='store_true', help='Diagnostic comparison only: disable foot correction')
    p.add_argument('--view', action='store_true', help='Open the single selected corrected clip after generation')
    p.add_argument('--reuse_baseline_dir', type=Path, help='Optional existing GMR PKL directory; missing clips are generated headlessly')
    p.add_argument('--skip_existing', action='store_true', help='Leave existing exported clips unchanged')
    p.add_argument('--continue_on_error', action='store_true', help='Continue batch, quarantining failed clips and reporting a nonzero exit status')
    p.add_argument('--dry_run', action='store_true', help='List selected clips and outputs without running IK')
    p.add_argument('--max_frames', type=int, help='Explicit prefix probe, not a full clip export')
    a = p.parse_args()
    root = (a.bvh_dir or REPO / 'library/motion_reference/kuavo_s54/retarget_data/raw').resolve()
    out = a.out_dir.resolve()
    if (not out.is_relative_to(REPO)
            or out in (REPO, REPO / 'library', REPO / 'library/motion_reference/kuavo_s54/retarget_data')
            or out == root or root in out.parents):
        p.error('Use an independent output directory inside the S54 workspace')
    if a.max_frames is not None and a.max_frames < 2:
        p.error('--max_frames must be at least 2')
    available = sorted(root.glob('*.bvh'))
    if a.clips:
        for pattern in a.clips:
            if not any(fnmatch.fnmatch(v.stem, pattern) for v in available):
                p.error(f'No BVH matches {pattern!r} in {root}')
    clips = [v for v in available if (a.all or any(fnmatch.fnmatch(v.stem, s) for s in a.clips))
             and not any(fnmatch.fnmatch(v.stem, s) for s in a.exclude)]
    if not clips:
        p.error('No selected BVH clips')
    if a.view and len(clips) != 1:
        p.error('--view requires exactly one clip')
    if a.no_grounding:
        p.error('Training exports require grounding and self collision validation; use the standalone upper-body tool for diagnostics')
    if a.reuse_baseline_dir and not a.no_grounding:
        p.error('Ground-aware IK needs a fresh baseline; omit --reuse_baseline_dir')
    for clip in clips:
        print(f'{clip.stem}: {clip} -> {out / (clip.stem + ".csv")}', flush=True)
    print(f'Selected {len(clips)} clips; CPU only; prefix={a.max_frames}', flush=True)
    if a.dry_run:
        return
    ground_cfg = None if a.no_grounding else json.loads(a.ground_config.read_text())
    collision_cfg = json.loads(a.collision_config.read_text())
    from s54_reference_publish import finish_candidate, accepted
    out.mkdir(parents=True, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    manifest = out / 'diagnostics' / f'generation_{tag}.json'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for bvh in clips:
        stem = bvh.stem
        csv = out / f'{stem}.csv'
        metrics = out / 'diagnostics' / stem / 'upper_body_metrics.json'
        if csv.exists() or metrics.exists():
            if a.skip_existing and csv.exists():
                if not accepted(csv):
                    raise ValueError(f'Existing clip has no valid self collision certificate: {csv}; choose a new --out_dir')
                records.append(dict(clip=stem, status='SKIP_EXISTING'))
                manifest.write_text(json.dumps(records, indent=2))
                continue
            raise FileExistsError(f'Refusing to replace {stem}; choose a new --out_dir or use --skip_existing')
        stage = out / '.staging' / (stem + '_' + uuid.uuid4().hex[:10])
        stage.mkdir(parents=True)
        try:
            pk = stage / 'baseline' / f'{stem}_30fps.pkl'
            source = a.reuse_baseline_dir / pk.name if a.reuse_baseline_dir else None
            if source and source.exists():
                pk.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, pk)
                print(f'Reusing baseline {source}', flush=True)
            else:
                baseline(a.gmr, bvh, pk, a.max_frames, ground_cfg)
            cmd = [sys.executable, str(REPO / 'tools/retarget_s54_upper_body.py'), '--gmr', str(a.gmr),
                   '--bvh', str(bvh), '--baseline', str(pk), '--out_dir', str(stage), '--config', str(a.config)]
            if a.max_frames is not None:
                cmd += ['--max_frames', str(a.max_frames)]
            log = stage / 'diagnostics' / stem / 'generation.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('w') as f:
                child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in child.stdout:
                    print(line, end='', flush=True); f.write(line)
                status = child.wait()
            if status:
                raise RuntimeError(f'Upper-body export returned {status}; see {log}')
            result = json.loads((stage / 'diagnostics' / stem / 'upper_body_metrics.json').read_text())
            if not all(result['checks'].values()):
                raise ValueError(f'Acceptance checks failed: {metrics}')
            finish_candidate(stage, out, stem, collision_cfg, ground_cfg)
            records.append(dict(clip=stem, status='PASS', frames=result['frames'], fps=result['fps'],
                                max_frames=a.max_frames, grounding=bool(ground_cfg), self_collision=True))
        except Exception as exc:
            # Only files created for this new clip can reach this path. Failed
            # candidates are kept for review outside the training CSV glob.
            rejected = out / 'rejected' / stage.name
            if stage.exists():
                rejected.parent.mkdir(parents=True, exist_ok=True)
                stage.rename(rejected)
            records.append(dict(clip=stem, status='FAIL', error=str(exc), diagnostic=str(rejected)))
            manifest.write_text(json.dumps(records, indent=2))
            if not a.continue_on_error:
                raise
        manifest.write_text(json.dumps(records, indent=2))
    print(f'Generation manifest: {manifest}')
    if any(r['status'] == 'FAIL' for r in records):
        raise SystemExit('Some clips failed; their CSVs are quarantined under rejected/. See manifest.')
    if a.view:
        subprocess.run([sys.executable, str(REPO / 'tools/view_s54_reference.py'),
                        '--motion_dir', str(out), '--clip', clips[0].stem], check=True)


if __name__ == '__main__':
    main()
