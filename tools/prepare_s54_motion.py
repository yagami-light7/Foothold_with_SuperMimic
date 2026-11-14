"""Losslessly preserve raw Leju-GMR CSV; validate and expand into isolated training CSV."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from supermimic.reference_quality import validate_root_motion
from supermimic.s54 import (REPO, ROBOT_ID, JOINT_NAMES, CSV_JOINT_NAMES, HEAD_JOINT_NAMES,
                            SOURCE_FPS, model_contract, checkpoint_contract, actuator_parameters)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(src, dst, expected_frames=None, raw_dir=None):
    src, dst = Path(src).resolve(), Path(dst).resolve()
    raw_dir = Path(raw_dir).resolve() if raw_dir is not None else REPO / "library/motion_reference/kuavo_s54/retarget_data/reference/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    preserved = raw_dir / src.name
    if preserved.exists() and sha(preserved) != sha(src):
        raise ValueError(f"Raw archive differs; refusing overwrite: {preserved}")
    if src != preserved: shutil.copy2(src, preserved)
    x = np.loadtxt(preserved, delimiter=",", ndmin=2)
    if x.shape[1] != 7 + len(CSV_JOINT_NAMES) or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError(f"Expected finite (N,34) raw motion, got {x.shape}")
    if expected_frames is not None and len(x) != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames, got {len(x)}")
    norm_err = np.abs(np.linalg.norm(x[:, 3:7], axis=1)-1)
    if norm_err.max() > 1e-5: raise ValueError(f"Root quaternion norm error: {norm_err.max()}")
    root_quality = validate_root_motion(x, SOURCE_FPS, str(src))
    _, rows = model_contract()
    y = np.empty((len(x), 7 + len(JOINT_NAMES))); y[:, :7] = x[:, :7]
    stats = {}; pars = actuator_parameters()
    for i, r in enumerate(rows):
        n = r["name"]
        y[:, 7+i] = x[:, 7+CSV_JOINT_NAMES.index(n)] if n in CSV_JOINT_NAMES else r["neutral"]
        q = y[:, 7+i]; lo, hi = r["range"]
        excess = np.maximum(np.maximum(lo-q, q-hi), 0)
        speed = np.abs(np.diff(q)) * SOURCE_FPS
        stats[n] = dict(min=float(q.min()), max=float(q.max()),
                        range_violations=int((excess > 1e-6).sum()), max_excess_rad=float(excess.max()),
                        max_speed_rad_s=float(speed.max()), speed_above_urdf=int((speed > pars[n]["velocity"]).sum()),
                        jumps_above_pi=int((np.abs(np.diff(q)) > np.pi).sum()))
    metadata = dict(**checkpoint_contract(), fps=SOURCE_FPS, quat="xyzw", units="radian",
                    frames=len(x), shape=list(y.shape), raw_csv=str(preserved.relative_to(REPO)),
                    raw_sha256=sha(preserved), head_fill={n: rows[JOINT_NAMES.index(n)]["neutral"] for n in HEAD_JOINT_NAMES},
                    quat_max_norm_error=float(norm_err.max()), statistics=stats,
                    root_max_speed_m_s=float(np.linalg.norm(np.diff(x[:, :3],axis=0)*SOURCE_FPS,axis=1).max()),
                    root_max_rotation_step_rad=float((2*np.arccos(np.clip(np.abs(np.sum(x[1:,3:7]*x[:-1,3:7],axis=1)),0,1))).max()),
                    root_motion_validation=root_quality,
                    modifications="Only append model-default heads. No clipping, smoothing or retiming.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # A previously misplaced raw CSV is archived before replacing it with the full version.
    if dst.exists() and dst != src:
        old = np.loadtxt(dst, delimiter=",", ndmin=2)
        if old.shape == x.shape and np.array_equal(old, x):
            dst.unlink()  # byte-for-byte archive is already verified above
        elif old.shape != y.shape or not np.allclose(old, y, rtol=0, atol=1e-15):
            raise ValueError(f"Refusing to replace unrelated output {dst}")
    if src == dst and src != preserved:
        if sha(src) != sha(preserved): raise ValueError("Archive verification failed")
        src.unlink()
    if any(s["range_violations"] or s["jumps_above_pi"] for s in stats.values()):
        report = dst.with_suffix(".rejected.json")
        report.write_text(json.dumps(metadata, indent=2))
        raise ValueError(f"Invalid motion; no training output written. See {report}")
    np.savetxt(dst, y, delimiter=",", fmt="%.17g")
    metadata["csv_sha256"] = sha(dst)
    dst.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"PASS raw {x.shape} → {y.shape}, fps={SOURCE_FPS}, quat err={norm_err.max():.3g}; head={metadata['head_fill']}")
    print(f"No clipping. {sum(s['speed_above_urdf'] for s in stats.values())} joint intervals exceed URDF velocity; see {dst.with_suffix('.json')}")
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=REPO / "library/motion_reference/kuavo_s54/retarget_data/reference/raw/dance1_subject1.csv")
    p.add_argument("--output", type=Path, default=REPO / "library/motion_reference/kuavo_s54/retarget_data/reference/dance1_subject1.csv")
    p.add_argument("--expected_frames", type=int, default=None)
    p.add_argument("--raw_dir", type=Path, help="Separate raw archive for an independent reference dataset")
    a=p.parse_args(); prepare(a.input,a.output,a.expected_frames,a.raw_dir)
