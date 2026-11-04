"""CPU regression checks for real mesh/capsule support and signed penetration."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from supermimic.s54 import model_contract
from tools.data.s54_grounding import FootGeometry, interpolate
from tools.validation.validate_s54_grounding import native_gaps


def main():
    model, rows = model_contract()
    rng = np.random.default_rng(54)
    addresses = [r['qposadr'] for r in rows]
    values = np.tile(np.r_[model.qpos0[:3], model.qpos0[[4, 5, 6, 3]], model.qpos0[addresses]], (24, 1))
    values[:, 3:7] = Rotation.random(24, random_state=rng).as_quat()
    bounds = np.array([r['range'] for r in rows])
    values[:, 7:] = rng.uniform(bounds[:, 0], bounds[:, 1], (24, 29))
    geometry = FootGeometry(model)
    before = geometry.gaps(values)
    values[:, 2] += .123
    after = geometry.gaps(values)
    translation_error = float(np.max(abs(after - before - .123)))
    assert translation_error < 1e-12
    _, support_error, _ = native_gaps(model, values)
    assert support_error < 1e-7
    # Deliberately penetrate the plane; do not rely on random poses happening
    # to penetrate. Signed distances must not be clipped to zero by the guard.
    values[:, 2] -= 3.
    collisions, negative_error, support = native_gaps(model, values)
    assert support.max() < 0 and collisions.max() < 0
    assert negative_error < 1e-7
    dense, i, a = interpolate(values, 150)
    assert np.allclose(dense[::5, :3], values[:, :3], rtol=0, atol=1e-12)
    assert np.allclose(dense[::5, 7:], values[:, 7:], rtol=0, atol=1e-12)
    assert np.allclose(abs(np.sum(dense[::5, 3:7] * values[:, 3:7], axis=1)), 1., atol=1e-12)
    assert np.max(abs(np.linalg.norm(dense[:, 3:7], axis=1) - 1)) < 1e-12
    # 150 Hz constraints contain both source and 50 Hz control grids.
    control, _, _ = interpolate(values, 50)
    assert np.allclose(dense[::3], control, rtol=0, atol=1e-12)
    print(json.dumps(dict(status='PASS', random_poses=24, signed_penetration='PASS',
                          translation_error_m=translation_error,
                          independent_mesh_capsule_support_error_m=max(support_error, negative_error),
                          interpolation_30_50_150hz='PASS'), indent=2))


if __name__ == '__main__':
    main()
