"""M3.4 adapter regression: split isolation, native timing, FK, goals, sampling."""
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from supermimic.m34 import entries, load, motion_library, balanced_bin_probs
from supermimic.kinematics import S54Kinematics
from test_s54_cpu import fk_test, core_test


def main():
    root = Path('library/motion_reference/kuavo_s54/datasets/s54_flat_dataset/dataset_export')
    opened = []
    original = np.load
    def guarded(path, *args, **kwargs):
        opened.append(Path(path))
        assert Path(path).resolve().is_relative_to((root/'train').resolve()), path
        return original(path, *args, **kwargs)
    kin = S54Kinematics()
    with patch('numpy.load', guarded):
        m = motion_library(root, kin, 'cpu')
    assert len(opened) == 14 and len(m.names) == 14 and m.num_frames == 13714
    assert m.clip_len.tolist() == [r['frames'] for r in entries(root)[2]]
    for iteration, expected in [(0, (.8, .1, .1)), (1000, (1/3,)*3)]:
        probs = balanced_bin_probs(m, torch.randn(m.num_bins), 100., iteration)
        assert torch.isfinite(probs).all() and (probs > 0).all() and abs(float(probs.sum())-1) < 1e-6
        for mode, weight in zip(('straight', 'turn', 'spin'), expected):
            clip_mask = torch.tensor([x == mode for x in m.modes])
            assert abs(float(probs[clip_mask[m.bin_clip]].sum())-weight) < 1e-6
    future = m.future_frames(m.clip_end, 10)
    assert torch.equal(future, m.clip_end[:, None].expand_as(future))
    assert torch.isfinite(m.goal(m.clip_end, 10)).all()
    meta, manifest, rows = entries(root)
    row = rows[0]
    original_data = load(root, row, meta, manifest)
    for key, mutate in [('quaternion_order', lambda d: np.array('xyzw')),
                        ('joint_names', lambda d: d['joint_names'][::-1]),
                        ('joint_pos', lambda d: np.full_like(d['joint_pos'], np.nan))]:
        broken = copy.deepcopy(original_data); broken[key] = mutate(broken)
        class FakeNPZ:
            def __enter__(self): return broken
            def __exit__(self, *_): pass
        with patch('numpy.load', return_value=FakeNPZ()):
            try: load(root, row, meta, manifest)
            except ValueError: pass
            else: raise AssertionError(f'Accepted invalid {key}')
    report = dict(status='PASS', train_only_npz_opened=len(opened), frames=m.num_frames,
                  fk=fk_test(kin, m, 32), algorithm=core_test(kin, m))
    out = Path('results/s54_m34_cpu.json'); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
