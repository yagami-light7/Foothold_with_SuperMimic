"""M3.4 NPZ contract and MotionLib adapter; never resamples 50 Hz references.

Training opens train files only. Test metadata may be inspected to enforce the
split, but test arrays never enter a training MotionLib or its normalizers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ARRAYS = ('fps', 'joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w',
          'body_lin_vel_w', 'body_ang_vel_w')
MODES = ('straight', 'turn', 'spin')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entries(root, split='train'):
    root = Path(root).resolve()
    if split not in ('train', 'test'):
        raise ValueError('Explicit train/test split required')
    meta = json.loads((root/'metadata.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    expected = dict(fps=50, dof_count=29, body_count=30, quaternion_order='wxyz',
                    velocity_frame='world_link_origin_J_qdot', position_units='m', joint_units='rad',
                    root_body_name='base_link')
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError('M3.4 metadata contract mismatch')
    rows = manifest['motions']
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate motion IDs')
    for row in rows:
        path = (root/row['path']).resolve()
        if (not path.is_relative_to(root/row['split']) or path.suffix != '.npz'
                or row['split'] not in ('train', 'test')):
            raise ValueError(f'Unsafe split/path: {row}')
    for row in rows:
        if row['split'] == 'train' and row['path'].startswith('test/'):
            raise ValueError('Test reference marked as training')
    train_hashes = {r['motion_content_sha256'] for r in rows if r['split'] == 'train'}
    if train_hashes & {r['motion_content_sha256'] for r in rows if r['split'] == 'test'}:
        raise ValueError('Exact train/test content leakage')
    selected = [r for r in rows if r['split'] == split and r['offline_accepted'] is True]
    if not selected or (split == 'train' and any(r['motion_type'] not in MODES for r in selected)):
        raise ValueError('Missing accepted motions or unexpected training mode')
    return meta, manifest, selected


def load(root, row, meta, manifest):
    path = Path(root)/row['path']
    if sha256(path) != manifest['file_sha256'].get(row['path']):
        raise ValueError(f'NPZ checksum mismatch: {path}')
    with np.load(path, allow_pickle=False) as z:
        data = dict(z)
    n = row['frames']
    shapes = dict(joint_pos=(n, 29), joint_vel=(n, 29), body_pos_w=(n, 30, 3),
                  body_quat_w=(n, 30, 4), body_lin_vel_w=(n, 30, 3), body_ang_vel_w=(n, 30, 3))
    if n < 2 or any(data[k].shape != shape or not np.isfinite(data[k]).all() for k, shape in shapes.items()):
        raise ValueError(f'Nonfinite/invalid motion arrays: {path}')
    for key in ('joint_names', 'body_names'):
        if list(data[key]) != meta[key] or len(set(meta[key])) != len(meta[key]):
            raise ValueError(f'{key} ordering mismatch: {path}')
    if (float(np.asarray(data['fps']).item()) != 50 or str(np.asarray(data['quaternion_order']).item()) != 'wxyz'
            or str(np.asarray(data['dataset_split']).item()) != row['split'] or not bool(np.asarray(data['accepted']).item())):
        raise ValueError(f'Motion contract mismatch: {path}')
    if str(np.asarray(data['velocity_frame']).item()) != meta['velocity_frame']:
        raise ValueError(f'World link-origin velocity contract mismatch: {path}')
    if not np.isclose((n-1)*.02, row['duration_s'], atol=1e-6):
        raise ValueError(f'Frame count/duration clock mismatch: {path}')
    if np.max(np.abs(np.linalg.norm(data['body_quat_w'], axis=-1)-1)) > 1e-5:
        raise ValueError(f'Nonunit body quaternion: {path}')
    # The export encodes a uniform clock by fps and frame count. Validate any
    # explicit timestamp array as well; do not infer a second resampling clock.
    for key in ('timestamps', 'time_s', 'timestamps_s'):
        if key in data and (data[key].shape != (n,) or not np.isfinite(data[key]).all()
                            or not np.allclose(np.diff(data[key]), .02, atol=1e-7, rtol=1e-5)):
            raise ValueError(f'Invalid timestamps: {path}')
    digest = hashlib.sha256()
    for key in ARRAYS:
        a = np.ascontiguousarray(data[key])
        digest.update(key.encode()); digest.update(str(a.shape).encode()); digest.update(a.tobytes())
    if digest.hexdigest() != row['motion_content_sha256']:
        raise ValueError(f'Motion content checksum mismatch: {path}')
    return data


def dataset_contract(root):
    _, manifest, selected = entries(root, 'train')
    return dict(format='s54_flat_m3_4_v1', split='train', fps=50,
                metadata_sha256=sha256(Path(root)/'metadata.json'),
                manifest_sha256=sha256(Path(root)/'manifest.json'),
                motions={r['id']: r['motion_content_sha256'] for r in selected})


def motion_library(root, kin, device, split='train', ids=None):
    import torch
    # train.py enables TF32 for network throughput. FK contract validation and
    # reference construction need full float32 mantissas, not TF32 matmuls.
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        return _motion_library(root, kin, device, split, ids)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _motion_library(root, kin, device, split='train', ids=None):
    """Build the existing MotionLib interface after all-body FK/velocity checks."""
    import torch
    from .motion import MotionLib, BIN_FRAMES
    from .rotations import quat_to_mat
    from .s54 import JOINT_NAMES, SOURCE_MJCF

    meta, manifest, selected = entries(root, split)
    if (sha256(SOURCE_MJCF) != meta['model']['original_mjcf_sha256']
            or sha256(Path(root)/meta['model']['mjcf_file']) != meta['model']['mjcf_sha256']):
        raise ValueError('Dataset/source dynamics model checksum mismatch')
    if meta['joint_names'] != JOINT_NAMES or meta['body_names'] != kin.body_names:
        raise ValueError('Dataset and current 29-joint/30-body model ordering differ')
    if ids is not None:
        if not set(ids) <= {r['id'] for r in selected}:
            raise ValueError('Unknown or wrong-split motion IDs')
        selected = [r for r in selected if r['id'] in ids]
    if not selected:
        raise ValueError('No motions selected')
    # Reuse goal/time/bin methods without running the legacy CSV resampler.
    m = MotionLib.__new__(MotionLib)
    m.device = device
    m.names = [r['id'] for r in selected]
    m.modes = [r['motion_type'] for r in selected]
    m.split = split
    m.rows = selected
    m.validation = []
    values = {k: [] for k in ('root_p', 'root_R', 'root_v', 'root_w', 'q', 'qd',
                              'body_p', 'body_R', 'body_v', 'body_w')}
    lens = []
    root_index = meta['body_names'].index(meta['root_body_name'])
    for row in selected:
        d = load(root, row, meta, manifest)
        tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        p = tensor(d['body_pos_w'][:, root_index]); R = quat_to_mat(tensor(d['body_quat_w'][:, root_index]))
        v = tensor(d['body_lin_vel_w'][:, root_index]); w = tensor(d['body_ang_vel_w'][:, root_index])
        q, qd = tensor(d['joint_pos']), tensor(d['joint_vel'])
        if ((q < kin.q_lo-1e-5) | (q > kin.q_hi+1e-5)).any():
            raise ValueError(f'Joint limits violated: {row["id"]}')
        with torch.no_grad():
            actual = kin.forward(p, R, v, w, q, qd, all_bodies=True)
            expected = (tensor(d['body_pos_w']), quat_to_mat(tensor(d['body_quat_w'])),
                        tensor(d['body_lin_vel_w']), tensor(d['body_ang_vel_w']))
            errors = [float((a-b).abs().max()) for a, b in zip(actual, expected)]
        if any(e > tol for e, tol in zip(errors, (1e-4, 1e-4, 1e-3, 1e-3))):
            raise ValueError(f'All-body dataset FK/velocity mismatch {row["id"]}: {errors}')
        m.validation.append(dict(id=row['id'], max_errors=errors))
        for key, x in zip(('root_p', 'root_R', 'root_v', 'root_w', 'q', 'qd'), (p, R, v, w, q, qd)):
            values[key].append(x)
        for key, x in zip(('body_p', 'body_R', 'body_v', 'body_w'), actual):
            values[key].append(x[:, kin.tracked_idx])
        lens.append(len(q))
    for key, xs in values.items():
        setattr(m, key, torch.cat(xs))
    m.clip_len = torch.tensor(lens, device=device)
    m.clip_start = torch.cumsum(torch.tensor([0]+lens[:-1], device=device), 0)
    m.clip_end = m.clip_start+m.clip_len-1
    m.num_frames = sum(lens)
    m.clip_id = torch.repeat_interleave(torch.arange(len(lens), device=device), m.clip_len)
    Rt = m.root_R.transpose(-1, -2)
    m.root_v_local = (Rt @ m.root_v.unsqueeze(-1)).squeeze(-1)
    m.root_w_local = (Rt @ m.root_w.unsqueeze(-1)).squeeze(-1)
    m.root_gz = m.root_R[:, 2, 2]
    bins = [(c, int(m.clip_start[c])+s) for c, n in enumerate(lens) for s in range(0, n-1, BIN_FRAMES)]
    m.bin_clip = torch.tensor([c for c, _ in bins], device=device)
    m.bin_start = torch.tensor([s for _, s in bins], device=device)
    m.num_bins = len(bins)
    m.frame_bin = torch.zeros(m.num_frames, dtype=torch.long, device=device)
    for b, (c, s) in enumerate(bins):
        m.frame_bin[s:min(s+BIN_FRAMES, int(m.clip_end[c])+1)] = b
    return m


def balanced_bin_probs(motion, bin_G, G_max, iteration, straight_iters=1000):
    """Equal class -> equal clip -> adaptive bins, independent of clip length.

    Warm start emphasizes straight (80/10/10); the subsequent distribution is
    exactly class balanced. Difficulty changes only bin weights within a clip.
    """
    import torch
    weights = (.8, .1, .1) if iteration < straight_iters else (1/3,)*3
    probs = torch.zeros_like(bin_G)
    for mode, weight in zip(MODES, weights):
        clips = [i for i, name in enumerate(motion.modes) if name == mode]
        if not clips:
            raise ValueError(f'Missing training mode: {mode}')
        for c in clips:
            mask = motion.bin_clip == c
            difficulty = torch.softmax(-25*bin_G[mask]/G_max, 0)
            probs[mask] = weight/len(clips)*(.5/int(mask.sum())+.5*difficulty)
    return probs
