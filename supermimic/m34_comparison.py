"""Explicit, single-source GMR-Raw PD comparison; inaccessible to the trainer."""
from pathlib import Path
import numpy as np
import torch

from .m34 import entries, motion_library, sha256
from .rotations import quat_to_mat


def raw_library(root, kin, device, clean_id):
    meta, manifest, rows = entries(root, 'train')
    row = next(r for r in rows if r['id'] == clean_id)
    relative = row['raw_comparison_path']
    path = (Path(root)/relative).resolve()
    if not path.is_relative_to((Path(root)/'comparison_only/raw_gmr').resolve()):
        raise ValueError('Raw comparison must use the named 29-joint NPZ, never original pickle')
    if sha256(path) != manifest['file_sha256'][relative]:
        raise ValueError('Raw comparison checksum mismatch')
    with np.load(path, allow_pickle=False) as z: d = dict(z)
    scalar = lambda k: np.asarray(d[k]).item()
    if (scalar('accepted') or scalar('dataset_split') != 'comparison_only'
            or scalar('data_role') != 'original_GMR_comparison_only_not_cleaned'
            or scalar('fps') != 50 or scalar('quaternion_order') != 'wxyz'
            or scalar('velocity_frame') != meta['velocity_frame']
            or scalar('model_sha256') != meta['model']['original_mjcf_sha256']):
        raise ValueError('Raw comparison contract mismatch')
    if list(d['joint_names']) != meta['joint_names'] or list(d['body_names']) != meta['body_names']:
        raise ValueError('Raw comparison joint/body order mismatch')
    n = len(d['joint_pos'])
    shapes = dict(joint_pos=(n, 29), joint_vel=(n, 29), body_pos_w=(n, 30, 3),
                  body_quat_w=(n, 30, 4), body_lin_vel_w=(n, 30, 3), body_ang_vel_w=(n, 30, 3))
    if n < 2 or any(d[k].shape != shape or not np.isfinite(d[k]).all() for k, shape in shapes.items()):
        raise ValueError('Invalid raw comparison arrays')
    if np.max(np.abs(np.linalg.norm(d['body_quat_w'], axis=-1)-1)) > 1e-5:
        raise ValueError('Raw comparison has nonunit quaternion')
    # Start from the fully validated clean model/interface, then replace every
    # per-frame field and index with this explicit, non-training raw reference.
    m = motion_library(root, kin, device, 'train', [clean_id])
    tensor = lambda a: torch.as_tensor(a, device=device, dtype=torch.float32)
    i = meta['body_names'].index('base_link')
    m.root_p = tensor(d['body_pos_w'][:, i]); m.root_R = quat_to_mat(tensor(d['body_quat_w'][:, i]))
    m.root_v = tensor(d['body_lin_vel_w'][:, i]); m.root_w = tensor(d['body_ang_vel_w'][:, i])
    m.q = tensor(d['joint_pos']); m.qd = tensor(d['joint_vel'])
    if ((m.q < kin.q_lo-1e-5) | (m.q > kin.q_hi+1e-5)).any():
        raise ValueError('Raw comparison exceeds model joint limits; retain as reference failure')
    expected = (tensor(d['body_pos_w']), quat_to_mat(tensor(d['body_quat_w'])),
                tensor(d['body_lin_vel_w']), tensor(d['body_ang_vel_w']))
    actual = kin.forward(*m.ref_state(torch.arange(n, device=device)), all_bodies=True)
    errors = [float((a-b).abs().max()) for a, b in zip(actual, expected)]
    if any(e > t for e, t in zip(errors, (1e-4, 1e-4, 1e-3, 1e-3))):
        raise ValueError(f'Raw comparison FK/velocity mismatch: {errors}')
    m.body_p, m.body_R, m.body_v, m.body_w = [x[:, kin.tracked_idx] for x in actual]
    Rt = m.root_R.transpose(-1, -2)
    m.root_v_local = (Rt @ m.root_v.unsqueeze(-1)).squeeze(-1)
    m.root_w_local = (Rt @ m.root_w.unsqueeze(-1)).squeeze(-1)
    m.root_gz = m.root_R[:, 2, 2]
    m.names = [clean_id+'_raw']; m.split = 'comparison_only'; m.num_frames = n
    m.clip_len = torch.tensor([n], device=device); m.clip_start = torch.tensor([0], device=device)
    m.clip_end = torch.tensor([n-1], device=device); m.clip_id = torch.zeros(n, dtype=torch.long, device=device)
    m.bin_start = torch.arange(0, n-1, 50, device=device); m.num_bins = len(m.bin_start)
    m.bin_clip = torch.zeros(m.num_bins, dtype=torch.long, device=device)
    m.frame_bin = (torch.arange(n, device=device)//50).clamp_max(m.num_bins-1)
    m.validation = [dict(id=m.names[0], max_errors=errors, role='comparison_only', paired_clean=clean_id)]
    return m
