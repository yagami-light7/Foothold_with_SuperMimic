"""不可变的离线多参考库，训练时只进行 GPU 索引。"""

import json
from pathlib import Path

import numpy as np
import torch

from ..motion import CONTROL_DT
from ..motion_reference import MotionReferenceProvider
from ..roban import JOINT_NAMES, actuator_parameters
from .m4_reference import GeneratedMotionClip


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / 'configs/roban/m5_0_motion_library.json'
REFERENCE_ROOT = REPO.parent / 'Humanoid_Online_Motion_Generator/outputs'
SOURCE = REPO / 'outputs/m4_11/b_hand_huber/checkpoints/iter_1250.pt'
DATA_FIELDS = ('root_p', 'root_R', 'root_v', 'root_w', 'q', 'qd', 'quality',
               'body_p', 'body_R', 'body_v', 'body_w', 'foot_p', 'foot_R', 'foot_v',
               'contact', 'contact_valid', 'phase', 'root_v_local', 'root_w_local', 'root_gz')


def read_manifest(path=MANIFEST):
    value = json.loads(Path(path).read_text())
    if not isinstance(value.get('reference_version'), str) or not value['reference_version']:
        raise ValueError('离线参考库需要指定不变的版本')
    entries = [entry for split in ('train', 'heldout') for entry in value.get(split, [])]
    ids = [entry['clip_id'] for entry in entries]
    files = [entry['file_path'] for entry in entries]
    if not value.get('train') or len(ids) != len(set(ids)) or len(files) != len(set(files)):
        raise ValueError('参考库需要非空训练集、唯一片段名与互斥的文件路径')
    return value


def load_clip(entry, kin, device, root, split):
    path = (Path(root) if split == 'train' else REPO) / entry['file_path']
    clip = GeneratedMotionClip(path, kin, device, clip_id=entry['clip_id'], source=entry['file_path'])
    if clip.num_frames != entry['num_frames']:
        raise ValueError(f'{entry["clip_id"]} 的实际帧数与清单不一致')
    with np.load(path, allow_pickle=False) as source:
        fields = dict(step_length='step_length_m', step_frequency='step_frequency_hz', yaw_rate='yaw_rate_rad_s')
        for field, metadata in fields.items():
            if abs(float(source['meta__' + metadata]) - entry[field]) > 1e-8:
                raise ValueError(f'{entry["clip_id"]} 的 {field} 与清单不一致')
    velocity = clip.q.new_tensor([actuator_parameters()[name]['velocity'] for name in JOINT_NAMES])
    if torch.any(clip.qd.abs() > velocity + 1e-4):
        raise ValueError(f'{entry["clip_id"]} 的参考速度超过模型限位')
    if clip.num_frames < 3 or not (clip.contact[[0, -1]].all() and clip.contact_valid[[0, -1]].all()):
        raise ValueError(f'{entry["clip_id"]} 的首末帧必须是有效双脚支撑')
    clip.validation.update(parameters=dict(entry), dataset_split=split, valid_start_indices=[0],
                           max_velocity_ratio=float((clip.qd.abs() / velocity).max()), fk=clip.alignment)
    return clip


class MultiReferenceProvider(MotionReferenceProvider):
    """轨迹按原长度连续存放，局部帧只在所属片段内推进。"""

    def __init__(self, entries, kin, device, *, root, version):
        if not entries or len({entry['clip_id'] for entry in entries}) != len(entries):
            raise ValueError('必须指定非空且互不重复的训练片段')
        self.device, self.kin = device, kin
        self.names = [entry['clip_id'] for entry in entries]
        clips = [load_clip(entry, kin, device, root, 'train') for entry in entries]
        lengths = [clip.num_frames for clip in clips]
        self.clip_len = torch.tensor(lengths, dtype=torch.long, device=device)
        self.clip_start = self.clip_len.cumsum(0) - self.clip_len
        self.clip_end = self.clip_start + self.clip_len - 1
        self.clip_id = torch.arange(len(clips), device=device).repeat_interleave(self.clip_len)
        self.num_frames = sum(lengths)
        self.num_bins = len(clips)
        self.bin_start = self.clip_start
        self.bin_clip = torch.arange(self.num_bins, device=device)
        self.frame_bin = self.clip_id
        for field in DATA_FIELDS:
            setattr(self, field, torch.cat([getattr(clip, field) for clip in clips]))
        self.foot_indices, self.corners = clips[0].foot_indices, clips[0].corners
        self.sole_offsets = clips[0].sole_offsets
        for clip in clips[1:]:
            torch.testing.assert_close(self.sole_offsets, clip.sole_offsets, atol=1e-7, rtol=0)
        self.stance_p = self.target_foot_p = self.foot_p
        events = [len(clip.touchdown_time) for clip in clips]
        self.event_clip = self.bin_clip.repeat_interleave(torch.tensor(events, device=device))
        seconds = self.clip_start.to(torch.float64) * CONTROL_DT
        self.liftoff_time = torch.cat([clip.liftoff_time + t for clip, t in zip(clips, seconds)])
        self.touchdown_time = torch.cat([clip.touchdown_time + t for clip, t in zip(clips, seconds)])
        self.touchdown_side = torch.cat([clip.touchdown_side for clip in clips])
        self.validation = dict(version=version, indexing='packed_f_with_episode_anchor_v1', dt=CONTROL_DT,
                               clips=[clip.validation for clip in clips])

    def packed_frames(self, ids, local_frames):
        torch._assert_async(((ids >= 0) & (ids < self.num_bins)).all(), '参考片段编号越界')
        torch._assert_async(((local_frames >= 0) & (local_frames < self.clip_len[ids])).all(), '参考局部帧越界')
        return self.clip_start[ids] + local_frames

    def get_current(self, ids, local_frames):
        f = self.packed_frames(ids, local_frames)
        return self.sample(f[:, None], self.anchor(self.clip_start[ids]))

    def get_future(self, ids, local_frames, offsets):
        f = self.packed_frames(ids, local_frames)
        torch._assert_async((offsets >= 0).all(), '未来参考偏移必须非负')
        future = torch.minimum(f[:, None] + offsets[None], self.clip_end[ids, None])
        return self.sample(future, self.anchor(self.clip_start[ids]))

    def sample(self, f, anchor):
        low, high = self.clip_start[anchor.clip_id, None], self.clip_end[anchor.clip_id, None]
        torch._assert_async(((f >= low) & (f <= high)).all(), '参考查询跨越了 Episode 所属片段')
        return super().sample(f, anchor)
