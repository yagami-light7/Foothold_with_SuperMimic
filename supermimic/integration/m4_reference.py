"""将 M3.7.1 F2 原始 NPZ 接入 Roban 的有限参考时钟。"""
import numpy as np
import torch

from ..motion import CONTROL_DT
from ..motion_reference import MotionReferenceProvider
from ..roban import JOINT_NAMES, ROBOT_ID, ROOT_BODY, foot_geometry
from ..rotations import quat_slerp, quat_to_mat, so3_log


class GeneratedMotionClip(MotionReferenceProvider):
    """以生成器的原始时钟读取一条有限长度参考。"""

    def __init__(self, path, kin, device='cpu', src_fps=None, *, clip_id, source, legacy_f2=False):
        self.clip_name, self.source_path, self.legacy_f2 = clip_id, source, legacy_f2
        super().__init__([path], kin, device, src_fps, clip_names=[clip_id])
        control = self.at_time(self.control_time)
        pos_err = (self.foot_p - control['foot_p']).norm(dim=-1).max()
        rot_err = so3_log(self.foot_R @ control['foot_R'].transpose(-1, -2)).norm(dim=-1).max()
        self.alignment = dict(sole_position_max_m=float(pos_err), sole_orientation_max_rad=float(rot_err))
        if pos_err > .002 or rot_err > .002:
            raise ValueError(f'{self.clip_name} 足部与 Roban 模型 FK 不一致：{self.alignment}')
        self.foot_p, self.foot_R, self.foot_v = (control[k] for k in ('foot_p', 'foot_R', 'foot_v'))
        # 支撑点由生成器固定，保留原始足部轨迹。
        self.stance_p = self.target_foot_p = self.foot_p

    def _validation_robot(self):
        return dict(robot=ROBOT_ID, joint_names=JOINT_NAMES, root_body=ROOT_BODY)

    def _foot_sole_offsets(self):
        return self._source_sole_offsets

    def _load(self, path, explicit_fps):
        if not path.is_file():
            raise FileNotFoundError(f'找不到 {self.clip_name} 参考 {path}')
        with np.load(path, allow_pickle=False) as source:
            self._read_source(source, path, explicit_fps)
        frames = round(self.duration / CONTROL_DT)
        if abs(frames * CONTROL_DT - self.duration) > 1e-8:
            raise ValueError(f'{self.clip_name} 终点必须落在当前训练控制步网格上')
        self.control_time = torch.arange(frames + 1, device=self.device, dtype=torch.float64) * CONTROL_DT
        data = self.at_time(self.control_time)
        quality = (data['validity']['kinematic_valid'] & data['validity']['optimization_converged']).float()
        return dict(p=data['root_p'], R=data['root_R'], v=data['root_v'], w=data['root_w'],
                    q=data['q'], qd=data['qd'], quality=quality, contact=data['contact'],
                    contact_valid=data['contact_valid'], phase=data['phase'])

    def _read_source(self, source, path, explicit_fps):
        def array(key, shape, binary=False):
            if key not in source:
                raise ValueError(f'{self.clip_name} 缺少字段 {key}：{path}')
            x = source[key]
            if x.shape != shape:
                raise ValueError(f'F2 字段 {key} 应为 {shape}，实际 {x.shape}')
            if binary and not np.isin(x, [0, 1]).all():
                raise ValueError(f'F2 字段 {key} 必须是二值掩码')
            return x

        def numeric(key, shape):
            x = array(key, shape)
            if x.dtype.kind not in 'fiu' or not np.isfinite(x).all():
                raise ValueError(f'F2 字段 {key} 必须为有限数')
            return x

        def tensor(x, dtype=torch.float32):
            return torch.as_tensor(x, device=self.device, dtype=dtype)

        for key, expected in dict(case_id=self.clip_name,
                                  schema_version='roban_motion_v2', frame='world_Z_up_m',
                                  foot_frame='sole_projection_of_link_origin',
                                  robot_model=ROBOT_ID, quaternion_order='wxyz').items():
            if array('meta__' + key, ()).item() != expected:
                raise ValueError(f'F2 元数据 {key} 应为 {expected}')

        source_clip = str(array('meta__source_clip_id', ()).item())
        if not source_clip or (self.legacy_f2 and source_clip != 'forward_012'):
            raise ValueError(f'{self.clip_name} 的源片段身份不符')
        n = len(source['time'])
        if n < 3:
            raise ValueError(f'{self.clip_name} 需要至少三帧')
        fps = float(numeric('meta__sample_rate_hz', ()))
        self.duration = float(numeric('meta__duration_s', ()))
        time = numeric('time', (n,))
        if abs(fps - 1. / CONTROL_DT) > 1e-8 or (self.legacy_f2 and (n != 261 or abs(self.duration - 5.2) > 1e-8)) or (
            explicit_fps is not None and abs(explicit_fps - fps) > 1e-8
        ):
            raise ValueError(f'{self.clip_name} 源时间约定不符')
        if abs(time[0]) > 1e-9 or abs(time[-1] - self.duration) > 1e-9 or not np.allclose(
            np.diff(time), 1 / fps, atol=1e-8, rtol=0
        ):
            raise ValueError('F2 时间戳与帧数、采样率或持续时间不符')
        self.time = tensor(time, torch.float64)
        self.joint_names = tuple(array('meta__joint_names', (len(JOINT_NAMES),)).astype(str))
        if self.joint_names != tuple(JOINT_NAMES):
            raise ValueError('F2 关节名称和顺序必须与 Roban 模型一致')

        offset = numeric('meta__sole_offset_local', (2, 3))
        sole_plane = np.array([foot['sole'][2] for foot in foot_geometry()])
        if not np.allclose(offset[:, :2], 0, atol=1e-6) or not np.allclose(offset[:, 2], sole_plane, atol=1e-6):
            raise ValueError('F2 足部必须使用踝关节原点在实际足底平面上的投影')
        self._source_sole_offsets = tensor(offset)

        sides = ('left', 'right')
        root_p = numeric('root_position', (n, 3))
        root_quat = numeric('root_quaternion', (n, 4))
        q = numeric('joint_position', (n, len(JOINT_NAMES)))
        qd = numeric('joint_velocity', q.shape)
        root_v = numeric('diag__root_velocity', (n, 3))
        root_w = numeric('diag__root_angular_velocity', (n, 3))
        foot_p = np.stack([numeric(side + '_foot_position', (n, 3)) for side in sides], axis=1)
        foot_quat = np.stack([numeric(side + '_foot_quaternion', (n, 4)) for side in sides], axis=1)
        foot_v = np.gradient(foot_p, time, axis=0, edge_order=2)
        for quat in (root_quat, foot_quat):
            if not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=2e-5, rtol=0):
                raise ValueError('F2 根或足部四元数未归一化')
        for value, key in ((root_p, 'root_position'), (root_quat, 'root_quaternion'), (q, 'joint_position')):
            if not np.allclose(value[0], numeric('diag__initial_' + key, value.shape[1:]), atol=1e-6, rtol=0):
                raise ValueError(f'F2 初始参考与初始状态诊断不一致：{key}')

        contact = np.column_stack([array(side + '_contact', (n,), binary=True) for side in sides]).astype(bool)
        gait_phase = numeric('gait_phase', (n,))
        if not contact.any(axis=-1).all() or np.any((gait_phase < 0) | (gait_phase >= 1)):
            raise ValueError('F2 接触日程或全局步态相位无效')
        count = len(source['diag__step_foot_id'])
        foot_ids = array('diag__step_foot_id', (count,)).astype(str).tolist()
        lift = numeric('diag__step_lift_off_time', (count,))
        touch = numeric('diag__step_touchdown_time', (count,))
        if self.legacy_f2 and (foot_ids != ['left', 'right', 'left', 'right'] or
                               not np.allclose(touch, [1.6, 2.6, 3.6, 4.6])):
            raise ValueError('F2 的四个交替计划落脚事件与实验定义不符')
        if not (np.all((lift >= 0) & (lift < touch) & (touch <= self.duration)) and
                np.all(np.diff(touch) > 0)):
            raise ValueError(f'{self.clip_name} 抬脚和落脚时间必须有序且位于参考范围内')
        side_indices = [sides.index(side) for side in foot_ids]
        self.touchdown_side = tensor(side_indices, torch.long)
        self._phase_by_side = tensor(np.eye(2)[np.asarray(side_indices, dtype=np.int64)], torch.float64)
        self.touchdown_time = tensor(touch, torch.float64)
        self.liftoff_time = tensor(lift, torch.float64)
        event_frames, event_sides = np.where(~contact[:-1] & contact[1:])
        if event_sides.tolist() != side_indices or not np.allclose(time[event_frames + 1], touch):
            raise ValueError('F2 计划落脚事件与接触日程不一致')

        mask_fields = dict(kinematic_valid='kinematic_valid', optimization_converged='optimization_converged',
                           dynamics_feasible='dynamics_feasible', contact_kinematics_valid='dy_contact_kinematics_valid',
                           contact_and_dynamics_valid='dy_contact_and_dynamics_valid')
        validity = {key: array('diag__' + field, (n,), binary=True).astype(bool)
                    for key, field in mask_fields.items()}
        if not (validity['kinematic_valid'] & validity['optimization_converged']).all():
            raise ValueError('F2 包含运动学无效或优化未收敛的姿态')
        self.validity = {key: tensor(mask, torch.bool) for key, mask in validity.items()}
        self._source_contact = tensor(contact, torch.bool)
        # 周期相位在单圈内部做线性插值，跨圈后回到 [0,1)。
        unwrapped = np.unwrap(gait_phase * (2 * np.pi)) / (2 * np.pi)
        self._source_phase = tensor(unwrapped, torch.float64)
        self._linear_values = {key: tensor(value) for key, value in dict(
            root_p=root_p, root_v=root_v, root_w=root_w, q=q, qd=qd, foot_p=foot_p, foot_v=foot_v).items()}
        self._root_quat, self._foot_quat = tensor(root_quat), tensor(foot_quat)

        # 记录可移植的数据契约，绝对路径由训练入口记录。
        angle = float(2 * np.arccos(np.clip(abs(root_quat[0] @ root_quat[-1]), 0, 1)))
        self.validation.update(
            contact=f'{self.clip_name} 计划标签，独立记录动力学诊断，不代表实测接触', duration_s=self.duration,
            touchdown_s=touch.tolist(), touchdown_foot=foot_ids, liftoff_s=lift.tolist(),
            valid_frames={key: int(mask.sum()) for key, mask in validity.items()},
            endpoints=dict(loop=False, root_position_gap_m=float(np.linalg.norm(root_p[-1] - root_p[0])),
                           root_orientation_gap_rad=angle, joint_position_gap_rad=float(np.linalg.norm(q[-1] - q[0])),
                           root_velocity_gap_m_s=float(np.linalg.norm(root_v[-1] - root_v[0])),
                           root_angular_velocity_gap_rad_s=float(np.linalg.norm(root_w[-1] - root_w[0])),
                           joint_velocity_gap_rad_s=float(np.linalg.norm(qd[-1] - qd[0])), contact=contact[[0, -1]].tolist()))
        self.validation['files'].append(dict(
            source=self.source_path, clip=source_clip, source_fps=fps,
            control_dt=CONTROL_DT, source_frames=n, frame='world_Z_up_m', quaternion_order='wxyz',
            foot_frame='sole_projection_of_link_origin', sole_offset_local=offset.tolist()))

    def at_time(self, time):
        """按秒采样。连续量插值，接触分段保持，末帧冻结。"""
        t = torch.as_tensor(time, dtype=torch.float64, device=self.device)
        torch._assert_async(torch.isfinite(t).all(), 'F2 采样时间必须为有限数')
        t = t.clamp(0, self.duration)
        lo = (torch.searchsorted(self.time, t.contiguous(), right=True) - 1).clamp(0, len(self.time) - 2)
        hi = lo + 1
        ratio = ((t - self.time[lo]) / (self.time[hi] - self.time[lo])).clamp(0, 1)
        discrete = (torch.searchsorted(self.time, (t + 1e-9).contiguous(), right=True) - 1).clamp_min(0)
        contact = self._source_contact[discrete]

        def linear(x):
            weight = ratio.to(x.dtype).reshape(t.shape + (1,) * (x.ndim - 1))
            return torch.lerp(x[lo], x[hi], weight)

        def mask_at(mask):
            # 跨帧插值只有在两个端点都有效时才有效；恰在帧上时沿用原掩码。
            return torch.where((t - self.time[discrete]).abs() < 1e-9, mask[discrete], mask[lo] & mask[hi])

        validity = {key: mask_at(mask) for key, mask in self.validity.items()}
        swing = ((t[..., None] - self.liftoff_time) / (self.touchdown_time - self.liftoff_time)).clamp(0, 1)
        active = (t[..., None] >= self.liftoff_time) & (t[..., None] < self.touchdown_time)
        phase = ((swing * active)[..., None] * self._phase_by_side).sum(-2)
        phase = torch.where(contact, 0, phase).float()
        out = {key: linear(value) for key, value in self._linear_values.items()}
        root_quat = quat_slerp(self._root_quat[lo], self._root_quat[hi], ratio.float())
        foot_quat = quat_slerp(self._foot_quat[lo], self._foot_quat[hi], ratio.float()[..., None])
        out.update(time=t, root_quat=root_quat, root_R=quat_to_mat(root_quat),
                   foot_quat=foot_quat, foot_R=quat_to_mat(foot_quat), contact=contact,
                   contact_valid=validity['kinematic_valid'][..., None].expand_as(contact),
                   phase=phase, gait_phase=linear(self._source_phase).remainder(1).float(), validity=validity)
        return out


class F2MotionReference(GeneratedMotionClip):
    def __init__(self, path, kin, device='cpu', src_fps=None):
        super().__init__(path, kin, device, src_fps, clip_id='F2',
                         source='m3_7_1/forward/F2/reference.npz', legacy_f2=True)
