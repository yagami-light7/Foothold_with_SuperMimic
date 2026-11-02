"""回放时绘制目标足印：只更新渲染标记，不参与碰撞、控制或奖励计算。"""

from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply, quat_mul

from .foot_frames import HEEL_LOCAL_POS, TOE_LOCAL_POS
from .heading import quat_from_yaw

if TYPE_CHECKING:
    from .foot_placement_env import S54FootPlacementEnv


class FootstepVisualizer:
    """青色为左脚目标，橙色为右脚目标；亮色为摆动脚，暗色为支撑脚。

    足印轮廓内的箭头表示目标脚尖方向，中心圆点表示目标 sole 原点。
    紫色圆点表示当前摆动相位的高度参考，不表示已经规划了完整的足端轨迹。
    """

    def __init__(self, env: "S54FootPlacementEnv") -> None:
        self.env = env
        device = env.device
        # 轮廓长度取 toe/heel 标记间距；宽度仅用于示意，不是精确碰撞模型。
        half_length = (TOE_LOCAL_POS[0] - HEEL_LOCAL_POS[0]) / 2
        half_width = 0.045
        # 四条边加三条箭头线段；全部以目标 sole 原点为中心。
        segments = torch.tensor([
            [[-half_length, half_width], [half_length, half_width]],
            [[-half_length, -half_width], [half_length, -half_width]],
            [[half_length, -half_width], [half_length, half_width]],
            [[-half_length, -half_width], [-half_length, half_width]],
            [[-0.030, 0.0], [0.065, 0.0]],
            [[0.065, 0.0], [0.040, 0.018]],
            [[0.065, 0.0], [0.040, -0.018]],
        ], device=device)
        delta = segments[:, 1] - segments[:, 0]
        self.local_centers = torch.zeros(7, 3, device=device)
        self.local_centers[:, :2] = segments.mean(dim=1)
        self.local_rotations = quat_from_yaw(torch.atan2(delta[:, 1], delta[:, 0]))
        self.line_scales = torch.ones(7, 3, device=device) * 0.004
        self.line_scales[:, 0] = torch.linalg.vector_norm(delta, dim=-1)
        colors = {
            "left_swing": (0.0, 0.9, 1.0),
            "right_swing": (1.0, 0.45, 0.0),
            "left_stance": (0.0, 0.28, 0.4),
            "right_stance": (0.4, 0.18, 0.0),
        }
        self.outlines = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/Footstep_targets",
            markers={name: sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ) for name, color in colors.items()},
        ))
        self.centers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/Footstep_centers_and_height",
            markers={name: sim_utils.SphereCfg(
                radius=0.008,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ) for name, color in {
                "left_center": colors["left_swing"],
                "right_center": colors["right_swing"],
                "swing_height_reference": (0.9, 0.1, 1.0),
            }.items()},
        ))
        self.updates = 0

    @torch.no_grad()
    def update(self) -> None:
        """在执行动作前读取当前目标，绘制本次奖励实际使用的足印与相位。"""
        sampler = self.env.goal_sampler
        positions = torch.stack((sampler.left_target_pos_w, sampler.right_target_pos_w), dim=1)
        rotations = torch.stack((sampler.left_target_quat_w, sampler.right_target_quat_w), dim=1)
        count = self.env.num_envs
        # 必须按 [环境, 左右脚, 线段] 排列，Newton 才能正确筛选可见环境。
        q = rotations[:, :, None, :].expand(-1, -1, 7, -1)
        offset = self.local_centers.expand(count, 2, -1, -1)
        translations = positions[:, :, None, :] + quat_apply(q.reshape(-1, 4), offset.reshape(-1, 3)).reshape(count, 2, 7, 3)
        # 只把绘图抬高 6 mm，避免与地面重叠闪烁；不修改 sampler 的真实目标。
        translations[..., 2] += 0.006
        orientations = quat_mul(q.reshape(-1, 4), self.local_rotations.expand(count, 2, -1, -1).reshape(-1, 4))
        left = sampler.left_is_swing
        prototypes = torch.stack((torch.where(left, 0, 2), torch.where(left, 3, 1)), dim=1)
        env_ids = torch.arange(count, device=positions.device)
        self.outlines.visualize(
            translations=translations.reshape(-1, 3), orientations=orientations,
            scales=self.line_scales.expand(count, 2, -1, -1).reshape(-1, 3),
            marker_indices=prototypes[:, :, None].expand(-1, -1, 7).reshape(-1),
            environment_ids=env_ids[:, None].expand(-1, 14).reshape(-1),
        )
        swing_target = torch.where(left[:, None], positions[:, 0], positions[:, 1]).clone()
        local_phase = torch.where(left, sampler.phase * 2, (sampler.phase - 0.5) * 2).clamp(0, 1)
        # v2只规定抬脚窗口及最低净空；旧奖励仍显示原来的正弦Z参考。
        reward_cfg = self.env.cfg.reward_cfg
        from configs.s54_flat_reward import FlatWalkRewardCfg
        if isinstance(reward_cfg,FlatWalkRewardCfg):
            height = ((local_phase-.5).abs() <= 2*reward_cfg.swing_window_half_width).float()
        else:
            height = torch.sin(torch.pi*local_phase)
        swing_target[:, 2] += reward_cfg.swing_height*height
        points = torch.cat((positions, swing_target[:, None]), dim=1)
        points[..., 2] += 0.009
        self.centers.visualize(
            translations=points.reshape(-1, 3),
            marker_indices=torch.arange(3, device=positions.device).expand(count, -1).reshape(-1),
            environment_ids=env_ids[:, None].expand(-1, 3).reshape(-1),
        )
        self.updates += 1
