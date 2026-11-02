"""S54 foot-link to sole-frame transformations.

The reference sole frame is defined at the midpoint
between the central toe and heel markers in the S54 URDF.

Quaternion convention: xyzw.
"""

import torch

from isaaclab.utils.math import combine_frame_transforms, quat_apply


TOE_LOCAL_POS = (0.170840, 0.0, -0.0595)
HEEL_LOCAL_POS = (-0.073164, 0.0, -0.0595)

SOLE_LOCAL_POS = (
    0.5 * (TOE_LOCAL_POS[0] + HEEL_LOCAL_POS[0]),
    0.0,
    -0.0595,
)


def get_sole_pose(
    foot_link_pos_w: torch.Tensor,
    foot_link_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert foot-link world pose to sole-frame world pose.

    Args:
        foot_link_pos_w: [N, 3]
        foot_link_quat_w: [N, 4], xyzw

    Returns:
        sole_pos_w: [N, 3]
        sole_quat_w: [N, 4], xyzw
    """

    num_envs = foot_link_pos_w.shape[0]

    offset = foot_link_pos_w.new_tensor(
        SOLE_LOCAL_POS
    ).unsqueeze(0).expand(num_envs, -1)

    sole_pos_w, sole_quat_w = combine_frame_transforms(
        foot_link_pos_w,
        foot_link_quat_w,
        offset,
    )

    return sole_pos_w, sole_quat_w


def get_sole_velocity(foot_link_quat_w, foot_link_linear_velocity_w, foot_link_angular_velocity_w):
    """足底点速度=v_link+ω×r；不能把质心速度或两帧差分直接当成脚底速度。"""
    local = foot_link_linear_velocity_w.new_tensor(SOLE_LOCAL_POS).expand_as(foot_link_linear_velocity_w)
    offset_w = quat_apply(foot_link_quat_w,local)
    return foot_link_linear_velocity_w+torch.cross(foot_link_angular_velocity_w,offset_w,dim=-1)
