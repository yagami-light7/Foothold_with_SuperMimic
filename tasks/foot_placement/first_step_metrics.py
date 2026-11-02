"""首个摆动半周期的物理指标；不改变奖励，不把提前摔倒从成功率分母中剔除。"""

import torch


class FirstStepMetrics:
    def __init__(self, num_envs, device):
        self.air = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.touchdown = torch.zeros_like(self.air)
        self.support = torch.ones_like(self.air)
        self.xy_sum = torch.zeros(num_envs, device=device)
        self.xy_count = torch.zeros_like(self.xy_sum)

    def reset(self, ids):
        self.air[ids] = False
        self.touchdown[ids] = False
        self.support[ids] = True
        self.xy_sum[ids] = 0
        self.xy_count[ids] = 0

    def observe(self, first, local, swing_fz, stance_fz, lift, xy, valid, threshold, height):
        middle = first & valid & (local > .2) & (local < .8)
        late = first & valid & (local >= .8)
        self.air |= middle & (swing_fz < threshold) & (lift > max(.01, .6 * height))
        self.touchdown |= late & (swing_fz > threshold) & (xy < .03)
        self.support &= ~middle | (stance_fz > threshold)
        self.xy_sum += xy * late
        self.xy_count += late

    def finish(self, finished, valid, started_left):
        # 成功：中段离地、末段落点3cm内接触、摆动中段始终有支撑，且没有提前结束。
        success = valid & self.air & self.touchdown & self.support
        result = {}
        for side, start in (("left", started_left), ("right", ~started_left)):
            selected = finished & start
            if selected.any():
                result[f"Startup/{side}_success"] = success[selected].float()
            measured = selected & (self.xy_count > 0)
            if measured.any():
                result[f"Startup/{side}_xy_error_m"] = self.xy_sum[measured] / self.xy_count[measured]
        return result


def summarize_first_steps(sums, counts):
    """无末段样本时输出None和计数，避免将提前失败误报为零误差。"""
    return {side: {
        "count": counts.get(f"Startup/{side}_success", 0),
        "success_rate": (sums.get(f"Startup/{side}_success", 0) / counts[f"Startup/{side}_success"]
                         if counts.get(f"Startup/{side}_success", 0) else None),
        "late_xy_error_m": (sums.get(f"Startup/{side}_xy_error_m", 0) / counts[f"Startup/{side}_xy_error_m"]
                            if counts.get(f"Startup/{side}_xy_error_m", 0) else None),
        "xy_measured_count": counts.get(f"Startup/{side}_xy_error_m", 0),
    } for side in ("left", "right")}
