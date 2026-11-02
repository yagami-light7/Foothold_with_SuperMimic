# S54 平地课程运行手册

以下训练命令已准备供审阅，**本次没有执行 100 次或更长的训练**。每个阶段先评估，达到 REPORT.md 门槛后再手动进入下一阶段。所有路径均以项目根目录为工作目录。

```bash
cd /mnt/sage_data/workspace/KUAVO_S54_Newton

# 便利函数：使用 Isaac Lab 的现有 uv 环境。
run_s54() {
  PYTHONPATH="$PWD" uv run \
    --project /mnt/sage_data/workspace/IsaacLab_3_0 \
    --no-sync isaaclab -p "$@"
}
```

## 重新执行诊断

```bash
run_s54 tests/sim/action_authority_test.py --all_joints \
  --output reports/recheck/authority --viz none

run_s54 tests/unit/test_preflight.py
run_s54 tests/unit/test_rewards.py
run_s54 tests/sim/smoke_goal_env.py --num_envs 16 --viz none
run_s54 tests/sim/smoke_reward_env.py --num_envs 16 --steps 110 --viz none

# 独立比较控制和奖励；不更新网络。
run_s54 scripts/preflight_signals.py --stage lift --mode random \
  --std 0.15 --num_envs 32 --steps 500 \
  --output reports/recheck/lift_random.json --viz none

# 单因素消融：保持其他课程参数不变。
run_s54 scripts/preflight_signals.py --stage lift --mode random \
  --ablation no_smoothing --std 0.15 --num_envs 32 --steps 500 \
  --output reports/recheck/no_smoothing.json --viz none

# 另两个消融参数：--ablation legacy_scale / legacy_reward
run_s54 scripts/reward_landscape.py --output reports/recheck/reward_landscape
```

Authority 的单关节 offset 限定≤0.3 rad，目标仍受机器人软限位约束。它仅连接仿真；输出 controller 配置、solver torque limits、逐步 joint/sole/force 数据和 reset 标志。

## 第一阶段：较大动作权限下的站立

```bash
run_s54 scripts/train.py --stage stand \
  --num_envs 64 --max_iterations 100 --seed 42 \
  --run_name stand_v2_s42 --viz none
```

100×64×64=409,600 个训练样本。配置实际值保存到输出目录的 `env_cfg.json`、`agent_cfg.json` 和 `manifest.json`。本轮的较短测试产物在 `reports/preflight_2026-10-29/pilot_runs/`，不应作为站立成功模型直接晋级。

将训练结束打印的真实路径填入变量，然后评估：

```bash
STAND_CKPT="logs/rsl_rl/s54_foot_placement/实际输出目录/model_final.pt"
run_s54 scripts/preflight_signals.py --stage stand --mode policy \
  --checkpoint "$STAND_CKPT" --num_envs 64 --steps 1000 \
  --output reports/eval_stand_s42.json --viz none

run_s54 scripts/play.py --checkpoint "$STAND_CKPT" \
  --num_envs 16 --steps 500 --viz newton_gl
```

## 第二到第四阶段：通过门槛后迁移

```bash
# 站立门槛通过后：原地交替抬脚。
run_s54 scripts/train.py --stage lift --warm_start "$STAND_CKPT" \
  --num_envs 64 --max_iterations 100 --seed 42 \
  --run_name lift_v2_s42 --viz none

# 填入 lift 模型真实路径，先评估左右脚成功半周期。
LIFT_CKPT="logs/rsl_rl/s54_foot_placement/实际lift输出目录/model_final.pt"
run_s54 scripts/preflight_signals.py --stage lift --mode policy \
  --checkpoint "$LIFT_CKPT" --num_envs 64 --steps 1000 \
  --output reports/eval_lift_s42.json --viz none

# 抬脚门槛通过后：3 cm 前向小步。
run_s54 scripts/train.py --stage step --warm_start "$LIFT_CKPT" \
  --num_envs 64 --max_iterations 100 --seed 42 \
  --run_name step_v2_s42 --viz none

STEP_CKPT="logs/rsl_rl/s54_foot_placement/实际step输出目录/model_final.pt"
run_s54 scripts/preflight_signals.py --stage step --mode policy \
  --checkpoint "$STEP_CKPT" --num_envs 64 --steps 1000 \
  --output reports/eval_step_s42.json --viz none

# 小步门槛通过后：6 cm 连续平地足印。
run_s54 scripts/train.py --stage walk --warm_start "$STEP_CKPT" \
  --num_envs 64 --max_iterations 100 --seed 42 \
  --run_name walk_v2_s42 --viz none
```

这些 100 次更新是有界试训预算，**不保证足以学会步态**。晋级依据是存活和成功半周期指标，不能仅看 reward、air ratio 或平均前进速度。连续交替还需检查 `.jsonl` 和可视化。通过后再另行决定数百万到更大样本的预算。

## 恢复、消融与配置语义

```bash
# 同一阶段继续：保留 actor、critic、optimizer 和 iteration。
run_s54 scripts/train.py --stage lift --resume "$LIFT_CKPT" \
  --num_envs 64 --max_iterations 100 --seed 42 \
  --run_name lift_v2_continue_s42 --viz none

# LR 与归一化预热只作独立消融，不同时改多个因素。
run_s54 scripts/train.py --stage lift --num_envs 64 --max_iterations 3 \
  --learning_rate 0.0003 --normalization_warmup_steps 0 \
  --seed 42 --run_name lift_lr_ablation_s42 --viz none
```

- 新课程默认：LR=1e-4 fixed，std=0.15，rollout=64，entropy=0.002，归一化预热=0。
- `--warm_start` 只转移 actor；跨阶段重置 critic 和 optimizer。`--resume` 要求同阶段。
- `play.py` 优先恢复 checkpoint 同目录的 task 配置；独立复制的 `model_final.pt` 也包含配置。显式 `--stage` 是手动覆盖，应谨慎使用。
- `--stage legacy` 供旧 baseline 对照。新课程的逐关节 scale 会覆盖标量 `action_scale`；旧模型直接用新 scale 回放不构成公平比较。
- 关闭 GUI 用 `--viz none` 训练；可视化用于检查足序、支撑、躯干和滑移。
- 目前没有速度命令、停止或转向课程，不应把 `walk` 解释为已具备这些能力。
