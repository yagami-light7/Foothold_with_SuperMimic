# 首步XY倍率2.0：12小时续训入口

> 结果更新（2026-11-02）：本轮已完成；以下为启动前配置记录。实际结果见[12小时训练结果报告](../walk_retention_12h_results_2026-11-02/REPORT.md)。

按用户要求准备，未启动训练。新配置：`configs/walk_retention_12h.json`；入口：`scripts/train_walk_retention_12h.sh`。

```bash
cd /mnt/sage_data/workspace/KUAVO_S54_Newton
bash scripts/train_walk_retention_12h.sh
```

从 `logs/walk/2026-11-02_00-32-11_walk_retention_30m/training/00000_retention/model_interrupted.pt` 完整恢复actor、critic、优化器和归一化，新建独立实验，不使用旧目录的resume-dir更改奖励。

唯一奖励改动：`first_swing_xy_multiplier: 1.5 → 2.0`。常规摆动倍率仍为2，因此首步最终倍率从3变为4；首步之后的奖励沿用原配置。旧半小时JSON保留，以便对照。

时长12h；2048环境；rollout64、mini-batches4、epochs4；有效LR=2e-5 fixed；episode10s；随机左右起步；15格固定配额、每格至少41环境、另205环境固定40cm/2Hz；Action Scale、PD和其余奖励沿用。所有自动eval关闭，SwanLab online/compact。输出：`logs/walk/<时间>_walk_retention_firstxy2_12h/`。

每250次PPO更新保存checkpoint，时间截止另存interrupted；50000是单进程及更新上限，12h墙钟预算优先，保存/上传退出可能额外耗时。支持`--checkpoint FILE`、`--hours 10`、`--dry-run`；`--resume-dir DIR`仅用于恢复本次已创建的实验并继承其剩余预算。

已通过shell语法、JSON/schema和CLI dry-run校验；核对新旧计划仅首步倍率不同、起点存在、12h预算、有效学习率、15格配额、无eval与在线上传设置。未运行GPU训练或eval；dry-run证据见本目录`dry_run.json`。此改动尚无训练结果。
