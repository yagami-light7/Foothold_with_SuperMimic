# S54 迁移检查单

先后依赖：MJCF/CSV 名称契约 → 无内置地面的仿真 MJCF → USD importer →
PyTorch FK → MotionLib/MDP 维度 → articulation 名称映射 → WM/BPTT/critic → evaluation/视频。

- [x] M0：模型检查、27→29、30 Hz metadata、范围/连续性统计
- [x] M1：保留 Leju-GMR 运动学，显式碰撞 mask，导出并验证 USD
- [x] M2 CPU：固定 link/关节锚点 FK，配置维度，checkpoint 身份检查；Newton 集成已通过
- [x] M3：独立容器与新 repo mount，保护旧 G1
- [x] M4：CPU/MuJoCo 对照 → Newton 1 env → 16 env → 30 iterations
- [x] M5：完整 checkpoint evaluation、真实轨迹和对比视频

初始检查：robot.py 硬编码 G1 名称、motor、初始姿态和 USD；kinematics.py
忽略 fixed bodies 且未处理 joint anchor/qpos0；motion.py 未验证 36 列，固定切片；
mdp.py/env.py/buffer.py/train.py/evaluate.py 固定关节、context、goal 与 critic 维度；
env.py 假设 simulator 顺序恰好等于 CSV；render.py 固定 pelvis 和两个 36-qpos 模型；
两份 shell 脚本固定旧容器名。原始代码快照保存在 `.cache/port_baseline/`。

现场：已复用 supermimic:latest 创建独立 supermimic_s54，mount 指向新 repo。
当前用户通过 sg docker 使用已登记的 docker 组；沙盒外 CUDA 与容器内 CUDA 均正常。
端到端入口退出0，30 iteration checkpoint / evaluation / NPZ / 150帧视频均已验证。
阶段最终状态、命令与证据见 S54_PORTING_REPORT.md。
