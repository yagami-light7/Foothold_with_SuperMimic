# SuperMimic KUAVO S54

S54 迁移入口：[README_S54.md](README_S54.md)。真实测试状态、关节映射、假设与阻塞见
[迁移报告](docs/S54_PORTING_REPORT.md)。

保留 Isaac Lab 3 Kit-less + Newton/MJWarp + 可微 PyTorch World Model / BPTT Policy / Critic。
已在独立 S54 容器完成 CPU/FK、Newton 1/16 环境、30 iteration WM+BPTT 训练、
checkpoint 评估、NPZ 与 5 秒对比视频。评估 return=0.0006181、失败1/1；仅证明单动作流程跑通，未达到跟踪性能目标。

上游算法：[zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)。
原 G1 数据、checkpoint、容器和原项目未修改。
