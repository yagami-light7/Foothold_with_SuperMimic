# 工具入口

所有命令从仓库根目录运行。`tools/` 根目录只保留包入口、本文及迁移索引。

| 目录 | 用途 | 常用入口 |
|---|---|---|
| `train/` | 训练、恢复、课程及消融命令生成 | `run_s54_stage1_walk.sh`、`run_s54_motion_prior.py` |
| `tests/` | CPU单元/回归测试及fixtures | `python -m unittest discover -s tools/tests -t . -v` |
| `validation/` | Newton验证、评估、诊断及导出 | `evaluate_s54_motion_prior.py`、`test_env_s54.py` |
| `data/` | 数据准备、重定向、模型转换与QA产物生成 | `prepare_s54_motion_prior.py`、`generate_s54_reference.py` |
| `visualization/` | 播放、渲染、曲线与参考轨迹预览 | `play_s54_foot_tracking.py`、`render_s54_motion_prior.py` |
| `ops/` | SwanLab同步、监控与进程监督 | `sync_swanlab.py` |
| `benchmarks/` | 显存/吞吐基准 | `benchmark_s54_fresh.py` |
| `setup/` | 环境与数据下载 | `setup.sh`、`download_data.sh` |

需要GPU的 `test_env*`、`test_*_newton`、reset/spawn检查放在 `validation/`，不随CPU unittest discovery启动。

Stage 1当前推荐入口：

```bash
RUN="s54_stage1_walk_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
bash tools/train/run_s54_stage1_walk.sh start "$RUN"
tmux attach -t "$RUN"
```

默认配置：[stage1_walk_3090.json](../configs/motion_prior/stage1/stage1_walk_3090.json)。参数与恢复方法见 [Stage 1训练说明](../docs/experiment/STAGE1_WALK_TRAINING.md)。根目录 `run_train.sh` 是旧Docker入口的兼容转发，实际脚本及旧参数在 `tools/train/run_train.sh`。

2026-11-05迁移了91个脚本、23个JSON配置。[layout_migration.json](layout_migration.json)列出每个旧路径与新路径。脚本的交叉导入、shell根目录定位及运行文档同步更新；历史 `logs/`、`results/`、`reports/`、checkpoint及源码快照不移动。旧checkpoint中的配置路径由 `supermimic/config_paths.py`映射到新位置；历史实验记录保留原始命令。
