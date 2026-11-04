# S54 训练启动器、动作筛选与 SwanLab

根目录 `tools/train/run_train.sh` 包含完整启动逻辑；原 `tools/train_s54.sh` 已移至根目录并替换旧入口。
保留 Isaac Lab 3.0 Kit-less、Newton/MJWarp、PyTorch World Model/BPTT 路线。

## 配置

编辑 `tools/train/run_train.sh` 顶部的变量：

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `MOTION_DIR` | `data/retarget_data/reference_startup_fixed` | 已验收 Reference 目录 |
| `CLIPS` | `("dance1_subject1")` | 参与训练的 clip 名称；空数组表示全部 |
| `SWANLAB_ENABLED` | `true` | 启用实验记录 |
| `SWANLAB_MODE` | `online` | 同步云端；`offline` 用于本地记录 |
| `SWANLAB_PROJECT` | `SuperMimic` | 云端项目 |
| `SWANLAB_WORKSPACE` | 空 | 使用默认工作空间；可填写团队空间 |
| `SWANLAB_EXPERIMENT` | 空 | 自动使用本次训练名称 |
| `SWANLAB_POLL_SECONDS` | `5` | 读取新训练日志的间隔 |

训练超参数保持本次修改前的设置：2048 env、10000 iterations、WM 1024×16、
Policy batch 256、horizon 32、BF16。`smoke` 使用同样的环境数与 batch，缩短为 30 iterations。

## 登录与启动

复用宿主机 `.venv-swanlab` 和已构建的 S54 Docker 镜像。若没有宿主机 SDK：

```bash
python3 -m venv .venv-swanlab
.venv-swanlab/bin/python -m pip install 'swanlab>=0.10.2,<0.11'
```

尚未登录时执行一次：

```bash
.venv-swanlab/bin/swanlab login
```

也可使用环境变量 `SWANLAB_API_KEY`；密钥无需写进训练脚本。
登录方式见 [SwanLab 官方文档](https://docs.swanlab.cn/api/cli-swanlab-login.html)。
启动器会在开始 GPU 训练前检查 SDK 并验证云端登录。

在项目根目录运行：

```bash
# 查看本次参数和选择的动作，不启动训练或访问云端。
bash tools/train/run_train.sh --dry-run train s54_dance_cloud --clips dance1_subject1

# 训练指定动作，并同步到云端。
bash tools/train/run_train.sh train s54_dance_cloud --clips dance1_subject1

# 多动作；每段都需要已经成功生成 Reference。
bash tools/train/run_train.sh train s54_dance_walk --clips dance1_subject1 walk1_subject1

# 选择目录顶层的全部合格 CSV。
bash tools/train/run_train.sh train s54_all_clips --all-clips

# 指定云端项目、团队空间和实验名称。
bash tools/train/run_train.sh train s54_team_run --clips 'dance*' \
  --swanlab-project SuperMimic --swanlab-workspace TEAM --swanlab-name 'S54 Dance'

# 30 iteration smoke / 从已有 Checkpoint 续训到脚本配置的总 iteration。
bash tools/train/run_train.sh smoke s54_cloud_smoke --clips dance1_subject1
bash tools/train/run_train.sh resume s54_cloud_resume logs/s54_dance_cloud/last.pt --clips dance1_subject1
```

从已完成的 10000 次继续到总计 20000 次（追加 10000 次，其他超参数不变）：

```bash
bash tools/train/run_train.sh resume s54_motion_20k logs/s54_motion/last.pt \
  --iterations 20000 --clips dance1_subject1 aiming1_subject1
```

`--iterations` 是总目标，必须大于 Checkpoint 已完成的 iteration。
完整续训使用 `last.pt`，它包含 Critic 与优化器；编号 `ckpt_*.pt` 用于评估。
恢复后学习率按原来的绝对 iteration 延续，Replay 会重新采集填充，
已超过 `wm_only_iters` 时直接继续 Policy/Critic 更新。日志写入新的运行目录。

`--clips` 接收名称或加引号的通配符，可选 `.csv` 后缀。每个名称/模式都必须匹配；
重复选择会去重。只搜索顶层 CSV，不递归读取 `raw/`、`.staging/` 或 `rejected/`。
训练加载器检查完整数据契约、根部运动连续性、CSV 哈希和足底/自碰撞证书。
旧 `reference_safe` 中部分 clip 的启动瞬移会被拒绝；使用修复后的默认目录。
新训练增加头部角度与角速度跟踪项，并将奖励配置保存到 `args.json` 和 Checkpoint。
历史实验评估保留当时的奖励定义，便于比较，详见 [修复报告](../reports/s54_startup_head_fix.md)。
实际解析后的 clip 名称会写入本次 `args.json`、Checkpoint 配置和 SwanLab 配置。

启动器核对项目挂载后，会自动启动处于 `exited` 或 `created` 状态的现有 S54 容器，
然后检查 GPU 和 SwanLab。运行中的容器会直接复用。
若当前终端尚未继承 docker 组，可用 `sg docker -c 'bash tools/train/run_train.sh ...'`。
每个实验必须使用新的 `s54_*` 名称，避免覆盖已有日志与 Checkpoint。

## 日志与同步

`train`/`resume` 在 S54 容器的 tmux 会话中运行；`smoke` 在前台运行。
宿主机的 SwanLab 进程使用 `nohup setsid` 在独立进程会话中跟随同一训练目录，终端关闭后继续同步。
训练结束会写入 `train.exitcode`，同步进程补齐最后的 CSV 记录后退出。
训练失败或 tmux/容器消失时，同步进程会记录已有指标并报告异常状态。

每次实验的文件位于 `logs/s54_RUN/`：

| 文件 | 用途 |
| --- | --- |
| `stdout.log` | 训练输出 |
| `args.json` | 超参数与实际参与训练的 clips |
| `train_log.csv` | 本地训练指标 |
| `swanlab_sync.log` | 云端实验链接和同步进度/错误 |
| `swanlab_sync_status.json` | 最近同步的 iteration 与同步状态 |
| `swanlab/` | SDK 本地实验记录 |
| `train.exitcode` | 训练退出码；`0` 表示正常结束 |

上传的指标包含 WM/Policy/Critic loss、仿真与想象 reward、episode return、
终止比例、策略梯度与存活率、训练速度、模拟步数、耗时与学习率。
`online` 的云端行为见 [SwanLab init 官方文档](https://docs.swanlab.cn/api/py-init.html)。

本地排查可加 `--swanlab-mode offline`，关闭记录可加 `--no-swanlab`。
如需手动补传已有训练日志，以下命令上传一次当前快照：

```bash
.venv-swanlab/bin/python tools/ops/sync_swanlab.py \
  --run_dir logs/s54_dance_cloud --project SuperMimic
```

## 验证

```bash
bash -n tools/train/run_train.sh tools/train/run_s54_job.sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/tests/test_s54_launcher.py
.venv-swanlab/bin/python tools/ops/sync_swanlab.py --check --mode offline
```

回归覆盖动作选择与目录隔离、Dry run、训练退出码、CSV 部分写入、重复 step、
不同运行日志隔离，以及正常/失败训练的同步收尾。云端实际连接由启动时的登录检查验证。

## 2026-10-28 修复后续训

已验证原配置 30 次 Smoke，完整 aiming 与 dance 均无失败。当前正式运行是
`logs/s54_motion_15k_fixed`，从原始 10000 次 `last.pt` 继续到总计 15000 次。
头部姿态奖励已加入，训练和云端新增 `Head/RMSErrorDeg`、`Head/NormalizedPenalty`。
头部回正需要后续学习，参考 [修复报告](../reports/s54_startup_head_fix.md)。

续接同一个云端实验使用 `SWANLAB_RUN_ID=1ohsjt5p SWANLAB_RESUME=must`；
同步器显式传递 `id` 与 `resume`，ID 不存在会报错。新日志目录独立保存，原 10000 次权重保持原样。
查看当前训练：

```bash
docker exec -it supermimic_s54 tmux attach -t s54_motion_15k_fixed
tail -f logs/s54_motion_15k_fixed/stdout.log
```

旧 `reference_safe` 的 43 段中有 22 段根部启动检查失败。继续生成应使用新默认目录，
不要把旧 CSV 直接复制进修复目录。导出进度与缺失名单见 [导出审计](../reports/s54_retarget_inventory_20261028.md)。
