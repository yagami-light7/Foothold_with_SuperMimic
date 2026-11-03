# 中断后的SwanLab曲线补传

适用场景：`train_4060.py`监督器退出，`train.py`仍在运行，本地PPO日志持续增长，但云端曲线停止。

训练数据主要在 `training/<chunk>/metrics.jsonl` 和 TensorBoard 的 `events.out.tfevents.*`。原 `swanlog/` 只包含监督器退出前接收到的数据；直接对旧 `swanlog/` 执行 `swanlab sync`，不能生成监督器退出后尚未记录到其中的曲线。

## 补传到原实验

在服务器项目根目录执行。先确保 [replay_swanlab.py](../scripts/replay_swanlab.py) 已同步到服务器；脚本复用仓库已有的 `scripts/swanlab_metrics.py`。

先预检，不联网、不启动训练、不创建输出目录：

```bash
python3 scripts/replay_swanlab.py \
  --run-dir logs/walk/2026-11-02_22-38-25_3090 \
  --start-step 142 \
  --dry-run
```

检查输出的 `run_id` 为 `5h4q90di`、`project` 为 `S54-Locomotion`，以及 `first_step`、`last_step`、`records_to_submit` 是否符合预期。然后补传：

```bash
.venv-swanlab/bin/python -u scripts/replay_swanlab.py \
  --run-dir logs/walk/2026-11-02_22-38-25_3090 \
  --start-step 142
```

实验ID默认从 `state.json` 读取，项目和工作空间默认从 `launcher.json` 读取；SDK使用 `resume="must"`，不会在原实验找不到时静默新建一条实验，也不覆盖原训练配置。

不传 `--start-step` 时，默认从 `state.json` 的 `updates+1` 开始。本例监督器停在141，因此默认也会从142开始。该计数表示监督器处理过多少行，不保证每一条都已经到达云端；若需要核查前段缺口，可使用下面的新建完整曲线方式。

## 单独建立一条完整恢复曲线

```bash
.venv-swanlab/bin/python -u scripts/replay_swanlab.py \
  --run-dir logs/walk/2026-11-02_22-38-25_3090 \
  --new-run
```

这会新建名字以 `_recovered` 结尾的实验，默认从Step 1重放全部本地PPO记录，保留原实验。原实验的普通resume接口会跳过已存在指标早于或等于其最新记录的Step，因此它适合追加尾段；不要依靠它修复已上传区间内部的空洞。

## 参数

| 参数 | 默认值 / 含义 |
|---|---|
| `--run-dir DIR` | 必填；原训练实验根目录 |
| `--start-step N` | 正整数，包含此Step；原实验默认 `updates+1`，新实验默认1 |
| `--end-step N` | 包含此Step；默认补到启动时的本地日志快照末尾 |
| `--new-run` | 新建完整恢复曲线，不修改原实验 |
| `--project NAME` | 默认沿用 `launcher.json`；缺省时为 `S54-Locomotion` |
| `--workspace NAME` | 默认沿用原工作空间 |
| `--detail compact\|full` | 默认compact，复用现有精简曲线筛选与命名 |
| `--print-every N` | 默认250，每提交N条打印一次进度 |
| `--dry-run` | 只校验数据和预览，不导入SwanLab SDK、不联网 |

## 数据范围和状态

- Step按 `training/*/metrics.jsonl` 的chunk顺序和完整记录数累计，保持原监督器的横轴；不直接使用包含旧checkpoint历史的RSL iteration编号。
- 固定启动时各文件的字节边界，跳过最后未写完整的一行。训练仍在追加时，本次不会无限追赶；后续新增数据可再次补传。
- 默认恢复奖励、归一化回报、episode长度、首步、PPO损失/学习率/动作噪声等核心训练曲线。完整诊断可用 `--detail full`。
- 不修改 `state.json`、`launcher.json`、原始日志或模型，不启动GPU仿真、训练或eval。
- 监督器退出后没有记录的GPU监控、没有执行的eval等无法重建；脚本不会伪造这些曲线。
- 补传记录的云端写入时间是补传时间；原始采集时间保留在JSONL中，原始墙钟曲线不会自动重现。
- SDK自己的补传日志写在 `swanlog_replay/`。网络异常时可以重试；普通resume会跳过相应指标已存在的旧Step。终端“已提交”指提交给SDK，不等于服务器已确认全部接收；完成后应核对云端最后Step。
- 恢复原实验时，若全部训练段的manifest均为complete，则以success结束；否则以aborted结束，保留训练中断的语义。新建的补传任务正常结束记为success，不代表policy已通过物理评估。

若训练已停止，建议先完整补传，再评估最新checkpoint。补传不会使监督器重新接管训练，也不会修复已经失效的12小时时限。
