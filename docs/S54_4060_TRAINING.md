# RTX 4060 8GB：约4GB显存训练配置

入口：`tools/train_s54_4060.sh`。已在本机RTX4060 Laptop完成30次迭代实测，
正式20万次训练尚未启动。保留Isaac Lab Kit-less、Newton/MJWarp、PyTorch WM+BPTT+Critic。
复用现有`supermimic_s54`容器和镜像，不改变GPU功耗限制，不影响原G1。

## 默认配置

| 参数 | 默认值 |
|---|---:|
| num_envs | 256 |
| buffer_steps / warmup_steps | 1024 / 256 |
| wm_batch / wm_batches / wm_window | 2048 / 4 / 8 |
| pi_batch / horizon | 576 / 32 |
| goal_frames / hist | 10 / 10 |
| iterations | 200000 |
| wm_only_iters | 1000 |
| utd / lr | 0.4 / 0.0002 |
| ckpt_every / log_every | 2000 / 100 |
| amp / compilation | BF16 / eager (`--no_compile`) |

每个replay时间步含256条并行transition。正式训练前1000次仅更新WM；
smoke使用相同环境数、batch和horizon，仅将迭代数设为30、WM-only设为10。
增大batch增加有效训练样本，4次WM更新增加世界模型拟合量；没有加入占显存或烧功耗的空操作。
batch与horizon属于训练超参数，长期收敛仍需用固定协议evaluation验证。

## 实测结果（2026-10-26）

| 配置 | 总显存采样最大值 | GPU利用率采样平均/最大 | 功耗采样平均/最大 |
|---|---:|---:|---:|
| WM1024×2 / PI512 / H32 | 3718MiB | 59.9% / 100% | 47.6W / 57.7W |
| **默认：WM2048×4 / PI576 / H32** | **3948MiB（3.86GiB）** | **78.6% / 100%** | **56.9W / 66.2W** |

表中每500ms采样一次，平均值仅取总显存>1000MiB的样本，包括相关初始化/保存开销，
不是长时间稳态测量。默认配置最大采样温度63℃；Torch峰值allocated3319MiB、reserved3622MiB。
Torch数据不含Warp，不能把它当作总显存。采样也可能错过瞬时峰值。
驱动报告power.limit为N/A，因此不能声称已达到额定满功耗；实测已经有100%利用率的计算区间。
大batch提高了持续负载，实际功耗仍受笔记本供电、散热与驱动控制。

默认配置第30次：wm_loss=0.6649、pi_loss=-0.2526、v_loss=0.0038、pi_gradnorm=4.39665；
全部损失/梯度finite，WM/Policy/Critic checkpoint检查通过。末10次约1.46iteration/s，
这是短测试速度，不能保证长期速度或跟踪质量。只测试30次，不宣称20万次稳定收敛。

证据：

- `logs/s54_4060_4gb_probe2/args.json`、`train_log.csv`、`runtime_stats.json`
- `logs/s54_4060_4gb_probe2/ckpt_000030.pt`、`last.pt`
- `results/s54/4060_4gb_probe2.log`、`4060_4gb_probe2_gpu.csv`

## 启动、观察和停止

宿主机新项目根目录执行。当前用户已在docker组，但会话未继承时用`sg docker`。

```bash
cd /mnt/sage_data/workspace/supermimic_s54
# 只打印参数，不启动训练
bash tools/train_s54_4060.sh --dry-run train
# 正式训练：新日志目录logs/s54_4060，后台tmux运行
sg docker -c 'bash tools/train_s54_4060.sh train'
tail -f logs/s54_4060/stdout.log
```

如需自行重复显存验证，使用新名字；此前结果受保护：

```bash
sg docker -c 'bash tools/train_s54_4060.sh smoke s54_4060_check2'
nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv -l 1
```

脚本检查新repo mount、至少6000MiB空闲显存、无现有GPU compute任务；
拒绝原G1容器和已有输出目录。正式训练保存每2000次的编号checkpoint与可恢复的last.pt。
每次checkpoint同时记录runtime_stats.json。默认100个编号checkpoint约7GiB，另有last.pt和日志。

只中断本次S54训练，保留最近保存的checkpoint：

```bash
sg docker -c 'docker exec supermimic_s54 tmux send-keys -t s54_4060 C-c'
```

## 断点恢复与参数覆盖

恢复必须用last.pt（含三个优化器与Critic）；编号checkpoint仅用于评估。
为保留原日志，恢复输出到新目录。ITERATIONS是总目标，不是额外次数。

```bash
sg docker -c 'ITERATIONS=300000 bash tools/train_s54_4060.sh resume s54_4060_resume logs/s54_4060/last.pt'
# 若希望先跑2000次观察收益，再决定是否继续：
sg docker -c 'ITERATIONS=2000 CKPT_EVERY=500 bash tools/train_s54_4060.sh train s54_4060_pilot'
```

支持NUM_ENVS、BUFFER_STEPS、WARMUP_STEPS、WM_BATCH、WM_BATCHES、PI_BATCH、HORIZON、
ITERATIONS、WM_ONLY_ITERS、CKPT_EVERY、LOG_EVERY、LR、SEED、MOTION_DIR环境变量。
从30次smoke恢复也可以，但正式学习建议新初始化；30次产物用于接口验证。

## 训练后评估与视频

训练完成或停止后再评估，避免两个GPU任务争抢显存：

```bash
sg docker -c 'docker exec supermimic_s54 python supermimic/evaluate.py \
  --run_dir logs/s54_4060 --motion_dir data/retarget_data/reference \
  --ckpt logs/s54_4060/ckpt_200000.pt --record logs/s54_4060/traj_200000.npz'
sg docker -c 'docker exec -e MUJOCO_GL=egl supermimic_s54 python tools/visualization/render.py \
  logs/s54_4060/traj_200000.npz logs/s54_4060/dance_200000.mp4 \
  --clips dance1_subject1 --seconds 10 --width 960 --height 540 --fps 30'
```

中途评估请换成实际已存在的编号checkpoint。关注norm_return、norm_return_framew和n_failed，
不要仅用显存、功耗或imag_rew判断学习是否成功。
当前训练库仅dance1_subject1；扩大迭代和并行数仍是单动作训练，不代表扩大动作数据集。
