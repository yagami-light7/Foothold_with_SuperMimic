# KUAVO S54 SuperMimic

Motion→Foot 直接迁移、可选配对 BPTT Bridge、上身先验与有界实验结果见
[MotionPrior_FootTrajectory 实验记录](experiment/MotionPrior_FootTrajectory.md#12-实施与实验记录2026-11-04)。
新入口通过接口 smoke；真实 Standing Gate 仍未通过。

足部轨迹与可切换 Goal 接口见 [S54_FOOT_TRAJECTORY.md](docs/S54_FOOT_TRAJECTORY.md)。
足部模式默认使用未来 1 秒的 10 个双脚目标，共 230 维；保留原 Motion 模式作为默认。
独立 Foot MDP 与 Standing 最小训练见 [训练前检查报告](docs/S54_FOOT_TRACKING_TRAINING.md)。
Standing 第 96 步失败的配对诊断、实际接触证据和有界 3090 配置见 [STANDING_DIAGNOSIS.md](STANDING_DIAGNOSIS.md)。
Foot 训练已去除 Motion 数据依赖；短测试通过梯度/同步检查，但无噪声评估仍提前终止，尚不适合长时间正式训练。

已实现 S54 数据、模型准备、FK、MDP、网络维度、训练入口、评估与渲染适配。
沙盒外通过 `sg docker` 接通宿主机；独立容器内已完成 USD 导入、Newton 1/16 环境和 30 iteration 训练。
原先的设备/权限错误仅来自沙盒，不是宿主机故障。评估、NPZ 与 5 秒视频也已完成。结果见 [报告](docs/S54_PORTING_REPORT.md)。

## 架构与数据契约

`supermimic/s54.py` 是机器人名称、路径、关节维度和物理参数来源的统一入口。
以 Leju-GMR `assets/kuavo_s54/xml/biped_s54.xml` 为运动学基准，root 为 `base_link`。
原模型 `nq=36,nv=35,nu=29`。14 个 tracked bodies 来自该模型；可微 FK 保留全部 30 个 body。

Raw CSV 为 3945×34：root xyz、quat **xyzw**、左腿6/右腿6/waist1/左臂7/右臂7。
30 FPS；所有角度为 rad。补齐 `zhead_1_joint/zhead_2_joint` 为 MJCF qpos0=0，
已检查合法范围，输出 3945×36；只输出完整文件到训练目录，raw 另存。
JSON sidecar 包含名称顺序、FPS、四元数约定、源/输出 SHA256、头部填充值及范围/速度统计。
Loader 要求该 metadata，拒绝 G1、混入的 raw、损坏或未重新验证的 CSV。
30→50 Hz 采用线性插值与 SLERP，得到 6574 帧。内部四元数为 wxyz；Isaac Lab 3 接口为 xyzw。

仿真 MJCF 副本 `s54_sim.xml` 移除 world floor 和源 motor/sensor，保持运动学、惯量不变。
47 个 collision proxy 的 mask 显式设为1，场景创建唯一地面；自碰撞关闭。
源 torque motors 移除，避免与 articulation 的 implicit PD 双重驱动；effort 和 armature 从源 MJCF 读取。
速度限值仅取本地 S54 URDF 的 limit 字段，不采用其运动学。PD 是有记录的 sim-only 假设，尚未调参。
已有 `data/s54_usd/biped_s54.usd` 只有27个可动关节且腰部 COM 不同，禁止复用；已从源 MJCF 生成并验证新 USD。

## 准备数据与 CPU 验证

在新 repo 根目录执行（当前已实际执行）：

```bash
cd /mnt/sage_data/workspace/supermimic_s54
python tools/copy_s54_sources.py --gmr /mnt/sage_data/workspace/Leju-GMR
python tools/inspect_s54_model.py
python tools/prepare_s54_motion.py --expected_frames 3945
python tools/prepare_s54_asset.py
OPENBLAS_NUM_THREADS=1 python tools/test_s54_cpu.py
```

CPU 依赖为 numpy、scipy、torch、mujoco；不导入 Isaac Lab。
当前测试环境 torch 2.14.1+cu130、mujoco 3.17.0。完整网络单次 WM/8步 BPTT/critic 梯度测试
使用参考运动增量作诊断目标，不产生训练 checkpoint，不是动力学性能实验。

USD 静态检查需 `pxr`（复用镜像内已有依赖即可）。当前会话仅为检查旧 USD 安装了工作区隔离依赖：

```bash
python -m pip install --target .cache/usd_python usd-core
PYTHONPATH=.cache/usd_python python tools/check_s54_usd.py \
  --usd data/s54_usd/biped_s54.usd --out results/s54/old_usd_alignment.json
# 该旧资产预期 exit 1；新资产必须 exit 0
```

## 独立容器与完整 smoke

以下须在有 Docker 权限及 GPU 的宿主机执行。先查看 `nvidia-smi` 与 `docker ps`，
等原 G1 GPU 工作结束、至少空闲 5000 MiB 后再启动 S54 测试。无需 rebuild 镜像。
默认容器 `supermimic_s54`；脚本明确拒绝原容器名 `supermimic`。

```bash
cd /mnt/sage_data/workspace/supermimic_s54
sudo docker ps --format '{{.Names}} {{.Status}} {{.Image}}'
nvidia-smi
sudo env CONTAINER_NAME=supermimic_s54 DOCKER_IMAGE=supermimic:latest \
  bash docker/run_container.sh 0 32g
sudo docker inspect --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}' supermimic_s54
sudo docker exec supermimic_s54 python -c 'import torch; print(torch.cuda.get_device_name(0))'
# 顺序执行 M0、CPU、USD、1 env、16 env、30 iterations、evaluation、视频；失败立即停止
sudo docker exec supermimic_s54 bash tools/run_s54_smoke.sh
```

容器应把本新 repo 挂载到 `/workspace/supermimic`。不要进入旧容器执行这些命令。
`run_s54_smoke.sh` 拒绝覆盖已存在的 `logs/s54_smoke`。若本机 Docker 无需 sudo，可去掉 sudo。当前用户已属于 docker 组但会话未继承时，可用 `sg docker -c 'docker ps'`；本次实际执行采用该方式。
现有容器无需再次 docker run，但必须检查 mount 正确；脚本不会删除、停止或重建任何已有容器。

分步执行资产和环境检查：

```bash
sudo docker exec supermimic_s54 python tools/mjcf_to_usd.py
sudo docker exec supermimic_s54 python tools/check_s54_usd.py
sudo docker exec supermimic_s54 python tools/test_env_s54.py --num_envs 1
sudo docker exec supermimic_s54 python tools/test_env_s54.py --num_envs 16
```

Kit-less importer 注册其 schema，输出 `data/s54_newton_usd/s54_sim/s54_sim.usda`，
检查 joint frames/axes/ranges/COM/mass、Z-up/metre、Physics=mujoco variant 并写出 contract。
环境要求该 contract，显式构造 CSV canonical→sim 与 inverse permutation，所有读写都按名称映射。
严格 smoke 的 FK 阈值为位置1e-4 m、姿态1e-3 rad；不能通过则先定位 importer/root/joint frames。
Smoke 读取真实接触 buffer，要求两脚均出现地面接触；不通过时保留诊断，不放宽检查。

## 30 iteration 训练

训练入口默认参数已限制为16 env、30 iterations的小smoke。先通过上述1/16 env测试，再运行（tmux 后台，正常 shell 退出不代表训练完成）：

```bash
sudo docker exec -w /workspace/supermimic supermimic_s54 python supermimic/train.py --log_dir logs/s54_smoke \
  --motion_dir data/retarget_data/reference_startup_fixed --num_envs 16 --iterations 30 \
  --buffer_steps 64 --warmup_steps 32 --wm_window 8 --wm_batches 1 --wm_batch 64 \
  --pi_batch 32 --horizon 8 --wm_only_iters 10 --ckpt_every 30 --log_every 10 \
  --amp bf16 --no_compile
sudo docker exec supermimic_s54 tail -f logs/s54_smoke/stdout.log
# 训练结束后运行；明确检查 checkpoint 身份和日志有限性
sudo docker exec supermimic_s54 python tools/check_s54_run.py
```

`--no_compile` 只关闭编译以缩短 smoke 启动，保持原 WM loss、BPTT imagined rollout、critic bootstrap。
梯度非有限时立即失败。`last.pt` 含 critic/optimizer，可用 `--resume logs/s54_smoke/last.pt` 恢复；
编号 checkpoint 用于独立评估。S54 权重重新初始化；无 S54 contract 的旧权重直接拒绝。
训练日志路径限定本 repo `logs/s54_*`，新训练拒绝覆盖已有 args/训练日志/checkpoint；
允许 wrapper 先创建 stdout.log。恢复训练会重新填充未保存到 checkpoint 的 replay buffer。

## 评估与渲染

```bash
sudo docker exec supermimic_s54 python supermimic/evaluate.py \
  --run_dir logs/s54_smoke --motion_dir data/retarget_data/reference_startup_fixed \
  --ckpt logs/s54_smoke/ckpt_000030.pt --record logs/s54_smoke/traj_30.npz
sudo docker exec supermimic_s54 python tools/check_s54_run.py --require_eval
sudo docker exec -e MUJOCO_GL=egl supermimic_s54 python tools/render.py \
  logs/s54_smoke/traj_30.npz logs/s54_smoke/dance_30.mp4 \
  --clips dance1_subject1 --start 0 --seconds 5 --width 960 --height 540 --fps 30
```

评估每条 clip 从第一帧开始，friction=1，无随机化/噪声，失败后停止计入回报。
输出 `norm_return,norm_return_framew,n_failed,failed,per_clip` 并保存 `_eval.json`；
npz 带机器人/关节/控制步长/四元数约定，renderer 拒绝错误 contract。
视频关节按 MJCF 名称赋值，Reference 和 Policy 均为 S54，支持 clip/start/seconds/resolution/fps。
渲染默认使用灰白外壳、灰色关节与黑色末端的 Policy、青色半透明 Reference、蓝灰天空和深灰棋盘地面，并降低高光。
配色仅应用于渲染器内存中的模型副本；重新运行 `tools/render.py` 即可导出新配色视频。
默认 ghost 的 xy 与 heading 对齐策略匹配 tracking reward；`--global_ghost` 可看全局漂移。
评估保存真实 terminal state；终止后 Policy 姿态冻结并标记 terminated，Reference 继续播放，
不伪造后续存活帧。
宿主机文件在 `logs/s54_smoke/`，无需拷出容器。

Reference 独立预览可在当前 CPU 环境重跑：

```bash
MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 python tools/preview_s54_reference.py
ffprobe -v error -show_entries stream=nb_frames,duration,width,height -of json \
  results/s54/reference_preview.mp4
```

已生成5秒/150帧/480×360 Reference-only 视频。它不是 Policy comparison 或 evaluation。

S54 动作数据统一放在 `data/retarget_data/`，77 段原始人体 BVH 位于 `raw/`。
足底、上肢及 link 自碰撞修正版默认输出到 `data/retarget_data/reference_startup_fixed/`。
旧实验与 Reference 保留便于核对。部分旧 clip 的首帧 IK 未收敛，不能直接用于续训。
2026-10-28 新增首帧 IK 收敛、根部连续性检查及头部奖励，见 [诊断报告](reports/s54_startup_head_fix.md)。
目录与命令说明见 `data/retarget_data/README.md`，实测见 `reports/s54_self_collision_fix.md`，足底阶段历史见 `reports/s54_foot_grounding.md`。
训练时请将上述示例中的 `--motion_dir` 改成 `data/retarget_data/reference_startup_fixed`。

```bash
# 生成 dance1 并打开查看；已生成的合格结果直接复用。
MUJOCO_GL=glfw OPENBLAS_NUM_THREADS=1 /data/conda_envs/leju_gmr/bin/python tools/generate_s54_reference.py \
  --clips dance1_subject1 --skip_existing --view

# 其他 clips，顺序处理；也可使用 --clips walk1_subject1 walk2_subject1。
OPENBLAS_NUM_THREADS=1 /data/conda_envs/leju_gmr/bin/python tools/generate_s54_reference.py \
  --all --exclude dance1_subject1 --skip_existing --continue_on_error

# 生成完成后，再执行验证或渲染。
OPENBLAS_NUM_THREADS=1 /data/conda_envs/leju_gmr/bin/python tools/validate_s54_upper_body.py \
  --motion_dir data/retarget_data/reference_startup_fixed
MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 /data/conda_envs/leju_gmr/bin/python \
  tools/compare_s54_reference.py --candidate_dir data/retarget_data/reference_startup_fixed \
  --seconds 15 --out results/s54/upper_body_comparison.mp4
```

生成器先在导出时间线之外使首帧 IK 收敛，再执行根部连续性检查、足底 IK、上肢修正、非相邻 link 避碰、关节速度约束、平滑抬升，
并证明关节插值区间无自碰撞，复核实际 50 Hz 训练数据。无需旧 Reference。
训练只接受通过全部检查且证书绑定当前 CSV 的数据；失败 clip 隔离，不进入训练。
`--dry_run` 只查看计划；
`--max_frames 300` 仅作前约 10 秒粗测，生成完整训练数据时省略。
上肢修正的历史结果见 `reports/s54_upper_body_fix.md`；报告对应的旧数据已按用户要求清除。
原始 Leju-GMR 文件、机器人资产、G1 数据、训练日志和 checkpoint 均未修改。

## 初次适配 Smoke 的历史限制

当前仅一个 dance clip，可宣称单动作适配 smoke；不能与 G1 40-clip 0.82/0 failed 公平对标。
关节无实质超限，但532个关节采样区间超过 URDF 速度限制，未裁剪/平滑；Newton 已证明接口与接触正确；PD playback 会失稳，参数与数据可执行性仍需调优。
主要性能指标应为固定评估协议下的 Evaluation Normalized Return 对训练时间/iteration 和失败 clip 数。
`sim_rew/imag_rew` 只作诊断。实际 `ckpt_000030.pt` 评估 `norm_return=norm_return_framew=0.0006181039`，失败1/1；策略在0.76秒触发终止。
对比视频 `logs/s54_smoke/dance_30.mp4` 为960×540、30 FPS、150帧；终止后冻结真实末状态并标记，Reference 继续运动。
这证明端到端单动作适配 smoke 通过，不代表已学会跟踪。

## 初次适配时准备的小规模训练命令（历史记录）

保留此次 smoke 结果，以新目录从 last.pt 继续到300 iterations。先检查 GPU 空闲：

```bash
sudo docker exec -w /workspace/supermimic supermimic_s54 python supermimic/train.py --log_dir logs/s54_pilot_300 \
  --motion_dir data/retarget_data/reference_startup_fixed --resume logs/s54_smoke/last.pt \
  --num_envs 16 --iterations 300 --buffer_steps 64 --warmup_steps 32 \
  --wm_window 8 --wm_batches 1 --wm_batch 64 --pi_batch 32 --horizon 8 \
  --wm_only_iters 10 --ckpt_every 100 --log_every 10 --amp bf16 --no_compile
sudo docker exec supermimic_s54 python supermimic/evaluate.py \
  --run_dir logs/s54_pilot_300 --motion_dir data/retarget_data/reference_startup_fixed \
  --ckpt logs/s54_pilot_300/ckpt_000300.pt --record logs/s54_pilot_300/traj_300.npz
```

300为总迭代数（从30继续270次）。正式扩容前检查固定协议评估收益、失败时间和 PD/参考速度问题。

## 当前训练入口与 SwanLab

统一入口是根目录 `run_train.sh`，超参数、默认 clips 与云端配置直接在该文件顶部修改。
默认读取 `data/retarget_data/reference_startup_fixed`，训练 `dance1_subject1`，启用 SwanLab 云端同步。
当前脚本保留用户设置的 2048 env、10000 iterations、WM batch 1024×16、Policy batch 256。
使用说明见 [训练启动器](docs/S54_LAUNCHER.md)。旧 [4060训练指南](docs/S54_4060_TRAINING.md)
保留历史配置的性能数据，不代表当前配置的显存与功耗。

```bash
# 首次使用 SwanLab 时登录；之后复用宿主机凭据。
.venv-swanlab/bin/swanlab login
bash run_train.sh --dry-run train s54_dance_cloud --clips dance1_subject1
sg docker -c 'bash run_train.sh train s54_dance_cloud --clips dance1_subject1'
tail -f logs/s54_dance_cloud/swanlab_sync.log
```
