# Codex 任务书：SuperMimic G1 → KUAVO S54 迁移（Isaac Lab 3 Kit-less + Newton/MJWarp）

> **执行方式**：在新复制的 `/mnt/sage_data/workspace/supermimic_s54` 仓库根目录阅读本文，并实际修改代码、运行最小测试、提交可复现的命令和结果。不是只给出设计方案。不得修改原始 `supermimic_vibed` 项目。

## 0. 任务目标、约束和已知事实

目标：在保留 `zita-ch/supermimic_vibed` 的 **可微 PyTorch World Model + BPTT Policy + Critic** 算法、Isaac Lab 3.0 **kit-less** 启动方式和 **Newton/MuJoCo-Warp** 后端的前提下，将运动跟踪机器人由 Unitree G1 替换为 **KUAVO S54**。达到可以训练、评估并渲染 Reference 与 Policy 对比视频的程度。**不做 sim2real，不需要 ROS2，不迁移到 PhysX，不将 BPTT 换成 PPO。**

已知现场：

- 新项目预期：`/mnt/sage_data/workspace/supermimic_s54`；原 G1 项目：`/mnt/sage_data/workspace/supermimic_vibed`；重定向项目：`/mnt/sage_data/workspace/Leju-GMR`。
- G1 原项目已经训练成功，日志/Checkpoint 不得更改；原容器名 `supermimic`，已构建的镜像 `supermimic:latest`、`isaac-lab-kitless:main-46d6cc3` 尽可能复用。
- 本地 RTX 4060 Laptop 显存约 8 GiB；先做 1–16 env 的 smoke test，再逐渐扩容。**不得擅自执行长时间完整训练、占满 GPU 或中断原 G1 训练。**
- S54 数据已经通过 Leju-GMR 生成：`/mnt/sage_data/workspace/Leju-GMR/output/kuavo_s54/csv/dance1_subject1.csv`；3945 帧，30 FPS，`shape=(3945,34)`，全部 finite。CSV 每行是 `root_pos xyz (3) + root_quat xyzw (4) + Leg 12 / Waist 1 / Arms 14 (27)`；没有头部两个关节。
- Leju-GMR 的 S54 运动学源模型：`/mnt/sage_data/workspace/Leju-GMR/assets/kuavo_s54/xml/biped_s54.xml`（依赖同级资产目录中的 meshes）。模型打印出 floating base 6 DoF、`leg_l1..6`、`leg_r1..6`、`waist_yaw`、`zarm_l1..7`、`zarm_r1..7`、`zhead_1`、`zhead_2`：**29 个单自由度关节**，其中 CSV 仅包含前 27 个。
- S54 根 body 为 `base_link`；不能沿用 G1 的 `pelvis`、`torso_link` 以及 G1 关节名。
- 用户以前还测试过 KUAVO S54 的 USD/URDF。优先检查是否能与 Leju-GMR MJCF **运动学严格匹配**；若不匹配，以生成 CSV 的 Leju-GMR MJCF 为运动学基准，构建与之一致的、适用于 Newton/MJWarp 的仿真 USD。**不要假设名称相同便意味着关节顺序、轴向和零位相同。**
- 原 G1 `supermimic/motion.py` 默认 CSV 30 Hz，控制 50 Hz，`raw[:, 7:36]` 提取 29 维并重采样。原 `tools/validation/test_env.py` 可作为 FK 与 simulator 对齐测试参考。

### 交付的最小验收条件

1. `dance1_subject1.csv` 可以安全加载，源帧率正确，27 个参考关节与 S54 MJCF 名称/次序对应，**两个头部关节处理明确**。
2. S54 USD/MJCF/IsaacLab articulation 的可动关节、坐标系、根姿态和 FK 与 Reference 完整一致；不存在重复地面或明显碰撞配置错误。
3. `tools/validation/test_env_s54.py --num_envs 1` 或等效脚本可以在 kit-less + Newton/MJWarp 上重置 S54、前进仿真、计算跟踪奖励；对 FK 和 roundtrip 打印明确误差。
4. 30 iteration 级别的极小训练 smoke 不崩溃，WM/Policy loss 与梯度有限且写出 checkpoint。
5. 指定 checkpoint 能完整运行 evaluation，输出 `norm_return`, `norm_return_framew`, `n_failed`, `per_clip`，并能生成轨迹 `.npz` 和 S54 对比视频。
6. 提供 README，包括复制数据、镜像/容器、测试、训练、评估、渲染的真实可执行命令。对未完成部分说明精确阻塞原因，不以假数据或绕过检查伪造成功。

## 1. 工作方式与风险控制

- **首先 inspect** 现有仓库的 `supermimic/{robot,kinematics,motion,mdp,env,networks,train,evaluate}.py`、`tools/{test_env,render,mjcf_to_usd}.py`、`docker/run_container.sh`、`tools/train/run_train.sh`、S54 assets。列出硬编码 G1 的地方及依赖关系，创建简短迁移检查单，然后进入实现。不得只回复分析。
- 不要直接在原 G1 仓库修改文件。不要删除或覆盖 `logs/g1_two_motions/`、`data/lafan1_g1`、原容器 `supermimic` 或原镜像。不要使用破坏性清理、`git reset --hard` 等命令。
- 优先最小改动、可测试、可恢复；给每个阶段独立提交（如果工作目录有 Git），或保留清晰 diff。优先原算法不变、机器人与维度配置可替换；如果第一次目标更适合以 S54 为主进行最小迁移，也要避免散落无解释的魔法数字。
- 没有 GPU 或 Docker 权限时，完成所有静态检查/CPU 测试，并报告需要由用户在本地运行的真实命令；不得声称 GPU smoke 已通过。

## 2. 阶段 M0：资产和数据契约（先执行，禁止直接开训）

### M0.1 检查机器人的完整 DoF

编写 `tools/validation/inspect_s54_model.py`（或同等脚本）使用 MuJoCo 读取 `biped_s54.xml` 并记录 `nq`, `nv`, `nu`、root body/freejoint、关节名称、`jnt_qposadr`、轴向、关节范围、默认 `qpos0`、控制器/actuator 列表、body 名称、碰撞 geom。检查 S54 USD 是否有与该模型一致的 root 和关节。列出不一致项而不是默默重排。

### M0.2 固定关节映射

Leju-GMR `scripts/pkl_to_csv.py` 把关节序列输出为 `leg_l1..6`, `leg_r1..6`, `waist_yaw`, `zarm_l1..7`, `zarm_r1..7` 共 27 列。须验证名称、轴和次序恰好与项目使用的 MJCF / Isaac articulation 对上。**不要依据排列位置盲目将 numpy 数组塞入仿真器。** 建立 joint-name → CSV-index → sim-index 的映射，至少在启动时 assert 映射是一一对应。

### M0.3 27 → 29 参考补齐方案（第一版优先）

推荐 V1：仿真采用完整 **29 DoF**（包含 `zhead_1_joint`、`zhead_2_joint`），Reference 原始 27 个姿态按名称映射到 29 维；缺失的两维头部角取所用 MJCF 关节默认/合法中立角度，而不是不经检查就设零。验证头部范围、四元数归一化、弧度单位、动作连续性。应保存并记录转换前 CSV、转换后完整参考及 metadata（`fps=30`, 机器人型号、joint names、quat=xyzw、头部填充值）。

这样 `N_J=29` 的形状暂时能够保持与 G1 同阶，便于跑通第一版；**这不意味着 G1 已训练的模型权重兼容 S54**。S54 的 Policy 和 World Model 必须重新初始化训练。未来可进一步重构为仅控制 27 DoF + 两个头部 PD 固定，但不作为 V1 的先决条件。

编写 `tools/data/prepare_s54_motion.py`：输入 34 列原始 CSV，输出项目可加载的安全表示（建议另一个 36 列 CSV：3+4+29；不要覆盖原始文件）。**特别注意目录隔离：原始 34 列文件存于 `data/lafan1_s54_raw/`（或 `data/lafan1_s54/raw/` 子目录），完整 36 列文件才存于训练入口 `data/lafan1_s54/*.csv`。原 `MotionLib.from_dir` 会 glob 根目录所有 `*.csv`，不能同时混入 34 列/36 列版本。若用户此前已把原始 CSV 放到 `data/lafan1_s54/dance1_subject1.csv`，先无损移动至 raw 目录，再生成完整 CSV。同时做 NaN/Inf、shape、头部关节范围、其余 27 关节范围、异常 `dq/dt`、Root quaternion 单位范数检查。关节超限必须记录统计并报告，不要悄悄大范围裁剪且不记日志。

### M0.4 动作帧率

Leju-GMR CSV 是 **30 FPS**；原 `MotionLib` 按 30→50 Hz 内插根位置/关节角并对四元数做 SLERP。不要把 30 FPS 的数据误认为 50 FPS。可以保留原重采样逻辑，但务必改成通用 joint dimension，或验证 36 列头部补齐后的完整兼容性。

## 3. 阶段 M1：统一机器人资产和仿真配置

1. 以 `biped_s54.xml` 为运动学基准。根据它生成或验证 Isaac Lab 3 + Newton/MJWarp 可用的 USD；参考仓库 `tools/data/mjcf_to_usd.py` 的 Kit-less 离线 importer 使用方式，但不能直接沿用硬编码的 G1 路径。检查 mesh 依赖和 USD 的 Physics variant / articulation root。
2. **重要：Leju-GMR `biped_s54.xml` 在 worldbody 自带 floor geom，而 `TrackingSceneCfg` 也声明 `GroundPlaneCfg()`。必须避免重复地面。** 最好在生成的 S54 专用 MJCF/USD 资产副本中去掉训练不需要的内置 floor，再由 Isaac Lab 创建地面；不要随便修改 Leju-GMR 源模型。
3. 检查碰撞 geom 的 `contype`/`conaffinity`、足底与地面的实际接触，惯量、质量、基础 root height，避免“看起来站着但无碰撞”。禁用部分自碰撞可以作为明确记录的模拟假设，不得静默禁用地面接触。
4. 新建 `supermimic/robots/s54.py`，或重构 `robot.py` 为 S54 cfg：USD 路径、29 joint names、default pose、track body names、effort/velocity limits、PD stiffness/damping/armature/action scale。**绝不能复制 G1 的 `motor_of` 数值作为 S54 的物理事实**；优先从 S54 模型和配置获取，缺失值则采用明确、可追踪的 sim-only 参数并说明假设。
5. 从实际 MJCF 选出与原 G1 14 个 tracked bodies 语义对应的 S54 body（base, 左右膝/足, 腰/上身, 左右肩/肘/手等），确认各 body 在 MuJoCo 及 Isaac articulation 均存在。若 tracked body 数变化，同步修改 reward / critic 与维度配置。

## 4. 阶段 M2：适配 SuperMimic 核心模块

需要检查并按依赖顺序修改：

- **`supermimic/robot.py`**：S54 ArticulationCfg，joint names/order，actuator，root body、default q；避免原 G1 常量误用。
- **`supermimic/kinematics.py`**：替换 `G1Kinematics`/G1 MJCF；实现与 S54 body hierarchy 一致的可微 FK（位置/姿态/速度）。特别核对 fixed body、root link offset、joint axis 和四元数约定。不要为了通过测试而放宽 FK 一致性检查。
- **`supermimic/motion.py`**：S54 参考数据读取、完整 29 维输出、30→50 Hz 重采样、未来帧目标构建。检查 `(q,qd)` 与真实 joint order 相同。
- **`supermimic/mdp.py`**：S54 tracked bodies、初始姿态、观测与 tracking reward、软/硬终止逻辑；保持原 SuperMimic 核心损失/奖赏表达式和有效归一化口径。若 `N_J=29` 且 tracked bodies=14，部分维度可暂时不变，但必须通过 assert 证明，不要碰巧跑通。
- **`supermimic/env.py`**：`TrackingSceneCfg` 改为 S54、root body、torso/COM body、关节 position targets、环境随机化、reset Reference、帧索引、执行器范围，确保 IsaacLab Articulation joint order 与映射完全相同。训练仍用 NewtonCfg + MJWarpSolverCfg + Kit-less。验证坐标系/四元数转换（MuJoCo wxyz、CSV xyzw、Isaac Lab 接口）正确。
- **`supermimic/networks.py` 和 `train.py`**：更新 robot-specific state/obs/goal/xi/critic dims，不能加载 G1 policy checkpoint 训练 S54；保留 WM 目标/损失、BPTT imagined rollout、critic bootstrap、训练参数语义。
- **`supermimic/evaluate.py`**：评估当前 S54 motion_dir 中每条动作，固定确定性环境，返回 `norm_return`, `norm_return_framew`, `n_failed`, `failed`, `per_clip`，支持 `--record`。
- **`tools/visualization/render.py`**：S54 真实模型与 ghost 都是 S54，支持指定 clip、start、seconds、resolution、fps；检查关节 qpos 数量、root 姿态与 references，不可保留 G1 MJCF 硬编码。

请在提交前逐文件全文搜索 `G1`, `g1_29dof`, `29`, `14`, `73`, `71`, `33`, `36`, `torso_link`, `pelvis`, `root` 等疑似硬编码；逐处判断是可保留的算法常量还是必须参数化的机器人常量。没有修改理由的残留 G1 专用路径为失败。

## 5. 阶段 M3：独立 Docker 容器，不影响 G1

原 `docker/run_container.sh` 用了 `docker run --name supermimic`；原 `tools/train/run_train.sh` 用 `docker exec supermimic`，所以**复制代码以后不能直接照原样启动**。改成读取环境变量，例如：

```bash
# docker/run_container.sh（示意）
NAME="${CONTAINER_NAME:-supermimic}"
docker run -d --name "$NAME" ... -v "$REPO":/workspace/supermimic ...

# tools/train/run_train.sh（示意）
NAME="${CONTAINER_NAME:-supermimic}"
docker exec "$NAME" tmux new-session ...
```

让新工作区可用：

```bash
cd /mnt/sage_data/workspace/supermimic_s54
CONTAINER_NAME=supermimic_s54 bash docker/run_container.sh 0 32g
# 由新工作区的同名路径挂载到新容器里的 /workspace/supermimic
sudo docker exec supermimic_s54 python -c 'import torch; print(torch.cuda.get_device_name(0))'
```

以上仅为目标命令：由你检查权限和正在运行的容器后再实际执行。新容器必须挂载**新 repo**，不能继续执行旧容器 `supermimic` 中的旧 G1 源码。复用现有镜像，不默认 Docker rebuild。原 G1 训练/评估同时占用 8 GB 显存时，不要并发启动 S54 GPU 测试。

## 6. 阶段 M4：测试（必须按序，失败则先修）

1. **纯 CPU 数据/模型测试**：`tools/validation/inspect_s54_model.py`、`tools/data/prepare_s54_motion.py`，断言原始 3945 帧/30 FPS、最终单个 36 列文件/29 维、关节映射、无 NaN、quat 单位范数、关节有效性。
2. **MuJoCo FK 测试**：使用原生 MuJoCo `mj_forward` 对随机合法姿态/参考帧计算 body 位置/姿态，与 PyTorch FK 比较。输出最大位置误差、旋转误差；设置显式数值阈值（若无法达到如 1e-4 m、1e-3 rad，须报告原因）。不能用相同错误实现互相验证。
3. **Isaac Lab+Newton S54 单环境 smoke**：1 env reset Reference → 读取 joint/root state → 环境步进 20–100 步 → 检查重力/地面接触、NaN、关节 position target、生效的驱动和奖励；打印 tracked body FK 与仿真 body pose 的最大误差。
4. **16 环境 smoke**：检查批量 rollout、reset、termination、CUDA memory。
5. **极小 WM+BPTT 训练 smoke**：30 iterations，参数可从下例开始（须根据 CLI 参数实际支持和 RAM 调整）：

```bash
CONTAINER_NAME=supermimic_s54 bash tools/train/run_train.sh s54_smoke \
  --motion_dir data/lafan1_s54 \
  --num_envs 16 --iterations 30 \
  --buffer_steps 64 --warmup_steps 32 \
  --wm_window 8 --wm_batches 1 --wm_batch 64 \
  --pi_batch 32 --horizon 8 --wm_only_iters 10 \
  --ckpt_every 30 --log_every 10 --amp bf16
```

检查日志 `wm_loss`, `wm_err8`, `pi_loss`, `v_loss`, `sim_rew`, `imag_rew` 都是有限数字，真实 `sim_rew` 有意义；至少保存 checkpoint。若发现算法代码存在启动约束，请根据源码修复/调整并记录理由。

## 7. 阶段 M5：独立评估、轨迹和视频

先在训练完成、GPU 空闲后评估 smoke checkpoint：

```bash
sudo docker exec supermimic_s54 python supermimic/evaluate.py \
  --run_dir logs/s54_smoke \
  --motion_dir data/lafan1_s54 \
  --ckpt logs/s54_smoke/ckpt_000030.pt \
  --record logs/s54_smoke/traj_30.npz
```

再验证新的 S54 渲染器输出视频，例如 `dance1_subject1` 前 5 秒；注意渲染结果必须能从宿主机访问。若截图/视频出现穿模、机器人倒置、关节镜像或僵尸 Reference，追查几何坐标及映射，不可仅靠修改相机掩盖。

**科学评价**：主性能指标为固定评估协议下的 `Evaluation Normalized Return` vs training wall time / iterations，并报告失败 Clip 数量。`sim_rew` 与 WM `imag_rew` 仅作训练诊断。仅使用一个 dance clip 时只能宣称 *S54 单动作适配 smoke*，不能与原 G1 40 Clip 实验的 0.82 / 0 failed 直接宣称公平对标。

## 8. 实际交付格式

完成后在新项目提供：

- `README_S54.md`：当前实现架构、来源模型、Reference 数据契约、镜像与容器命令、测试与小规模训练命令、Evaluation/Render 命令、限制与假设。
- `docs/S54_PORTING_REPORT.md`：G1→S54 修改文件清单、各模块变更理由、关节映射表、头部关节处理、MJCF↔USD 对齐报告、碰撞设置、PD/effort 配置来源、实验结果和残余问题。
- `tools/validation/inspect_s54_model.py`、`tools/data/prepare_s54_motion.py`、`tools/validation/test_env_s54.py` 或同等可重跑脚本。
- 修改后的实际可运行源代码、明确区分的容器与日志目录。
- 对每个 Milestone 明确标注 `PASS / FAIL / NOT RUN` 和真实命令/日志摘要。尽量为每阶段创建单独 git commit；不允许为了完成任务而编造测试结果。

## 9. 执行优先级

按 **M0 数据契约 → M1 资产转换/核对 → M2 核心适配 → M3 独立容器 → M4 验证 → M5 评估与渲染** 执行。尽可能在当前回合完成可完成的代码实现和测试；遇到真实缺失的模型/权限/显存阻塞时，给出精确故障命令、定位结论和下一条操作，而不是泛泛说“需要用户确认”。**暂不执行长时间正式训练**，先确保一次端到端 smoke test 可靠。
