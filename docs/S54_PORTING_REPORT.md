# S54 迁移报告（2026-10-26）

> 数据清理说明（2026-10-27）：本文为历史实验记录，对应旧 Reference 和中间版本已删除。当前仅保留 `data/retarget_data/raw/` 的原始 BVH；重新生成请使用 `README_S54.md` 的命令，默认输出到 `data/retarget_data/reference/`。

目标路线保留 Isaac Lab 3 Kit-less + Newton/MJWarp + PyTorch World Model/BPTT/Critic。
代码、CPU、USD 与 Newton 1/16 环境验证已完成，30 iteration smoke、checkpoint evaluation、NPZ 与对比视频全部完成。
原 `supermimic_vibed`、原容器、原镜像、G1 数据和日志未修改；只执行小规模 GPU smoke，未启动正式训练。

## 阶段状态与命令

| 阶段 | 状态 | 实际结果/证据 |
|---|---|---|
| M0 数据/模型契约 | PASS | 3945×34 → 3945×36，30 Hz，29 维，metadata/sha；`results/s54/prepare_motion.log` |
| M1 MJCF 副本 | PASS | 无内置 floor，47 proxies；29 源 motors 与 sensors 删除，由 Isaac PD 驱动；运动学/质量/惯量逐数组完全相等 |
| M1 旧 USD 核对 | FAIL | 缺两个可动头关节，无 Physics=mujoco variant，waist_yaw COM 相差0.04409894 m；不使用 |
| M1 新 USD 导入与动态对齐 | PASS | 新 USD 静态对齐通过；Newton reset/step FK 位置误差均小于 6e-7 m |
| M2 CPU FK/MDP/WM/BPTT/Critic | PASS | 完整网络单次梯度诊断与真实 Newton 30 iteration、8步 BPTT 均通过 |
| M3 独立容器 | PASS | 复用 supermimic:latest；新 repo mount 与 CUDA 实测正常，原 G1 容器仅 sleep |
| M4 CPU/native MuJoCo | PASS | 数据、独立 FK/速度对照、地面接触、梯度和维度测试 |
| M4 Newton 1 env / 16 env | PASS | 各60步；roundtrip/FK、双脚接触、唯一地面、驱动/finite 均通过；newton_1env.json/newton_16env.json |
| M4 30 iteration 训练 | PASS | 真实 Newton rollout + WM/BPTT/Critic；finite 日志与 ckpt_000030.pt/last.pt，check_s54_run.py 通过 |
| M5 checkpoint evaluation / npz / Policy视频 | PASS | norm_return=0.0006181039，失败1/1；39帧真实轨迹（含terminal）、150帧/5秒/960×540对比视频 |
| M5 Reference 预览/双模型渲染构建 | PASS | Reference-only 5秒/150帧/480×360视频；另实测双S54模型离屏渲染，非 Policy视频 |
| Python/shell 静态检查 | PASS | `python -m compileall -q supermimic tools`; `bash -n docker/run_container.sh run_train.sh tools/*.sh` |

重跑命令与完整宿主机/容器流程见 [README_S54.md](../README_S54.md)。
顺序入口 `sudo docker exec supermimic_s54 bash tools/run_s54_smoke.sh` 在第一个失败阶段停止。

实际 CPU 命令：

```bash
python tools/copy_s54_sources.py --gmr /mnt/sage_data/workspace/Leju-GMR
python tools/inspect_s54_model.py
python tools/prepare_s54_motion.py --expected_frames 3945
python tools/prepare_s54_asset.py
OPENBLAS_NUM_THREADS=1 python tools/test_s54_cpu.py
PYTHONPATH=.cache/usd_python python tools/check_s54_usd.py --usd data/s54_usd/biped_s54.usd --out results/s54/old_usd_alignment.json
MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 python tools/preview_s54_reference.py
```

最后的旧 USD 检查预期 exit 1，详见 JSON；其它上述 CPU 命令成功。

## 模型和关节映射

Leju-GMR 源 MJCF 和本地 `data/s54_description/xml/biped_s54.xml` 字节相同。
SHA256：`9a92e5b125808d9021b0d52cac7ec22b58de260cee9392bf2d667c56ff8da446`。
源模型 nq=36/nv=35/nu=29，floating `base_link`，freejoint 无名称，默认 root=(0,0,0.82)、quat wxyz=(1,0,0,0)。
源 qpos0 的直腿姿态过低；articulation 初始 spawn 使用原生 geom-floor distance 推导的安全高度
0.9917 m（最低代理离地2 cm），随后 reset 完整写入 Reference root，不偏移参考。
仿真副本仅给 freejoint 命名 `floating_base_joint`，不改变根 frame 或零位。

表中 index/column 全部 **0-based**。`sim-index` 在真实 articulation 启动时由名称求得并打印，
本次真实 Newton articulation 的 sim-index 与 canonical index 相同（0–28），见 newton 日志。29 个 canonical→sim 和 inverse 数组都 assert 一一对应，
root/joint reset、state read、position target 写入均应用映射。两头部关节 qpos0=0 且范围分别
[-1.5708,1.5708]、[-0.523599,0.523599]，不是未经检查直接补零。

| canonical index | joint | raw CSV column | full CSV column | MuJoCo qposadr | axis | range rad |
|---|---|---|---|---|---|---|
| 0 | `leg_l1_joint` | 7 | 7 | 7 | [1.0, 0.0, 0.0] | [-0.314159, 0.663225] |
| 1 | `leg_l2_joint` | 8 | 8 | 8 | [0.0, 0.0, 1.0] | [-0.785398, 1.91986] |
| 2 | `leg_l3_joint` | 9 | 9 | 9 | [0.0, 1.0, 0.0] | [-1.91986, 1.5708] |
| 3 | `leg_l4_joint` | 10 | 10 | 10 | [0.0, 1.0, 0.0] | [0.0, 2.53073] |
| 4 | `leg_l5_joint` | 11 | 11 | 11 | [0.0, 1.0, 0.0] | [-0.872665, 0.349066] |
| 5 | `leg_l6_joint` | 12 | 12 | 12 | [1.0, 0.0, 0.0] | [-0.261799, 0.261799] |
| 6 | `leg_r1_joint` | 13 | 13 | 13 | [1.0, 0.0, 0.0] | [-0.663225, 0.314159] |
| 7 | `leg_r2_joint` | 14 | 14 | 14 | [0.0, 0.0, 1.0] | [-1.91986, 0.785398] |
| 8 | `leg_r3_joint` | 15 | 15 | 15 | [0.0, 1.0, 0.0] | [-1.91986, 1.5708] |
| 9 | `leg_r4_joint` | 16 | 16 | 16 | [0.0, 1.0, 0.0] | [0.0, 2.53073] |
| 10 | `leg_r5_joint` | 17 | 17 | 17 | [0.0, 1.0, 0.0] | [-0.872665, 0.349066] |
| 11 | `leg_r6_joint` | 18 | 18 | 18 | [1.0, 0.0, 0.0] | [-0.261799, 0.261799] |
| 12 | `waist_yaw_joint` | 19 | 19 | 19 | [0.0, 0.0, 1.0] | [-3.14, 3.14] |
| 13 | `zarm_l1_joint` | 20 | 20 | 20 | [0.0, 1.0, 0.0] | [-3.14159, 1.5708] |
| 14 | `zarm_l2_joint` | 21 | 21 | 21 | [1.0, 0.0, 0.0] | [-0.349066, 3.49066] |
| 15 | `zarm_l3_joint` | 22 | 22 | 22 | [0.0, 0.0, 1.0] | [-1.48353, 1.48353] |
| 16 | `zarm_l4_joint` | 23 | 23 | 23 | [0.0, 1.0, 0.0] | [-2.61799, 0.0] |
| 17 | `zarm_l5_joint` | 24 | 24 | 24 | [0.0, 0.0, 1.0] | [-1.5708, 1.5708] |
| 18 | `zarm_l6_joint` | 25 | 25 | 25 | [1.0, 0.0, 0.0] | [-1.309, 0.698132] |
| 19 | `zarm_l7_joint` | 26 | 26 | 26 | [0.0, 1.0, 0.0] | [-0.698132, 0.698132] |
| 20 | `zarm_r1_joint` | 27 | 27 | 27 | [0.0, 1.0, 0.0] | [-3.14159, 1.5708] |
| 21 | `zarm_r2_joint` | 28 | 28 | 28 | [1.0, 0.0, 0.0] | [-3.49066, 0.349066] |
| 22 | `zarm_r3_joint` | 29 | 29 | 29 | [0.0, 0.0, 1.0] | [-1.48353, 1.48353] |
| 23 | `zarm_r4_joint` | 30 | 30 | 30 | [0.0, 1.0, 0.0] | [-2.61799, 0.0] |
| 24 | `zarm_r5_joint` | 31 | 31 | 31 | [0.0, 0.0, 1.0] | [-1.5708, 1.5708] |
| 25 | `zarm_r6_joint` | 32 | 32 | 32 | [1.0, 0.0, 0.0] | [-0.698132, 1.309] |
| 26 | `zarm_r7_joint` | 33 | 33 | 33 | [0.0, 1.0, 0.0] | [-0.698132, 0.698132] |
| 27 | `zhead_1_joint` | 无，补中立值 | 34 | 34 | [0.0, 0.0, 1.0] | [-1.5708, 1.5708] |
| 28 | `zhead_2_joint` | 无，补中立值 | 35 | 35 | [0.0, 1.0, 0.0] | [-0.523599, 0.523599] |

Leju-GMR `scripts/pkl_to_csv.py` 的 kuavo_s54 layout 已阅读：12腿+1腰+14臂连续27维。
本工具进一步断言编译 MJCF 的完整 hinge 名称顺序正好为上述27维加两头。
源CSV位于 `retarget_data/lafan1_s54_raw/`，训练目录根只含一个完整 CSV；不覆盖/混放raw。
输出带 metadata，加载时要求模型fingerprint、名称顺序、fps/units/quat、SHA和frame count均一致。

Quaternion norm 最大误差 1.22e-15；无 NaN/Inf；无超过1e-6 rad的关节硬限位违规，
无大于pi的单步关节跳变。532个joint intervals超过URDF速度限值，最高约38.11 rad/s；均保留并记录，未裁剪/平滑。
30→50 Hz重采样输出6574帧，持续131.46s（原始最后时间131.4667s，按控制时间网格向下取整）。

## 资产、碰撞、惯量与控制

旧 USD 的27个可动关节 frames/axes/ranges 可通过1e-5级静态比较，
但两头部不可动、缺 Physics=mujoco variant，腰部 COM 不一致，故整体 FAIL。
旧 USD COM=[0.0420494713,-0.000608409,0.170535]，MJCF=[-0.00204947,-0.000608409,0.170535]。
`old_usd_alignment.json` 和 `existing_usd_inspection.json` 保存真实检查明细。
新 USD importer 的源是生成的 S54 专用 MJCF，绝不使用旧27-DoF USD。
静态 checker 在 session layer 选择 Physics=mujoco，不写原 USD；比对 root、坐标轴/单位、
父子 links、joint anchors/local rotations/axes/ranges、mass/COM/inertia tensor。
静态通过后已执行 Newton roundtrip 和 FK。修正 importer 重跑逻辑后再次导出，新资产单环境复验也通过；证据为 import_rerun.log / newton_after_reimport.json。

源 floor contype=1/conaffinity=0，左足 proxy 1/2，右足2/1；左足与该 floor 没有 mask 配对。
仿真副本移除 world geoms，用47个proxy的显式1/1 mask和唯一场景GroundPlane。
保留视觉mesh为0/0，不以视觉mesh充当碰撞；禁用自碰撞为明确的V1模拟假设。
源29 torque motors和sensor被删除，避免导入第二套controller；关节effort限制仍保留，Isaac PD由cfg提供。
原生CPU自由落体试验（关闭自碰撞）中左右足都与地面产生实际contact，峰值合力约4725N；
机器人随后倒下，root下降约0.948m。该测试仅证明重力与地面响应，不是稳定站立验收，也不是Newton接触验收。

所有质量/惯量采用源MJCF。effort采用源motor ctrlrange（gear=1），与joint actuatorfrcrange取更严格值；
armature采用MJCF的0.05 kg m²。速度只读取URDF limit（不采用URDF运动学）。
头部 motor ctrlrange 均为 ±14.1 Nm，但 MJCF joint actuatorfrcrange 分别限制为 ±1.5 / ±12 Nm；实际 effort 取两者交集，未使用 URDF effort。
PD尚无硬件标定：腿/腰kp=200，上臂kp=80，腕kp=20，头kp=10；
kd=2sqrt(kp*armature)，action_scale=0.25*effort/kp。
这些是可追踪、待调的sim-only参数；没有复用G1 motor物理数值。

| joint | effort Nm | velocity rad/s | kp Nm/rad (sim-only) | kd (sim-only) |
|---|---|---|---|---|
| `leg_l1_joint` | 127 | 10.4 | 200 | 6.32456 |
| `leg_l2_joint` | 71 | 8.7 | 200 | 6.32456 |
| `leg_l3_joint` | 132 | 12.7 | 200 | 6.32456 |
| `leg_l4_joint` | 280 | 10.4 | 200 | 6.32456 |
| `leg_l5_joint` | 91.6 | 17.8 | 200 | 6.32456 |
| `leg_l6_joint` | 30 | 17.8 | 200 | 6.32456 |
| `leg_r1_joint` | 127 | 10.4 | 200 | 6.32456 |
| `leg_r2_joint` | 71 | 8.7 | 200 | 6.32456 |
| `leg_r3_joint` | 132 | 12.7 | 200 | 6.32456 |
| `leg_r4_joint` | 280 | 10.4 | 200 | 6.32456 |
| `leg_r5_joint` | 91.6 | 17.8 | 200 | 6.32456 |
| `leg_r6_joint` | 30 | 17.8 | 200 | 6.32456 |
| `waist_yaw_joint` | 102 | 8.7 | 200 | 6.32456 |
| `zarm_l1_joint` | 66 | 18.8 | 80 | 4 |
| `zarm_l2_joint` | 75 | 8 | 80 | 4 |
| `zarm_l3_joint` | 57 | 7.5 | 80 | 4 |
| `zarm_l4_joint` | 75 | 8 | 80 | 4 |
| `zarm_l5_joint` | 14.1 | 17.5 | 20 | 2 |
| `zarm_l6_joint` | 14.1 | 17.5 | 20 | 2 |
| `zarm_l7_joint` | 14.1 | 17.5 | 20 | 2 |
| `zarm_r1_joint` | 66 | 18.8 | 80 | 4 |
| `zarm_r2_joint` | 75 | 8 | 80 | 4 |
| `zarm_r3_joint` | 57 | 7.5 | 80 | 4 |
| `zarm_r4_joint` | 75 | 8 | 80 | 4 |
| `zarm_r5_joint` | 14.1 | 17.5 | 20 | 2 |
| `zarm_r6_joint` | 14.1 | 17.5 | 20 | 2 |
| `zarm_r7_joint` | 14.1 | 17.5 | 20 | 2 |
| `zhead_1_joint` | 1.5 | 5.23 | 10 | 1.41421 |
| `zhead_2_joint` | 12 | 5.23 | 10 | 1.41421 |

14 tracked bodies：`base_link, leg_l1_link, leg_l4_link, leg_l6_link, leg_r1_link, leg_r4_link, leg_r6_link, waist_yaw, zarm_l2_link, zarm_l4_link, zarm_l7_link, zarm_r2_link, zarm_r4_link, zarm_r7_link`。
分别对应base、左右hip-roll/knee/foot、上身、左右shoulder-roll/elbow/hand末端link。
全部在源MJCF及真实Isaac articulation中存在，运行时名称检查与sim-index双向映射已通过。

## 核心实现与实际数值

可微FK保留fixed bodies、joint pivot、局部frame旋转与qpos0，所有速度是link-origin的世界速度。
与原生 `mj_forward` 的xpos/xmat及 `mj_jacBody*qvel` 独立比较，不用同一实现互相验证。
128个样本一半是整个clip均匀抽帧，一半是完整合法joint range随机姿态，全部30个body参与。

| 检查 | 最大误差 |
|---|---|
| FK position | 9.35927107e-07 m（阈值1e-4） |
| FK rotation | 2.48794985e-07 rad（阈值1e-3） |
| FK linear velocity | 4.15033321e-06 m/s（阈值1e-4） |
| FK angular velocity | 3.48713765e-06 rad/s（阈值1e-4） |
| CPU state roundtrip | 2.98023224e-07 |

STATE=71、OBS=99、XI=33、每goal frame=73，由NUM_J=29计算；TRACKED=14；
WM输出为70，critic输入K=10时914，CPU测试K=2时330。维度与原G1相同不代表权重可迁移。
`assert_checkpoint`拒绝缺失/不匹配S54身份，编号checkpoint和last.pt都保存身份。
原WM损失、多步预测、BPTT、soft termination和critic bootstrap保留；没有PPO或PhysX切换。
`--no_compile`仅为小smoke关闭torch.compile；运行时梯度非有限会明确失败。

完整网络CPU单次Adam更新结果（参考增量诊断目标，**不是 simulator learning 曲线**）：

| quantity | value |
|---|---|
| wm_loss | 0.0244247112 |
| pi_loss (8-step BPTT) | -0.310268968 |
| v_loss | 0.159635574 |
| WM gradient norm | 0.253824174 |
| Policy gradient norm | 0.0972179472 |
| Critic gradient norm | 1.46676326 |

CPU诊断不写checkpoint，也不生成伪装成evaluation的npz。
原reward中的soft-limit penalty保持不变，即使Reference关节处于合法硬限位，也可能有该惩罚；
`reference_reward_diagnostic.json`给出全clip统计，不能当作Policy return。

Evaluation仍按每clip完整长度归一化，失败后的回报为0，保留norm_return/framew/n_failed/failed/per_clip。
真实terminal state会写入recording；视频可在终止后冻结Policy并标记terminated，Reference继续播放。
NPZ与checkpoint都有robot/joint/quaternion契约，renderer按名称写qpos；两模型nq=72/nv=70已实测构建渲染。
`results/s54/reference_preview.mp4`为真实参考预览（150帧/5秒/480×360），首帧已人工检查未见倒置/镜像。
`two_model_render_check.png`是仅验证solid/ghost构建的中立姿态静态图，不是Policy rollout。

## 实际 GPU smoke 与评估结果

端到端命令退出码0，完整输出 `results/s54/end_to_end.log`。使用原镜像
`supermimic:latest`（ID `sha256:1db595225a30868e568db6e0b9d37092f88825421c73d4e5b8b36fa53b8dde17`），
容器内 torch2.11.0+cu130、MuJoCo/MJWarp3.15.0、Warp1.17.0、Newton1.8.0.dev0、
isolated importer6.2.0.0；设备RTX4060 Laptop 8GiB。版本证据 `container_runtime.json/log`。
之前主机 CPU 的 torch2.14.1/MuJoCo3.17 测试也保留，端到端容器重复执行了同一 CPU 测试。

| 项目 | 1 env（60步） | 16 env（60步） |
|---|---:|---:|
| roundtrip 最大绝对误差 | 3.7253e-9 | 3.7253e-9 |
| reset FK 位置 m | 2.9802e-7 | 2.9802e-7 |
| step FK 位置 m | 4.0536e-7 | 5.7481e-7 |
| step FK 旋转 rad | 2.7205e-7 | 4.3773e-7 |
| step link 线速度误差 m/s | 1.8980e-6 | 2.3419e-6 |
| step link 角速度误差 rad/s | 1.0212e-6 | 1.4404e-6 |
| collidable world shapes | 1 | 1 |
| peak contacts | 44 | 704 |
| 双足实际接触 | PASS | PASS |
| joint target 引起最大角变化 rad | 0.10666 | 0.10666 |
| mean normalized reward（诊断） | 0.295883 | 0.295883 |
| failed envs | 1 | 16 |

Reference PD playback 后失稳；其成功标准是接口、驱动、接触、有限状态与严格FK，不是稳定跟踪。
真实训练自动reset、并行rollout以及domain randomization也随30iteration通过。
Torch reported peak allocation约30.15MiB仅包含这些环境测试中的Torch张量，不包含Warp/CUDA全部显存。
训练进程的一次nvidia-smi采样为890MiB总占用；未采集完整GPU峰值，不将采样冒充峰值。

实际参数16env/buffer64/warmup32/wm_window8/wm_batches1/wm_batch64/pi_batch32/horizon8，
前10iteration仅WM；随后20iteration训练Policy/Critic。BF16、no_compile，原损失不变。
train_log.csv最后一行（对应最近10次聚合）：

| metric | iteration30 |
|---|---:|
| wm_loss | 2.05465 |
| wm_err8 | 3.90504 |
| pi_loss | -0.0272224 |
| v_loss | 0.000442697 |
| sim_rew | 0.0556074 |
| imag_rew | 0.032508 |
| pi_gradnorm | 0.40546 |
| sim_steps（含warmup） | 1712 |
| logged training wall_h | 0.00148651 |

日志wall_h约5.35秒只计warmup后的训练循环，不含容器启动、资产导入、仿真初始化和warmup，
不能拿它当全流程耗时。三个优化器均用error_if_nonfinite=True检查梯度；loss有限性显式检查。
`ckpt_000030.pt`含WM/Policy，`last.pt`还含Critic和三组optimizer；
`check_s54_run.py --require_eval`验证finite参数、迭代、S54身份及所有评估字段通过。

固定评估从clip首帧开始，完整长度6573个transition作为分母：
`norm_return=norm_return_framew=0.0006181038916110992`，`n_failed=1`，
`failed=[dance1_subject1]`，`per_clip={dance1_subject1:0.0006}`。
策略在0.76s终止，NPZ共39帧（含真实terminal），所有数组finite；没有伪造存活轨迹。
视频`logs/s54_smoke/dance_30.mp4`为960×540、30FPS、150帧/5秒。
检查0.2s与3s截图：模型无倒置或关节镜像，后者明确标注terminated；Reference继续运动，
前5秒参考关节最大变化2.07997rad，头部参考始终为合法0。

**M0–M5工程smoke通过；跟踪性能尚未达标。** 单动作/30iteration结果不能和G1四十clip公平对标。
下一步小规模300总iteration的恢复训练命令在README_S54.md；仅提供命令，未执行。

## 修改文件

实际新增/修改源码及入口：

- `supermimic/buffer.py`
- `supermimic/env.py`
- `supermimic/evaluate.py`
- `supermimic/kinematics.py`
- `supermimic/mdp.py`
- `supermimic/motion.py`
- `supermimic/robot.py`
- `supermimic/s54.py`
- `supermimic/train.py`
- `tools/bench.py`
- `tools/check_s54_run.py`
- `tools/check_s54_usd.py`
- `tools/copy_s54_sources.py`
- `tools/download_data.sh`
- `tools/inspect_s54_model.py`
- `tools/mjcf_to_usd.py`
- `tools/prepare_s54_asset.py`
- `tools/prepare_s54_motion.py`
- `tools/preview_s54_reference.py`
- `tools/render.py`
- `tools/run_s54_smoke.sh`
- `tools/sync_swanlab.py`
- `tools/test_env.py`
- `tools/test_env_s54.py`
- `tools/test_reset.py`
- `tools/test_s54_cpu.py`
- `tools/test_spawn.py`
- `tools/urdf_to_mjcf.py`
- `docker/run_container.sh`
- `run_train.sh`
- `README.md`
- `README_S54.md`

各模块理由：s54.py定义契约与参数来源；robot.py仅负责Isaac cfg；kinematics.py修复固定body与锚点通用FK；
motion.py验证数据/metadata并通用维度；mdp.py和buffer.py从契约计算切片和shape；
env.py维护canonical/sim双向映射、root/COM、USD contract和reset后forward；
train.py采用S54默认数据、16env/30iteration安全默认值、身份验证、安全日志路径和finite gradient；允许tee预建stdout.log，恢复训练重新填充replay，新日志目录也写CSV表头；evaluate.py输出完整per_clip/JSON和terminal state；
render.py构建两个S54模型、按名称写qpos并按精确fps采样；shell脚本隔离容器/mount/log并停止失败流程。
networks.py本身无机器人路径，原本已从mdp导入NUM_J/INV_DIM/XI_DIM，因此保留网络实现，已实测完整维度与梯度。
旧test_env入口转发严格S54 smoke，其余测试/benchmark默认改S54；旧URDF转换入口明确拒绝改变基准；
download_data.sh改为S54源文件安全复制/校验，不再调用旧G1容器；sync_swanlab默认仅S54日志（未执行上传）。

最终全文搜索无运行代码中的G1模型/关节/数据路径残留。仅G1权重/容器/日志保护提示、
NUM_J=29/TRACKED=14契约断言及reward角速度std=3.14保留。完整搜索见hardcode_audit.txt。
原算法说明文档MILESTONES.md保留历史背景；当前验收只以本报告为准。
根目录原文件均为untracked，Git尚无初始commit，当前.git为受保护只读路径，未写Git元数据。
保留迁移前快照 `.cache/port_baseline/` 和可review的 `results/s54/port.patch`，未操作原repo。

## 执行环境与下一步

先前的 runtime_probe.json / *_attempt.log 记录的是沙盒内失败尝试，保留作历史证据。
2026-10-26 沙盒外实测 RTX 4060 8 GiB 正常。用户 light 已登记为 docker 组成员，
但会话 supplementary groups 缺 docker；`sg docker` 可直接使用该组，不需 sudo 密码。
复用 `supermimic:latest`，创建 `supermimic_s54`，仅挂载新 repo 到 `/workspace/supermimic`。
启动前原 `supermimic` 容器仅运行 sleep infinity，GPU 约70 MiB；未中断任何 G1 作业。

实际入口：

```bash
sg docker -c 'CONTAINER_NAME=supermimic_s54 bash docker/run_container.sh 0 32g'
sg docker -c 'docker exec supermimic_s54 bash tools/run_s54_smoke.sh' > results/s54/end_to_end.log 2>&1
```

导入脚本另修复重复运行时 importer 自动追加 `_1` 目录、误验证旧文件的问题：
每次使用独立临时目录，验证 importer 返回的新文件后发布到固定路径。
PD 为 sim-only 假设；参考有532处速度超 URDF 上限，实际跟踪性能仍需训练与调参。
仅单动作 smoke，不能与 G1 四十动作性能直接公平对标；未执行长时间正式训练。

接口核对参考固定版本官方源码：
[SimulationContext.forward](https://github.com/isaac-sim/IsaacLab/blob/46d6cc3d7b115906fad12efd372fbf69f4e3d3ed/source/isaaclab/isaaclab/sim/simulation_context.py)，
[Newton contact export](https://github.com/isaac-sim/IsaacLab/blob/46d6cc3d7b115906fad12efd372fbf69f4e3d3ed/source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py)。

## 后续：4060 4GB训练配置

新增 `tools/train_s54_4060.sh` 和 `docs/S54_4060_TRAINING.md`；train.py仅增加checkpoint时GPU资源统计，
不修改训练算法。已实际验证256env / WM2048×4 / PI576 / horizon32的30iteration，
总显存采样最高3948MiB、GPU利用率最高100%、功耗最高66.15W；梯度和checkpoint检查通过。
该结果是短资源测试，未执行20万iteration正式训练。完整配置/采样与恢复命令见4060训练指南。
