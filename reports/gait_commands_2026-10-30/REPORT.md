# S54 左右独立节奏接入与3小时训练配置

日期：2026-10-30（Asia/Shanghai）。本次只进行了CPU测试、离线迁移、无梯度rollout和训练入口只加载验证，机器人PPO更新次数为0，未创建新的SwanLab训练实验。

## 一键启动

```bash
cd /mnt/sage_data/workspace/KUAVO_S54_Newton
bash scripts/history_train/train_stride_cadence_3h.sh
```

脚本会建立独立日志目录、在线上传SwanLab，并从以下模型完整恢复actor、critic、归一化和Adam状态，自动迁移103→105维，原文件只读：

`logs/straight/2026-10-29_23-43-32_random_step_6to12cm/training/00064_straight/model_interrupted.pt`

该文件是上一轮时间预算结束时的有效最终保存，原iteration为6789。新实验的新增update/sample计数从0开始。

| 参数 | 本次启动值 | 对照 |
|---|---|---|
| 墙钟预算 | 3小时，含初始化与eval | 上轮12小时 |
| 并行环境 / rollout | 2048 / 64 | 沿用 |
| PPO epochs / minibatches | 4 / 4 | 沿用 |
| LR / schedule | 1e-4 / fixed | 沿用 |
| gamma / GAE lambda | .995 / .95 | 沿用 |
| entropy coefficient | .002 | 沿用，std从旧模型恢复 |
| 物理 / 控制频率 | 500 / 50 Hz | 沿用 |
| episode | 10秒 / 500控制步 | 沿用 |
| 水平目标步长 | 每episode Uniform(.06,.15)m | 原来.06–.12m |
| 左 / 右节奏 | 分别独立Uniform(.65,.85)Hz | 原来均为固定.75Hz |
| 观测 | 105维：原87维 + 左右节奏2维 + Goal16维 | 旧103/104维可恢复或迁移 |
| 训练分段 | 250次更新 | 原来100；不改变PPO rollout大小 |
| 完整eval间隔 | 每4分段，即约1000次新增更新 | 原来每100次更新 |
| eval内容 | 随机档、3个固定步长、4个左右节奏角点 | 角点使用最大步长 |
| SwanLab | S54-Locomotion / online | 沿用原SDK和登录方式 |

每个训练分段继续保存checkpoint和上传曲线。通过eval后仍继续到时间预算。3小时是否能学好新命令尚未验证；预算耗尽时优先保存模型，可能没有对最后一个模型执行完整eval。

左右参数控制的是各自半周期时长：`T_L=1/(2*f_L)`、`T_R=1/(2*f_R)`；严格交替，因此两只脚长期完成的循环次数相同，总循环频率为`2*f_L*f_R/(f_L+f_R)`。例如.65/.85Hz对应左.769秒、右.588秒。跨换脚边界后按剩余时间使用下一侧频率，避免相位超调误差。命令仅在reset采样，episode内保持不变。

奖励权重、action scale、动作滤波、PD和力矩限制均未改变。`rewards.py`仅新增日志参考和归一化函数，PPO仍使用原始奖励。

## 归一化回报

新增指标均在[0,1]内，采用固定公式而非训练历史min/max：

`normalized_return = clip(sum(reward)/sum(phase_reference_upper),0,1) * min(episode_steps/500,1)`

逐相位参考上界考虑实际权重、dt、摆动中段/末段窗口，以及clearance的站立基准归一化。动作变化惩罚的参考最好值为0。该参考是分项理论上界，不保证物理上能同时达到。

- `straight/Episode/normalized_return`：已完成episode的归一化回报，包含提前终止的时长损失。
- `straight/Reward/normalized`：逐控制步奖励质量，便于episode完成前观察趋势。
- `straight/Episode/normalized_quality`：去掉存活时长因子的episode质量。
- `straight/Episode/survival_fraction`：episode时长占完整时长的比例。
- `Eval/straight/normalized_return`：eval内已完成episode得分的均值；没有完成episode时本地JSON为null，不伪造完成回报。

例如奖励质量.8、只坚持半个episode，归一化回报为.4。原始`Reward/total`和累计回报仍保留，归一化指标不反馈给PPO。

该值不是严格落脚成功率。现有奖励touchdown容差5cm，eval容差3cm；高归一化回报仍可能伴随落脚失败，需要同时看左右成功率、连续成功步数、方向误差及跌倒数。

## 实际验证

1. 30项CPU回归全部通过，覆盖独立采样、局部reset、种子、跨边界剩余时间、等频1000步逐元素一致、103/104/105维布局、模型/Adam迁移、归一化边界、eval间隔与失败角点不得通过。合成模型测试执行Adam步骤检查形状，不是机器人PPO训练。
2. 原观测、奖励、课程回归脚本通过。
3. Pyright basic检查48个Python文件：0错误、0警告。
4. 实际最终checkpoint迁移：1024组输入的actor动作最大差异0、critic输出最大差异0；归一化count保留，4个Adam矩阵补零扩维，iteration仍为6789。
5. `train.py --validate_only` GPU检查通过：导出的env_cfg.json和实际网络均105维，完整恢复PPO状态，optimizer_updates=0。
6. GPU rollout结果如下，均无优化器更新；相位、奖励重算误差均为0，验证了reset后的命令与累计量处理。

| 场景 | 环境×控制步 | 维度 | terminated / truncated | 归一化episode回报 | 左 / 右严格成功率 |
|---|---:|---:|---:|---:|---:|
| 旧模型，固定12cm，.75/.75Hz | 8×600 | 103 | 0 / 8 | .96993 | .875 / 1.000 |
| 旧模型迁移，固定12cm，.65/.85Hz | 8×600 | 105 | 0 / 8 | .93250 | .8594 / .3594 |
| 随机动作，随机步长/左右节奏 | 8×400 | 105 | 46 / 0 | .04573 | 0 / 0 |

非对称节奏下右脚成功率明显下降，说明新任务仍需学习。迁移保持初始函数，不代表旧策略已经掌握新命令。随机动作短测验证了提前跌倒会显著降低归一化episode回报。

## 修改文件

| 文件 | 作用 |
|---|---|
| tasks/foot_placement/gait.py（新增） | 有效循环频率、跨边界相位推进 |
| tasks/foot_placement/goal_sampler.py | 左右独立命令、范围验证、episode级采样 |
| tasks/foot_placement/spec.py、observations.py | 103/104/105维契约，命令位于Goal之前 |
| tasks/foot_placement/foot_placement_env_cfg.py、foot_placement_env.py | 命令参数、观测接入、统计与reset清零 |
| tasks/foot_placement/rewards.py | 新增参考上界和归一化记录；原奖励公式/权重不变 |
| configs/s54_curriculum.py | 配置恢复与导出前维度同步 |
| agents/checkpoint_compat.py（新增） | actor/critic、归一化、Adam迁移及共用加载入口 |
| scripts/migrate_gait_checkpoint.py（新增） | 独立目录离线迁移，不覆盖原模型 |
| scripts/gait_cli.py（新增） | 训练/回放/eval/监督器共用参数 |
| scripts/train.py、play.py | 自动加载迁移、命令覆盖与维度同步 |
| scripts/preflight_signals.py | 独立节奏校验、真实名义速度、左右失败原因、归一化回报 |
| scripts/train_4060.py | 参数转发、分布恢复保护、低频eval和角点门槛、SwanLab曲线 |
| scripts/history_train/train_stride_cadence_3h.sh（新增） | 3小时一键启动与dry-run |
| tests/unit/test_gait_commands.py（新增）、test_train_4060.py | CPU回归 |

## 可复现产物与命令

- `before_source.tar.gz`：接管时的源文件快照，包含用户此前的手工修改。
- `before_sha256.json`、`after_sha256.json`、`changed_files.json`、`changes.patch`：修改记录。
- `migration_validation.json`、`migrated_final/model_105.pt`：实际迁移验证与独立副本。
- `train_validate_105/`、`train_validate_105.log`：完整加载检查与实际导出参数。
- `cpu_tests.log`、`static_check.json`、`launch_dry_run.log`：回归、类型检查与启动命令。
- `eval_legacy_103.*`、`eval_asymmetric.*`、`eval_random_resets.*`：GPU短测原始结果与轨迹。

```bash
# 预览启动命令，不创建训练实验或上传数据
bash scripts/history_train/train_stride_cadence_3h.sh --dry-run

# CPU回归
/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python -m unittest \
  tests.unit.test_gait_commands tests.unit.test_random_steps \
  tests.unit.test_heading_control tests.unit.test_train_4060

# 示例：自定义两侧不同的范围
bash scripts/history_train/train_stride_cadence_3h.sh --left-range .65 .75 --right-range .75 .85

# 中断后继续同一实验；恢复原命令分布与云端实验身份
bash scripts/history_train/train_stride_cadence_3h.sh --resume-dir logs/straight/实际运行目录
```
