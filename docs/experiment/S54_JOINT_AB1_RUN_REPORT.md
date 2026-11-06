# S54_JOINT_AB1 运行报告

生成时间：2026-11-06T14:53:17+08:00。
实验：`s54_joint_ab1_20261106_025912_seed54`；模式：`formal`；状态：**COMPLETED_BY_WALL_CAP**。
实际iteration：**14484**；累计运行wall：**42419 s** / 43200 s。
停止/失败原因：Training stopped with reserved final eval/checkpoint time。

本实验是 [SuperMimic Appendix B/G.1](https://arxiv.org/pdf/2610.28480) 的Foot Goal扩展：双接口在同一训练进程、同一Newton Scene中采样，共享唯一WM与Decoder；Foot只更新Encoder，Motion更新Motion Encoder与Decoder。它使用Stage1 warm-start，并非原论文Velocity从零实验的复现。

## 固定配置与来源

用户最新要求为12h，正式配置为43200s / 50000 iterations双上限；旧方案中的8h/15000为较早提案。初始化、评估、保存计入运行wall；resume保留已用预算，进程未运行期间不计入累计计算wall。

Teacher：[logs/s54_stage1_walk_20261105_014722/ckpt_004772.pt](../../logs/s54_stage1_walk_20261105_014722/ckpt_004772.pt)；SHA256 `1cf556f384cefa8635975ca4f8cb8c0e144a98292c5fb2fb8cff33e9b07c3fc4`。
源预训练：iteration 4772、12478464 real transitions、7170.713 s，单列成本。
源码版本：`ac0cab7f1dc9837d1f13169ccddb9a8d87cf0ade`；工作树逐文件哈希以 [logs/s54_joint_ab1_20261106_025912_seed54/source_snapshot/manifest.json](../../logs/s54_joint_ab1_20261106_025912_seed54/source_snapshot/manifest.json) 为准。

| 参数 | 固定值 |
| --- | --- |
| experiment | `S54_JOINT_AB1` |
| version | `1` |
| mode | `formal` |
| teacher | `logs/s54_stage1_walk_20261105_014722/ckpt_004772.pt` |
| teacher_sha256 | `1cf556f384cefa8635975ca4f8cb8c0e144a98292c5fb2fb8cff33e9b07c3fc4` |
| reference_manifest | `results/s54/motion_prior_stage0_verified/manifest.json` |
| foot_task_config | `configs/motion_prior/stage2/stage2_walking.json` |
| foot_config | `configs/motion_prior/stage2/conservative_footsteps.json` |
| wall_seconds | `43200` |
| iterations | `50000` |
| num_envs_motion | `512` |
| num_envs_foot | `512` |
| hist | `10` |
| goal_frames | `10` |
| motion_goal_dt | `0.02` |
| foot_goal_dt | `0.1` |
| wm_window | `8` |
| wm_batches | `16` |
| wm_batch | `1024` |
| horizon | `32` |
| pi_batch_motion | `128` |
| pi_batch_foot | `128` |
| buffer_steps | `1024` |
| warmup_steps | `128` |
| wm_only_iters | `200` |
| recovery_wm_iters | `20` |
| utd | `0.4` |
| lr_wm | `5e-05` |
| lr_motion_encoder | `2e-05` |
| lr_decoder | `2e-05` |
| lr_foot_encoder | `0.0001` |
| lr_critic | `0.0001` |
| lr_decay_iters | `50000` |
| gamma | `0.99` |
| lam | `0.95` |
| action_noise | `0.03` |
| critic_epochs | `2` |
| critic_minibatches | `4` |
| amp | `bf16` |
| seed | `54` |
| prior_config | `None` |
| paired_latent_weight | `0.0` |
| log_every | `20` |
| gradient_audit_every | `100` |
| checkpoint_every | `500` |
| checkpoint_seconds | `1800` |
| keep_checkpoints | `6` |
| evaluation_seconds | `1800` |
| full_evaluation_seconds | `21600` |
| evaluation_timeout | `600` |
| full_evaluation_timeout | `900` |
| final_reserve_seconds | `960` |
| motion_probe_offsets | `[0, 60, 120]` |
| motion_probe_duration | `10` |
| motion_retention_drop | `0.05` |
| motion_retention_patience | `2` |
| max_actor_gradnorm | `1000` |
| max_critic_loss | `10000` |
| unstable_update_patience | `10` |
| foot_eval_seeds | `[54, 55, 56]` |
| foot_eval_durations | `[2, 5, 10]` |
| swanlab_project | `Foothold_with_SuperMimic` |
| swanlab_workspace | `yagami_light` |

启动命令/时间：[logs/s54_joint_ab1_20261106_025912_seed54/launch.json](../../logs/s54_joint_ab1_20261106_025912_seed54/launch.json)；参数：[logs/s54_joint_ab1_20261106_025912_seed54/args.json](../../logs/s54_joint_ab1_20261106_025912_seed54/args.json)；逐tensor迁移：[logs/s54_joint_ab1_20261106_025912_seed54/transfer_mapping.json](../../logs/s54_joint_ab1_20261106_025912_seed54/transfer_mapping.json)。

Motion使用QA通过的LAFAN1 walk1_subject1（730D Goal、dt .02s），Foot使用230D Goal、dt .1s的100%保守alternating Quintic，未加载全身reference。Foot步长增量U[.02,.06]m、swing U[.65,.90]s、DS U[.15,.25]s、高度U[.03,.045]m、yaw=0；reset采样整段计划。原Reward、物理终止和控制保持不变。
Motion原域随机化及噪声；Foot原固定域和配置reset噪声。两套Goal/ObsNorm独立；Motion严格加载合法源Norm与兼容Critic，Foot新建；所有optimizer新建。Foot最后6D观测mask为显式有损适配。WM133→70、状态71、action29、ξ33、控制dt .02s；Goal不输入WM。

## 实际样本与资源

| 计数 | 当前值 |
| --- | --- |
| real_motion | 18605056 |
| real_foot | 18605056 |
| wm_updates | 231744 |
| wm_windows_motion | 118652928 |
| wm_windows_foot | 118652928 |
| wm_valid_steps_motion | 937408364 |
| wm_valid_steps_foot | 922348324 |
| actor_motion | 14284 |
| actor_foot | 14284 |
| critic_motion | 114272 |
| critic_foot | 114272 |
| imagined_motion | 58507264 |
| imagined_foot | 58507264 |
| recovery_real_motion | 0 |
| recovery_real_foot | 0 |
| eval_real_motion | 47565 |
| eval_real_foot | 25834 |
| events | 749 |

真实训练合计 **37210112**；加一次源预训练共 **49688576**。独立评估另计 **73399**。
real_*包含warmup与恢复补采，recovery_real_*是子集，不重复相加。WM windows/valid steps为Replay重用，imagined_*为BPTT预测，均不是新增Newton样本。UTD=.4平均每iteration采2.5个vector steps；H32=.64s想象，WM window8=.16s监督，均不是PPO rollout。每iteration总16次WM更新，每次M/F各512 windows。

资源证据：[logs/s54_joint_ab1_20261106_025912_seed54/hardware_before.json](../../logs/s54_joint_ab1_20261106_025912_seed54/hardware_before.json)、[logs/s54_joint_ab1_20261106_025912_seed54/runtime_stats.json](../../logs/s54_joint_ab1_20261106_025912_seed54/runtime_stats.json)。Torch峰值不含Warp；device_used为保存时快照，非全程峰值；collection FPS仅计采集，不代表端到端训练吞吐。

```json
{
  "iteration": 14484,
  "gpu": "NVIDIA GeForce RTX 3090",
  "peak_allocated_MiB": 2783.75830078125,
  "peak_reserved_MiB": 2840.0,
  "device_used_snapshot_MiB": 3980.0625,
  "collection_fps": 8342.443491163967,
  "wall_seconds": 42419.000219393,
  "counters": {
    "real_motion": 18605056,
    "real_foot": 18605056,
    "wm_updates": 231744,
    "wm_windows_motion": 118652928,
    "wm_windows_foot": 118652928,
    "wm_valid_steps_motion": 937408364,
    "wm_valid_steps_foot": 922348324,
    "actor_motion": 14284,
    "actor_foot": 14284,
    "critic_motion": 114272,
    "critic_foot": 114272,
    "imagined_motion": 58507264,
    "imagined_foot": 58507264,
    "recovery_real_motion": 0,
    "recovery_real_foot": 0,
    "eval_real_motion": 47565,
    "eval_real_foot": 25834,
    "events": 749
  }
}
```

## Gate、梯度隔离与恢复

门槛记录：[logs/s54_joint_ab1_20261106_025912_seed54/preflight_gates.json](../../logs/s54_joint_ab1_20261106_025912_seed54/preflight_gates.json)。

```json
{
  "passed": true,
  "num_envs_per_interface": 512,
  "cpu_tests": 54,
  "cpu_log": "results/s54/joint_ab1_preflight_20261106_021910/cpu_tests_release.log",
  "proofs": {
    "one": {
      "run": "logs/s54_joint_ab1_smoke_release1_20261106_025731_seed54",
      "status": "COMPLETED_BY_ITER_CAP",
      "iteration": 4,
      "counters": {
        "real_motion": 138,
        "real_foot": 138,
        "wm_updates": 4,
        "wm_windows_motion": 32,
        "wm_windows_foot": 32,
        "wm_valid_steps_motion": 256,
        "wm_valid_steps_foot": 256,
        "actor_motion": 3,
        "actor_foot": 3,
        "critic_motion": 24,
        "critic_foot": 24,
        "imagined_motion": 384,
        "imagined_foot": 384,
        "recovery_real_motion": 0,
        "recovery_real_foot": 0,
        "eval_real_motion": 0,
        "eval_real_foot": 0,
        "events": 5
      },
      "checkpoint_sha256": "bddaaac865fea4f4f92f52e6b121cd8f0390eb27c860b37f27024b6dff438e2f",
      "gradient_audits": 6,
      "runtime": {
        "iteration": 4,
        "gpu": "NVIDIA GeForce RTX 3090",
        "peak_allocated_MiB": 1418.1787109375,
        "peak_reserved_MiB": 1482.0,
        "device_used_snapshot_MiB": 2396.0625,
        "collection_fps": 25.46102493006525,
        "wall_seconds": 28.445491157996003,
        "counters": {
          "real_motion": 138,
          "real_foot": 138,
          "wm_updates": 4,
          "wm_windows_motion": 32,
          "wm_windows_foot": 32,
          "wm_valid_steps_motion": 256,
          "wm_valid_steps_foot": 256,
          "actor_motion": 3,
          "actor_foot": 3,
          "critic_motion": 24,
          "critic_foot": 24,
          "imagined_motion": 384,
          "imagined_foot": 384,
          "recovery_real_motion": 0,
          "recovery_real_foot": 0,
          "eval_real_motion": 0,
          "eval_real_foot": 0,
          "events": 5
        }
      }
    },
    "scale256": {
      "run": "logs/s54_joint_ab1_smoke_scale256_20261106_021910_seed54",
      "status": "COMPLETED_BY_ITER_CAP",
      "iteration": 8,
      "counters": {
        "real_motion": 37888,
        "real_foot": 37888,
        "wm_updates": 128,
        "wm_windows_motion": 65536,
        "wm_windows_foot": 65536,
        "wm_valid_steps_motion": 524130,
        "wm_valid_steps_foot": 524097,
        "actor_motion": 6,
        "actor_foot": 6,
        "critic_motion": 48,
        "critic_foot": 48,
        "imagined_motion": 24576,
        "imagined_foot": 24576,
        "recovery_real_motion": 0,
        "recovery_real_foot": 0,
        "eval_real_motion": 0,
        "eval_real_foot": 0,
        "events": 9
      },
      "checkpoint_sha256": "4a47e6ae7f76e92b75bc771c9ab38dad435e508b1451a4b626d2b4b4acbab762",
      "gradient_audits": 12,
      "runtime": {
        "iteration": 8,
        "gpu": "NVIDIA GeForce RTX 3090",
        "peak_allocated_MiB": 2275.79052734375,
        "peak_reserved_MiB": 2334.0,
        "device_used_snapshot_MiB": 3378.0625,
        "collection_fps": 5971.528640227624,
        "wall_seconds": 44.010669432987925,
        "counters": {
          "real_motion": 37888,
          "real_foot": 37888,
          "wm_updates": 128,
          "wm_windows_motion": 65536,
          "wm_windows_foot": 65536,
          "wm_valid_steps_motion": 524130,
          "wm_valid_steps_foot": 524097,
          "actor_motion": 6,
          "actor_foot": 6,
          "critic_motion": 48,
          "critic_foot": 48,
          "imagined_motion": 24576,
          "imagined_foot": 24576,
          "recovery_real_motion": 0,
          "recovery_real_foot": 0,
          "eval_real_motion": 0,
          "eval_real_foot": 0,
          "events": 9
        }
      }
    },
    "scale512": {
      "run": "logs/s54_joint_ab1_smoke_scale512_20261106_021910_seed54",
      "status": "COMPLETED_BY_ITER_CAP",
      "iteration": 12,
      "counters": {
        "real_motion": 80896,
        "real_foot": 80896,
        "wm_updates": 192,
        "wm_windows_motion": 98304,
        "wm_windows_foot": 98304,
        "wm_valid_steps_motion": 786250,
        "wm_valid_steps_foot": 785797,
        "actor_motion": 10,
        "actor_foot": 10,
        "critic_motion": 80,
        "critic_foot": 80,
        "imagined_motion": 40960,
        "imagined_foot": 40960,
        "recovery_real_motion": 0,
        "recovery_real_foot": 0,
        "eval_real_motion": 0,
        "eval_real_foot": 0,
        "events": 13
      },
      "checkpoint_sha256": "c23da5506e7dcba6d36c360cc947ed4e405e07a4e57a6a03c6943dbfb922a5ba",
      "gradient_audits": 20,
      "runtime": {
        "iteration": 12,
        "gpu": "NVIDIA GeForce RTX 3090",
        "peak_allocated_MiB": 2752.34814453125,
        "peak_reserved_MiB": 2808.0,
        "device_used_snapshot_MiB": 3948.0625,
        "collection_fps": 11112.371735551787,
        "wall_seconds": 63.855788259010296,
        "counters": {
          "real_motion": 80896,
          "real_foot": 80896,
          "wm_updates": 192,
          "wm_windows_motion": 98304,
          "wm_windows_foot": 98304,
          "wm_valid_steps_motion": 786250,
          "wm_valid_steps_foot": 785797,
          "actor_motion": 10,
          "actor_foot": 10,
          "critic_motion": 80,
          "critic_foot": 80,
          "imagined_motion": 40960,
          "imagined_foot": 40960,
          "recovery_real_motion": 0,
          "recovery_real_foot": 0,
          "eval_real_motion": 0,
          "eval_real_foot": 0,
          "events": 13
        }
      }
    },
    "release512": {
      "run": "logs/s54_joint_ab1_smoke_release512_20261106_025600_seed54",
      "status": "COMPLETED_BY_ITER_CAP",
      "iteration": 12,
      "counters": {
        "real_motion": 80896,
        "real_foot": 80896,
        "wm_updates": 192,
        "wm_windows_motion": 98304,
        "wm_windows_foot": 98304,
        "wm_valid_steps_motion": 786268,
        "wm_valid_steps_foot": 785847,
        "actor_motion": 10,
        "actor_foot": 10,
        "critic_motion": 80,
        "critic_foot": 80,
        "imagined_motion": 40960,
        "imagined_foot": 40960,
        "recovery_real_motion": 0,
        "recovery_real_foot": 0,
        "eval_real_motion": 0,
        "eval_real_foot": 0,
        "events": 13
      },
      "checkpoint_sha256": "dc86795e14b39c4c518975db2152a82405023a5528b06d802112ba2f5249d12a",
      "gradient_audits": 20,
      "runtime": {
        "iteration": 12,
        "gpu": "NVIDIA GeForce RTX 3090",
        "peak_allocated_MiB": 2752.34326171875,
        "peak_reserved_MiB": 2808.0,
        "device_used_snapshot_MiB": 3948.0625,
        "collection_fps": 11371.272301632811,
        "wall_seconds": 64.43346343497979,
        "counters": {
          "real_motion": 80896,
          "real_foot": 80896,
          "wm_updates": 192,
          "wm_windows_motion": 98304,
          "wm_windows_foot": 98304,
          "wm_valid_steps_motion": 786268,
          "wm_valid_steps_foot": 785847,
          "actor_motion": 10,
          "actor_foot": 10,
          "critic_motion": 80,
          "critic_foot": 80,
          "imagined_motion": 40960,
          "imagined_foot": 40960,
          "recovery_real_motion": 0,
          "recovery_real_foot": 0,
          "eval_real_motion": 0,
          "eval_real_foot": 0,
          "events": 13
        }
      }
    },
    "resume16": {
      "run": "logs/s54_joint_ab1_smoke_eval_resume_v3_20261106_021910_seed54",
      "status": "COMPLETED_BY_ITER_CAP",
      "iteration": 8,
      "counters": {
        "real_motion": 4416,
        "real_foot": 4416,
        "wm_updates": 16,
        "wm_windows_motion": 512,
        "wm_windows_foot": 512,
        "wm_valid_steps_motion": 4096,
        "wm_valid_steps_foot": 4055,
        "actor_motion": 6,
        "actor_foot": 6,
        "critic_motion": 48,
        "critic_foot": 48,
        "imagined_motion": 3072,
        "imagined_foot": 3072,
        "recovery_real_motion": 2048,
        "recovery_real_foot": 2048,
        "eval_real_motion": 1500,
        "eval_real_foot": 762,
        "events": 11
      },
      "checkpoint_sha256": "3c5859e93ffab51710cab3a2312ea3ec4e207cb3257e3c9b4a7a44d7ad888047",
      "gradient_audits": 12,
      "runtime": {
        "iteration": 8,
        "gpu": "NVIDIA GeForce RTX 3090",
        "peak_allocated_MiB": 1461.9697265625,
        "peak_reserved_MiB": 1516.0,
        "device_used_snapshot_MiB": 2430.0625,
        "collection_fps": 398.47841443279896,
        "wall_seconds": 188.79172224598005,
        "counters": {
          "real_motion": 4416,
          "real_foot": 4416,
          "wm_updates": 16,
          "wm_windows_motion": 512,
          "wm_windows_foot": 512,
          "wm_valid_steps_motion": 4096,
          "wm_valid_steps_foot": 4055,
          "actor_motion": 6,
          "actor_foot": 6,
          "critic_motion": 48,
          "critic_foot": 48,
          "imagined_motion": 3072,
          "imagined_foot": 3072,
          "recovery_real_motion": 2048,
          "recovery_real_foot": 2048,
          "eval_real_motion": 1500,
          "eval_real_foot": 762,
          "events": 11
        }
      }
    }
  },
  "evaluations": {
    "motion": "logs/s54_joint_ab1_smoke_eval_resume_v3_20261106_021910_seed54/evaluation/iteration_000000_1793903861163416462/motion/summary.json",
    "foot": "logs/s54_joint_ab1_smoke_eval_resume_v3_20261106_021910_seed54/evaluation/iteration_000000_1793903861163416462/foot/summary.json"
  },
  "video_summary": "results/s54/joint_ab1_preflight_20261106_021910/media_preflight.json",
  "video": "results/s54/joint_ab1_preflight_20261106_021910/preview_motion.gif",
  "preserved_failed_preflight": {
    "run": "s54_joint_ab1_smoke_eval_resume_v2_20261106_021910_seed54",
    "reason": "paired_predictions tuple was incorrectly treated as summary dict; fixed and regression-tested"
  },
  "limitations": "Engineering gates only; initial Foot survival fails and no convergence/holdout claim. Resume refills Replay; Newton contact warm state is fresh."
}
```

| 实测接口 | 审计数 | ΔD min/max | ΔEncoder min/max | latent grad min | ΔWM_actor max |
| --- | --- | --- | --- | --- | --- |
| motion | 142 | 0.008080651 / 0.01286752 | 0.007275579 / 0.01133346 | 0.001879186 | 0 |
| foot | 142 | 0 / 0 | 0.02647034 / 0.05819876 | 0.004061607 | 0 |

原始梯度证据：[logs/s54_joint_ab1_20261106_025912_seed54/gradient_audit.jsonl](../../logs/s54_joint_ab1_20261106_025912_seed54/gradient_audit.jsonl)。Motion optimizer只持有E_m+D，Foot只持有E_f；每次Actor均检查WM/另一Encoder/Foot-D未被更新，审计周期另测差值。CPU测试覆盖共享identity、Critic隔离、旧Motion推理、Replay命令快照/跨reset掩码、合法与错误恢复。

原子Checkpoint包含WM、唯一D、两Encoder/Critic、五个optimizer、Norm、RNG、计数、场景姿态/时钟/命令及观测历史。周期500iter或30min，保留6个周期版本，另保留initial/last/last_good_motion/best_foot。Replay不序列化：恢复补采warmup并有界WM-only recovery，成本单列。Newton接触暖启动未序列化，不承诺逐位轨迹连续。达到预算或有live pane时，resume在修改终态文件之前拒绝。

## 分接口训练趋势

曲线：[logs/s54_joint_ab1_20261106_025912_seed54/train_log.csv](../../logs/s54_joint_ab1_20261106_025912_seed54/train_log.csv)、[logs/s54_joint_ab1_20261106_025912_seed54/interface_metrics.jsonl](../../logs/s54_joint_ab1_20261106_025912_seed54/interface_metrics.jsonl)；SwanLab将Motion与Foot分开，不平均成一个Reward。

| 接口 | 窗口 | 条数 | sim reward | imag reward | episode return | ep steps | term fraction | WM8 | Actor grad | Critic loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| motion | 初期 20–200 | 10 | 0.8175783 | N/A | 0.7228878 | 279.1389 | 0 | 0.03473789 | N/A | N/A |
| motion | 中段 7160–7340 | 10 | 0.8245525 | 0.8440278 | 0.4740356 | 240.4644 | 0.0008333333 | 0.02392447 | 2.959711 | 0.004780889 |
| motion | 末期 14300–14480 | 10 | 0.8409237 | 0.857812 | 0.5299776 | 242.0328 | 0.001587302 | 0.02020268 | 2.789029 | 0.002579663 |
| foot | 初期 20–200 | 10 | 0.2887881 | N/A | 0.4640908 | 92.14645 | 0.939685 | 0.07541009 | N/A | N/A |
| foot | 中段 7160–7340 | 10 | 0.3302989 | 0.243739 | 0.3648229 | 118.7545 | 1 | 0.01514007 | 0.702138 | 0.001558638 |
| foot | 末期 14300–14480 | 10 | 0.3388911 | 0.2563475 | 0.3480563 | 137.9794 | 1 | 0.01190238 | 0.7803963 | 0.001557341 |

为CSV记录等权均值；早期窗口可能重叠，WM-only Actor空白为N/A。sim reward是单步/r_max，episode return除以实际执行步数，ep_len仅含已结束episode；term fraction=物理失败/done，Timeout/ClipEnd不当跌倒。imag reward来自Replay且乘soft survival，不能与sim均值之差当作WM bias。

## 独立Newton评估

主训练同步CUDA并暂停后，两个评估子进程串行运行。Motion先测固定offset0/60/120各10s的同协议Teacher基线；完整261.3s的0.820197只用于全片比较。Foot固定seed54/55/56与2/5/10s，物理终止优先于Timeout，终止后的后缀不补Reward。

| 接口/协议 | iteration | 平均NR | 物理终止/episodes | 实际秒均值 | 证据 |
| --- | --- | --- | --- | --- | --- |
| foot/short | 0 | 0.1179425 | 9/9 | 1.691111 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_000000_1793905169377155493/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_000000_1793905169377155493/foot/summary.json) |
| motion/short | 0 | 0.8615972 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_000000_1793905169377155493/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_000000_1793905169377155493/motion/summary.json) |
| foot/short | 769 | 0.2475893 | 6/9 | 3.271111 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_000769_1793907119406321607/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_000769_1793907119406321607/foot/summary.json) |
| motion/short | 769 | 0.8913507 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_000769_1793907119406321607/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_000769_1793907119406321607/motion/summary.json) |
| foot/short | 1459 | 0.2144451 | 6/9 | 2.648889 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_001459_1793909097076036806/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_001459_1793909097076036806/foot/summary.json) |
| motion/short | 1459 | 0.8866256 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_001459_1793909097076036806/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_001459_1793909097076036806/motion/summary.json) |
| foot/short | 2127 | 0.210578 | 6/9 | 2.597778 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_002127_1793911060308866588/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_002127_1793911060308866588/foot/summary.json) |
| motion/short | 2127 | 0.8729643 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_002127_1793911060308866588/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_002127_1793911060308866588/motion/summary.json) |
| foot/short | 2802 | 0.2075649 | 6/9 | 2.475556 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_002802_1793913018654430501/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_002802_1793913018654430501/foot/summary.json) |
| motion/short | 2802 | 0.8761602 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_002802_1793913018654430501/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_002802_1793913018654430501/motion/summary.json) |
| foot/short | 3471 | 0.2210963 | 6/9 | 2.573333 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_003471_1793914977500028213/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_003471_1793914977500028213/foot/summary.json) |
| motion/short | 3471 | 0.8857247 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_003471_1793914977500028213/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_003471_1793914977500028213/motion/summary.json) |
| foot/short | 4140 | 0.2139636 | 6/9 | 2.555556 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_004140_1793916934583684647/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_004140_1793916934583684647/foot/summary.json) |
| motion/short | 4140 | 0.8794188 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_004140_1793916934583684647/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_004140_1793916934583684647/motion/summary.json) |
| foot/short | 4826 | 0.2157429 | 6/9 | 2.555556 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_004826_1793918890564096294/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_004826_1793918890564096294/foot/summary.json) |
| motion/short | 4826 | 0.8736431 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_004826_1793918890564096294/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_004826_1793918890564096294/motion/summary.json) |
| foot/short | 5507 | 0.2202648 | 6/9 | 2.435556 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_005507_1793920849129239912/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_005507_1793920849129239912/foot/summary.json) |
| motion/short | 5507 | 0.8827969 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_005507_1793920849129239912/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_005507_1793920849129239912/motion/summary.json) |
| foot/short | 6192 | 0.2210586 | 6/9 | 2.615556 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_006192_1793922802585637256/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_006192_1793922802585637256/foot/summary.json) |
| motion/short | 6192 | 0.883786 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_006192_1793922802585637256/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_006192_1793922802585637256/motion/summary.json) |
| foot/short | 6852 | 0.2187291 | 6/9 | 2.602222 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_006852_1793924759529955629/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_006852_1793924759529955629/foot/summary.json) |
| motion/short | 6852 | 0.8977033 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_006852_1793924759529955629/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_006852_1793924759529955629/motion/summary.json) |
| foot/short | 7513 | 0.2181903 | 6/9 | 2.646667 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_007513_1793926719642080183/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_007513_1793926719642080183/foot/summary.json) |
| motion/short | 7513 | 0.8706979 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_007513_1793926719642080183/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_007513_1793926719642080183/motion/summary.json) |
| motion/full | 7513 | 0.8247583 | 0/1 | 261.3 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_007513_1793926879580708018/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_007513_1793926879580708018/motion/summary.json) |
| foot/short | 8033 | 0.2128271 | 6/9 | 2.611111 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_008033_1793928680149831633/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_008033_1793928680149831633/foot/summary.json) |
| motion/short | 8033 | 0.8750935 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_008033_1793928680149831633/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_008033_1793928680149831633/motion/summary.json) |
| foot/short | 8711 | 0.1953788 | 7/9 | 2.082222 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_008711_1793930639262834709/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_008711_1793930639262834709/foot/summary.json) |
| motion/short | 8711 | 0.8736174 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_008711_1793930639262834709/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_008711_1793930639262834709/motion/summary.json) |
| foot/short | 9387 | 0.1953429 | 8/9 | 1.635556 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_009387_1793932584984329976/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_009387_1793932584984329976/foot/summary.json) |
| motion/short | 9387 | 0.8721629 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_009387_1793932584984329976/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_009387_1793932584984329976/motion/summary.json) |
| foot/short | 10068 | 0.2110741 | 7/9 | 2.082222 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_010068_1793934519481904539/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_010068_1793934519481904539/foot/summary.json) |
| motion/short | 10068 | 0.8868333 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_010068_1793934519481904539/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_010068_1793934519481904539/motion/summary.json) |
| foot/short | 10741 | 0.2216039 | 6/9 | 2.513333 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_010741_1793936465133383290/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_010741_1793936465133383290/foot/summary.json) |
| motion/short | 10741 | 0.890106 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_010741_1793936465133383290/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_010741_1793936465133383290/motion/summary.json) |
| foot/short | 11423 | 0.2292541 | 6/9 | 2.573333 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_011423_1793938421121704477/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_011423_1793938421121704477/foot/summary.json) |
| motion/short | 11423 | 0.892087 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_011423_1793938421121704477/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_011423_1793938421121704477/motion/summary.json) |
| foot/short | 12095 | 0.2312012 | 6/9 | 2.624444 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_012095_1793940381457316197/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_012095_1793940381457316197/foot/summary.json) |
| motion/short | 12095 | 0.8616949 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_012095_1793940381457316197/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_012095_1793940381457316197/motion/summary.json) |
| foot/short | 12778 | 0.2356903 | 6/9 | 2.631111 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_012778_1793942340789305234/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_012778_1793942340789305234/foot/summary.json) |
| motion/short | 12778 | 0.8899488 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_012778_1793942340789305234/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_012778_1793942340789305234/motion/summary.json) |
| foot/short | 13446 | 0.2384845 | 6/9 | 2.695556 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_013446_1793944302190486575/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_013446_1793944302190486575/foot/summary.json) |
| motion/short | 13446 | 0.8753168 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_013446_1793944302190486575/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_013446_1793944302190486575/motion/summary.json) |
| foot/short | 14121 | 0.2381594 | 6/9 | 2.664444 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_014121_1793946263441632261/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_014121_1793946263441632261/foot/summary.json) |
| motion/short | 14121 | 0.8884666 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_014121_1793946263441632261/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_014121_1793946263441632261/motion/summary.json) |
| foot/short | 14484 | 0.2478737 | 6/9 | 2.626667 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_014484_1793947401205263713/foot/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_014484_1793947401205263713/foot/summary.json) |
| motion/short | 14484 | 0.8941111 | 0/3 | 10 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_014484_1793947401205263713/motion/summary.json](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_014484_1793947401205263713/motion/summary.json) |

### 最新Foot固定时限

| 时限 | Survival | NR | Foot RMSE/m | Scheduled TD/m | 速度/m/s | Slip proxy/m/s | contact准确率 | 原因 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2s | 3/3 | 0.4275562 | 0.734797 | 0.8222293 | -0.9122115 | 0.1225916 | 0.585 | ['timeout', 'timeout', 'timeout'] |
| 5s | 0/3 | 0.2110395 | 1.580195 | 1.098201 | -1.256479 | 0.1890048 | 0.5779433 | ['height', 'height', 'height'] |
| 10s | 0/3 | 0.1050254 | 1.572234 | 1.103262 | -1.259542 | 0.1970449 | 0.5714623 | ['height', 'height', 'height'] |

### 最新姿态及Motion跟踪

| 接口 | Root roll/pitch max deg | Body aligned RMSE/m | Root world XY RMSE/m | Upper pose RMSE/rad | Upper速度/加速度RMS |
| --- | --- | --- | --- | --- | --- |
| motion | 6.89315 / 9.470187 | 0.047797 | 0.133527 | 0.2903603 | 0.6838963 / 13.18014 |
| foot | 8.062491 / 19.74967 | N/A | N/A | NOT EVALUATED | 0.6717747 / 13.80865 |

除Root峰值跨episode取max，其余指标为episode等权均值，非池化RMSE。短失败前缀误差低不代表完整成功。Foot没有配对上身参考，pose误差NOT EVALUATED，不以动作幅度代替自然性。接触为Newton末子步Fz>1N；slip proxy为承载踝link XY速度，非接触点滑移；Scheduled TD为计划事件，非传感器实际触地时刻。

### WM同起点同动作配对

| 接口 | Horizon | Height RMSE/m | Pitch RMSE/deg | Foot FK RMSE/m | Normalized Reward MAE | Reward bias |
| --- | --- | --- | --- | --- | --- | --- |
| motion | 1 | 0.0001262964 | 0.01671602 | 0.001023706 | 0.001313054 | 0.00034571 |
| motion | 4 | 0.0003144015 | 0.07622022 | 0.002454644 | 0.0009255473 | 0.0004220743 |
| motion | 8 | 0.0004469516 | 0.1158385 | 0.004076633 | 0.0007898898 | 0.0004293722 |
| motion | 16 | 0.0008383899 | 0.3413475 | 0.006154728 | 0.0008190299 | 0.0004113195 |
| motion | 32 | 0.001393321 | 0.9374525 | 0.009374147 | 0.0009382246 | -2.761901e-05 |
| foot | 1 | 8.514801e-05 | 0.008255593 | 0.0005854299 | 0.0005749028 | -0.0002886386 |
| foot | 4 | 0.0002315808 | 0.032184 | 0.001487999 | 0.0004823203 | -0.0002795399 |
| foot | 8 | 0.0003190456 | 0.07446986 | 0.002508251 | 0.0004612988 | -0.0002301964 |
| foot | 16 | 0.000671017 | 0.2204581 | 0.005158036 | 0.0006196282 | 1.658785e-05 |
| foot | 32 | 0.002292303 | 0.7703168 | 0.01302299 | 0.001878978 | 0.001386024 |

这是已访问Newton起点、相同开环动作/ξ/时钟的WM预测误差；不是足对目标的tracking误差。Reward MAE为每窗口平均Reward差绝对值的均值，逐步MAE另存JSON，不混称。epoch/episode等权统计仍有覆盖局限，不认证全动作空间梯度。

## 产物、SwanLab和操作

| 产物 | 路径/状态 |
| --- | --- |
| last.pt | [logs/s54_joint_ab1_20261106_025912_seed54/last.pt](../../logs/s54_joint_ab1_20261106_025912_seed54/last.pt) |
| initial.pt | [logs/s54_joint_ab1_20261106_025912_seed54/initial.pt](../../logs/s54_joint_ab1_20261106_025912_seed54/initial.pt) |
| last_good_motion.pt | [logs/s54_joint_ab1_20261106_025912_seed54/last_good_motion.pt](../../logs/s54_joint_ab1_20261106_025912_seed54/last_good_motion.pt) |
| best_foot.pt | [logs/s54_joint_ab1_20261106_025912_seed54/best_foot.pt](../../logs/s54_joint_ab1_20261106_025912_seed54/best_foot.pt) |
| train_log.csv | [logs/s54_joint_ab1_20261106_025912_seed54/train_log.csv](../../logs/s54_joint_ab1_20261106_025912_seed54/train_log.csv) |
| interface_metrics.jsonl | [logs/s54_joint_ab1_20261106_025912_seed54/interface_metrics.jsonl](../../logs/s54_joint_ab1_20261106_025912_seed54/interface_metrics.jsonl) |
| gradient_audit.jsonl | [logs/s54_joint_ab1_20261106_025912_seed54/gradient_audit.jsonl](../../logs/s54_joint_ab1_20261106_025912_seed54/gradient_audit.jsonl) |
| wm_pair_metrics.jsonl | [logs/s54_joint_ab1_20261106_025912_seed54/wm_pair_metrics.jsonl](../../logs/s54_joint_ab1_20261106_025912_seed54/wm_pair_metrics.jsonl) |
| stdout.log | [logs/s54_joint_ab1_20261106_025912_seed54/stdout.log](../../logs/s54_joint_ab1_20261106_025912_seed54/stdout.log) |
| evaluation.log | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation.log](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation.log) |
| training_status.json | [logs/s54_joint_ab1_20261106_025912_seed54/training_status.json](../../logs/s54_joint_ab1_20261106_025912_seed54/training_status.json) |
| swanlab_cloud_final_readback.json | [logs/s54_joint_ab1_20261106_025912_seed54/swanlab_cloud_final_readback.json](../../logs/s54_joint_ab1_20261106_025912_seed54/swanlab_cloud_final_readback.json) |
| foot preview iter0 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_000000_1793905169377155493/foot/preview_foot.gif](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_000000_1793905169377155493/foot/preview_foot.gif) |
| motion preview iter0 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_000000_1793905169377155493/motion/preview_motion.gif](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_000000_1793905169377155493/motion/preview_motion.gif) |
| motion preview iter7513 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_007513_1793926879580708018/motion/preview_motion.gif](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_007513_1793926879580708018/motion/preview_motion.gif) |
| foot preview iter14484 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_014484_1793947401205263713/foot/preview_foot.gif](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_014484_1793947401205263713/foot/preview_foot.gif) |
| motion preview iter14484 | [logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_014484_1793947401205263713/motion/preview_motion.gif](../../logs/s54_joint_ab1_20261106_025912_seed54/evaluation/iteration_014484_1793947401205263713/motion/preview_motion.gif) |

视频只重放已有状态且限制到实际执行区间；Motion的并列LAFAN1 kinematic reference仅用于展示，不是真实Newton轨迹。

```json
{
  "run_id": "5a300fd7075d46c2",
  "experiment_name": "S54_JOINT_AB1_20261106_025912_seed54",
  "project": "Foothold_with_SuperMimic",
  "workspace": "yagami_light",
  "status": "verified",
  "updated_at": "2026-11-06T06:46:25.210906+00:00",
  "local_events": 749,
  "last_iteration": 14484,
  "upload_policy": "core_only_v1",
  "max_metric_count": 32,
  "url": "https://swanlab.cn/@yagami_light/Foothold_with_SuperMimic/runs/5a300fd7075d46c2/chart",
  "terminal_training_status": "COMPLETED_BY_WALL_CAP",
  "note": "unique event step; Progress/Iteration records actual learner iteration"
}
```

[SwanLab正式run](https://swanlab.cn/@yagami_light/Foothold_with_SuperMimic/runs/5a300fd7075d46c2/chart)，稳定ID `5a300fd7075d46c2`。

仅formal上传，同项目Foothold_with_SuperMimic，新时间戳实验。独立CPU/tmux不依赖SSH代理，异常重试，本地JSONL/SDK持久化；训练结束离线重放、sync及最终读回，只有verified与final_sync.exitcode=0才确认最终同步。唯一event_step为默认横轴，Progress/Iteration记录实际迭代。

按用户2026-11-06要求，上传器已改为32项核心指标白名单：Motion/Foot各10项（真实/想象Reward、Episode Return/Length、物理终止比例、WM Loss/Error8、Actor/Critic Loss及Actor梯度）；Motion短/全片评估Return各1项；Foot 2/5/10秒Survival、10秒Return/Foot RMSE/实际前向速度共6项；迭代/时长/FPS/峰值显存共4项。评估各offset/seed等权汇总，短评估与全片协议分开。本地JSONL、逐seed评估、WM多步误差和梯度审计保留完整；原始计数不再逐项上传。最终读回验收改为最新核心WM Loss及终态Iteration/WallHours。已上传的旧图表仍存在：当前SDK不支持修改已有图表的隐藏状态，需要在WebUI隐藏，不能声称已删除。完整策略见swanlab_core_policy.json。

```bash
tmux attach -t s54_joint_ab1_20261106_025912_seed54
tail -F logs/s54_joint_ab1_20261106_025912_seed54/stdout.log
/home/rm/workspace/IsaacLab_3_0/.venv/bin/python -B tools/train/run_s54_joint_ab1.py status --run s54_joint_ab1_20261106_025912_seed54
/home/rm/workspace/IsaacLab_3_0/.venv/bin/python -B tools/train/run_s54_joint_ab1.py stop --run s54_joint_ab1_20261106_025912_seed54
/home/rm/workspace/IsaacLab_3_0/.venv/bin/python -B tools/train/run_s54_joint_ab1.py resume --run s54_joint_ab1_20261106_025912_seed54
/home/rm/workspace/IsaacLab_3_0/.venv/bin/python -B tools/validation/report_s54_joint_ab1.py --run_dir logs/s54_joint_ab1_20261106_025912_seed54
```

按Ctrl-b再d离开tmux，SSH断开仍运行；服务器须保持开机。已有活跃训练/已耗尽预算的run拒绝resume，不加载旧ABC checkpoint。

## 结论边界与下一步

COMPLETED_BY_WALL_CAP/ITER_CAP仅表示有界工作流完成，不是收敛。Motion短探针的同协议Teacher绝对NR下降>.05或任一物理失败，连续两次触发STOPPED_BY_GATE；完整协议的实现仅检查平均NR是否低于.820197−.05，不另用物理失败触发，不能将两种Gate混称。Foot初步进展需10s Survival≥2/3且速度≥−.01m/s，仍需接触/自然性验收。
非有限数、持续超阈梯度/Critic loss、OOM或写Checkpoint失败均留故障并停止，不在线改Reward/终止/方法。当前缺独立Walking holdout与Standing/Turning模板，泛化NOT EVALUATED；单训练seed不能声称统计显著。
若Foot未达标，分别检查真实终止/速度、WM32配对及reward/bootstrap梯度；不以失败前缀RMSE或延长WM-only掩盖控制失败。后续对照须匹配Foot或总真实样本预算，并单列新增Motion监督与Stage1成本。

**此为已记录终态的证据汇总；能力与上传状态以上表为准。**

## 最终验收与补充证据

最终Checkpoint CPU审计：[results/s54/s54_joint_ab1_20261106_025912_seed54/final_checkpoint_audit.json](../../results/s54/s54_joint_ab1_20261106_025912_seed54/final_checkpoint_audit.json)，passed=True；278个张量均有限；五个Adam step与真实更新计数相符；RNG含Python/NumPy/Torch/CUDA；两套512×10×99历史、环境/命令/时钟及Replay schema均已保存。Shared Decoder只序列化一次。
最终last.pt SHA256：`ef5671d4e223ab004c1f69837fffca6f461ca1c2c661c111b5afc2f82e6e5544`。
**真实结论：Motion短探针NR=0.8941111，同协议初始Teacher=0.8615972；物理失败0/3。Foot 10s Survival=0/3，实际前向速度=-1.259542m/s；Foot进展门槛=FAIL。**
Foot短时存活改善不能等同足轨迹跟踪行走成功。Motion支路保留原有动作能力，也不能证明任意Foot Goal能获得自然上身；Foot无配对全身参考，本轮没有合格的自然性结论。
训练端到端吞吐（包含初始化/评估/保存）=877.2039 Newton transitions/s、0.3414508 iterations/s；与只计采集区间的FPS分开。

同合同零动作PD对照：[results/s54/s54_joint_ab1_20261106_025912_seed54/zero_pd_comparison.json](../../results/s54/s54_joint_ab1_20261106_025912_seed54/zero_pd_comparison.json)；真实原始评估 [results/s54/s54_joint_ab1_20261106_025912_seed54/zero_pd_newton/summary.json](../../results/s54/s54_joint_ab1_20261106_025912_seed54/zero_pd_newton/summary.json)；命令/预算/退出状态 [results/s54/s54_joint_ab1_20261106_025912_seed54/zero_pd_run.json](../../results/s54/s54_joint_ab1_20261106_025912_seed54/zero_pd_run.json)。
三种控制器的9组命令张量、初始物理状态及ξ已逐项配对检查；不以历史Standing零动作数据代替本次alternating任务。

| 控制器 | 时限/s | Survival | 实际执行/s | NR | 世界足位置RMSE/m | 前向速度/m/s |
| --- | --- | --- | --- | --- | --- | --- |
| initial_frozen_decoder | 2 | 0/3 | 1.693333 | 0.2208211 | 0.1712955 | 0.5619058 |
| initial_frozen_decoder | 5 | 0/3 | 1.693333 | 0.08839859 | 0.1717615 | 0.5623501 |
| initial_frozen_decoder | 10 | 0/3 | 1.686667 | 0.04460779 | 0.1684036 | 0.5591936 |
| final_joint | 2 | 3/3 | 2 | 0.4275562 | 0.734797 | -0.9122115 |
| final_joint | 5 | 0/3 | 2.946667 | 0.2110395 | 1.580195 | -1.256479 |
| final_joint | 10 | 0/3 | 2.933333 | 0.1050254 | 1.572234 | -1.259542 |
| zero_action_pd | 2 | 1/3 | 1.913333 | 0.660145 | 0.04310358 | -0.3817004 |
| zero_action_pd | 5 | 0/3 | 1.92 | 0.264296 | 0.04356042 | -0.3859166 |
| zero_action_pd | 10 | 0/3 | 1.92 | 0.132148 | 0.04356043 | -0.3859169 |
零动作PD在正式训练和评估进程退出、GPU锁释放后串行执行，进程耗时70.66981s，新增863条Newton评估样本，未计入训练更新/Replay预算；结束时间2026-11-06T06:49:56.174049+00:00，原12h截止2026-11-06T14:59:14.159719+08:00。
最终Foot较初始随机Foot Encoder和零动作PD延长了失败前缀，但5/10秒均无成功，后退更快、足位置误差更大；NR也低于零动作PD。不能据此宣布联合训练已解决行走。上述RMSE均为世界坐标足link与固定目标的欧氏误差，同一计算定义，且各自失败前缀长度不同，不是固定10秒成功轨迹上的精度比较。
接口Normalizer分布审计（差异是事实，尚未证明是控制失败原因）：[results/s54/s54_joint_ab1_20261106_025912_seed54/normalization_audit_001000.json](../../results/s54/s54_joint_ab1_20261106_025912_seed54/normalization_audit_001000.json)。
与历史C在2000 iterations的真实预算对照（同总样本不等于同Foot样本、Actor更新或Motion监督）：[results/s54/s54_joint_ab1_20261106_025912_seed54/budget_comparison_002000_vs_C.json](../../results/s54/s54_joint_ab1_20261106_025912_seed54/budget_comparison_002000_vs_C.json)。
42项历史ABC/Teacher文件内容、大小和mtime未改动：[results/s54/s54_joint_ab1_20261106_025912_seed54/historical_final_preservation.json](../../results/s54/s54_joint_ab1_20261106_025912_seed54/historical_final_preservation.json)。

**按用户追加要求建立的精简数据组**：[Foothold_with_SuperMimic_Core](https://swanlab.cn/@yagami_light/Foothold_with_SuperMimic_Core/runs/6d21bea029fa4c4f/chart)；状态 `verified`，稳定ID `6d21bea029fa4c4f`。
仅重放同一次正式训练的32项核心指标；746个记录点，横轴为真实iteration，最大14484。每20 iterations记录一次，不应把点数误读为训练迭代数。同一iteration的训练/评估/终态合并；完整raw JSONL仍为749个event。
逐项最新值云端读回证据：[logs/s54_joint_ab1_20261106_025912_seed54/swanlab_core_export/cloud_readback.json](../../logs/s54_joint_ab1_20261106_025912_seed54/swanlab_core_export/cloud_readback.json)；身份与源实验关联：[logs/s54_joint_ab1_20261106_025912_seed54/swanlab_core_export/identity.json](../../logs/s54_joint_ab1_20261106_025912_seed54/swanlab_core_export/identity.json)。原SwanLab实验保留，不额外启动任何训练。
核心上传器CPU测试4项通过；仅上传逻辑与报告后处理在运行末期按用户要求调整，训练器/Reward/终止/物理/冻结profile均未修改。变更回执：[logs/s54_joint_ab1_20261106_025912_seed54/swanlab_core_policy_change.json](../../logs/s54_joint_ab1_20261106_025912_seed54/swanlab_core_policy_change.json)。

最小下一步建议：先离线检查Foot末期轨迹的后退、触地和reward/bootstrap梯度，再分别验证Decoder输入的观测/latent分布及长H动作梯度；对照需固定真实Foot样本和Motion监督预算。现有证据不足以在本轮后自动开展D/F/G或无界延长训练；本轮未执行这些新训练。
