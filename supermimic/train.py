"""SuperMimic training (Algorithm A.1): collect -> fit world model -> BPTT policy + critic update."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time
import signal
import subprocess
import random
import numpy as np

PROCESS_START = time.monotonic()
STOP_REQUESTED = False
def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

M48_GROUPS = {
    'a': ('a_startup_only', 10, 0.),
    'b': ('b_smoothness_only', 0, .02),
    'c': ('c_startup_smoothness', 10, .02),
}
M49_GROUPS = {'a': 'a_long_baseline', 'b': 'b_original_prior', 'c': 'c_soft_prior'}
M49_GRADIENT_PAIRS = (
    ('foot', 'body_prior'), ('foot', 'soft_body_prior'), ('foot', 'upper_action_smoothness'),
    ('upper_action_smoothness', 'body_prior'), ('upper_action_smoothness', 'soft_body_prior'),
    ('bootstrap', 'body_prior'), ('bootstrap', 'soft_body_prior'),
)

from supermimic.roban_runtime import M4_GUARDED_MAPPINGS, configure_robot, add_roban_args
ROBOT_KIND = configure_robot(sys.argv[1:])
from isaaclab.app import add_launcher_args, launch_simulation
from supermimic.goals import GoalSource, add_goal_args, resolve_goal_config
from supermimic.foot_mdp import FootTrackingMDP, resolve_foot_task, assert_task_checkpoint
from supermimic.integration.m410_reward import (GROUPS as M410_GROUPS, GROUP_REWARDS, HUBER_TERMS, ROOT_TERMS,
                                                SHOULDER_JOINTS, WHOLE_BODY_TERMS, tolerance_metric_names)
from supermimic.integration.m411_hand import GROUPS as M411_GROUPS, HAND_CONFIG, HAND_SIDES, SOURCE as M411_SOURCE

M410_GRADIENT_PAIRS = tuple((base, term) for base in ('foot', 'bootstrap') for term in WHOLE_BODY_TERMS)
M410_WM_METRICS = ('wm_root_position_rmse', 'wm_root_orientation_rmse_deg', 'wm_upper_joint_rmse')
M411_GRADIENT_PAIRS = tuple((name, 'hand_huber') for name in
                             ('foot', 'root', 'joint_huber', 'upper_action_smoothness', 'bootstrap'))

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--iterations", type=int, default=30)
parser.add_argument("--buffer_steps", type=int, default=64)
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=8)
parser.add_argument("--goal_frames", type=int, default=10)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=1)
parser.add_argument("--wm_batch", type=int, default=64)
parser.add_argument("--pi_batch", type=int, default=32)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument('--gradient_audit_every',type=int,default=0,help='Periodically split reward/bootstrap Actor gradients; 0 disables')
parser.add_argument('--actor_lr', type=float, default=None, help='Actor LR; default inherits --lr')
parser.add_argument('--critic_real_batch',type=int,default=0,help='Optional real physical-termination return anchors; 0 disables')
parser.add_argument('--critic_real_window',type=int,default=512)
parser.add_argument('--critic_real_recent_steps',type=int,default=512)
parser.add_argument('--critic_real_weight',type=float,default=0.25)
parser.add_argument('--critic_lr', type=float, default=None, help='Critic LR; default inherits --lr')
parser.add_argument('--recovery_wm_iters', type=int, default=20, help='WM-only updates after replay refill on resume')
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--no_critic", action="store_true")
parser.add_argument("--warmup_steps", type=int, default=32, help="sim steps before any update")
parser.add_argument("--wm_only_iters", type=int, default=10, help="world-model-only iterations at the start")
parser.add_argument("--lr_decay_iters", type=int, default=250_000)
parser.add_argument("--ckpt_every", type=int, default=30)
parser.add_argument("--log_every", type=int, default=10)
parser.add_argument("--motion_dir", type=str, default=os.path.join(REPO, "library", "motion_reference", "kuavo_s54", "retarget_data", "reference_startup_fixed"))
parser.add_argument('--m34_dataset', help='M3.4 dataset_export root; strictly train split, native 50 Hz NPZ')
parser.add_argument('--m34_straight_iters', type=int, default=1000, help='Straight emphasis before equal three-class sampling')
parser.add_argument('--m34_eval_every', type=int, default=1000, help='Independent frozen train/test Newton eval interval; 0 disables')
parser.add_argument('--m34_eval_seeds', nargs='+', type=int, default=[0, 1, 2], help='0 nominal; nonzero seeds use reset perturbations')
parser.add_argument("--clips", nargs="+", help="Clip names or quoted glob patterns; omit to train all top-level CSVs")
parser.add_argument("--log_dir", type=str, default=os.path.join(REPO, "logs", "s54_smoke"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
parser.add_argument("--no_compile", action="store_true", help="eager PyTorch for a short smoke; same WM/BPTT algorithm")
parser.add_argument('--foot_task_config', default=None, help='Independent foot MDP JSON, including standing/alternating mode')
parser.add_argument('--check_foot_gradients', action='store_true', help='Measure pure foot reward Actor gradients and real/replay time consistency')
parser.add_argument('--curriculum_config', help='Enable the bounded curriculum (JSON)')
parser.add_argument('--max_device_memory_mib',type=float,default=None,help='Optional device-memory stop threshold at telemetry intervals')
parser.add_argument('--fresh_start', action='store_true', help='Assert fresh models/progress; forbid resume and transfer')
parser.add_argument('--init_weights', help='Explicit transfer of compatible WM/Actor weights; fresh optimizers/critic')
parser.add_argument('--m451_source', help='M4.5.1 短训的 M4.3 第 4500 轮初始化来源')
parser.add_argument('--m46_group', choices=('w002', 'w005'), help='M4.6 单因素正式实验组')
parser.add_argument('--m46_check', action='store_true', help='仅核对 M4.6 运行时配置后退出')
parser.add_argument('--m46_smoke', action='store_true', help='隔离目录内执行固定两轮 Newton 冒烟测试')
parser.add_argument('--m47_startup_steps', type=int, default=0, help='上身启动过渡控制步数')
parser.add_argument('--upper_action_smoothness_weight', type=float, default=0., help='上身原始动作二阶差分权重')
parser.add_argument('--m47_smoke', action='store_true', help='隔离目录内验证 M4.7 的采集、回放和联合更新')
parser.add_argument('--m48_group', choices=M48_GROUPS, help='M4.8 正式消融组')
parser.add_argument('--m48_check', action='store_true', help='仅解析并核对 M4.8 生效配置')
parser.add_argument('--m48_smoke', action='store_true', help='在临时目录验证 M4.8 采集和联合更新')
parser.add_argument('--m49_group', choices=M49_GROUPS, help='M4.9 正式消融组')
parser.add_argument('--m49_check', action='store_true', help='仅核对 M4.9 生效配置')
parser.add_argument('--m49_smoke', action='store_true', help='在临时目录验证 M4.9 姿态先验及联合更新')
parser.add_argument('--m410_group', choices=M410_GROUPS, help='M4.10 正式消融组')
parser.add_argument('--m410_check', action='store_true', help='仅核对 M4.10 生效配置')
parser.add_argument('--m410_smoke', action='store_true', help='隔离验证 M4.10 奖励和联合更新')
parser.add_argument('--m411_group', choices=M411_GROUPS, help='M4.11 正式微调组')
parser.add_argument('--m411_check', action='store_true', help='核对 M4.11 生效配置')
parser.add_argument('--m411_smoke', action='store_true', help='隔离验证 M4.11 的两个训练阶段')
parser.add_argument('--m411_critic_warmup', type=int, default=0, help='M4.11 Critic 单独更新轮次')
parser.add_argument('--m50_library', help='M5.0 离线多参考清单')
parser.add_argument('--m50_reference_root', help='训练参考根目录，离线生成数据只读加载')
parser.add_argument('--m50_adaptation_iters', type=int, default=0, help='冻结 Actor 的 WM/Critic 适应轮数')
parser.add_argument('--m50_smoke', type=int, choices=(8, 32, 256), help='在隔离目录运行指定环境数的 M5.0 冒烟')
parser.add_argument('--m50_check', action='store_true', help='仅核对 M5.0 生效配置')
parser.add_argument('--m51_group', choices=('a', 'b', 'c'), help='M5.1 足部支撑消融组')
parser.add_argument('--m51_critic_warmup', type=int, default=0, help='M5.1 冻结 Actor 与 WM 的 Critic 预热轮次')
parser.add_argument('--m52_group', choices=('a', 'b', 'c'), help='M5.2 根部倾斜消融组')
parser.add_argument('--stair_config', help='Opt-in T1.5 audited mixed flat/stair experiment design')
parser.add_argument('--init_from_flat', help='Explicit Flat -> Stair transfer; source checkpoint is read-only')
parser.add_argument('--init_from_motion', help='Cross-task Stage 1 initialization; never a resume')
parser.add_argument('--init_from_bridge', help='Initialize Quintic Foot RL from an optional paired Motion-reward bridge')
parser.add_argument('--transfer_components', choices=['wm','wm_decoder'],default='wm_decoder')
parser.add_argument('--obs_adapter', choices=['identity','mask_reference_orientation'],default='identity')
parser.add_argument('--motion_obs_mode', choices=['legacy_v1', 'root_xy_heading_v2', 'roban_proprio_v1'], default='root_xy_heading_v2')
parser.add_argument('--motion_reward_config', help='Motion reward JSON object or JSON file; defaults to requested tracking-v2 weights')
parser.add_argument('--freeze_decoder',action='store_true')
parser.add_argument('--train_decoder',action='store_true',help='Explicitly finetune D during direct Foot RL; bridge always freezes D')
parser.add_argument('--reference_manifest',help='Hashed clip-level train/held-out manifest')
parser.add_argument('--manifest_split',choices=['train','heldout'],default='train')
parser.add_argument('--prior_config',help='Optional upper-body prior config; default off')
parser.add_argument('--wall_seconds',type=float,help='Safe wall-clock stop, including initialization; finite bounded experiments')
parser.add_argument('--reward_term_logging',action='store_true',help='Persist per-term mean/std and weighted prior contributions locally')
parser.add_argument('--paired_latent_weight',type=float,default=0.,help='Optional same-clip latent alignment; never applies to random Quintic goals')
parser.add_argument('--m4_early_iteration', type=int, help='M4 早期检查点迭代，M4.1 和 M4.2 默认取 WM-only 结束轮')
add_goal_args(parser)
add_launcher_args(parser)
add_roban_args(parser, sys.argv[1:])
args = parser.parse_args()
m47_context_requested = bool(args.m47_startup_steps or args.upper_action_smoothness_weight)
m48 = args.m48_group is not None
m49 = args.m49_group is not None
m410 = args.m410_group is not None
m411 = args.m411_group is not None
m50 = args.m50_library is not None
m51 = args.m51_group is not None
m52 = args.m52_group is not None
fresh_run = False
hand_tracking = m411 or m50
whole_body = m410 or hand_tracking
upper_ablation = m48 or m49 or m410 or m411
upper_smoke = args.m48_smoke or args.m49_smoke or args.m410_smoke or args.m411_smoke
upper_check = args.m48_check or args.m49_check or args.m410_check or args.m411_check
if sum((m48, m49, m410, m411)) > 1 or any(
        (check or smoke) and not enabled for enabled, check, smoke in (
            (m48, args.m48_check, args.m48_smoke), (m49, args.m49_check, args.m49_smoke),
            (m410, args.m410_check, args.m410_smoke), (m411, args.m411_check, args.m411_smoke))):
    parser.error('请选择对应的一组 M4.8 至 M4.11 实验入口')
if args.m411_critic_warmup and not m411:
    parser.error('Critic 单独预热需要 M4.11 实验入口')
if not m50 and any((args.m50_check, args.m50_smoke, args.m50_adaptation_iters, args.m50_reference_root, m51, m52)):
    parser.error('M5.0 参数需要 --m50_library')
if args.m51_critic_warmup and not m51:
    parser.error('M5.1 Critic 预热需要对应消融组配置')
if m52 and args.m51_group != 'c':
    parser.error('M5.2 继承 M5.1-C 的足部奖励配置')
if m50:
    from supermimic.integration.m50_reference import MANIFEST as M50_MANIFEST, read_manifest as read_m50_manifest
    from supermimic.integration.m50_task import CONFIG as M50_CONFIG, OUTPUT as M50_OUTPUT, TRAIN_CLIPS

    try:
        config = json.loads(M50_CONFIG.read_text())
        parent = json.loads((Path(REPO) / 'configs/roban/m4/m4_11_b.json').read_text())
        del parent['m411_critic_warmup']
        expected = dict(parent, m50_adaptation_iters=1000, iterations=6000, m4_eval_every=500)
        config_path = M50_CONFIG
        if m51:
            from supermimic.integration.m51_task import (CONFIGS, GRAD_PAIRS as M51_GRAD_PAIRS, GROUPS,
                LOG_TERMS as M51_LOG_TERMS, OUTPUT as M51_OUTPUT, TERMS as M51_TERMS, expected_config)

            if config != expected or args.fresh_start:
                raise ValueError('M5.1 从冻结的 M5.0 配置与第 5500 轮权重初始化')
            config_path = CONFIGS[args.m51_group]
            expected = expected_config(config, args.m51_group)
            config = json.loads(config_path.read_text())
            experiment_dir = M51_OUTPUT / GROUPS[args.m51_group]
        if m52:
            from supermimic.integration.m52_task import (CONFIGS as M52_CONFIGS, GRAD_PAIRS as M52_GRAD_PAIRS,
                GROUPS as M52_GROUPS, OUTPUT as M52_OUTPUT, TRANSFER as M52_TRANSFER, expected_config as expected_m52)

            if config != expected:
                raise ValueError('M5.2 需要冻结的 M5.1-C 配置')
            expected = expected_m52(config, args.m52_group)
            config_path = M52_CONFIGS[args.m52_group]
            config = json.loads(config_path.read_text())
            experiment_dir = M52_OUTPUT / M52_GROUPS[args.m52_group]
        if (not args.roban_config or Path(args.roban_config).resolve() != config_path.resolve() or config != expected
                or Path(args.m50_library).resolve() != M50_MANIFEST or args.m4_reference or upper_ablation
                or args.m46_group or args.m451_source or args.device != 'cuda:0'):
            raise ValueError('M5.0 需要独立清单、冻结的 B 组配置与 CUDA:0')
        if not args.m50_reference_root or not Path(args.m50_reference_root).is_dir():
            raise ValueError('M5.0 需要生成器的只读参考根目录')
        manifest = read_m50_manifest(args.m50_library)
        if tuple(entry['clip_id'] for entry in manifest['train']) != TRAIN_CLIPS:
            raise ValueError('M5.0 正式训练清单需要原始 F1、F2、F3')
        output = Path(args.log_dir).resolve()
        if args.m50_smoke:
            prefix = 'm52-smoke-' if m52 else 'm51-smoke-' if m51 else 'm50-smoke-'
            if (output.name != 'checkpoints' or output.parent.parent != Path('/tmp')
                    or not output.parent.name.startswith(prefix) or not output.parent.is_dir()):
                raise ValueError(f'冒烟输出需要已有 /tmp/{prefix}*/checkpoints')
        else:
            fresh_run = args.fresh_start or (args.resume and output.parent == M50_OUTPUT / 'fresh')
            expected_out = experiment_dir if m51 else M50_OUTPUT / 'fresh' if fresh_run else M50_OUTPUT
            if output != expected_out / 'checkpoints':
                raise ValueError(f'M5.0 本次检查点目录为 {expected_out / "checkpoints"}')
        if args.resume and Path(args.resume).resolve(strict=True) != output / 'last.pt':
            raise ValueError('M5.0 恢复需要同一运行目录的 last.pt')
        canonical = ['--roban_config', args.roban_config, '--m50_library', args.m50_library,
                     '--m50_reference_root', args.m50_reference_root, '--device', 'cuda:0', '--log_dir', args.log_dir]
        if args.resume:
            canonical += ['--resume', args.resume]
        if args.fresh_start:
            canonical += ['--fresh_start']
        if args.m50_smoke:
            canonical += ['--m50_smoke', str(args.m50_smoke)]
        if args.m50_check:
            canonical += ['--m50_check']
        actual = vars(parser.parse_args(canonical))
        if vars(args) != actual:
            raise ValueError(f'M5.0 参数差异 {sorted(key for key in vars(args) if vars(args)[key] != actual[key])}')
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.m50_smoke:
        setattr(args, 'm51_critic_warmup' if m51 else 'm50_adaptation_iters', 2)
        vars(args).update(num_envs=args.m50_smoke, iterations=2, warmup_steps=34,
                          ckpt_every=1, m4_eval_every=0, log_every=1, gradient_audit_every=1)

if not m49 and (args.soft_body_prior_weight or args.soft_body_prior_tolerance is not None):
    parser.error('Soft Body Prior 需要 M4.9 F2 实验入口')
if args.m47_smoke:
    group = {'a':(0.,0.), 'b':(.02,0.), 'c':(.02,.02)}
    out = Path(args.log_dir).resolve()
    scratch = out.parent.parent
    if (not args.m4_reference or not args.roban_config or not args.fresh_start or args.resume
            or args.m451_source or args.m46_group or args.m46_smoke or args.m46_check or upper_ablation
            or Path(args.roban_config).resolve() != Path(REPO)/'configs/roban/m4/m4_3.json'
            or out.name != 'checkpoints' or scratch.parent != Path('/tmp')
            or not scratch.name.startswith('m47-smoke-') or not scratch.is_dir()
            or (args.upper_action_smoothness_weight,args.body_prior_weight) != group.get(out.parent.name)
            or args.m47_startup_steps not in (5,10) or args.m4_unclipped_pd_target):
        parser.error('M4.7 冒烟测试须使用 M4.3/F2、Fresh Start 和 /tmp/m47-smoke-*/{a,b,c}/checkpoints')
    vars(args).update(num_envs=8, iterations=2, warmup_steps=16, wm_only_iters=1, recovery_wm_iters=0,
                      m4_early_iteration=1, ckpt_every=1, m4_eval_every=0, log_every=1, gradient_audit_every=1)
elif m47_context_requested and not (upper_ablation or m50):
    parser.error('启动和平滑配置需要独立的实验入口')


def milestone_name(iteration):
    return f'iter_{iteration:04d}.pt' if hand_tracking else f'iter_{iteration}.pt'

if upper_ablation:
    revision = 11 if m411 else 10 if m410 else 9 if m49 else 8
    phase = f'M4.{revision}'
    group_key = getattr(args, f'm4{revision}_group')
    group_dir = (M411_GROUPS if m411 else M410_GROUPS if m410 else M49_GROUPS if m49 else
                 {key: entry[0] for key, entry in M48_GROUPS.items()})[group_key]
    config_dir = Path(REPO) / 'configs/roban/m4'
    selected = config_dir / f'm4_{revision}_{group_key}.json'
    formal = Path(REPO) / f'outputs/m4_{revision}' / group_dir / 'checkpoints'
    output = Path(args.log_dir).resolve()
    try:
        if not args.roban_config or Path(args.roban_config).resolve() != selected:
            raise ValueError(f'{phase} 本组需要 {selected.name}')
        candidate = json.loads(selected.read_text())
        if m411:
            baseline = json.loads((config_dir / 'm4_10_c.json').read_text())
            expected_config = dict(baseline, iterations=2000, wm_only_iters=0, recovery_wm_iters=0,
                                   m411_critic_warmup=200, actor_lr=5e-5, ckpt_every=250, m4_eval_every=250)
            if group_key == 'b':
                expected_config['roban_reward_config'] = baseline['roban_reward_config'] | HAND_CONFIG
        elif m410:
            baseline = json.loads((config_dir / 'm4_9_a.json').read_text())
            expected_config = dict(baseline, roban_reward_config=baseline['roban_reward_config'] | GROUP_REWARDS[group_key])
        elif m49:
            baseline = json.loads((config_dir / 'm4_8_c.json').read_text())
            expected_config = dict(baseline, iterations=10000)
            if group_key == 'b':
                expected_config['body_prior_weight'] = .02
            elif group_key == 'c':
                if not args.soft_body_prior_weight > 0. or args.soft_body_prior_tolerance is None:
                    raise ValueError('M4.9-C 需要离线标定的正权重与容许偏差')
                expected_config.update(soft_body_prior_weight=args.soft_body_prior_weight,
                                       soft_body_prior_tolerance=args.soft_body_prior_tolerance)
        else:
            _, steps, weight = M48_GROUPS[group_key]
            baseline = json.loads((config_dir / 'm4_3.json').read_text())
            expected_config = dict(baseline, m47_startup_steps=steps, upper_action_smoothness_weight=weight)
        if baseline.get('body_prior_weight', 0.) != 0. or candidate != expected_config:
            raise ValueError(f'{phase} 的配置包含未授权的基线或消融参数差异')
        invalid_start = args.fresh_start if m411 else args.fresh_start == bool(args.resume)
        if not args.m4_reference or args.device != 'cuda:0' or invalid_start:
            raise ValueError(f'{phase} 需要原 F2、CUDA:0 和本组初始化方式或同组恢复')
        if upper_smoke:
            scratch = output.parent.parent
            if ((not m411 and not args.fresh_start) or upper_check or output != scratch / group_dir / 'checkpoints'
                    or scratch.parent != Path('/tmp') or not scratch.name.startswith(f'm4{revision}-smoke-') or not scratch.is_dir()):
                raise ValueError(f'{phase} Smoke 需要独立的 /tmp/m4{revision}-smoke-* 组目录')
        elif output != formal or formal != formal.resolve():
            raise ValueError(f'{phase} 本组正式检查点目录为 {formal}')
        if args.resume:
            checkpoint = Path(args.resume).absolute()
            stride = 1 if upper_smoke else expected_config['ckpt_every']
            names = {'initial.pt', 'last.pt', *(milestone_name(i) for i in range(stride, expected_config['iterations']+1, stride))}
            if m411:
                names.add('critic_warmup.pt')
            if (checkpoint.parent != output or checkpoint.name not in names
                    or checkpoint.resolve(strict=True) != checkpoint):
                raise ValueError(f'{phase} 只能恢复当前组的 Initial、Last 或周期里程碑')
        # 用正式入口重新解析默认值，禁止悄悄覆盖实验组的有效训练参数。
        canonical = ['--roban_config', args.roban_config, '--m4_reference', args.m4_reference,
                     '--device', 'cuda:0', f'--m4{revision}_group', group_key, '--log_dir', args.log_dir]
        if args.resume:
            canonical += ['--resume', args.resume]
        elif args.fresh_start:
            canonical.append('--fresh_start')
        if upper_smoke:
            canonical.append(f'--m4{revision}_smoke')
        if upper_check:
            canonical.append(f'--m4{revision}_check')
        expected = vars(parser.parse_args(canonical))
        changed = [key for key, value in vars(args).items() if value != expected[key]]
        if changed:
            raise ValueError(f'{phase} 不允许覆盖生效参数：{", ".join(changed)}')
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if upper_smoke:
        vars(args).update(num_envs=8, iterations=2, warmup_steps=34, wm_only_iters=1, recovery_wm_iters=0,
                          m4_early_iteration=1, m4_eval_every=0, ckpt_every=1, log_every=1, gradient_audit_every=1)
        if m411:
            vars(args).update(m411_critic_warmup=2, wm_only_iters=0)
elif args.roban_config and Path(args.roban_config).name in {f'm4_{v}_{name}.json' for v in (8, 9, 10, 11) for name in 'abc'}:
    parser.error('M4.8 至 M4.11 配置需要对应的实验组入口')
m46 = args.m46_group is not None
if m46:
    group = args.m46_group
    weight = {'w002': .02, 'w005': .05}[group]
    config_dir = Path(REPO) / 'configs/roban/m4'
    selected_config = config_dir / f'm4_6_{group}.json'
    output = Path(args.log_dir).resolve()
    formal_output = Path(REPO) / 'outputs' / 'm4_6' / group / 'checkpoints'
    smoke_root = output.parent.parent
    try:
        if not args.roban_config or Path(args.roban_config).resolve() != selected_config:
            raise ValueError(f'M4.6 {group} 必须使用 {selected_config.name}')
        baseline = json.loads((config_dir / 'm4_3.json').read_text())
        if baseline.get('body_prior_weight', 0.) != 0.:
            raise ValueError('M4.6 的 M4.3 基线必须是零上身先验权重')
        if json.loads(selected_config.read_text()) != dict(baseline, body_prior_weight=weight):
            raise ValueError('M4.6 配置相对 M4.3 只允许改变上身先验权重')
        if not args.m4_reference or args.device != 'cuda:0' or args.m4_unclipped_pd_target:
            raise ValueError('M4.6 需要原始 F2、CUDA:0 和 Hard Clamp')
        if args.fresh_start == bool(args.resume):
            raise ValueError('M4.6 必须明确选择 --fresh_start 或同组 --resume')
        if args.m46_smoke:
            if (not args.fresh_start or args.m46_check or output.parent.name != group
                    or output.name != 'checkpoints' or smoke_root.parent != Path('/tmp')
                    or not smoke_root.name.startswith('m46-smoke-') or not smoke_root.is_dir()):
                raise ValueError('M4.6 冒烟测试只能从头运行在现有 /tmp/m46-smoke-*/<group>/checkpoints')
        elif output != formal_output or formal_output != formal_output.resolve():
            raise ValueError(f'M4.6 {group} 的正式检查点目录必须为 {formal_output}')
        if args.resume and (Path(args.resume).absolute() != formal_output / 'last.pt'
                            or Path(args.resume).resolve(strict=True) != formal_output / 'last.pt'):
            raise ValueError('M4.6 只允许从同组 checkpoints/last.pt 恢复')
        # 使用同一解析器重建受控入口，拒绝任何改变有效训练参数的命令行覆盖。
        canonical = ['--roban_config', args.roban_config, '--m4_reference', args.m4_reference,
                     '--device', 'cuda:0', '--m46_group', group, '--log_dir', args.log_dir]
        canonical += ['--fresh_start'] if args.fresh_start else ['--resume', args.resume]
        if args.m46_check:
            canonical.append('--m46_check')
        if args.m46_smoke:
            canonical.append('--m46_smoke')
        expected = vars(parser.parse_args(canonical))
        changed = [name for name, value in vars(args).items() if value != expected[name]]
        if changed:
            raise ValueError(f'M4.6 不允许覆盖基线有效参数：{", ".join(changed)}')
        args.m4_reference = str(Path(args.m4_reference).resolve(strict=True))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.m46_check:
        print('M4.6 effective config ' + json.dumps(vars(args), ensure_ascii=False), flush=True)
        sys.exit(0)
    if args.m46_smoke:
        vars(args).update(num_envs=8, iterations=2, warmup_steps=16, wm_only_iters=1, recovery_wm_iters=0,
                          m4_early_iteration=1, ckpt_every=1, m4_eval_every=0, log_every=1, gradient_audit_every=1)
elif args.m46_check or args.m46_smoke:
    parser.error('M4.6 检查需要 --m46_group')
elif args.roban_config and Path(args.roban_config).name in ('m4_6_w002.json', 'm4_6_w005.json'):
    parser.error('M4.6 配置需要 --m46_group，请使用 tools/run_m46_ablation.sh')
m4 = bool(args.m4_reference or m50)
residual_action = m4 and args.m4_action_mapping == 'reference_residual'
joint_range_action = m4 and args.m4_action_mapping == 'joint_range'
standing_action = m4 and args.m4_action_mapping == 'standing_offset'
guarded_action = m4 and args.m4_action_mapping in M4_GUARDED_MAPPINGS
m43 = standing_action and bool(args.roban_reward_config.get('swing_aware'))
m44 = m43 and args.m4_unclipped_pd_target
m45 = m43 and not m44 and args.body_prior_weight > 0
m451 = args.m451_source is not None
if args.m4_early_iteration is None:
    args.m4_early_iteration = max(1, args.wm_only_iters) if guarded_action else 8
try:
    if ROBOT_KIND == 'roban':
        if args.goal_mode != 'roban_foot' or any((args.m34_dataset,args.stair_config,args.foot_task_config,args.curriculum_config,args.prior_config,args.init_from_motion,args.init_from_bridge,args.init_from_flat)):
            raise ValueError('Roban uses its unified reference/task adapter; S54-only options are incompatible')
        if args.obs_adapter != 'identity' or args.freeze_decoder or args.paired_latent_weight or args.check_foot_gradients or (args.gradient_audit_every and not m43) or args.critic_real_batch:
            raise ValueError('Use Roban foot gradient tests; legacy S54 adapters/audits are incompatible')
        if m52:
            from supermimic.integration.m52_task import RootTiltRewardConfig as reward_config_type
        elif m51:
            from supermimic.integration.m51_task import StanceFootRewardConfig as reward_config_type
        else:
            from supermimic.roban_task import FootRewardConfig as reward_config_type
        reward_cfg = reward_config_type(**dict(args.roban_reward_config, body_prior_weight=args.body_prior_weight,
                                            soft_body_prior_weight=args.soft_body_prior_weight,
                                            soft_body_prior_tolerance=args.soft_body_prior_tolerance))
        if reward_cfg.active_whole_body() and not whole_body:
            raise ValueError('全身跟踪奖励需要 M4.10 或 M4.11 F2 实验入口')
        if reward_cfg.hand_huber and not hand_tracking:
            raise ValueError('手部跟踪奖励需要 M4.11 或 M5.0 实验入口')
    if m4 and (ROBOT_KIND != 'roban' or args.init_weights or args.no_critic
               or args.reward_term_logging or args.wm_use_terrain or args.m4_early_iteration < 1):
        raise ValueError('M4 使用 Roban 平地任务和完整 SuperMimic 更新，仅支持同任务恢复')
    if not m4 and args.m4_action_mapping != 'neutral':
        raise ValueError('M4 动作映射需要 --m4_reference')
    if args.m4_unclipped_pd_target and (not m44 or not args.m4_fault_trace):
        raise ValueError('未裁剪训练使用 M4.3 摆动奖励、standing_offset 和物理故障监测')
    if m451 and (not m43 or m44 or not args.m4_fault_trace or args.fresh_start
                 or args.body_prior_weight not in (0., .02, .05, .10) or not 200 <= args.iterations <= 300):
        raise ValueError('M4.5.1 需要带故障监测的 F2 Hard Clamp、指定上身权重及 200–300 轮短训')
    if guarded_action and (not math.isfinite(args.action_noise) or args.action_noise < 0):
        raise ValueError('M4 探索噪声的标准差必须有限且非负')
    if args.m4_fault_trace and not guarded_action:
        raise ValueError('物理异常快照需要 M4.1 或 M4.2 动作映射')
    if args.m4_eval_every < 0 or (args.m4_eval_every and not standing_action):
        raise ValueError('m4_eval_every 必须为非负数，定期 M4 评估需要 M4.2 动作映射')
    if args.m34_eval_every < 0 or not args.m34_eval_seeds or len(set(args.m34_eval_seeds)) != len(args.m34_eval_seeds) or any(s < 0 for s in args.m34_eval_seeds):
        raise ValueError('Invalid M3.4 evaluation interval or seed protocol')
    goal_spec, foot_config = resolve_goal_config(args)
    if args.goal_mode != 'foot_trajectory' and (args.foot_task_config or args.check_foot_gradients):
        raise ValueError('Foot task configuration/gradient checks require --goal_mode foot_trajectory')
    task_cfg = resolve_foot_task(args.foot_task_config) if args.goal_mode == 'foot_trajectory' else None
    if args.goal_mode=='reference_foot' and (args.foot_task_config or args.curriculum_config):
        raise ValueError('Paired bridge uses Motion reward/reset and its own reference-foot goal')
    if args.init_from_motion and args.goal_mode=='motion': raise ValueError('init_from_motion is a cross-task Foot initializer')
    if args.init_from_bridge and args.goal_mode!='foot_trajectory': raise ValueError('Bridge initialization targets Quintic Foot RL')
    if sum(bool(x) for x in (args.resume,args.init_weights,args.init_from_motion,args.init_from_bridge,args.init_from_flat))>1:
        raise ValueError('Choose exactly one resume/initialization source')
    if args.freeze_decoder and args.train_decoder: raise ValueError('Conflicting Decoder update flags')
    if args.goal_mode=='reference_foot' and args.train_decoder: raise ValueError('Sequential bridge freezes Decoder; joint Motion training is not enabled')
    if args.init_from_motion and args.transfer_components=='wm_decoder': args.obs_adapter='mask_reference_orientation'
    if args.init_from_motion and args.transfer_components=='wm_decoder' and not args.train_decoder: args.freeze_decoder=True
    if args.init_from_bridge:
        args.obs_adapter='mask_reference_orientation'
        if not args.train_decoder: args.freeze_decoder=True
    if args.goal_mode=='reference_foot':
        if not (args.init_from_motion or args.resume): raise ValueError('Paired bridge requires Stage 1 initialization or same-task resume')
        if args.goal_frames!=10: raise ValueError('The paired Motion encoder contract currently requires K=10')
        args.freeze_decoder=True;args.obs_adapter='mask_reference_orientation'
    if not math.isfinite(args.paired_latent_weight) or args.paired_latent_weight<0 or (args.paired_latent_weight and args.goal_mode!='reference_foot'):
        raise ValueError('Latent alignment requires paired reference data and a nonnegative weight')
    if args.wall_seconds is not None and (not math.isfinite(args.wall_seconds) or args.wall_seconds<=0): raise ValueError('Invalid wall_seconds')
    if args.curriculum_config and (task_cfg is None or task_cfg.command_mode != 'curriculum'):
        raise ValueError('Curriculum requires command_mode=curriculum foot task')
    if task_cfg is not None and task_cfg.command_mode == 'curriculum' and not args.curriculum_config:
        raise ValueError('Curriculum task requires --curriculum_config')
    if args.lr<=0 or any(v is not None and v<=0 for v in (args.actor_lr,args.critic_lr)):
        raise ValueError('Learning rates must be positive')
    if args.critic_real_batch<0 or args.critic_real_window<1 or args.critic_real_recent_steps<1 or args.critic_real_weight<0:
        raise ValueError('Invalid real Critic anchor configuration')
    if args.critic_real_batch and task_cfg is None: raise ValueError('Real value anchors currently require the foot task')
    if args.gradient_audit_every<0: raise ValueError('gradient_audit_every must be nonnegative')
    if args.recovery_wm_iters<0: raise ValueError('recovery_wm_iters must be nonnegative')
    if not 0<=args.gamma<1: raise ValueError('gamma must be in [0,1)')
    if args.max_device_memory_mib is not None and args.max_device_memory_mib<=0:
        raise ValueError('max_device_memory_mib must be positive')
    if args.fresh_start and (args.resume or args.init_weights or args.init_from_motion or args.init_from_bridge):
        raise ValueError('--fresh_start cannot load --resume or --init_weights')
    if args.resume and args.init_weights: raise ValueError('Choose resume OR init_weights')
    if task_cfg is not None and args.horizon >= task_cfg.episode_steps:
        raise ValueError('Foot episode_steps must be greater than the imagination horizon')
except (ValueError, TypeError, OSError) as exc:
    parser.error(str(exc))
log_path = Path(args.log_dir).resolve()
if m4:
    m4_root = Path('outputs') / ('m4_5_1' if m451 else 'm4_5' if m45 else 'm4_4_unclipped' if m44 else 'm4_3' if m43 else
                               {'joint_range': 'm4_1', 'standing_offset': 'm4_2'}.get(args.m4_action_mapping, 'm4_0'))
    if not (m46 or upper_ablation or args.m47_smoke or m50) and (not log_path.is_relative_to(Path(REPO) / m4_root) or log_path.name != "checkpoints"):
        parser.error(f"M4 检查点目录必须位于 {m4_root} 下，且目录名为 checkpoints")
    if m451 and log_path.parent != Path(REPO) / m4_root / f'w{round(args.body_prior_weight*100):03d}':
        parser.error('M4.5.1 权重和短训输出组名不一致')
    metrics_path = log_path.parent/'logs'/'train.csv'
    args_path = log_path.parent/'config'/'train.json'
else:
    if not log_path.is_relative_to(Path(REPO)/"logs") or not log_path.name.startswith("roban_" if ROBOT_KIND == "roban" else "s54_"):
        parser.error("S54 logs must be under this repository logs/s54_*; existing G1 logs are protected")
    metrics_path = log_path/'train_log.csv'
    args_path = log_path/'args.json'
existing_training = (args_path.exists() or metrics_path.exists()
                     or any(log_path.glob('*.pt')))
if not args.resume and not (upper_check or args.m50_check) and existing_training:
    parser.error("Training artifacts already exist: use a fresh s54_* run or --resume last.pt")
if args.buffer_steps < max(args.wm_window, args.hist) or args.warmup_steps < args.wm_window:
    parser.error("buffer_steps >= max(wm_window,hist) and warmup_steps >= wm_window required")
if args.pi_batch * args.horizon < args.critic_minibatches:
    parser.error("Empty critic minibatches: increase pi_batch*horizon")

from supermimic.reference_selection import select_reference_files
from supermimic.motion_prior_data import load_manifest
stair_cfg=None
if args.stair_config:
    from supermimic.stair_training import read_design, contract as stair_contract
    stair_cfg=read_design(args.stair_config)
    if args.goal_mode!='motion' or args.m34_dataset or args.reference_manifest or args.curriculum_config:
        parser.error('T1.5 uses Motion goals and its own explicit train assignment/sampling')
    if not (args.init_from_flat or args.resume):parser.error('T1.5 requires Flat initialization or strict Stair resume')
elif args.init_from_flat:
    parser.error('init_from_flat requires stair_config and prerequisite gates')
motion_files = []
data_contract=None
if task_cfg is None:
    try:
        if m50:
            args.m50_library = str(Path(args.m50_library).resolve(strict=True))
            args.m50_reference_root = str(Path(args.m50_reference_root).resolve(strict=True))
            args.motion_dir = None
        elif m4:
            reference = Path(args.m4_reference).resolve(strict=True)
            args.m4_reference = str(reference)
            args.motion_dir = None
            motion_files = [str(reference)]
        elif ROBOT_KIND == 'roban':
            from supermimic.roban_runtime import select_roban_files
            motion_files = select_roban_files(args.motion_dir, args.clips)
        elif stair_cfg:
            data_contract=stair_contract(stair_cfg)
            from supermimic.m34 import entries as stair_flat_entries
            motion_files=[str(Path(stair_cfg['flat_dataset'])/row['path'])
                          for row in stair_flat_entries(stair_cfg['flat_dataset'],'train')[2]]
            motion_files.append(str(stair_cfg.get('stair_npz',Path(stair_cfg['bundle'])/'test/stair_up_4.npz')))
        elif args.m34_dataset:
            if args.goal_mode != 'motion' or args.reference_manifest or args.clips or args.m34_straight_iters < 0:
                raise ValueError('M3.4 requires motion goals, all train clips, and no legacy manifest')
            from supermimic.m34 import entries, dataset_contract
            _, _, rows = entries(args.m34_dataset, 'train')
            motion_files = [str(Path(args.m34_dataset)/r['path']) for r in rows]
            data_contract = dataset_contract(args.m34_dataset)
        elif args.reference_manifest:
            motion_files,data_contract=load_manifest(args.reference_manifest,args.manifest_split)
        else: motion_files = select_reference_files(args.motion_dir, args.clips)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
else:
    if args.m34_dataset: parser.error('M3.4 uses the original Motion encoder, not a Foot task')
    args.motion_dir = None
    args.foot_task_config = asdict(task_cfg)
# Record the resolved names, including when all clips or a glob was requested.
args.clips = list(TRAIN_CLIPS) if m50 else ['F2'] if m4 else [Path(path).stem for path in motion_files]
if upper_check or args.m50_check:
    if m52:
        phase = f'M5.2-{args.m52_group.upper()}'
    elif m51:
        phase = f'M5.1-{args.m51_group.upper()}'
    elif m50:
        phase = 'M5.0'
    print(f'{phase} effective config ' + json.dumps(vars(args), ensure_ascii=False), flush=True)
    sys.exit(0)
print(f"Training clips ({len(args.clips)}): {', '.join(args.clips)}" if task_cfg is None else
      f'Independent S54 foot task: {task_cfg.command_mode}; no Motion reference loaded', flush=True)

import torch
import torch.nn as nn

from supermimic.control import ACTION_LIMIT, bound_action, bound_bootstrap, return_upper_bound
from supermimic.buffer import ReplayBuffer
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import RobotKinematics
from supermimic.mdp import CRITIC_EXTRA, INV_DIM, MDP, OBS_DIM, RewardCfg, resolve_motion_reward, invariant_features, normalize_xi, unflatten_state
from supermimic.motion import MotionLib
import supermimic.networks as networks
from supermimic.networks import Critic, Policy, WorldModel
from supermimic.transfer import (physical_contract,policy_contract,assert_policy_checkpoint,initialize_from_motion)
from supermimic.robot_contract import (default_joint_pos_list, NUM_J, checkpoint_contract, assert_checkpoint)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(args.seed)
random.seed(args.seed); np.random.seed(args.seed)
from supermimic.curriculum import Curriculum, TASKS
from supermimic.curriculum_runtime import atomic_save, rng_state, restore_rng, emit, evaluate
curriculum = Curriculum(json.load(open(args.curriculum_config))) if args.curriculum_config else None
signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
dev = "cuda:0"

if args.goal_mode == 'motion':
    motion_reward_cfg = (RewardCfg() if args.motion_obs_mode == 'legacy_v1' and args.motion_reward_config is None
                         else resolve_motion_reward(args.motion_reward_config))
else:
    if args.motion_reward_config is not None:
        raise ValueError('motion_reward_config applies to Motion training only')
    args.motion_obs_mode = 'legacy_v1'
    motion_reward_cfg = RewardCfg()
args.motion_reward_config = asdict(motion_reward_cfg) if args.goal_mode == 'motion' else None

args.reward_config = asdict(
    motion_reward_cfg if task_cfg is None else task_cfg)
os.makedirs(args.log_dir, exist_ok=True)


kin = RobotKinematics(dev)
terrain_provider = None
if ROBOT_KIND == 'roban':
    from supermimic.roban_runtime import build_task
    motion, goals, mdp, terrain_provider = build_task(args, kin, motion_files, dev)
    if m4:
        goal_spec = goals.spec
    else:
        (log_path/'roban_reference_validation.json').write_text(json.dumps(motion.validation, indent=2))
    data_contract = motion.validation
elif stair_cfg:
    from supermimic.stair_training import library as stair_library
    motion=stair_library(stair_cfg,kin,dev)
    (log_path/'stair_data_validation.json').write_text(json.dumps(motion.validation,indent=2))
elif args.m34_dataset:
    from supermimic.m34 import motion_library
    motion = motion_library(args.m34_dataset, kin, dev, 'train')
    (log_path/'m34_data_validation.json').write_text(json.dumps(motion.validation, indent=2))
else:
    motion = MotionLib(motion_files, kin, dev) if task_cfg is None else None
if ROBOT_KIND != 'roban':
    default_q = torch.tensor(default_joint_pos_list(), device=dev)
    goals = GoalSource(motion, kin, goal_spec, foot_config,
                    command_mode='alternating' if task_cfg is None else task_cfg.command_mode)
    mdp = (
        MDP(kin, motion, default_q, motion_reward_cfg, observation_mode=args.motion_obs_mode)
        if task_cfg is None
        else FootTrackingMDP(kin, default_q, goals, task_cfg))
if stair_cfg:
    from supermimic.stair_runtime import StairTrackingMDP
    if motion_reward_cfg.root_position_mode!='xyz':parser.error('Stair task requires explicit XYZ root reward')
    mdp=StairTrackingMDP(kin,motion,default_q,motion_reward_cfg,observation_mode=args.motion_obs_mode)

if args.goal_mode=='reference_foot':
    from supermimic.reference_foot import ReferenceFootGoalSource,PairedMotionMDP
    goals=ReferenceFootGoalSource(motion,kin,goal_spec,foot_config)
    mdp=PairedMotionMDP(kin,motion,default_q,goals,RewardCfg())
if args.prior_config:
    if task_cfg is None: raise ValueError('Dynamic prior adapter currently requires independent Foot RL')
    from supermimic.upper_prior import PriorFootMDP,resolve_prior
    prior_cfg=resolve_prior(args.prior_config)
    args.prior_config=asdict(prior_cfg)
    mdp=PriorFootMDP(kin,default_q,goals,task_cfg,prior_cfg)
if curriculum:
    goals.curriculum = curriculum
    goals.command_duration = task_cfg.episode_steps * .02 + goal_spec.frames * goal_spec.dt + 1
obs_dim = mdp.obs_dim
args.obs_dim = obs_dim
action_context = m4 and mdp.action_history_enabled
reward_scale = mdp.max_step_reward
value_upper = return_upper_bound(reward_scale,args.gamma) if mdp.is_foot else None
imagination_contract=dict(version=1,action_clip=ACTION_LIMIT,bootstrap_upper_bound=value_upper)
if joint_range_action:
    imagination_contract.update(version=2, action_clip=None, action_transform='tanh', action_range=[-1., 1.])
if standing_action:
    imagination_contract.update(version=3, action_clip=None, action_transform='linear',
                                wm_action='standing_equivalent_of_limit_clamped_pd_target')
    if m44:
        imagination_contract.update(version=4, pd_target_clip=False, wm_action='raw_linear_pd_target')
    if action_context:
        imagination_contract.update(version=5, startup_steps=mdp.startup_steps,
                                    wm_action='standing_equivalent_of_executed_pd_target',
                                    action_history='two_previous_raw_actions_within_episode')
actor_lr=args.lr if args.actor_lr is None else args.actor_lr
critic_lr=args.lr if args.critic_lr is None else args.critic_lr
K, Hh, HZ = args.goal_frames, args.hist, args.horizon
GOAL_DIM = goal_spec.dim
CRITIC_IN = mdp.critic_dim + GOAL_DIM

wm = WorldModel(terrain_provider=terrain_provider).to(dev)
if args.goal_mode=='reference_foot':
    from supermimic.networks import DualEncoderPolicy
    pi=DualEncoderPolicy(hist=Hh).to(dev)
else:
    pi = Policy(obs_dim, GOAL_DIM, hist=Hh, obs_adapter=args.obs_adapter, squash_action=joint_range_action).to(dev)
if ROBOT_KIND == 'roban':
    # Critic width is the E0/E1 ablation. Its initialization must not change
    # subsequent shared reset/action-noise RNG draws merely by consuming more RNG.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        critic = Critic(CRITIC_IN).to(dev)
else:
    critic = Critic(CRITIC_IN).to(dev)
opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
opt_pi = torch.optim.Adam(pi.parameters(), lr=actor_lr, betas=(0.9, 0.95))
opt_v = torch.optim.Adam(critic.parameters(), lr=critic_lr)
start_it, wall_offset, sim_steps = 0, 0.0, 0
pretrain_iterations = args.m51_critic_warmup if m51 else args.m50_adaptation_iters if m50 else args.m411_critic_warmup
warmup_label = 'after_adaptation' if m50 and not m51 else 'critic_warmup'
supervision_steps=0
physics_metadata = mdp.physics_contract(args) if m4 else physical_contract(obs_dim=obs_dim)
if stair_cfg:
    physics_metadata['solver']={**physics_metadata['solver'],'use_mujoco_contacts':False}
    physics_metadata['collision_decimation']=1
    physics_metadata['terrain_task']=data_contract
transfer_report=None
critic_warmup_done = 0
git_sha=os.environ.get('S54_SOURCE_GIT_SHA')
if not git_sha:
    git_result=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,text=True,capture_output=True)
    git_sha=git_result.stdout.strip() if git_result.returncode==0 else 'unavailable-in-container'
if m43:
    from importlib.metadata import version

    args.environment_versions = {p: version(p) for p in ('isaaclab', 'newton', 'mujoco-warp', 'warp-lang', 'torch')}
    args.git_commit = git_sha
if m46 or m49 or whole_body:
    if m50:
        source_status_key = 'M52_SOURCE_GIT_STATUS' if m52 else 'M51_SOURCE_GIT_STATUS' if m51 else 'M50_SOURCE_GIT_STATUS'
    else:
        source_status_key = f'M4{revision if upper_ablation else 6}_SOURCE_GIT_STATUS'
    args.git_worktree = os.environ.get(source_status_key)
    if args.git_worktree is None:
        result = subprocess.run(['git', 'status', '--porcelain'], cwd=REPO, text=True, capture_output=True)
        args.git_worktree = result.stdout.rstrip() if result.returncode == 0 else '容器未挂载 Git 元数据'
if whole_body:
    args.training_gpu = torch.cuda.get_device_name(dev)


def evaluate_m4_checkpoint(checkpoint=None):
    if m50:
        module = 'm52_evaluation' if m52 else 'm51_evaluation' if m51 else 'm50_evaluation'
        command = [sys.executable, '-m', f'supermimic.integration.{module}', 'periodic',
                   '--out', str(log_path.parent), '--library', args.m50_library,
                   '--reference-root', args.m50_reference_root, '--checkpoint', str(checkpoint or log_path / 'last.pt')]
        subprocess.run(command, cwd=REPO, check=True)
        return
    evaluator = 'm43_evaluation' if m43 else 'm4_evaluation'
    command = [sys.executable, '-m', 'supermimic.integration.' + evaluator,
               '--out', str(log_path.parent), '--m4_reference', args.m4_reference, '--device', dev]
    if not m43:
        command.append('--latest_only')
    if m46:
        command.append('--upper-metrics-only')
    elif upper_ablation:
        command.append(f'--m4{revision}-periodic')
    elif m45 or m451:
        command.append('--upper-body')
    subprocess.run(command, cwd=REPO, check=True)


def assert_periodic_final():
    summary = json.loads((log_path.parent / 'evaluation' / 'results.json').read_text())
    if m50:
        best = 'best_contact_stability.pt' if m51 else 'best_robust.pt'
        if (str(args.iterations) not in summary['milestones'] or not (log_path / 'best_multi_reference.pt').is_file()
                or not (log_path / best).is_file()):
            raise RuntimeError('M5.0 最终周期评估及候选权重尚未落盘')
        return
    best = 'best_foot.pt' if m411 else 'best.pt'
    if (not (log_path / milestone_name(args.iterations)).is_file()
            or str(args.iterations) not in summary['milestones']
            or summary.get('best_iteration') is None or not (log_path / best).is_file()):
        raise RuntimeError('最终周期评估或 Best 未落盘，请检查本组训练日志')


def write_m4_config():
    args_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(vars(args), indent=2)+'\n')
    os.replace(temporary, args_path)


launch_metadata=json.loads((log_path/'launch.json').read_text()) if (log_path/'launch.json').exists() else {}
source_snapshot=launch_metadata.get('source_snapshot')
if args.resume:
    ck = torch.load(args.resume, map_location=dev, weights_only=False)
    if m46 or upper_ablation or m50:
        saved_args = ck.get('args', {})
        # 启动器在写检查点前会展开无直播默认值，恢复前此参数仍是原始命令行默认值。
        resume_metadata = {'resume', 'fresh_start', 'm4_reference', 'roban_config', 'log_dir', 'livestream',
                           'git_worktree', 'git_commit', 'environment_versions'}
        if m46:
            resume_metadata.update(('m47_smoke', 'm47_startup_steps', 'upper_action_smoothness_weight',
                                    'm48_group', 'm48_check', 'm48_smoke'))
        if not m49:
            resume_metadata.update(('m49_group', 'm49_check', 'm49_smoke',
                                    'soft_body_prior_weight', 'soft_body_prior_tolerance'))
        if not m410:
            resume_metadata.update(('m410_group', 'm410_check', 'm410_smoke'))
        if not m411:
            resume_metadata.update(('m411_group', 'm411_check', 'm411_smoke', 'm411_critic_warmup'))
        if not m50:
            resume_metadata.update(('m50_library', 'm50_reference_root', 'm50_adaptation_iters', 'm50_smoke', 'm50_check'))
        if not m51:
            resume_metadata.update(('m51_group', 'm51_critic_warmup'))
        if whole_body:
            resume_metadata.add('training_gpu')
        differences = [key for key, value in vars(args).items() if key not in resume_metadata
                       and saved_args.get(key) != value]
        checkpoint_transfer = ck.get('transfer_report')
        strategy = (checkpoint_transfer or {}).get('strategy')
        if m52:
            valid_transfer = strategy == M52_TRANSFER
        elif m51:
            valid_transfer = strategy == 'm51_m50_5500_all_models'
        elif m50:
            valid_transfer = strategy == 'm50_b1250_all_models' or (checkpoint_transfer is None and fresh_run)
        else:
            valid_transfer = strategy == 'm411_c9000_all_models' if m411 else checkpoint_transfer is None
        if differences or not valid_transfer or not {'rng', 'step_acc'} <= ck.keys():
            raise ValueError(f'仅恢复同组原始实验的检查点，配置差异：{differences}')
        ahead = [path.name for path in log_path.glob('iter_*.pt')
                 if path.stem[5:].isdigit() and int(path.stem[5:]) > ck['iteration']]
        if ahead:
            raise ValueError(f'恢复检查点落后于已有里程碑：{ahead}')
        if upper_ablation and Path(args.resume).name != 'last.pt' and (log_path / 'last.pt').exists():
            raise ValueError('本组已有 Last，请从 Last 恢复')
    if m451 and (ck.get('transfer_report') or {}).get('strategy') != 'f2_body_prior_actor_wm_only':
        raise ValueError('M4.5.1 恢复只接受本短训生成的同权重检查点')
    assert_checkpoint(ck)
    if m4:
        mdp.assert_checkpoint_task(ck)
    if guarded_action:
        mdp.assert_wm_statistics(ck)
    if ck.get('imagination_contract') not in (None,imagination_contract):
        raise ValueError('Imagination action/value contract mismatch')
    if ck.get('imagination_contract') is None:
        print('Legacy checkpoint: applying real/imagined action consistency and valid foot return bound; '
              'network dimensions and physical task unchanged',flush=True)
    goal_spec.assert_checkpoint(ck, foot_config)
    if not m4:
        if ROBOT_KIND == 'roban':
            if ck.get('task_contract') != mdp.contract(): raise ValueError('Roban task/critic/terrain resume mismatch')
        else:
            assert_task_checkpoint(ck, mdp)
    assert_policy_checkpoint(ck,Hh,args.obs_adapter,not args.freeze_decoder,
                             obs_dim=obs_dim, observation_mode=args.motion_obs_mode)
    if ck.get('physical_contract',physics_metadata)!=physics_metadata: raise ValueError('Resume physics contract mismatch')
    if ck.get('data_contract',data_contract)!=data_contract: raise ValueError('Resume data manifest mismatch')
    if "critic" not in ck: raise ValueError("Resume requires last.pt with critic and optimizers")
    wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"]), critic.load_state_dict(ck["critic"])
    opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"]), opt_v.load_state_dict(ck["opt_v"])
    transfer_report=ck.get('transfer_report')
    supervision_steps=ck.get('stage_stats',{}).get('supervision_steps',0)
    if curriculum:
        if 'curriculum' not in ck: raise ValueError('Full curriculum resume requires curriculum/RNG state')
        curriculum.load_state_dict(ck['curriculum'])
    start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
    if hand_tracking:
        critic_warmup_done = ck['adaptation_iterations' if m50 else 'critic_warmup_iterations']
        if not 0 <= critic_warmup_done <= pretrain_iterations or (start_it and critic_warmup_done != pretrain_iterations):
            raise ValueError('检查点的适应轮次和联合轮次不一致')
        warmup_path = log_path / f'{warmup_label}.pt'
        if not start_it and critic_warmup_done == pretrain_iterations and not warmup_path.exists():
            os.link(args.resume, warmup_path)
        if m50 and not m51 and not start_it and critic_warmup_done == pretrain_iterations and args.m4_eval_every:
            evaluate_m4_checkpoint(warmup_path)
    if (m46 or upper_ablation or m50) and not 0 <= start_it <= args.iterations:
        raise ValueError(f'检查点轮次 {start_it} 超出本实验预算')
    if (m46 or upper_ablation or m50) and args.m4_eval_every and start_it > 0 and start_it % args.m4_eval_every == 0:
        # 进程可能在发布 Last 后、提交周期评估前中断。先补完同一冻结权重的轻量评估。
        milestone = log_path / milestone_name(start_it)
        if not milestone.exists():
            os.link(args.resume, milestone)
        if upper_ablation and not (log_path / 'last.pt').exists():
            os.link(args.resume, log_path / 'last.pt')
        evaluate_m4_checkpoint()
        if start_it == args.iterations:
            assert_periodic_final()
            print(f'M4 completed | iteration={start_it} | sim_steps={sim_steps} | '
                  f'wall={wall_offset + time.monotonic() - PROCESS_START:.1f}s', flush=True)
            sys.exit(0)
    if args.iterations <= start_it:
        raise ValueError(f'Resume checkpoint is at iteration {start_it}; --iterations must be larger '
                         f'(it is the total target, not additional iterations; got {args.iterations})')
    print(f'Resuming iteration {start_it} -> {args.iterations}; model and optimizer states restored. '
          'Replay buffer will be refilled.', flush=True)

if m411 and not args.resume:
    source_path = Path(REPO) / M411_SOURCE
    source = torch.load(source_path, map_location='cpu', weights_only=False, mmap=True)
    assert_checkpoint(source)
    mdp.assert_wm_statistics(source)
    task = mdp.contract()
    for key in HAND_CONFIG:
        task['config'].pop(key, None)
    contracts = dict(task_contract=task, physical_contract=physics_metadata, data_contract=data_contract,
                     goal_contract=goal_spec.contract(), imagination_contract=imagination_contract,
                     policy_contract=policy_contract(Hh, args.obs_adapter, not args.freeze_decoder,
                                                     obs_dim=obs_dim, observation_mode=args.motion_obs_mode))
    differences = [key for key, value in contracts.items() if source.get(key) != value]
    if source['iteration'] != 9000 or source['args'].get('m410_group') != 'c' or differences:
        raise ValueError(f'M4.11 初始化需要 M4.10-C9000 及一致的任务契约，差异 {differences}')
    for name, model in (('pi', pi), ('wm', wm), ('critic', critic)):
        model.load_state_dict(source[name], strict=True)
    transfer_report = dict(strategy='m411_c9000_all_models', source_path=str(M411_SOURCE),
                           source_iteration=source['iteration'], source_sim_steps=source['sim_steps'],
                           source_total_wall_seconds=source['wall_time'])
    del source

if m50 and not (args.resume or args.fresh_start):
    if m52:
        from supermimic.integration.m52_task import initialize_m51 as initialize_multi
    elif m51:
        from supermimic.integration.m51_task import initialize_m50 as initialize_multi
    else:
        from supermimic.integration.m50_task import initialize_b1250 as initialize_multi

    if (obs_dim, GOAL_DIM, CRITIC_IN) != (69, 320, 970):
        raise ValueError('M5.0 必须继承 Roban 的 69D 观测、320D 目标和 970D Critic')
    transfer_report = initialize_multi(
        dict(pi=pi, wm=wm, critic=critic), mdp, goals, physics_metadata,
        policy_contract(Hh, args.obs_adapter, not args.freeze_decoder, obs_dim=obs_dim,
                        observation_mode=args.motion_obs_mode), imagination_contract)

if m451 and not args.resume:
    from supermimic.transfer import initialize_f2_body_prior

    source = torch.load(args.m451_source, map_location='cpu', weights_only=False, mmap=True)
    mdp.assert_wm_statistics(source)
    transfer_report = initialize_f2_body_prior(source, wm, pi, dict(
        task_contract=mdp.contract(), physical_contract=physics_metadata, data_contract=data_contract,
        goal_contract=goal_spec.contract(), imagination_contract=imagination_contract,
        policy_contract=policy_contract(Hh, args.obs_adapter, not args.freeze_decoder,
                                        obs_dim=obs_dim, observation_mode=args.motion_obs_mode)))
    transfer_report['source_path'] = str(Path(args.m451_source).resolve())
    del source

if args.init_weights:
    transfer = torch.load(args.init_weights, map_location=dev, weights_only=False)
    assert_checkpoint(transfer)
    if ROBOT_KIND == 'roban':
        from supermimic.roban_runtime import initialize_compatible
        transfer_report = initialize_compatible(transfer,wm,pi,physics_metadata,hist=Hh,goal_contract=goal_spec.contract())
    else:
        goal_spec.assert_checkpoint(transfer)  # representation must match; sampling/task may change explicitly
        if transfer['args']['hist'] != args.hist: raise ValueError('Transfer requires matching Actor history')
        if transfer.get('policy_contract',{}).get('obs_adapter','identity')!=args.obs_adapter:
            raise ValueError('Compatible Actor transfer requires matching observation semantics')
        if transfer.get('policy_contract', {}).get('obs_dim', OBS_DIM) != obs_dim:
            raise ValueError('Actor observation dimension mismatch: new Root feedback requires a new 103D policy')
        if transfer.get('physical_contract',physics_metadata)!=physics_metadata:raise ValueError('Actor transfer physics mismatch')
        wm.load_state_dict(transfer['wm']); pi.load_state_dict(transfer['pi'])
        pi.goal_norm.mean.zero_(); pi.goal_norm.var.fill_(1); pi.goal_norm.count.zero_()
        print('TRANSFER: WM+Actor (goal normalization reset); fresh Critic/optimizers/replay/curriculum. Source iteration',
              transfer['iteration'], flush=True)

if args.init_from_flat:
    from supermimic.stair_training import initialize_flat
    from supermimic.m34 import sha256 as stair_sha256
    if (stair_cfg.get('source_checkpoint_sha256') and
            stair_sha256(args.init_from_flat)!=stair_cfg['source_checkpoint_sha256']):
        raise ValueError('Flat initialization differs from the pinned experiment checkpoint')
    transfer=torch.load(args.init_from_flat,map_location=dev,weights_only=False)
    transfer_report=initialize_flat(transfer,wm,pi,physics_metadata,goal_spec,Hh,
        normalizer_policy=stair_cfg.get('normalizer_policy','frozen_source'))
    transfer_report['source_path']=str(Path(args.init_from_flat).resolve())
    (log_path/'flat_to_stair_transfer.json').write_text(json.dumps(transfer_report,indent=2))
    print('T1.5: Flat WM+MotionEncoder+SharedDecoder+normalizers copied; fresh reward-specific Critic/optimizers/replay',flush=True)

if args.init_from_motion:
    transfer=torch.load(args.init_from_motion,map_location=dev,weights_only=False)
    transfer_report=initialize_from_motion(transfer,wm,pi,hist=Hh,components=args.transfer_components)
    transfer_report['source_path']=str(Path(args.init_from_motion).resolve())
if args.init_from_bridge:
    from supermimic.transfer import initialize_from_bridge
    transfer=torch.load(args.init_from_bridge,map_location=dev,weights_only=False)
    transfer_report=initialize_from_bridge(transfer,wm,pi,hist=Hh)
    transfer_report['source_path']=str(Path(args.init_from_bridge).resolve())
if transfer_report:
    if not (m451 or hand_tracking):
        (log_path/'transfer_report.json').write_text(json.dumps(transfer_report,indent=2))
    print('Cross-task initialization:',json.dumps({k:v for k,v in transfer_report.items() if k!='mapping'}),flush=True)
if args.freeze_decoder:
    for parameter in pi.dec.parameters(): parameter.requires_grad_(False)
if args.goal_mode=='reference_foot':
    for parameter in pi.motion_enc.parameters():parameter.requires_grad_(False)
actor_parameters=tuple(p for p in pi.parameters() if p.requires_grad)

if not m4:
    args_temp=log_path/'args.json.tmp'
    args_temp.write_text(json.dumps(vars(args),indent=2))
    os.replace(args_temp,log_path/'args.json')


# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((NUM_J,), 3e-3), torch.full((NUM_J,), 5e-2))).to(dev)  # fmt: skip


def _wm_action(a, f):
    # 采样、回放与想象使用同一物理目标；Actor 历史仍保留实际输出的动作。
    return mdp.wm_action(a, f) if m4 else a


def _wm_rollout(s, a, xi):
    S = unflatten_state(s[:, 0])
    preds = []
    for j in range(args.wm_window):
        S = wm.step(S, a[:, j], normalize_xi(xi[:, j]))
        preds.append(S)
    return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)


def _wm_err(S_hat, s1, mask):
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    e = (wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2  # (B, k, D)
    e = e.mean(-1)
    return (e * mask).sum() / s1.shape[0], e[:, -1]


def _imag_step(S, o_hist, a_prev, f, xi, xi_n, g, ep_step, command, startup_q=None):
    a = pi(o_hist, g)
    clip_fraction = a.new_zeros(())
    if standing_action:
        clip_fraction = mdp.target_clip_mask(a.detach()).float().mean()
    elif not joint_range_action:
        clip_fraction = (a.abs() > ACTION_LIMIT).float().mean()
        a = bound_action(a)
    if action_context:
        target = mdp.executed_target(a, startup_q, ep_step)
        executed = mdp.effective_action(target)
    else:
        target = None
        executed = _wm_action(a, f)
    S1 = wm.step(S, executed, xi_n)
    f1 = f + 1
    step1 = ep_step + 1
    o1 = mdp.add_noise(goals.obs(mdp, S1, a, xi, f1, command))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    return S1, a, o_hist, f1, goals.goal(S1, f1, step1, command), step1, clip_fraction, target


def _reward_term(S, f, a, a_prev, ep_step, command, **history):
    if mdp.is_foot or getattr(mdp,'is_reference_foot',False):
        r, terms = mdp.reward(S,f,a,a_prev,terms=True,command=command,ep_step=ep_step,**history)
        foot_reward = terms['foot_objective']
    else:
        r = mdp.reward(S,f,a,a_prev)
        foot_reward = torch.zeros_like(r)
    done = mdp.soft_termination(S,f,command=command,ep_step=ep_step)
    if m4:
        # 参考结束后没有下一步，想象轨迹的 Critic 同样在终点停止 bootstrap。
        done = torch.where(f >= motion.clip_end[motion.clip_id[f]], 1., done)
    baseline = terms['m410_reward'] if hand_tracking else r
    return r, done, foot_reward, baseline


def _critic_input(S, f, xi_n, g, ep_step, command):
    return torch.cat((mdp.critic_features(S, f, xi_n,command=command,ep_step=ep_step), g), -1)


compile_fn = (lambda f: f) if args.no_compile else torch.compile
wm_rollout = compile_fn(_wm_rollout)
wm_err = compile_fn(_wm_err)
imag_step = compile_fn(_imag_step)
reward_term = compile_fn(_reward_term)
critic_input_c = compile_fn(_critic_input)


def lr_at(it: int) -> float:
    if it >= args.lr_decay_iters:
        return args.lr * 0.03
    return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


# ------------------------------------------------------------------------------------------------
m52_update_batch = None


def wm_update():
    global m52_update_batch
    w = buffer.sample_windows(args.wm_batch, args.wm_window, include_frames=residual_action)
    if m52:
        m52_update_batch = dict(update='wm', samples=w)
    executed = w['a_exec'] if action_context else _wm_action(w['a'],w.get('f'))
    S_hat = wm_rollout(w['s'], executed, w['xi'])
    loss, e_last = wm_err(S_hat, w["s1"], w["mask"])
    opt_wm.zero_grad(set_to_none=True)
    if not torch.isfinite(loss): raise FloatingPointError("Nonfinite WM/Policy loss")
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(wm.parameters(), 1.0, error_if_nonfinite=True)
    opt_wm.step()
    m = w["mask"][:, -1]
    return loss.detach() / args.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1), grad_norm.detach()


@torch.no_grad()
def m43_wm_metrics():
    # 旁路抽样复用正常 Replay 的遮罩，恢复随机状态并保持训练更新完全相同。
    if buffer.size < 8:
        return {}
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        batch = buffer.sample_windows(args.wm_batch, 8, include_frames=True, include_reward=True)
    state = unflatten_state(batch['s'][:, 0])
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    values = {}
    for k in range(8):
        f, a = batch['f'][:, k], batch['a'][:, k]
        executed = batch['a_exec'][:,k] if action_context else _wm_action(a,f)
        state = wm.step(state, executed, normalize_xi(batch['xi'][:, k]))
        if k not in (0, 3, 7):
            continue
        valid = batch['mask'][:, k].bool()
        if not valid.any():
            continue
        pred = tuple(x[valid] for x in state)
        true = unflatten_state(batch['s1'][:, k][valid])
        frame = f[valid] + 1
        ep_step = batch['ep_step'][:, k][valid].long() + 1 if m50 else frame
        command = motion.anchor(frame - ep_step)
        history = {name:batch[name][:,k][valid] for name in ('a_prev2','pd_target')} if action_context else {}
        reward = mdp.reward(pred, frame, a[valid], batch['a_prev'][:, k][valid], command=command, ep_step=ep_step, **history)
        error = (wm.state_error(pred, true) / std).square().mean(-1)
        foot_sq = (motion.feet(pred)[0] - motion.feet(true)[0]).square().sum(-1).mean(-1)
        values[f'wm_error_{k+1}'] = float(error.mean())
        values[f'wm_foot_rmse_{k+1}'] = float(foot_sq.mean().sqrt())
        values[f'wm_reward_bias_{k+1}'] = float((reward - batch['r'][:, k][valid]).mean() / reward_scale)
        if m50:
            for clip_id, name in enumerate(motion.names):
                select = command.clip_id == clip_id
                if select.any():
                    values[f'{name}/wm_error_{k+1}'] = float(error[select].mean())
                    values[f'{name}/wm_foot_rmse_{k+1}'] = float(foot_sq[select].mean().sqrt())
        if whole_body:
            position = (pred[0] - true[0]).square().sum(-1).mean().sqrt()
            trace = (pred[1] * true[1]).sum((-1, -2))
            orientation = torch.acos(((trace - 1.) * .5).clamp(-1., 1.)).rad2deg().square().mean().sqrt()
            upper = mdp.waist_ids + mdp.upper_ids
            joints = (pred[4][:, upper] - true[4][:, upper]).square().mean().sqrt()
            values.update({f'{name}_{k+1}': float(value) for name, value in
                           zip(M410_WM_METRICS, (position, orientation, joints))})
    return values


def sample_valid_starts(B: int):
    starts, hists, n = [], [], 0
    attempts = 0
    while n < B:
        attempts += 1
        if attempts > 1000: raise RuntimeError("Replay contains no valid imagination starts; reduce horizon or use longer clips")
        s, h = buffer.sample_starts(2 * B, Hh)
        f = s["f"]
        ok = (s['ep_step']+HZ <= task_cfg.episode_steps if mdp.is_foot else
              f + HZ <= motion.clip_end[motion.clip_id[f]])
        idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
        starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
        n += len(idx)
    cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
    return cat(starts), cat(hists)


PROFILE = bool(int(os.environ.get("SM_PROFILE", "0")))


def policy_update(audit_gradients=False, *, critic_only=False):
    global supervision_steps, m52_update_batch
    if PROFILE:
        torch.cuda.synchronize(); tq0 = time.time()
    s, h = sample_valid_starts(args.pi_batch)
    if m52:
        m52_update_batch = dict(update='critic' if critic_only else 'actor_critic', samples=s, history=h)
    if upper_smoke or args.m50_smoke:
        # 起点从真实 Replay 抽取，上一帧的 Previous Action 应等于起点的上两步动作。
        torch.testing.assert_close(s['a_prev2'], h['a_prev'][:, -2], rtol=0, atol=0)
    S = unflatten_state(s["s"])
    xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
    ep_step = s['ep_step']
    command = buffer.commands(s['command_id'], frames=f, episode_steps=ep_step) if goals.is_foot else None
    xi_n = normalize_xi(xi)
    with torch.no_grad():
        o_hist = goals.obs(mdp, unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"], command)
    o_hist = mdp.add_noise(o_hist)
    wm.zero_grad(set_to_none=True)  # Actor gradients may traverse WM inputs, never its parameters.
    set_requires_grad(wm, False)
    set_requires_grad(critic, False)
    B = args.pi_batch
    audit_freeze=args.check_foot_gradients or audit_gradients or args.goal_mode=='reference_foot' or args.roban_gradient_audit
    wm_versions=tuple(p._version for p in wm.parameters()) if audit_freeze else None
    decoder_versions=tuple(p._version for p in pi.dec.parameters()) if audit_freeze and args.freeze_decoder else None
    # 预热仅拟合 Critic，梯度审计时才保留通过冻结 WM 的计算图。
    with torch.set_grad_enabled(not critic_only or audit_gradients):
        g0 = goals.goal(S, f, ep_step, command)
        g = g0
        S0 = S
        Ss, As, Aps, Fs, Gs, Steps = [], [], [], [], [], []
        previous_two, targets = [], []
        startup_q, a_prev2 = s.get('startup_q'), s.get('a_prev2')
        clip_fractions=[]
        for t in range(HZ):
            Aps.append(a_prev)
            if action_context:
                previous_two.append(a_prev2)
                # 保留想象步之间的原始动作梯度；只有 Replay 读取的历史天然无梯度。
                a_prev2 = a_prev
            S, a_prev, o_hist, f, g, ep_step, clip_fraction, target = imag_step(
                S, o_hist, a_prev, f, xi, xi_n, g, ep_step, command, startup_q)
            if action_context:
                targets.append(target)
            Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
            Steps.append(ep_step); clip_fractions.append(clip_fraction)
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
        f_all = flat(Fs)
        command_all = command.repeat(HZ) if command is not None else None
        history = dict(a_prev2=flat(previous_two),pd_target=flat(targets)) if action_context else {}
        r, d, foot_reward, baseline = reward_term(S_all, f_all, flat(As), flat(Aps), flat(Steps), command_all, **history)
        r, d = r.view(HZ, B), d.view(HZ, B)
        alive = torch.cumprod(1 - d, 0)  # c_{t+1}
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0)
        c = alive[-1]
        r_sum = (c_t * r).detach().sum(0)
        rews, dones = r.detach(), d.detach()
        reward_J = J
        raw_v = torch.zeros_like(J); bootstrap = torch.zeros_like(J)
        if not args.no_critic:
            raw_v = critic(_critic_input(S, f, xi_n, g, ep_step, command))
            bootstrap = (args.gamma**HZ) * c * bound_bootstrap(raw_v,value_upper)
            J = J + bootstrap
        loss = -J.mean()
        latent_loss=torch.zeros((),device=dev)
        if args.paired_latent_weight:
            with torch.no_grad(): target_latent=pi.encode_motion(motion.goal(f_all,K))
            latent_loss=(pi.encode(flat(Gs))-target_latent).square().mean()
            loss=loss+args.paired_latent_weight*latent_loss
            supervision_steps+=HZ*B
    foot_gn = torch.zeros((), device=dev)
    if (mdp.is_foot and (args.check_foot_gradients or audit_gradients)) or (ROBOT_KIND == 'roban' and args.roban_gradient_audit):
        # Exclude direct action costs, posture, base terms, critic and survival derivatives.
        foot_loss = -(disc*c_t.detach()*foot_reward.view(HZ,B)).sum(0).mean()
        fg = torch.autograd.grad(foot_loss,actor_parameters,retain_graph=True,allow_unused=True)
        foot_gn = torch.stack([g.detach().float().square().sum() for g in fg if g is not None]).sum().sqrt()
        if not torch.isfinite(foot_gn): raise FloatingPointError('Nonfinite pure foot Actor gradient')
        if ROBOT_KIND == 'roban' and foot_gn <= 0:
            raise RuntimeError('Roban primary foot reward produced no Actor gradient')
    gradient_diagnostics={}
    if audit_gradients:
        params=actor_parameters
        rg=torch.autograd.grad(-reward_J.mean(),params,retain_graph=True,allow_unused=True)
        bg=(torch.autograd.grad(-bootstrap.mean(),params,retain_graph=True,allow_unused=True)
            if bootstrap.requires_grad else tuple(None for _ in params))
        def grad_norm(gs):
            return sum((g.detach().float().square().sum() for g in gs if g is not None),torch.zeros((),device=dev)).sqrt()
        def grad_dot(left, right):
            return sum(((x.detach().float()*y.detach().float()).sum() for x,y in zip(left,right)
                        if x is not None and y is not None), torch.zeros((), device=dev))
        rn,bn=grad_norm(rg),grad_norm(bg)
        gradient_diagnostics=dict(actor_reward_gradnorm=rn,actor_bootstrap_gradnorm=bn,
            actor_grad_cosine=grad_dot(rg,bg)/(rn*bn).clamp_min(1e-20),pure_foot_gradnorm=foot_gn)
        if mdp.is_foot or m43:
            _,heads=mdp.reward(S_all,f_all,flat(As),flat(Aps),True,command=command_all,ep_step=flat(Steps),**history)
            if m43:
                names = ('swing_position', 'swing_clearance', 'touchdown', 'upright', 'survival')
                if m45 or m451 or action_context:
                    names += ('body_prior',)
                if action_context:
                    names += ('upper_action_smoothness',)
                objectives = {name: heads[f'{name}_contribution'] for name in names}
                objectives['foot'] = foot_reward
                if m49:
                    objectives['soft_body_prior'] = heads.get('soft_body_prior_contribution', torch.zeros_like(foot_reward))
                if whole_body:
                    objectives.update({term: heads.get(f'{term}_contribution', torch.zeros_like(foot_reward))
                                       for term in WHOLE_BODY_TERMS})
                    gradient_diagnostics['imagined_baseline_reward'] = (
                        c_t.detach() * heads['baseline_reward'].view(HZ, B)).mean().detach()
                if hand_tracking:
                    objectives.update(hand_huber=heads['hand_huber_contribution'],
                                      root=sum(objectives[term] for term in ROOT_TERMS),
                                      joint_huber=sum(objectives[term] for term in HUBER_TERMS))
                if m51:
                    objectives.update({term: heads[f'{term}_contribution'] for term in M51_TERMS})
                    objectives.update(foot_position=heads['swing_position_contribution']+heads['stance_position_contribution'],
                                      swing_orientation=heads['swing_orientation_contribution'])
                if m52:
                    objectives['root_tilt'] = heads['root_tilt_contribution']
            else:
                objectives={k:v for k,v in heads.items() if k.endswith('_contribution')}
                objectives.update(base_height=heads['base_height']*task_cfg.w_base_height*.02,
                                  base_upright=heads['base_upright']*task_cfg.w_upright*.02,
                                  foot=heads['foot_objective'])
            pairs = (M410_GRADIENT_PAIRS + M411_GRADIENT_PAIRS if hand_tracking else
                     M410_GRADIENT_PAIRS if m410 else M49_GRADIENT_PAIRS if m49 else ())
            if m51:
                pairs += M51_GRAD_PAIRS
            if m52:
                pairs += M52_GRAD_PAIRS
            part_grads = {'bootstrap': bg}
            for key,values in objectives.items():
                objective=-(disc*c_t.detach()*values.view(HZ,B)).sum(0).mean()
                # 零权重奖励项是常量，分项审计应返回零范数。
                gradients=(torch.autograd.grad(objective,params,retain_graph=True,allow_unused=True)
                           if objective.requires_grad else ())
                gradient_diagnostics['term_'+key+'_actor_gradnorm']=grad_norm(gradients)
                if any(key in pair for pair in pairs):
                    part_grads[key] = gradients
            for left,right in pairs:
                x,y = part_grads[left], part_grads[right]
                gradient_diagnostics[f'grad_cosine_{left}_{right}'] = grad_dot(x,y)/(grad_norm(x)*grad_norm(y)).clamp_min(1e-20)
            if hand_tracking:
                foot_gn = gradient_diagnostics['term_foot_actor_gradnorm']
        if m50:
            # 同一批想象起点按参考分组度量梯度一致性，更新仍用原始全批目标。
            clip_grads = {}
            for clip_id, name in enumerate(motion.names):
                selected = command.clip_id == clip_id
                if selected.any():
                    clip_grads[name] = torch.autograd.grad(-J[selected].mean(), params, retain_graph=True, allow_unused=True)
                    gradient_diagnostics[f'{name}/actor_gradnorm'] = grad_norm(clip_grads[name])
            for i, name in enumerate(motion.names):
                for other in motion.names[i+1:]:
                    if name in clip_grads and other in clip_grads:
                        left, right = clip_grads[name], clip_grads[other]
                        gradient_diagnostics[f'clip_grad_cosine_{name}_{other}'] = (
                            grad_dot(left, right) / (grad_norm(left) * grad_norm(right)).clamp_min(1e-20))
    if m52 and gradient_diagnostics and not torch.stack(tuple(gradient_diagnostics.values())).isfinite().all():
        raise FloatingPointError('M5.2 nonfinite 分项梯度')
    if PROFILE:
        torch.cuda.synchronize(); tq1 = time.time()
    if not torch.isfinite(loss): raise FloatingPointError("Nonfinite WM/Policy loss")
    gn = torch.zeros((), device=dev)
    if not critic_only:
        opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0, error_if_nonfinite=True)
        nn.utils.clip_grad_value_(pi.parameters(), 0.1)
        opt_pi.step()
    if PROFILE:
        torch.cuda.synchronize(); tq2 = time.time()
    if audit_freeze:
        if tuple(p._version for p in wm.parameters())!=wm_versions or any(p.grad is not None for p in wm.parameters()):
            raise RuntimeError('Actor update mutated WM parameters/gradients')
        if decoder_versions is not None and (tuple(p._version for p in pi.dec.parameters())!=decoder_versions
                or any(p.grad is not None for p in pi.dec.parameters())):
            raise RuntimeError('Frozen Decoder changed during Encoder adaptation')
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
    v_loss = torch.zeros(())
    critic_grad_norms = []
    target_mean=torch.zeros((),device=dev); target_max=target_mean
    real_loss_log=[];real_count=0
    if not args.no_critic:
        with torch.no_grad():
            real_X=real_Y=None
            if args.critic_real_batch:
                real=buffer.sample_mc_returns(args.critic_real_batch,args.critic_real_window,args.gamma,args.critic_real_recent_steps)
                real_count=len(real['s'])
                if real_count:
                    real_S=unflatten_state(real['s']);real_c=buffer.commands(real['command_id'])
                    real_g=goals.goal(real_S,real['f'],real['ep_step'],real_c)
                    real_X=_critic_input(real_S,real['f'],normalize_xi(real['xi']),real_g,real['ep_step'],real_c)
                    real_Y=real['mc_return']
            S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
            f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
            g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
            xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
            step_c = torch.cat((s['ep_step'].unsqueeze(0),torch.stack(Steps)),0).flatten()
            command_c = command.repeat(HZ+1) if command is not None else None
            X = critic_input_c(S_c, f_c, xi_c, g_c, step_c, command_c).view(HZ + 1, B, -1)
        critic.norm.update(X)
        if real_X is not None: critic.norm.update(real_X)
        with torch.no_grad():
            V = bound_bootstrap(critic(X),value_upper)
            R = V[-1]
            targets = []
            for t in range(HZ - 1, -1, -1):
                R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                targets.append(R)
            targets = torch.stack(targets[::-1])  # (H, B)
            target_mean=targets.mean(); target_max=targets.max()
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        mb = n // args.critic_minibatches
        for _ in range(args.critic_epochs):
            perm = torch.randperm(n, device=dev)
            real_perm=torch.randperm(real_count,device=dev) if real_count else None
            for i in range(args.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                total_v_loss=v_loss
                if real_count:
                    ri=real_perm[i::args.critic_minibatches]
                    if len(ri):
                        real_v_loss=((critic(real_X[ri])-real_Y[ri])**2).mean()
                        real_loss_log.append(real_v_loss.detach())
                        total_v_loss=total_v_loss+args.critic_real_weight*real_v_loss
                opt_v.zero_grad(set_to_none=True)
                if not torch.isfinite(total_v_loss): raise FloatingPointError("Nonfinite critic loss")
                total_v_loss.backward()
                critic_grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
                if m46 or action_context:
                    critic_grad_norms.append(critic_grad_norm.detach())
                opt_v.step()
    if PROFILE:
        torch.cuda.synchronize()
        print(f"[prof] imagine fwd {1e3*(tq1-tq0):.0f} ms | bwd {1e3*(tq2-tq1):.0f} ms | critic {1e3*(time.time()-tq2):.0f} ms", flush=True)
    diagnostics=dict(**gradient_diagnostics,paired_latent_loss=latent_loss.detach(),critic_real_samples=torch.tensor(float(real_count),device=dev),
        **({'critic_real_mc_loss':torch.stack(real_loss_log).mean()} if real_loss_log else {}),critic_raw_value_mean=raw_v.detach().mean(),critic_raw_value_max=raw_v.detach().max(),
        critic_target_mean=target_mean,critic_target_max=target_max,
        actor_bootstrap_mean=bootstrap.detach().mean(),actor_reward_return_mean=reward_J.detach().mean(),
        imagined_action_clip_fraction=torch.stack(clip_fractions).mean().detach())
    if m46 or action_context:
        diagnostics['critic_gradnorm'] = torch.stack(critic_grad_norms).mean()
    if whole_body:
        diagnostics.update(critic_target_min=targets.min(), critic_target_negative_ratio=(targets < 0.).float().mean())
    if hand_tracking:
        diagnostics['imagined_m410_reward'] = (c_t * baseline.view(HZ, B)).mean().detach()
    if m50:
        for i, name in enumerate(motion.names):
            selected = command.clip_id == i
            if selected.any():
                diagnostics[f'{name}/imagined_reward'] = (r_sum[selected].mean() / HZ).detach()
    return loss.detach(), v_loss.detach(), (r_sum / HZ / reward_scale).mean(), gn.detach(), c.mean().detach(), foot_gn, diagnostics


# ------------------------------------------------------------------------------------------------
if stair_cfg:
    from supermimic.stair_runtime import make_stair_sim
    sim_cfg=make_stair_sim(dev,args)
elif m4:
    from supermimic.integration.m4_task import M4TrackingSim, make_m4_sim
    sim_cfg = make_m4_sim(dev, args)
else:
    sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    if stair_cfg:
        from supermimic.stair_training import environment as stair_environment
        env=stair_environment(sim_cfg,args.num_envs,motion,mdp,stair_cfg,gamma=args.gamma,goals=goals)
        env.stair_iteration=start_it
    elif m4:
        if m50:
            from supermimic.integration.m50_task import ClipTrainingStats, MultiReferenceSim

            sim_type, stats_type = MultiReferenceSim, ClipTrainingStats
            if m51:
                from supermimic.integration.m51_task import StanceFootSim, StanceTrainingStats

                sim_type, stats_type = StanceFootSim, StanceTrainingStats
            if m52:
                from supermimic.integration.m52_task import RootTiltSim, TiltTrainingStats

                sim_type, stats_type = RootTiltSim, TiltTrainingStats
            env = sim_type(sim_cfg, args.num_envs, motion, mdp, gamma=args.gamma, goals=goals,
                           seed=args.seed, iteration=start_it + 1)
            if args.resume:
                env.restore_curriculum(ck['m50_curriculum'])
                env.reset(torch.arange(env.N, device=dev))
            clip_stats = stats_type(motion)
        else:
            env = M4TrackingSim(sim_cfg, args.num_envs, motion, mdp, gamma=args.gamma, goals=goals)
        if args.m4_fault_trace:
            from supermimic.integration.m4_dynamics import TrainingDynamics

            env.dynamics = TrainingDynamics(env, log_path.parent/'dynamics_fault.npz', args)
        if not (upper_ablation or m50) and (not m46 or not args_path.exists()):
            write_m4_config()
        elif m46 and args.resume:
            print(f'M4.6 从第 {start_it} 轮恢复，Git {git_sha}，工作树 {args.git_worktree!r}', flush=True)
    else:
        env = TrackingSim(sim_cfg, args.num_envs, motion, mdp, gamma=args.gamma, goals=goals)
    if args.m34_dataset:
        env.m34_iteration = start_it
        env.m34_straight_iters = args.m34_straight_iters
    buffer = ReplayBuffer(args.buffer_steps,args.num_envs,dev,foot_commands=goals.is_foot,action_context=action_context,
                          static_motion=motion if m50 else None)
    obs_hist = torch.zeros(args.num_envs, Hh, obs_dim, device=dev)

    collection_frames=0; collection_seconds=0.

    @torch.no_grad()
    def collect(n_steps: int, stats: dict):
        global obs_hist, sim_steps, collection_frames, collection_seconds
        collect_started=time.monotonic()
        for _ in range(n_steps):
            if upper_smoke and buffer.size == 5:
                # 在其它世界继续推进时，单独复位首个世界以覆盖异步 Reset。
                env.reset(torch.tensor([0], device=dev))
            if args.m50_smoke and buffer.size in (5, 6, 7):
                # 同一环境依次复位到 F1、F3、F2，其余环境保留自己的时钟。
                index = (0, 2, 1)[buffer.size - 5]
                env.reset(torch.tensor([0], device=dev), motion.clip_start[index:index+1])
            S = env.state()
            o = mdp.add_noise(goals.obs(mdp, S, env.a_prev, env.xi, env.f, env.command))
            fresh = env.ep_step == 0
            obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
            g = goals.goal(S, env.f, env.ep_step, env.command)
            normalizers_online = (critic_warmup_done >= pretrain_iterations
                                  and (not stair_cfg or stair_cfg.get('normalizer_policy') == 'source_online'))
            if normalizers_online:
                pi.obs_norm.update(o)
                pi.goal_norm.update(g)
            if guarded_action:
                if args.m4_fault_trace:
                    a, logits = pi(obs_hist, g, action_noise=args.action_noise, return_logits=True)
                    env.dynamics.set_action(logits, a, current_it + 1, sim_steps)
                else:
                    a = pi(obs_hist, g, action_noise=args.action_noise)
            else:
                a = pi(obs_hist, g)
                a = bound_action(a + args.action_noise * torch.randn_like(a))
            try:
                out = env.step(a)
            except FloatingPointError as exc:
                if guarded_action:
                    exc.add_note(f'训练轮次={current_it + 1}，此前已采集转移={sim_steps}，保留上一个检查点。')
                    if args.m4_fault_trace:
                        message, _ = env.dynamics.report()
                        print(f'Newton | {message}', flush=True)
                raise
            if stair_cfg:
                from supermimic.stair_training import validate_transition
                validate_transition(out,mdp,stair_cfg,log_path)
                replay_reward=mdp.reward(unflatten_state(out['s1']),out['f']+1,out['a'],out['a_prev'])
                error=(out['r']-replay_reward).abs().max()
                if error>1e-5:raise RuntimeError(f'Stair real/replay reward timing mismatch: {float(error)}')
                stats['reward_sync'].append(error)
            if args.goal_mode=='reference_foot' and not torch.equal(out['command'].start_frame+out['ep_step'],out['f']):
                raise RuntimeError('Paired reference frame and immutable command clock diverged')
            if args.reward_term_logging:
                record=dict(sim_steps=sim_steps+args.num_envs,terms={
                    k:dict(mean=float(v.float().mean()),std=float(v.float().std(unbiased=False)))
                    for k,v in out['terms'].items()})
                with (log_path/'reward_terms.jsonl').open('a') as sink: sink.write(json.dumps(record)+'\n')
            if not torch.isfinite(out["s1"]).all(): raise FloatingPointError("Nonfinite Newton state; refusing to poison normalization/replay")
            if guarded_action and not out['r'].isfinite().all():
                raise FloatingPointError('M4 Newton 奖励出现非有限值')
            buffer.add(out)
            if m50:
                clip_stats.record(out)
            if curriculum: curriculum.observe(out['command'],out['ep_step'])
            valid = out.get('valid')
            keep = slice(None) if valid is None else valid
            count = args.num_envs if valid is None else int(valid.count_nonzero())
            sim_steps += count
            if count == 0:
                continue
            S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
            if normalizers_online or (m50 and not m51):
                executed = out['a_exec'] if action_context else _wm_action(out['a'],out['f'])
                wm.in_norm.update(wm.inputs(S0, executed, normalize_xi(out['xi']))[keep])
                wm.out_norm.update(wm.targets(S0, S1)[keep])
            stats["r"].append(out["r"][keep].mean() / reward_scale)
            stats["head_error"].append(out['terms']['head_position_error'][keep].mean())
            stats["head_penalty"].append(out['terms']['head_penalty'][keep].mean() / reward_scale)
            if m4:
                stats['foot_error_sq'].append(out['terms']['foot_rmse'][keep].square().mean())
                stats['root_error_sq'].append(out['terms']['root_position_error'][keep].square().mean())
            if m43:
                frame = out['f'][keep] + 1
                p, R = S1[0][keep], S1[1][keep]
                trace = (R * motion.root_R[frame]).sum((-1, -2))
                stats['root_xy_rmse_m'].append((p[:, :2] - motion.root_p[frame, :2]).square().sum(-1).mean())
                stats['root_orientation_rmse_deg'].append(torch.acos(((trace-1.)*.5).clamp(-1., 1.)).rad2deg().square().mean())
                stats['action_rate_rms'].append((out['a'][keep] - out['a_prev'][keep]).square().mean())
                for name in reward_metric_names:
                    stats[name].append(out['terms'][name][keep].mean())
                if m49 or whole_body:
                    for name in posture_metric_names:
                        value = out['terms'].get(name)
                        if value is not None:
                            stats[name].append(value[keep].mean())
            if goals.is_foot and (args.check_foot_gradients or args.goal_mode=='reference_foot'
                                  or args.m47_smoke or upper_smoke or args.m50_smoke or (m4 and sim_steps == startup_steps + count)):
                command = (buffer.commands(out['command_id'], frames=out['f'], episode_steps=out['ep_step'])
                           if m50 else out['command'])
                restored = goals.goal(S0,out['f'],out['ep_step'],command)
                history = {key:out[key] for key in ('a_prev2','pd_target')} if action_context else {}
                if action_context:
                    target = mdp.executed_target(out['a'],out['startup_q'],out['ep_step'])
                    torch.testing.assert_close(target,out['pd_target'],rtol=0,atol=0)
                    torch.testing.assert_close(mdp.effective_action(target),out['a_exec'],rtol=0,atol=0)
                recomputed = mdp.reward(S1,out['f']+1,out['a'],out['a_prev'],
                                        command=command,ep_step=out['ep_step']+1,**history)
                goal_error = (g-restored)[keep].abs().max()
                reward_error = (out['r']-recomputed)[keep].abs().max()
                if goal_error > 1e-5 or reward_error > 1e-5:
                    raise RuntimeError('Real/replay foot command or reward timing mismatch')
                stats['goal_sync'].append(goal_error)
                stats['reward_sync'].append(reward_error)
            if "ep_len" in out:
                if m4:
                    stats["ep_len"].append(out["ep_len"].sum())
                    stats["ep_ret"].append(out["ep_ret_norm"].sum())
                else:
                    stats["ep_len"].append(out["ep_len"].mean())
                    stats["ep_ret"].append(out["ep_ret_norm"].mean())
                stats["term"].append(out["term"].float().sum())
                stats["done"].append(out["done"].float().sum())

        collection_frames+=n_steps*args.num_envs
        collection_seconds+=time.monotonic()-collect_started

    t_start = time.monotonic() - wall_offset - (time.monotonic()-PROCESS_START)
    def elapsed(): return time.monotonic()-t_start
    current_it = start_it
    startup_steps = sim_steps
    step_acc = float(ck.get('step_acc',0.)) if args.resume else 0.
    if args.resume and 'rng' in ck: restore_rng(ck['rng'])
    last_milestone_saved = ((start_it, critic_warmup_done) if (m46 or upper_ablation or m50)
                            and (log_path / 'last.pt').is_file() else None)

    def save_checkpoint(label=None, *, initial=False):
        global last_milestone_saved
        progress = (current_it, critic_warmup_done)
        if (m44 or m45 or m451 or action_context) and not initial and last_milestone_saved == progress:
            return
        ckpt = {"robot_contract": physics_metadata['robot'] if m4 else checkpoint_contract(), "goal_contract": goal_spec.contract(),
                "checkpoint_schema_version":3 if ROBOT_KIND == 'roban' else 2,
                "task_contract": mdp.contract(),
                "physical_contract":physics_metadata,"policy_contract":policy_contract(
                    Hh,args.obs_adapter,not args.freeze_decoder,obs_dim=obs_dim,observation_mode=args.motion_obs_mode),
                "data_contract":data_contract,"git_sha":git_sha,"transfer_report":transfer_report,
                "source_snapshot":source_snapshot,
                "stage_stats":dict(stage='bridge' if args.goal_mode=='reference_foot' else 'stage1' if args.goal_mode=='motion' else 'stage2',
                    current_stage_real_steps=sim_steps,upstream_real_steps=transfer_report.get('source_total_real_steps',transfer_report.get('source_sim_steps',0)) if transfer_report else 0,
                    upstream_wall_seconds=transfer_report.get('source_total_wall_seconds',0.) if transfer_report else 0.,
                    supervision_steps=supervision_steps),
                "wm":wm.state_dict(), "pi":pi.state_dict(), "critic":critic.state_dict(),
                "opt_wm":opt_wm.state_dict(), "opt_pi":opt_pi.state_dict(), "opt_v":opt_v.state_dict(),
                "iteration":current_it, "wall_time":elapsed(), "sim_steps":sim_steps, "args":vars(args),
                "saved_at":time.strftime('%Y-%m-%d %H:%M:%S'), "rng":rng_state(), "step_acc":step_acc,
                "replay_restore":"fresh episode reset + refill; WM-only recovery warmup",
                "curriculum":curriculum.state_dict() if curriculum else None,
                "imagination_contract":imagination_contract}
        if m411:
            ckpt.update(critic_warmup_iterations=critic_warmup_done,
                        replay_restore='同组从固定场景重新采集 Replay')
        if m50:
            ckpt.update(adaptation_iterations=critic_warmup_done, m50_curriculum=env.curriculum_state(),
                        replay_restore='同一不可变参考库重新 Reset 和采集 Replay')
        finite=all(torch.isfinite(value).all() for model in (wm,pi,critic)
                   for value in model.state_dict().values() if value.is_floating_point())
        if not finite:
            if m4:
                raise FloatingPointError('M4 模型或归一化状态出现非有限值')
            atomic_save(ckpt,log_path/'failure_nonfinite.pt')
            print('Nonfinite model/normalization state saved separately; previous latest preserved',flush=True)
            return
        if m4:
            if initial:
                initial_name = ('initial_m51_c1500.pt' if m52 else 'initial_m50_5500.pt' if m51 else
                                'initial_b1250.pt' if m50 and transfer_report else 'initial.pt')
                atomic_save(ckpt, log_path / initial_name)
                return
            atomic_save(ckpt, log_path/'last.pt')
            if m44 or m45 or m451 or action_context:
                last_milestone_saved = progress
            if (m50 or m411) and label == warmup_label:
                os.link(log_path / 'last.pt', log_path / f'{warmup_label}.pt')
            milestone_interval = args.ckpt_every if m44 or m45 or m451 or action_context else 1000
            if m43 and current_it > 0 and (current_it % milestone_interval == 0 or current_it == args.iterations):
                stage = log_path / milestone_name(current_it)
                if not stage.exists():
                    os.link(log_path/'last.pt', stage)
            if not m43 and current_it == args.m4_early_iteration and not (log_path/'early.pt').exists():
                os.link(log_path/'last.pt', log_path/'early.pt')
            return
        atomic_save(ckpt, log_path/'last.pt')
        if curriculum or args.m34_dataset or stair_cfg:
            # latest is an atomic hard-link alias of the full last.pt.
            temp=log_path/'latest.pt.tmp'
            if temp.exists(): temp.unlink()
            os.link(log_path/'last.pt',temp); os.replace(temp,log_path/'latest.pt')
        if label: atomic_save(ckpt,log_path/(label+'.pt'))
        atomic_save(ckpt,log_path/f'ckpt_{current_it:06d}.pt')
        if curriculum:
            for path in sorted(log_path.glob('ckpt_*.pt'))[:-curriculum.cfg['keep_checkpoints']]: path.unlink()
        if stair_cfg:
            for path in sorted(log_path.glob('ckpt_*.pt'))[:-stair_cfg.get('keep_periodic_checkpoints',8)]:path.unlink()
            if hasattr(env,'reset_validator'):
                (log_path/'reset_statistics.json').write_text(json.dumps(env.reset_validator.stats,indent=2)+'\n')
                (log_path/'reset_geometry_audit.json').write_text(json.dumps(env.reset_geometry_audit,indent=2)+'\n')
        free,total=torch.cuda.mem_get_info()
        (log_path/'runtime_stats.json').write_text(json.dumps(dict(iteration=current_it,gpu=torch.cuda.get_device_name(),
            torch_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
            torch_peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
            device_used_at_checkpoint_MiB=(total-free)/2**20,device_total_MiB=total/2**20,
            wall_seconds=elapsed(),note='Torch peaks exclude Warp; device snapshot includes Warp'),indent=2))

    stats = {"r": [], "head_error": [], "head_penalty": [], "ep_len": [], "ep_ret": [], "term": [], "done": [], 'goal_sync': [], 'reward_sync': []}
    if m4:
        stats.update(foot_error_sq=[], root_error_sq=[])
        if not args.resume:
            # 初始权重须保存于任何采样、归一化和训练更新之前。
            save_checkpoint(initial=True)
        if (upper_ablation or m50) and not args_path.exists():
            write_m4_config()
    if m43:
        from supermimic.roban import ELBOW_JOINT_NAMES, JOINT_NAMES

        reward_metric_names = tuple(f'{name}_contribution' for name in (
            'swing_position', 'swing_clearance', 'swing_orientation', 'stance_position', 'contact_schedule',
            'touchdown', 'upright', 'survival', 'slip', 'joint_limit', 'torque', 'action_rate', 'collision',
            'self_collision', 'body_prior'))
        if m46 or action_context:
            reward_metric_names += ('body_prior', 'foot_objective')
        if action_context:
            reward_metric_names += ('upper_action_smoothness', 'upper_action_smoothness_contribution')
        if m51:
            reward_metric_names += ('m50_reward', *(term+'_contribution' for term in M51_TERMS))
        rms_names = ('root_xy_rmse_m', 'root_orientation_rmse_deg', 'action_rate_rms')
        stats.update({name: [] for name in (*reward_metric_names, *rms_names)})
        wm_metric_names = tuple(f'{name}_{k}' for name in ('wm_error', 'wm_foot_rmse', 'wm_reward_bias') for k in (1, 4, 8))
        grad_metric_names = tuple(f'term_{name}_actor_gradnorm' for name in
                                  ('swing_position', 'swing_clearance', 'touchdown', 'upright', 'survival', 'foot'))
        grad_metric_names += ('actor_bootstrap_gradnorm', 'actor_grad_cosine')
        extended_action_names = ('actor_raw_abs_p95', 'actor_raw_abs_p99', 'joint_speed_max_rad_s',
                                 'joint_speed_p99_rad_s', 'wm_effective_action_rms') + tuple(f'clamp_{name}' for name in JOINT_NAMES)
        if m44:
            extended_action_names += ('raw_target_out_of_range_ratio',) + tuple(
                f'raw_target_{metric}_{name}' for name in JOINT_NAMES
                for metric in ('out_of_range', 'excess_p95', 'excess_p99', 'excess_max'))
        if m45 or m451 or action_context:
            grad_metric_names += ('term_body_prior_actor_gradnorm',)
        if action_context:
            grad_metric_names += ('term_upper_action_smoothness_actor_gradnorm',)
        if m46 or action_context:
            grad_metric_names += ('wm_gradnorm', 'critic_gradnorm')
        m43_columns = (*rms_names, *reward_metric_names, *wm_metric_names, *grad_metric_names,
                       *extended_action_names, 'epa_resets')
        if m49:
            posture_metric_names = ('body_prior_arm', 'body_prior_waist', 'soft_body_prior',
                                    'soft_body_prior_contribution', 'soft_body_prior_arm', 'soft_body_prior_waist',
                                    'soft_body_prior_dead_zone_violation_rad')
            stats.update({name: [] for name in posture_metric_names})
            posture_grad_names = ('term_soft_body_prior_actor_gradnorm',
                                  *(f'grad_cosine_{left}_{right}' for left,right in M49_GRADIENT_PAIRS))
            grad_metric_names += posture_grad_names
            m43_columns += (*posture_metric_names, *posture_grad_names,
                            *(f'raw_{joint}_{q}' for joint in ELBOW_JOINT_NAMES for q in ('p05','p50','p95')),
                            *(f'near_limit_{joint}' for joint in ELBOW_JOINT_NAMES))
        if whole_body:
            posture_metric_names = ('baseline_reward', *(f'{term}_contribution' for term in WHOLE_BODY_TERMS),
                                    *(name for term in HUBER_TERMS for name in tolerance_metric_names(term)))
            stats.update({name: [] for name in posture_metric_names})
            more_audits = ('imagined_baseline_reward', 'critic_target_min', 'critic_target_negative_ratio',
                           *(f'term_{term}_actor_gradnorm' for term in WHOLE_BODY_TERMS),
                           *(f'grad_cosine_{base}_{term}' for base, term in M410_GRADIENT_PAIRS))
            grad_metric_names += more_audits
            m43_columns += (*posture_metric_names, *more_audits, 'imagined_reward',
                            *(f'{name}_{k}' for name in M410_WM_METRICS for k in (1, 4, 8)),
                            *(f'near_limit_{joint}' for joint in ELBOW_JOINT_NAMES + SHOULDER_JOINTS))
        if hand_tracking:
            hand_error_names = tuple(f'hand_{side}{axis}_error_sq' for side in HAND_SIDES
                                     for axis in ('', '_x', '_y', '_z'))
            hand_metrics = ('m410_reward', 'hand_huber_contribution',
                            *(f'hand_{side}_outside_ratio' for side in HAND_SIDES))
            posture_metric_names += hand_metrics + hand_error_names
            stats.update({name: [] for name in (*hand_metrics, *hand_error_names)})
            hand_audits = ('imagined_m410_reward', 'actor_reward_gradnorm',
                           *(f'term_{term}_actor_gradnorm' for term in ('root', 'joint_huber', 'hand_huber')),
                           *(f'grad_cosine_{left}_{right}' for left,right in M411_GRADIENT_PAIRS))
            grad_metric_names += hand_audits
            m43_columns += (*hand_metrics, *(name.replace('_error_sq', '_rmse_m') for name in hand_error_names),
                            *hand_audits, 'stage', 'actor_lr_effective', 'real_imagined_reward_gap')
        if upper_ablation or m50:
            m43_columns += ('total_reward',)
        if m50:
            clip_names = motion.names
            clip_audit_columns = tuple(f'{name}/{key}' for name in clip_names for key in ('imagined_reward', 'actor_gradnorm'))
            clip_audit_columns += tuple(f'clip_grad_cosine_{name}_{other}' for i, name in enumerate(clip_names)
                                        for other in clip_names[i+1:])
            grad_metric_names += clip_audit_columns
            clip_wm_columns = tuple(f'{name}/wm_{metric}_{span}' for name in clip_names
                                    for metric in ('error', 'foot_rmse') for span in (1, 4, 8))
            m43_columns += (clip_stats.columns + clip_audit_columns + clip_wm_columns
                            + tuple(f'{name}/real_imagined_gap' for name in clip_names)
                            + ('reset_probability', 'reset_scale', 'wm_lr_effective', 'critic_lr_effective'))
        if m51:
            contact_audits = tuple(f'term_{term}_actor_gradnorm' for term in M51_LOG_TERMS)
            contact_audits += tuple(f'grad_cosine_{left}_{right}' for left, right in M51_GRAD_PAIRS)
            grad_metric_names += contact_audits
            m43_columns += clip_stats.contact_columns + contact_audits
        if m52:
            tilt_audits = ('term_root_tilt_actor_gradnorm',
                           *(f'grad_cosine_{left}_{right}' for left, right in M52_GRAD_PAIRS))
            grad_metric_names += tilt_audits
            m43_columns += clip_stats.root_columns + tilt_audits
    # Replay is deliberately not checkpointed. Refill it on resume as well as on
    # the first run; otherwise sampling the fresh empty buffer fails immediately.
    # Fresh replay after resume: refill, then a short WM-only recovery before Actor updates.
    recovery_until = start_it + (args.recovery_wm_iters if args.resume else 0)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = metrics_path.open("a")
    action_metrics = ('actor_raw_mean', 'actor_raw_std', 'actor_raw_abs_max', 'pd_target_clamp_ratio',
                      'torque_saturation_ratio', 'joint_limit_excess_rad', 'invalid_transitions', 'physics_exceptions')
    if log_f.tell() == 0:
        columns = "iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr,head_rms_deg,head_penalty_norm,foot_actor_gradnorm,goal_sync_max_error,reward_sync_max_error,completed_episodes"
        if m4:
            columns += ",num_envs,collect_fps,train_fps,device_used_mib,torch_peak_mib,survival_rate,foot_rmse_m,root_rmse_m"
        if standing_action:
            columns += ',' + ','.join(action_metrics) + ',imagined_pd_target_clamp_ratio'
        if m43:
            columns += ',' + ','.join(m43_columns)
        log_f.write(columns+'\n')
    t_log = time.time()
    last_log_iteration = start_it if critic_warmup_done >= pretrain_iterations else critic_warmup_done
    last_log_steps = sim_steps
    last_collection_seconds = 0.
    agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": [], 'fgn': []}
    diag_agg={}
    try:
        collect(args.warmup_steps, stats)
        if upper_smoke:
            ages = buffer.data['ep_step'][:buffer.size]
            starts = ages == 0
            torch.testing.assert_close(ages, buffer.data['f'][:buffer.size].to(ages.dtype), rtol=0, atol=0)
            if not (ages[5, 0] == 0 and (ages[5, 1:] == 5).all()
                    and all((ages == tick).any() for tick in (0, 5, 10, 11))):
                raise RuntimeError(f'{phase} Smoke 未覆盖异步 Reset 或启动过渡的完整时段')
            q_reset = unflatten_state(buffer.data['s'][:buffer.size][starts])[4]
            data = {name: buffer.data[name][:buffer.size] for name in ('a', 'a_prev', 'a_prev2', 'startup_q')}
            torch.testing.assert_close(data['startup_q'][starts], q_reset.clamp(mdp.kin.q_lo, mdp.kin.q_hi), rtol=0, atol=0)
            for lag, key in ((1, 'a_prev'), (2, 'a_prev2')):
                valid = ages[lag:] >= lag
                torch.testing.assert_close(data[key][lag:][valid], data['a'][:-lag][valid], rtol=0, atol=0)
                missing = data[key][ages < lag]
                torch.testing.assert_close(missing, torch.zeros_like(missing), rtol=0, atol=0)
            print(f'{phase} Smoke 覆盖步龄 0/5/10/11，异步 Reset 和 Replay 动作历史通过', flush=True)
        if args.m50_smoke:
            frames = buffer.data['f'][5:8, 0]
            ages = buffer.data['ep_step'][5:8, 0]
            ids = buffer.data['command_id'][5:8, 0]
            restored = buffer.commands(ids, frames=frames, episode_steps=ages)
            torch.testing.assert_close(restored.clip_id, torch.tensor([0, 2, 1], device=dev), atol=0, rtol=0)
            torch.testing.assert_close(buffer.data['ep_step'][5:8, 1],
                                       torch.arange(5, 8, device=dev, dtype=ages.dtype), atol=0, rtol=0)
            if ids.unique().numel() != 3 or buffer.command_store:
                raise RuntimeError('M5.0 Replay 的异步 Episode 身份不正确')
            print('M5.0 Smoke 通过 F1→F3→F2 异步复位、旧回放和其余环境时钟核对', flush=True)
        for update in range(start_it + critic_warmup_done, args.iterations + pretrain_iterations):
            critic_only = update < pretrain_iterations
            it = update if critic_only else update - pretrain_iterations
            if stair_cfg:env.stair_iteration=it
            if args.m34_dataset: env.m34_iteration = it
            reserve=stair_cfg.get('final_eval_reserve_seconds',0) if stair_cfg and args.m34_eval_every else 0
            budget_time = time.monotonic()-PROCESS_START if m4 else elapsed()
            if STOP_REQUESTED or (args.wall_seconds is not None and budget_time>=args.wall_seconds-reserve) or (curriculum and elapsed() >= curriculum.cfg['budget_seconds']-45):
                print('Safe stop requested / wall-clock budget reserve reached',flush=True); break
            lr = lr_at(update) if m50 else args.lr if critic_only else lr_at(it)
            for o,scale in ((opt_wm,1.),(opt_pi,actor_lr/args.lr),(opt_v,critic_lr/args.lr)):
                for gparam in o.param_groups:
                    gparam["lr"] = actor_lr if hand_tracking and o is opt_pi else lr*scale
            step_acc += 1.0 / args.utd
            n = int(step_acc)
            step_acc -= n
            prof = PROFILE and it % 20 == 0 and it > args.wm_only_iters + 5
            if prof:
                torch.cuda.synchronize(); tp0 = time.time()
            if m50:
                env.set_iteration(0 if critic_only else it + 1)
            m52_update_batch = None
            collect(n, stats)
            if prof:
                torch.cuda.synchronize(); tp1 = time.time()
            if not critic_only or (m50 and not m51):
                for _ in range(args.wm_batches):
                    l, e8, wm_grad_norm = wm_update()
                    if m46 or action_context:
                        diag_agg.setdefault('wm_gradnorm', []).append(wm_grad_norm)
                agg["wm"].append(l), agg["wm8"].append(e8)
            if prof:
                torch.cuda.synchronize(); tp2 = time.time()
            warmup_complete = critic_only and it + 1 == pretrain_iterations
            if critic_only or it >= max(args.wm_only_iters,recovery_until):
                audit_gradients = warmup_complete or (args.gradient_audit_every > 0 and (it + 1) % args.gradient_audit_every == 0)
                pl, vl, ir, gn, surv, fgn, diagnostics = policy_update(audit_gradients, critic_only=critic_only)
                for key,value in diagnostics.items(): diag_agg.setdefault(key,[]).append(value)
                if prof:
                    torch.cuda.synchronize(); tp3 = time.time()
                    print(f"[prof] collect {1e3*(tp1-tp0):.0f} ms ({n} steps) | wm {1e3*(tp2-tp1):.0f} ms | policy+critic {1e3*(tp3-tp2):.0f} ms", flush=True)
                agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
                if args.check_foot_gradients or audit_gradients or args.roban_gradient_audit: agg['fgn'].append(fgn)
            first_joint = not critic_only and it == start_it
            if ((it + 1) % args.log_every == 0 or (stair_cfg and it == start_it)
                    or (m4 and (update == start_it + critic_warmup_done and it == 0
                               or first_joint or warmup_complete or it + 1 == args.iterations))):
                mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
                interval = time.time() - t_log
                ips = ((it+1-last_log_iteration) if stair_cfg or m4 else args.log_every) / interval
                last_log_iteration=it+1
                t_log = time.time()
                wall_h = elapsed() / 3600
                term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
                head_rms_deg = math.sqrt(mean(stats['head_error']) / 2) * 180 / math.pi
                row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
                       mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), mean(stats["ep_len"]), mean(stats["ep_ret"]), term_frac, ips, lr,
                       head_rms_deg, mean(stats['head_penalty']), mean(agg['fgn']),
                       max((float(x) for x in stats['goal_sync']),default=0.),
                       max((float(x) for x in stats['reward_sync']),default=0.),
                       sum(float(x) for x in stats['done'])]  # fmt: skip
                if m4:
                    completed = row[21]
                    if completed:
                        row[11] = sum(float(x) for x in stats['ep_len']) / completed
                        row[12] = sum(float(x) for x in stats['ep_ret']) / completed
                    free, total = torch.cuda.mem_get_info(dev)
                    device_mib = (total-free)/2**20
                    nframes = sim_steps-last_log_steps
                    collection_dt = collection_seconds-last_collection_seconds
                    row.extend((args.num_envs, nframes/max(collection_dt, 1e-9), nframes/interval, device_mib,
                                torch.cuda.max_memory_allocated(dev)/2**20,
                                1-term_frac if completed else '',
                                math.sqrt(mean(stats['foot_error_sq'])), math.sqrt(mean(stats['root_error_sq']))))
                    last_log_steps, last_collection_seconds = sim_steps, collection_seconds
                    if args.max_device_memory_mib is not None and device_mib >= args.max_device_memory_mib:
                        STOP_REQUESTED = True
                        print(f'M4 显存达到阈值（{device_mib:.0f} MiB），保存后停止。', flush=True)
                dynamics_message, dynamics_metrics = env.dynamics.report() if args.m4_fault_trace else ('', {})
                if standing_action:
                    clips = diag_agg.get('imagined_action_clip_fraction', [])
                    row.extend(dynamics_metrics.get(key, '') for key in action_metrics)
                    row.append(mean(clips) if clips else '')
                if m43:
                    extra = {key: mean(stats[key]) for key in reward_metric_names}
                    if m49 or whole_body:
                        extra.update({key: mean(stats[key]) for key in posture_metric_names})
                    extra.update({key: math.sqrt(mean(stats[key])) for key in rms_names})
                    extra.update(m43_wm_metrics(), epa_resets=env.epa_resets)
                    extra.update(dynamics_metrics)
                    extra.update({key: mean(diag_agg[key]) for key in grad_metric_names if key in diag_agg})
                    if upper_ablation or m50:
                        extra['total_reward'] = row[10] * reward_scale
                    if whole_body:
                        extra['imagined_reward'] = row[7] * reward_scale
                    if hand_tracking:
                        extra.update({name.replace('_error_sq', '_rmse_m'): math.sqrt(mean(stats[name]))
                                      for name in hand_error_names})
                        extra.update(stage=warmup_label.replace('after_adaptation', 'adaptation') if critic_only else 'joint',
                                     actor_lr_effective=0. if critic_only else opt_pi.param_groups[0]['lr'],
                                     real_imagined_reward_gap=extra['total_reward'] - extra['imagined_reward'])
                    if m50:
                        extra.update(clip_stats.read(buffer, env))
                        extra.update(reset_probability=env.perturb_probability, reset_scale=env.perturb_scale,
                                     wm_lr_effective=opt_wm.param_groups[0]['lr'],
                                     critic_lr_effective=opt_v.param_groups[0]['lr'])
                        for name in motion.names:
                            if f'{name}/imagined_reward' in extra:
                                extra[f'{name}/real_imagined_gap'] = extra[f'{name}/reward'] - extra[f'{name}/imagined_reward']
                    row.extend(extra.get(key, '') for key in m43_columns)
                log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
                log_f.flush()
                if diag_agg:
                    if not m4:
                        with (log_path/'gradient_terms.jsonl').open('a') as sink:
                            sink.write(json.dumps(dict(iteration=it+1,**{k:mean(v) for k,v in diag_agg.items()}))+'\n')
                    if not curriculum:diag_agg={}
                resets = f' | joint_resets {env.joint_resets}' if guarded_action else ''
                if standing_action:
                    resets += f' | epa_resets {env.epa_resets}'
                if hand_tracking:
                    stage_label = (('WM/Critic adaptation | ' if m50 and not m51 else 'Critic warm-up | ')
                                   if critic_only else 'Joint | ')
                else:
                    stage_label = ''
                print(f"{stage_label}it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} "
                      f"| sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | head {head_rms_deg:.1f}deg | actor_grad {row[8]:.3g} foot_grad {row[18]:.3g} | {ips:.2f} it/s{resets}", flush=True)  # fmt: skip
                if dynamics_message:
                    print(f'Newton | {dynamics_message}', flush=True)
                if standing_action and clips:
                    print(f'BPTT | PD 目标裁剪 {mean(clips):.2%}', flush=True)
                stats = {k: [] for k in stats}
                agg = {k: [] for k in agg}
            if critic_only:
                critic_warmup_done = it + 1
                if warmup_complete:
                    save_checkpoint(warmup_label)
                    last_log_iteration = 0
                    if m50 and not m51:
                        print(f'M5.0 WM/Critic 适应完成 | {critic_warmup_done} 轮', flush=True)
                        if args.m4_eval_every and not STOP_REQUESTED:
                            evaluate_m4_checkpoint(log_path / 'after_adaptation.pt')
                    elif m52:
                        print(f'M5.2-{args.m52_group.upper()} Critic 预热完成 | {critic_warmup_done} 轮', flush=True)
                    elif m51:
                        print(f'M5.1-{args.m51_group.upper()} Critic 预热完成 | {critic_warmup_done} 轮', flush=True)
                    else:
                        print(f'M4.11 Critic warm-up complete | {critic_warmup_done} iterations', flush=True)
                continue
            current_it = it+1
            if stair_cfg and (current_it % args.log_every==0 or current_it==start_it+1):
                free,total=torch.cuda.mem_get_info()
                if args.max_device_memory_mib is not None and (total-free)/2**20>=args.max_device_memory_mib:
                    STOP_REQUESTED=True
                    print('Configured device memory limit reached; saving before safe stop',flush=True)
                (log_path/'training_status.json').write_text(json.dumps(dict(status='running',
                    iteration=current_it,wall_seconds=elapsed(),budget_seconds=args.wall_seconds)))
            if curriculum:
                if current_it % args.log_every == 0:
                    gpu_stats={}
                    try:
                        gpu_sample=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2,check=True)
                        utilization,memory=gpu_sample.stdout.splitlines()[0].split(',')
                        gpu_stats={'Performance/GPUUtilization':float(utilization),'Performance/DeviceUsedMiB':float(memory)}
                        memory_limit=getattr(args,'max_device_memory_mib',None)
                        if memory_limit is not None and float(memory)>=memory_limit:
                            STOP_REQUESTED=True
                            print(f'Device memory {memory} MiB reached configured {memory_limit} MiB; safe save/stop requested',flush=True)
                    except (OSError,ValueError,subprocess.SubprocessError) as exc:
                        print(f'GPU telemetry unavailable: {exc}',flush=True)
                    status_temp=log_path/'training_status.json.tmp'
                    status_temp.write_text(json.dumps(dict(status='running',iteration=current_it,wall_seconds=elapsed(),
                        stage=curriculum.stage,budget_seconds=curriculum.cfg['budget_seconds'])))
                    os.replace(status_temp,log_path/'training_status.json')
                    emit(log_path,current_it,curriculum.scalars() | gpu_stats | {
                        **{f'Optimization/{k}':float(sum(v)/len(v)) for k,v in diag_agg.items()},
                        'Optimization/ActorLR':opt_pi.param_groups[0]['lr'],
                        'Optimization/CriticLR':opt_v.param_groups[0]['lr'],
                        'Performance/PeakTorchMiB':torch.cuda.max_memory_allocated()/2**20,
                        'Performance/CollectionFPS':collection_frames/max(collection_seconds,1e-9)})
                    diag_agg={}
                remaining=curriculum.cfg['budget_seconds']-elapsed()
                final_due=120 < remaining < 300 and not getattr(curriculum,'final_evaluated',False)
                due=elapsed()-curriculum.last_eval>=curriculum.cfg['eval_seconds'] or final_due
                available=elapsed()*curriculum.cfg['eval_fraction']-curriculum.eval_spent
                pending_full=curriculum.passes>=curriculum.cfg['consecutive_passes']-1
                full_reserve=[45,90,150,210,270,270][curriculum.stage]
                full=(pending_full and remaining>600 and available>=full_reserve) or (final_due and available>=120)
                evaluation_minimum=getattr(curriculum,'eval_min_seconds',10.)
                allowance=min(300. if full else max(90.,evaluation_minimum),available,max(0.,remaining-60))
                if pending_full and not full: due=False  # accumulate the 5% quota for a complete stage gate
                if due and current_it % args.log_every == 0 and allowance>=evaluation_minimum and remaining>120 and not STOP_REQUESTED:
                    # Full evaluation only after a successful fast gate, at most on stage transitions.
                    report=evaluate(env,pi,goals,mdp,curriculum,Hh,full,
                                    deadline=time.monotonic()+allowance,rich=final_due,critic=critic,wm=wm)
                    obs_hist.zero_()
                    if final_due and full: curriculum.final_evaluated=True
                    curriculum.eval_spent+=report['wall_seconds']; curriculum.last_eval=elapsed()
                    stage_before=curriculum.stage
                    complete=not any(r['status'] in ('budget_skipped','incomplete') or r.get('incomplete_rate',0)>0 for r in report['rows'])
                    curriculum.eval_min_seconds=report['minimum_next_eval_seconds']
                    changed=curriculum.update(report['metrics'],elapsed(),allow_promotion=full,bucket_metrics=report['bucket_metrics'],wm_metrics=report.get('wm_metrics')) if complete else False
                    (log_path/f'eval_{current_it:06d}.json').write_text(json.dumps(report,indent=2))
                    scalars={f'Evaluation/{task}/{key}':value for task,m in report['metrics'].items() for key,value in m.items()}
                    for i,row in enumerate(report['rows']):
                        if row['status']=='evaluated':
                            scalars.update({f'Evaluation/Command{i}/{k}':v for k,v in row.items() if isinstance(v,(int,float))})
                    emit(log_path,current_it,scalars | curriculum.scalars())
                    for name,tasks in [('standing',['standing']),('walking',['short','medium','long']),
                                       ('overall',[t for t,w in zip(TASKS,
                                                                 curriculum.cfg['weights'][stage_before]) if w>0])]:
                        if not complete or not all(t in report['metrics'] for t in tasks): continue
                        # Score is actual success first, then tracking error. No WM-loss selection.
                        score=min(report['metrics'][t]['success_rate'] for t in tasks)-.1*max(report['metrics'][t]['foot_rmse'] for t in tasks)
                        if name=='overall':
                            if report['metrics'].get('standing',{}).get('success_rate',0)<curriculum.cfg['standing_threshold']: continue
                            score += stage_before
                        if score>curriculum.best.get(name,-float('inf')):
                            curriculum.best[name]=score; save_checkpoint('best_'+name)
                    if curriculum.failures>=3:
                        print(f'Curriculum C{curriculum.stage} stalled for {curriculum.failures} complete evaluations; '
                              'difficulty reduced; no forced promotion',flush=True)
                    if changed:
                        save_checkpoint(f'stage_C{stage_before}_end')
                        print(f'Curriculum transition C{stage_before} -> C{curriculum.stage}',flush=True)
                if elapsed()-curriculum.last_save>=curriculum.cfg['checkpoint_seconds']:
                    curriculum.last_save=elapsed(); save_checkpoint()
            evaluate_m4 = args.m4_eval_every and not STOP_REQUESTED and (
                current_it % args.m4_eval_every == 0 or current_it == args.iterations)
            if curriculum and current_it == args.wm_only_iters:
                save_checkpoint('wm_calibrated')
                print(f'WM-only phase complete at iteration {current_it}; saved wm_calibrated.pt; Actor/Critic updates begin next iteration',flush=True)
            elif (current_it % args.ckpt_every == 0 or current_it == args.iterations or evaluate_m4
                  or (stair_cfg and current_it==start_it+1) or (m4 and current_it == args.m4_early_iteration)):
                save_checkpoint()
            if evaluate_m4:
                # 使用冻结检查点独立评估，训练环境和 Replay 时钟留在当前进程。
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                eval_started = time.time()
                evaluate_m4_checkpoint()
                t_log += time.time() - eval_started
            if args.m34_dataset and args.m34_eval_every and current_it % args.m34_eval_every == 0:
                # No restart/refill: replay, optimizer, RNG and simulator state
                # remain in this process while separate Newton children evaluate.
                evaluation_dir = log_path/'evals'/f'iter_{current_it:06d}'
                evaluation_dir.mkdir(parents=True, exist_ok=True)
                frozen_checkpoint = evaluation_dir/'checkpoint.pt'
                if not frozen_checkpoint.exists():
                    save_checkpoint()
                    # A hard link preserves this exact serialization even when
                    # last.pt / ckpt_*.pt are atomically replaced on final save.
                    os.link(log_path/'last.pt', frozen_checkpoint)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                from supermimic.m34_evaluation import evaluate_checkpoint
                print(f'Iteration {current_it}: paused training for independent frozen Newton evaluation', flush=True)
                evaluation_started = time.time()
                evaluate_checkpoint(frozen_checkpoint, args.m34_dataset, evaluation_dir, args.m34_eval_seeds)
                print(f'Iteration {current_it}: evaluation complete; continuing existing replay/optimizers', flush=True)
                # Exclude evaluation duration from the next throughput window.
                t_log += time.time()-evaluation_started
            if stair_cfg and args.m34_eval_every and current_it % args.m34_eval_every == 0:
                evaluation_dir=log_path/'evals'/f'iter_{current_it:06d}'
                evaluation_dir.mkdir(parents=True,exist_ok=True)
                frozen_checkpoint=evaluation_dir/'checkpoint.pt'
                if not frozen_checkpoint.exists():
                    save_checkpoint()
                    os.link(log_path/'last.pt',frozen_checkpoint)
                torch.cuda.synchronize();torch.cuda.empty_cache()
                from supermimic.stair_evaluation import evaluate_checkpoint as evaluate_stairs
                print(f'Iteration {current_it}: independent frozen stair/flat Newton evaluation',flush=True)
                evaluation_started=time.time()
                evaluation=evaluate_stairs(frozen_checkpoint,stair_cfg,evaluation_dir,args.m34_eval_seeds,
                    timeout_seconds=min(stair_cfg.get('eval_timeout_seconds',300),
                        max(.1,args.wall_seconds-elapsed())) if args.wall_seconds else None)
                # Select solely on TRAIN references. The designated four-step
                # stair is a training action; no held-out flat test is opened.
                if evaluation['flat_retention_passed']:
                    score=evaluation['stair_score']
                    best_path=log_path/'best_score.json'
                    previous=json.loads(best_path.read_text()) if best_path.exists() else {}
                    if score>previous.get('score',-float('inf')):
                        temp=log_path/'best.pt.tmp'
                        if temp.exists():temp.unlink()
                        os.link(frozen_checkpoint,temp);os.replace(temp,log_path/'best.pt')
                        best_path.write_text(json.dumps(dict(iteration=current_it,score=score,
                            selection='stair train return; flat train retention >= 0.85',
                            checkpoint_sha256=evaluation['checkpoint_sha256']),indent=2)+'\n')
                t_log+=time.time()-evaluation_started
        if curriculum and not STOP_REQUESTED:
            remaining=curriculum.cfg['budget_seconds']-elapsed()
            allowance=min(180.,elapsed()*curriculum.cfg['eval_fraction']-curriculum.eval_spent)
            if remaining>allowance+60 and allowance>=35:
                final_report=evaluate(env,pi,goals,mdp,curriculum,Hh,full=True,
                                      deadline=time.monotonic()+allowance,critic=critic,wm=wm)
                curriculum.eval_spent+=final_report['wall_seconds']
                (log_path/'eval_final.json').write_text(json.dumps(final_report,indent=2))
        if (m46 or upper_ablation or m50) and args.m4_eval_every and not STOP_REQUESTED and current_it == args.iterations:
            assert_periodic_final()
    except (FloatingPointError, RuntimeError) as error:
        if m52 and m52_update_batch and 'nonfinite' in str(error).lower().replace('-', ''):
            snapshot = dict(m52_update_batch, reason=str(error), sim_steps=sim_steps, iteration=current_it,
                            critic_warmup_iterations=critic_warmup_done, rng=rng_state())
            try:
                atomic_save(snapshot, log_path/'failure_update.pt')
            except OSError as snapshot_error:
                error.add_note(f'保存更新诊断失败 {snapshot_error}')
        raise
    finally:
        # Signals set a flag; exceptions also leave the last complete optimizer state.
        if not m4 or sys.exc_info()[0] is None:
            save_checkpoint()
        log_f.close()
        final_status='error' if sys.exc_info()[0] is not None else 'stopped' if STOP_REQUESTED or current_it<args.iterations else 'completed'
        if m4:
            print(f'M4 {final_status} | iteration={current_it} | sim_steps={sim_steps} | wall={elapsed():.1f}s',flush=True)
        else:
            (log_path/'training_status.json').write_text(json.dumps(dict(status=final_status,iteration=current_it,wall_seconds=elapsed(),
                stop_requested=STOP_REQUESTED,budget_seconds=curriculum.cfg['budget_seconds'] if curriculum else args.wall_seconds)))
    if (m46 or upper_ablation or m50) and final_status == 'stopped':
        sys.exit(75)

# The parent simulator is closed before the final independent physics rollout.
# Smoke has eval_every=0: its checkpoint is evaluated by the explicit gate command.
if stair_cfg and args.m34_eval_every:
    periodic=log_path/'evals'/f'iter_{current_it:06d}'/'summary.json'
    if not periodic.exists():
        torch.cuda.empty_cache()
        final_dir=log_path/'evals'/f'final_{current_it:06d}'
        final_dir.mkdir(parents=True,exist_ok=True)
        frozen=final_dir/'checkpoint.pt'
        if not frozen.exists():os.link(log_path/'last.pt',frozen)
        from supermimic.stair_evaluation import evaluate_checkpoint as evaluate_stairs
        evaluate_stairs(frozen,stair_cfg,final_dir,args.m34_eval_seeds,
            timeout_seconds=min(stair_cfg.get('eval_timeout_seconds',300),
                max(.1,args.wall_seconds-elapsed())) if args.wall_seconds else None)
