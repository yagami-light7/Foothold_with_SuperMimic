"""验证 M4.6 配置及限定两轮的真实 Newton 冒烟测试。"""

import argparse
from copy import deepcopy
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / 'configs' / 'roban'
GROUPS = {'w002': .02, 'w005': .05}
MODELS = ('pi', 'wm', 'critic')
OPTIMIZERS = ('opt_pi', 'opt_wm', 'opt_v')
SOURCES = ('resume', 'init_weights', 'init_from_flat', 'init_from_motion', 'init_from_bridge', 'm451_source')
RUNTIME_PATHS = {'roban_config', 'log_dir', 'm4_reference'}
SMOKE_OVERRIDES = dict(num_envs=8, iterations=2, warmup_steps=16, wm_only_iters=1, recovery_wm_iters=0,
                       m4_early_iteration=1, ckpt_every=1, m4_eval_every=0, log_every=1, gradient_audit_every=1)
CONFIG_PREFIX = 'M4.6 effective config '


def require(valid, message):
    if not valid:
        raise ValueError(message)


def check_files():
    baseline = json.loads((CONFIGS / 'm4_3.json').read_text())
    require(baseline.get('body_prior_weight', 0.) == 0., 'M4.3 的上身先验权重必须为零')
    for group, weight in GROUPS.items():
        path = CONFIGS / f'm4_6_{group}.json'
        candidate = json.loads(path.read_text())
        require(candidate == dict(baseline, body_prior_weight=weight), f'{path.name} 存在额外配置差异')
    print('配置文件通过：两组仅改变上身先验权重。', flush=True)
    return baseline


def trainer(group, checkpoint_dir, reference, mode):
    cmd = [sys.executable, 'supermimic/train.py', '--roban_config', f'configs/roban/m4_6_{group}.json',
           '--m4_reference', str(reference), '--device', 'cuda:0', '--log_dir', str(checkpoint_dir),
           '--m46_group', group, '--fresh_start', mode]
    return run_checked(cmd, f'{group} {mode}')


def run_checked(cmd, context):
    result = subprocess.run(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f'{context} 退出码为 {result.returncode}\n{result.stdout}')
    return result.stdout


def check_effective(baseline, reference):
    effective = {}
    for group, weight in GROUPS.items():
        output = trainer(group, f'outputs/m4_6/{group}/checkpoints', reference, '--m46_check')
        lines = [line.removeprefix(CONFIG_PREFIX) for line in output.splitlines() if line.startswith(CONFIG_PREFIX)]
        require(len(lines) == 1, f'{group} 未返回训练入口解析结果\n{output}')
        cfg = json.loads(lines[0])
        expected = dict(baseline, body_prior_weight=weight, iterations=5000, seed=0, wm_only_iters=1000,
                        m4_unclipped_pd_target=False, fresh_start=True)
        mismatch = sorted(k for k in expected if cfg.get(k) != expected[k])
        require(not mismatch, f'{group} 运行时配置与基线不同：{mismatch}')
        require(not any(cfg.get(key) for key in SOURCES), f'{group} 试图加载已有权重')
        require(Path(cfg['m4_reference']).resolve() == reference, f'{group} 运行时 F2 路径不一致')
        effective[group] = cfg
    a, b = effective.values()
    exclusive = RUNTIME_PATHS | {'m46_group', 'body_prior_weight'}
    mismatch = sorted(k for k in (a.keys() | b.keys()) - exclusive if a.get(k) != b.get(k))
    require(not mismatch, f'两组运行时参数出现额外差异：{mismatch}')
    print('真实训练入口解析通过：正式配置为 5000 轮，Hard Clamp 和 Fresh Start 已启用。', flush=True)
    return effective


def read_checkpoint(path):
    import torch

    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def verify_fresh(checkpoint, group):
    args = checkpoint['args']
    require(args['fresh_start'] and not any(args.get(key) for key in SOURCES), f'{group} 初始化来源不空')
    require(checkpoint['iteration'] == checkpoint['sim_steps'] == 0 and checkpoint['transfer_report'] is None,
            f'{group} Initial 已包含训练或迁移进度')
    require(all(not checkpoint[key]['state'] for key in OPTIMIZERS), f'{group} Initial 已包含优化器历史')
    sensors = checkpoint['data_contract']
    require(sensors['frames'] == 261 and sensors['duration_s'] == 5.2 and sensors['files'][0]['source_fps'] == 50,
            f'{group} F2 参考契约不对')
    physical = checkpoint['physical_contract']
    require(physical['control_dt'] == .02 and physical['solver']['num_substeps'] == 8,
            f'{group} 物理子步契约不对')
    policy = checkpoint['policy_contract']
    require((policy['hist'], policy['obs_dim'], policy['action_dim'], checkpoint['goal_contract']['dim']) == (10, 69, 21, 320)
            and checkpoint['critic']['norm.mean'].numel() == 970, f'{group} Actor/Critic 输入输出维度不对')
    require(not args.get('m4_unclipped_pd_target') and
            checkpoint['task_contract']['wm_action'] == 'standing_equivalent_of_limit_clamped_pd_target',
            f'{group} 未应用 Hard Clamp')
    for name in MODELS:
        norm_counts = [value for key, value in checkpoint[name].items() if key.endswith('_norm.count') or key == 'norm.count']
        require(norm_counts and all(value.item() == 0. for value in norm_counts), f'{group} {name} 归一化已被更新')


def check_smoke_group(group, cfg, run, output):
    import torch

    require('M4 completed | iteration=2 |' in output, f'{group} 未完成两轮\n{output}')
    require(not any(flag in output for flag in ('CUDA error', 'Warning: EPA horizon', 'nefc overflow')),
            f'{group} 捕获到严重求解器告警\n{output}')
    folder = run / 'checkpoints'
    initial, final = (read_checkpoint(folder / file) for file in ('initial.pt', 'last.pt'))
    verify_fresh(initial, group)
    require(all((folder / f'iter_{iteration}.pt').is_file() for iteration in (1, 2)), f'{group} 未写入周期检查点')
    require(final['iteration'] == SMOKE_OVERRIDES['iterations'] and final['sim_steps'] > 0 and final.get('rng'),
            f'{group} 最终检查点缺少进度或随机数状态')

    stored = initial['args']
    # 训练入口解析通过后还会展开 F2 文件和启动器默认值，以实际检查点与基线对照这些字段。
    expected = set(cfg) - RUNTIME_PATHS - {'m46_check', 'm46_smoke', 'motion_dir', 'clips', 'foot_config', 'livestream'}
    mismatch = sorted(key for key in expected if stored.get(key) != SMOKE_OVERRIDES.get(key, cfg[key]))
    require(not mismatch, f'{group} 实际运行的冒烟参数出现额外变化：{mismatch}')
    require(initial['task_contract']['config']['body_prior_weight'] == GROUPS[group], f'{group} 上身奖励未启用')

    for model, optimizer in zip(MODELS, OPTIMIZERS):
        state = final[optimizer]['state']
        require(state and all(float(item['step']) > 0 for item in state.values()), f'{group} {model} 优化器未更新')
        tensors = final[model]
        require(all(torch.isfinite(value).all() for value in tensors.values()), f'{group} {model} 有非有限参数或归一化')
        require(any(not torch.equal(initial[model][key], value) for key, value in tensors.items()
                    if key.endswith(('.weight', '.bias'))), f'{group} {model} 优化未改变模型参数')

    with (run / 'logs' / 'train.csv').open() as csv_file:
        metrics = list(csv.DictReader(csv_file))
    require([int(row['iteration']) for row in metrics] == [1, 2], f'{group} 训练指标缺少两轮记录')
    fields = ('wm_loss', 'wm_gradnorm', 'pi_loss', 'v_loss', 'sim_rew', 'imag_rew', 'body_prior_contribution',
              'pi_gradnorm', 'critic_gradnorm', 'term_body_prior_actor_gradnorm')
    values = {field: float(metrics[-1][field]) for field in fields}
    require(all(math.isfinite(value) for value in values.values()), f'{group} 奖励、损失或梯度非有限')
    require(all(values[field] > 0. for field in fields if field.endswith('gradnorm')),
            f'{group} WM、Actor、Critic 或上身奖励没有收到梯度')
    for row in metrics:
        require(all(float(row[key]) == 0 for key in ('epa_resets', 'invalid_transitions', 'physics_exceptions')),
                f'{group} 出现 EPA 重置或无效转移')
    print(f'{group} 通过两轮 Newton 冒烟：{final["sim_steps"]} 条转移，'
          f'WM/Actor/Critic 梯度 {values["wm_gradnorm"]:.6g}/{values["pi_gradnorm"]:.6g}/{values["critic_gradnorm"]:.6g}，'
          f'上身奖励 {values["body_prior_contribution"]:.6g}，上身梯度 {values["term_body_prior_actor_gradnorm"]:.6g}。',
          flush=True)
    return initial


def compare_initial(a, b, historical):
    import torch

    for name in MODELS:
        prim, other, old = (ck[name] for ck in (a, b, historical))
        require(prim.keys() == other.keys() == old.keys(), f'{name} 初始化网络结构有差异')
        require(all(prim[k].shape == old[k].shape and prim[k].dtype == old[k].dtype for k in prim),
                f'{name} 张量形状或类型与历史 M4.3 不符')
        require(all(torch.equal(value, other[key]) for key, value in prim.items()), f'{name} 两组种子 0 初始化不一致')
        historical_equal = sum(torch.equal(value, old[key]) for key, value in prim.items())
        weights = [value for key, value in prim.items() if key.endswith(('.weight', '.bias'))]
        # 逐张量精确比较已足够，使用元素个数和 L2 范数记录可读指纹。
        norm = math.sqrt(sum(float(value.double().square().sum()) for value in weights))
        print(f'{name} 两组完全相等，历史 M4.3 逐张量相等 {historical_equal}/{len(prim)}，'
              f'参数元素 {sum(value.numel() for value in weights)}，L2={norm:.9g}。', flush=True)
    for key in ('goal_contract', 'policy_contract', 'data_contract', 'imagination_contract'):
        require(a[key] == b[key] == historical[key], f'{key} 与 M4.3 基线不一致')
    require(a['physical_contract'] == b['physical_contract'], '两组物理契约不一致')
    baseline_physical = dict(historical['physical_contract'], backend_versions=a['physical_contract']['backend_versions'])
    require(a['physical_contract'] == baseline_physical, '两组物理契约与 M4.3 基线不一致')
    for group, ck in zip(GROUPS, (a, b)):
        expected = deepcopy(historical['task_contract'])
        expected['config']['body_prior_weight'] = GROUPS[group]
        require(ck['task_contract'] == expected, f'{group} 任务与历史 M4.3 的差异超出上身奖励权重')
        baseline_args = historical['args']
        ignored = RUNTIME_PATHS | SMOKE_OVERRIDES.keys() | {'git_commit', 'body_prior_weight', 'reward_config',
                                                            'environment_versions'}
        common = (ck['args'].keys() & baseline_args.keys()) - ignored
        mismatch = sorted(key for key in common if ck['args'][key] != baseline_args[key])
        require(not mismatch, f'{group} 生效参数与历史 M4.3 存在额外差异：{mismatch}')


def check_periodic(run, reference):
    # 使用第一组冒烟的末轮权重，仅由正式的轻量周期入口复评一个检查点。
    run_checked([sys.executable, '-m', 'supermimic.integration.m43_evaluation', '--out', str(run),
                 '--m4_reference', str(reference), '--device', 'cuda:0', '--upper-metrics-only'], 'M4.6 周期评估')
    evaluation = run / 'evaluation'
    summary = json.loads((evaluation / 'results.json').read_text())
    metrics = summary['milestones']['2']['aggregate']
    require(metrics['valid_for_best'] and summary['best_iteration'] == 2, '周期评估未记录有效的 Best')
    for key in ('arm_rmse_rad', 'waist_rmse_rad', 'body_prior_contribution'):
        require(math.isfinite(metrics['upper_body'][key]), f'周期评估缺少有效的上身指标 {key}')
    checkpoints = run / 'checkpoints'
    require((checkpoints / 'best.pt').samefile(checkpoints / 'iter_2.pt'), '周期评估 Best 未链接到实际检查点')
    require({file.name for file in evaluation.iterdir()} == {'results.json'}, '周期评估产生了额外文件')
    print('单检查点周期评估通过，已核对步态 Best、上身指标和轻量归档。', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference', type=Path, default=Path('/m4_reference.npz'), help='真实 F2 参考文件')
    parser.add_argument('--config-only', action='store_true', help='只验证两份 JSON 文件的静态差异')
    args = parser.parse_args()
    baseline = check_files()
    if args.config_only:
        return
    reference = args.m4_reference.resolve(strict=True)
    effective = check_effective(baseline, reference)

    import torch

    torch.set_num_threads(2)
    require(torch.cuda.is_available(), 'Newton 冒烟测试需要项目容器内的 CUDA')
    with tempfile.TemporaryDirectory(prefix='m46-smoke-', dir='/tmp') as scratch:
        initial = {}
        for group, cfg in effective.items():
            run = Path(scratch) / group
            output = trainer(group, run / 'checkpoints', reference, '--m46_smoke')
            initial[group] = check_smoke_group(group, cfg, run, output)
            if group == 'w002':
                check_periodic(run, reference)
        historical = read_checkpoint(REPO / 'outputs/m4_3/checkpoints/initial.pt')
        compare_initial(*initial.values(), historical)
    print('M4.6 两组配置、初始化和 Newton 冒烟测试均通过，临时数据已清理。', flush=True)


if __name__ == '__main__':
    main()
