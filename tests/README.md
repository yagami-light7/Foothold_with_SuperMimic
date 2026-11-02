# S54 测试脚本

测试统一放在本目录：

```text
tests/
  unit/   # CPU回归、Goal/Observation/Reward和配置检查
  sim/    # Isaac Lab/Newton仿真、Smoke和Action Authority检查
```

从项目根目录运行CPU回归（40项）：

```bash
/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python -m unittest discover \
  -s tests/unit -t . -p 'test_*.py' -v
```

只检查四阶段课程调度，不依赖Torch或GPU：

```bash
python3 -m unittest tests.unit.test_command_curriculum -v
```

以下5个文件使用`main()`中的断言检查，需要单独运行：

```bash
/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python tests/unit/test_robot_cfg.py
/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python tests/unit/test_goal_sampler.py
/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python tests/unit/test_observation.py
/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python tests/unit/test_rewards.py
/mnt/sage_data/workspace/IsaacLab_3_0/.venv/bin/python tests/unit/test_preflight.py
```

脚本已自动添加项目根目录，直接运行无需设置`PYTHONPATH`。`test_goal_sampler.py`会在GPU可用时使用GPU，只有张量运算，不启动物理仿真。

需要物理仿真时，使用Isaac Lab解释器或原来的`isaaclab -p`入口，例如：

```bash
uv run --project /mnt/sage_data/workspace/IsaacLab_3_0 --no-sync \
  isaaclab -p tests/sim/test_s54_newton.py --num-steps 100 --viz none

uv run --project /mnt/sage_data/workspace/IsaacLab_3_0 --no-sync \
  isaaclab -p tests/sim/smoke_goal_env.py --num_envs 16 --viz none
```

`tests/sim/`还包含接触、站立、足端状态、重力、奖励Smoke和`action_authority_test.py`；每个脚本可加`--help`查看参数。仿真检查由用户显式启动，默认unittest发现过程不会导入它们。

VS Code测试面板配置为发现`tests/unit/`。训练、回放及训练监督器使用的评估工具继续位于`scripts/`；历史日志、checkpoint和报告中的SHA256清单保留原始记录。
