# S54 Retarget 导出与速度审计（2026-10-28）

77 段原始 BVH 中，已有 43 段顶层 CSV，34 段尚未导出。未完成的 .staging 候选不计入已导出。

| 批处理状态 | 数量 |
| --- | --- |
| 已导出 | 43 |
| 未通过导出质量检查 | 16 |
| run1_subject2 中断 | 1 |
| 尚未处理 | 17 |

旧 reference_safe 的 43 段中，21 段通过本次实际轨迹审计，22 段根部启动检查失败；它们的关节限位与 30/50 Hz 关节速度检查均通过，问题集中在根部启动瞬态。
修复后的 aiming1_subject1 和未修改的 dance1_subject1 位于 reference_startup_fixed，两段均通过全部审计并通过完整 Newton 仿真。
跨目录去重后，22 段有可用且通过审计的完整导出；其余 55 段需要生成或重新生成。此处不把几何证书通过等同于动态跟踪必然成功。

本次续训沿用原来的 aiming1_subject1、dance1_subject1 两段，仅读取 reference_startup_fixed，不会把其他旧输出或中断候选混入训练。

轨迹审计直接重算根部速度、角速度、启动脉冲、关节限位和 30/50 Hz 关节速度，并核对完整帧数、机器人契约、CSV 哈希及证书。昂贵的逐网格碰撞检查使用原有绑定 CSV 的已验收证书，本工具没有逐段重跑全套网格碰撞。

[完整审计 JSON](../results/s54/retarget_inventory/inventory.json) · [34 段未导出名单](../results/s54/retarget_inventory/missing_exports.txt) · [55 段需新导出名单](../results/s54/retarget_inventory/needs_valid_export.txt)

## 未导出名单

### 已失败（16）

- `aiming1_subject4`
- `aiming2_subject3`
- `dance2_subject1`
- `dance2_subject3`
- `dance2_subject4`
- `fallAndGetUp1_subject4`
- `fallAndGetUp2_subject2`
- `fallAndGetUp3_subject1`
- `fight1_subject2`
- `fight1_subject3`
- `fight1_subject5`
- `multipleActions1_subject3`
- `obstacles3_subject3`
- `obstacles5_subject3`
- `obstacles6_subject1`
- `obstacles6_subject5`

### 本次中断（1）

- `run1_subject2`

### 尚未处理（17）

- `run1_subject5`
- `run2_subject1`
- `run2_subject4`
- `sprint1_subject2`
- `sprint1_subject4`
- `walk1_subject1`
- `walk1_subject2`
- `walk1_subject5`
- `walk2_subject1`
- `walk2_subject3`
- `walk2_subject4`
- `walk3_subject1`
- `walk3_subject2`
- `walk3_subject3`
- `walk3_subject4`
- `walk3_subject5`
- `walk4_subject1`

## 复查命令

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /data/conda_envs/leju_gmr/bin/python \
  tools/audit_s54_retarget.py --required_clips aiming1_subject1 dance1_subject1
```

后续重新生成请使用当前生成器的新默认目录。它在导出时间线之外完成首帧 IK 收敛，未收敛、根部异常或其他证书失败的候选会被拒绝。旧目录保留用于核对，未覆盖原始训练数据。
