# 日志整理与目标足印可视化

日期：2026-10-29，Asia/Shanghai。

## 日志归类

`logs`从约1.4GB整理到84MB，仅保留10月28日晚间至今的7个关键训练记录：

| 分类 | 保留内容 |
|---|---|
| stand | 累计200更新的站立通过点；累计600更新的lift通过点 |
| step | 28日23:16 baseline；5300更新迁移起点；14249更新最终自转对照 |
| walk | 暂无模型；此前课程尚未进入walk |
| straight | 100更新的统计折算消融；保留统计的100/200/300更新实验 |

详细目录、checkpoint及原路径对应关系见`../../logs/README.md`和`relocation.json`。

其余记录移入项目内`.log_archive/2026-10-29_before_cleanup/logs`，没有永久删除，因此本次减少的是logs目录体积，没有释放备份所占磁盘。备份已加入.gitignore。53个保留checkpoint的SHA256移动前后一致；历史143个训练分块、14249次更新和142次评估均可读。旧原始log/metrics保持原文；可继续使用的straight state/launcher路径及文档命令已更新。

## 为什么步子小

用户原始默认值确实是`[0.12, ±0.19]m`，分别表示“摆动脚目标相对支撑脚向前12cm”和“横向脚间距19cm”。`goal_sampler.py`及环境默认值仍然如此。

但`configure_stage()`把step/straight的前向偏移覆盖为0.03m，walk预设为0.06m；checkpoint保存的配置优先于默认值。当前模型实际训练目标是`[0.03, ±0.19]m`，因此可见步幅小有明确配置原因。

这里的3cm是一次交替落脚时，相对支撑脚的前向目标差。理想连续交替下，同一只脚两次落点的间距约6cm，名义前向速度约`2 × 0.75Hz × 0.03m = 0.045m/s`。原12cm目标在相同频率下对应约0.18m/s。这些是理想几何参考，不是实测脚位移或速度保证。

抬脚高度是另一参数：当前straight为3cm，旧step为2cm。之前heading报告把它写成相同，现已更正说明；本次没有修改训练配置数值、action scale、PD或模型。

## 转向改善与当前限制

同为64环境、1000步独立评估：旧12小时最终模型平均绝对偏航角速度为0.6438rad/s，当前straight模型为0.3394rad/s。与5300更新迁移起点比较，最大累计朝向误差从31.57°降到15.55°。这支持“累计转向有所改善”的观察，但仍超过0.15rad/s及10°直行门槛。

当前沿固定参考方向速度为0.0430m/s，接近3cm课程的名义速度；同时左/右半周期成功率为0/86.5%，末期XY误差4.35cm，连续8个成功半周期比例为0。这些结果不能证明左右交替步态已经稳定，也不能仅凭躯干前移推断每只脚都达到了目标。

原始证据：`../posttrain_2026-10-29/eval_final_14249.json`、`../heading_v1_2026-10-29/baseline_old_step.json`、`../heading_v1_2026-10-29/normalization_preserved_300_updates.json`。

## 可视化与回放

新增`tasks/foot_placement/visualization.py`，仅绘制marker，不创建物理碰撞体：

- 青色为左脚目标，橙色为右脚目标；摆动脚轮廓亮、支撑脚轮廓暗。
- 轮廓内箭头表示目标脚尖方向；中心圆点表示目标sole原点。
- 紫点表示当前相位的摆动高度参考，其XY仍是最终落点，不代表完整的足端摆动轨迹。
- 轮廓长度使用toe/heel间距，宽度仅用于示意；绘图抬高6–9mm避免与地面闪烁，sampler目标不变。

play默认开启目标显示，`--no-show-goals`可关闭。`--policy-only`让全部环境执行策略；不传时仍是前半策略、后半零动作对照。窗口关闭后会退出，统计按实际执行步数计算。

```bash
cd /mnt/sage_data/workspace/KUAVO_S54_Newton
PYTHONPATH="$PWD" uv run \
  --project /mnt/sage_data/workspace/IsaacLab_3_0 --no-sync \
  isaaclab -p scripts/play.py \
  --checkpoint logs/straight/latest/model_final.pt \
  --num_envs 4 --steps 1000 \
  --policy-only --show-goals --viz newton_gl
```

实际启动会打印水平目标`[0.030, ±0.190]m`、抬脚高度0.030m和步态频率0.75Hz。未加12cm回放覆盖参数，避免把超出当前训练范围的目标误作模型训练时的任务。恢复12cm应在后续训练中逐步扩大目标并独立评估。

## 验证

- Newton GL：4环境、600步带目标回放完成；无跌倒，正常timeout一次/环境。
- 真实Newton marker探针：4环境，56条线段、12个参考点；变换均有限，目标/phase逐元素保持不变。见`marker_probe.json`、`marker_probe.log`。
- CPU heading回归测试3项通过；旧step无渲染/零动作对照回放另见`play_baseline_smoke.log`。
- 静态检查42文件，0错误、0警告，结果见`static_check_summary.json`。

修改文件：`.gitignore`、`scripts/play.py`、新增`tasks/foot_placement/visualization.py`；`configs/s54_curriculum.py`仅补注释。另更新了heading/posttrain文档中的可执行路径和历史分析脚本读取位置，并生成日志索引；原始checkpoint参数不变。
