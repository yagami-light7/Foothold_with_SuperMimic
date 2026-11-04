# S54 浏览器播放

当前服务：`s54_play_viser_8080` tmux，监听 `0.0.0.0:8080`；checkpoint为 `best_standing.pt`（iteration35480）。每回合最多10秒，物理终止或Timeout后重置，seed从9541逐回合增加；这是实际Newton闭环执行，不是WM预测或预录动画。原物理终止、成功标准与权重不变，viewer不计成功率。

同一局域网访问：<http://192.168.1.8:8080>。

远程电脑可以在**本地终端**建立SSH端口转发：

```bash
ssh -N -L 8080:127.0.0.1:8080 rm@192.168.1.8
```

服务器地址、SSH端口按平时连接服务器的参数替换；然后打开 <http://127.0.0.1:8080>。VS Code Remote SSH也可以在Ports面板转发服务器8080端口。

浏览器支持鼠标旋转/缩放，侧栏有Pause Simulation、Pause Rendering和Reset Episode。默认相机允许自由操作；只有显式添加 `--follow_camera` 才逐帧跟随。服务在tmux中运行，SSH断开不会停止；浏览器断开也不会停止循环。

服务器管理命令：

```bash
tmux attach -t s54_play_viser_8080  # Ctrl-b，然后d离开
tail -n 20 -F logs/s54_play_viser_8080/stdout.log
# 停止浏览服务、释放GPU
tmux kill-session -t s54_play_viser_8080
```

当前完整启动命令保存在 `logs/s54_play_viser_8080/launch_command.txt`。独立运行脚本的参数为：

```bash
/home/rm/workspace/IsaacLab_3_0/.venv/bin/python tools/play_s54_foot_tracking.py \
  --checkpoint logs/s54_curriculum_fresh_auditfix_20261104_032902/best_standing.pt \
  --viz viser --viser_host 0.0.0.0 --viser_port 8080 --headless --loop --seconds 10
```

已有服务时无需重复启动。停止现有服务后如需长期重新运行，可在tmux会话内执行上面的命令。

已验证：HTTP200，WebSocket握手成功，收到34组BatchedMeshes及持续SceneNodeUpdate；验证记录 `logs/s54_play_viser_8080/connectivity_check.json`。进程单环境显存约508MiB。未开启第三方public share，未新开训练。
