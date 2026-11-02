"""离线迁移S54 checkpoint到105维；不启动仿真、训练或SwanLab。"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from agents.checkpoint_compat import checkpoint_task_config, migrate_checkpoint_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-dim", type=int, choices=(104, 105), default=105)
    args = parser.parse_args()
    if args.output.exists() or args.output.parent.resolve() == args.checkpoint.parent.resolve():
        raise ValueError("请输出到独立目录的新文件，避免覆盖原checkpoint或任务配置")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint_task_config(args.checkpoint)
    # 将伴随配置也嵌入模型，复制单个pt文件时仍可恢复任务。
    saved.setdefault("infos", None)
    saved["infos"] = saved["infos"] or {}
    saved["infos"]["env_cfg"] = config
    migrated, report = migrate_checkpoint_dict(saved, args.target_dim)
    config = migrated["infos"]["env_cfg"]
    config_file = args.output.parent / "env_cfg.json"
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError("输出目录已有不同任务配置，请使用独立目录")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, args.output)
    config_file.write_text(json.dumps(config, indent=2))
    args.output.with_suffix(".migration.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({**report, "output": str(args.output), "optimizer_updates": 0}, indent=2))


if __name__ == "__main__":
    main()
