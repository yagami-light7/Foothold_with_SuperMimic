"""Resolve moved repository configs without changing saved task/physics contracts."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def resolve_config_path(value):
    path=Path(value)
    if path.is_file():return path
    text=path.as_posix()
    key='configs/'+text.rsplit('/configs/',1)[1] if '/configs/' in text else text
    aliases=json.loads((ROOT/'tools/layout_migration.json').read_text())['configs']
    return ROOT/aliases[key] if key in aliases else path
