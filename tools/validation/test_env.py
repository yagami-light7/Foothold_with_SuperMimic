"""Compatibility entrypoint for the strict S54 Newton smoke."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

from tools.validation.test_env_s54 import main
if __name__ == "__main__": main()
