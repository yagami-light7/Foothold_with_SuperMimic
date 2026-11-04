"""Resolve a fresh S54 profile into one safely quoted argv token per line."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse
import json
from pathlib import Path
import re

FORBIDDEN={'resume','init_weights','log_dir'}


def profile_args(path,saved_args=None,for_resume=False):
    cfg=json.loads(Path(path).read_text())['train']
    if not cfg.get('fresh_start') or cfg.get('wm_only_iters',0)<1000:
        raise ValueError('Fresh profile requires fresh_start=true and >=1000 WM-only iterations')
    if FORBIDDEN.intersection(cfg): raise ValueError('Profiles cannot load checkpoints or override log_dir')
    if saved_args:
        saved=json.loads(Path(saved_args).read_text())
        cfg={k:saved.get(k,v) for k,v in cfg.items()}
    out=[]
    for key,value in cfg.items():
        if not re.fullmatch('[a-z][a-z0-9_]*',key): raise ValueError('Invalid argument name')
        if for_resume and key=='fresh_start': continue
        if isinstance(value,bool):
            if value: out.append('--'+key)
        elif isinstance(value,(str,int,float)):
            text=str(value)
            if '\n' in text or '\r' in text or '\x00' in text: raise ValueError('Invalid argument value')
            out.extend(('--'+key,text))
        elif value is not None: raise ValueError('Profile arguments must be scalars')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('profile');p.add_argument('--saved-args')
    p.add_argument('--for-resume',action='store_true');a=p.parse_args()
    print('\n'.join(profile_args(a.profile,a.saved_args,a.for_resume)))
