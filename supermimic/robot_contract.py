"""Select a robot before importing algorithm modules; historical runs default to S54.

The CLI sets SUPERMIMIC_ROBOT once, before imports. Mixing robot contracts in one
interpreter is deliberately unsupported: replay/network shapes are process-local.
"""
import os

ROBOT_KIND = os.environ.get('SUPERMIMIC_ROBOT', 's54')
if ROBOT_KIND == 'roban':
    from .roban import *  # noqa: F403
elif ROBOT_KIND == 's54':
    from .s54 import *  # noqa: F403
else:
    raise ValueError(f'Unknown SUPERMIMIC_ROBOT={ROBOT_KIND!r}')
