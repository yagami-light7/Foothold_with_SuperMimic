#!/bin/bash
# S54 is retargeted locally by Leju-GMR; no G1 downloads or old-container actions.
set -euo pipefail
cd "$(dirname "$0")/../.."
python tools/data/copy_s54_sources.py "$@"
python tools/data/prepare_s54_motion.py --expected_frames 3945
python tools/data/prepare_s54_asset.py
