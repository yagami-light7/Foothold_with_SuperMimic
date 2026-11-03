"""Read existing BVH/GMR targets without solving IK or writing source data.

Run with the existing Leju-GMR Python; the main diagnosis consumes the small NPZ.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gmr', type=Path, default=Path('/mnt/sage_data/workspace/Leju-GMR'))
    p.add_argument('--out', type=Path, default=Path('reports/s54_reference_figures/human_targets.npz'))
    a = p.parse_args()
    sys.path.insert(0, str(a.gmr))
    # Loading by filename avoids the unrelated ROS package also named `scripts`.
    spec = importlib.util.spec_from_file_location('s54_bvh_source', a.gmr / 'scripts/bvh_to_pkl.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
    frames, height = module.load_motion_data_kuavo_s54(
        'lafan1', str(a.gmr / 'assets/lafan1_bvh/dance1_subject1.bvh'), 0.01)
    gmr = GeneralMotionRetargeting('bvh_lafan1', 'kuavo_s54', actual_human_height=height, verbose=False)
    names = list(dict.fromkeys(entry[0] for entry in gmr.ik_match_table1.values()))
    positions, rotations, target_p, target_q = [], [], [], []
    for frame in frames:
        positions.append([frame[n][0] for n in names])
        rotations.append([frame[n][1] for n in names])
        gmr.update_targets(dict(frame), offset_to_ground=False)
        target_p.append([gmr.scaled_human_data[n][0] for n in names])
        target_q.append([gmr.scaled_human_data[n][1] for n in names])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, names=np.array(names), positions=positions, quats_wxyz=rotations,
                        target_positions=target_p, target_quats_wxyz=target_q,
                        human_height=height, source_fps=30.0, position_scale_to_meter=0.01,
                        ground_offset=gmr.ground_offset)
    print(json.dumps(dict(frames=len(frames), human_height=height, ground_offset=gmr.ground_offset,
                          output=str(a.out)), indent=2))


if __name__ == '__main__':
    main()
