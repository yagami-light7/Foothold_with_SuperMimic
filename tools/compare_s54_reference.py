"""Render BVH human / original S54 / corrected S54 at identical source times.

CPU-only usage: MUJOCO_GL=osmesa python tools/compare_s54_reference.py
The yellow human is the BVH skeleton, rigidly aligned at the baseline root.
"""
import argparse
import json
import os
os.environ.setdefault('MUJOCO_GL', 'osmesa')
from pathlib import Path
import pickle
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from supermimic.s54 import MJCF_PATH, SOURCE_MJCF
from render import apply_render_style, set_pose


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate_dir', type=Path, default=Path('library/motion_reference/kuavo_s54/retarget_data/reference_startup_fixed'))
    p.add_argument('--baseline', type=Path, help='Baseline PKL/CSV; defaults to candidate_dir/baseline/<clip>_30fps.pkl')
    p.add_argument('--clip', default='dance1_subject1')
    p.add_argument('--out', type=Path, default=Path('results/s54/upper_body_comparison.mp4'))
    p.add_argument('--start', type=float, default=0)
    p.add_argument('--seconds', type=float, default=15)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--panel_width', type=int, default=480)
    p.add_argument('--height', type=int, default=600)
    a = p.parse_args()
    a.baseline = a.baseline or a.candidate_dir / 'baseline' / f'{a.clip}_30fps.pkl'
    if a.start < 0 or a.seconds <= 0:
        raise ValueError('Invalid start/seconds')
    corrected = np.loadtxt(a.candidate_dir / f'{a.clip}.csv', delimiter=',')
    if a.baseline.suffix == '.pkl':
        with a.baseline.open('rb') as f:
            source = pickle.load(f)
        original = np.c_[source['root_pos'], source['root_rot'], source['dof_pos'], corrected[:, -2:]]
    else:
        original = np.loadtxt(a.baseline, delimiter=',')
    compare_columns = list(range(20))
    ground_report = a.candidate_dir / 'diagnostics' / a.clip / 'grounding_metrics.json'
    if ground_report.exists():
        guard = np.load(a.candidate_dir / 'diagnostics' / a.clip / 'before_ground_guard.npz')
        if not np.allclose(corrected[:, 2] - guard['csv'][:, 2], guard['lift_m'], rtol=0, atol=1e-12):
            raise ValueError('Grounding root lift differs from its recorded trajectory')
        compare_columns.remove(2)
    expected = corrected
    adjustment_path = a.candidate_dir / 'diagnostics' / a.clip / 'self_collision_adjustment.npz'
    if adjustment_path.exists():
        adjustment = np.load(adjustment_path)
        expected = adjustment['before_csv']
        if not np.array_equal(corrected[:, 7:34], adjustment['joint_positions']):
            raise ValueError('Final joints differ from the recorded collision repair')
    if not np.array_equal(original[:len(corrected), compare_columns], expected[:, compare_columns]):
        raise ValueError('Root/lower body differs between the comparison inputs')
    with np.load(a.candidate_dir / 'reference' / f'{a.clip}_human.npz') as h:
        names, positions, quats = list(h['names']), h['positions'].copy(), h['quats_wxyz'].copy()
    cfg = json.loads((a.candidate_dir / 'diagnostics' / a.clip / 'upper_body_metrics.json').read_text())['config']
    root_offset = Rotation.from_quat(np.array(cfg['human_root_to_robot_wxyz'])[[1, 2, 3, 0]]).as_matrix()
    bones = [('Hips', 'Spine'), ('Spine', 'Spine1'), ('Spine1', 'Spine2'), ('Spine2', 'Neck'), ('Neck', 'Head')]
    for side in ('Left', 'Right'):
        bones.extend([(side + x, side + y) for x, y in [('Shoulder', 'Arm'), ('Arm', 'ForeArm'),
                                                      ('ForeArm', 'Hand'), ('UpLeg', 'Leg'), ('Leg', 'Foot'), ('Foot', 'Toe')]])
        bones.extend([('Hips', side + 'UpLeg'), ('Spine2' if 'Spine2' in names else 'Spine', side + 'Shoulder')])
    bones = [(names.index(x), names.index(y)) for x, y in bones if x in names and y in names]
    root = ET.parse(MJCF_PATH).getroot()
    root.find('compiler').set('meshdir', str(SOURCE_MJCF.parent.parent / 'meshes'))
    apply_render_style(root, a.panel_width, a.height)
    world = root.find('worldbody')
    ET.SubElement(world, 'geom', name='comparison_ground', type='plane', size='0 0 .01', material='groundplane', group='4')
    model = mj.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mj.MjData(model)
    cam = mj.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.4, 130, -12
    robot_opt, human_opt = mj.MjvOption(), mj.MjvOption()
    robot_opt.geomgroup[3] = 0
    robot_opt.geomgroup[4] = 1
    human_opt.geomgroup[:] = 0
    human_opt.geomgroup[4] = 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                           '-s', f'{3 * a.panel_width}x{a.height}', '-r', str(a.fps), '-i', '-',
                           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(a.out)], stdin=subprocess.PIPE)
    frames = 0
    try:
        with mj.Renderer(model, a.height, a.panel_width) as renderer:
            for i in range(round(a.seconds * a.fps)):
                t = a.start + i / a.fps
                f = round(t * 30)
                if f >= len(corrected):
                    break
                row = corrected[f]
                robot_R = Rotation.from_quat(row[3:7]).as_matrix()
                human_R = Rotation.from_quat(quats[f, names.index('Hips')][[1, 2, 3, 0]]).as_matrix() @ root_offset
                points = (positions[f] - positions[f, names.index('Hips')]) @ (robot_R @ human_R.T).T + row[:3]
                cam.lookat[:] = row[:3] + [0, 0, .12]
                panels = []
                for panel, label in enumerate(['Human BVH (root aligned)', 'Original S54 reference', 'Corrected S54 reference']):
                    value = original[f] if panel == 1 else corrected[f]
                    set_pose(model, data, value[:3], Rotation.from_quat(value[3:7]).as_matrix(), value[7:])
                    mj.mj_forward(model, data)
                    renderer.update_scene(data, camera=cam, scene_option=human_opt if panel == 0 else robot_opt)
                    if panel == 0:
                        for b, c in bones:
                            geom = renderer.scene.geoms[renderer.scene.ngeom]
                            mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array([.95, .65, .12, 1.]))
                            mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_CAPSULE, .014, points[b], points[c])
                            renderer.scene.ngeom += 1
                    img = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(img)
                    draw.rectangle((0, 0, a.panel_width, 43), fill='#202830')
                    draw.text((10, 8), label, fill='white')
                    draw.text((10, 25), f't = {t:.2f}s   source = 30 FPS', fill='white')
                    panels.append(img)
                combined = Image.new('RGB', (3 * a.panel_width, a.height))
                for panel, img in enumerate(panels):
                    combined.paste(img, (panel * a.panel_width, 0))
                if i == 0 or abs(t - 5) < .5 / a.fps or abs(t - 10) < .5 / a.fps:
                    combined.save(a.out.with_name(a.out.stem + f'_{f:06d}.png'))
                ff.stdin.write(np.asarray(combined).tobytes())
                frames += 1
    finally:
        ff.stdin.close()
        status = ff.wait()
    if status or frames == 0:
        raise RuntimeError(f'Video failed: ffmpeg={status}, frames={frames}')
    print(f'PASS comparison video: {a.out}; frames={frames}, fps={a.fps}')


if __name__ == '__main__':
    main()
