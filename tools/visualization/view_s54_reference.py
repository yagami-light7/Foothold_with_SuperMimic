"""Loop one exported S54 reference in a native desktop viewer; Space pauses."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse
import os
os.environ.setdefault('MUJOCO_GL', 'glfw')
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco as mj
import glfw
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from supermimic.s54 import MJCF_PATH, SOURCE_MJCF
from supermimic.motion import MotionLib
from supermimic.kinematics import S54Kinematics
from tools.visualization.render import apply_render_style, set_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion_dir', type=Path, default=REPO / 'data/retarget_data/reference_startup_fixed')
    parser.add_argument('--clip', default='dance1_subject1')
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--seconds', type=float, help='Optional automatic viewer shutdown for a desktop smoke test')
    args = parser.parse_args()
    if args.start < 0 or args.speed <= 0 or (args.seconds is not None and args.seconds <= 0):
        parser.error('Use nonnegative --start and positive --speed/--seconds')
    torch.set_num_threads(4)
    motion = MotionLib([str(args.motion_dir / f'{args.clip}.csv')], S54Kinematics(), 'cpu')
    root = ET.parse(MJCF_PATH).getroot()
    root.find('compiler').set('meshdir', str(SOURCE_MJCF.parent.parent / 'meshes'))
    apply_render_style(root, 960, 720)
    ET.SubElement(root.find('worldbody'), 'geom', name='reference_floor', type='plane',
                  size='0 0 0.01', material='groundplane')
    model = mj.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mj.MjData(model)
    paused = [False]

    def key_callback(window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_SPACE:
            paused[0] = not paused[0]
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

    local_time = args.start
    begin = previous = time.perf_counter()
    # Keep the OpenGL context and its teardown on the main thread. This avoids
    # the passive viewer's asynchronous window-close race on this desktop.
    if not glfw.init():
        raise RuntimeError('Cannot initialize desktop GLFW; run from your desktop terminal')
    window = glfw.create_window(960, 720, f'S54 reference: {args.clip} | Space: pause | Esc: close', None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError('Cannot create the reference viewer window')
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    glfw.set_key_callback(window, key_callback)
    camera = mj.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.4, 135, -12
    option = mj.MjvOption()
    option.geomgroup[3] = 0
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150)
    cursor = [None]

    def scroll(window, dx, dy):
        mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, 0., -.05 * dy, camera)

    def mouse(window, x, y):
        old = cursor[0]
        cursor[0] = (x, y)
        if old is not None and glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            height = max(glfw.get_window_size(window)[1], 1)
            mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ROTATE_H,
                             (x - old[0]) / height, (y - old[1]) / height, camera)

    glfw.set_scroll_callback(window, scroll)
    glfw.set_cursor_pos_callback(window, mouse)
    print(f'Viewing {args.clip}; Space pauses, drag rotates, scroll zooms, Esc closes.', flush=True)
    try:
        while not glfw.window_should_close(window):
            now = time.perf_counter()
            if args.seconds and now - begin >= args.seconds:
                break
            if not paused[0]:
                local_time += (now - previous) * args.speed
            previous = now
            frame = round(local_time * 50) % motion.num_frames
            set_pose(model, data, motion.root_p[frame].numpy(), motion.root_R[frame].numpy(), motion.q[frame].numpy())
            mj.mj_forward(model, data)
            camera.lookat[:] = [data.qpos[0], data.qpos[1], .8]
            mj.mjv_updateScene(model, data, option, None, camera, mj.mjtCatBit.mjCAT_ALL, scene)
            width, height = glfw.get_framebuffer_size(window)
            if width and height:
                mj.mjr_render(mj.MjrRect(0, 0, width, height), scene, context)
            glfw.swap_buffers(window)
            glfw.poll_events()
    finally:
        context.free()
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == '__main__':
    main()
