"""Render S54 evaluation recordings; grayscale policy and cyan reference, offscreen MuJoCo.

CPU headless fallback: MUJOCO_GL=osmesa python tools/visualization/render.py traj.npz out.mp4
"""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
os.environ.setdefault('MUJOCO_GL','egl')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from supermimic.s54 import MJCF_PATH, SOURCE_MJCF, ROOT_BODY, JOINT_NAMES, ROBOT_ID


POLICY_RGBA = '0.78 0.78 0.78 1'
POLICY_JOINT_RGBA = '0.30 0.30 0.30 1'
POLICY_DARK_RGBA = '0.10 0.10 0.10 1'
REFERENCE_RGBA = '0.08 0.75 0.95 0.38'


def apply_render_style(root, width, height):
    """Style the in-memory render copy; keep the training asset/contract untouched."""
    asset = root.find('asset')
    for texture in asset.findall('texture'):
        if texture.get('type') == 'skybox':
            texture.set('rgb1', '0.12 0.20 0.32')
            texture.set('rgb2', '0.48 0.61 0.74')
        elif texture.get('name') == 'groundplane':
            texture.set('rgb1', '0.22 0.26 0.30')
            texture.set('rgb2', '0.31 0.36 0.40')
            texture.set('markrgb', '0.40 0.45 0.49')
    palette = dict(render_policy=POLICY_RGBA, render_joint=POLICY_JOINT_RGBA,
                   render_dark=POLICY_DARK_RGBA, render_reference=REFERENCE_RGBA)
    for name, rgba in palette.items():
        ET.SubElement(asset, 'material', name=name, rgba=rgba,
                      specular='0.12', shininess='0.18', reflectance='0')
    floor = asset.find("material[@name='groundplane']")
    floor.set('specular', '0'); floor.set('reflectance', '0'); floor.set('texrepeat', '1 1')
    world = root.find('worldbody')
    for geom in world.find(f"body[@name='{ROOT_BODY}']").iter('geom'):
        if geom.get('type') == 'mesh':
            mesh = geom.get('mesh', '')
            if mesh.endswith(('foot_roll', 'hand_pitch_nohand')) or mesh in ('head_camera', 'head_rader'):
                material = 'render_dark'
            elif mesh == 'waist_yaw' or mesh.endswith(('leg_roll', 'leg_yaw', 'foot_pitch',
                                                      'arm_roll', 'arm_yaw', 'hand_yaw', 'hand_roll')):
                material = 'render_joint'
            else:
                material = 'render_policy'
            geom.set('material', material); geom.set('rgba', palette[material])
    # One diffuse key light plus a soft headlight avoids washing out the colors.
    for light in list(world.findall('light')):
        world.remove(light)
    ET.SubElement(world, 'light', pos='-3 -4 6', dir='3 4 -6', directional='true',
                  diffuse='0.65 0.65 0.65', specular='0.10 0.10 0.10', castshadow='true')
    visual = root.find('visual')
    if visual is None: visual = ET.SubElement(root, 'visual')
    def visual_element(name):
        element = visual.find(name)
        return ET.SubElement(visual, name) if element is None else element
    visual_element('headlight').attrib.update(diffuse='0.35 0.35 0.35', ambient='0.22 0.22 0.22',
                                             specular='0.05 0.05 0.05')
    visual_element('rgba').set('haze', '0.38 0.48 0.60 1')
    visual_element('global').attrib.update(offwidth=str(width), offheight=str(height))


def two_robot_xml(width=960,height=540):
    root=ET.parse(MJCF_PATH).getroot()
    root.find('compiler').set('meshdir',str(SOURCE_MJCF.parent.parent/'meshes'))
    apply_render_style(root, width, height)
    world=root.find('worldbody');body=world.find(f"body[@name='{ROOT_BODY}']")
    ghost=deepcopy(body)
    for element in ghost.iter():
        if 'name' in element.attrib: element.set('name','ref_'+element.get('name'))
        if element.tag=='geom':
            element.set('contype','0');element.set('conaffinity','0')
            if element.get('type')=='mesh':
                element.set('material','render_reference');element.set('rgba',REFERENCE_RGBA)
    world.append(ghost)
    ET.SubElement(world,'geom',name='render_floor',type='plane',size='0 0 0.01',material='groundplane',contype='0',conaffinity='0')
    return ET.tostring(root,encoding='unicode')


def quat(R):
    q=np.empty(4);mujoco.mju_mat2Quat(q,np.asarray(R,dtype=np.float64).reshape(-1));return q


def set_pose(m,d,p,R,q,prefix=''):
    body=m.body(prefix+ROOT_BODY).id
    free=int(m.body_jntadr[body]);adr=int(m.jnt_qposadr[free])
    assert m.jnt_type[free]==mujoco.mjtJoint.mjJNT_FREE
    d.qpos[adr:adr+3]=p;d.qpos[adr+3:adr+7]=quat(R)
    assert q.shape==(len(JOINT_NAMES),)
    for i,n in enumerate(JOINT_NAMES):d.qpos[m.jnt_qposadr[m.joint(prefix+n).id]]=q[i]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('traj');ap.add_argument('out');ap.add_argument('--clips',nargs='*')
    ap.add_argument('--seconds',type=float,default=5);ap.add_argument('--start',type=float,default=0)
    ap.add_argument('--fps',type=int,default=25);ap.add_argument('--width',type=int,default=960);ap.add_argument('--height',type=int,default=540)
    ap.add_argument('--title',default='');ap.add_argument('--global_ghost',action='store_true')
    a=ap.parse_args()
    if a.fps<=0 or a.seconds<=0 or a.start<0 or a.width%2 or a.height%2:ap.error('Positive fps/seconds, nonnegative start and even video dimensions required')
    with np.load(a.traj,allow_pickle=False) as archive:d=dict(archive)
    if str(d.get('robot',''))!=ROBOT_ID or list(d.get('joint_names',[]))!=JOINT_NAMES or str(d.get('quat',''))!='wxyz':
        raise ValueError('Recording must contain the S54 robot/joint/quaternion contract from evaluate.py')
    if d['q'].shape[-1]!=len(JOINT_NAMES) or d['ref_q'].shape[-1]!=len(JOINT_NAMES):raise ValueError('Recording joint dimension differs')
    names=list(d['names']);clips=a.clips or names
    if any(n not in names for n in clips):ap.error(f'Unknown clip; available: {names}')
    dt=float(d['control_dt']);model=mujoco.MjModel.from_xml_string(two_robot_xml(a.width,a.height))
    assert model.nq==2*(7+len(JOINT_NAMES))
    data=mujoco.MjData(model)
    cam=mujoco.MjvCamera();cam.distance,cam.azimuth,cam.elevation=4.2,135,-12
    opt=mujoco.MjvOption();opt.geomgroup[3]=0
    try:font=ImageFont.load_default(size=18)
    except TypeError:font=ImageFont.load_default()
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    ff=None;written=0
    with mujoco.Renderer(model,a.height,a.width) as renderer:
        ff=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{a.width}x{a.height}',
            '-r',str(a.fps),'-i','-','-c:v','libx264','-pix_fmt','yuv420p','-crf','20',a.out],stdin=subprocess.PIPE)
        try:
            for clip in clips:
                c=names.index(clip);T=len(d['root_p']);first=int(round(a.start/dt))
                end=min(int(d['clip_len'][c]),int(np.ceil((a.start+a.seconds)/dt)))
                if first>=end:raise ValueError('Requested start beyond available rollout')
                # Sample exact output times; fps > control fps safely duplicates frames.
                times=np.arange(a.start,end*dt,1/a.fps);indices=np.minimum(np.rint(times/dt).astype(int),end-1)
                failed=np.flatnonzero(~d['alive'][:,c]);fell=int(failed[0]) if len(failed) else None
                for t in indices:
                    tt=min(t,T-1,fell) if fell is not None else min(t,T-1)
                    # The full reference continues after policy termination. Only the
                    # genuinely recorded terminal policy pose is frozen and labelled.
                    f=int(d['clip_start'][c])+t
                    R=np.empty(9);mujoco.mju_quat2Mat(R,np.asarray(d['root_quat'][tt,c],dtype=np.float64));R=R.reshape(3,3)
                    set_pose(model,data,d['root_p'][tt,c],R,d['q'][tt,c])
                    rp=d['ref_root_p'][f].copy();rR=d['ref_root_R'][f].copy()
                    if not a.global_ghost:
                        # Match reward anchoring in BOTH xy and heading.
                        rel=R@rR.T;yaw=np.arctan2(rel[1,0],rel[0,0]);co,si=np.cos(yaw),np.sin(yaw)
                        dR=np.array([[co,-si,0],[si,co,0],[0,0,1]])
                        rp[:2]=d['root_p'][tt,c,:2];rR=dR@rR
                    set_pose(model,data,rp,rR,d['ref_q'][f],prefix='ref_')
                    mujoco.mj_forward(model,data)
                    cam.lookat[:]=[data.qpos[0],data.qpos[1],.85]
                    renderer.update_scene(data,camera=cam,scene_option=opt)
                    img=Image.fromarray(renderer.render());draw=ImageDraw.Draw(img)
                    label=f'{a.title} {clip} t={(f-int(d["clip_start"][c]))*dt:.2f}s'
                    if fell is not None and t>=fell:label+=' [terminated]'
                    for xy,txt in [((12,10),label),((12,a.height-30),'gray: S54 policy | cyan: S54 reference')]:
                        bb=draw.textbbox(xy,txt,font=font);draw.rectangle((bb[0]-4,bb[1]-3,bb[2]+4,bb[3]+3),fill='#152332');draw.text(xy,txt,font=font,fill='#edf3f8')
                    ff.stdin.write(np.asarray(img).tobytes());written+=1
        finally:
            ff.stdin.close();code=ff.wait()
    if code or not written:raise RuntimeError(f'ffmpeg exited {code}, frames={written}')
    print(f'wrote {a.out}: {written} frames at {a.fps} fps')


if __name__=='__main__':main()
