"""Render a genuine S54 reference preview, explicitly NOT a policy comparison/evaluation."""
import argparse
import os
os.environ.setdefault('MUJOCO_GL','osmesa')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import subprocess
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import torch
from PIL import Image,ImageDraw
from supermimic.s54 import MJCF_PATH,SOURCE_MJCF
from supermimic.motion import MotionLib
from supermimic.kinematics import S54Kinematics
from render import set_pose, apply_render_style


def main():
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=5);p.add_argument('--start',type=float,default=0)
 p.add_argument('--motion_dir',default='data/retarget_data/reference_startup_fixed');p.add_argument('--clip',help='Clip stem; defaults to the first clip')
 p.add_argument('--width',type=int,default=480);p.add_argument('--height',type=int,default=360);p.add_argument('--fps',type=int,default=30)
 p.add_argument('--out',default='results/s54/reference_preview.mp4');a=p.parse_args()
 torch.set_num_threads(4);motion=MotionLib.from_dir(a.motion_dir,S54Kinematics(),'cpu')
 clip=motion.names.index(a.clip) if a.clip else 0
 if a.start<0 or a.seconds<=0:raise ValueError('start must be nonnegative and seconds must be positive')
 root=ET.parse(MJCF_PATH).getroot();root.find('compiler').set('meshdir',str(SOURCE_MJCF.parent.parent/'meshes'))
 apply_render_style(root,a.width,a.height)
 ET.SubElement(root.find('worldbody'),'geom',name='preview_floor',type='plane',size='0 0 0.01',material='groundplane')
 m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m)
 cam=mujoco.MjvCamera();cam.distance,cam.azimuth,cam.elevation=3.5,130,-15
 opt=mujoco.MjvOption();opt.geomgroup[3]=0
 Path(a.out).parent.mkdir(parents=True,exist_ok=True)
 ff=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{a.width}x{a.height}',
  '-r',str(a.fps),'-i','-','-c:v','libx264','-pix_fmt','yuv420p',a.out],stdin=subprocess.PIPE)
 with mujoco.Renderer(m,a.height,a.width) as renderer:
  for i in range(round(a.seconds*a.fps)):
   local_f=round((a.start+i/a.fps)*50)
   if local_f>=int(motion.clip_len[clip]):break
   f=int(motion.clip_start[clip])+local_f
   set_pose(m,d,motion.root_p[f].numpy(),motion.root_R[f].numpy(),motion.q[f].numpy());mujoco.mj_forward(m,d)
   cam.lookat[:]=[d.qpos[0],d.qpos[1],.8];renderer.update_scene(d,camera=cam,scene_option=opt)
   img=Image.fromarray(renderer.render());draw=ImageDraw.Draw(img)
   draw.rectangle((0,0,a.width,24),fill='white');draw.text((8,7),'S54 REFERENCE ONLY - not policy evaluation',fill='black')
   if i==0:img.save(str(Path(a.out).with_suffix('.png')))
   ff.stdin.write(np.asarray(img).tobytes())
 ff.stdin.close();assert ff.wait()==0
 print(f'Reference-only preview: {a.out}')


if __name__=='__main__':main()
