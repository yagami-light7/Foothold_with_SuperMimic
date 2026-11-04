"""Same-camera full-body replay of recorded Newton states; rendering only, no simulation."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse
import os
from pathlib import Path
import subprocess
import shutil
import sys
import xml.etree.ElementTree as ET
os.environ.setdefault('MUJOCO_GL','egl')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from tools.visualization.render import apply_render_style,set_pose
from supermimic.s54 import MJCF_PATH,SOURCE_MJCF,JOINT_NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',nargs='+',required=True);p.add_argument('--labels',nargs='+',required=True)
    p.add_argument('--out',required=True);p.add_argument('--seconds',type=float,default=2.)
    p.add_argument('--fps',type=int,default=25)
    a=p.parse_args()
    if len(a.records)!=len(a.labels) or len(a.records)>4 or not 0<a.seconds<=10 or not 1<=a.fps<=50:p.error('Use 1–4 labelled recordings, <=10s, <=50fps')
    output=Path(a.out)
    if output.exists():p.error('Output exists')
    gif=output.suffix.lower()=='.gif'
    if not gif and not shutil.which('ffmpeg'):p.error('ffmpeg unavailable; use --out comparison.gif for Pillow GIF output')
    width,height=480,540;root=ET.parse(MJCF_PATH).getroot()
    root.find('compiler').set('meshdir',str(SOURCE_MJCF.parent.parent/'meshes'))
    apply_render_style(root,width,height)
    ET.SubElement(root.find('worldbody'),'geom',name='render_floor',type='plane',size='0 0 .01',material='groundplane',contype='0',conaffinity='0')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));data=mujoco.MjData(model)
    records=[]
    for path in a.records:
        with np.load(path,allow_pickle=False) as z:d=dict(z)
        if 'state' in d:
            x=d['state'][:,0];d['root_p']=x[:,:3];d['root_quat']=x[:,3:7];d['q']=x[:,13:42]
        else:
            if list(d['joint_names'])!=JOINT_NAMES:raise ValueError('Expected S54 recording')
            d['root_p']=d['root_p'][:,0];d['root_quat']=d['root_quat'][:,0];d['q']=d['q'][:,0]
        records.append(d)
    cam=mujoco.MjvCamera();cam.distance=3.2;cam.azimuth=135;cam.elevation=-12
    option=mujoco.MjvOption();option.geomgroup[3]=0
    output.parent.mkdir(parents=True,exist_ok=True);frames=0
    with mujoco.Renderer(model,height,width) as renderer:
        gif_frames=[]
        encoder=None if gif else subprocess.Popen(['ffmpeg','-n','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s',f'{width*len(records)}x{height}','-r',str(a.fps),'-i','-','-c:v','libx264','-pix_fmt','yuv420p',str(output)],stdin=subprocess.PIPE)
        try:
            for t in np.arange(0,a.seconds,1/a.fps):
                panels=[]
                for d,label in zip(records,a.labels):
                    tick=round(t/.02);index=min(tick,len(d['q'])-1)
                    R=np.empty(9);mujoco.mju_quat2Mat(R,d['root_quat'][index].astype(float))
                    set_pose(model,data,d['root_p'][index],R.reshape(3,3),d['q'][index]);mujoco.mj_forward(model,data)
                    cam.lookat[:]=[d['root_p'][index,0],d['root_p'][index,1],.72]
                    renderer.update_scene(data,camera=cam,scene_option=option)
                    panel=Image.fromarray(renderer.render());draw=ImageDraw.Draw(panel)
                    text=f'{label}\nt={t:.2f}s | recorded Newton states\n'+('[recording ended; pose held]' if tick>=len(d['q'])-1 else '')
                    draw.rectangle((8,8,width-8,65),fill='#152332');draw.multiline_text((12,12),text,fill='white')
                    panels.append(np.asarray(panel))
                combined=np.concatenate(panels,axis=1)
                if gif:gif_frames.append(Image.fromarray(combined).quantize(colors=128))
                else:encoder.stdin.write(combined.tobytes())
                frames+=1
        finally:
            if encoder:encoder.stdin.close();code=encoder.wait()
            else:code=0
        if gif:gif_frames[0].save(output,save_all=True,append_images=gif_frames[1:],duration=round(1000/a.fps),loop=0)
    if code:raise RuntimeError(f'ffmpeg failed {code}')
    print(f'{output}: {frames} frames, {a.fps} fps, fixed shared camera settings')


if __name__=='__main__':main()
