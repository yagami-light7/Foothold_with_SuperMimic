"""Render existing Stage2 Newton NPZ: robot + generated foot references; no rollout."""
import argparse,json,os,sys,subprocess,math,xml.etree.ElementTree as ET
from pathlib import Path
from copy import deepcopy
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def main():
 p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--fps',type=int,default=25);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 os.environ.setdefault('MUJOCO_GL','egl')
 import numpy as np,torch,mujoco
 from PIL import Image,ImageDraw,ImageFont
 from supermimic.mdp import unflatten_state
 from supermimic.kinematics import S54Kinematics
 from supermimic.s54 import MJCF_PATH,SOURCE_MJCF,FOOT_BODY_NAMES
 from tools.visualization.render import apply_render_style,set_pose,REFERENCE_RGBA
 from tools.visualization.render_s54_joint_comparison import resolve_ffmpeg
 from tools.visualization.policy_preview import gpu_slot
 data=np.load(a.record);row=json.loads(str(data['summary']));S=unflatten_state(torch.from_numpy(data['state']))
 torch.set_num_threads(4);kin=S54Kinematics('cpu');foot_ids=[kin.tracked_names.index(n) for n in FOOT_BODY_NAMES]
 with torch.no_grad():actual_feet=kin.forward(*S)[0][:,foot_ids].numpy()
 target_feet=data['reference_foot_p'];path_stride=max(1,math.ceil(len(target_feet)/150))
 def path_line(scene,first,last,color):
  if np.linalg.norm(last-first)<1e-8:return
  if scene.ngeom>=scene.maxgeom:raise RuntimeError('Too many trajectory markers')
  geom=scene.geoms[scene.ngeom]
  mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).flatten(),np.asarray(color,dtype=np.float32))
  mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_LINE,2.,np.asarray(first,dtype=float),np.asarray(last,dtype=float));scene.ngeom+=1
 width,height=960,720;root=ET.parse(MJCF_PATH).getroot();root.find('compiler').set('meshdir',str(SOURCE_MJCF.parent.parent/'meshes'))
 apply_render_style(root,width,height);world=root.find('worldbody')
 for side,name in enumerate(FOOT_BODY_NAMES):
  body=ET.SubElement(world,'body',name=f'command_foot_{side}',mocap='true')
  for source in root.find(f'.//body[@name="{name}"]').findall('geom'):
   if source.get('type')!='mesh':continue
   geom=deepcopy(source);geom.attrib.pop('name',None);geom.attrib.update(material='render_reference',rgba=REFERENCE_RGBA,contype='0',conaffinity='0');body.append(geom)
 ET.SubElement(world,'geom',name='render_floor',type='plane',size='0 0 .01',material='groundplane',contype='0',conaffinity='0')
 a.out.parent.mkdir(parents=True,exist_ok=True);temp=a.out.with_name(a.out.stem+'.partial.mp4');font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
 with gpu_slot():
  m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m);cam=mujoco.MjvCamera();cam.azimuth=125;cam.elevation=-13;cam.distance=3.6
  opt=mujoco.MjvOption();opt.geomgroup[3]=0
  encoder=subprocess.Popen([resolve_ffmpeg(),'-n','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(a.fps),'-i','-','-an','-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],stdin=subprocess.PIPE)
  frames=round(row['duration_s']*a.fps)
  try:
   with mujoco.Renderer(m,height,width) as r:
    for frame in range(frames):
     t=frame/a.fps;i=min(round(t/.02),len(S[0])-1);ended=t>(len(S[0])-1)*.02
     set_pose(m,d,S[0][i].numpy(),S[1][i].numpy(),S[4][i].numpy())
     for side in (0,1):
      bid=m.body(f'command_foot_{side}').mocapid;d.mocap_pos[bid]=data['reference_foot_p'][i,side];d.mocap_quat[bid]=data['reference_foot_quat'][i,side]
     mujoco.mj_forward(m,d);cam.lookat[:]=[float(S[0][i,0]),float(S[0][i,1]),.75];r.update_scene(d,camera=cam,scene_option=opt)
     for side in (0,1):
      for j in range(path_stride,len(target_feet),path_stride):path_line(r.scene,target_feet[j-path_stride,side],target_feet[j,side],(.1,.8,.95,.8))
      for j in range(path_stride,i+1,path_stride):path_line(r.scene,actual_feet[j-path_stride,side],actual_feet[j,side],(.95,.48,.12,1.))
     im=Image.fromarray(r.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,width,85),fill='#16232f')
     draw.multiline_text((15,12),f"Stage2: {row['label']} | ENV0 t={i*.02:.2f}s | gray: robot | cyan: goal | orange: actual path\nEval aggregate ({row.get('evaluation_num_envs','multiple')} env): Foot RMSE={row['foot_rmse']:.3f}m | Survival={row['survival_rate']:.2f} | Success={row['success_rate']:.2f}",font=font,fill='white')
     if ended:
      draw.rectangle((0,height-50,width,height),fill='#7e2525');draw.text((15,height-40),'ACTUAL TRACE ENDED: '+row['stop_reasons'][0]+' | pose and target frozen',font=font,fill='white')
     encoder.stdin.write(np.asarray(im).tobytes())
     if frame==0:im.save(a.out.with_suffix('.png'))
     if frame%100==0:print(f'Render {row["label"]}: {frame}/{frames} frames',flush=True)
   encoder.stdin.close()
   if encoder.wait(timeout=60):raise RuntimeError('ffmpeg failed')
  except BaseException:encoder.kill();encoder.wait();temp.unlink(missing_ok=True);raise
 temp.replace(a.out);a.out.with_suffix('.json').write_text(json.dumps(dict(record=str(a.record),frames=frames,fps=a.fps,no_new_rollout=True,full_body_reference=False,reference='generated quintic feet',row=row),indent=2));print(a.out,flush=True)
if __name__=='__main__':main()
