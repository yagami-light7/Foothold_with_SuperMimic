"""Render saved M7 states only; rendering never supplies physics evidence."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ['SUPERMIMIC_ROBOT']='roban'
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout',required=True,type=Path)
    p.add_argument('--dataset_root',type=Path,default=os.environ.get('PMT_DATASET_ROOT','/mnt/sage_data/workspace/Foothold/PMT/TCRS/outputs/roban_pillar_dataset'))
    p.add_argument('--width',type=int,default=560);p.add_argument('--height',type=int,default=420)
    args=p.parse_args()
    import mujoco
    import numpy as np
    from PIL import Image,ImageDraw
    from supermimic.integration.pmt_dataset_loader import load_catalog
    from supermimic.integration.contact_diagnostics import RobotGeometry
    report=json.loads((args.rollout/'evaluation_metrics.json').read_text())
    samples=load_catalog(args.dataset_root,seed=44)
    sample=next(s for s in samples if s.name==report['scene_id'])
    actual=dict(np.load(args.rollout/'actual_trajectory.npz',allow_pickle=False))
    ref=dict(np.load(args.rollout/'reference_trajectory.npz',allow_pickle=False))
    video=args.rollout/'replay.mp4'
    if video.exists():raise FileExistsError(video)
    geometry=RobotGeometry(sample);model=geometry.model
    model.vis.global_.offwidth=args.width;model.vis.global_.offheight=args.height
    model.vis.headlight.ambient[:]=.55;model.vis.headlight.diffuse[:]=.7
    renderer=mujoco.Renderer(model,height=args.height,width=args.width)
    camera=mujoco.MjvCamera();camera.azimuth=130;camera.elevation=-20;camera.distance=2.9
    options=mujoco.MjvOption();options.geomgroup[1]=1
    encoder=subprocess.Popen(['ffmpeg','-n','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{args.width*2}x{args.height}',
        '-r','50','-i','-','-an','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    try:
        for i in range(len(actual['time_s'])):
            center=(actual['root_pos'][i]+ref['root_pos'][i])/2
            camera.lookat[:]=[center[0],center[1],.65]
            panels=[]
            for name,data,pp,active in [('Saved Newton state' if report['kinematic_playback_pass'] is None else 'Kinematic pose writes',actual,actual['pillar_pose'][i],actual['pillar_active'][i]),
                                        ('PMT final-IK reference',ref,ref['pillar_pose'][i],ref['pillar_active'][i])]:
                geometry.set_robot(data['root_pos'][i],data['root_quat'][i],data['joint_pos'][i])
                geometry.set_pillars(pp,active)
                renderer.update_scene(geometry.data,camera=camera,scene_option=options)
                image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
                draw.rectangle((0,0,args.width,54),fill='#12202d')
                label=f'{name}\nt={actual["time_s"][i]:.2f}s ref={actual["reference_frame"][i]} / 287 | {report["status"]}'
                draw.multiline_text((9,8),label,fill='white')
                draw.rectangle((0,args.height-45,args.width,args.height),fill='#12202d')
                draw.multiline_text((9,args.height-40),('KINEMATIC ONLY: no dynamic control claim' if report['kinematic_playback_pass'] is not None else
                    f'{actual["record_phase"][i]} | termination: {report["termination_reason"][:45]}'),fill='white')
                panels.append(image)
            frame=Image.new('RGB',(args.width*2,args.height));frame.paste(panels[0],(0,0));frame.paste(panels[1],(args.width,0))
            encoder.stdin.write(np.asarray(frame).tobytes())
            if i==len(actual['time_s'])//2:frame.save(args.rollout/'preview.png')
            if (i+1)%100==0:print(f'Render {args.rollout.name}: {i+1}/{len(actual["time_s"])}',flush=True)
    finally:
        encoder.stdin.close();code=encoder.wait();renderer.close()
    if code:raise RuntimeError(f'ffmpeg failed: {code}')
    probe=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,r_frame_rate,width,height','-of','json',str(video)],capture_output=True,text=True,check=True)
    evidence=json.loads(probe.stdout);require=evidence['streams'][0]
    if int(require['nb_read_frames'])!=len(actual['time_s']):raise RuntimeError('Video missing recorded states')
    (args.rollout/'video_verification.json').write_text(json.dumps(dict(status='PASS',rendering='saved Newton/kinematic states; MuJoCo rendering does not run dynamics',**evidence),indent=2)+'\n')
    print(video,flush=True)


if __name__=='__main__':main()
