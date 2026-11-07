"""Render recorded Stage15 Newton trajectories with original full-body ghosts."""
import argparse,json,os,sys,subprocess,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--offset',type=float,default=0.);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--diagnostic',action='store_true');p.add_argument('--fps',type=int,default=25);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    os.environ.setdefault('MUJOCO_GL','egl')
    import torch,numpy as np,mujoco
    torch.set_num_threads(4)
    from PIL import Image,ImageDraw,ImageFont
    from contextlib import ExitStack
    from tools.visualization.render import load_recording,two_robot_xml,set_pose
    from tools.visualization.render_s54_joint_comparison import resolve_ffmpeg,playback_index
    from tools.visualization.policy_preview import gpu_slot
    cfg=json.loads((a.run_dir/'profile.json').read_text());controllers=[]
    names=[('eval_a','motion','Teacher Motion'),('eval_a','foot','Phase A | Foot only')]
    if (a.run_dir/'eval_b/summary.json').exists():names.append(('eval_b','foot','Phase B | Foot only'))
    if a.diagnostic and (a.run_dir/'diagnostic_masked_oracle/summary.json').exists():names.append(('diagnostic_masked_oracle','foot','Diagnostic | Teacher latent + mask'))
    for phase,role,title in names:
        summary=json.loads((a.run_dir/phase/'summary.json').read_text())
        row=next(r for r in summary['rows'][role] if math.isclose(r['offset_s'],a.offset))
        record=load_recording(Path(row['record']),str(ROOT/cfg['reference_manifest']))
        controllers.append((title,row,record))
    width,height=512,640;totalwidth=width*len(controllers);seconds=10.;count=round(a.fps*seconds)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    titlefont=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    a.out.parent.mkdir(parents=True,exist_ok=True);temp=a.out.with_name('.'+a.out.stem+'.partial.mp4')
    with gpu_slot(),ExitStack() as stack:
        scenes=[]
        for _,_,record in controllers:
            m=mujoco.MjModel.from_xml_string(two_robot_xml(width,height));d=mujoco.MjData(m)
            r=stack.enter_context(mujoco.Renderer(m,height,width));cam=mujoco.MjvCamera();cam.azimuth=135;cam.elevation=-12
            option=mujoco.MjvOption();option.geomgroup[3]=0;scenes.append((m,d,r,cam,option))
        encoder=subprocess.Popen([resolve_ffmpeg(),'-n','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{totalwidth}x{height}','-r',str(a.fps),'-i','-',
            '-an','-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],stdin=subprocess.PIPE)
        try:
            for frame in range(count):
                images=[]
                for (title,row,record),(m,d,r,cam,opt) in zip(controllers,scenes):
                    i,ended=playback_index(frame/a.fps,row['episode_lengths'][0],.02)
                    mat=np.empty(9);mujoco.mju_quat2Mat(mat,record['root_quat'][i,0].astype(float))
                    set_pose(m,d,record['root_p'][i,0],mat.reshape(3,3),record['q'][i,0])
                    rf=int(record['reference_frame'][i,0]);rp=record['ref_root_p'][rf]
                    set_pose(m,d,rp,record['ref_root_R'][rf],record['ref_q'][rf],prefix='ref_');mujoco.mj_forward(m,d)
                    ap=record['root_p'][i,0];center=(ap[:2]+rp[:2])*.5
                    cam.lookat[:]=[center[0],center[1],.8];cam.distance=max(4.,3+1.4*np.linalg.norm(ap[:2]-rp[:2]))
                    r.update_scene(d,camera=cam,scene_option=opt);im=Image.fromarray(r.render());draw=ImageDraw.Draw(im)
                    draw.rectangle((0,0,width,90),fill='#152332');draw.text((10,8),title,font=titlefont,fill='white')
                    draw.multiline_text((10,36),f'offset {a.offset:g}s | t={i*.02:.2f}s | Motion NR={row["norm_return"]:.3f}\nFoot RMSE={row["metrics"]["foot_position_rmse_m"]:.3f}m',font=font,fill='#d6e7f1')
                    draw.rectangle((0,height-65,width,height),fill='#152332')
                    draw.multiline_text((10,height-58),'Gray: real Newton policy | Cyan: LAFAN1 reference\nFull reference is visualization/scoring only for Foot',font=font,fill='#c6eaf7')
                    if ended and row['n_failed']:
                        draw.rectangle((5,height-110,width-5,height-70),fill='#85252a')
                        draw.text((12,height-101),f'TERMINATED {i*.02:.2f}s: {row["stop_reasons"][0]}',font=font,fill='white')
                    images.append(np.asarray(im))
                pixels=np.concatenate(images,1);encoder.stdin.write(pixels.tobytes())
                if frame==0:Image.fromarray(pixels).save(a.out.with_suffix('.png'))
                if frame%125==0:print(f'Render offset={a.offset:g}: {frame}/{count} frames',flush=True)
            encoder.stdin.close()
            if encoder.wait(timeout=60):raise RuntimeError('MP4 encoder failed')
        except BaseException:
            encoder.kill();encoder.wait();temp.unlink(missing_ok=True);raise
    temp.replace(a.out)
    a.out.with_suffix('.json').write_text(json.dumps(dict(controllers=[c[0] for c in controllers],offset=a.offset,frames=count,fps=a.fps,
        seconds=count/a.fps,actual_recordings=[c[1]['record'] for c in controllers],terminated_panels='Freeze actual AND reference, labelled; no reward after termination',
        phase_b='NOT RUN' if not (a.run_dir/'eval_b/summary.json').exists() else 'Recorded',no_new_rollouts=True),indent=2))
    print(a.out,flush=True)
if __name__=='__main__':main()
