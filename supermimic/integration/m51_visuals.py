"""由冻结终评归档生成正面、侧面、俯视回放和足部诊断曲线。"""

from argparse import ArgumentParser
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'
    os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco
import numpy as np
import torch

from ..kinematics import RobotKinematics
from ..motion import CONTROL_DT
from ..roban import FOOT_BODY_NAMES, JOINT_NAMES, model_contract
from .m410_visuals import render_video
from .m48_evaluation import load_archive
from .m50_evaluation import TRACE_FIELDS
from .m50_reference import MANIFEST, REFERENCE_ROOT, load_clip, read_manifest
from .m51_audit import task
from .m51_foot import foot_axes
from .m51_metrics import read_guide_heading, trace_signals
from .m51_task import OUTPUT, SOURCE


FEET = ('#1685a5', '#d17025')
COLORS = ('#343b45', '#2266aa', '#2b8967', '#ba4f26', '#77519c')


def sole_profiles(motion):
    # 用实际两个圆柱和一个矩形碰撞代理的外缘绘制脚掌。
    model,_ = model_contract()
    profiles = []
    for name, sole in zip(FOOT_BODY_NAMES, motion.sole_offsets.numpy()):
        parts = []
        angle = np.linspace(0., 2*np.pi, 33)
        for g in np.flatnonzero(model.geom_bodyid == model.body(name).id):
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            sx, sy = model.geom_size[g,:2]
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                xy = np.column_stack((sx*np.cos(angle), sx*np.sin(angle)))
            elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                xy = np.array([[-sx,-sy],[sx,-sy],[sx,sy],[-sx,sy],[-sx,-sy]])
            else:
                raise ValueError('未知足底碰撞轮廓')
            xy += model.geom_pos[g,:2]-sole[:2]
            parts.append(np.column_stack((xy,np.zeros(len(xy)))))
        profiles.append(parts)
    return profiles


def cumulative_slip(speed, loaded):
    values = np.zeros_like(speed)
    for t in range(1,len(speed)):
        values[t] = (values[t-1]+speed[t]*CONTROL_DT*1000)*loaded[t]
    return values


def curves(target, traces, signals, labels):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    def styled(ax, title):
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.grid(alpha=.22)
        ax.legend(fontsize=7)

    with PdfPages(target) as pdf:
        for metrics in (('Course-relative yaw (deg)', 'Angular speed (rad/s)', 'Normal force (N)', 'Slip speed (mm/s)'),
                        ('Yaw error (deg)', 'Pitch error (deg)', 'Roll error (deg)', 'Hip yaw / ankle pitch / roll (deg)')):
            fig, axes = plt.subplots(2,4,figsize=(16,8),layout='constrained')
            for side, limb in enumerate('lr'):
                qids = [JOINT_NAMES.index(f'leg_{limb}{j}_joint') for j in (3,5,6)]
                for (name,trace),color in zip(traces.items(),COLORS):
                    signal, t = signals[name], trace['frame']*CONTROL_DT
                    if metrics[0].startswith('Course'):
                        line = (signal['local'][:,side],signal['omega'][:,side],
                                signal['force'][:,side] if name != 'reference' else np.full_like(t,np.nan),
                                1000*signal['speed'][:,side])
                    else:
                        q = np.rad2deg(trace['state'][:,13+np.asarray(qids)])
                        line = (*signal['error'][:,side].T,q[:,0])
                        for j,style in ((1,'--'),(2,':')):
                            axes[side,3].plot(t,q[:,j],style,color=color,alpha=.65)
                    for ax,y in zip(axes[side],line):
                        ax.plot(t,y,color=color,label=labels[name])
                for ax,title in zip(axes[side],metrics):
                    styled(ax,f'{limb.upper()} {title}')
            pdf.savefig(fig)
            plt.close(fig)
        fig, axes = plt.subplots(2,2,figsize=(13,8),layout='constrained')
        for side,limb in enumerate('LR'):
            for (name,trace),color in zip(traces.items(),COLORS):
                loaded = trace['contact'][:,side]
                s, pos = signals[name], trace['foot'][..., :2]
                for xy, style in ((pos[:,side],'-'),(s['heel_xy'][:,side],'--'),(s['toe_xy'][:,side],':')):
                    poly = np.where(loaded[:,None],xy,np.nan)
                    axes[side,0].plot(poly[:,0],poly[:,1],style,color=color)
                if name != 'reference':
                    axes[side,1].scatter(s['force'][:,side],1000*s['speed'][:,side],s=5,color=color,label=labels[name])
            axes[side,0].set(title=f'{limb} loaded sole / heel / toe paths (m)',aspect='equal')
            axes[side,1].set(title=f'{limb} contact vs slip',xlabel='Vertical force (N)',ylabel='Sole speed (mm/s)')
            axes[side,1].legend(fontsize=8)
            for ax in axes[side]:
                ax.grid(alpha=.22)
        pdf.savefig(fig)
        plt.close(fig)


def top_view(target, motion, traces, signals, labels):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    axes_local = np.asarray(foot_axes()[0])
    profiles = sole_profiles(motion)
    distances = {name:cumulative_slip(signals[name]['speed'],trace['contact']) for name,trace in traces.items()}
    fig, axs = plt.subplots(2,3,figsize=(19.2,10.8),dpi=100,layout='constrained')
    flat = axs.ravel()
    points = motion.foot_p.numpy().reshape(-1,3)
    flat[-1].axis('off')
    key = ('Solid: measured collision outlines', 'Dashed: reference collision outlines',
           'Filled center: actual contact ≥10 N', 'Arrow: foot heading', 'Slip: distance in current support')
    flat[-1].text(.06,.9,chr(10).join(key),va='top',fontsize=18)
    writer = FFMpegWriter(fps=25,codec='libx264',extra_args=['-pix_fmt','yuv420p','-threads','4'])
    with writer.saving(fig,target,100):
        for frame in range(0,motion.num_frames,2):
            fig.suptitle(f'Top view | {motion.clip_name} | {frame*CONTROL_DT:.2f} s',fontsize=23)
            for ax,(name,trace) in zip(flat,traces.items()):
                ax.clear()
                i = min(frame,len(trace['frame'])-1)
                for side,color in enumerate(FEET):
                    for candidate,index,style in ((traces['reference'],frame,'--'),(trace,i,'-')):
                        p,R = candidate['foot'][index,side],candidate['foot_R'][index,side]
                        for polygon in profiles[side]:
                            xy = polygon@R.T+p
                            ax.plot(xy[:,0],xy[:,1],style,color=color,alpha=.65)
                        arrow = R@axes_local[side]
                        ax.arrow(*p[:2],*(.14*arrow[:2]),width=.0015,color=color,linestyle=style,alpha=.65)
                    p = trace['foot'][i,side]
                    loaded = trace['contact'][i,side]
                    ax.scatter(*p[:2],facecolor=color if loaded else 'white',edgecolor=color,s=42)
                    s = signals[name]
                    ax.text(.02,.92-.08*side,f'{"LR"[side]}  {s["error"][i,side,0]:+.1f}°   '
                            f'{s["omega"][i,side]:+.2f} rad/s   {distances[name][i,side]:.1f} mm',
                            transform=ax.transAxes,color=color,fontsize=13)
                ax.set(xlim=(points[:,0].min()-.20,points[:,0].max()+.22),
                       ylim=(points[:,1].min()-.19,points[:,1].max()+.19),title=labels[name],aspect='equal')
                ax.grid(alpha=.2)
            writer.grab_frame()
    plt.close(fig)


def run(args):
    import matplotlib
    matplotlib.use('Agg')

    info, arrays = load_archive(args.out/'cuda_evaluation.npz')
    winners = info.get('winners', {})
    if set(winners) != {'a','b','c'}:
        raise ValueError('需要完成 M5.0 与 A/B/C 的独立 CUDA 终评')
    labels = dict(reference='Reference',m50='M5.0 #5500')
    labels.update({winners[k]:f'M5.1-{k.upper()} #{winners[k][1:]}' for k in 'abc'})
    kin = RobotKinematics('cpu')
    item = next(x for x in read_manifest(args.library)['train'] if x['clip_id'] == args.clip)
    motion = load_clip(item,kin,'cpu',args.reference_root,'train')
    guide = read_guide_heading(args.reference_root/item['file_path'])
    mdp = task(kin,motion,torch.load(SOURCE,map_location='cpu',weights_only=False,mmap=True))
    reference = motion.at_time(motion.control_time)
    state = torch.cat([reference[k] for k in ('root_p','root_quat','root_v','root_w','q','qd')],-1).numpy()
    traces = dict(reference=dict(frame=np.arange(motion.num_frames),state=state,foot=motion.foot_p.numpy(),
                                  foot_R=motion.foot_R.numpy(),foot_velocity=motion.foot_v.numpy(),
                                  contact=motion.contact.numpy(),force=np.zeros_like(motion.foot_p.numpy())))
    rows = {}
    for name in list(labels)[1:]:
        row = rows[name] = info['cases'][name]['seen'][args.clip][str(args.seed)]
        p = row['prefix']
        traces[name] = {key:arrays[p+key] for key in (*TRACE_FIELDS,'foot_angular_velocity') if p+key in arrays}
    signals = {name:trace_signals(trace,mdp,guide) for name,trace in traces.items()}
    curves(args.out/f'{args.clip}_foot_contact.pdf',traces,signals,labels)
    with TemporaryDirectory(dir=args.out) as scratch:
        directory = Path(scratch)
        def panels(names, _seed, _info, _arrays, _reference):
            return [dict(state=traces[name]['state'][:,:13+len(JOINT_NAMES)],feet=traces[name]['foot'],
                         title=labels[name],row=dict(rows[name],whole_body=rows[name]['body']) if name in rows else None)
                    for name in names]
        sections = [(f'足部跟踪 · {label}',args.seed,list(labels),azimuth)
                    for label,azimuth in (('正面',180),('侧面',90))]
        render_video(info,arrays,motion,directory/'views.mp4',sections=sections,
                     panels_from=panels,banner=f'Roban {args.clip}',show_hand_errors=True)
        top_view(directory/'top.mp4',motion,traces,signals,labels)
        index = directory/'clips.txt'
        index.write_text(chr(10).join(('file views.mp4','file top.mp4','')))
        target = directory/'comparison.mp4'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-f','concat','-safe','1','-i',str(index),
                        '-c','copy','-movflags','+faststart',str(target)],check=True)
        target.replace(args.out/f'{args.clip}_foot_contact.mp4')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUTPUT)
    parser.add_argument('--reference-root',type=Path,default=REFERENCE_ROOT)
    parser.add_argument('--library',type=Path,default=MANIFEST)
    parser.add_argument('--clip',choices=('F1','F2','F3'),default='F2')
    parser.add_argument('--seed',type=int,default=1001)
    torch.set_num_threads(2)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
