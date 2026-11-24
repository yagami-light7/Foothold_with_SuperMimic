"""从冻结诊断归档生成 M4.5.1 的配对曲线和一份对照视频。"""

import argparse
import json
import os
from pathlib import Path

if __name__ == '__main__':
    os.environ['SUPERMIMIC_ROBOT'] = 'roban'
    os.environ.setdefault('MUJOCO_GL','egl')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..kinematics import RobotKinematics
from .m42v3_rollout import render_comparison
from .m45_upper import ARMS
from .m4_reference import F2MotionReference


FACTORS = ('xy','height','roll_pitch','yaw','joint_position','root_velocity','joint_velocity','combined')
NAMES = {'m43_best':'M4.3 #4500','m43_final':'M4.3 #5000','m45_4500':'M4.5 #4500','m45_final':'M4.5 #5000'}
AUDIT_NAMES = {'m43_4500':'M4.3 #4500','m43_5000':'M4.3 #5000','m45_4500':'M4.5 #4500','m45_5000':'M4.5 #5000'}
DATA = ('baseline_success','prior_success','prior_fall')
DATA_NAMES = ('M4.3 success','M4.5 success','M4.5 falls')


def evidence_plot(perturb, audit, output):
    fig, axes = plt.subplots(3,2,figsize=(13,10),layout='constrained')
    trials = {name:perturb['models'][name]['trials'] for name in ('m43_best','m45_final')}
    x = np.arange(len(FACTORS))
    for i,(model, data) in enumerate(trials.items()):
        rows = [data[f'{k}_x1']['aggregate'] for k in FACTORS]
        axes[0,0].bar(x+(i-.5)*.35,[r['full']/r['n'] for r in rows],width=.35,label=NAMES[model])
        for j, factor in enumerate(perturb['risk_factors']):
            rows = [data[f'{factor}_scan_x{s:g}']['aggregate'] for s in perturb['scales']]
            axes[0,1].plot(perturb['scales'],[r['full']/r['n'] for r in rows],marker=('o','s')[i],
                           ls=('-', '--', ':')[j],label=f'{NAMES[model]} {factor}',color=('tab:blue','tab:orange')[i])
    axes[0,0].set_xticks(x,FACTORS,rotation=28,ha='right')
    axes[0,0].set_title('Complete four steps | paired 1101–1106')
    axes[0,1].set_title('Amplitude | paired 1107–1112')
    axes[0,1].set_xlabel('Perturbation scale')
    for ax in axes[0]:
        ax.set_ylim(-.03,1.05)
        ax.set_ylabel('Success fraction')
        ax.legend(fontsize=7)

    bptt = audit['bptt']['prior_fall']
    phases = list(bptt['m45_5000'])
    for label, data in bptt.items():
        axes[1,0].plot(phases,[data[p].get('body_foot') if p in data else None for p in phases],'.-',label=AUDIT_NAMES[label])
    axes[1,0].tick_params(axis='x',rotation=22)
    axes[1,0].set_title('Body/foot gradient cosine | matched failed starts')
    axes[1,0].axhline(0,color='.6',lw=.7)
    axes[1,0].legend(fontsize=7)
    for label in AUDIT_NAMES:
        stats = [audit['prediction'][dataset][label]['8']['all'] for dataset in DATA]
        for ax,key,title in ((axes[1,1],'foot_m','WM 8-step foot RMSE (mm)'),(axes[2,0],'root_height_m','WM 8-step root-height error (mm)')):
            ax.plot(DATA_NAMES,[row[key]*1000 for row in stats],'.-',label=AUDIT_NAMES[label])
            ax.set_title(title)
    for label in ('m43_4500','m45_5000'):
        for term in ('foot','body'):
            values = [audit['physics'][dataset]['finite_difference'][label][term]['all'].get('sign') for dataset in DATA]
            axes[2,1].plot(DATA_NAMES,values,'.-',label=f'{AUDIT_NAMES[label]} {term}')
    axes[2,1].set_ylim(-.03,1.05)
    axes[2,1].set_title('WM vs Newton gradient sign | informative slopes')
    axes[1,1].legend(fontsize=7)
    axes[2,1].legend(fontsize=7)
    for ax in axes.flat:
        ax.grid(axis='y',alpha=.2)
    fig.savefig(output/'robustness.png',dpi=145)
    plt.close(fig)


def read_trace(record, prefix):
    return {key:record[prefix+key] for key in ('state','time','foot','force','contact','termination')}


def trajectory_visuals(output, reference, seed, nominal_seed):
    motion = F2MotionReference(reference,RobotKinematics('cpu'))
    t = motion.control_time.numpy()
    ref = motion.at_time(t)
    with np.load(output/'repro_rollouts.npz',allow_pickle=False) as old, np.load(output/'perturb_rollouts.npz',allow_pickle=False) as new:
        archive = {'seeds':np.asarray([nominal_seed])}
        nominal = [(f'm43_4500_independent_r1_s{nominal_seed}_',f'M4.3 #4500 nominal | seed {nominal_seed}'),
                   (f'm45_5000_independent_r1_s{nominal_seed}_',f'M4.5 #5000 nominal | seed {nominal_seed}')]
        combined = [(f'{label}_combined_x1_s{seed}_',f'{NAMES[label]} combined x1 | seed {seed}')
                    for label in ('m43_best','m45_4500','m45_final')]
        fig,ax = plt.subplots(3,2,figsize=(12,9),layout='constrained')
        for source, selections in ((old,nominal),(new,combined)):
            for prefix,label in selections:
                x = read_trace(source,prefix)
                archive.update({prefix+k:v for k,v in x.items()})
                time, state = x['time'],x['state']
                if source is old:
                    ax[2,1].plot(time,x['force'][:,1,2],label=label)
                    continue
                n = time <= 1.8+1e-8
                delta = state[:,13:34]-ref['q'].numpy()[:len(time)]
                for canvas,value in ((ax[0,0],state[:,2]),(ax[1,0],x['force'][:,1,2]),(ax[2,0],x['foot'][:,0,2]),
                                     (ax[0,1],np.sqrt((delta[:,ARMS]**2).mean(-1))),(ax[1,1],abs(delta[:,0]))):
                    canvas.plot(time[n],value[n],label=label)
        ax[0,0].plot(t[t<=1.8],ref['root_p'].numpy()[t<=1.8,2],'k--',lw=.9,label='F2')
        ax[2,0].plot(t[t<=1.8],ref['foot_p'].numpy()[t<=1.8,0,2],'k--',lw=.9,label='F2')
        for canvas,title in zip(ax.flat,('Root height (m)','Arm error (rad)','Right foot force (N)','Waist error (rad)',
                                        'Left sole height (m)','Right foot force: fourth touchdown (N)')):
            canvas.set_title(title)
            canvas.set_xlabel('F2 time (s)')
            canvas.grid(alpha=.2)
            if canvas is not ax[2,1]:
                canvas.axvspan(.8,1.6,color='.85',alpha=.5)
        ax[1,0].axhline(10,color='.5',ls='--',lw=.8)
        ax[2,1].set_xlim(4.35,4.85)
        ax[2,1].axvline(4.6,color='.6',lw=.9)
        ax[2,1].axhline(10,color='.5',ls='--',lw=.8)
        ax[0,0].legend(fontsize=7)
        ax[2,1].legend(fontsize=7)
        fig.savefig(output/'contact_and_posture.png',dpi=145)
        plt.close(fig)
    pairs = [(nominal[0],nominal[1],False), *[(None,row,False) for row in combined], (combined[0],combined[-1],True)]
    render_comparison(archive,reference,output/'robustness_comparison.mp4',paired_chapters=pairs,robustness_overlay=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m4_reference',required=True,type=Path)
    parser.add_argument('--out',default=Path('outputs/m4_5_1'),type=Path)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--nominal-seed',type=int,default=1001)
    args = parser.parse_args()
    torch.set_num_threads(4)
    evidence_plot(json.loads((args.out/'perturb_results.json').read_text()),json.loads((args.out/'audit.json').read_text()),args.out)
    trajectory_visuals(args.out,args.m4_reference,args.seed,args.nominal_seed)


if __name__ == '__main__':
    main()
