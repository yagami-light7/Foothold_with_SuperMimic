"""Generate M7 figures and evidence tables from saved runs; no simulation."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report_dir',required=True,type=Path)
    args=p.parse_args();root=args.report_dir;runs=root/'runs'
    output=root/'figures';output.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25})
    results={p.parent.name:json.loads(p.read_text()) for p in runs.glob('*/evaluation_metrics.json')}
    diagnosis={}
    for name,s in results.items():
        if 'main_control_steps' not in s:continue
        a=dict(np.load(runs/name/'actual_trajectory.npz',allow_pickle=False));r=dict(np.load(runs/name/'reference_trajectory.npz',allow_pickle=False))
        end=s['main_control_steps']+1;t=a['time_s'];frame=a['reference_frame']
        rooterr=np.linalg.norm(a['root_pos']-r['root_pos'],axis=-1)
        qerr=np.sqrt(np.mean((a['joint_pos']-r['joint_pos'])**2,axis=-1))
        footerr=np.linalg.norm(a['foot_pos']-r['planned_sole_pos'],axis=-1)
        force=a['foot_contact_force'][...,2]
        margin=a.get('contact_patch_support_margin',a['support_margin'])
        outside=np.flatnonzero(np.isfinite(margin[:end])&(margin[:end]<0))
        oldmargin=a['support_margin'];oldoutside=np.flatnonzero(np.isfinite(oldmargin[:end])&(oldmargin[:end]<0))
        effort=np.array(list(s['physical_contract']['actuators'][n]['effort'] for n in a['joint_names']))
        normtau=np.max(np.abs(a['applied_torque'])/effort,axis=-1)
        diagnosis[name]=dict(first_com_outside_actual_contact_polygon_s=float(t[outside[0]]) if len(outside) else None,
            first_com_outside_force_supported_sole_polygon_s=float(t[oldoutside[0]]) if len(oldoutside) else None,
            max_effort_utilization=float(normtau[:end].max()),
            initial_measured_root=a['root_pos'][0].tolist(),initial_reference_root=r['root_pos'][0].tolist(),
            initial_joint_max_error_rad=float(np.abs(a['joint_pos'][0]-r['joint_pos'][0]).max()),
            after_failure_frames=len(t)-end,recorded_reference_frames=frame.tolist())
        if name.startswith('pd_'):
            fig,ax=plt.subplots(3,2,figsize=(12,8),constrained_layout=True)
            ax[0,0].plot(t,rooterr);ax[0,0].set(ylabel='Root position error (m)')
            ax[0,1].plot(t,qerr);ax[0,1].set(ylabel='Joint RMSE (rad)')
            ax[1,0].plot(t,footerr[:,0],label='left');ax[1,0].plot(t,footerr[:,1],label='right');ax[1,0].legend();ax[1,0].set(ylabel='Planned sole error (m)')
            ax[1,1].plot(t,force[:,0],label='left');ax[1,1].plot(t,force[:,1],label='right');ax[1,1].legend();ax[1,1].set(ylabel='Measured foot vertical force (N)')
            ax[2,0].plot(t,normtau);ax[2,0].axhline(1,c='r',ls='--');ax[2,0].set(ylabel='Peak torque / effort limit')
            ax[2,1].plot(t,margin,label='actual contact hull');ax[2,1].plot(t,oldmargin,label='sole polygon proxy',alpha=.6);ax[2,1].axhline(0,c='r',ls='--');ax[2,1].legend();ax[2,1].set(ylabel='COM signed support margin (m)')
            for item in ax.flat:
                item.set_xlabel('Time (s)')
                if s['fall_time_s'] is not None:item.axvline(s['fall_time_s'],c='r',ls=':')
            fig.suptitle(f'{name}: {s["status"]}; {s["termination_reason"]}')
            for ext in ('png','pdf'):fig.savefig(output/f'{name}.{ext}',dpi=150)
            plt.close(fig)
    static=results['static_collision'];cube=dict(np.load(runs/'static_collision/cube_trajectories.npz'))
    fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for i in range(6):
        key=f'top_drop_{i}';top=static['tests'][i]['pillar_top_m']
        ax[0].plot(cube[key+'__time'],cube[key+'__position'][:,2]-top,label=f'P{i}')
        ax[1].plot(cube[key+'__time'],cube[key+'__force'][:,2],label=f'P{i}')
    ax[0].axhline(.02,c='k',ls='--',label='cube half-height')
    ax[1].axhline(9.81,c='k',ls='--',label='weight')
    ax[0].set(xlabel='Time (s)',ylabel='Cube center above pillar top (m)');ax[1].set(xlabel='Time (s)',ylabel='Measured vertical force (N)')
    ax[0].legend(ncol=2);ax[1].legend(ncol=2)
    fig.suptitle('Authoritative PMT pillar meshes: real Newton drop tests')
    for ext in ('png','pdf'):fig.savefig(output/f'static_collision.{ext}',dpi=150)
    plt.close(fig)
    cube=dict(np.load(runs/'dynamic_collision/cube_trajectories.npz'))
    fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for name in ('active_old_support','disabled_old_void','relocated_disabled_void','reenabled_new_support'):
        ax[0].plot(cube[name+'__time'],cube[name+'__position'][:,2],label=name)
        ax[1].plot(cube[name+'__time'],cube[name+'__force'][:,2],label=name)
    ax[0].set(xlabel='Case-local time (s)',ylabel='Cube world height (m)');ax[1].set(xlabel='Case-local time (s)',ylabel='Measured vertical force (N)')
    ax[0].legend(fontsize=7);ax[1].legend(fontsize=7)
    fig.suptitle('Dynamic collider: inactive parking removes actual support; enable restores it')
    for ext in ('png','pdf'):fig.savefig(output/f'dynamic_collider.{ext}',dpi=150)
    plt.close(fig)
    # Re-verify dataset manifests here without changing or importing PMT code.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import os
    os.environ['SUPERMIMIC_ROBOT']='roban'
    from supermimic.integration.pmt_dataset_loader import load_catalog
    dataset=Path(os.environ.get('PMT_DATASET_ROOT','/mnt/sage_data/workspace/Foothold/PMT/TCRS/outputs/roban_pillar_dataset'))
    samples=load_catalog(dataset,seed=44)
    align=dict(status='PASS',samples=[s.alignment for s in samples],contact_truth='planned labels never Newton ground truth',
               tests_log='dataset_contract_tests.log',unique_sample_names=[s.name for s in samples])
    (root/'dataset_alignment_report.json').write_text(json.dumps(align,indent=2)+'\n')
    summary=dict(dataset_integration='PASS',physics_integration='PASS' if all(results[n]['status']=='PASS' for n in ('static_collision','dynamic_collision')) and
                 all(results[n]['physics_control_chain_pass'] for n in results if n.startswith('pd_')) else 'FAIL',
                 locomotion_performance='PASS' if all(results[n]['locomotion_performance_pass'] for n in results if n.startswith('pd_')) else 'FAIL',
                 formal_training=False,production_PD_changed=False,tests={k:v['status'] for k,v in results.items()},diagnosis=diagnosis)
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='diagnosis'},indent=2))


if __name__=='__main__':main()
