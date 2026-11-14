"""Read-only S54 reference QA, clip-level train/held-out manifest and paired FK export."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import subprocess

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from supermimic.kinematics import S54Kinematics
from supermimic.motion import MotionLib,CONTROL_DT
from supermimic.reference_selection import select_reference_files
from supermimic.motion_prior_data import ContactConfig,extract_reference,sha256
from supermimic.s54 import checkpoint_contract,JOINT_NAMES
from supermimic.reference_foot import ReferenceFootGoalSource
from supermimic.goals import GoalSpec
from supermimic.foot_trajectory import AlternatingConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion_dir',default='library/motion_reference/kuavo_s54/retarget_data/reference_startup_fixed')
    p.add_argument('--clips',nargs='+',default=['walk*','dance*'])
    p.add_argument('--heldout',nargs='+',default=['dance1_subject1'])
    p.add_argument('--seed',type=int,default=54)
    p.add_argument('--contact_config')
    p.add_argument('--out',default='results/s54/motion_prior_stage0')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    files=select_reference_files(a.motion_dir,a.clips)
    heldout=set(select_reference_files(a.motion_dir,a.heldout))
    if not heldout.issubset(files): p.error('Held-out selection must be included in --clips')
    if len(heldout)==len(files): p.error('At least one disjoint train clip is required')
    cfg=ContactConfig(**(json.loads(Path(a.contact_config).read_text()) if a.contact_config else {}))
    out=Path(a.out)
    if not a.dry_run:
        if out.exists(): p.error('Output exists; choose a fresh QA directory')
        out.mkdir(parents=True)
    kin=S54Kinematics('cpu');clips=[];data={};seen={}
    for path in files:
        source=Path(path).resolve();meta=json.loads(source.with_suffix('.json').read_text())
        split='heldout' if path in heldout else 'train'; identity=meta.get('raw_sha256',meta['csv_sha256'])
        if identity in seen and seen[identity]!=split: raise ValueError('Raw source clip leaks across splits')
        seen[identity]=split
        entry=dict(name=source.stem,path=str(source),sidecar=str(source.with_suffix('.json')),
                   csv_sha256=sha256(source),sidecar_sha256=sha256(source.with_suffix('.json')),
                   source_identity=identity,original_path=meta.get('raw_csv'),source_fps=meta['fps'],
                   quat_source=meta['quat'],quat_internal='wxyz',joint_names=JOINT_NAMES,split=split)
        try:
            motion=MotionLib([str(source)],kin,'cpu');d=extract_reference(motion,kin,0,cfg)
            n=len(d['time']);dt=np.diff(d['time'])
            entry.update(status='PASS',reason='S54 loader, certificate, checksum, limits, FK and time-grid passed',
                source_frames=meta['frames'],control_frames=n,duration_s=float(d['time'][-1]),control_dt=CONTROL_DT,
                max_time_error_s=float(abs(dt-CONTROL_DT).max()),contact_fraction=d['contact'].mean(0).tolist(),
                estimated_landings=(np.diff(d['contact'].astype(int),axis=0)==1).sum(0).tolist(),
                foot_gap_min_m=d['sole_gap'].min(0).tolist(),
                root_speed_p95_m_s=float(np.quantile(np.linalg.norm(d['root_v_local'],axis=-1),.95)),
                max_joint_step_rad=float(abs(np.diff(d['q'],axis=0)).max()))
            data[source.stem]=d
            if not a.dry_run:
                np.savez_compressed(out/(source.stem+'_paired.npz'),**d)
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                limit=min(n,1000);t=d['time'][:limit]
                fig,axes=plt.subplots(4,1,figsize=(13,10),sharex=True)
                for side,name in enumerate(('left','right')):
                    axes[0].plot(t,d['sole_gap'][:limit,side],label=name)
                    axes[1].plot(t,np.linalg.norm(d['foot_v'][:limit,side],axis=-1),label=name)
                    axes[2].step(t,d['contact'][:limit,side].astype(float)+side*1.2,label=name)
                axes[3].plot(t,d['root_p'][:limit,2],label='root height')
                for ax,title in zip(axes,['Sole gap (m)','Foot speed (m/s)','Estimated contact (offset R)','Root height (m)']):
                    ax.set_ylabel(title);ax.grid(alpha=.2);ax.legend()
                axes[0].axhline(cfg.enter_height,ls='--',c='gray');axes[-1].set_xlabel('s')
                fig.tight_layout();fig.savefig(out/(source.stem+'_contacts.png'),dpi=130);plt.close(fig)
        except (ValueError,FileNotFoundError) as exc: entry.update(status='FAIL',reason=str(exc))
        clips.append(entry)
    report=dict(version=1,robot_contract=checkpoint_contract(),seed=a.seed,
                git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                contact_config=asdict(cfg),clips=clips,
                missing_categories=[k for k in ('stand','turn','slow_walk','start_stop') if not any(k in e['name'] for e in clips)],
                scope='Geometric/dataset QA; not Newton stability or measured contact-force certification')
    if not a.dry_run:
        (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        # A small reproducible pairing example proves both interfaces share the same current time.
        train=[e['path'] for e in clips if e['split']=='train' and e['status']=='PASS']
        if train:
            m=MotionLib(train,kin,'cpu');g=ReferenceFootGoalSource(m,kin,GoalSpec('reference_foot'),AlternatingConfig(),cfg)
            torch.manual_seed(a.seed);f=torch.randint(0,int(m.clip_end[0])-50,(16,));S=m.ref_state(f)
            cmd=g.new_command(S,frames=f);step=torch.zeros(16,dtype=torch.long)
            np.savez_compressed(out/'paired_goal_samples.npz',clip_id=m.clip_id[f].numpy(),frame_id=f.numpy(),
                control_time=(f*CONTROL_DT).numpy(),motion_goal=m.goal(f,10).numpy(),
                foot_goal=g.goal(S,f,step,cmd).numpy(),phase=g.phase[f].numpy())
            from supermimic.upper_prior import build_template_bank
            build_template_bank(out/'manifest.json',data,out/'upper_templates.npz')
    print(json.dumps(report,indent=2))
    if any(e['status']!='PASS' for e in clips): sys.exit(1)


if __name__=='__main__':main()
