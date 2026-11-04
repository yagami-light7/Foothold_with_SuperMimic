"""Apply predeclared gates to bounded results; writes evidence, never launches training."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse
import json
from pathlib import Path


def teacher_acceptance(summary,protocol,manifest):
    cfg=protocol['teacher'];rows=summary['rows']
    mapping={k:k for k in ('body_position_world_rmse_m','body_orientation_world_rmse_deg',
        'root_orientation_tracking_rmse_deg','root_height_tracking_max_m','arm_pose_rmse_rad',
        'arm_velocity_rmse_rad_s','joint_pose_rmse_rad','joint_limit_violation_fraction')}
    checks={k:all(r['metrics'].get(k,float('inf'))<=cfg[v] for r in rows) for k,v in mapping.items()}
    checks.update(all_timeouts=all(r['n_failed']==0 and r['n_timeout']==1 for r in rows),
        complete=(len(rows)==len(cfg['durations_s'])*len(cfg['seeds'])*len(cfg['evaluation_offsets_s']) and
            {(r['seed'],r['duration_s'],r['start_seconds']) for r in rows}==
            {(seed,duration,offset) for seed in cfg['seeds'] for duration in cfg['durations_s'] for offset in cfg['evaluation_offsets_s']}),
        root_pitch_roll=all(max(r['metrics']['root_pitch_max_abs_deg'],r['metrics']['root_roll_max_abs_deg'])<cfg['root_pitch_roll_max_deg'] for r in rows),
        measured_contacts=all(r['metrics'].get('measured_contact_accuracy',-1)>=cfg['measured_contact_accuracy'] for r in rows),
        loaded_slip=all(r['metrics']['loaded_foot_samples']>0 and r['metrics']['actual_loaded_stance_slip_m_s']<cfg['loaded_stance_slip_m_s'] for r in rows),
        independent_walking_holdout=any(e['split']=='heldout' and e['status']=='PASS' and e['name'].startswith('walk') for e in manifest['clips']))
    low,high=cfg['arm_motion_std_ratio_range'];ratios=[]
    for r in rows:
        m=r['metrics'];reference=m.get('reference_arm_motion_std_rad',0.)
        if reference>.01:ratios.append(m['arm_motion_std_rad']/reference)
    checks['arm_movement_evidence']=bool(ratios) and all(low<=v<=high for v in ratios)
    return dict(qualified=all(checks.values()),checks=checks,arm_motion_std_ratios=ratios,
        support='same training Walking clip at fixed distinct offsets; dance is not Walking holdout',
        missing_categories=manifest['missing_categories'],real_steps=summary['eval_real_steps'],
        episode_durations_s=[sum(r['episode_lengths'])*.02 for r in rows])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--manifest',default='results/s54/motion_prior_stage0_verified/manifest.json')
    p.add_argument('--teacher-evaluation',default='teacher_train_windows');p.add_argument('--out',default='readiness.json')
    p.add_argument('--standing-evaluation',default='standing_before')
    a=p.parse_args();root=Path(a.root);target=root/a.out
    if target.exists():p.error('Existing result protected')
    protocol=json.loads((root/'protocol.json').read_text());wm=json.loads((root/'wm_comparison.json').read_text())
    standing=json.loads((root/a.standing_evaluation/'summary.json').read_text())
    teacher=json.loads((root/a.teacher_evaluation/'summary.json').read_text());manifest=json.loads(Path(a.manifest).read_text())
    acceptance=teacher_acceptance(teacher,protocol,manifest)
    gates=dict(wm_calibration=wm['passed']['after'],
        standing=standing['standing_gates']['all_timeouts'] and standing['standing_gates']['passed'],
        stage1_teacher=acceptance['qualified'],independent_walking_holdout=acceptance['checks']['independent_walking_holdout'])
    for gate,filename in [('engineering_isolation','runtime_isolation_audit.json'),('contact_edge_coverage','contact_edge_audit.json')]:
        path=root/filename
        gates[gate]=path.exists() and json.loads(path.read_text()).get('passed') is True
    # This does not certify engineering gates not represented by these files.
    result=dict(conclusion='READY_FOR_SHORT_ABLATION' if all(gates.values()) else 'BLOCKED',
        ability_gates=gates,teacher=acceptance,wm=wm['worst_seed_horizon8'],
        protocol=str(root/'protocol.json'),formal_training_started=False,
        engineering_requires_separate_audit=True)
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
