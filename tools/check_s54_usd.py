"""Compare USD joint frames/axes/limits with compiled Leju-GMR MJCF, not names alone."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation
from inspect_s54_model import inspect_usd
from supermimic.s54 import model_contract, JOINT_NAMES, TRACKED_BODIES, USD_PATH


def rot(q): return Rotation.from_quat(q).as_matrix()


def check(path):
    usd=inspect_usd(path); m,rows=model_contract(); joints={j['name']:j for j in usd['joints']}
    errors=[];details=[]
    if usd['up_axis']!='Z' or abs(usd['meters_per_unit']-1)>1e-9:errors.append('Expected Z-up metre units')
    if set(joints) != set(JOINT_NAMES): errors.append(dict(missing=sorted(set(JOINT_NAMES)-set(joints)),extra=sorted(set(joints)-set(JOINT_NAMES))))
    if not all(n in usd['bodies'] for n in TRACKED_BODIES): errors.append('Missing tracked rigid bodies')
    if not any(p.endswith('/base_link') or p.endswith('/floating_base_joint') for p in usd['roots']): errors.append('Articulation root must identify floating base_link')
    if not any('mujoco' in v.get('Physics',[]) for v in usd['variants'].values()): errors.append('No Physics=mujoco variant; asset is not validated for this Newton configuration')
    for name,props in usd['body_properties'].items():
        bid=next((b for b in range(1,m.nbody) if m.body(b).name==name),None)
        if bid is None:continue
        if props['mass'] is None or abs(props['mass']-m.body_mass[bid])>1e-5:
            errors.append(dict(body=name,usd_mass=props['mass'],mjcf_mass=float(m.body_mass[bid])))
        if props['center_of_mass'] is None or np.linalg.norm(np.array(props['center_of_mass'])-m.body_ipos[bid])>1e-5:
            errors.append(dict(body=name,usd_com=props['center_of_mass'],mjcf_com=m.body_ipos[bid].tolist()))
        if props['principal_axes_xyzw'] is None or props['diagonal_inertia'] is None:
            errors.append(dict(body=name,missing='explicit inertia tensor'));continue
        ur=rot(props['principal_axes_xyzw']);mr=rot(np.roll(m.body_iquat[bid],-1))
        ui=ur@np.diag(props['diagonal_inertia'])@ur.T;mi=mr@np.diag(m.body_inertia[bid])@mr.T
        if np.max(np.abs(ui-mi))>1e-5:
            errors.append(dict(body=name,inertia_tensor_max_error=float(np.max(np.abs(ui-mi)))))
    for row in rows:
        n=row['name']
        if n not in joints: continue
        j=joints[n]; jid=row['mj_joint_id']; b=int(m.jnt_bodyid[jid]);parent=m.body(int(m.body_parentid[b])).name
        R0,R1=rot(j['localRot0']),rot(j['localRot1']); R=R0@R1.T
        offset=np.array(j['localPos0'])-R@np.array(j['localPos1'])
        axis=R1 @ np.eye(3)['XYZ'.index(j['axis'])]
        expectedR=Rotation.from_quat(np.roll(m.body_quat[b],-1)).as_matrix()
        detail=dict(name=n, parent_ok=bool(j['body0'] and j['body0'][0].split('/')[-1]==parent),
                    child_ok=bool(j['body1'] and j['body1'][0].split('/')[-1]==m.body(b).name),
                    offset_error_m=float(np.linalg.norm(offset-m.body_pos[b])),
                    anchor_error_m=float(np.linalg.norm(np.array(j['localPos1'])-m.jnt_pos[jid])),
                    rotation_error_rad=float(Rotation.from_matrix(R.T@expectedR).magnitude()),
                    axis_error=float(np.linalg.norm(axis-m.jnt_axis[jid])),
                    range_error_rad=float(np.max(np.abs(np.deg2rad([j['lower_deg'],j['upper_deg']])-m.jnt_range[jid]))))
        if not (detail['parent_ok'] and detail['child_ok'] and detail['offset_error_m']<1e-5
                and detail['anchor_error_m']<1e-5 and detail['rotation_error_rad']<1e-5
                and detail['axis_error']<1e-5 and detail['range_error_rad']<1e-5): errors.append(detail)
        details.append(detail)
    return dict(path=str(path),**{'pass':not errors},errors=errors, joints=details,
                articulation_roots=usd['roots'],variants=usd['variants'],dynamic_backend_check='NOT RUN')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--usd',type=Path,default=USD_PATH);p.add_argument('--out',type=Path,default=Path('results/s54/usd_alignment.json'))
    a=p.parse_args();r=check(a.usd);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='joints'},indent=2));sys.exit(0 if r['pass'] else 1)
