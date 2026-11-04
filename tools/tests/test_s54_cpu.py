"""Independent native MuJoCo FK/velocity, motion, gradients, collisions, WM/BPTT tests.

This is a CPU diagnostic, NOT an Isaac Lab/Newton training/evaluation substitute.
"""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from supermimic.s54 import *
from supermimic.kinematics import S54Kinematics
from supermimic.motion import MotionLib
from supermimic.mdp import (MDP, OBS_DIM, STATE_DIM, CRITIC_EXTRA, INV_DIM, flatten_state,
                            unflatten_state, normalize_xi, MAX_STEP_REWARD)
from supermimic.networks import WorldModel, Policy, Critic
from supermimic.rotations import quat_to_mat


def fk_test(kin,motion,samples):
    rng=np.random.default_rng(54); m,rows=model_contract();d=mujoco.MjData(m)
    q=rng.uniform(kin.q_lo.numpy(),kin.q_hi.numpy(),(samples,NUM_J))
    p=rng.normal(size=(samples,3));R=Rotation.random(samples,random_state=rng).as_matrix()
    v=rng.normal(size=(samples,3));w=rng.normal(size=(samples,3));qd=rng.normal(size=(samples,NUM_J))
    # Half of samples are actual reference frames; the rest span the full legal pose range.
    count=samples//2;frames=torch.linspace(0,motion.num_frames-1,count).long()
    for dest,src in zip((p,R,v,w,q,qd),motion.ref_state(frames)): dest[:count]=src.numpy()
    S=tuple(torch.tensor(x,dtype=torch.float32) for x in (p,R,v,w,q,qd))
    got=kin.forward(*S,all_bodies=True)
    expected=[[],[],[],[]]
    for t in range(samples):
        d.qpos[:3]=p[t];d.qpos[3:7]=np.roll(Rotation.from_matrix(R[t]).as_quat(),1)
        d.qvel[:3]=v[t];d.qvel[3:6]=R[t].T @ w[t]
        for r in rows:
            d.qpos[r['qposadr']]=q[t,r['canonical_index']];d.qvel[r['dofadr']]=qd[t,r['canonical_index']]
        mujoco.mj_forward(m,d)
        bv=[];bw=[]
        for b in range(1,m.nbody):
            jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,b)
            bv.append(jp @ d.qvel);bw.append(jr @ d.qvel)
        for arr,val in zip(expected,(d.xpos[1:].copy(),d.xmat[1:].reshape(-1,3,3).copy(),bv,bw)): arr.append(val)
    expected=list(map(np.array,expected));actual=[x.detach().numpy().astype(float) for x in got]
    pos=float(np.linalg.norm(actual[0]-expected[0],axis=-1).max())
    relative=np.swapaxes(expected[1],-1,-2) @ actual[1]
    angle=float(Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude().max())
    vel=float(np.linalg.norm(actual[2]-expected[2],axis=-1).max())
    angvel=float(np.linalg.norm(actual[3]-expected[3],axis=-1).max())
    assert pos<1e-4 and angle<1e-3 and vel<1e-4 and angvel<1e-4,(pos,angle,vel,angvel)
    return dict(samples=samples,all_bodies=kin.num_bodies,max_pos_m=pos,max_rot_rad=angle,max_linvel_m_s=vel,max_angvel_rad_s=angvel)


def contact_test():
    root=ET.parse(MJCF_PATH).getroot();root.find('compiler').set('meshdir',str(SOURCE_MJCF.parent.parent/'meshes'))
    for geom in root.findall('.//geom'):
        if geom.get('contype')=='1': geom.set('contype','1');geom.set('conaffinity','2')
    ET.SubElement(root.find('worldbody'),'geom',name='test_floor',type='plane',size='0 0 0.01',contype='2',conaffinity='1')
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));m.opt.timestep=.002
    d=mujoco.MjData(m);d.qpos[2]=1.2;mujoco.mj_forward(m,d)
    start=float(d.qpos[2]);feet=set();max_force=0.;min_height=start
    for _ in range(600):
        mujoco.mj_step(m,d);min_height=min(min_height,float(d.qpos[2]))
        for i in range(d.ncon):
            c=d.contact[i];bodies=[m.body(int(m.geom_bodyid[g])).name for g in (c.geom1,c.geom2)]
            if 'world' in bodies:
                feet.update(b for b in bodies if b in ('leg_l6_link','leg_r6_link'))
                force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force);max_force=max(max_force,float(np.linalg.norm(force[:3])))
    assert np.isfinite(d.qpos).all() and start-min_height>.05 and len(feet)==2 and max_force>0
    return dict(backend='native MuJoCo CPU only',feet_contacted=sorted(feet),max_normal_and_tangent_force_N=max_force,root_drop_m=start-min_height)


def core_test(kin,motion):
    torch.manual_seed(54)
    mdp=MDP(kin,motion,torch.tensor(default_joint_pos_list()))
    frames=torch.linspace(0,max(0,motion.num_frames-10),4).long()
    frames=torch.maximum(motion.clip_start[motion.clip_id[frames]],
                         torch.minimum(frames,motion.clip_end[motion.clip_id[frames]]-8))
    S=motion.ref_state(frames)
    a=torch.zeros(len(frames),NUM_J);xi=torch.zeros(len(frames),XI_DIM);xi[:,0]=1
    o=mdp.obs(S,a,xi,frames);g=motion.goal(frames,2)
    assert o.shape==(4,OBS_DIM) and g.shape==(4,2*GOAL_FRAME_DIM)
    flat=flatten_state(S);roundtrip=unflatten_state(flat)
    assert flat.shape[-1]==STATE_DIM
    rt=float(max((x-y).abs().max() for x,y in zip(S,roundtrip)));assert rt<1e-5
    perm=list(reversed(JOINT_NAMES));c2s,s2c=name_mapping(perm)
    assert torch.equal(S[4][:,s2c][:,c2s],S[4])
    try: assert_checkpoint({'robot_contract':{'robot':'g1'}})
    except ValueError: pass
    else: raise AssertionError('G1 checkpoint accepted')
    # Meaningful backpropagation through actual full-size networks, FK and analytic reward.
    # Targets are reference kinematics here; these are NOT claimed as simulator training data.
    wm=WorldModel();pi=Policy(OBS_DIM,g.shape[-1],hist=2);critic=Critic(INV_DIM+CRITIC_EXTRA+g.shape[-1])
    xn=normalize_xi(xi);target=motion.ref_state(frames+1)
    wm.in_norm.update(wm.inputs(S,a,xn));wm.out_norm.update(wm.targets(S,target))
    wp=torch.optim.Adam(wm.parameters(),lr=2e-4)
    hat=wm.step(S,a,xn);loss=wm.state_error(hat,target).square().mean()
    # Zero-output initialization predicts mean increment; mixed frames ensure nonzero error.
    wp.zero_grad();loss.backward();wg=torch.nn.utils.clip_grad_norm_(wm.parameters(),1,error_if_nonfinite=True);wp.step()
    for p in wm.parameters(): p.requires_grad_(False)
    for p in critic.parameters(): p.requires_grad_(False)
    pi.obs_norm.update(o);pi.goal_norm.update(g)
    hist=o[:,None].repeat(1,2,1);sh=S;prev=a;f=frames.clone();survival=torch.ones(4);J=torch.zeros(4)
    opt=torch.optim.Adam(pi.parameters(),lr=2e-4)
    for t in range(8):
        act=pi(hist,motion.goal(f,2));sh=wm.step(sh,act,xn);f=torch.minimum(f+1,motion.clip_end[motion.clip_id[f]])
        rew=mdp.reward(sh,f,act,prev);J=J+.99**t*survival*rew
        survival=survival*(1-mdp.soft_termination(sh,f));prev=act
        hist=torch.cat((hist[:,1:],mdp.obs(sh,act,xi,f)[:,None]),1)
    X=torch.cat((mdp.critic_features(sh,f,xn),motion.goal(f,2)),-1)
    assert X.shape[-1]==INV_DIM+CRITIC_EXTRA+g.shape[-1]
    J=J+.99**8*survival*critic(X)
    pl=-J.mean();opt.zero_grad();pl.backward();pg=torch.nn.utils.clip_grad_norm_(pi.parameters(),1,error_if_nonfinite=True);opt.step()
    for p in critic.parameters():p.requires_grad_(True)
    vc=torch.optim.Adam(critic.parameters(),lr=2e-4);critic.norm.update(X.detach());vl=(critic(X.detach())-J.detach()).square().mean()
    vc.zero_grad();vl.backward();vg=torch.nn.utils.clip_grad_norm_(critic.parameters(),1,error_if_nonfinite=True);vc.step()
    assert all(torch.isfinite(x) for x in (loss,pl,vl,wg,pg,vg)) and wg>0 and pg>0
    return dict(state_dim=STATE_DIM,obs_dim=OBS_DIM,goal_frame_dim=GOAL_FRAME_DIM,xi_dim=XI_DIM,
                critic_dim=X.shape[-1],roundtrip_max_error=rt,wm_loss=float(loss.detach()),pi_loss=float(pl.detach()),v_loss=float(vl.detach()),
                wm_gradnorm=float(wg),pi_gradnorm=float(pg),v_gradnorm=float(vg),imagined_horizon=8,
                warning='CPU one-update gradient/shape diagnostic with reference increments; not Newton training or evaluation')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,default=128);ap.add_argument('--skip_core',action='store_true')
    ap.add_argument('--motion_dir',default='data/retarget_data/reference_safe')
    ap.add_argument('--out',type=Path,default=Path('results/s54/cpu_tests.json'));a=ap.parse_args()
    torch.set_num_threads(4)
    kin=S54Kinematics();motion=MotionLib.from_dir(a.motion_dir,kin,'cpu')
    assert motion.num_frames>=2 and motion.q.shape==(motion.num_frames,NUM_J)
    source_frames=sum(json.loads(p.with_suffix('.json').read_text())['frames'] for p in Path(a.motion_dir).glob('*.csv'))
    report=dict(motion=dict(source_frames=source_frames,clips=motion.names,source_fps=SOURCE_FPS,control_frames=motion.num_frames,control_fps=50),
                fk=fk_test(kin,motion,a.samples),contacts=contact_test())
    if not a.skip_core:report['core']=core_test(kin,motion)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
