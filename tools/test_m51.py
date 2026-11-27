"""运行足部数学测试、历史轨迹审计及限于两轮的 CUDA 冒烟。"""

from argparse import ArgumentParser, Namespace
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

os.environ['SUPERMIMIC_ROBOT'] = 'roban'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import torch

from supermimic.integration.m42v3_rollout import EffectiveFriction
from supermimic.integration.m50_evaluation import INVALID_TERMINATIONS, scene_for, trial
from supermimic.integration.m50_reference import MANIFEST, REFERENCE_ROOT, load_clip, read_manifest
from supermimic.integration.m51_audit import ARCHIVE, run_offline, task
from supermimic.integration.m51_foot import EPS, StanceSignals, TERMS, foot_axes, heading_error, stance_signals
from supermimic.integration.m51_metrics import contact_metrics, read_guide_heading
from supermimic.integration.m51_task import OUTPUT, SOURCE, StanceFootRewardConfig
from supermimic.kinematics import RobotKinematics
from supermimic.mdp import unflatten_state
from supermimic.motion import CONTROL_DT
from supermimic.roban import FOOT_BODY_NAMES, JOINT_NAMES, model_contract
from supermimic.rotations import axis_angle_to_mat, mat_to_quat, so3_exp, so3_log
from test_m50 import run_smoke


def unit_tests():
    cfg = StanceFootRewardConfig(stance_yaw_huber=.25, stance_angular_slip=.10)
    angles = lambda xs: torch.tensor(xs, dtype=torch.float64).deg2rad()
    axis = torch.tensor(foot_axes()[0], dtype=torch.float64)
    torch.testing.assert_close(axis, torch.tensor([[1., 0., 0.]]*2, dtype=axis.dtype), atol=0, rtol=0)
    def rotate(yaw):
        return axis_angle_to_mat(axis.new_tensor([0.,0.,1.]), angles(yaw))
    R = rotate([[0, 0], [2.9, -2.9], [8, -8], [13, -13], [23, -23], [179., -179.]])
    ref = rotate([[0, 0]]*5+[[-179,179]])
    y, valid = heading_error(R, ref, axis)
    torch.testing.assert_close(y[-1].rad2deg(), y.new_tensor([-2.,2.]), atol=1e-10, rtol=0.)
    horizontal = rotate(82.)
    torch.testing.assert_close(heading_error(horizontal@R, horizontal@ref, axis)[0], y, atol=1e-12, rtol=0.)
    # 退化投影对 atan2 输入及梯度均为有限值。
    up = axis_angle_to_mat(axis.new_tensor([0.,1.,0.]), angles([[90.,90.]])).requires_grad_()
    degenerate, ok = heading_error(up, ref[:1], axis)
    assert not ok.any() and torch.isfinite(torch.autograd.grad(degenerate.sum(), up)[0]).all()
    s = StanceSignals(y, valid, torch.zeros_like(y), torch.ones_like(valid),
                      torch.ones_like(y), torch.ones_like(y[:,0]))
    values, _ = s.raw_rewards(cfg)
    yaw = values[TERMS[0]]
    assert (yaw[[0,1,-1]] == 0.).all() and (yaw[2:4] > yaw[3:5]).all()
    assert (values[TERMS[1]] == 0.).all()
    plan = torch.tensor([[True, False], [True, False], [False,False]])
    projected = torch.ones_like(plan)
    y1 = angles([[13,30],[13,30],[90,90]]).requires_grad_()
    w = y1.new_tensor([[.60, 3.],[.6, 3.],[1.,1.]]).requires_grad_()
    proxy = y1.new_tensor([[1.,1.],[0.,1.],[1.,1.]]).requires_grad_()
    quality = y1.new_tensor([.4,.4,.4])
    values, _ = StanceSignals(y1, projected, w, plan, proxy, quality).raw_rewards(cfg)
    expected = -.4 * .5/(1.+EPS)
    expected_yaw = y1.new_tensor([expected,expected,0.])
    torch.testing.assert_close(values[TERMS[0]], expected_yaw, atol=1e-12, rtol=0.)
    torch.testing.assert_close(values[TERMS[1]], y1.new_tensor([expected,0.,0.]), atol=1e-12, rtol=0.)
    grad_y, grad_w, grad_proxy = torch.autograd.grad(sum(values.values()).sum(), (y1,w,proxy), allow_unused=True)
    assert grad_proxy is None and grad_y.isfinite().all() and grad_w.isfinite().all()
    assert (grad_y[:2,0] < 0.).all() and grad_w[0,0] < 0 and grad_w[1,0] == 0.
    # 同一足底位置绕水平轴滚动或绕法向扭转，两组只有角速度不同。
    ref_data = dict(foot_R=ref[:1,None], contact=plan[:1,None], contact_valid=projected[:1,None],
                    quality=quality[:1,None])
    actual = ref[:1]
    horizontal_w = y1.new_tensor([[[2.,0.,0.],[0.,2.,0.]]])
    ref_w = torch.zeros_like(horizontal_w)
    for omega, is_slipping in ((horizontal_w, False), (horizontal_w.roll(2,dims=-1), True)):
        signal = stance_signals(actual,omega,ref_data,ref_w,axis,proxy[:1])
        assert (signal.raw_rewards(cfg)[0][TERMS[1]].item() < 0.) == is_slipping
    from supermimic.integration.m410_reward import dead_zone_huber
    for dead, scale in ((cfg.stance_yaw_huber_tolerance,cfg.stance_yaw_huber_sigma), (.10,.50)):
        x = axis.new_tensor([dead-1e-8, dead+1e-8, dead+scale-1e-8, dead+scale+1e-8]).requires_grad_()
        grad = torch.autograd.grad(dead_zone_huber(x, dead, scale)[0].sum(), x)[0]
        assert abs(float(grad[1]-grad[0])) < 1e-6 and abs(float(grad[3]-grad[2])) < 1e-6
    print('奖励数学测试通过，含环绕、零接触、退化投影、Huber 连续和有限梯度', flush=True)
    return dict(passed=True, axes=foot_axes()[0], normalized_epsilon=EPS, dtype='float64')


def independent_fk():
    kin = RobotKinematics('cpu', dtype=torch.float64)
    model, rows = model_contract()
    data = mujoco.MjData(model)
    bodies = [model.body(name).id for name in FOOT_BODY_NAMES]
    ids = [kin.tracked_names.index(name) for name in FOOT_BODY_NAMES]
    joint_q = [row['qposadr'] for row in rows]
    joint_v = [row['dofadr'] for row in rows]
    with np.load(ARCHIVE, allow_pickle=False) as traces:
        samples = torch.from_numpy(np.concatenate([traces[f'5500_{name}_s1102_state'][::11]
                                                    for name in ('F1','F2','F3')])).double()
    p,R,v,w,q,qd = unflatten_state(samples)
    actual = kin.forward(p,R,v,w,q,qd)
    matrix, omega = [], []
    for pos,rot,vel,angular,angles,speed in zip(p,R,v,w,q,qd):
        data.qpos[:3], data.qpos[3:7], data.qpos[joint_q] = pos, mat_to_quat(rot), angles
        data.qvel[:3], data.qvel[3:6], data.qvel[joint_v] = vel, rot.T@angular, speed
        mujoco.mj_forward(model, data)
        matrix.append(data.xmat[bodies].copy().reshape(2,3,3))
        velocities = np.zeros((2,6))
        for i, bid in enumerate(bodies):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, velocities[i], 0)
        omega.append(velocities[:,:3])
    orientation = float(np.abs(actual[1][:, ids].numpy()-matrix).max())
    motion = actual[3][:, ids]
    analytic = float(np.abs(motion.numpy()-omega).max())
    eps = 1e-4
    plus = kin.forward(p+eps*v,so3_exp(w*eps)@R,v,w,q+eps*qd,qd)[1][:, ids]
    minus = kin.forward(p-eps*v,so3_exp(-w*eps)@R,v,w,q-eps*qd,qd)[1][:, ids]
    rate = so3_log(plus@minus.transpose(-1,-2))/(2*eps)
    difference = float((rate-motion).abs().max())
    assert orientation < 3e-6 and analytic < 3e-6 and difference < 2e-5
    print(f'独立 MuJoCo FK / 角速度验收通过，共 {len(samples)} 帧', flush=True)
    return dict(samples=len(samples), max_rotation_entry=orientation, max_omega_rad_s=analytic,
                max_centered_difference_rad_s=difference)


@torch.no_grad()
def numerical_invariance(root):
    kin = RobotKinematics('cpu')
    saved = torch.load(SOURCE, map_location='cpu', mmap=True, weights_only=False)
    motion = load_clip(read_manifest()['train'][1], kin, 'cpu', root, 'train')
    mdp = task(kin, motion, saved, 'c')
    f = torch.arange(1, 261, 11)
    s = motion.ref_state(f)
    u = s[4]*.4
    s = (*s[:4], s[4]+.01, s[5]+.25)
    q0 = motion.q[0].expand_as(u)
    target = mdp.executed_target(u,q0,f-1)
    anchor = motion.anchor(torch.zeros_like(f))
    r, heads = mdp.reward(s,f,u,u,True,command=anchor,ep_step=f,a_prev2=u,pd_target=target)
    rot = so3_exp(u.new_tensor([0.,0.,.91])).expand(len(f),3,3)
    x = u.new_tensor([1.2,-.8,0.]).expand(len(f),3)
    mul = lambda v: (rot@v.unsqueeze(-1)).squeeze(-1)
    again = (mul(s[0])+x,rot@s[1],mul(s[2]),mul(s[3]),s[4],s[5])
    cmd = motion.anchor(torch.zeros_like(f),rotation=rot,translation=x)
    _, result = mdp.reward(again,f,u,u,True,command=cmd,ep_step=f,a_prev2=u,pd_target=target)
    checks = {}
    for key in TERMS:
        term = key+'_contribution'
        torch.testing.assert_close(result[term], heads[term], atol=1e-8, rtol=1e-5)
        torch.testing.assert_close(heads[term], heads[key]*(float(getattr(mdp.cfg,key))*CONTROL_DT),atol=0,rtol=0)
        checks[key] = float((result[term]-heads[term]).abs().max())
    assert torch.isfinite(r).all()
    return checks


@torch.no_grad()
def cuda_probe(root):
    import warp as wp

    args = Namespace(reference_root=root, library=MANIFEST)
    readback = {}
    torch.cuda.reset_peak_memory_stats()
    for item, seed in zip(read_manifest()['train'], (1001,1104,1201)):
        clip = item['clip_id']
        with scene_for(args,SOURCE,item,'train') as (env, pi, contacts, solver):
            # 只复放前 140 个原动作，覆盖实际摆动、第一次触地和各自承重。
            with np.load(ARCHIVE, allow_pickle=False) as z:
                replay = z[f'5500_{clip}_s{seed}_raw_action'][:140]
            trace, row = trial(env,pi,contacts,solver,seed,perturb=seed != 1001,fixed_actions=replay)
            if row['termination'] in INVALID_TERMINATIONS or row['steps'] != len(replay):
                raise AssertionError(f'{clip} Newton 角速度探针提前终止 {row["termination"]}')
            mdp, device = env.mdp, env.device
            x = torch.as_tensor(trace['state'],device=device)
            state = unflatten_state(x)
            foot = mdp.motion.foot_indices
            bw = mdp.kin.forward(*state)[3]
            p,R,v = mdp.motion.feet(state)
            delta = {key: float((value-torch.as_tensor(trace[key],device=device)).abs().max())
                     for key,value in (('foot',p),('foot_R',R),('foot_velocity',v),('foot_angular_velocity',bw[:,foot]))}
            assert delta['foot'] < 2e-5 and delta['foot_R'] < 2e-5 and delta['foot_velocity'] < 5e-4
            assert delta['foot_angular_velocity'] < 5e-4
            native_R = torch.as_tensor(trace['foot_R'],device=device)
            fd = so3_log(native_R[1:]@native_R[:-1].transpose(-1,-2))/CONTROL_DT
            w = torch.as_tensor(trace['foot_angular_velocity'],device=device)
            midpoint = (w[1:]+w[:-1])*.5
            fd_rmse = float((fd-midpoint).square().mean().sqrt())
            guide = read_guide_heading(root/item['file_path'])
            measured = contact_metrics(trace,row,mdp,guide)
            record = dict(seed=seed, steps=row['steps'], max_fk_minus_newton=delta,
                          sampled_so3_rate_vs_midpoint_rad_s=fd_rmse,
                          proxy_agreement=measured['proxy_agreement'])
            if clip == 'F2':
                m = solver.model
                shape = env.physics.get_model()
                mu = wp.to_torch(shape.shape_material_mu).cpu().numpy()
                ids = [i for i in range(m.ngeom)
                       if m.geom_bodyid[i] == 0 or any(name in m.body(m.geom_bodyid[i]).name for name in FOOT_BODY_NAMES)]
                friction = EffectiveFriction(env)
                pair = friction.read()
                active = [float(value) for array in pair for value in array if np.isfinite(value)]
                record['contact_physics'] = dict(dt_s=env.physics_dt, friction_context=env.xi[0,0].item(),
                      shape_mu=np.unique(mu).tolist(), last_step_foot_floor_mu=active,
                      cone=int(m.opt.cone), iterations=int(m.opt.iterations), solver_tolerance=float(m.opt.tolerance),
                      geom=[dict(name=m.geom(i).name, type=int(m.geom_type[i]), size=m.geom_size[i].tolist(),
                                 solref=m.geom_solref[i].tolist(), solimp=m.geom_solimp[i].tolist(),
                                 friction=m.geom_friction[i].tolist()) for i in ids])
            readback[clip] = record
        torch.cuda.empty_cache()
        print(f'{clip} Newton Link 与可微 FK 角速度最大差 {delta["foot_angular_velocity"]:.3g} rad/s', flush=True)
    return readback


@torch.no_grad()
def reward_speed(root):
    device = 'cuda:0'
    kin = RobotKinematics(device)
    saved = torch.load(SOURCE, map_location='cpu', weights_only=False, mmap=True)
    motion = load_clip(read_manifest()['train'][1],kin,device,root,'train')
    cases = {'m50': task(kin,motion,saved)}
    cases.update({key: task(kin,motion,saved,key) for key in ('a','b','c')})
    f = torch.arange(256,device=device)+1
    s = motion.ref_state(f)
    a = s[4]*.2
    t = cases['a'].executed_target(a,motion.q[0].expand_as(a),f-1)
    anchor = motion.anchor(torch.zeros_like(f))
    measured = {}
    for label, mdp in cases.items():
        def value():
            return mdp.reward(s,f,a,a,True,command=anchor,ep_step=f,a_prev2=a,pd_target=t)
        for _ in range(10):
            value()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(80):
            value()
        torch.cuda.synchronize()
        measured[label] = (time.perf_counter()-start)/80*1000
    return measured


def publish(path, updates):
    existing = json.loads(path.read_text()) if path.is_file() else {}
    existing.update(updates)
    with TemporaryDirectory(dir=path.parent) as scratch:
        target = Path(scratch)/path.name
        target.write_text(json.dumps(existing,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
        target.replace(path)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all','offline','cuda','smoke'), default='all')
    parser.add_argument('--root', type=Path, default=REFERENCE_ROOT)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    path = args.out/'preflight.json'
    torch.set_num_threads(2)
    if args.stage in ('offline','all'):
        result = dict(unit=unit_tests(), independent_fk=independent_fk(), invariance=numerical_invariance(args.root),
                      offline=run_offline(args.root,args.out))
        publish(path,result)
    if args.stage in ('cuda','all'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA 当前不可用，实际 Newton 探针无法验收')
        result = dict(newton=cuda_probe(args.root),reward_ms_per_256=reward_speed(args.root))
        publish(path,result)
    if args.stage in ('smoke','all'):
        smokes = []
        for group, n in (('a',8),('b',32),('c',256)):
            smokes.append(run_smoke(args.root,n,group=group,resume_test=group != 'b'))
        publish(path,dict(smokes=smokes))
    print(f'M5.1 {args.stage} 验收结果已保存 {path}',flush=True)


if __name__ == '__main__':
    main()
