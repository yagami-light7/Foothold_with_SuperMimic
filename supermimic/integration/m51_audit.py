"""基于 M5.0 留存的真实轨迹核对足部奖励、WM 和 Actor 梯度。"""

from argparse import Namespace
import csv
import json

import numpy as np
import torch

from ..kinematics import RobotKinematics
from ..mdp import normalize_xi, unflatten_state
from ..motion import CONTROL_DT
from ..networks import Critic, Policy, WorldModel
from ..roban import JOINT_NAMES, XI_DIM
from .m410_evaluation import mean_metrics
from .m410_reward import HUBER_TERMS, ROOT_TERMS
from .m4_task import build_reference_tracking
from .m50_evaluation import DEFAULT_SEEDS, HOLDOUT_SEEDS, PERTURB_SEEDS, TRACE_FIELDS
from .m50_reference import MANIFEST, REPO, load_clip, read_manifest
from .m50_task import MultiReferenceGoalSpec, MultiReferenceMDP
from .m51_foot import TERMS, heading_error, rotation_error
from .m51_metrics import contact_metrics, magnitudes, read_guide_heading, rms, trace_signals
from .m51_task import CONFIGS, SOURCE, StanceFootMDP


ARCHIVE = REPO/'outputs/m5_0/cuda_evaluation.npz'
SPANS = (1, 4, 8, 16)
COHORTS = dict(nominal=DEFAULT_SEEDS, perturb=PERTURB_SEEDS, holdout_reset=HOLDOUT_SEEDS)
EXISTING_FOOT = ('swing_position', 'swing_orientation', 'swing_clearance', 'stance_position',
                 'contact_schedule', 'touchdown', 'slip')


def task(kin, motion, saved, group=None):
    cfg = Namespace(**saved['args'])
    spec = MultiReferenceGoalSpec('roban_foot', cfg.goal_frames, cfg.goal_dt)
    mdp_type = MultiReferenceMDP
    if group:
        cfg.roban_reward_config = json.loads(CONFIGS[group].read_text())['roban_reward_config']
        mdp_type = StanceFootMDP
    return build_reference_tracking(cfg, kin, motion, spec, mdp_type)[2]


def pre_states(trace, torch_device):
    to = lambda x: torch.as_tensor(x, device=torch_device)
    raw = to(trace['raw_action'])
    previous = torch.cat((raw.new_zeros((2, raw.shape[1])), raw))
    starts = to(trace['state'])
    return dict(states=starts, frame=to(trace['frame']).long(), raw=raw,
                previous=previous[1:-1], previous2=previous[:-2],
                target=to(trace['target']), effective=to(trace['effective_action']),
                foot_R=to(trace['foot_R']), foot_velocity=to(trace['foot_velocity']), contact=to(trace['contact']))


def score(mdp, x, i, state, anchor):
    f = x['frame'][i+1]
    return mdp.reward(state, f, x['raw'][i], x['previous'][i], True, command=anchor,
                      ep_step=f-anchor.start_frame, a_prev2=x['previous2'][i], pd_target=x['target'][i])


def cost_distribution(value):
    # 奖励保留原负号，P95 和峰值用惩罚幅度。
    return dict(mean=float(value.mean()), abs_p95=float(np.quantile(np.abs(value), .95)),
                abs_max=float(np.abs(value).max()))


@torch.no_grad()
def rescore(mdp, baseline, a_task, trace):
    device = mdp.default_q.device
    x = pre_states(trace, device)
    f = x['frame'][1:]
    command = mdp.motion.anchor(torch.zeros_like(f))
    states = unflatten_state(x['states'][1:])
    i = torch.arange(len(f), device=device)
    r, heads = score(mdp, x, i, states, command)
    a, _ = score(a_task, x, i, states, command)
    original, _ = score(baseline, x, i, states, command)
    torch.testing.assert_close(a, original, atol=0., rtol=0.)
    torch.testing.assert_close(heads['m50_reward'], original, atol=0., rtol=0.)
    reconstructed = (original-torch.as_tensor(trace['reward'], device=device)).abs().max().item()
    if reconstructed > 2e-6:
        raise AssertionError(f'M5.0 历史奖励重算误差 {reconstructed:.4g}')
    fids = mdp.motion.foot_indices
    _, rotation, _, angular = mdp.kin.forward(*states)
    ref = mdp._reference(f, command)
    clear = mdp._sole_clearance(torch.as_tensor(trace['foot'][1:], device=device), rotation[:, fids])
    proxy = ((mdp.cfg.contact_height-clear)/mdp.cfg.contact_height_std).sigmoid()
    signals = mdp.foot_signals(rotation[:, fids], angular[:, fids], ref, f, command, proxy)
    _, excess = signals.raw_rewards(mdp.cfg)
    planned, loaded = signals.plan.cpu().numpy(), trace['contact'][1:]
    group = np.asarray(loaded).sum(-1)
    amounts = {term: heads[term+'_contribution'].cpu().numpy() for term in (*EXISTING_FOOT, *TERMS)}
    torch.testing.assert_close(r, original+sum(heads[term+'_contribution'] for term in TERMS), atol=2e-8, rtol=0.)
    result = dict(original_foot_mean=float(sum(amounts[key] for key in EXISTING_FOOT).mean()),
                  original_components={key: float(amounts[key].mean()) for key in EXISTING_FOOT},
                  original_replay_max_error=reconstructed)
    for index, term in enumerate(TERMS):
        e = excess[index].cpu().numpy()
        effect = planned if index == 0 else planned & loaded
        result[term] = dict(all=cost_distribution(amounts[term]), dead_zone_exceeded=float((e[effect] > 0).mean()),
                            **{label: cost_distribution(amounts[term][group == count])
                               for count, label in ((1, 'single'), (2, 'double')) if (group == count).any()})
    return result


@torch.no_grad()
def wm_predictions(mdp, wm, traces, stride=10):
    device = mdp.default_q.device
    windows = {key: [] for key in ('states', 'frame', 'raw', 'previous', 'previous2', 'target', 'effective',
                                   'foot_R', 'foot_velocity', 'contact')}
    for trace in traces:
        row = pre_states(trace, device)
        starts = torch.arange(0, len(row['raw'])-SPANS[-1]+1, stride, device=device)
        if not len(starts):
            continue
        for key, value in row.items():
            state_field = len(value) == len(row['states'])
            offsets = torch.arange(SPANS[-1]+int(state_field), device=device)
            windows[key].append(value[starts[:, None]+offsets])
    data = {key: torch.cat(parts).transpose(0, 1) for key, parts in windows.items()}
    x = unflatten_state(data['states'][0])
    anchor = mdp.motion.anchor(torch.zeros_like(data['frame'][0]))
    xi = x[0].new_zeros((len(x[0]), XI_DIM))
    xi[:, 0] = 1.
    xi = normalize_xi(xi)
    results = dict(windows=len(x[0]), horizons={})
    for k in range(SPANS[-1]):
        x = wm.step(x, data['effective'][k], xi)
        if k+1 not in SPANS:
            continue
        actual = unflatten_state(data['states'][k+1])
        f = data['frame'][k+1]
        plan = mdp.motion.contact[f]
        mask = plan & data['contact'][k+1]
        _, angle, speed = mdp.motion.feet(x)
        direct_R = data['foot_R'][k+1]
        yaw, _ = heading_error(angle, direct_R, mdp.foot_axis)
        ids = mdp.motion.foot_indices
        predicted_w = mdp.kin.forward(*x)[3][:, ids]
        target_w = mdp.kin.forward(*actual)[3][:, ids]
        omega = (predicted_w-target_w)[..., 2]
        tangent = (speed-data['foot_velocity'][k+1])[..., :2]
        _, rp = score(mdp, data, k, x, anchor)
        _, rt = score(mdp, data, k, actual, anchor)
        metric = dict(yaw_rms_deg=rms(yaw.rad2deg().cpu().numpy()),
                      orientation_rms_deg=rms(rotation_error(angle, direct_R).rad2deg().cpu().numpy()),
                      stance_yaw_rms_deg=rms(yaw[mask].rad2deg().cpu().numpy()),
                      stance_omega_rms_rad_s=rms(omega[mask].cpu().numpy()),
                      stance_velocity_rms_m_s=rms(tangent[mask].norm(dim=-1).cpu().numpy()))
        for term in TERMS:
            key = f'{term}_contribution'
            delta = (rp[key]-rt[key]).cpu().numpy()
            metric[term] = dict(replay_mean=float(rt[key].mean()), imagined_mean=float(rp[key].mean()),
                                bias=float(delta.mean()), mae=float(np.abs(delta).mean()))
        results['horizons'][str(k+1)] = metric
    return results


def phase_masks(mdp, frames):
    t = frames.double() * CONTROL_DT
    near = (t[..., None]-mdp.motion.touchdown_time).abs().amin(-1) <= .10+1e-9
    support = mdp.motion.contact[frames].sum(-1)
    return dict(startup=frames <= 10, single=support == 1, double=support == 2,
                touchdown=near, walking=(frames > 10) & ~near & (t >= mdp.motion.liftoff_time[0]))


def bptt_audit(mdp, actor, wm, critic, cfg, traces):
    device = mdp.default_q.device
    parts, f0, previous, previous2, start_q, history = [], [], [], [], [], []
    n = len(JOINT_NAMES)
    horizon = SPANS[-1]
    for trace in traces:
        x = pre_states(trace, device)
        starts = torch.as_tensor([0, 4, *range(25, len(x['raw'])-horizon+1, 35)], device=device)
        idx = (starts[:, None] + torch.arange(-cfg['hist']+1, 1, device=device)).clamp_min(0)
        parts.append(x['states'][starts])
        f0.append(x['frame'][starts])
        previous.append(x['previous'][starts])
        previous2.append(x['previous2'][starts])
        start_q.append(x['states'][0, 13:13+n].expand(len(starts), -1))
        xi = x['states'].new_zeros((idx.numel(), XI_DIM))
        xi[:, 0] = 1.
        obs = mdp.obs(unflatten_state(x['states'][idx.flatten()]), x['previous'][idx.flatten()], xi, idx.flatten())
        history.append(obs.view(len(starts), cfg['hist'], -1))
    x = unflatten_state(torch.cat(parts))
    f = torch.cat(f0)
    command = mdp.motion.anchor(torch.zeros_like(f))
    a0, a1, reset_q, o = map(torch.cat, (previous, previous2, start_q, history))
    xi = x[0].new_zeros((len(f), XI_DIM))
    xi[:, 0] = 1.
    xi_n = normalize_xi(xi)
    g = mdp.goals.goal(x, f, f, command)
    controls, frames, terms, survived = [], [], [], []
    for _ in range(horizon):
        u = actor(o, g)
        p = mdp.executed_target(u, reset_q, f)
        x = wm.step(x, mdp.effective_action(p), xi_n)
        f = f+1
        g = mdp.goals.goal(x, f, f, command)
        _, heads = mdp.reward(x, f, u, a0, True, command=command, ep_step=f, a_prev2=a1, pd_target=p)
        o = torch.cat((o[:, 1:], mdp.obs(x, u, xi, f)[:, None]), 1)
        alive = (1-mdp.soft_termination(x, f)) * (f < int(mdp.motion.clip_end[0]))
        controls.append(u)
        frames.append(f)
        terms.append(heads)
        survived.append(alive)
        a1, a0 = a0, u
    gates = torch.stack(survived).cumprod(dim=0)
    discounts = cfg['gamma'] ** torch.arange(horizon, device=device)[:, None]
    weights = discounts*torch.cat((torch.ones_like(gates[:1]), gates[:-1].detach()))
    value = critic(torch.cat((mdp.critic_features(x, f, xi_n, command=command, ep_step=f), g), -1))
    bootstrap = cfg['gamma']**horizon * gates[-1] * value
    groups = dict(foot_position=('swing_position', 'stance_position'), swing_orientation=('swing_orientation',),
                  stance_yaw_huber=('stance_yaw_huber',), stance_angular_slip=('stance_angular_slip',),
                  root=ROOT_TERMS, joint_huber=HUBER_TERMS, hand_huber=('hand_huber',))
    scores = {name: torch.stack([sum(step[key+'_contribution'] for key in keys) for step in terms])
              for name, keys in groups.items()}
    params = tuple(actor.parameters())
    result = {}
    channels = {name: [JOINT_NAMES.index(f'leg_{side}{i}_joint') for side in 'lr' for i in ids]
                for name, ids in (('hip_yaw', (3,)), ('ankle', (5, 6)))}
    for phase, mask in phase_masks(mdp, torch.stack(frames)).items():
        if not bool(mask.any()):
            continue
        sample = mask.any(0)
        objective = {name: -(value*weights*mask).sum()/sample.sum() for name, value in scores.items()}
        objective['bootstrap'] = -bootstrap[sample].mean()
        gradients, summary = {}, {}
        for name, loss in objective.items():
            grad = torch.autograd.grad(loss, (*params, *controls), retain_graph=True, allow_unused=True)
            vec = torch.cat([x.reshape(-1) for x in grad[:len(params)] if x is not None])
            action_grad = torch.stack([u if u is not None else torch.zeros_like(c)
                                       for u, c in zip(grad[len(params):], controls)])
            if not (vec.isfinite().all() and action_grad.isfinite().all()):
                raise FloatingPointError(f'{phase}/{name} BPTT 梯度非有限')
            gradients[name] = vec
            summary[name] = dict(actor_norm=float(vec.norm()), action_norm=float(action_grad.norm()),
                                 **{part: float(action_grad[..., indices].norm()) for part, indices in channels.items()})
        pairs = {(a,b) for a in TERMS for b in gradients if a != b}
        cosines = {f'{a}/{b}': float(torch.nn.functional.cosine_similarity(gradients[a], gradients[b], dim=0))
                   for a, b in sorted(pairs)}
        result[phase] = dict(frame_samples=int(mask.sum()), objectives=summary, cosine=cosines)
    return result


def run_offline(root, out, *, device='cpu'):
    saved = torch.load(SOURCE, map_location='cpu', mmap=True, weights_only=False)
    kin = RobotKinematics(device)
    audit = dict(source=5500, baseline={}, wm={}, gradient={})
    wm, actor = WorldModel().to(device), Policy(69, 320, hist=saved['args']['hist']).to(device)
    critic = Critic(970).to(device)
    for name, model in (('wm',wm), ('pi',actor), ('critic',critic)):
        model.load_state_dict(saved[name])
        model.eval().requires_grad_(name == 'pi')
    rows = []
    with np.load(ARCHIVE, allow_pickle=False) as package:
        info = json.loads(package['metadata'].item())['seen']['5500']['trials']
        for entry in read_manifest(MANIFEST)['train']:
            clip = entry['clip_id']
            motion = load_clip(entry, kin, device, root, 'train')
            base, group_a, study = (task(kin, motion, saved, group) for group in (None, 'a', 'c'))
            if group_a.cfg.contract() != base.cfg.contract():
                raise AssertionError('A 组旧奖励契约与 M5.0 不一致')
            audit.setdefault('m50_reward_config', base.cfg.contract())
            guide = read_guide_heading(root/entry['file_path'])
            tracks, results = {}, {}
            for seed, row in info[clip].items():
                prefix = row['prefix']
                t = tracks[seed] = {key: package[prefix+key] for key in TRACE_FIELDS}
                contact = contact_metrics(t, row, study, guide)
                score_data = rescore(study, base, group_a, t)
                results[seed] = dict(contact=contact, rescore=score_data)
                rows.append(dict(clip=clip, seed=seed, stance_yaw_deg=contact['planned_stance']['yaw_rms_deg'],
                                 swing_yaw_deg=contact['swing']['yaw_rms_deg'], touchdown_yaw_deg=contact['touchdown_yaw_rmse_deg'],
                                 stance_omega_rms_rad_s=contact['support']['omega']['rms'],
                                 stance_slide_rms_m_s=contact['support']['speed']['rms'],
                                 stance_yaw_sum_deg=contact['support']['segment_mean']['yaw_total_deg'],
                                 stance_distance_mm=contact['support']['segment_mean']['distance_mm'],
                                 planned_load_ratio=contact['plan_load_fraction'],
                                 original_foot_mean=score_data['original_foot_mean'],
                                 yaw_reward=score_data[TERMS[0]]['all']['mean'],
                                 angular_slip_reward=score_data[TERMS[1]]['all']['mean']))
            audit['baseline'][clip] = {group: mean_metrics([results[str(seed)] for seed in seeds])
                                       for group, seeds in COHORTS.items()}
            audit['wm'][clip] = wm_predictions(study, wm, list(tracks.values()))
            audit['gradient'][clip] = bptt_audit(study, actor, wm, critic, saved['args'],
                                                [tracks[str(seed)] for seed in (1001, 1101, 1201)])
            print(f'{clip} 已审计 15 条历史轨迹，WM 窗口 {audit["wm"][clip]["windows"]}', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'foot_baseline.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return audit
