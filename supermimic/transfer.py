"""Explicit cross-task initialization, separate from strict same-task resume."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.metadata

import torch

from .mdp import OBS_DIM, MOTION_OBS_DIM, STATE_DIM, INV_DIM, RewardCfg, motion_reward_contract
from .motion import CONTROL_DT
from .s54 import (JOINT_NAMES, NUM_J, XI_DIM, MJCF_PATH, checkpoint_contract,
                  assert_checkpoint, actuator_parameters)

SIMULATION = dict(num_substeps=4, njmax=500, nconmax=128, cone='pyramidal',
                  integrator='implicitfast', use_mujoco_contacts=True)


def physical_contract(obs_dim=OBS_DIM):
    params = actuator_parameters()
    versions = {}
    for name in ('newton', 'mujoco-warp', 'warp-lang'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = 'unavailable'
    return dict(version=1, robot=checkpoint_contract(), state_dim=STATE_DIM,
                state='root_link_world_pose_velocity; canonical_q_qd; quaternion_wxyz',
                obs_dim=obs_dim, action_dim=NUM_J, xi_dim=XI_DIM, wm_input=INV_DIM+NUM_J+XI_DIM,
                wm_output=70, control_dt=CONTROL_DT, gravity=[0.,0.,-9.81], solver=SIMULATION,
                sim_mjcf_sha256=hashlib.sha256(MJCF_PATH.read_bytes()).hexdigest(),
                actuators=params, action_scale=[.25*params[n]['effort']/params[n]['stiffness'] for n in JOINT_NAMES],
                backend_versions=versions)


def policy_contract(hist, adapter='identity', decoder_trainable=True, *, obs_dim=OBS_DIM, observation_mode='legacy_v1'):
    if adapter not in ('identity', 'mask_reference_orientation'):
        raise ValueError('Unknown observation adapter')
    expected_dim = MOTION_OBS_DIM if observation_mode == 'root_xy_heading_v2' else OBS_DIM
    if observation_mode not in ('legacy_v1', 'root_xy_heading_v2') or obs_dim != expected_dim:
        raise ValueError('Policy observation mode/dimension mismatch')
    value = dict(version=1, hist=hist, obs_dim=obs_dim, z_dim=32, action_dim=29,
                obs_adapter=adapter, decoder_trainable=decoder_trainable,
                obs_reference_masked_channels=list(range(93,99)) if adapter != 'identity' else [])
    if observation_mode != 'legacy_v1':
        value.update(version=2, observation_mode=observation_mode,
                     feedback_channels=dict(root_xy_error=[99, 100], heading_error_sin_cos=[101, 102]))
    return value


def assert_policy_checkpoint(ck, hist, adapter, decoder_trainable, *, obs_dim=OBS_DIM, observation_mode='legacy_v1'):
    actual = ck.get('policy_contract')
    expected = policy_contract(hist,adapter,decoder_trainable,obs_dim=obs_dim,observation_mode=observation_mode)
    if actual is not None and actual != expected:
        raise ValueError('Policy history/observation-adapter/freeze contract mismatch')
    if actual is None and adapter != 'identity':
        raise ValueError('Legacy checkpoints have no adapted observation contract')
    if actual is None and observation_mode != 'legacy_v1':
        raise ValueError('Legacy 99D checkpoints cannot resume a 103D Root-feedback policy')


def _validate_shapes(module, state, label):
    wanted = module.state_dict()
    if wanted.keys() != state.keys() or any(wanted[k].shape != state[k].shape for k in wanted):
        raise ValueError(f'{label} state keys/shapes mismatch')
    if any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise ValueError(f'{label} contains nonfinite weights/normalization')


def _validate_motion_task(ck,paired=False):
    task=ck.get('task_contract') or {}
    try:
        expected=dict(**motion_reward_contract(RewardCfg(**task.get('config',{}))),
            mode='paired_motion_bridge' if paired else 'motion',obs_dim=99,
            critic_state_dim=135 if paired else 184,control_dt=CONTROL_DT)
        if paired:
            from .motion_prior_data import ContactConfig
            expected.update(goal_source='original_reference_fk',
                contact_config=asdict(ContactConfig(**task.get('contact_config',{}))),
                reset='Motion bins',termination='Motion reference height/tilt + clip end')
    except (TypeError,ValueError) as exc:
        raise ValueError('Invalid Motion reward/task contract') from exc
    if task!=expected: raise ValueError('Incompatible Motion reward/task contract')


def _report_new_modules(report,pi,modules):
    for name in modules:
        for key,value in getattr(pi,name).state_dict().items():
            report['mapping'].append(dict(source=None,target=f'pi.{name}.{key}',shape=list(value.shape),
                status='new',reason='New Foot interface or independent normalization; no compatible source semantics'))
    report['mapping'].append(dict(source=None,target='critic',status='new',input_dim=365,output_dim=1,
        reason='New goal, reward and privileged features; source critic/optimizer cannot be resumed'))


def initialize_from_motion(ck, wm, pi, *, hist, components='wm_decoder', expected_physics=None):
    """Validate everything before mutating; no optimizer/Critic/GoalNorm is copied.

    Foot decoder masks the six reference-relative orientation channels AFTER
    normalization. Projected gravity remains available. Decoder weights themselves
    are copied exactly; this adapter never changes the source Motion policy.
    """
    if components not in ('wm', 'wm_decoder'):
        raise ValueError('Transfer components must be wm or wm_decoder')
    assert_checkpoint(ck)
    if ck.get('physical_contract') != (expected_physics or physical_contract()):
        raise ValueError('Missing/incompatible physical/actuator/backend contract; legacy weights cannot be attested automatically')
    source_goal = ck.get('goal_contract', {})
    from .goals import GoalSpec
    if source_goal != GoalSpec('motion').contract():
        raise ValueError('init_from_motion requires a Stage 1 Motion 730D checkpoint')
    _validate_motion_task(ck)
    source_args=ck.get('args',{})
    if ck.get('policy_contract') is None: raise ValueError('Source Motion policy contract is missing')
    assert_policy_checkpoint(ck,source_args.get('hist'),source_args.get('obs_adapter','identity'),
                             not source_args.get('freeze_decoder',False))
    if components == 'wm_decoder' and ck.get('args',{}).get('hist') != hist:
        raise ValueError('Decoder history mismatch; use explicit WM-only transfer instead')
    if components == 'wm_decoder' and pi.obs_adapter != 'mask_reference_orientation':
        raise ValueError('Motion -> Foot decoder requires the explicit observation adapter')
    _validate_shapes(wm, ck['wm'], 'WM')
    decoder = {k.removeprefix('dec.'):v for k,v in ck['pi'].items() if k.startswith('dec.')}
    if components == 'wm_decoder': _validate_shapes(pi.dec, decoder, 'Decoder')
    if hasattr(pi,'motion_enc'):
        for target,prefix in [(pi.motion_enc,'enc.'),(pi.motion_goal_norm,'goal_norm.'),(pi.motion_obs_norm,'obs_norm.')]:
            _validate_shapes(target,{k.removeprefix(prefix):v for k,v in ck['pi'].items() if k.startswith(prefix)},prefix)
    report = dict(version=1, source_iteration=ck.get('iteration'), source_sim_steps=ck.get('sim_steps',0),
                  source_total_wall_seconds=ck.get('wall_time',0.)+ck.get('stage_stats',{}).get('upstream_wall_seconds',0.),
                  components=components, hist=hist, loaded=[], new=[], skipped=[], mapping=[])
    for prefix, state in [('wm',ck['wm']),('pi.dec',decoder if components=='wm_decoder' else {})]:
        for key,value in state.items():
            report['mapping'].append(dict(source=f'{prefix}.{key}',target=f'{prefix}.{key}',shape=list(value.shape),status='loaded',
                reason='Explicit cross-task initialization after robot, physics, task and tensor validation'))
        if state: report['loaded'].append(prefix)
    wm.load_state_dict(ck['wm'],strict=True)
    if components == 'wm_decoder': pi.dec.load_state_dict(decoder,strict=True)
    if hasattr(pi,'motion_enc'):
        for target,prefix in [(pi.motion_enc,'enc.'),(pi.motion_goal_norm,'goal_norm.'),(pi.motion_obs_norm,'obs_norm.')]:
            target.load_state_dict({k.removeprefix(prefix):v for k,v in ck['pi'].items() if k.startswith(prefix)},strict=True)
            for key,value in target.state_dict().items():
                report['mapping'].append(dict(source='pi.'+prefix+key,target='pi.motion_'+prefix+key,
                    shape=list(value.shape),status='loaded',reason='Frozen paired Motion supervision; independent source normalization'))
        for p in pi.motion_enc.parameters():p.requires_grad_(False)
        report['loaded'].extend(['frozen Motion encoder','independent Motion goal/obs norms'])
    report['new'] = ['pi.enc','pi.goal_norm','pi.obs_norm','critic','all optimizers','replay']
    if components=='wm': report['new'].append('pi.dec')
    _report_new_modules(report,pi,['enc','goal_norm','obs_norm']+(['dec'] if components=='wm' else []))
    report['skipped'] = (['Motion encoder','Motion goal/observation normalization'] if not hasattr(pi,'motion_enc') else [])+['Motion Critic','source optimizers']
    if components=='wm':report['skipped'].append('Motion Decoder')
    report['observation_semantics'] = ('Foot: reset observation normalization; mask normalized relative orientation '
        'channels 93:99 in every history frame; retain projected gravity and all other sensors')
    report['source_data_contract'] = deepcopy(ck.get('data_contract'))
    report['teacher_required_by_stage2'] = False
    return report


def initialize_from_bridge(ck,wm,pi,*,hist,expected_physics=None):
    assert_checkpoint(ck)
    if ck.get('physical_contract')!=(expected_physics or physical_contract()): raise ValueError('Bridge physics mismatch')
    from .goals import GoalSpec
    if ck.get('goal_contract')!=GoalSpec('reference_foot').contract():
        raise ValueError('Expected paired Motion-reward BPTT bridge checkpoint')
    _validate_motion_task(ck,paired=True)
    if ck.get('args',{}).get('hist')!=hist: raise ValueError('Bridge history mismatch')
    if ck.get('policy_contract',{}).get('obs_adapter')!=pi.obs_adapter:
        raise ValueError('Bridge observation semantics mismatch')
    assert_policy_checkpoint(ck,hist,pi.obs_adapter,False)
    foot_state={k:v for k,v in ck['pi'].items() if not k.startswith(('motion_enc.','motion_obs_norm.','motion_goal_norm.'))}
    _validate_shapes(wm,ck['wm'],'WM'); _validate_shapes(pi,foot_state,'Foot Actor')
    wm.load_state_dict(ck['wm'],strict=True);pi.load_state_dict(foot_state,strict=True)
    pi.goal_norm.mean.zero_();pi.goal_norm.var.fill_(1);pi.goal_norm.count.zero_()
    stage=ck.get('stage_stats',{})
    report=dict(version=1,loaded=['wm','pi.enc','pi.dec','pi.obs_norm'],
                new=['pi.goal_norm','critic','all optimizers','replay'],
                skipped=['frozen Motion encoder and Motion norms','Bridge GoalNorm','Bridge critic and optimizers'],
                source_sim_steps=ck.get('sim_steps',0),
                source_total_real_steps=ck.get('sim_steps',0)+stage.get('upstream_real_steps',0),
                source_total_wall_seconds=ck.get('wall_time',0.)+stage.get('upstream_wall_seconds',0.),
                mapping=[dict(source=prefix+k,target=prefix+k,shape=list(v.shape),status='loaded',
                    reason='Validated paired Foot interface; new Quintic goal normalization handled separately')
                    for prefix,state in [('wm.',ck['wm']),('pi.',foot_state)] for k,v in state.items()
                    if prefix=='wm.' or not k.startswith('goal_norm.')],
                observation_semantics='Same fixed command heading adapter; fresh Quintic Goal normalization',
                teacher_required_by_stage2=False)
    _report_new_modules(report,pi,['goal_norm'])
    return report
