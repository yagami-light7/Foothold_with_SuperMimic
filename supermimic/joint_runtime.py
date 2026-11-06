"""Validated joint contracts, canonical checkpoints and interface projections."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import torch
from .goals import GoalSpec,GoalSource
from .foot_trajectory import AlternatingConfig
from .foot_mdp import FootTrackingMDP,resolve_foot_task
from .kinematics import S54Kinematics
from .mdp import MDP,RewardCfg
from .motion import MotionLib
from .motion_prior_data import load_manifest
from .networks import WorldModel,Critic
from .s54 import default_joint_pos_list,checkpoint_contract
from .transfer import physical_contract,policy_contract
from .joint_models import JointPolicy,initialize_teacher,optimizers

ROOT=Path(__file__).resolve().parents[1]
TEACHER_SHA='1cf556f384cefa8635975ca4f8cb8c0e144a98292c5fb2fb8cff33e9b07c3fc4'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def json_write(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def profile_hash(cfg):return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()


def validate_config(cfg):
    if cfg['experiment']!='S54_JOINT_AB1' or cfg['version']!=1:raise ValueError('Unknown joint experiment contract')
    if cfg['teacher_sha256']!=TEACHER_SHA or digest(ROOT/cfg['teacher'])!=TEACHER_SHA:raise ValueError('Exact Stage1 Teacher missing or SHA256 mismatch')
    if cfg['mode'] not in ('formal','smoke'):raise ValueError('Unknown execution mode')
    if cfg['mode']=='formal' and (cfg['wall_seconds']!=43200 or cfg['iterations']!=50000):raise ValueError('Formal contract requires latest user 12h / 50000 cap')
    for k in ['num_envs_motion','num_envs_foot','iterations','wm_batches','wm_batch','pi_batch_motion','pi_batch_foot','wm_window','horizon']:
        if not isinstance(cfg[k],int) or cfg[k]<=0:raise ValueError('Invalid '+k)
    if cfg['num_envs_motion']!=cfg['num_envs_foot'] or cfg['pi_batch_motion']!=cfg['pi_batch_foot'] or cfg['wm_batch']%2:raise ValueError('Require exact 50/50 interface budgets')
    if cfg['mode']=='formal' and cfg['num_envs_motion'] not in (256,512):raise ValueError('Untested formal environment count')
    if cfg['prior_config'] is not None or cfg['paired_latent_weight']!=0:raise ValueError('Joint AB1 has no extra prior or latent/action distillation')
    if cfg['hist']!=10 or cfg['goal_frames']!=10 or cfg['motion_goal_dt']!=.02 or cfg['foot_goal_dt']!=.1:raise ValueError('Goal/history contracts changed')
    if cfg['buffer_steps']<max(cfg['wm_window'],cfg['hist']) or cfg['warmup_steps']<cfg['wm_window']:raise ValueError('Replay too short')
    if cfg['horizon']>=500 or not 0<=cfg['wm_only_iters']<cfg['iterations']:raise ValueError('No valid policy update budget')
    for k in ['lr_wm','lr_motion_encoder','lr_decoder','lr_foot_encoder','lr_critic','utd','wall_seconds']:
        if not math.isfinite(cfg[k]) or cfg[k]<=0:raise ValueError('Invalid '+k)
    if not 0<=cfg['gamma']<1 or not 0<=cfg['lam']<=1:raise ValueError('Invalid return settings')
    files,contract=load_manifest(str(ROOT/cfg['reference_manifest']),'train')
    if [Path(f).stem for f in files]!=['walk1_subject1']:raise ValueError('AB1 must use the declared Walking source, not Dance/holdout')
    return files,contract


def build(cfg,device):
    files,data=validate_config(cfg)
    kin=S54Kinematics(device);motion=MotionLib(files,kin,device)
    foot_cfg=AlternatingConfig(**json.loads((ROOT/cfg['foot_config']).read_text()))
    task=resolve_foot_task(str(ROOT/cfg['foot_task_config']))
    goals={'motion':GoalSource(motion,kin,GoalSpec('motion',10,.02)),
           'foot':GoalSource(None,kin,GoalSpec('foot_trajectory',10,.1),foot_cfg,command_mode=task.command_mode)}
    teacher=torch.load(ROOT/cfg['teacher'],map_location=device,weights_only=False)
    default=torch.tensor(default_joint_pos_list(),device=device)
    mdps={'motion':MDP(kin,motion,default,RewardCfg(**teacher['args']['reward_config'])),
          'foot':FootTrackingMDP(kin,default,goals['foot'],task)}
    pi=JointPolicy(cfg['hist']).to(device);wm=WorldModel().to(device)
    critics={name:Critic(mdp.critic_dim+goals[name].spec.dim).to(device) for name,mdp in mdps.items()}
    mapping=initialize_teacher(teacher,pi,wm,critics,data)
    mapping.update(teacher_sha256=cfg['teacher_sha256'],source_path=cfg['teacher'],
        contracts=dict(physics=physical_contract(),goals={n:g.spec.contract() for n,g in goals.items()},tasks={n:m.contract() for n,m in mdps.items()}))
    opts=optimizers(pi,wm,critics,cfg)
    return dict(kin=kin,motion=motion,goals=goals,mdps=mdps,policy=pi,wm=wm,critics=critics,optimizers=opts,mapping=mapping,data_contract=data)


def contract(cfg,objects):
    return dict(version=1,kind='joint_motion_foot',profile_sha256=profile_hash(cfg),teacher_sha256=cfg['teacher_sha256'],
                physical=physical_contract(),goals={n:g.spec.contract() for n,g in objects['goals'].items()},
                foot_sampling=json.loads(json.dumps(asdict(objects['goals']['foot'].foot_config))),
                tasks={n:m.contract() for n,m in objects['mdps'].items()},data=objects['data_contract'],
                update_permissions=dict(motion=['E_m','D'],foot=['E_f'],wm=['WM']),
                replay_restore='restore simulator poses/clocks/commands/history; refill separate replays, then bounded WM-only recovery; Newton contact warm state is fresh')


def assert_resume(checkpoint,expected):
    if checkpoint.get('joint_contract')!=expected:raise ValueError('Joint resume contract mismatch; single-interface A/B/C checkpoints are not joint resumes')


def project(checkpoint,name):
    """Portable single-interface inference view; never writes historical files."""
    cfg=checkpoint['profile'];joint=checkpoint['joint_contract']
    if name not in ('motion','foot'):raise ValueError('Unknown interface')
    state=checkpoint['policy']['motion']
    if name=='foot':state=dict(checkpoint['policy']['foot'],**{k:v for k,v in state.items() if k.startswith('dec.')})
    foot=name=='foot'
    args=dict(goal_mode='foot_trajectory' if foot else 'motion',hist=cfg['hist'],goal_frames=10,
              goal_dt=cfg['foot_goal_dt'] if foot else cfg['motion_goal_dt'],amp=cfg['amp'],
              foot_config=checkpoint['resolved_foot_config'] if foot else {},
              foot_task_config=joint['tasks']['foot']['config'] if foot else None,
              reward_config=joint['tasks'][name]['config'],seed=cfg['seed'],
              reference_manifest=str(ROOT/cfg['reference_manifest']) if not foot else None,
              manifest_split='train',motion_dir=None,clips=[] if foot else ['walk1_subject1'],
              obs_adapter='mask_reference_orientation' if foot else 'identity',freeze_decoder=foot,prior_config=None)
    return dict(robot_contract=checkpoint_contract(),checkpoint_schema_version=2,physical_contract=joint['physical'],
                goal_contract=joint['goals'][name],task_contract=joint['tasks'][name],
                policy_contract=policy_contract(cfg['hist'],args['obs_adapter'],not foot),data_contract=None if foot else joint['data'],
                args=args,wm=checkpoint['wm'],pi=state,critic=checkpoint['critics'][name],
                iteration=checkpoint['iteration'],sim_steps=checkpoint['counters']['real_'+name],
                projection=dict(joint_experiment='S54_JOINT_AB1',interface=name,decoder_owner='Motion only'))
