"""Bounded-launcher and isolation regressions; no simulator or GPU needed."""

# Support both direct execution and python -m from the repository layout.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_tool_root = str(_ToolPath(__file__).resolve().parents[2])
if _tool_root not in _tool_sys.path:
    _tool_sys.path.insert(0, _tool_root)

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import json

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('runner',ROOT/'tools/train/run_s54_motion_prior.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
summary_spec=importlib.util.spec_from_file_location('gates',ROOT/'tools/validation/summarize_s54_pre_ablation.py')
gates=importlib.util.module_from_spec(summary_spec);summary_spec.loader.exec_module(gates)


class BoundedTests(unittest.TestCase):
    def args(self,**kw):
        values=dict(profile='configs/motion_prior/validation/calibration_bounded.json',run='s54_test',stage='stage2',
            task='standing',group='C',source='source.pt',seed=54,manifest='manifest.json',resume=None,
            resume_iterations=24,python='python',wm_source=None)
        values.update(kw);return SimpleNamespace(**values)

    def test_calibration_is_wm_only_with_shutdown_reserve(self):
        cmd,profile=runner.command_for(self.args())
        self.assertEqual(cmd[cmd.index('--wm_only_iters')+1],'80')
        self.assertEqual(cmd[cmd.index('--iterations')+1],'80')
        self.assertEqual(cmd[cmd.index('--wall_seconds')+1],'270')
        self.assertIn('--wm_calibration',cmd);self.assertEqual(profile['wall_seconds'],300)

    def test_bridge_objective_has_no_bootstrap_or_latent(self):
        cmd,_=runner.command_for(self.args(stage='bridge',profile='configs/motion_prior/stage1/stage1_smoke.json'))
        self.assertIn('--no_critic',cmd)
        self.assertEqual(cmd[cmd.index('--paired_latent_weight')+1],'0.0')
        self.assertIn('--freeze_decoder',cmd)

    def test_bridge_groups_require_and_use_common_wm(self):
        with self.assertRaisesRegex(ValueError,'wm-source'):runner.command_for(self.args(group='D'))
        for group in 'DF':
            cmd,_=runner.command_for(self.args(group=group,wm_source='common_stage1.pt'))
            self.assertEqual(cmd[cmd.index('--init_stage2_wm')+1],'common_stage1.pt')

    def test_abc_differ_only_in_intended_initialization(self):
        commands=[runner.command_for(self.args(group=g,source=None if g=='A' else 'source.pt'))[0] for g in 'ABC']
        self.assertNotIn('--init_from_motion',commands[0])
        self.assertEqual(commands[1][commands[1].index('--transfer_components')+1],'wm')
        self.assertEqual(commands[2][commands[2].index('--transfer_components')+1],'wm_decoder')
        for command in commands:
            self.assertIn('--freeze_decoder',command)
            self.assertEqual(command[command.index('--obs_adapter')+1],'mask_reference_orientation')

    def test_extended_stage1_keeps_ablation_limits_and_common_decoder_contract(self):
        args=self.args(stage='stage1',source=None,profile='configs/motion_prior/stage1/stage1_walk_3090.json',stage1_train=True)
        cmd,profile=runner.command_for(args)
        self.assertIn('--fresh_start',cmd);self.assertNotIn('--freeze_decoder',cmd)
        self.assertEqual(cmd[cmd.index('--hist')+1],'10')
        self.assertEqual(cmd[cmd.index('--goal_mode')+1],'motion')
        self.assertEqual(cmd[cmd.index('--wm_only_iters')+1],'1000')
        self.assertEqual(cmd[cmd.index('--wall_seconds')+1],'7170')
        self.assertEqual(profile['wall_seconds'],7200)
        with self.assertRaises(ValueError):runner.command_for(self.args(stage='stage1',source=None,profile=args.profile))
        with self.assertRaisesRegex(ValueError,'only applies'):runner.command_for(self.args(stage1_train=True))

    def test_short_abc_is_explicit_and_rejects_other_objectives(self):
        profile='configs/motion_prior/stage2/abc_short_3090.json'
        with self.assertRaises(ValueError):runner.command_for(self.args(profile=profile))
        for overrides in (dict(stage='stage1'),dict(stage='bridge'),dict(group='D'),dict(task='curriculum')):
            with self.assertRaisesRegex(ValueError,'fixed-task'):
                runner.command_for(self.args(profile=profile,stage2_short=True,**overrides))
        with self.assertRaisesRegex(ValueError,'without Bridge/Prior'):
            runner.command_for(self.args(stage2_short=True))  # calibration profile must remain WM-only

    def test_short_abc_shares_all_training_knobs_except_initialization(self):
        profiles=[]
        for group in 'ABC':
            cmd,profile=runner.command_for(self.args(group=group,source=None if group=='A' else 'teacher.pt',
                task='walking',profile='configs/motion_prior/stage2/abc_short_3090.json',stage2_short=True))
            options={};i=3
            while i<len(cmd):
                key=cmd[i];value=True
                if i+1<len(cmd) and not cmd[i+1].startswith('--'):
                    value=cmd[i+1];i+=1
                options[key]=value;i+=1
            self.assertNotIn('--wall_seconds',options)
            self.assertIsNone(profile['wall_seconds'])
            self.assertEqual(options['--wm_only_iters'],'1000')
            self.assertEqual(options['--iterations'],'2000')
            for key,value in dict(num_envs='1024',horizon='32',wm_window='8',wm_batch='1024',wm_batches='16',
                                  pi_batch='256',hist='10',goal_frames='10',goal_dt='0.1',gamma='0.99',utd='0.4').items():
                self.assertEqual(options['--'+key],value)
            self.assertTrue(options['--freeze_decoder'])
            if group=='A':self.assertNotIn('--init_from_motion',options)
            else:self.assertEqual(options.pop('--transfer_components'),'wm' if group=='B' else 'wm_decoder')
            options.pop('--init_from_motion',None)
            profiles.append(options)
        self.assertEqual(profiles[0],profiles[1]);self.assertEqual(profiles[1],profiles[2])
        t=profile['train']
        self.assertEqual(t['num_envs']*(t['warmup_steps']+int(t['iterations']/t['utd'])),5251072)
        self.assertEqual(t['iterations']-t['wm_only_iters'],1000)

    def test_unlimited_training_wait_has_no_timeout(self):
        process=Mock();process.wait.return_value=0
        self.assertEqual(runner.wait_for_training(process,123.,None),0)
        process.wait.assert_called_once_with()

    def test_unlimited_training_keeps_manual_safe_stop(self):
        process=Mock();process.pid=12345;process.wait.side_effect=[KeyboardInterrupt(),0]
        with patch.object(runner.os,'killpg') as kill:
            self.assertEqual(runner.wait_for_training(process,123.,None),130)
        kill.assert_called_once_with(12345,runner.signal.SIGTERM)
        self.assertEqual(process.wait.call_args_list[-1].kwargs,dict(timeout=30))

    def test_small_validation_modes_still_require_finite_budget(self):
        profile=json.loads((ROOT/'configs/motion_prior/validation/calibration_bounded.json').read_text())
        profile['wall_seconds']=None
        source=Mock();source.read_text.return_value=json.dumps(profile)
        with patch.object(runner,'resolve_config_path',return_value=source):
            with self.assertRaisesRegex(ValueError,'only supported'):
                runner.command_for(self.args())

    def test_finite_training_wait_keeps_existing_shutdown_reserve(self):
        process=Mock();process.wait.return_value=0
        with patch.object(runner.time,'monotonic',return_value=100.):
            self.assertEqual(runner.wait_for_training(process,0.,300.),0)
        process.wait.assert_called_once_with(timeout=185.)

    def test_finite_training_timeout_requests_checkpoint_save(self):
        process=Mock();process.pid=12345
        process.wait.side_effect=[runner.subprocess.TimeoutExpired('train',185.),0]
        with patch.object(runner.time,'monotonic',side_effect=[100.,285.]),patch.object(runner.os,'killpg') as kill:
            self.assertEqual(runner.wait_for_training(process,0.,300.),124)
        kill.assert_called_once_with(12345,runner.signal.SIGTERM)
        self.assertEqual(process.wait.call_args_list[-1].kwargs,dict(timeout=15.))

    def test_teacher_timeout_alone_and_dance_holdout_never_qualify(self):
        protocol=json.loads((ROOT/'configs/motion_prior/validation/pre_ablation_gates.json').read_text())
        cfg=protocol['teacher']
        metrics={k:0. for k in ('body_position_world_rmse_m','body_orientation_world_rmse_deg',
            'root_orientation_tracking_rmse_deg','root_height_tracking_max_m','arm_pose_rmse_rad',
            'arm_velocity_rmse_rad_s','joint_pose_rmse_rad','joint_limit_violation_fraction',
            'root_pitch_max_abs_deg','root_roll_max_abs_deg','actual_loaded_stance_slip_m_s')}
        metrics.update(measured_contact_accuracy=1.,loaded_foot_samples=1000,
            arm_motion_std_rad=.1,reference_arm_motion_std_rad=.1)
        rows=[dict(seed=seed,duration_s=duration,start_seconds=offset,n_failed=0,n_timeout=1,
            episode_lengths=[round(duration/.02)],metrics=metrics.copy())
            for seed in cfg['seeds'] for duration in cfg['durations_s'] for offset in cfg['evaluation_offsets_s']]
        data=dict(rows=rows,eval_real_steps=100)
        manifest=dict(clips=[dict(name='dance1',split='heldout',status='PASS')],missing_categories=[])
        result=gates.teacher_acceptance(data,protocol,manifest)
        self.assertFalse(result['qualified']);self.assertFalse(result['checks']['independent_walking_holdout'])
        manifest['clips'][0]['name']='walk2'
        self.assertTrue(gates.teacher_acceptance(data,protocol,manifest)['qualified'])
        rows[0]['metrics']['body_position_world_rmse_m']=.5
        self.assertFalse(gates.teacher_acceptance(data,protocol,manifest)['qualified'])
        rows[0]=rows[1]
        self.assertFalse(gates.teacher_acceptance(data,protocol,manifest)['checks']['complete'])


if __name__=='__main__':unittest.main(verbosity=2)
