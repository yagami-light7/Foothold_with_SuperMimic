"""CPU engineering gates; synthetic Replay is not Newton performance evidence."""
import copy,json,tempfile,unittest
from pathlib import Path
import torch
from unittest.mock import patch
from supermimic.joint_runtime import ROOT,build,contract,assert_resume,validate_config,project
from supermimic.joint_updates import JointUpdates
from supermimic.joint_replay import JointReplay
from supermimic.mdp import flatten_state
from supermimic.networks import Policy
from supermimic.curriculum_runtime import atomic_save,rng_state,restore_rng


def clone(module):return {k:v.clone() for k,v in module.state_dict().items()}
def same(module,state):return all(torch.equal(v,module.state_dict()[k]) for k,v in state.items())


class JointGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg=json.loads((ROOT/'configs/motion_prior/joint_ab1/joint_ab1_3090_12h.json').read_text())
        cls.cfg.update(mode='smoke',num_envs_motion=2,num_envs_foot=2,horizon=2,wm_batch=4,wm_batches=1,
            wm_window=2,pi_batch_motion=2,pi_batch_foot=2,critic_epochs=1,critic_minibatches=1,iterations=4,wm_only_iters=1)

    def make(self):
        torch.manual_seed(54);o=build(self.cfg,'cpu');buf={}
        for name in ('motion','foot'):
            b=JointReplay(32,2,'cpu',interface=name);buf[name]=b
            mdp=o['mdps'][name];goals=o['goals'][name]
            cmd=goals.new_command(mdp.nominal_state(2)) if name=='foot' else None
            for j in range(32):
                f=torch.full((2,),j,dtype=torch.long)
                S=o['motion'].ref_state(f) if name=='motion' else mdp.nominal_state(2)
                S1=o['motion'].ref_state(f+1) if name=='motion' else mdp.nominal_state(2)
                xi=torch.zeros(2,33);xi[:,0]=1
                tr=dict(s=flatten_state(S),s1=flatten_state(S1),f=f,ep_step=f.int(),xi=xi,
                        a=torch.zeros(2,29),a_prev=torch.zeros(2,29),r=torch.zeros(2),
                        done=torch.zeros(2,dtype=torch.bool),term=torch.zeros(2,dtype=torch.bool),timeout=torch.zeros(2,dtype=torch.bool))
                if cmd is not None:tr.update(command=cmd,command_id=torch.tensor([100,101]))
                b.add(tr)
        u=JointUpdates(o['policy'],o['wm'],o['critics'],o['optimizers'],o['mdps'],o['goals'],buf,self.cfg)
        return o,u

    def test_single_decoder_independent_norms_and_source_inference(self):
        o,u=self.make();pi=o['policy']
        self.assertIs(pi.motion.dec,pi.foot.dec)
        self.assertIsNot(pi.motion.obs_norm,pi.foot.obs_norm)
        self.assertIsNot(pi.motion.goal_norm,pi.foot.goal_norm)
        teacher=torch.load(ROOT/self.cfg['teacher'],map_location='cpu',weights_only=False)
        source=Policy(99,730,hist=10);source.load_state_dict(teacher['pi'])
        obs=torch.randn(2,10,99);goal=torch.randn(2,730)
        with torch.no_grad():self.assertTrue(torch.equal(source(obs,goal),pi.motion(obs,goal)))
        self.assertEqual(len(o['optimizers']['motion'].state),0)
        self.assertEqual(len(o['optimizers']['foot'].state),0)

    def test_foot_backward_through_decoder_but_changes_only_foot_encoder_and_critic(self):
        o,u=self.make();pi=o['policy'];D=clone(pi.motion.dec);Em=clone(pi.motion.enc);Ef=clone(pi.foot.enc)
        W=clone(o['wm']);Vm=clone(o['critics']['motion'])
        r=u.actor_critic('foot',audit=True)
        self.assertGreater(r['latent_gradnorm'],0);self.assertGreater(r['pure_foot_gradnorm'],0)
        self.assertEqual(r['delta_decoder'],0);self.assertGreater(r['delta_encoder'],0)
        self.assertTrue(same(pi.motion.dec,D));self.assertTrue(same(pi.motion.enc,Em))
        self.assertFalse(same(pi.foot.enc,Ef));self.assertTrue(same(o['wm'],W));self.assertTrue(same(o['critics']['motion'],Vm))

    def test_motion_updates_shared_decoder_and_encoder_without_crossing_foot(self):
        o,u=self.make();Ef=clone(o['policy'].foot.enc);W=clone(o['wm']);Vf=clone(o['critics']['foot'])
        r=u.actor_critic('motion',audit=True)
        self.assertGreater(r['delta_decoder'],0);self.assertGreater(r['delta_encoder'],0)
        self.assertTrue(same(o['policy'].foot.enc,Ef));self.assertTrue(same(o['wm'],W));self.assertTrue(same(o['critics']['foot'],Vf))

    def test_wm_mixes_two_independent_replays_without_actor_updates(self):
        o,u=self.make();P=clone(o['policy']);W=clone(o['wm'])
        r=u.wm_update()
        self.assertEqual(r['motion']['sampled_windows'],2);self.assertEqual(r['foot']['sampled_windows'],2)
        self.assertFalse(same(o['wm'],W));self.assertTrue(same(o['policy'],P))
        self.assertNotIn('command_id',u.buffers['motion'].fields)
        self.assertIn('command_id',u.buffers['foot'].fields)

    def test_clock_mask_rejects_reset_and_foot_commands_are_snapshots(self):
        o,u=self.make();b=u.buffers['foot'];initial=b.command_store[100].target_pos.clone()
        cmd=b.commands(torch.tensor([100]));cmd.target_pos.add_(3)
        self.assertTrue(torch.equal(b.command_store[100].target_pos,initial))
        b.data['ep_step'][10:]=torch.arange(22)[:,None]
        # Force the sampled window to cross the explicit reset gap at t=10.
        from unittest.mock import patch
        with patch('torch.randint',side_effect=[torch.tensor([9]),torch.tensor([0])]):w=b.sample_windows(1,2)
        self.assertEqual(w['mask'].tolist(),[[1.,0.]])

    def test_resume_canonical_alias_optimizers_rng_and_wrong_contract_rejection(self):
        o,u=self.make();u.actor_critic('foot',audit=True);expected=contract(self.cfg,o)
        with patch('torch.cuda.is_available',return_value=False):
            state=dict(joint_contract=expected,policy=o['policy'].canonical_state(),
                       opts={n:v.state_dict() for n,v in o['optimizers'].items()},rng=rng_state(),iteration=3)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'last.pt';atomic_save(state,path);restored=torch.load(path,weights_only=False)
        assert_resume(restored,expected)
        with self.assertRaises(ValueError):assert_resume({'args':{}},expected)
        bad=copy.deepcopy(expected);bad['goals']['foot']['dt']=.2
        with self.assertRaises(ValueError):assert_resume(restored,bad)
        o['policy'].load_canonical(restored['policy'])
        self.assertIs(o['policy'].motion.dec,o['policy'].foot.dec)
        for n,opt in o['optimizers'].items():opt.load_state_dict(restored['opts'][n])
        self.assertEqual({int(x['step']) for x in o['optimizers']['foot'].state.values()},{1})
        restore_rng(restored['rng']);x=torch.rand(3);restore_rng(restored['rng']);self.assertTrue(torch.equal(x,torch.rand(3)))

    def test_wrong_teacher_and_goal_configuration_rejected(self):
        bad=copy.deepcopy(self.cfg);bad['teacher_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Teacher'):validate_config(bad)
        bad=copy.deepcopy(self.cfg);bad['foot_goal_dt']=.02
        with self.assertRaisesRegex(ValueError,'Goal'):validate_config(bad)

    def test_paired_diagnostic_summary_unpacks_window_rows(self):
        from supermimic.joint_evaluation import paired_summary
        summary={'horizons':{h:{} for h in ('1','4','8','16','32')}}
        with patch('supermimic.joint_evaluation.paired_predictions',return_value=(summary,[{'start':0}])):
            self.assertIs(paired_summary(None,None,None,None,'cpu',1),summary)

    def test_replay_rejects_cross_interface_commands_and_changed_sampling_contract(self):
        o,u=self.make()
        self.assertTrue(torch.equal(u.buffers['motion'].data['start_frame'],torch.zeros(32,2,dtype=torch.long)))
        with self.assertRaisesRegex(ValueError,'leaked'):u.buffers['motion'].add({'command':u.buffers['foot'].commands(torch.tensor([100]))})
        with self.assertRaisesRegex(ValueError,'requires'):u.buffers['foot'].add({'command':None})
        expected=contract(self.cfg,o);bad=copy.deepcopy(expected);bad['foot_sampling']['step_length']=[.1,.2]
        with self.assertRaisesRegex(ValueError,'mismatch'):assert_resume({'joint_contract':bad},expected)


if __name__=='__main__':unittest.main()
