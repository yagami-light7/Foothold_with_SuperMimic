import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.supervise_s54_run import verdict

class SupervisorTests(unittest.TestCase):
    def row(self,loss=8,reward=.7,length=90):
        return dict(v_loss=str(loss),sim_rew=str(reward),ep_len=str(length),wm_loss='0.1')
    def test_transient_value_spike_or_wm_spike_does_not_stop(self):
        self.assertIsNone(verdict([self.row(.03),self.row(8),self.row(.1)]))
        rows=[self.row() for _ in range(3)];rows[-1]['wm_loss']='10';self.assertIsNone(verdict(rows))
    def test_severe_joint_collapse_and_nonfinite_stop(self):
        self.assertIsNotNone(verdict([self.row(5000,-2,15) for _ in range(3)]))
        self.assertIsNone(verdict([self.row(5000,-2,15)]))
        self.assertIsNotNone(verdict([self.row(float('nan'))]))
        self.assertIsNotNone(verdict([self.row(2e6,.7,100) for _ in range(3)]))

if __name__=='__main__':unittest.main()
