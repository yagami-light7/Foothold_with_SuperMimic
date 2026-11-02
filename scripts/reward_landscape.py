"""Synthetic reward response versus lift height; this does not simulate balance."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from agents.rsl_rl_ppo_cfg import S54FootPlacementPPORunnerCfg
from configs.s54_curriculum import configure_stage
from tasks.foot_placement.foot_placement_env_cfg import S54FootPlacementEnvCfg
from tasks.foot_placement.rewards import compute_foot_placement_reward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/preflight_2026-10-29/reward_landscape"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    heights = torch.linspace(0, 0.1, 101)
    n = len(heights)
    rows = []
    fig, ax = plt.subplots(figsize=(7,4))
    summary = {}
    for stage in ('legacy','lift'):
        env, agent = S54FootPlacementEnvCfg(), S54FootPlacementPPORunnerCfg()
        configure_stage(env,agent,stage)
        left = torch.zeros(n,3)
        left[:,1] = 0.095
        right = left.clone();right[:,1] = -0.095
        target = left.clone();target[:,0] = env.step_length
        current = left.clone();current[:,2] = heights
        lf = torch.zeros(n,3);lf[:,2] = 330*(1-heights/0.015).clamp(0,1)
        rf = torch.zeros(n,3);rf[:,2] = 660-lf[:,2]
        gravity = torch.zeros(n,3);gravity[:,2] = -1
        reward, terms = compute_foot_placement_reward(current,right,target,right,lf,rf,
            torch.full((n,),0.25),torch.ones(n,dtype=torch.bool),gravity,
            torch.zeros(n,27),torch.zeros(n,27),0.02,env.reward_cfg,
            swing_has_lifted=heights>0.005)
        for i,h in enumerate(heights):
            rows.append({'stage':stage,'lift_m':h.item(),'reward_per_step':reward[i].item(),
                         **{key:value[i].item()*0.02 for key,value in terms.items()}})
        summary[stage]={'standing_mid_swing':reward[0].item(),'best_counterfactual':reward.max().item(),
                        'best_lift_m':heights[reward.argmax()].item(),
                        'desired_height':env.reward_cfg.swing_height}
        ax.plot(heights.numpy()*100,reward.numpy(),label=stage)
    ax.set(xlabel='Swing sole lift (cm)',ylabel='Reward / policy step',title='Mid-swing synthetic reward response (supported stance)')
    ax.legend();ax.grid(alpha=0.25);fig.tight_layout();fig.savefig(args.output.with_suffix('.png'),dpi=180)
    with args.output.with_suffix('.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    args.output.with_suffix('.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
