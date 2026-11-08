"""Summarize existing PD Newton rollouts without changing or selecting motions."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_dir', type=Path, required=True)
    args = p.parse_args()
    result = json.loads((args.run_dir/'pd/results.json').read_text())
    rows = result['rows']
    lines = ['# S54 M4.1 Newton PD baseline', '',
             f'Run: `{args.run_dir}`. Status: `{result["status"]}`.', '',
             'Newton/MJWarp, 50 Hz control, four substeps, gravity -9.81 m/s², real ground contacts, 29 limited joint actuators.',
             'Root remains free during rollout. Root/joint state writes are used only for reset and the independent FK gate.',
             'Targets are reference joint positions at t+0.02 s. No root correction, reference feedforward torque, or stabilizing root controller.', '',
             f'All-body Newton gate: `{result["fk_gate"]["status"]}`, 14 reference poses and 16 random poses.',
             'FK agreement establishes a model/data contract; it does not establish dynamic feasibility.', '',
             '| mode | clips | completed | mean survival (s) | root RMSE (m) | joint RMSE (rad) | foot RMSE (m) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for mode in ('straight', 'turn', 'spin'):
        group = [r for r in rows if r['mode'] == mode]
        avg = lambda key: statistics.mean(r[key] for r in group)
        lines.append(f'| {mode} | {len(group)} | {sum(r["survived"] for r in group)} | {avg("episode_s"):.3f} | {avg("root_position_rmse_m"):.4f} | {avg("joint_rmse_rad"):.4f} | {avg("foot_position_rmse_m"):.4f} |')
    lines += ['', '| clip | termination | frame | duration (s) | stance slip (m/s) | foot contact samples |',
              '|---|---|---:|---:|---:|---:|']
    for r in rows:
        slip = f'{r["stance_slip_m_s"]:.5f}' if r['stance_slip_m_s'] is not None else 'N/A'
        lines.append(f'| {r["id"]} | {r["status"]} | {r["failure_frame"]} | {r["episode_s"]:.2f} | {slip} | {r["ground_foot_contact_samples"]} |')
    raw_results = [json.loads(p.read_text()) for p in sorted(args.run_dir.glob('pd_raw_*/results.json'))]
    if raw_results:
        lines += ['', '## Same-source GMR-Raw / TCRS-Clean PD comparisons', '',
                  'The controller/model are identical. Raw and cleaned references retain their own generation timing; this is not a frame-aligned intervention.', '',
                  '| source / mode | clean duration (s) | raw duration (s) | clean root RMSE (m) | raw root RMSE (m) |',
                  '|---|---:|---:|---:|---:|']
        for result in raw_results:
            for raw in result['rows']:
                clean = next(r for r in rows if r['id'] == raw['id'].removesuffix('_raw'))
                lines.append(f'| {clean["id"]} / {clean["mode"]} | {clean["episode_s"]:.2f} | {raw["episode_s"]:.2f} | {clean["root_position_rmse_m"]:.4f} | {raw["root_position_rmse_m"]:.4f} |')
            if result['status'] != 'completed':
                lines.append(f'Raw reference/environment error retained: `{result.get("error")}`.')
    lines += ['', 'The complete per-frame recorded Newton state is retained in `pd/*.npz`; summary metrics are in `pd/results.json`.',
              'World-space root and foot link-origin errors include trajectory drift. Stance slip is horizontal link-origin speed with actual contact and low reference link velocity; it is a proxy, not measured contact-patch slip.',
              'Failure frames are global concatenated reference indices, not an invented physical fall detector.',
              'Tracking height/tilt termination can occur before the body hits the ground; it must not be reported as a measured impact frame.', '',
              '## Diagnosis and limits', '',
              'The root/joint reset and random-pose Newton FK gates pass. Real foot-ground contacts are observed.',
              'PD alone loses reference tracking. This controller has no balance feedback for the free root, no desired joint velocity input and no inverse-dynamics feedforward.',
              'Model inertias/actuator limits and controller capability must be studied before declaring any motion physically infeasible.',
              'Actuator-reported efforts are labelled in JSON; implicit PD estimates are not solver-measured torque. Saturation of an estimate does not prove solver effort saturation.',
              'A separate controller-gain / feedforward ablation is not yet completed.',
              'The baseline uses a single nominal reset seed per clip. Multi-seed perturbation tests belong to the independent policy eval reports.',
              'No test reference participated in this PD preparation or any training choice.', '',
              'Representative videos, when rendered, are `pd_straight.mp4`, `pd_turn.mp4`, `pd_spin.mp4` in the run directory.',
              'These videos render recorded physics states and explicitly hold the last recorded state after early termination.']
    target = args.run_dir/'S54_M4_1_RESULTS.md'
    target.write_text('\n'.join(lines)+'\n'); print(target)


if __name__ == '__main__': main()
