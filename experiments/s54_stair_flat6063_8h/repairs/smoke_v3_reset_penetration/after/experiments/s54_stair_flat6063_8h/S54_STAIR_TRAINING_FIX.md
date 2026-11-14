# Stair training startup repair — 2026-11-11

## Evidence and repair

The v3 failure was a real physics spike, not a negative-reward warning: every captured episode was at step 0; the largest joint velocity exceeded 5000 rad/s. FK of the saved reset poses showed 2–7 cm foot penetration. These states are retained in `logs/s54_stair_flat6063_smoke_v3/failure_physics_transition.pt`.

The reset fix draws from the original Flat perturbation ranges and rejects poses intersecting the original terrain collision geometry. Boxes use 15-axis SAT; capsules/spheres use exact segment/AABB distance; cylinders use conservative oriented boxes. The original 1 mm penetration tolerance is retained. After 64 unsuccessful draws, only a validated pose of the same reference frame may be used. The original motion library remains intact. Two marginally penetrating Flat frames (global IDs 11142 and 12405) are excluded from reset starts only; they remain future reference goals. Every sampling bin retains feasible starts. The 436 stair frames pass the same geometry check.

Reset conditioning alone was insufficient: v5/v6 showed a flat-ground contact spike after a few normal steps. Readback confirmed the temporary flat collider was a translated native Box: half extents `(12,12,.05)`, transform `(20,0,-.05)`, gap `.01`. Cross-world contact count was zero.

Saved-state Newton comparisons:

| Case | Max root speed m/s | Max root angular speed rad/s | Max joint speed rad/s |
|---|---:|---:|---:|
| Translated Box, captured v5 state | 74.21 | 650.36 | 712.02 |
| Original Flat analytic Plane, same state | 1.33 | 1.62 | 9.60 |
| Same finite Box geometry baked into a triangle Mesh | 1.34 | 1.84 | 9.52 |

These tests isolate an unstable contact path for the translated native Box in the installed Newton/MJWarp stack. They do not establish a universal Box bug or identify a vendor-kernel line. The controlled full-size v6/v7 runs both used per-substep collision refresh: v6 with Box failed in warmup; v7 with the equivalent Mesh completed all 64 iterations and 851968 real transitions.

Only the synthetic Flat training region changes representation. Its eight corner coordinates, 24×24×0.1 m dimensions, top z=0 and world bounds `[8,-12,-.1]` to `[32,12,0]` are preserved. The authoritative stair NPZ/Mesh/Height Map/Footstep and S54 USD are unchanged. No infinite plane is added under the stairs.

The Newton adapter also explicitly sets `collision_decimation=1`: contact is refreshed every 5 ms solver substep rather than the default once per 20 ms tick. The 20 ms control DT, four solver substeps, actuator limits, friction randomization and gravity are unchanged. This timing is recorded in checkpoint physical metadata and evaluation summaries.

The v4 reward consistency failure was a float32 rotation-matrix/quaternion roundtrip difference (~4e-5 reward). Real-step Stair rewards now use the exact 71D saved replay state. The strict 1e-5 timing/consistency gate is retained; the completed v7 Smoke reported zero reward-sync error. Original Flat environments retain their previous reward path.

## Learning smoke versus formal training

v7 completed physics, WM, Actor/Critic BPTT, gradients and checkpoints. Its WM loss fell from about .35 to .14 and 8-step error from .55 to .24. Independent evaluation correctly rejected this Smoke policy: 32 premature Actor updates after only 32 WM-only iterations degraded Flat tracking. No formal run inherited those weights.

The frozen original Flat 6063 checkpoint still reproduced **0.957207** normalized return on `straight_flat_18`, surviving its full 11.72 seconds with the new contact adapter. A diagnostic using original Actor weights plus v7 online normalizers reproduced **0.957095**; normalization drift does not explain the v7 failure. Both diagnostic outputs and the ablation checkpoint are retained under `repairs/smoke_v3_reset_penetration/` and are not final policies.

The technical Smoke is therefore 64 iterations with 63 WM-only iterations and one actual Actor/Critic update to verify BPTT and optimizer checkpointing. Its trained checkpoint is still independently evaluated under the unchanged Flat-retention gate (each selected Flat reference, seeds 0/1/2, mean return >=.85). The original failed v7 evaluation remains available. This is a technical smoke, not convergence evidence.

Formal training still initializes from the untouched Flat 6063 checkpoint, uses all 39 original core configuration values, and waits the original **1000 WM-only iterations** before Actor updates. Its ceiling is 10000 iterations / 28800 seconds, whichever occurs first. No PPO, Foot Encoder, new command policy or root-state writes during dynamics were introduced. Periodic frozen Newton evaluation runs every 1000 iterations; SwanLab receives scalar data only.

## Files and evidence

- `supermimic/stair_reset.py`: geometry-conditioned reset candidates.
- `supermimic/env.py`: reset-only hook and optional canonical replay-state reward path.
- `supermimic/stair_runtime.py`: equivalent finite Mesh flat collider and substep contact refresh.
- `supermimic/stair_training.py`: feasible reset starts, task contracts, source transfer checks.
- `supermimic/train.py`: reset diagnostics, first-iteration progress/checkpoint, existing WM/BPTT loop.
- `tools/diagnose_s54_stair_reset.py`: reproducible saved-state Newton A/B tool; no optimization.
- `repairs/smoke_v3_reset_penetration/newton_ab/`, `newton_conditioned/`, `box_v5/`, `plane_v5/`, `mesh_v5/`: retained physical tests.
- `repairs/smoke_v3_reset_penetration/source_flat_current/`, `source_weights_smoke_norms/`: frozen baseline/normalizer ablation.
- `logs/s54_stair_flat6063_smoke_v3` through `v7`: all failed/completed attempts preserved.
- `logs/s54_stair_flat6063_smoke_v8`: final technical smoke and its frozen evaluation.
- `smoke_gate_v8.json`: actual gate result, generated only after execution.
- `logs/s54_stair_flat6063_8h/`: formal run, created only when the gate passes.

Run (already invoked for this repair; do not launch a duplicate):

```bash
export ISAAC_PYTHON=/mnt/sage_data/workspace/Toolbox/IsaacLab_3_0/.venv/bin/python
"$ISAAC_PYTHON" tools/run_s54_stair.py night
tail -F logs/s54_stair_flat6063_night_v8.log
```

Monitor the formal run with `tail -F logs/s54_stair_flat6063_8h/stdout.log`. Current execution status and final repair verification are recorded separately in `verification.json`; this document makes no convergence or stair-generalization claim.

## Independent evaluation timeout

The v8 Smoke completed with finite gradients, zero reward-sync error and Flat straight mean normalized return 0.919984. Its first independent evaluation attempt exhausted the old 300-second **total** timeout while evaluating Turn (only seed 0 finished). This was an orchestration timeout, not a completed retention verdict. The failed gate/status/partial traces are preserved. Both the total evaluation timeout and final-evaluation budget reserve are now 1200 seconds; the 8h cumulative cap, physics limits and retention threshold are unchanged. The gate is rerun against the same immutable v8 Smoke checkpoint before formal launch.

## Verified execution

The retry completed all 4 references × 3 seeds. Mean normalized returns were Flat Straight **0.919845**, Turn **0.946847**, Spin **0.951152** (all nine Flat episodes completed) and Stair **0.111858** (none of three stair episodes completed). The unchanged >=.85 per-Flat-reference gate passed. The stair score is a training-action Smoke diagnostic, not a stair convergence or held-out result.

Formal training started in tmux at **2026-11-11 02:45 Asia/Shanghai**. Its actual args match all 39 pinned Flat core fields. Iteration 1 collected **528384** cumulative real transitions, saved `logs/s54_stair_flat6063_8h/ckpt_000001.pt`, and has finite WM/Policy/Critic state. The WM optimizer has 16 steps; Actor/Critic optimizers are correctly empty during the original first-1000-iterations WM-only phase. The run remains active; it has not finished or demonstrated stair convergence.

Core-only SwanLab upload is active and has acknowledged iteration 1: [formal run](https://swanlab.cn/@yagami_light/SuperMimic-S54-Stair/runs/9f95eb54e9684e1b). No GIF or video upload was requested. The total budget remains 28800 seconds, including periodic evaluations; each complete evaluation and the final reserve allow up to 1200 seconds. Both the failed 300-second attempt and completed retry remain available.

Final code/config snapshots, 25 passing CPU regression test results, saved-state Newton A/B data and machine-readable diagnosis are retained under `repairs/smoke_v3_reset_penetration/`. See `verification.json` for checkpoint hash and exact launch metadata.
