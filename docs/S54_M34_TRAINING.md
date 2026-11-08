# M3.4 → M4 short training entry

Run from `supermimic_s54_main` on the Docker host:

```bash
bash tools/run_s54_m34.sh train s54_m34_run01 configs/s54_m34_4060.json
docker exec -it supermimic_s54_m34 tmux attach -t s54_m34_run01
```

The launcher creates a dedicated container if absent and checks its mount.
The older `supermimic_s54` container in this installation mounts a different
checkout. It must not be reused without checking the mount. The original
containers, code paths, CSV data and checkpoints are preserved.

The job runs CPU regression, a real Newton PD baseline with random-pose FK
regression, a 30-iteration WM/BPTT/Critic smoke, and a fresh unified policy
training run. All training parameters come from the current JSON; user edits are
preserved. The initial verified short run used 128 environments, 100 WM-only
iterations and 1,000 total iterations. It was a bounded experiment, not a
convergence claim. Edit the JSON before launching a new run to change the budget.
`wall_seconds` includes training initialization and periodic evaluations;
CPU preflight, PD, smoke and final rendering add to total pipeline time.

Train data is exactly the 14 accepted `train/*/*.npz` entries in `manifest.json`.
The adapter retains 50 Hz, supplied world link-origin velocities and wxyz
quaternions. It validates checksums, names/order, shapes, units, clock, quaternion
norms, limits, source-model hashes, and all-frame 30-body FK/velocity consistency.
It never reuses the old CSV collision certificate or claims a continuous
collision certificate. Newton random-pose validation remains a separate gate.

The unchanged SuperMimic train loop obtains WM transition labels from Newton
replay, and optimizes the Motion Encoder/shared decoder through the learned WM
with BPTT and Critic. No PPO, foot encoder or command policy is introduced.
Sampling is class → clip → adaptive return-weighted bins. Initial weights are
80/10/10 straight/turn/spin; after `m34_straight_iters` they become exactly equal.
These weights are precommitted and receive no test feedback. There is no new
standing-only curriculum in this initial integration.

Every `eval_every=1000` completed iterations (including WM warmup), training
saves an immutable snapshot, pauses, and launches separate Newton processes to
evaluate train and test. Training then continues in the same process with its
existing replay, optimizers, RNG, normalization and simulator state. Smoke
disables periodic evaluation. Set `eval_every=0` to request final evaluation only.
Seed 0 is nominal; other configured `eval_seeds` use separately reported reset
perturbations. No test arrays are opened by the optimizer process or its
normalizers, and test results never select weights, samplers or checkpoints.
If the final iteration was already evaluated, its results are reused; an
off-boundary stop gets one final frozen evaluation. Tests
Forward/Turn/Spin measure same-template parameter generalization; Backward is
cross-mode extrapolation. Each episode stops at physical tracking termination
or reference completion. Diagnostic truncation never counts as survival.

Artifacts in `logs/s54_m34_run01/`:

- `pipeline_config.json`, `source/`, `pipeline_status.json`, `stdout.log`.
- `pd/results.json`, recorded Newton trajectories.
- `args.json`, `last.pt`, atomic `latest.pt`, periodic `ckpt_*.pt`.
- `train_log.csv`, `gradient_terms.jsonl`, `runtime_stats.json`.
- `evals/iter_001000/checkpoint.pt`, `summary.json`, `status.json` (then 002000, etc.).
- `evals/*/attempt_*/train_*/results.json`, `test_*/results.json` and NPZ trajectories.
- `train_*.mp4`, `test_*.mp4`: rendering of actual Newton rollout states.
- `training_curves.png`, `S54_M4_2_RESULTS.md`.

The original smoke attempts are retained under `logs/s54_m34_smoke*`.
Two setup failures discovered and fixed: TF32 precision during strict FK
validation, and the worktree Git pointer being inaccessible inside a container.
Only FK/reference construction temporarily disables TF32; network training
retains the existing precision settings. The host supplies the source SHA.

Resume with all model/optimizer/RNG state, fresh replay and the existing
WM-only replay recovery. Set `iterations` in the copied config to a total target
greater than the checkpoint's iteration:

```bash
bash tools/run_s54_m34.sh resume s54_m34_resume01 configs/s54_m34_longer.json logs/s54_m34_run01/last.pt
```

For an independent frozen evaluation only:

```bash
docker exec supermimic_s54_m34 python tools/m34_rollout.py \
  --dataset data/s54_flat_dataset/dataset_export --split test \
  --checkpoint logs/s54_m34_run01/last.pt --out results/s54_m34_test01 --seeds 0
```

Use a new output directory each time; diagnostics are never overwritten.
Actuator-reported torque fields can be implicit-controller estimates and are
labelled accordingly. They are not presented as measured solver effort.
SwanLab host credentials/SDK were not found in this checkout; CSV/JSON logs
are always local and can be uploaded with the existing `tools/sync_swanlab.py`.
The uploader defaults to `--core_only`: only whitelisted core hyperparameters
and WM/Policy/Critic losses, simulator reward, episode return/length/termination,
Actor gradient norm, throughput and simulated-step count are sent. Full task and
reward dictionaries, file paths, clip lists, head diagnostics and arbitrary
curriculum telemetry remain local. A failed training job is marked failed after
its final available metrics are drained. Automatic host/device probes are off.
Three explicit same-source GMR-Raw PD comparisons are supported separately;
their references retain their own generation timing. This is not a frame-aligned
controller ablation or a continuous collision certificate. PD failure alone
does not invalidate a motion.

Validation: the CPU adapter/FK/gradient regression and real Newton random-pose
gate passed for the initial integration. Periodic-eval orchestration and SwanLab
whitelist regressions cover immutable checkpoints, train/test seed isolation,
failed-attempt retention, final-result reuse and filtered cloud uploads.
