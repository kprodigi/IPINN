# Hyperparameter Optimization (Optuna TPE)

This directory holds `tune_v20.py`, the Optuna entry point used to retune the
forward training hyperparameters of each approach (DDNS, Soft-PINN, Hard-PINN)
on the unseen-θ=60° validation protocol before running the full pipeline.

## Why HPO is split out

The forward training search space differs by approach (Soft and DDNS share
the `SoftPINNNet` backbone; Hard adds warmup and SWA knobs that DDNS/Soft
don't have), and Hard-PINN trials cost roughly 2× a Soft-PINN trial. Running
the three studies as separate SLURM jobs lets each get the right wall budget
and gives independent SQLite study databases that survive preemption.

## Recommended workflow (4 stages)

```
Stage 1 ── HPO forward      ─→ best hyperparameters per approach
Stage 2 ── apply best       ─→ patch composite_design_v20.get_model_config
Stage 3 ── retrain forward  ─→ full-budget forward models with new HPs
Stage 4 ── HPO inverse      ─→ tune GP-BO + classifier weights on Stage 3 models
```

Stage 1 is what this directory covers. Stages 2–4 are scheduled separately
and use the artifacts written here.

## Search space summary

| Knob | DDNS | Soft-PINN | Hard-PINN |
|------|------|-----------|-----------|
| `hidden_layers` | {64-32, 128-64, 128-64-32, 256-128, 256-128-64} | same as DDNS | {32-32, 64-32, 64-64, 128-64, 128-64-32} |
| `batch_size` | {32, 64, 128} | {32, 64, 128} | {8, 16} |
| `lr`, `weight_decay`, `dropout` | log-scaled ranges | same | same |
| `softplus_beta`, `smoothl1_beta`, `grad_clip` | — | tuned | tuned |
| Physics weights | — | `w_phys` | `w_load`, `w_energy`, `w_monotonicity`, `w_angle_smooth`, `w_curvature` |
| `warmup_epochs` | — | — | [40, 150] |
| `swa_pct` | — | — | [0.10, 0.35] |

`hidden_layers` for Hard-PINN is bounded smaller than for Soft/DDNS because
Hard-PINN computes `dE/dd` via autograd through the network — wider networks
inflate trial cost without reliably improving validation R².

The TPE sampler is configured with `n_startup_trials=15`, `n_ei_candidates=50`,
`multivariate=True`, `group=True` to handle correlated knobs jointly.

## Running locally (smoke)

```bash
# Tiny 5-trial × 30-epoch sanity check (~5 min on CPU):
python hpo/tune_v20.py --approach soft --output_dir ./hpo_dry --dry_hpo
```

## Running on SLURM

```bash
# From the repo root.  Each call submits one SLURM job:
bash slurm/submit_hpo_ddns.sh        # 80 trials,  48h wall
bash slurm/submit_hpo_soft.sh        # 120 trials, 72h wall
bash slurm/submit_hpo_hard.sh        # 150 trials, 96h wall
```

To run more or fewer trials override `N_TRIALS`:

```bash
N_TRIALS=200 bash slurm/submit_hpo_hard.sh
```

To resume after preemption simply rerun the same command — Optuna sees the
already-completed trials in `${OUTPUT_DIR}/tune_v20_study_<approach>.db`
and only schedules the remainder up to `N_TRIALS`.  Each approach owns
its own SQLite file so concurrent SLURM jobs don't race on schema init.

## Outputs

Each approach writes the following to `--output_dir` (default `./hpo_v20`):

```
<output_dir>/
  tune_v20_study_<approach>.db     # per-approach SQLite study (resumable)
  best_params_<approach>.json      # copy-pasteable into composite_design_v20
  trial_history_<approach>.csv     # every trial's params + R² + duration
  hpo_log_<approach>.txt           # full Optuna log (per-trial reports)
```

## Objective

For every trial the objective is the **mean validation load R²** across
`--n_seeds` ensemble members on the unseen-θ=60° protocol.  The default
`n_seeds=3` (raised from v_19's 2) reduces seed noise that misled TPE on
marginal hyperparameter differences.

## Determinism

The SLURM scripts export `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`OMP_NUM_THREADS`, and `MKL_NUM_THREADS` to match `slurm/submit_pipeline.sh`'s
determinism profile.  Runs on the same hardware with the same `--base_seed`
are bitwise reproducible.
