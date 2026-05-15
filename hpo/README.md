# Hyperparameter optimisation (Optuna TPE)

This directory holds the Optuna-based hyperparameter-optimisation (HPO)
driver and the per-member training launchers used to assemble the forward
and inverse-design ensembles after the optimal hyperparameters are
known.

## File layout

| File | Purpose |
|------|---------|
| `hpo_search.py` | Optuna TPE entry point. Tunes the hyperparameters of one approach (DDNS / Soft-PINN / Hard-PINN) on the unseen-angle protocol. SQLite-backed, resumable, supports multiple workers attached to one study. |
| `forward_member.py` | Trains a single ensemble member of the unseen-angle forward surrogate. Designed for one SLURM array task per member. |
| `forward_merge.py` | Aggregates the per-member partial bundles, applies the Tukey-fence convergence filter, writes the final forward bundle and results JSON. |
| `inverse_member.py` | Trains a single ensemble member of the full-data Hard-PINN inverse-design surrogate. One SLURM array task per member. |
| `inverse_merge.py` | Aggregates the inverse-design partials and writes the pretrained inverse surrogate bundle. |
| `compare_methods.py` | Cross-approach comparison: physics-correctness metrics on saved bundles (E(0), F(0), R² lift, etc.). |

## Workflow

```
1. hpo_search.py     ── one Optuna study per approach
                        on the unseen-angle protocol
2. (paste best params into composite_design.get_model_config)
3. forward_member.py ── M parallel SLURM tasks per approach
4. forward_merge.py  ── one merge per approach (Tukey filter + bundle)
5. inverse_member.py ── M parallel SLURM tasks (full-data Hard-PINN)
6. inverse_merge.py  ── one merge step (writes pretrained surrogate)
7. composite_design.py --mode inverse --use_pretrained_inverse ...
                     ── classifier training + GP-BO + Pareto +
                        sensitivity studies + figures
```

## Search-space summary

| Knob | DDNS | Soft-PINN | Hard-PINN |
|------|------|-----------|-----------|
| `hidden_layers` | `{64-32, 128-64, 128-64-32, 256-128, 256-128-64}` | same as DDNS | `{32-32, 64-32, 64-64, 64-64-64, 128-64, 128-64-32}` |
| `batch_size` | `{32, 64, 128}` | `{32, 64, 128}` | `{16, 32}` |
| `lr`, `weight_decay`, `dropout` | log-scaled ranges | log-scaled ranges | log-scaled ranges |
| `softplus_beta`, `smoothl1_beta` | tuned | tuned | tuned |
| Data weights | `w_data_load`, `w_data_energy` | `w_data_load`, `w_data_energy` | `w_load`, `w_energy` |
| Work-energy residual weight | — | `w_phys` (log-scaled) | architectural (no weight) |
| Paired E(0)/F(0) BC penalty | — | `w_bc` (log-scaled) | architectural (no weight) |
| Collocation ratio | — | `colloc_ratio` | — |
| `grad_clip` | — | — | tuned |
| `warmup_epochs` | — | — | `[40, 150]` |
| `swa_pct` | — | — | `[0.10, 0.35]` |

Hard-PINN's `hidden_layers` choices are smaller than Soft/DDNS because
computing `F = dE/dd` via autograd through the network makes wider
architectures expensive without reliably improving validation R².

The Hard-PINN search space contains no physics-loss weights because the
three core physics constraints (work-energy identity, `E(0) = 0`,
`F(0) = 0`) are enforced architecturally via slope subtraction. The
Soft-PINN search space contains the *same three constraints* as soft
penalties (`w_phys` for the work-energy residual, `w_bc` for the paired
boundary penalty).

## Sampler configuration

The Optuna TPE sampler is configured with:

- `n_startup_trials = 15` — random search before TPE engages
- `multivariate = True` — jointly model correlated parameters
- `group = True` — keep categoricals grouped under multivariate
- `seed = --seed` argument — full reproducibility

A small set of warm-start parameter dictionaries (see `WARM_START` at
the top of `hpo_search.py`) is enqueued before any TPE proposals so the
search starts with at least one informed prior. Pass `--no_warm_starts`
to disable.

## Running locally (smoke)

```bash
# Tiny 5-trial × 20-epoch sanity check (~3 min on CPU)
python hpo/hpo_search.py --approach hard --n_trials 5 --hpo_epochs 20 \
    --n_seeds 1 --output_dir ./hpo_smoke --dry_run --force_cpu
```

## Running on SLURM

```bash
# From the repo root, one approach per invocation
APPROACH=ddns N_TRIALS=80  bash slurm/submit_hpo.sh
APPROACH=soft N_TRIALS=120 bash slurm/submit_hpo.sh
APPROACH=hard N_TRIALS=100 HPO_EPOCHS=120 bash slurm/submit_hpo.sh
```

Multiple parallel workers can attach to the same study by passing
`N_WORKERS > 1`:

```bash
APPROACH=hard N_TRIALS=200 N_WORKERS=4 bash slurm/submit_hpo.sh
```

To resume after preemption simply rerun the same command — Optuna
detects the already-completed trials in
`${OUTPUT_DIR}/hpo_study_<approach>.db` and only schedules the
remainder up to `N_TRIALS`. Each approach owns its own SQLite file so
concurrent SLURM jobs do not race on schema initialisation.

## Outputs

Each approach writes to `--output_dir` (default `./hpo_out`):

```
<output_dir>/
    hpo_study_<approach>.db          per-approach SQLite study (resumable)
    hpo_best_params_<approach>.json  copy-pasteable best parameters
    hpo_history_<approach>.csv       every trial's params + R² + state
    hpo_log_<approach>.txt           full Optuna log
    slurm_logs/                      per-task stdout/stderr
```

## Objective

For every trial the objective is the **mean validation load R²** across
`--n_seeds` ensemble members on the unseen-angle protocol
(θ\* = 60°). The default `n_seeds = 2` is a reasonable trade-off
between per-trial noise and per-trial wall-clock; raise to `3` for
higher-fidelity rankings near the search ceiling.

## Determinism

`PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`OMP_NUM_THREADS=4`, and `MKL_NUM_THREADS=4` are exported by the SLURM
wrapper. Runs on the same hardware with the same `--seed` are bitwise
reproducible.
