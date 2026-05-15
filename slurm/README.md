# HPC submission scripts (SLURM)

This directory holds every SLURM script the project uses, plus the
`hpc_run_stage.py` per-stage CLI dispatcher that `submit_pipeline.sh`
invokes inside each job.

All scripts assume an SDSU/SDSMT-style cluster with `cuda/12.3`,
`cudnn/9.0.0-cuda12`, and an `ipinn` conda environment. Edit the
`CUDA_MODULE` / `CUDNN_MODULE` / `CONDA_ENV` variables at the top of
each script (or override via the environment) for other clusters.

## Files

| File | Purpose | Wall | Memory |
|------|---------|------|--------|
| `submit_pipeline.sh` | Full forward + inverse + analysis pipeline (11 jobs in a dependency chain) | up to 120 h per stage | 16–64 G |
| `submit_hpo.sh` | Optuna HPO launcher, parameterised by `APPROACH=ddns\|soft\|hard` | 72 h (configurable) | 32 G |
| `submit_forward.sh` | SLURM array launcher for the per-member forward ensemble (one task per member, plus a dependent merge job) | 24 h per task | 32 G |
| `submit_inverse.sh` | SLURM array launcher for the per-member inverse-design surrogate (one task per member, plus dependent merge + analysis jobs) | 24 h per task | 32 G |
| `hpc_run_stage.py` | Per-stage CLI used by `submit_pipeline.sh` (not invoked directly) | — | — |

## Run from the repo root

Every script resolves paths relative to the repo root. Invoke like:

```bash
# From the repo root:
bash slurm/submit_pipeline.sh
APPROACH=hard N_TRIALS=120 bash slurm/submit_hpo.sh
N_CONCURRENT=10 bash slurm/submit_forward.sh
```

If you `cd` into `slurm/` and invoke from there, the default
`--data_dir ./data` resolves incorrectly. Use absolute paths in that
case:

```bash
DATA_DIR=/absolute/path/to/data bash submit_pipeline.sh
```

## Common environment overrides

| Variable | Default | What it controls |
|----------|---------|------------------|
| `DATA_DIR` | `./data` | Input directory containing `LC1.xlsx`, `LC2.xlsx` |
| `OUTPUT_DIR` | `./results_paper` (pipeline / inverse) or `./hpo_out` (HPO) | Where every artifact lands |
| `PARTITION` | `gpu` | SLURM partition |
| `CONDA_ENV` | `ipinn` | Conda environment to attach |
| `CUDA_MODULE` | `cuda/12.3` | `module load` target |
| `CUDNN_MODULE` | `cudnn/9.0.0-cuda12` | `module load` target |
| `SEED` | `2026` | Global seed; also set as `PYTHONHASHSEED` |
| `N_ENSEMBLE` | `20` | Ensemble size for pipeline / forward / inverse |
| `N_CONCURRENT` | `10` | Maximum array tasks running concurrently |
| `N_TRIALS` | `100` (HPO) | Total Optuna trials per approach |
| `N_SEEDS` | `2` (HPO) | Ensemble members trained per trial |
| `HPO_EPOCHS` | `200` (HPO) | Per-trial epoch cap during HPO |

Examples:

```bash
# Use a specific data path and a non-default output directory
DATA_DIR=/scratch/$USER/ipinn_data \
OUTPUT_DIR=/scratch/$USER/ipinn_results \
bash slurm/submit_pipeline.sh

# 200-trial Hard-PINN HPO, 4 parallel workers sharing the same study
APPROACH=hard N_TRIALS=200 N_WORKERS=4 bash slurm/submit_hpo.sh

# Forward ensemble for the Soft approach, 10 concurrent members
APPROACH=soft N_CONCURRENT=10 bash slurm/submit_forward.sh
```

## Pipeline dependency graph

```
 prep
 │
 ├──┬──┬──┬──┬──┬──┐
 │  │  │  │  │  │  │
 train (forward) train (inverse)
   for each (protocol, approach)
 │  │  │  │  │  │  │
 └──┴──┴──┴──┴──┴──┘
 │                  │
 forward_analysis  train_inverse
 │                  │
 └────────┬─────────┘
          │
   inverse_analysis
          │
       aggregate
```

`submit_pipeline.sh` submits 11 jobs and prints the IDs at the end.
Cancel the whole chain with `scancel <space-separated IDs>`.

## Determinism

Every job exports the same environment:

```bash
export PYTHONHASHSEED=${SEED}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

`PYTHONHASHSEED` is set before Python starts; `CUBLAS_WORKSPACE_CONFIG`
enables deterministic cuBLAS algorithms.

## Resuming after preemption

- **Pipeline.** The per-stage staging directory (`<output_dir>/.staging/`)
  caches every upstream result via atomic `pickle.dump` + `os.replace`.
  Resubmit failed jobs only — completed stages are skipped because
  their downstream consumers find the cached pickles.
- **HPO.** Optuna's SQLite study DB
  (`<output_dir>/hpo_study_<approach>.db` — one per approach to avoid
  concurrent schema-init races) survives any preemption. Rerun the same
  `bash slurm/submit_hpo.sh` command — Optuna picks up where it left
  off and schedules only the remainder up to `N_TRIALS`.
- **Forward / inverse arrays.** Each array task writes its partial
  bundle (`parts_<approach>/member_<idx>.pt`) on completion. Failed
  tasks write `member_<idx>.FAILED.json` instead. The downstream merge
  is dependency-chained with `afterany`, so it runs even when some
  members fail; it consolidates whatever partials were produced and
  reports missing/failed members in the merge log.

## Logs

All SLURM stdout/stderr lands in `<output_dir>/slurm_logs/`:

- `submit_pipeline.sh`: `<stage>_<jobid>.{out,err}`
- `submit_hpo.sh`: `hpo_<approach>_<jobid>_<worker>.{out,err}`
- `submit_forward.sh`: `forward_<approach>_m<idx>_<jobid>.{out,err}`,
  `forward_<approach>_merge_<jobid>.{out,err}`
- `submit_inverse.sh`: `inverse_hard_m<idx>_<jobid>.{out,err}`,
  `inverse_hard_merge_<jobid>.{out,err}`,
  `inverse_hard_analysis_<jobid>.{out,err}`
