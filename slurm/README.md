# HPC Submission Scripts (SLURM)

This directory holds every SLURM script the project uses, plus the
`hpc_run_stage.py` per-stage CLI dispatcher that `submit_pipeline.sh`
invokes inside each job.

All scripts assume an SDSU/SDSMT-style cluster with `cuda/12.3`,
`cudnn/9.0.0-cuda12`, and a `conda activate ipinn` environment.  Edit the
`CUDA_MODULE` / `CUDNN_MODULE` / `CONDA_ENV` variables at the top of each
script (or override via the environment) for other clusters.

## Files

| File | Purpose | Wall | Memory |
|------|---------|------|--------|
| `submit_pipeline.sh`  | Full forward + inverse + analysis pipeline (11 jobs in dependency chain) | up to 120h per stage | 16–64 G |
| `submit_hpo_ddns.sh`  | Optuna HPO for DDNS                  | 120h | 32 G |
| `submit_hpo_soft.sh`  | Optuna HPO for Soft-PINN             | 120h | 32 G |
| `submit_hpo_hard.sh`  | Optuna HPO for Hard-PINN             | 120h | 32 G |
| `hpc_run_stage.py`    | per-stage CLI used by `submit_pipeline.sh` (not invoked directly) | — | — |

## Run from the repo root

Every script resolves paths relative to the repo root.  Invoke them like:

```bash
# From the repo root:
bash slurm/submit_pipeline.sh
bash slurm/submit_hpo_soft.sh
```

If you `cd` into `slurm/` and run the scripts from there, the default
`--data_dir ./data` will resolve incorrectly.  Use absolute paths in that
case:

```bash
DATA_DIR=/absolute/path/to/data bash submit_pipeline.sh
```

## Common environment overrides

| Variable | Default | What it controls |
|----------|---------|------------------|
| `DATA_DIR` | `./data` | Input directory containing `LC1.xlsx`, `LC2.xlsx` |
| `OUTPUT_DIR` | `./results_paper` (pipeline) or `./hpo_v20` (HPO) | Where every artifact lands |
| `PARTITION` | `gpu` | SLURM partition |
| `CONDA_ENV` | `ipinn` | Conda environment to activate |
| `CUDA_MODULE` | `cuda/12.3` | `module load` target |
| `CUDNN_MODULE` | `cudnn/9.0.0-cuda12` | `module load` target |
| `SEED` | `2026` | Global seed; also set as `PYTHONHASHSEED` |
| `N_ENSEMBLE` | `20` | Pipeline ensemble size |
| `N_TRIALS` | 80 / 120 / 150 | HPO trial budget per approach |
| `N_SEEDS` | `3` | HPO members trained per trial |
| `HPO_EPOCHS` | `200` | HPO per-trial epoch cap |

Examples:

```bash
# Use a specific data path and a non-default output directory:
DATA_DIR=/scratch/$USER/ipinn_data \
OUTPUT_DIR=/scratch/$USER/ipinn_results \
bash slurm/submit_pipeline.sh

# 200-trial Hard-PINN HPO instead of the 150 default:
N_TRIALS=200 bash slurm/submit_hpo_hard.sh
```

## Pipeline dependency graph

```
                    prep
                     |
   +------+------+------+------+------+------+------+
   |      |      |      |      |      |      |      |
  train  train  train  train  train  train  train
  rand   rand   rand   uns    uns    uns    inv
  ddns   soft   hard   ddns   soft   hard
   |      |      |      |      |      |      |
   +------+------+------+------+------+------+
                          |
                  forward_analysis      train_inverse
                          |                    |
                          +--------------------+
                                    |
                            inverse_analysis
                                    |
                                aggregate
```

`submit_pipeline.sh` submits 11 jobs and prints the IDs at the end.  Cancel
the whole chain with `scancel <space-separated IDs>`.

## Determinism

Every job exports the same trio that the single-machine path uses:

```bash
export PYTHONHASHSEED=${SEED}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

`PYTHONHASHSEED` is set before Python starts; `CUBLAS_WORKSPACE_CONFIG`
enables deterministic cuBLAS algorithms.

## Resuming after preemption

- **Pipeline:** the per-stage staging directory (`<output_dir>/.staging/`)
  caches every upstream result via atomic `pickle.dump` + `os.replace`.
  Resubmit failed jobs only — completed stages will be skipped because
  their downstream consumers find the cached pickles.
- **HPO:** Optuna's SQLite study DB
  (`<output_dir>/tune_v20_study_<approach>.db` — one per approach to
  avoid concurrent schema-init races) survives any preemption.  Just
  rerun the same `bash slurm/submit_hpo_*.sh` command — Optuna picks up
  where it left off and only schedules the remainder up to `N_TRIALS`.

## Logs

All SLURM stdout/stderr lands in `<output_dir>/slurm_logs/`:

- `submit_pipeline.sh`: `<stage>_<jobid>.{out,err}`
- `submit_hpo_*.sh`:    `hpo_<approach>_<jobid>.{out,err}`
