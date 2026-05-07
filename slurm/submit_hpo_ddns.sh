#!/bin/bash
# ==========================================================================
# submit_hpo_ddns.sh — SLURM submission for v_20 DDNS HPO
# ==========================================================================
#
# Optuna TPE search over DDNS's data-fit weights and key optimizer
# hyperparameters on the unseen-θ=60° protocol.  DDNS is the data-driven
# baseline (no physics losses), so the search space is the smallest of the
# three approaches — fewer trials are sufficient.
#
# Single-GPU, single-node; resumable on SLURM preemption (the SQLite study
# DB is reused automatically the next time you submit).
#
# Usage (run from the repo root so the relative defaults resolve):
#   bash slurm/submit_hpo_ddns.sh                          # defaults below
#   N_TRIALS=100 bash slurm/submit_hpo_ddns.sh             # override count
#   DATA_DIR=/path OUTPUT_DIR=/scratch/hpo bash slurm/submit_hpo_ddns.sh
#
# Resume after preemption: just rerun the same command — Optuna sees the
# already-completed trials in $OUTPUT_DIR/tune_v20_study.db and only
# schedules the remainder up to N_TRIALS.
# ==========================================================================

set -euo pipefail

# ---- Configurable variables (override via environment) -------------------
# DATA_DIR defaults to the repo's ``data/`` directory (LC1.xlsx, LC2.xlsx).
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./hpo_v20}"
PARTITION="${PARTITION:-gpu}"
CONDA_ENV="${CONDA_ENV:-ipinn}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.3}"
CUDNN_MODULE="${CUDNN_MODULE:-cudnn/9.0.0-cuda12}"
N_TRIALS="${N_TRIALS:-80}"          # smaller search space → fewer trials needed
N_SEEDS="${N_SEEDS:-3}"             # ensemble members per trial (raised from 2 — reduces TPE seed noise)
HPO_EPOCHS="${HPO_EPOCHS:-200}"     # per-trial epoch cap (production: 600)
SEED="${SEED:-2026}"
WALL="${WALL:-48:00:00}"            # DDNS is fastest of the three; 48h ample
MEM="${MEM:-32G}"
JOB_NAME="${JOB_NAME:-hpo_v20_ddns}"

mkdir -p "${OUTPUT_DIR}/slurm_logs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# tune_v20.py lives in ../hpo/ relative to this script
TUNE="$SCRIPT_DIR/../hpo/tune_v20.py"

if [[ ! -f "$TUNE" ]]; then
    echo "ERROR: $TUNE not found"; exit 1
fi

JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${OUTPUT_DIR}/slurm_logs/hpo_ddns_%j.out
#SBATCH --error=${OUTPUT_DIR}/slurm_logs/hpo_ddns_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --time=${WALL}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# Determinism (matches submit_pipeline.sh).  PYTHONHASHSEED must be set
# BEFORE Python starts; CUBLAS_WORKSPACE_CONFIG enables deterministic cuBLAS.
export PYTHONHASHSEED=${SEED}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPTUNA_STORAGE_FILE_LOCK="${OPTUNA_STORAGE_FILE_LOCK:-false}"

module purge
module load ${CUDA_MODULE}
module load ${CUDNN_MODULE}

source \$HOME/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

echo "=== HPO DDNS  Node: \$(hostname)  GPU: \${CUDA_VISIBLE_DEVICES:-none} ==="
echo "=== Start: \$(date)  N_TRIALS=${N_TRIALS}  N_SEEDS=${N_SEEDS}  EPOCHS=${HPO_EPOCHS} ==="

python ${TUNE} \\
    --approach ddns \\
    --data_dir ${DATA_DIR} \\
    --output_dir ${OUTPUT_DIR} \\
    --n_trials ${N_TRIALS} \\
    --n_seeds ${N_SEEDS} \\
    --hpo_epochs ${HPO_EPOCHS} \\
    --base_seed ${SEED}
EXIT_CODE=\$?

echo "=== End: \$(date)  Exit: \${EXIT_CODE} ==="
exit \${EXIT_CODE}
EOF
)

echo "Submitted DDNS HPO: job ${JOB_ID}"
echo "  Logs:        ${OUTPUT_DIR}/slurm_logs/hpo_ddns_${JOB_ID}.{out,err}"
echo "  Best params: ${OUTPUT_DIR}/best_params_ddns.json     (written when study completes)"
echo "  Trial CSV:   ${OUTPUT_DIR}/trial_history_ddns.csv"
echo "  Monitor:     squeue -j ${JOB_ID}"
echo "  Cancel:      scancel ${JOB_ID}"
