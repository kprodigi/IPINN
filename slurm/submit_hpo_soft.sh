#!/bin/bash
# ==========================================================================
# submit_hpo_soft.sh — SLURM submission for v_20 Soft-PINN HPO
# ==========================================================================
#
# Optuna TPE search over Soft-PINN's physics-loss weights and key optimizer
# hyperparameters on the unseen-θ=60° protocol.  Single-GPU, single-node;
# resumable on SLURM preemption (the SQLite study DB is reused automatically
# the next time you submit).
#
# Usage (run from the repo root so the relative defaults resolve):
#   bash slurm/submit_hpo_soft.sh                          # defaults below
#   N_TRIALS=100 bash slurm/submit_hpo_soft.sh             # override count
#   DATA_DIR=/path OUTPUT_DIR=/scratch/hpo bash slurm/submit_hpo_soft.sh
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
N_TRIALS="${N_TRIALS:-120}"         # target completed trials (search adds hidden_layers + batch_size categoricals)
N_SEEDS="${N_SEEDS:-3}"             # ensemble members per trial (raised from 2 — reduces TPE seed noise)
HPO_EPOCHS="${HPO_EPOCHS:-200}"     # per-trial epoch cap
SEED="${SEED:-2026}"
WALL="${WALL:-72:00:00}"            # 80 trials × ~5–15 min/trial fits in <24h on a V100
MEM="${MEM:-32G}"
JOB_NAME="${JOB_NAME:-hpo_v20_soft}"

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
#SBATCH --output=${OUTPUT_DIR}/slurm_logs/hpo_soft_%j.out
#SBATCH --error=${OUTPUT_DIR}/slurm_logs/hpo_soft_%j.err
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
# Optuna stores the study DB at ${OUTPUT_DIR}/tune_v20_study.db; if your
# scratch is NFS, set OPTUNA_STORAGE_FILE_LOCK=true to avoid lock issues.
export OPTUNA_STORAGE_FILE_LOCK="${OPTUNA_STORAGE_FILE_LOCK:-false}"

module purge
module load ${CUDA_MODULE}
module load ${CUDNN_MODULE}

source \$HOME/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

echo "=== HPO Soft-PINN  Node: \$(hostname)  GPU: \${CUDA_VISIBLE_DEVICES:-none} ==="
echo "=== Start: \$(date)  N_TRIALS=${N_TRIALS}  N_SEEDS=${N_SEEDS}  EPOCHS=${HPO_EPOCHS} ==="

python ${TUNE} \\
    --approach soft \\
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

echo "Submitted Soft-PINN HPO: job ${JOB_ID}"
echo "  Logs:        ${OUTPUT_DIR}/slurm_logs/hpo_soft_${JOB_ID}.{out,err}"
echo "  Best params: ${OUTPUT_DIR}/best_params_soft.json     (written when study completes)"
echo "  Trial CSV:   ${OUTPUT_DIR}/trial_history_soft.csv"
echo "  Monitor:     squeue -j ${JOB_ID}"
echo "  Cancel:      scancel ${JOB_ID}"
