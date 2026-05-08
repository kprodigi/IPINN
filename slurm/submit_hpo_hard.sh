#!/bin/bash
# ==========================================================================
# submit_hpo_hard.sh — SLURM submission for v_20 Hard-PINN HPO
# ==========================================================================
#
# Optuna TPE search over Hard-PINN's loss weights (w_load, w_energy,
# w_monotonicity, w_angle_smooth, w_curvature) plus key optimizer knobs
# (lr, weight_decay, dropout, softplus_beta, smoothl1_beta, grad_clip).
# Architecture, warmup, and SWA settings are held fixed at v_19's HPO-found
# values; this is a re-tune around v_20's architectural E(0)=0 BC and the
# load-only checkpoint score, not a rediscovery of the architecture.
#
# Hard-PINN trials are roughly 2× the cost of Soft-PINN trials due to
# autograd through dE/dd + warmup + SWA — budget more wall-time accordingly.
#
# Usage (run from the repo root so the relative defaults resolve):
#   bash slurm/submit_hpo_hard.sh                          # defaults below
#   N_TRIALS=120 bash slurm/submit_hpo_hard.sh             # override count
#   DATA_DIR=/path OUTPUT_DIR=/scratch/hpo bash slurm/submit_hpo_hard.sh
#
# Resume after preemption: rerun the same command — Optuna picks up where
# it left off via the shared SQLite study DB.
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
N_TRIALS="${N_TRIALS:-100}"         # cut from 150 — Hard trials are intrinsically expensive (autograd through dE/dd)
N_SEEDS="${N_SEEDS:-3}"             # ensemble members per trial (raised from 2 — reduces TPE seed noise)
HPO_EPOCHS="${HPO_EPOCHS:-120}"     # cut from 200 — enough to see whether a config converges within a 120h wall
SEED="${SEED:-2026}"
WALL="${WALL:-120:00:00}"           # uniform 120h wall across all three HPO scripts
MEM="${MEM:-32G}"
JOB_NAME="${JOB_NAME:-hpo_v20_hard}"

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
#SBATCH --output=${OUTPUT_DIR}/slurm_logs/hpo_hard_%j.out
#SBATCH --error=${OUTPUT_DIR}/slurm_logs/hpo_hard_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --time=${WALL}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

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

echo "=== HPO Hard-PINN  Node: \$(hostname)  GPU: \${CUDA_VISIBLE_DEVICES:-none} ==="
echo "=== Start: \$(date)  N_TRIALS=${N_TRIALS}  N_SEEDS=${N_SEEDS}  EPOCHS=${HPO_EPOCHS} ==="

python ${TUNE} --approach hard --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --n_trials ${N_TRIALS} --n_seeds ${N_SEEDS} --hpo_epochs ${HPO_EPOCHS} --base_seed ${SEED}
EXIT_CODE=\$?

echo "=== End: \$(date)  Exit: \${EXIT_CODE} ==="
exit \${EXIT_CODE}
EOF
)

echo "Submitted Hard-PINN HPO: job ${JOB_ID}"
echo "  Logs:        ${OUTPUT_DIR}/slurm_logs/hpo_hard_${JOB_ID}.{out,err}"
echo "  Best params: ${OUTPUT_DIR}/best_params_hard.json     (written when study completes)"
echo "  Trial CSV:   ${OUTPUT_DIR}/trial_history_hard.csv"
echo "  Monitor:     squeue -j ${JOB_ID}"
echo "  Cancel:      scancel ${JOB_ID}"
