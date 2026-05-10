#!/bin/bash
# ==========================================================================
# submit_stage2_v16.sh — Stage 2 production retrain (v_16 cfg in v_20 pipeline)
# ==========================================================================
#
# Submits THREE parallel SLURM jobs (one per approach: ddns, soft, hard).
# Each trains an M=20 ensemble at full v_16 epoch budget (600/800/800)
# using v_16's hardcoded HPO-best cfg, with v_20's architectural BC.
#
# This is the path to the documented v_16 production R² (~0.85 for Hard).
# Skips Optuna entirely — no HPO; just the retrain.
#
# Usage (run from the repo root):
#   bash slurm/submit_stage2_v16.sh
#   N_ENSEMBLE=10 bash slurm/submit_stage2_v16.sh         # smaller ensemble
#   OUTPUT_DIR=/scratch/$USER/stage2 bash slurm/submit_stage2_v16.sh
#
# Resume after preemption: each approach's bundle is written only at the
# end, so a preempted job means re-running that approach from scratch.
# Lower wall-time risk by submitting Hard first if the queue is busy.
# ==========================================================================

set -euo pipefail

# ---- Configurable variables (override via environment) -------------------
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_stage2_v16}"
PARTITION="${PARTITION:-gpu}"
CONDA_ENV="${CONDA_ENV:-ipinn}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.3}"
CUDNN_MODULE="${CUDNN_MODULE:-cudnn/9.0.0-cuda12}"
N_ENSEMBLE="${N_ENSEMBLE:-20}"
SEED="${SEED:-2026}"
WALL="${WALL:-120:00:00}"          # uniform 120h wall (Hard at M=20×800ep is the long pole)
MEM="${MEM:-32G}"

mkdir -p "${OUTPUT_DIR}/slurm_logs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$SCRIPT_DIR/../hpo/stage2_v16.py"

if [[ ! -f "$LAUNCHER" ]]; then
    echo "ERROR: $LAUNCHER not found"; exit 1
fi


submit_one() {
    local APPROACH="$1"
    local JOB_NAME="stage2_${APPROACH}"

    JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${OUTPUT_DIR}/slurm_logs/${JOB_NAME}_%j.out
#SBATCH --error=${OUTPUT_DIR}/slurm_logs/${JOB_NAME}_%j.err
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

module purge
module load ${CUDA_MODULE}
module load ${CUDNN_MODULE}

# Bypass conda activate (silently fails on some SDSMT compute nodes —
# see commit ff09ee5).  Pin the env's python by absolute path.
PYTHON_BIN="\$HOME/miniconda3/envs/${CONDA_ENV}/bin/python"
if [[ ! -x "\$PYTHON_BIN" ]]; then
    echo "ERROR: '\$PYTHON_BIN' not executable on \$(hostname)"
    exit 1
fi

echo "=== env check on \$(hostname) ==="
echo "  PYTHON_BIN: \$PYTHON_BIN"
"\$PYTHON_BIN" -c "import sys, torch; print(f'  python {sys.version.split()[0]}  torch {torch.__version__}  cuda={torch.cuda.is_available()}')"

echo "=== STAGE 2 ${APPROACH^^}  Node: \$(hostname)  GPU: \${CUDA_VISIBLE_DEVICES:-none} ==="
echo "=== Start: \$(date)  M=${N_ENSEMBLE} ==="

"\$PYTHON_BIN" ${LAUNCHER} --approach ${APPROACH} --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --n_ensemble ${N_ENSEMBLE} --seed ${SEED}
EXIT_CODE=\$?

echo "=== End: \$(date)  Exit: \${EXIT_CODE} ==="
exit \${EXIT_CODE}
EOF
)
    echo "Submitted ${JOB_NAME}: job ${JOB_ID}"
    echo "  Logs:   ${OUTPUT_DIR}/slurm_logs/${JOB_NAME}_${JOB_ID}.{out,err}"
    echo "  Result: ${OUTPUT_DIR}/stage2_${APPROACH}_results.json     (written when training completes)"
    echo "  Bundle: ${OUTPUT_DIR}/stage2_${APPROACH}_bundle.pt        (trained models + scalers)"
}

# Submit Hard first (longest runtime), then Soft, then DDNS — gives Hard
# priority in the queue when slots are scarce.
submit_one hard
submit_one soft
submit_one ddns

echo ""
echo "=== Stage 2 production retrain submitted (3 parallel jobs) ==="
echo "  Output: ${OUTPUT_DIR}"
echo "  Monitor:    squeue -u \$USER"
echo "  Per-approach R² will land in ${OUTPUT_DIR}/stage2_<approach>_results.json"
