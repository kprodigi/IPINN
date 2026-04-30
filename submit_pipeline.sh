#!/bin/bash
# ==========================================================================
# submit_pipeline.sh — SDSU HPC SLURM submission for IPINN parallel pipeline
# ==========================================================================
#
# Usage:
#   bash submit_pipeline.sh                          # defaults
#   DATA_DIR=/path/to/data OUTPUT_DIR=/path/to/results bash submit_pipeline.sh
#   bash submit_pipeline.sh --dry_run                # CI-scale test
#
# Submits 11 jobs with dependency chains:
#   prep  →  7 parallel training jobs  →  forward_analysis + inverse_analysis  →  aggregate
#
# Monitor:  squeue -u $USER
# Cancel:   scancel <jobids printed at end>
# ==========================================================================

set -euo pipefail

# ---- Configurable variables (override via environment or flags) ----
DATA_DIR="${DATA_DIR:-.}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_paper}"
PARTITION="${PARTITION:-gpu}"
CONDA_ENV="${CONDA_ENV:-ipinn}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.3}"     # default on the SDSMT cluster
CUDNN_MODULE="${CUDNN_MODULE:-cudnn/9.0.0-cuda12}"
N_ENSEMBLE="${N_ENSEMBLE:-20}"
SEED="${SEED:-2026}"
DRY_RUN_FLAG=""
STRICT_FLAG="--strict_paper"
# Per user instruction: every stage runs on the GPU partition with a single
# GPU allocated, even CPU-bound prep/aggregate.  This avoids cross-partition
# queue waits and keeps the dependency chain on one queue.
DEFAULT_GRES="${DEFAULT_GRES:-gpu:1}"

# Parse command-line overrides
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)     DATA_DIR="$2";     shift 2;;
        --output_dir)   OUTPUT_DIR="$2";   shift 2;;
        --partition)    PARTITION="$2";    shift 2;;
        --conda_env)    CONDA_ENV="$2";    shift 2;;
        --n_ensemble)   N_ENSEMBLE="$2";   shift 2;;
        --seed)         SEED="$2";         shift 2;;
        --dry_run)      DRY_RUN_FLAG="--dry_run"; STRICT_FLAG=""; shift;;
        *)              echo "Unknown flag: $1"; exit 1;;
    esac
done

COMMON="--data_dir $DATA_DIR --output_dir $OUTPUT_DIR --seed $SEED --n_ensemble $N_ENSEMBLE $STRICT_FLAG $DRY_RUN_FLAG"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC="$SCRIPT_DIR/hpc_run_stage.py"

mkdir -p "${OUTPUT_DIR}/slurm_logs"

# ---- Helper: submit a single stage ----
# All stages run on ${PARTITION} (default: gpu) with ${DEFAULT_GRES} (default
# gpu:1), per the "GPU always" instruction.  Pass GRES="" only if you have a
# specific reason to skip the GPU allocation for a single stage.
submit() {
    local STAGE="$1" WALL="$2" GRES="$3" MEM="$4" DEPS="$5"
    local GRES_LINE="" DEP_LINE=""
    # Use DEFAULT_GRES when the per-call GRES is the empty string (so existing
    # call sites don't need editing).
    if [[ -z "$GRES" ]]; then GRES="$DEFAULT_GRES"; fi
    GRES_LINE="#SBATCH --gres=$GRES"
    [[ -n "$DEPS" ]] && DEP_LINE="#SBATCH --dependency=$DEPS"

    sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=ipinn_${STAGE}
#SBATCH --output=${OUTPUT_DIR}/slurm_logs/${STAGE}_%j.out
#SBATCH --error=${OUTPUT_DIR}/slurm_logs/${STAGE}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --time=${WALL}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
${GRES_LINE}
${DEP_LINE}

# ---- Environment (SDSMT cluster) ----
# Determinism env vars: PYTHONHASHSEED must be set BEFORE Python starts;
# CUBLAS_WORKSPACE_CONFIG enables deterministic cuBLAS algorithms.
export PYTHONHASHSEED=${SEED}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Ensure SLURM job name shows up in nvidia-smi so we can correlate
# memory usage to a stage.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

module purge
module load ${CUDA_MODULE}
module load ${CUDNN_MODULE}
source activate ${CONDA_ENV}

echo "=== Stage: ${STAGE}  Node: \$(hostname)  GPU: \${CUDA_VISIBLE_DEVICES:-none} ==="
echo "=== Start: \$(date)   PYTHONHASHSEED=${SEED}   CUDA: \$(nvcc --version 2>/dev/null | tail -1) ==="

python ${HPC} --stage ${STAGE} ${COMMON}
EXIT_CODE=\$?

echo "=== End: \$(date)  Exit: \${EXIT_CODE} ==="
exit \${EXIT_CODE}
EOF
}

# ====================================================================
# Wall-time policy
# --------------------------------------------------------------------
# All stages get a generous 120-hour ceiling so a slow node, an HPO
# re-run, or a single retried ensemble member never overruns the wall.
# SLURM bills only what is used, not what is requested, so over-asking
# is free in dollars; the only cost is queue priority on partitions
# that prefer short jobs. If your cluster caps single-job wall time
# below 120h, edit ``MAX_WALL`` below.
# ====================================================================
MAX_WALL="120:00:00"
PREP_WALL="04:00:00"     # CPU prep is minutes; 4h is huge headroom
AGG_WALL="04:00:00"      # aggregation is minutes; same

# ====================================================================
# Stage 0: prep (CPU, fast)
# ====================================================================
JOB_PREP=$(submit "prep" "${PREP_WALL}" "" "16G" "")
echo "  prep            : $JOB_PREP"

# ====================================================================
# Stage 1: 7 parallel GPU training jobs
# ====================================================================
D="afterok:${JOB_PREP}"

JOB_RD=$(submit "train_random_ddns"  "${MAX_WALL}" "gpu:1" "32G" "$D")
JOB_RS=$(submit "train_random_soft"  "${MAX_WALL}" "gpu:1" "32G" "$D")
JOB_RH=$(submit "train_random_hard"  "${MAX_WALL}" "gpu:1" "32G" "$D")
JOB_UD=$(submit "train_unseen_ddns"  "${MAX_WALL}" "gpu:1" "32G" "$D")
JOB_US=$(submit "train_unseen_soft"  "${MAX_WALL}" "gpu:1" "32G" "$D")
JOB_UH=$(submit "train_unseen_hard"  "${MAX_WALL}" "gpu:1" "32G" "$D")
JOB_IV=$(submit "train_inverse"      "${MAX_WALL}" "gpu:1" "32G" "$D")

echo "  train (x7)      : $JOB_RD $JOB_RS $JOB_RH $JOB_UD $JOB_US $JOB_UH $JOB_IV"

# ====================================================================
# Stage 2: forward_analysis (needs all 6 training jobs complete)
# ====================================================================
D_FWD="afterok:${JOB_RD}:${JOB_RS}:${JOB_RH}:${JOB_UD}:${JOB_US}:${JOB_UH}"
JOB_FWD=$(submit "forward_analysis" "${MAX_WALL}" "gpu:1" "48G" "$D_FWD")
echo "  forward_analysis: $JOB_FWD"

# ====================================================================
# Stage 3: inverse_analysis (needs train_inverse + forward_analysis)
# ====================================================================
D_INV="afterok:${JOB_IV}:${JOB_FWD}"
JOB_INV=$(submit "inverse_analysis" "${MAX_WALL}" "gpu:1" "64G" "$D_INV")
echo "  inverse_analysis: $JOB_INV"

# ====================================================================
# Stage 4: aggregate (needs forward + inverse)
# ====================================================================
D_AGG="afterok:${JOB_FWD}:${JOB_INV}"
JOB_AGG=$(submit "aggregate" "${AGG_WALL}" "" "16G" "$D_AGG")
echo "  aggregate       : $JOB_AGG"

# ====================================================================
# Summary
# ====================================================================
ALL_JOBS="$JOB_PREP $JOB_RD $JOB_RS $JOB_RH $JOB_UD $JOB_US $JOB_UH $JOB_IV $JOB_FWD $JOB_INV $JOB_AGG"
echo ""
echo "=== Pipeline submitted: 11 jobs ==="
echo "  Chain: prep -> 7 parallel training -> forward + inverse -> aggregate"
echo "  Output: $OUTPUT_DIR"
echo "  Monitor:    squeue -u \$USER"
echo "  Cancel all: scancel $ALL_JOBS"
