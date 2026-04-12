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
# Submits 12 jobs with dependency chains:
#   prep  →  8 parallel training jobs  →  forward_analysis + inverse_analysis  →  aggregate
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
N_ENSEMBLE="${N_ENSEMBLE:-20}"
SEED="${SEED:-2026}"
DRY_RUN_FLAG=""
STRICT_FLAG="--strict_paper"

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
submit() {
    local STAGE="$1" WALL="$2" GRES="$3" MEM="$4" DEPS="$5"
    local GRES_LINE="" DEP_LINE=""
    [[ -n "$GRES" ]] && GRES_LINE="#SBATCH --gres=$GRES"
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

# ---- Environment ----
module purge
module load cuda          # adjust to your cluster's module name
source activate ${CONDA_ENV}

echo "=== Stage: ${STAGE}  Node: \$(hostname)  GPU: \${CUDA_VISIBLE_DEVICES:-none} ==="
echo "=== Start: \$(date) ==="

python ${HPC} --stage ${STAGE} ${COMMON}
EXIT_CODE=\$?

echo "=== End: \$(date)  Exit: \${EXIT_CODE} ==="
exit \${EXIT_CODE}
EOF
}

# ====================================================================
# Stage 0: prep (CPU, fast)
# ====================================================================
JOB_PREP=$(submit "prep" "00:30:00" "" "16G" "")
echo "  prep            : $JOB_PREP"

# ====================================================================
# Stage 1: 8 parallel GPU training jobs
# ====================================================================
D="afterok:${JOB_PREP}"

JOB_RD=$(submit "train_random_ddns"  "48:00:00" "gpu:1" "32G" "$D")
JOB_RS=$(submit "train_random_soft"  "48:00:00" "gpu:1" "32G" "$D")
JOB_RH=$(submit "train_random_hard"  "48:00:00" "gpu:1" "32G" "$D")
JOB_UD=$(submit "train_unseen_ddns"  "48:00:00" "gpu:1" "32G" "$D")
JOB_US=$(submit "train_unseen_soft"  "48:00:00" "gpu:1" "32G" "$D")
JOB_UH=$(submit "train_unseen_hard"  "48:00:00" "gpu:1" "32G" "$D")
JOB_IV=$(submit "train_inverse"      "48:00:00" "gpu:1" "32G" "$D")
JOB_LC=$(submit "loao_cv"            "24:00:00" "gpu:1" "32G" "$D")

echo "  train (x8)      : $JOB_RD $JOB_RS $JOB_RH $JOB_UD $JOB_US $JOB_UH $JOB_IV $JOB_LC"

# ====================================================================
# Stage 2: forward_analysis (needs all 6 training + LOAO complete)
# ====================================================================
D_FWD="afterok:${JOB_RD}:${JOB_RS}:${JOB_RH}:${JOB_UD}:${JOB_US}:${JOB_UH}:${JOB_LC}"
JOB_FWD=$(submit "forward_analysis" "24:00:00" "gpu:1" "48G" "$D_FWD")
echo "  forward_analysis: $JOB_FWD"

# ====================================================================
# Stage 3: inverse_analysis (needs train_inverse + forward_analysis)
# ====================================================================
D_INV="afterok:${JOB_IV}:${JOB_FWD}"
JOB_INV=$(submit "inverse_analysis" "48:00:00" "gpu:1" "64G" "$D_INV")
echo "  inverse_analysis: $JOB_INV"

# ====================================================================
# Stage 4: aggregate (needs forward + inverse)
# ====================================================================
D_AGG="afterok:${JOB_FWD}:${JOB_INV}"
JOB_AGG=$(submit "aggregate" "01:00:00" "" "16G" "$D_AGG")
echo "  aggregate       : $JOB_AGG"

# ====================================================================
# Summary
# ====================================================================
ALL_JOBS="$JOB_PREP $JOB_RD $JOB_RS $JOB_RH $JOB_UD $JOB_US $JOB_UH $JOB_IV $JOB_LC $JOB_FWD $JOB_INV $JOB_AGG"
echo ""
echo "=== Pipeline submitted: 12 jobs ==="
echo "  Chain: prep -> 8 parallel training -> forward + inverse -> aggregate"
echo "  Output: $OUTPUT_DIR"
echo "  Monitor:    squeue -u \$USER"
echo "  Cancel all: scancel $ALL_JOBS"
