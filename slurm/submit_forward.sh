#!/bin/bash
# ==========================================================================
# submit_forward.sh — Forward Hard retrain, parallel across GPUs
# ==========================================================================
#
# Submits a SLURM array of N tasks (one ensemble member per task, 1 GPU
# each) and a dependent merge job that gathers the per-member partial
# bundles into the final forward_hard_bundle.pt + forward_hard_results.json.
#
# Pipeline:
# 1. Array job (--array=0-(M-1)): hpo/forward_member.py, one member per task.
# 2. Merge job (--dependency=afterok:<array_job>): hpo/forward_merge.py
# reads parts_hard/member_*.pt, applies Tukey-fence convergence filter,
# computes ensemble metrics, writes the final bundle.
#
# Default M=20. Wall time per task is 24 h (the new slope-subtraction BC
# is 2-3x slower than value-only BC; previous per-member time was
# ~3.5 h, so new per-member is ~7-11 h; 24h is a safe ceiling).
#
# Usage (from the repo root):
# bash slurm/submit_forward.sh
#
# Environment overrides:
# M_ENSEMBLE=20 total ensemble size
# N_CONCURRENT=10 max array tasks running concurrently
# (clamps SLURM_ARRAY's %N qualifier)
# APPROACH=hard ddns | soft | hard
# DATA_DIR=./data
# OUTPUT_DIR=./results_forward
# PARTITION=gpu
# CONDA_ENV=ipinn
# CUDA_MODULE=cuda/12.3
# CUDNN_MODULE=cudnn/9.0.0-cuda12
# SEED=2026
# MEMBER_WALL=24:00:00 per-array-task wall time
# MERGE_WALL=01:00:00 merge job wall time (CPU-only, fast)
# MEM=32G
#
# Example: 10 GPUs in flight, the rest queued, finishing in ~2 waves
# N_CONCURRENT=10 bash slurm/submit_forward.sh
# ==========================================================================

set -euo pipefail

# ---- Configurable variables ----------------------------------------------
M_ENSEMBLE="${M_ENSEMBLE:-20}"
N_CONCURRENT="${N_CONCURRENT:-10}"
APPROACH="${APPROACH:-hard}"
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_forward}"
PARTITION="${PARTITION:-gpu}"
CONDA_ENV="${CONDA_ENV:-ipinn}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.3}"
CUDNN_MODULE="${CUDNN_MODULE:-cudnn/9.0.0-cuda12}"
SEED="${SEED:-2026}"
MEMBER_WALL="${MEMBER_WALL:-24:00:00}"
MERGE_WALL="${MERGE_WALL:-01:00:00}"
MEM="${MEM:-32G}"
# Optional held-out angle for leave-one-angle-out (LOAO) production retrains.
# When set, members are written to ``parts_<approach>_t<theta>/`` and the
# merge writes ``forward_<approach>_t<theta>_{bundle.pt,results.json,log.txt}``
# so multiple folds can coexist in the same OUTPUT_DIR without overwriting.
# Leave empty for the single-angle (theta_star=60) default protocol.
THETA_STAR="${THETA_STAR:-}"

# ---- Sanity ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MEMBER_LAUNCHER="$REPO_ROOT/hpo/forward_member.py"
MERGE_LAUNCHER="$REPO_ROOT/hpo/forward_merge.py"
for f in "$MEMBER_LAUNCHER" "$MERGE_LAUNCHER"; do
 if [[ ! -f "$f" ]]; then echo "ERROR: $f not found"; exit 1; fi
done
case "$APPROACH" in
 ddns|soft|hard) ;;
 *) echo "ERROR: APPROACH must be ddns|soft|hard, got '$APPROACH'"; exit 1 ;;
esac
LAST_IDX=$((M_ENSEMBLE - 1))

# Build the launcher flag string + the per-fold partials directory name.
# When THETA_STAR is set, jobs / parts / outputs are tagged with the held-
# out angle so multiple LOAO folds can run concurrently in the same
# OUTPUT_DIR without overwriting each other.
if [[ -n "${THETA_STAR}" ]]; then
    THETA_INT=$(printf '%d' "${THETA_STAR%.*}")
    THETA_FLAG="--theta_star ${THETA_STAR}"
    APPROACH_TAG="${APPROACH}_t${THETA_INT}"
    PARTS_SUBDIR="parts_${APPROACH}_t${THETA_INT}"
else
    THETA_FLAG=""
    APPROACH_TAG="${APPROACH}"
    PARTS_SUBDIR="parts_${APPROACH}"
fi

mkdir -p "${OUTPUT_DIR}/slurm_logs"
mkdir -p "${OUTPUT_DIR}/${PARTS_SUBDIR}"

# ---- Submit array job (per-member training) -------------------------------
ARRAY_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=forward_${APPROACH_TAG}_arr
#SBATCH --output=${OUTPUT_DIR}/slurm_logs/forward_${APPROACH_TAG}_m%a_%A.out
#SBATCH --error=${OUTPUT_DIR}/slurm_logs/forward_${APPROACH_TAG}_m%a_%A.err
#SBATCH --partition=${PARTITION}
#SBATCH --time=${MEMBER_WALL}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-${LAST_IDX}%${N_CONCURRENT}

export PYTHONHASHSEED=${SEED}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

module purge
module load ${CUDA_MODULE}
module load ${CUDNN_MODULE}

# Bypass conda activate (silently fails on some SDSMT compute nodes —
# see commit ff09ee5). Pin the env's python by absolute path.
PYTHON_BIN="\$HOME/miniconda3/envs/${CONDA_ENV}/bin/python"
if [[ ! -x "\$PYTHON_BIN" ]]; then
 echo "ERROR: '\$PYTHON_BIN' not executable on \$(hostname)"
 exit 1
fi

MEMBER_IDX=\${SLURM_ARRAY_TASK_ID}

echo "=== env check on \$(hostname) ==="
echo " PYTHON_BIN: \$PYTHON_BIN"
"\$PYTHON_BIN" -c "import sys, torch; print(f' python {sys.version.split()[0]} torch {torch.__version__} cuda={torch.cuda.is_available()}')"

echo "=== FORWARD ${APPROACH^^} member \${MEMBER_IDX} Node: \$(hostname) GPU: \${CUDA_VISIBLE_DEVICES:-none} ==="
echo "=== Start: \$(date) ==="

# Single-line invocation: the SDSMT cluster's sbatch eats backslash-newline
# line-continuations inside unquoted heredocs (see commit d7da39a), making
# the python call exit silently with "unrecognized arguments".
"\$PYTHON_BIN" ${MEMBER_LAUNCHER} --approach ${APPROACH} --member_idx \${MEMBER_IDX} --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --n_ensemble ${M_ENSEMBLE} --seed ${SEED} ${THETA_FLAG}
EXIT_CODE=\$?

echo "=== End: \$(date) Exit: \${EXIT_CODE} ==="
exit \${EXIT_CODE}
EOF
)
echo "Submitted array job ${ARRAY_JOB_ID} (${M_ENSEMBLE} tasks, up to ${N_CONCURRENT} concurrent)"
echo " Per-member logs: ${OUTPUT_DIR}/slurm_logs/forward_${APPROACH_TAG}_m<idx>_${ARRAY_JOB_ID}.{out,err}"
echo " Partial bundles: ${OUTPUT_DIR}/${PARTS_SUBDIR}/member_*.pt"

# ---- Submit merge job (depends on the whole array succeeding) ------------
# Use ``afterany`` not ``afterok`` so a single failed member doesn't block
# the merge — forward_merge.py tolerates missing/failed parts and
# reports them in the log.
MERGE_JOB_ID=$(sbatch --parsable --dependency=afterany:${ARRAY_JOB_ID} <<EOF
#!/bin/bash
#SBATCH --job-name=forward_${APPROACH_TAG}_merge
#SBATCH --output=${OUTPUT_DIR}/slurm_logs/forward_${APPROACH_TAG}_merge_%j.out
#SBATCH --error=${OUTPUT_DIR}/slurm_logs/forward_${APPROACH_TAG}_merge_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --time=${MERGE_WALL}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

export PYTHONHASHSEED=${SEED}
export CUBLAS_WORKSPACE_CONFIG=:4096:8

module purge
module load ${CUDA_MODULE}
module load ${CUDNN_MODULE}

PYTHON_BIN="\$HOME/miniconda3/envs/${CONDA_ENV}/bin/python"

echo "=== FORWARD ${APPROACH^^} merge Node: \$(hostname) ==="
echo "=== Start: \$(date) ==="

# Single-line invocation (see d7da39a / multi-line heredoc continuation bug).
"\$PYTHON_BIN" ${MERGE_LAUNCHER} --approach ${APPROACH} --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --n_ensemble ${M_ENSEMBLE} --seed ${SEED} ${THETA_FLAG}
EXIT_CODE=\$?

echo "=== End: \$(date) Exit: \${EXIT_CODE} ==="
exit \${EXIT_CODE}
EOF
)
echo "Submitted merge job ${MERGE_JOB_ID} (depends on ${ARRAY_JOB_ID})"
echo " Merge log: ${OUTPUT_DIR}/slurm_logs/forward_${APPROACH_TAG}_merge_${MERGE_JOB_ID}.{out,err}"
echo " Final bundle: ${OUTPUT_DIR}/forward_${APPROACH_TAG}_bundle.pt"
echo " Final results: ${OUTPUT_DIR}/forward_${APPROACH_TAG}_results.json"
echo ""
echo "=== Forward ${APPROACH_TAG^^} parallel retrain submitted ==="
echo " Monitor: squeue -u \$USER"
echo " Per-member done when: ls ${OUTPUT_DIR}/${PARTS_SUBDIR}/*.pt | wc -l"
