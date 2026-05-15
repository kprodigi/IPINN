#!/bin/bash
# ==========================================================================
# submit_inverse.sh — Parallel inverse-design surrogate training
# ==========================================================================
#
# Trains the M=20 full-data Hard-PINN inverse-design surrogate in parallel
# (one SLURM task per ensemble member, 1 GPU each), then merges the
# partial bundles, then runs the downstream inverse-design pipeline
# (classifier + GP-BO + Pareto + Jacobian + ablations) using the
# pretrained surrogate.
#
# Pipeline (three SLURM jobs, dependency-chained):
#   1. Array job        hpo/inverse_member.py (one member per array task)
#   2. Merge job        hpo/inverse_merge.py (gather + Tukey filter)
#                       writes inverse_pretrained_hard.pt
#   3. Analysis job     composite_design.py --mode inverse \
#                       --use_pretrained_inverse <bundle>
#                       skips training, runs classifier + GP-BO + ablations
#
# Default M=20.  Member training wall is 24 h (sequential would be ~7-11 h
# per member with the new slope-subtraction BC; 24 h is a safe ceiling).
# Analysis wall is 24 h (classifier is fast; GP-BO over 5 targets + Pareto
# + Jacobian + λ-sensitivity adds up).
#
# Usage (from the repo root):
#   bash slurm/submit_inverse.sh
#
# Environment overrides:
#   M_ENSEMBLE=20                  total ensemble size
#   N_CONCURRENT=10                max array tasks running concurrently
#   DATA_DIR=./data
#   OUTPUT_DIR=./results_paper     (NOTE: different default from stage 2)
#   PARTITION=gpu
#   CONDA_ENV=ipinn
#   CUDA_MODULE=cuda/12.3
#   CUDNN_MODULE=cudnn/9.0.0-cuda12
#   SEED=2026
#   MEMBER_WALL=24:00:00
#   MERGE_WALL=01:00:00
#   ANALYSIS_WALL=24:00:00
#   MEM=32G
#   SKIP_ANALYSIS=0                if 1, only submit array + merge (no analysis)
# ==========================================================================

set -euo pipefail

# ---- Configurable variables ----------------------------------------------
M_ENSEMBLE="${M_ENSEMBLE:-20}"
N_CONCURRENT="${N_CONCURRENT:-10}"
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_paper}"
PARTITION="${PARTITION:-gpu}"
CONDA_ENV="${CONDA_ENV:-ipinn}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.3}"
CUDNN_MODULE="${CUDNN_MODULE:-cudnn/9.0.0-cuda12}"
SEED="${SEED:-2026}"
MEMBER_WALL="${MEMBER_WALL:-24:00:00}"
MERGE_WALL="${MERGE_WALL:-01:00:00}"
ANALYSIS_WALL="${ANALYSIS_WALL:-24:00:00}"
MEM="${MEM:-32G}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-0}"

# ---- Sanity ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MEMBER_LAUNCHER="$REPO_ROOT/hpo/inverse_member.py"
MERGE_LAUNCHER="$REPO_ROOT/hpo/inverse_merge.py"
ANALYSIS_LAUNCHER="$REPO_ROOT/composite_design.py"
for f in "$MEMBER_LAUNCHER" "$MERGE_LAUNCHER" "$ANALYSIS_LAUNCHER"; do
    if [[ ! -f "$f" ]]; then echo "ERROR: $f not found"; exit 1; fi
done
LAST_IDX=$((M_ENSEMBLE - 1))

mkdir -p "${OUTPUT_DIR}/slurm_logs"
mkdir -p "${OUTPUT_DIR}/parts_inverse_hard"

PRETRAINED_BUNDLE="${OUTPUT_DIR}/inverse_pretrained_hard.pt"

# ---- Submit array job (per-member training) -------------------------------
ARRAY_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=inverse_hard_arr
#SBATCH --output=${OUTPUT_DIR}/slurm_logs/inverse_hard_m%a_%A.out
#SBATCH --error=${OUTPUT_DIR}/slurm_logs/inverse_hard_m%a_%A.err
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

PYTHON_BIN="\$HOME/miniconda3/envs/${CONDA_ENV}/bin/python"
if [[ ! -x "\$PYTHON_BIN" ]]; then
    echo "ERROR: '\$PYTHON_BIN' not executable on \$(hostname)"
    exit 1
fi

MEMBER_IDX=\${SLURM_ARRAY_TASK_ID}

echo "=== env check on \$(hostname) ==="
echo "  PYTHON_BIN: \$PYTHON_BIN"
"\$PYTHON_BIN" -c "import sys, torch; print(f'  python {sys.version.split()[0]}  torch {torch.__version__}  cuda={torch.cuda.is_available()}')"

echo "=== INVERSE HARD member \${MEMBER_IDX}  Node: \$(hostname)  GPU: \${CUDA_VISIBLE_DEVICES:-none} ==="
echo "=== Start: \$(date) ==="

# Single-line invocation: the SDSMT cluster's sbatch eats backslash-newline
# continuations inside unquoted heredocs (see commit d7da39a).
"\$PYTHON_BIN" ${MEMBER_LAUNCHER} --member_idx \${MEMBER_IDX} --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --n_ensemble ${M_ENSEMBLE} --seed ${SEED}
EXIT_CODE=\$?

echo "=== End: \$(date)  Exit: \${EXIT_CODE} ==="
exit \${EXIT_CODE}
EOF
)
echo "Submitted inverse training array job ${ARRAY_JOB_ID}"
echo "  (${M_ENSEMBLE} tasks, up to ${N_CONCURRENT} concurrent)"

# ---- Submit merge job (depends on the array) -----------------------------
MERGE_JOB_ID=$(sbatch --parsable --dependency=afterany:${ARRAY_JOB_ID} <<EOF
#!/bin/bash
#SBATCH --job-name=inverse_hard_merge
#SBATCH --output=${OUTPUT_DIR}/slurm_logs/inverse_hard_merge_%j.out
#SBATCH --error=${OUTPUT_DIR}/slurm_logs/inverse_hard_merge_%j.err
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

echo "=== INVERSE HARD merge  Node: \$(hostname) ==="
echo "=== Start: \$(date) ==="

# Single-line invocation (see d7da39a).
"\$PYTHON_BIN" ${MERGE_LAUNCHER} --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --n_ensemble ${M_ENSEMBLE} --seed ${SEED}
EXIT_CODE=\$?

echo "=== End: \$(date)  Exit: \${EXIT_CODE} ==="
exit \${EXIT_CODE}
EOF
)
echo "Submitted inverse merge job ${MERGE_JOB_ID} (depends on ${ARRAY_JOB_ID})"

if [[ "${SKIP_ANALYSIS}" == "1" ]]; then
    echo ""
    echo "SKIP_ANALYSIS=1: not submitting the downstream analysis job."
    echo "Run it manually after the merge completes:"
    echo "  python composite_design.py --mode inverse \\"
    echo "      --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} \\"
    echo "      --use_pretrained_inverse ${PRETRAINED_BUNDLE}"
    exit 0
fi

# ---- Submit analysis job (depends on merge, uses pretrained bundle) ------
ANALYSIS_JOB_ID=$(sbatch --parsable --dependency=afterok:${MERGE_JOB_ID} <<EOF
#!/bin/bash
#SBATCH --job-name=inverse_hard_analysis
#SBATCH --output=${OUTPUT_DIR}/slurm_logs/inverse_hard_analysis_%j.out
#SBATCH --error=${OUTPUT_DIR}/slurm_logs/inverse_hard_analysis_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --time=${ANALYSIS_WALL}
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

PYTHON_BIN="\$HOME/miniconda3/envs/${CONDA_ENV}/bin/python"

echo "=== INVERSE HARD analysis (GP-BO + classifier + Pareto + ...)  Node: \$(hostname) ==="
echo "=== Start: \$(date) ==="

# Single-line invocation (see d7da39a).
"\$PYTHON_BIN" ${ANALYSIS_LAUNCHER} --mode inverse --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --n_ensemble ${M_ENSEMBLE} --seed ${SEED} --use_pretrained_inverse ${PRETRAINED_BUNDLE}
EXIT_CODE=\$?

echo "=== End: \$(date)  Exit: \${EXIT_CODE} ==="
exit \${EXIT_CODE}
EOF
)
echo "Submitted inverse analysis job ${ANALYSIS_JOB_ID} (depends on ${MERGE_JOB_ID})"
echo ""
echo "=== Inverse-design parallel pipeline submitted ==="
echo "  Output:    ${OUTPUT_DIR}"
echo "  Monitor:   squeue -u \$USER"
echo "  Partials:  ${OUTPUT_DIR}/parts_inverse_hard/member_*.pt"
echo "  Pretrained:${PRETRAINED_BUNDLE}                  (after merge)"
echo "  Final:     ${OUTPUT_DIR}/inverse_models.pt + analysis_results.pt"
echo "             (after analysis; then run --mode replot for figures)"
