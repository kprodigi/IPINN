#!/bin/bash
# ==========================================================================
# submit_hpo.sh — Hyperparameter optimisation launcher for one approach
# ==========================================================================
#
# Submits a SLURM job that runs hpo/hpo_search.py for one forward-design
# approach (ddns | soft | hard) on the unseen-angle protocol.  Multiple
# workers can be launched against the same --output_dir (and the same
# --study_name, derived from the approach) to share a single SQLite-backed
# Optuna study; Optuna serialises trial scheduling through SQLite locks
# so each trial is picked up by exactly one worker.
#
# Usage (from the repo root):
#   APPROACH=hard bash slurm/submit_hpo.sh
#   APPROACH=soft N_TRIALS=120 N_WORKERS=4 bash slurm/submit_hpo.sh
#   APPROACH=ddns N_TRIALS=80  bash slurm/submit_hpo.sh
#
# Environment overrides (all optional):
#   APPROACH=hard                    ddns | soft | hard   (required)
#   N_TRIALS=100                     Total Optuna trials across all workers
#   N_STARTUP=15                     Random-search trials before TPE engages
#   N_SEEDS=2                        Ensemble members trained per trial
#   HPO_EPOCHS=200                   Per-trial training epoch budget
#   N_WORKERS=1                      Parallel worker count (one SLURM task each)
#   SEED=2026                        Base seed
#   DATA_DIR=./data
#   OUTPUT_DIR=./hpo_out
#   PARTITION=gpu
#   CONDA_ENV=ipinn
#   CUDA_MODULE=cuda/12.3
#   CUDNN_MODULE=cudnn/9.0.0-cuda12
#   WALL=72:00:00
#   MEM=32G
#   NO_WARM_STARTS=0                 Set to 1 to pass --no_warm_starts
#
# Outputs (per OUTPUT_DIR)
# ------------------------
#   hpo_study_<approach>.db          SQLite study (resumable)
#   hpo_best_params_<approach>.json  Best params, copy-paste into cfg
#   hpo_history_<approach>.csv       Per-trial history (params + R^2)
#   hpo_log_<approach>.txt           Full Optuna log
#   slurm_logs/<approach>_*.{out,err}
# ==========================================================================

set -euo pipefail

# ---- Required ------------------------------------------------------------
APPROACH="${APPROACH:?Set APPROACH=ddns|soft|hard}"
case "$APPROACH" in
    ddns|soft|hard) ;;
    *) echo "ERROR: APPROACH must be ddns|soft|hard, got '$APPROACH'"; exit 1 ;;
esac

# ---- Defaults ------------------------------------------------------------
N_TRIALS="${N_TRIALS:-100}"
N_STARTUP="${N_STARTUP:-15}"
N_SEEDS="${N_SEEDS:-2}"
HPO_EPOCHS="${HPO_EPOCHS:-200}"
N_WORKERS="${N_WORKERS:-1}"
SEED="${SEED:-2026}"
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./hpo_out}"
PARTITION="${PARTITION:-gpu}"
CONDA_ENV="${CONDA_ENV:-ipinn}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.3}"
CUDNN_MODULE="${CUDNN_MODULE:-cudnn/9.0.0-cuda12}"
WALL="${WALL:-72:00:00}"
MEM="${MEM:-32G}"
NO_WARM_STARTS="${NO_WARM_STARTS:-0}"

WARM_FLAG=""
if [[ "$NO_WARM_STARTS" == "1" ]]; then
    WARM_FLAG="--no_warm_starts"
fi

# ---- Paths ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER="$REPO_ROOT/hpo/hpo_search.py"
if [[ ! -f "$LAUNCHER" ]]; then
    echo "ERROR: $LAUNCHER not found"; exit 1
fi

mkdir -p "${OUTPUT_DIR}/slurm_logs"

# ---- Submit one job per worker; all workers share the same study --------
LAST_WORKER=$((N_WORKERS - 1))
ARRAY_FLAG=""
if [[ "$N_WORKERS" -gt 1 ]]; then
    ARRAY_FLAG="#SBATCH --array=0-${LAST_WORKER}"
fi

JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=hpo_${APPROACH}
#SBATCH --output=${OUTPUT_DIR}/slurm_logs/hpo_${APPROACH}_%A_%a.out
#SBATCH --error=${OUTPUT_DIR}/slurm_logs/hpo_${APPROACH}_%A_%a.err
#SBATCH --partition=${PARTITION}
#SBATCH --time=${WALL}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
${ARRAY_FLAG}

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

echo "=== env check on \$(hostname) ==="
echo "  PYTHON_BIN: \$PYTHON_BIN"
"\$PYTHON_BIN" -c "import sys, torch; print(f'  python {sys.version.split()[0]}  torch {torch.__version__}  cuda={torch.cuda.is_available()}')"

WORKER_ID=\${SLURM_ARRAY_TASK_ID:-0}
echo "=== HPO ${APPROACH^^} worker \${WORKER_ID}  Node: \$(hostname)  GPU: \${CUDA_VISIBLE_DEVICES:-none} ==="
echo "=== Start: \$(date) ==="

# Single-line invocation (heredoc + backslash-newline line continuations are
# unreliable on this cluster's sbatch — see previous diagnostics).
"\$PYTHON_BIN" ${LAUNCHER} --approach ${APPROACH} --n_trials ${N_TRIALS} --n_startup_trials ${N_STARTUP} --n_seeds ${N_SEEDS} --hpo_epochs ${HPO_EPOCHS} --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --seed ${SEED} ${WARM_FLAG}
EXIT_CODE=\$?

echo "=== End: \$(date)  Exit: \${EXIT_CODE} ==="
exit \${EXIT_CODE}
EOF
)
echo "Submitted HPO job ${JOB_ID} (approach=${APPROACH}, n_workers=${N_WORKERS})"
echo "  Logs:    ${OUTPUT_DIR}/slurm_logs/hpo_${APPROACH}_${JOB_ID}_*.{out,err}"
echo "  Study:   ${OUTPUT_DIR}/hpo_study_${APPROACH}.db"
echo "  Best:    ${OUTPUT_DIR}/hpo_best_params_${APPROACH}.json   (written when search completes)"
echo "  History: ${OUTPUT_DIR}/hpo_history_${APPROACH}.csv        (rewritten each completion)"
echo ""
echo "Monitor:  squeue -u \$USER  ;  tail -f ${OUTPUT_DIR}/hpo_log_${APPROACH}.txt"
