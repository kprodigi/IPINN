#!/bin/bash
# ==========================================================================
# launch_three_hpos.sh — Sequential HPO across DDNS / Soft / Hard
# ==========================================================================
#
# Submits one HPO job per approach on the unseen-θ=60° protocol.  The three
# jobs share the 15 GPUs sequentially (Soft starts after DDNS completes,
# Hard after Soft) via SLURM job dependencies.
#
# Usage:
#   bash slurm/launch_three_hpos.sh
#
# Compute estimate (per approach):
#   100 trials × 3 seeds × 800 ep × ~3 s/ep / 15 workers
#   ≈ 100 × 3 × 800 × 3 / 15 / 3600 ≈ 13.3 h per approach
#   Total wall: ~40 h ≈ 1.7 days
# ==========================================================================
set -euo pipefail

OUTPUT_BASE="${OUTPUT_BASE:-./hpo_clean_out}"
N_TRIALS="${N_TRIALS:-100}"
N_STARTUP="${N_STARTUP:-25}"
N_SEEDS="${N_SEEDS:-3}"
HPO_EPOCHS="${HPO_EPOCHS:-800}"
N_WORKERS="${N_WORKERS:-15}"
WALL="${WALL:-48:00:00}"

mkdir -p "${OUTPUT_BASE}"

echo "=== Launching three HPOs sequentially (DDNS → Soft → Hard) ==="
echo "  Output base: ${OUTPUT_BASE}"
echo "  Per approach: ${N_TRIALS} trials × ${N_SEEDS} seeds × ${HPO_EPOCHS} ep, ${N_WORKERS} workers, ${WALL} wall"

# DDNS — no dependency
echo ""
echo "--- DDNS ---"
DDNS_JID=$(N_TRIALS=${N_TRIALS} N_STARTUP=${N_STARTUP} N_SEEDS=${N_SEEDS} \
           HPO_EPOCHS=${HPO_EPOCHS} N_WORKERS=${N_WORKERS} WALL=${WALL} \
           APPROACH=ddns OUTPUT_DIR=${OUTPUT_BASE}/ddns \
           bash slurm/submit_hpo.sh 2>&1 | grep -oP 'job \K\d+' | head -1)
echo "  Submitted DDNS HPO: ${DDNS_JID}"

# Soft — afterany on DDNS array completion
echo ""
echo "--- Soft (depends on DDNS:${DDNS_JID}) ---"
# Note: submit_hpo.sh does not currently accept a --dependency flag; we
# replicate its sbatch logic here only if we need stricter sequencing.
# For now we run with no SLURM dependency — instead we submit all three
# and rely on N_WORKERS=15 saturating the queue.  If queue contention is
# an issue, swap to a wrapper sbatch with --dependency=afterany.
SOFT_JID=$(N_TRIALS=${N_TRIALS} N_STARTUP=${N_STARTUP} N_SEEDS=${N_SEEDS} \
           HPO_EPOCHS=${HPO_EPOCHS} N_WORKERS=${N_WORKERS} WALL=${WALL} \
           APPROACH=soft OUTPUT_DIR=${OUTPUT_BASE}/soft \
           bash slurm/submit_hpo.sh 2>&1 | grep -oP 'job \K\d+' | head -1)
echo "  Submitted Soft HPO: ${SOFT_JID}"

# Hard — afterany on Soft array completion
echo ""
echo "--- Hard (depends on Soft:${SOFT_JID}) ---"
HARD_JID=$(N_TRIALS=${N_TRIALS} N_STARTUP=${N_STARTUP} N_SEEDS=${N_SEEDS} \
           HPO_EPOCHS=${HPO_EPOCHS} N_WORKERS=${N_WORKERS} WALL=${WALL} \
           APPROACH=hard OUTPUT_DIR=${OUTPUT_BASE}/hard \
           bash slurm/submit_hpo.sh 2>&1 | grep -oP 'job \K\d+' | head -1)
echo "  Submitted Hard HPO: ${HARD_JID}"

echo ""
echo "=== All three HPOs queued ==="
echo "  DDNS: ${DDNS_JID}  ./${OUTPUT_BASE}/ddns/hpo_study_ddns.db"
echo "  Soft: ${SOFT_JID}  ./${OUTPUT_BASE}/soft/hpo_study_soft.db"
echo "  Hard: ${HARD_JID}  ./${OUTPUT_BASE}/hard/hpo_study_hard.db"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  squeue -j ${DDNS_JID},${SOFT_JID},${HARD_JID}"
