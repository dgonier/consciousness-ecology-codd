#!/usr/bin/env bash
# Run SFT three times with seeds 0/1/2, all output to a single combined log.
# Each seed is a fresh Python process so weights/optim/scheduler all reset.
#
# Usage:
#   bash scripts/run_three_seeds.sh /path/to/output.log
#
# Honors TROPHIC_STEPS / TROPHIC_LR / TROPHIC_EVAL_EVERY env vars (with
# sensible defaults for a comparison run).
set -euo pipefail

LOG="${1:?usage: run_three_seeds.sh <log-path>}"
STEPS="${TROPHIC_STEPS:-1000}"
LR="${TROPHIC_LR:-5e-4}"
EVAL_EVERY="${TROPHIC_EVAL_EVERY:-100}"
LOG_EVERY="${TROPHIC_LOG_EVERY:-25}"

cd "$(dirname "$0")/.."
source .venv/bin/activate

for SEED in 0 1 2; do
    {
        echo ""
        echo "############################################################"
        echo "## SEED ${SEED}  steps=${STEPS}  lr=${LR}  $(date -Iseconds)"
        echo "############################################################"
    } >> "$LOG"

    PYTHONUNBUFFERED=1 \
    TROPHIC_STEPS="$STEPS" \
    TROPHIC_LR="$LR" \
    TROPHIC_EVAL_EVERY="$EVAL_EVERY" \
    TROPHIC_LOG_EVERY="$LOG_EVERY" \
    TROPHIC_SEED="$SEED" \
    python -u scripts/train_sft.py >> "$LOG" 2>&1

    {
        echo ""
        echo "## SEED ${SEED} END $(date -Iseconds)"
    } >> "$LOG"
done

echo "all three seeds done -> $LOG"
