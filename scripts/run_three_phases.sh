#!/usr/bin/env bash
# Run all three training phases sequentially:
#   1. Cross-model channel (E_analyst_math)
#   2. Re-SFT with the integrated 4-channel predator
#   3. Reward-RL on top
set -euo pipefail

cd "$(dirname "$0")/.."
LOG_DIR="${TROPHIC_LOG_DIR:-/tmp/claude-1000/-home-dgonier-ecology-experiment/62ff8417-b521-43e5-97aa-8ea8724dd4a9/tasks}"
mkdir -p "$LOG_DIR"

source .venv/bin/activate

if [ "${SKIP_PHASE1:-0}" != "1" ]; then
  echo "===== PHASE 1: cross-model channel training ====="
  LOG1="$LOG_DIR/phase1_cross_model.log"
  echo "started: $(date -Iseconds)" > "$LOG1"
  PYTHONUNBUFFERED=1 \
    TROPHIC_CMRL_STEPS="${PHASE1_STEPS:-200}" \
    TROPHIC_CMRL_LR=5e-5 \
    TROPHIC_GROUP_SIZE=4 \
    TROPHIC_TEMP=0.7 \
    TROPHIC_SEED=1 \
    python -u scripts/train_cross_model.py >> "$LOG1" 2>&1
  echo "phase 1 done: $(date -Iseconds)" >> "$LOG1"
else
  echo "===== PHASE 1: SKIPPED (SKIP_PHASE1=1) ====="
fi

echo "===== PHASE 2: re-SFT with 4-channel predator ====="
LOG2="$LOG_DIR/phase2_sft_v4.log"
echo "started: $(date -Iseconds)" > "$LOG2"
# Remove stale 3-channel checkpoint so the 4-channel one can take its place.
rm -f checkpoints/sft_seed1_best.pt checkpoints/sft_seed1_final.pt
echo "[orch] removed stale sft_seed1 checkpoints (different channel topology)" >> "$LOG2"
PYTHONUNBUFFERED=1 \
  TROPHIC_STEPS="${PHASE2_STEPS:-800}" \
  TROPHIC_LR=5e-4 \
  TROPHIC_EVAL_EVERY=100 \
  TROPHIC_LOG_EVERY=50 \
  TROPHIC_SEED=1 \
  python -u scripts/train_sft.py >> "$LOG2" 2>&1
echo "phase 2 done: $(date -Iseconds)" >> "$LOG2"

echo "===== PHASE 3: reward-RL on top of SFT v4 ====="
LOG3="$LOG_DIR/phase3_reward_rl.log"
echo "started: $(date -Iseconds)" > "$LOG3"
PYTHONUNBUFFERED=1 \
  TROPHIC_RL_STEPS="${PHASE3_STEPS:-300}" \
  TROPHIC_RL_LR=5e-5 \
  TROPHIC_LOG_EVERY=10 \
  TROPHIC_EVAL_EVERY=50 \
  TROPHIC_GROUP_SIZE=4 \
  TROPHIC_TEMP=0.9 \
  TROPHIC_KL_BETA=0.05 \
  TROPHIC_SEED=1 \
  TROPHIC_CKPT="$(pwd)/checkpoints/sft_seed1_best.pt" \
  python -u scripts/train_reward_rl.py >> "$LOG3" 2>&1
echo "phase 3 done: $(date -Iseconds)" >> "$LOG3"

echo "all three phases complete"
