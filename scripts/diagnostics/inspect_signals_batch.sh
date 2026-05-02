#!/usr/bin/env bash
# Capture signal JSONL for a list of scenarios. Output goes to logs/signals/.
# Usage:
#   scripts/diagnostics/inspect_signals_batch.sh <ckpt> <scenario1> [<scenario2> ...]
# Example:
#   scripts/diagnostics/inspect_signals_batch.sh checkpoints/sft_seed32_stable_best.pt \
#     stocknet_test_AAPL_2015-10-01 stocknet_test_AAPL_2015-10-02
set -e
CKPT="${1:?usage: $0 <ckpt> <scenario1> [scenario2 ...]}"
shift
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/signals
for sc in "$@"; do
  echo "[batch] $sc"
  .venv/bin/python -u scripts/diagnostics/inspect_signals_jsonl.py \
    --ckpt "$CKPT" --scenario "$sc" --seed 32 \
    > "logs/jsonl_${sc}.log" 2>&1
done
echo "[batch] done. files:"
ls -lh logs/signals/
