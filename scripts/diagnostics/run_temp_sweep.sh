#!/bin/bash
# DIAG 2: temperature sweep on ipo_seed8_best.pt over the StockNet smoke set.
# Tests whether decode-time greediness is masking direction conditioning.
set -e
cd /home/dgonier/ecology_experiment/trophic
echo "=== TEMPERATURE SWEEP on ipo_seed8_best.pt ==="
echo "Reference (T=0.0 / greedy): MCC=0.000, ACC=0.720"
echo ""
for T in 0.5 1.0 1.5; do
  echo "=========================================="
  echo "=== T=$T ==="
  echo "=========================================="
  TROPHIC_CKPT=checkpoints/ipo_seed8_best.pt TROPHIC_SEED=8 \
    STOCKNET_MAX_PER_TICKER=10 TROPHIC_DECODE_TEMP=$T \
    timeout 1500 .venv/bin/python -u scripts/eval_stocknet.py 2>&1 \
    | grep -E "ACCURACY|MCC|TP=|abstained|parse_failed|RESULTS|^  [A-Z]+:" \
    | head -20
  echo ""
done
echo "=== SWEEP COMPLETE ==="
