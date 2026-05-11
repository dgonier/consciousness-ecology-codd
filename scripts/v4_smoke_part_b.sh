#!/usr/bin/env bash
# v4 phase-2.5 smoke gate — Part B: 3-day ECO-QWEN mini-sweep.
#
# Runs scripts/run_firehose_loop.py for 3 trading days, ECO-QWEN path only,
# with all v4 features enabled (--apex-pm via local vLLM). Parses the per-day
# eval JSONL afterwards to compute order count, rejection rate, and horizon
# distribution.
#
# Pre-flight: verifies vLLM at localhost:8001 is reachable.
# Exit codes:
#   0 — PASS or SOFT-FAIL (continue to phase 3 OK)
#   1 — HARD FAIL (do NOT unblock phase 3)
#   2 — vLLM unreachable
#
# Outputs:
#   /tmp/v4_smoke_b.log — full runner stdout
#   /tmp/v4_smoke_b_out/ — eval JSONL + pass artifacts
#   tasks_v4/SMOKE_RESULTS.md is written by the python parser below.

set -uo pipefail

ROOT="/home/dgonier/ecology_experiment/trophic"
cd "$ROOT" || exit 1

LOG="/tmp/v4_smoke_b.log"
OUT_DIR="/tmp/v4_smoke_b_out"
VLLM_BASE="http://localhost:8001/v1"

echo "── v4 SMOKE Part B — 3-day ECO-QWEN mini-sweep ──"
echo "vLLM base: $VLLM_BASE"

if ! curl -sf "$VLLM_BASE/models" > /dev/null 2>&1 ; then
    echo "FATAL: vLLM unreachable at $VLLM_BASE/models — start the server first"
    echo "SMOKE_PART_B: FAIL vllm_unreachable"
    exit 2
fi
echo "vLLM healthy"

# Clean prior smoke outputs
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo
echo "Launching runner — output at $LOG"
echo "Expected wall-time: ~5-10 min (3 days × ~2 min/day on Qwen-4B)"
echo

# Time-bound the runner; 20 min hard ceiling per mission spec.
START_T=$(date +%s)
timeout 1200 .venv/bin/python -u scripts/run_firehose_loop.py \
    --path ecology \
    --apex-pm \
    --apex-via-vllm \
    --vllm-base "$VLLM_BASE" \
    --limit 3 \
    --no-lm-cache \
    --out "$OUT_DIR" \
    > "$LOG" 2>&1
RC=$?
END_T=$(date +%s)
ELAPSED=$((END_T - START_T))
echo "runner exit=$RC  elapsed=${ELAPSED}s  log=$LOG"

if [[ $RC -ne 0 ]]; then
    echo "Runner failed (rc=$RC). Last 60 lines of log:"
    tail -60 "$LOG"
fi

# Find the eval JSONL the runner wrote
EVAL_JSONL=$(ls -t "$OUT_DIR"/eval_*.jsonl 2>/dev/null | head -1)
if [[ -z "$EVAL_JSONL" ]]; then
    echo "FATAL: no eval JSONL written to $OUT_DIR — runner aborted before write"
    echo "SMOKE_PART_B: FAIL no_eval_jsonl"
    exit 1
fi
echo "eval jsonl: $EVAL_JSONL"

# Parse & write SMOKE_RESULTS.md via python helper. Pass runner exit code
# + elapsed time so the report can flag exceptions and slow runs.
.venv/bin/python -u scripts/v4_smoke_part_b_report.py \
    --log "$LOG" \
    --eval-jsonl "$EVAL_JSONL" \
    --runner-rc "$RC" \
    --elapsed-s "$ELAPSED" \
    --report-out tasks_v4/SMOKE_RESULTS.md

REPORT_RC=$?
exit $REPORT_RC
