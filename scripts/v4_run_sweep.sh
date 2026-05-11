#!/usr/bin/env bash
# v4_run_sweep.sh — launch the full 66-day v4 sweep.
#
# Path matrix (9): BARE-{QWEN,SONNET,OPUS}, ECOLOGY-{QWEN,SONNET,OPUS}
# (interrogator-enabled), ORACLE-{QWEN,OPUS}, ECO-DEBATE-QWEN (NEW).
# ORACLE-SONNET is intentionally dropped (was -6.73% in v3.3).
#
# `--async-paths` ON by default per phase3-A-05e coordinator note:
# v3.3 shipped the async dispatcher but never exercised it on a real
# sweep. Within-phase parallelism in the debate (16 calls/day at ~4s/
# call → ~16s/day) combined with cross-path parallelism is expected to
# drop per-day wall-clock from sum-across-paths to max-across-paths.
# Estimated total sweep: ~20-40 min vs ~6h sync.
#
# Outputs (relative to repo root):
#   data/firehose_eval/runs/run_<date>_pm_v4/
#     eval.jsonl             — one line per (path, date)
#     sweep.log              — full stdout
#     run_meta.json          — final equities + headline findings (written by hand from sweep.log)
#     debate_transcripts.jsonl — phase-1..4 dumps for the debate path
#
# Usage:
#   bash scripts/v4_run_sweep.sh
#   bash scripts/v4_run_sweep.sh --max-days 3        # smoke
#   RUN_LABEL=run_2026-05-10_pm_v4_run2 bash scripts/v4_run_sweep.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# Pre-flight ──────────────────────────────────────────────────────────
curl -sf -m 3 http://localhost:8001/v1/models > /dev/null || {
  echo "FATAL: vLLM port 8001 unreachable (Qwen apex would fail)"; exit 1
}

# Bedrock credentials are only required if any Sonnet/Opus path is active.
# `all-9` exercises Sonnet (BARE-SONNET, ECOLOGY-SONNET) and Opus
# (BARE-OPUS, ECOLOGY-OPUS, ORACLE-OPUS), so do the check by default.
if ! aws sts get-caller-identity > /dev/null 2>&1; then
  echo "FATAL: aws cli not configured for Bedrock"
  echo "       (needed for BARE-SONNET / ECOLOGY-SONNET / BARE-OPUS /"
  echo "        ECOLOGY-OPUS / ORACLE-OPUS in all-9)"
  exit 1
fi

# Archive layout ───────────────────────────────────────────────────────
RUN_LABEL="${RUN_LABEL:-run_$(date +%Y-%m-%d)_pm_v4}"
ARCHIVE="data/firehose_eval/runs/${RUN_LABEL}"
LOG="/tmp/${RUN_LABEL}.log"
mkdir -p "${ARCHIVE}/passes"

echo "v4 sweep label: ${RUN_LABEL}"
echo "archive       : ${ARCHIVE}"
echo "log           : ${LOG}"

# Sweep ────────────────────────────────────────────────────────────────
# Note: `--async-paths` ON, `--use-interrogator` ON, Dow 30 universe.
# Pass-through extra args (e.g. --max-days 3 for a smoke).
.venv/bin/python -u scripts/run_firehose_loop.py \
    --path all-9 \
    --universe dow30 \
    --apex-pm \
    --apex-via-vllm \
    --vllm-base http://localhost:8001/v1 \
    --max-tokens 6144 \
    --use-interrogator \
    --use-strategy-committee \
    --use-tax-aware-validators \
    --philosophy-weights tasks_v4/philosophy_weights.yaml \
    --debate-predators 4 \
    --async-paths \
    --run-label "${RUN_LABEL}" \
    "$@" \
    2>&1 | tee "${LOG}"

# Archive ──────────────────────────────────────────────────────────────
cp "${LOG}" "${ARCHIVE}/sweep.log"
LATEST_EVAL=$(ls -t data/firehose_eval/eval_*.jsonl 2>/dev/null | head -1 || true)
if [[ -n "${LATEST_EVAL}" ]]; then
  cp "${LATEST_EVAL}" "${ARCHIVE}/eval.jsonl"
  echo "copied eval JSONL: ${LATEST_EVAL} -> ${ARCHIVE}/eval.jsonl"
fi
DEBATE_T="data/firehose_eval/debate_transcripts/${RUN_LABEL}.jsonl"
if [[ -f "${DEBATE_T}" ]]; then
  cp "${DEBATE_T}" "${ARCHIVE}/debate_transcripts.jsonl"
  echo "copied debate transcripts: ${DEBATE_T} -> ${ARCHIVE}/debate_transcripts.jsonl"
fi

echo "v4 sweep archived to ${ARCHIVE}"
