#!/usr/bin/env bash
# 3-day ECO-DEBATE-QWEN smoke gate (phase3-A-05f).
#
# Runs the runner for 3 days on the eco-debate-qwen path only, with
# the apex on local vLLM (port 8001) and the PMInterrogator enabled.
# After the run, hands the log + the eval JSONL + the debate
# transcripts to `v4_debate_smoke_report.py` which writes the verdict
# report to tasks_v4/DEBATE_SMOKE_RESULTS.md.
#
# Pre-flight: vLLM at $VLLM_BASE must respond on /v1/models. If not,
# the script exits 2 with a FATAL — we don't want to fall back to
# in-process Qwen here because the smoke gate is specifically the
# vLLM path that 05g will sweep on.
set -euo pipefail

cd /home/dgonier/ecology_experiment/trophic

VLLM_BASE="${VLLM_BASE:-http://localhost:8001/v1}"
if ! curl -sf "${VLLM_BASE}/models" > /dev/null; then
    echo "FATAL: vLLM unreachable at ${VLLM_BASE}" >&2
    exit 2
fi

LOG=/tmp/v4_debate_smoke.log
OUT_DIR=/tmp/v4_debate_smoke_out
mkdir -p "${OUT_DIR}"
# Wipe the runner log on each invocation — appending across runs caused
# the report parser to double-count days.
rm -f "${LOG}"

# Wipe any prior transcript so this smoke's diversity check is over
# THIS smoke only — the runner appends to the per-run_label jsonl.
TRANSCRIPT_DIR=data/firehose_eval/debate_transcripts
mkdir -p "${TRANSCRIPT_DIR}"
TRANSCRIPT_FILE="${TRANSCRIPT_DIR}/v4_debate_smoke.jsonl"
rm -f "${TRANSCRIPT_FILE}"

echo "==> running 3-day eco-debate-qwen smoke (log: ${LOG})"
.venv/bin/python -u scripts/run_firehose_loop.py \
    --path eco-debate-qwen \
    --universe dow30 \
    --apex-pm \
    --apex-via-vllm \
    --vllm-base "${VLLM_BASE}" \
    --max-tokens 4096 \
    --max-days 3 \
    --use-interrogator \
    --run-label v4_debate_smoke \
    --no-lm-cache \
    --output "${LOG}" 2>&1 | tee "${LOG}"

RC=${PIPESTATUS[0]}
echo "==> runner exit code: ${RC}"

# Locate the most recent eval jsonl (the runner writes
# data/firehose_eval/eval_<epoch>.jsonl).
EVAL_JSONL=$(ls -t data/firehose_eval/eval_*.jsonl 2>/dev/null | head -1)
echo "==> eval jsonl: ${EVAL_JSONL:-<none>}"
echo "==> transcript: ${TRANSCRIPT_FILE}"

.venv/bin/python -u scripts/v4_debate_smoke_report.py \
    --log "${LOG}" \
    --eval "${EVAL_JSONL:-/dev/null}" \
    --transcript "${TRANSCRIPT_FILE}" \
    --out tasks_v4/DEBATE_SMOKE_RESULTS.md \
    --runner-rc "${RC}"

REPORT_RC=$?
echo "==> report exit code: ${REPORT_RC}"
echo "==> see tasks_v4/DEBATE_SMOKE_RESULTS.md"
exit ${REPORT_RC}
