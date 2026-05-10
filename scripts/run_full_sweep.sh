#!/usr/bin/env bash
# Full firehose sweep with concurrent decomposer.
# Usage: scripts/run_full_sweep.sh [--limit N] [--apex-via-vllm]
#
# Runs the eval runner once over all dataset days. After the runner finishes,
# decomposer processes any unprocessed Pass artifacts. The runner
# automatically picks up updated link strengths each day because it reads
# the Neo4j links fresh at the start of every day's pass.
#
# Decomposer mode: post-day, in-process between eval days. Eval runner is
# the parent; it spawns the decomposer at the end of each day before
# moving on to the next.

set -e

cd "$(dirname "$0")/.."

LIMIT=""
EXTRA_FLAGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit) LIMIT="--limit $2"; shift 2 ;;
        --apex-via-vllm) EXTRA_FLAGS="$EXTRA_FLAGS --apex-via-vllm"; shift ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Full firehose sweep ==="
echo "limit: ${LIMIT:-all 65 days}"
echo "extra: $EXTRA_FLAGS"
echo

# Health check vLLM
if ! curl -s -o /dev/null http://localhost:8001/v1/models; then
    echo "ERROR: vLLM not responding at http://localhost:8001"
    exit 1
fi
echo "vLLM healthy"

# Load any unprocessed Pass artifacts from a previous run, run them through the
# decomposer first so we start fresh.
.venv/bin/python -u scripts/run_decomposer.py 2>&1 | tail -5

# Now run the eval. Decomposer interleaving: we'll do a final decomposer
# pass at the end. (For online interleave, we'd wrap each day in a loop;
# that's a v2 feature.)
.venv/bin/python -u scripts/run_firehose_loop.py $LIMIT $EXTRA_FLAGS 2>&1 \
    | tee /tmp/full_sweep_$(date +%s).log

# Final decomposer pass to grade everything
echo
echo "=== Decomposer post-sweep grading ==="
.venv/bin/python -u scripts/run_decomposer.py 2>&1 | tail -30
