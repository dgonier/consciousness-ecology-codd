# Mission 05f: debate-smoke-gate (3-day debate verification)

**Handle**: phase3-A-05f
**Phase**: 3 (sequential, after 05e)
**Mission file**: phase3-A-05f-debate-smoke-gate.md
**Dependencies**: phase3-A-05e (runner wiring)
**Blocks**: phase3-A-05g (full sweep)

---

## Before You Start

```bash
grep -E "@all|@phase3|@phase3-A|@phase3-A-05f" tasks_v4/scratchpad.md | tail -30
```

```bash
sed -i 's/phase3-A-05f:PENDING/phase3-A-05f:RUNNING/' tasks_v4/scratchpad.md 2>/dev/null || true
```

---

## Goal

A second smoke gate — analog to phase 2.5 but specifically for the
DEBATE mechanism. The cost of full v4 sweep is now ~110+ min on top of
8 control paths; a debate-specific failure on day 2 would cost real time.

This gate catches:
- Debate orchestration crashes mid-round.
- Token-budget blowouts (per-predator round-2 visibility overruns context).
- Per-predator validators rejecting >50% of commits.
- Predators not actually differentiating (all 4 propose identical orders).
- Thesis book lifecycle bugs (theses never close, never expire, etc.).

---

## Files to Create / Modify

**Create:**
- `scripts/v4_debate_smoke.sh` — 3-day ECO-DEBATE-QWEN-only smoke wrapper.
- `scripts/v4_debate_smoke_report.py` — parser + verdict + report writer.
- `tasks_v4/DEBATE_SMOKE_RESULTS.md` — the report.

**No source modifications** in this mission. Observe; don't change.

---

## Pass Criteria

- Run completes without unhandled exception.
- **Per-day predator coverage**: all 4 predators produce at least one
  output (no silent crashes).
- **Order count**: total across all 4 predators ≤ 16 per day (4 per
  predator average ceiling).
- **Rejection rate** (per predator OR aggregate): < 50%.
- **Diversity**: across all 4 predators × 3 days, at least 3 of the 4
  philosophies should have opened at least one thesis. If 3+ predators
  open zero theses, the diversity goal isn't met — soft fail.
- **Token budget**: no day's debate orchestration exceeds 80% of Qwen's
  16K context window on any predator's round 2 or round 3 call. (Parse
  this from the runner's log — DSPy / vLLM log prompt token counts.)

---

## Failure Modes

| Symptom | Severity | Action |
|---|---|---|
| Crash mid-round | HARD-FAIL | Post @all; do not unblock 05g. |
| Rejection > 50% | HARD-FAIL | Post @phase3-A-05c/05d; do not unblock. |
| 3+ predators silent | HARD-FAIL | Post @phase3-A-05d (orchestration bug). |
| 3+ predators open zero theses | SOFT-FAIL | Post @phase3-A-05d (prompt) + @phase3-A-05b (thesis schema usability). MAY unblock with flag. |
| Token > 80% on any call | SOFT-FAIL | Post @phase3-A-05d (compression). MAY unblock with flag. |
| Order count > 16/day average | SOFT-FAIL | Post @phase3-A-05d (per-predator over-emission). MAY unblock. |

---

## Implementation Steps

1. **Wrapper script** (`v4_debate_smoke.sh`):

   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   cd /home/dgonier/ecology_experiment/trophic
   curl -sf http://localhost:8001/v1/models > /dev/null || {
     echo "FATAL: vLLM unreachable"; exit 2
   }
   LOG=/tmp/v4_debate_smoke.log
   OUT=/tmp/v4_debate_smoke_out
   mkdir -p "$OUT"
   .venv/bin/python -u scripts/run_firehose_loop.py \
       --path eco-debate-qwen \
       --universe dow30 \
       --apex-pm --apex-via-vllm \
       --vllm-base http://localhost:8001/v1 \
       --max-tokens 4096 \
       --max-days 3 \
       --use-interrogator \
       --output "$LOG"
   .venv/bin/python scripts/v4_debate_smoke_report.py "$LOG" "$OUT"
   ```

2. **Report parser** (`v4_debate_smoke_report.py`):

   Parses the runner log + eval JSONL + debate_transcripts. Produces
   `tasks_v4/DEBATE_SMOKE_RESULTS.md` with:
   - Verdict at top: PASS / SOFT-FAIL / HARD-FAIL.
   - Per-day per-predator breakdown: round counts, orders, rejections,
     validated, equity delta, theses opened/closed.
   - Diversity check: which philosophies actually behaved differently
     (overlap between predators' chosen tickers, % of orders unique to
     each predator).
   - Token budget summary: max prompt tokens observed per round/predator
     (parsed from DSPy log lines if available).
   - Sample debate transcript excerpt: one good round-2 objection
     verbatim if found.
   - Soft-fails / hard-fails called out at the top.

3. **Diversity metric**:

   For each predator, list the unique tickers they opened theses on.
   Compute Jaccard similarity between each pair of predators over the
   3 days. If average pairwise Jaccard > 0.7, predators are too similar —
   diversity soft-fail.

---

## Acceptance Criteria

- [ ] `scripts/v4_debate_smoke.sh` runs without unhandled exception.
- [ ] `tasks_v4/DEBATE_SMOKE_RESULTS.md` exists with all required sections.
- [ ] Verdict line at top of report.
- [ ] If PASS or soft-fail-ok: phase3-A-05g unblocked.
- [ ] If hard-fail: 05g stays blocked, MESSAGES posted with diagnosis.

---

## Testing Conditions

### 1. Smoke runs

```bash
bash scripts/v4_debate_smoke.sh
echo "exit code: $?"
```
**Expected**: exits 0 (or the script halts with a clear FATAL on infra).

### 2. Report exists with verdict

```bash
test -f tasks_v4/DEBATE_SMOKE_RESULTS.md && head -5 tasks_v4/DEBATE_SMOKE_RESULTS.md
```
**Expected**: shows verdict line and section headers.

### 3. All 4 predators present

```bash
grep -E "momentum|value|mean_revert|event_driven" tasks_v4/DEBATE_SMOKE_RESULTS.md | wc -l
```
**Expected**: `>= 4` (each appears at least once in per-predator breakdown).

---

## Coordination

- This is the gate before the 110-min sweep. Be honest in the report.
- If you find a debate-specific bug that none of 05a-05e caught, post
  the diagnosis in MESSAGES with specific recommendation (which mission
  needs to fix, what specifically).
- The debate path is genuinely novel — expect to discover ≥1 unforeseen
  failure mode. Document, don't gloss.

---

## When Done

```bash
grep -E "@all|@phase3" tasks_v4/scratchpad.md | tail -30
```

If verdict is **PASS or soft-fail-ok**:

```bash
sed -i 's/phase3-A-05f:RUNNING/phase3-A-05f:DONE/' tasks_v4/scratchpad.md
sed -i 's/phase3-A-05g:BLOCKED.*$/phase3-A-05g:PENDING/' tasks_v4/scratchpad.md

cat >> tasks_v4/scratchpad.md <<'EOF'

[YYYY-MM-DD HH:MM] phase3-A-05f > @phase3-A-05g: debate smoke gate PASSED (with X soft-fails). 05g unblocked. See DEBATE_SMOKE_RESULTS.md.
EOF

mv tasks_v4/phase3-A-05f-debate-smoke-gate.md tasks_v4/completed/
```

If verdict is **HARD-FAIL**:

```bash
# STATUS stays RUNNING.
cat >> tasks_v4/scratchpad.md <<'EOF'

[YYYY-MM-DD HH:MM] phase3-A-05f > @all: debate smoke gate HARD-FAILED. <describe>. See DEBATE_SMOKE_RESULTS.md. 05g remains BLOCKED.
EOF
# Do not move mission file.
```
