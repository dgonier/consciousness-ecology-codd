# Mission 05: wire-interrogator-and-run

**Handle**: phase3-A (agent A continuing from phase 1)
**Phase**: 3 (sequential, after phase 2 + smoke gate)
**Mission file**: phase3-A-05-wire-interrogator-and-run.md
**Dependencies**: phase2-B:02, phase2-C:03, phase2-D:04, phase2_5:smoke
**Blocks**: phase4-PARKED:06 (deferred future work)

---

## Before You Start

```bash
grep -E "@all|@phase3|@phase3-A" tasks_v4/scratchpad.md | tail -50
cat tasks_v4/SMOKE_RESULTS.md
```

**Hard gate**: if `SMOKE_RESULTS.md` shows hard-fails, STOP. Do not run
a 66-day sweep with broken parts. Post in MESSAGES requesting triage.

```bash
sed -i 's/phase3-A:05:PENDING/phase3-A:05:RUNNING/' tasks_v4/scratchpad.md
```

---

## Goal

Two sub-deliverables:

**A. Wire the existing `InterrogatorHerbivore` into the PM firehose
pipeline.** The interrogator agent exists at
`trophic/agents/interrogator_herbivore.py` — it uses Qwen3-4B as a planner
and Qwen2.5-Math-1.5B as a math-specialist solver to produce
math-grounded synthesis broadcasts. It's wired into training pipelines
(SFT, IPO, GRPO, RL) but **never into the PM/firehose runner**. v3.3's
ECO path therefore omitted the math-interrogator signal entirely.

**B. Run the v4 sweep on the same 66-day window as v3.3.** Same 9 paths
(3 models × 3 pipelines), same data, all v4 features enabled.

This is a long-running GPU mission. Plan for 4-6 hours of sweep time
plus 1-2 hours of integration / sanity work.

---

## Files to Create / Modify

**Modify:**
- `scripts/run_firehose_loop.py` — integrate `InterrogatorHerbivore`
  into the ECO pipeline only (not BARE, not ORACLE). The interrogator's
  output broadcast is added to the producer broadcasts that the herbivore
  / predator stack already consumes. Must be opt-in via a flag
  (`--use-interrogator`) defaulting to ON for v4 sweeps.
- `trophic/agents/interrogator_herbivore.py` — verify the `forward()` /
  `__call__` API can be invoked from the firehose runner without an active
  training loop. If it can't, add a thin `predict_only()` method.

**Create:**
- `scripts/v4_run_full_sweep.sh` — the actual launch script for the 66-day
  9-way sweep, mirroring v3.3's invocation.
- `tasks_v4/V4_RUN_LOG.md` — a markdown log you update as the sweep
  progresses (similar to how v3.3 was tracked in conversation).
- `data/firehose_eval/runs/run_<DATE>_pm_v4/` — final archive directory
  following the v3.3 convention (`eval.jsonl`, `sweep.log`,
  `run_meta.json`).

---

## Implementation Steps

### Step 1: integrate the interrogator

1. **Read `interrogator_herbivore.py` end-to-end.** Understand its
   producer-payload reading and its synthesis-broadcast output.

2. **Identify where ECO's producer broadcasts are aggregated** in
   `run_firehose_loop.py`. Likely there's a step like
   `broadcasts = collect_producer_broadcasts(...)` followed by
   `herb_outputs = run_herbivore_stack(broadcasts)`.

3. **Inject the interrogator as an additional herbivore** alongside the
   existing technical / quantitative herbivores. The interrogator
   consumes the same producer broadcasts and emits one
   `<synthesis kind="interrogator">` broadcast per scenario per day.

4. **Verify the apex prompt change.** The ECO PM apex prompt should now
   include the interrogator synthesis in its observations payload. Confirm
   visually: dump one day's full apex prompt and grep for `kind="interrogator"`.

5. **CLI flag**: add `--use-interrogator` (default True for ECO paths,
   ignored for BARE/ORACLE). Add `--no-use-interrogator` to disable for
   debugging.

### Step 2: smoke-test the interrogator integration

```bash
# Single-day, ECO-QWEN only, with interrogator on.
.venv/bin/python -u scripts/run_firehose_loop.py \
    --path ecology-qwen \
    --apex-pm --apex-via-vllm \
    --vllm-base http://localhost:8001/v1 \
    --max-tokens 4096 \
    --max-days 1 \
    --use-interrogator \
    --output /tmp/v4_interrogator_smoke.log
```

Confirm:
- Apex prompt for that day contains an interrogator synthesis broadcast.
- The interrogator's MathHost was actually invoked (look for
  `MathHost.solve` or similar log lines).
- Run completes without exception.

### Step 3: full sweep

`scripts/v4_run_full_sweep.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/dgonier/ecology_experiment/trophic

# Ensure local vLLM is up at port 8001 with --max-model-len 16384
curl -sf http://localhost:8001/v1/models > /dev/null || {
    echo "FATAL: local vLLM on port 8001 not reachable"
    exit 1
}

# Bedrock credentials present?
aws sts get-caller-identity > /dev/null || {
    echo "FATAL: aws cli not configured for Bedrock"
    exit 1
}

LOG=/tmp/pm_sweep_v4.log
ARCHIVE=data/firehose_eval/runs/run_$(date +%Y-%m-%d)_pm_v4

.venv/bin/python -u scripts/run_firehose_loop.py \
    --path all-9 \
    --apex-pm \
    --apex-via-vllm \
    --vllm-base http://localhost:8001/v1 \
    --max-tokens 4096 \
    --use-interrogator \
    --use-strategy-committee \
    --use-tax-aware-validators \
    --philosophy-weights tasks_v4/philosophy_weights.yaml \
    2>&1 | tee "$LOG"

mkdir -p "$ARCHIVE/passes"
cp "$LOG" "$ARCHIVE/sweep.log"
# Latest eval JSONL written by the runner:
LATEST_EVAL=$(ls -t data/firehose_eval/eval_*.jsonl | head -1)
cp "$LATEST_EVAL" "$ARCHIVE/eval.jsonl"

echo "v4 sweep archived to $ARCHIVE"
```

### Step 4: update `V4_RUN_LOG.md` daily

The runner emits a `[N/66]` line at each day start and per-path `PM:` lines.
Mirror v3.3's tracking style: a daily totals table comparing v3.3 vs v4
per path. Snippet:

```markdown
## Day N/66 (<date>)

| Path | v3.3 equity | v4 equity | Δ |
|---|---|---|---|
| BARE-QWEN | $... | $... | ... |
| ECO-QWEN  | $... | $... | ... |
| ...
```

### Step 5: build the archive

After PORTFOLIO RESULTS lands:

1. Copy log + eval JSONL into the archive dir (the script above does this).
2. Write `run_meta.json` mirroring v3.3's format with v4-specific fields
   (`feature_flags="...interrogator=on,strategy_committee=on,tax_aware=on,full_window_oracle=on"`).
3. Compute pre-tax equities by adding back tax_owed from the final
   PORTFOLIO RESULTS block.
4. Compare against v3.3 baseline. Document headline findings.

### Step 6: post-sweep analysis

Open questions to answer in `V4_RUN_LOG.md`:

- **Did BARE-QWEN's lead hold, shrink, or invert vs ECO-QWEN?** The v4
  thesis is ECO should now beat BARE *net*, not just pre-tax.
- **Order count delta**: v3.3 ECO-QWEN had 295 orders. v4 target is <150.
- **Rejection rate**: what fraction of orders were rejected by the new
  validators? Is the apex learning to comply over the 66 days, or does the
  rejection rate stay flat?
- **Horizon distribution**: of all v4 orders, what % were h1 vs h5 vs h20
  vs h60? Healthy distribution is ~10% h1, 30% h5, 40% h20, 20% h60.
  Heavy h1 = the apex is ignoring the prompt.
- **Oracle-Opus**: full-window prompt should yield <100 orders and >+15%
  pre-tax. Did it?

---

## Acceptance Criteria

- [ ] Interrogator is invoked on every ECO day (verifiable in sweep log).
- [ ] v4 sweep ran 66 days end-to-end without crash.
- [ ] Archive directory exists with `eval.jsonl`, `sweep.log`,
      `run_meta.json` populated.
- [ ] `V4_RUN_LOG.md` has all 66 daily updates and a final headline section.
- [ ] Headline comparison vs v3.3 written.
- [ ] If v4 underperforms v3.3, the failure mode is documented (not hidden).

---

## Testing Conditions

### 1. Interrogator smoke run completes

```bash
.venv/bin/python -u scripts/run_firehose_loop.py \
    --path ecology-qwen --apex-pm --apex-via-vllm \
    --vllm-base http://localhost:8001/v1 --max-tokens 4096 \
    --max-days 1 --use-interrogator --output /tmp/v4_interrogator_smoke.log
echo "exit code: $?"
```
**Expected**: exit 0, log file contains 1 `PM:` line and at least one
indication that the interrogator was invoked (e.g., `MathHost` log entry
or `interrogator` broadcast in the apex prompt dump).

### 2. Full sweep ran to PORTFOLIO RESULTS

```bash
grep -c "PORTFOLIO RESULTS" /tmp/pm_sweep_v4.log
```
**Expected**: `1`.

### 3. Archive directory is complete

```bash
ls data/firehose_eval/runs/run_*_pm_v4/
```
**Expected**: contains `eval.jsonl`, `sweep.log`, `run_meta.json`,
`passes/` (may be empty if not used in v4).

### 4. run_meta.json has the expected fields

```bash
.venv/bin/python -c "
import json, glob
path = sorted(glob.glob('data/firehose_eval/runs/run_*_pm_v4/run_meta.json'))[-1]
d = json.load(open(path))
needed = ['label','snapshotted_at','git_sha','n_days','first_date','last_date',
          'cmd','feature_flags','final_equities_net','final_equities_pretax',
          'order_counts','headline_findings']
missing = [k for k in needed if k not in d]
assert not missing, f'missing keys in {path}: {missing}'
print('run_meta.json has all expected keys')
"
```
**Expected**: `run_meta.json has all expected keys`.

### 5. V4_RUN_LOG.md exists and has a final headline section

```bash
test -f tasks_v4/V4_RUN_LOG.md && grep -c "Day 66" tasks_v4/V4_RUN_LOG.md
```
**Expected**: file exists, has at least 1 `Day 66` marker.

### 6. No regressions in unit tests

```bash
.venv/bin/python -m pytest tests/ -x -q
```
**Expected**: green.

---

## Coordination

- **You will be running for hours.** Use background bash + Monitor for
  the sweep itself; do interactive work (analysis, log writing) in
  foreground.
- **If you see catastrophic divergence early** (e.g., day 3 has every
  path < $90K, or order count is 50+/day), HALT the sweep and post in
  MESSAGES @phase2-D + @phase2-B. Better to triage at hour 1 than at
  hour 6.
- **Spin down the Modal Hexis vLLM at end of sweep** (task #11 from the
  v3.3 task list — confirm with user whether it should be auto-stopped
  or left up).

---

## When Done

```bash
grep -E "@all|@phase3" tasks_v4/scratchpad.md | tail -30
sed -i 's/phase3-A:05:RUNNING/phase3-A:05:DONE/' tasks_v4/scratchpad.md

cat >> tasks_v4/scratchpad.md <<'EOF'

[YYYY-MM-DD HH:MM] phase3-A > @all: v4 sweep complete. Archive at data/firehose_eval/runs/run_<date>_pm_v4/. Headline comparison vs v3.3 in tasks_v4/V4_RUN_LOG.md. Interrogator was wired into ECO path and active on all 66 days. <Add headline result here: e.g., "ECO-QWEN +XX.XX% net vs v3.3's +8.32% — primary win was order count drop from 295 to NNN.">
EOF

git mv tasks_v4/phase3-A-05-wire-interrogator-and-run.md tasks_v4/completed/
```
