# Mission 05g: full-sweep (66-day v4 9-way + DEBATE)

**Handle**: phase3-A-05g
**Phase**: 3 (sequential, after 05f)
**Mission file**: phase3-A-05g-full-sweep.md
**Dependencies**: phase3-A-05f (debate smoke gate)
**Blocks**: phase4-PARKED:06 (deferred)

---

## Before You Start

```bash
grep -E "@all|@phase3|@phase3-A|@phase3-A-05g" tasks_v4/scratchpad.md | tail -50
cat tasks_v4/completed/DEBATE_SMOKE_RESULTS.md 2>/dev/null | head -30
cat tasks_v4/completed/SMOKE_RESULTS.md 2>/dev/null | head -10
```

**Hard gate**: do not start if the debate smoke gate report shows HARD-FAIL.

```bash
sed -i 's/phase3-A-05g:PENDING/phase3-A-05g:RUNNING/' tasks_v4/scratchpad.md 2>/dev/null || true
```

---

## Goal

Run the v4 sweep. 66 days, 9 paths (one of which is debate), Dow 30
universe, interrogator wired into ECO, tax-aware validators, multi-horizon
forecasts, full-window oracle.

This is a multi-hour GPU-bound mission. Plan accordingly.

---

## Files to Create / Modify

**Create:**
- `tasks_v4/V4_RUN_LOG.md` — running daily log similar to how v3.3 was tracked.
- `data/firehose_eval/runs/run_<DATE>_pm_v4/` — final archive directory.

**No source modifications** in this mission.

---

## Implementation Steps

### Step 1: pre-flight

```bash
# Local vLLM up?
curl -sf http://localhost:8001/v1/models | head
# AWS Bedrock auth working?
aws sts get-caller-identity | head
# Disk space for ~50MB of logs + eval JSONL?
df -h .
```

If any of these fail, halt and post @all.

### Step 2: launch

```bash
bash scripts/v4_run_sweep.sh
```

Expected runtime: 4-6 hours total. Use background bash + Monitor for the
sweep; do interactive work (V4_RUN_LOG.md updates, sanity inspection) in
foreground.

### Step 3: daily tracking

Update `tasks_v4/V4_RUN_LOG.md` after each day's PORTFOLIO RESULTS line.
Format mirrors v3.3 conversation tracking:

```markdown
## Day N/66 (<date>)

| Path | Equity | vs $100K | Δ vs day N-1 | Inv % | Notes |
|---|---|---|---|---|---|
| BARE-QWEN | $... | ... | ... | ... | |
| ECO-QWEN  | $... | ... | ... | ... | |
| ECO-DEBATE-QWEN | $... | ... | ... | ... | per-predator: M:..., V:..., MR:..., ED:... |
| ...
```

For ECO-DEBATE-QWEN, include the per-predator breakdown — that's the
leaderboard that emerges as the run progresses.

### Step 4: halt conditions

If you observe any of these during the run, **halt the sweep** and post
@all:

- All 9 paths down >10% by day 10.
- Any path crashes mid-sweep with an unhandled exception.
- The debate path's predator capital distribution diverges to >90/10
  within 20 days (one predator dominating, others fully starved) — that's
  evolutionary pressure faster than expected; pause to inspect.
- Order count for any path exceeds 50/day average for 3+ consecutive days
  (rejection-loop runaway).

Halting is reversible (re-launch from a checkpoint if one exists, or
re-run); a bad sweep that runs to completion costs more.

### Step 5: archive at completion

The launch script (`v4_run_sweep.sh`) handles archiving. Verify:

```bash
ls data/firehose_eval/runs/run_$(date +%Y-%m-%d)_pm_v4/
# Expected: eval.jsonl, sweep.log, run_meta.json, passes/, debate_transcripts.jsonl
```

Write `run_meta.json` mirroring v3.3 schema with v4-specific fields:

```json
{
  "label": "run_<DATE>_pm_v4",
  "snapshotted_at": "...",
  "git_sha": "...",
  "n_days": 66,
  "first_date": "...",
  "last_date": "...",
  "note": "9-path sweep: 3 BARE, 3 ECO (with interrogator), 2 ORACLE (sonnet dropped), 1 ECO-DEBATE-QWEN with 4 predators. Dow 30 universe. Multi-horizon forecasts + strategy committee + tax-aware validators.",
  "cmd": "...",
  "feature_flags": "universe=dow30,interrogator=on,strategy_committee=on,tax_aware=on,full_window_oracle=on,debate=on,async=on,context=16384",
  "final_equities_net": {...},
  "final_equities_pretax": {...},
  "order_counts": {...},
  "debate_predator_equities_net": {
    "momentum": ...,
    "value": ...,
    "mean_revert": ...,
    "event_driven": ...
  },
  "debate_predator_theses_count": {
    "momentum": {"opened": N, "matured": N, "invalidated": N, "expired": N},
    ...
  },
  "headline_findings": [...]
}
```

### Step 6: post-sweep analysis

Open questions to answer in V4_RUN_LOG.md's final section:

1. **Did v4 ECO-QWEN beat v3.3 BARE-QWEN?** v3.3's bar is +9.67% net. The
   v4 thesis depends on ECO finally net-beating BARE thanks to multi-horizon
   discipline.
2. **Order count compression**: v3.3 ECO-QWEN had 295 orders. v4 target
   is <150. How did it land?
3. **Did the debate path beat single-apex ECO?** ECO-QWEN vs ECO-DEBATE-QWEN
   final net return. The win/lose margin tells us whether per-predator
   debate generates real lift.
4. **Which philosophy won the debate run?** Leaderboard of the 4 predators
   inside ECO-DEBATE-QWEN. End-of-run capital share = each predator's
   fitness signal.
5. **Thesis lifecycle stats**: how many theses opened total, what fraction
   matured vs invalidated vs expired. High invalidation rate = the apex's
   forecasts are unstable. High maturation rate without alpha = thesis
   matures but doesn't pay.
6. **Oracle paradox revisited**: ORACLE-QWEN and ORACLE-OPUS with the
   full-window prompt — did the buy-low-sell-high directive convert the
   alpha that v3.3 lost to churn?
7. **Sonnet without ORACLE-SONNET**: did dropping the worst Oracle hurt
   anything informative? (Probably not — and that's a positive finding.)

### Step 7: comparison artifact

Write `tasks_v4/V4_VS_V3_3.md` — a single table comparing v3.3 vs v4 on
the 8 paths that exist in both, plus a separate row for the new
ECO-DEBATE-QWEN. Pre-tax and net columns. Plus headline analytic
paragraph (3-5 sentences) on what changed and why.

---

## Acceptance Criteria

- [ ] All 9 paths ran 66 days end-to-end without crash.
- [ ] Archive directory complete with eval.jsonl, sweep.log, run_meta.json, debate_transcripts.jsonl.
- [ ] `V4_RUN_LOG.md` has 66 daily updates + final headline section.
- [ ] `V4_VS_V3_3.md` comparison written.
- [ ] If v4 underperforms v3.3 net, the failure mode is documented (not hidden).
- [ ] Predator leaderboard captured.

---

## Testing Conditions

### 1. Sweep completed cleanly

```bash
grep -c "PORTFOLIO RESULTS" /tmp/pm_sweep_v4.log
```
**Expected**: `1`.

### 2. Archive complete

```bash
ls data/firehose_eval/runs/run_*_pm_v4/ | head -20
```
**Expected**: `eval.jsonl`, `sweep.log`, `run_meta.json`, `debate_transcripts.jsonl`, `passes/`.

### 3. Comparison artifacts

```bash
test -f tasks_v4/V4_RUN_LOG.md && grep -c "Day 66" tasks_v4/V4_RUN_LOG.md
test -f tasks_v4/V4_VS_V3_3.md
```
**Expected**: file exists with at least 1 "Day 66" marker; comparison file exists.

### 4. run_meta has v4 fields

```bash
.venv/bin/python -c "
import json, glob
path = sorted(glob.glob('data/firehose_eval/runs/run_*_pm_v4/run_meta.json'))[-1]
d = json.load(open(path))
needed = ['final_equities_net','final_equities_pretax','order_counts',
          'debate_predator_equities_net','debate_predator_theses_count','headline_findings']
missing = [k for k in needed if k not in d]
assert not missing, f'missing keys: {missing}'
print('run_meta v4 schema OK')
"
```
**Expected**: `run_meta v4 schema OK`.

### 5. No regressions in tests

```bash
.venv/bin/python -m pytest tests/ -x -q
```
**Expected**: green.

---

## Coordination

- This is the longest mission of the project. Budget 6+ hours.
- The user has invested a full day in v4 architecture; the run is the
  payoff. Be precise in the daily log.
- After completion, **stop Modal Hexis vLLM** if any deploys remain
  (per v3.3 task #11 pattern).
- Document the win/lose verdict in the final section of V4_RUN_LOG.md
  honestly. If ECO-DEBATE-QWEN loses to ECO-QWEN, that's a real
  finding — say so.

---

## When Done

```bash
grep -E "@all|@phase3" tasks_v4/scratchpad.md | tail -30
sed -i 's/phase3-A-05g:RUNNING/phase3-A-05g:DONE/' tasks_v4/scratchpad.md

cat >> tasks_v4/scratchpad.md <<'EOF'

[YYYY-MM-DD HH:MM] phase3-A-05g > @all: v4 sweep complete. Headline: <one-sentence verdict>. Archive at data/firehose_eval/runs/run_<DATE>_pm_v4/. Comparison vs v3.3 at tasks_v4/V4_VS_V3_3.md. Predator leaderboard: <M%/V%/MR%/ED% capital share>.
EOF

mv tasks_v4/phase3-A-05g-full-sweep.md tasks_v4/completed/
```
