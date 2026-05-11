# Mission 05e: runner-wiring (10th path + interrogator + drop ORACLE-SONNET)

**Handle**: phase3-A-05e
**Phase**: 3 (sequential, after 05a, 05c, 05d all DONE)
**Mission file**: phase3-A-05e-runner-wiring.md
**Dependencies**: phase3-A-05a (Dow 30), phase3-A-05c (sub-portfolio), phase3-A-05d (debate)
**Blocks**: phase3-A-05f (smoke gate), phase3-A-05g (sweep)

---

## Before You Start

```bash
grep -E "@all|@phase3|@phase3-A|@phase3-A-05e" tasks_v4/scratchpad.md | tail -50
```

Verify contracts:
```bash
.venv/bin/python -c "
from trophic.data.dow30_universe import DOW30
from trophic.beliefs.investment_thesis import InvestmentThesis, ThesisBook
from trophic.agents.predator_sub_portfolio import PredatorSubPortfolio
from trophic.agents.apex_portfolio import make_debate_portfolio
from trophic.beliefs.debate import run_debate, DebateRound3Commit
print('all 05a/05b/05c/05d contracts present')
"
```

```bash
sed -i 's/phase3-A-05e:PENDING/phase3-A-05e:RUNNING/' tasks_v4/scratchpad.md 2>/dev/null || true
```

---

## Goal

Wire everything together in `scripts/run_firehose_loop.py`:

1. **Add the 10th path**: `ECO-DEBATE-QWEN` (4-predator debate ecology, local Qwen-4B).
2. **Drop ORACLE-SONNET**: that was v3.3's worst path (-6.73%) and we're conserving compute.
3. **Wire the InterrogatorHerbivore** into both ECO-QWEN and ECO-DEBATE-QWEN paths (it currently only exists in training, never in PM).
4. **Default to Dow 30** universe for v4 sweep; legacy mode preserved.

After this mission lands, the sweep matrix is **9 paths** total:
- BARE-QWEN, BARE-SONNET, BARE-OPUS (3 controls)
- ECO-QWEN, ECO-SONNET, ECO-OPUS (3 ecology paths, now with interrogator)
- ORACLE-QWEN, ORACLE-OPUS (2 oracle paths, ORACLE-SONNET dropped)
- ECO-DEBATE-QWEN (1 debate path)

---

## Files to Create / Modify

**Modify:**
- `scripts/run_firehose_loop.py` — heavy edits.
- `trophic/agents/interrogator_herbivore.py` — verify it has a `predict_only()`-style entry point invocable outside the training loop. If not, add one.

**Create:**
- `scripts/v4_run_sweep.sh` — wrapper launch script for the full v4 sweep (used by 05g).

---

## Implementation Steps

### Step 1: path enumeration

The runner's `--path` CLI flag currently accepts strings like
`all-9`, `ecology-qwen`, etc. Refactor:

```python
ALL_PATHS_V4 = (
    "bare-qwen",
    "bare-sonnet",
    "bare-opus",
    "ecology-qwen",
    "ecology-sonnet",
    "ecology-opus",
    "oracle-qwen",
    # NOTE: oracle-sonnet dropped in v4 (was -6.73% in v3.3)
    "oracle-opus",
    "eco-debate-qwen",   # NEW
)
# Convenience aliases
PATH_GROUPS = {
    "all-9":          ALL_PATHS_V4,             # the v4 sweep matrix
    "bare":           ALL_PATHS_V4[0:3],
    "ecology":        ALL_PATHS_V4[3:6],
    "oracle":         ALL_PATHS_V4[6:8],
    "debate":         (ALL_PATHS_V4[8],),
    "v3-3-baseline":  (...) # legacy 9-way for back-compat re-runs
}
```

### Step 2: interrogator wiring (ECO paths only)

Find where ECO's producer broadcasts are collected before the apex sees
them. Inject the InterrogatorHerbivore's synthesis as one additional
herbivore broadcast.

```python
from trophic.agents.interrogator_herbivore import InterrogatorHerbivore

# Initialize once per run:
interrogator = InterrogatorHerbivore(...)  # use existing constructor; defaults to Qwen3-4B planner + math host

# Per-day, per-ECO-path:
def collect_eco_broadcasts(day_state, producer_broadcasts):
    interrogator_synth = interrogator.predict_only(producer_broadcasts, day_state)
    return [*producer_broadcasts, interrogator_synth]
```

If `predict_only` doesn't exist, add a thin wrapper. The forward method
of `InterrogatorHerbivore` likely takes a batch shape that doesn't apply
in single-scenario inference; the wrapper unwraps that.

Feature flag: `--use-interrogator` (default ON for ECO-* paths; ignored for BARE/ORACLE).

### Step 3: debate path setup

For the `eco-debate-qwen` path:

```python
def setup_debate_path(starting_cash: float = 100_000) -> ApexPortfolio:
    from trophic.agents.apex_portfolio import make_debate_portfolio
    return make_debate_portfolio(
        total_starting_cash=starting_cash,
        predators=(
            ("momentum",     "momentum"),
            ("value",        "value"),
            ("mean_revert",  "mean_revert"),
            ("event_driven", "event_driven"),
        ),
    )

def run_debate_day(portfolio: ApexPortfolio, today: str, days_remaining: int,
                   watchlist: list[str], observations: str, lm) -> dict:
    from trophic.beliefs.debate import run_debate
    from trophic.beliefs.validators import run_tax_aware_chain
    commits = run_debate(portfolio, today, days_remaining, watchlist, observations, lm)
    summary = {"predator_results": {}}
    for pid, commit in commits.items():
        sub = portfolio.sub_portfolios[pid]
        # Per-predator validation
        pred_state = portfolio.state_for_predator(pid, today, current_prices)
        validated, rejected = portfolio.filter_tax_aware_for_predator(pid, commit.final_orders, today, ...)
        validated, scaler_rejected = sub.validate_orders(validated, current_prices, today, ...)  # reuse v3.3 logic per sub-portfolio
        # Apply orders against sub-portfolio
        for order in validated:
            apply_order_to_sub(sub, order)
        # Update thesis book
        for new_thesis in commit.final_theses_to_open:
            sub.thesis_book.add(new_thesis)
        for tid in commit.final_theses_to_close:
            sub.thesis_book.close(tid, status="manual", today=today, ...)
        # Sweep overdue
        sub.thesis_book.sweep_overdue(today_idx=today_idx, ...)
        summary["predator_results"][pid] = {
            "validated": len(validated),
            "rejected": len(rejected) + len(scaler_rejected),
            "open_theses": len(sub.thesis_book.active_theses()),
        }
    return summary
```

### Step 4: per-day dispatch

The per-day loop now dispatches to each of the 9 paths. The debate path
takes longer (~3× LM calls per day) so its position in the dispatch
order matters if running async.

If `--async-paths` is set (the v3.3 async refactor exists but never ran a
sweep), dispatch all 9 paths concurrently. The debate path internally
parallelizes its 4 predators, so total per-day concurrency is up to 12
LM calls in flight.

If sequential, dispatch order is BARE → ECO → ORACLE → DEBATE.

### Step 5: archive structure

For the v4 sweep, archive shape:

```
data/firehose_eval/runs/run_<YYYY-MM-DD>_pm_v4/
├── eval.jsonl                       # one line per (path, date)
├── sweep.log                        # full stdout
├── run_meta.json                    # mirrors v3.3 schema
├── debate_transcripts.jsonl         # NEW: per-day debate transcripts (from 05d)
└── passes/                          # if any pass-level artifacts written
```

Per-path equity / orders / rejections live in `eval.jsonl`. For the debate
path, each JSONL line should include `predator_breakdown` with the four
predators' per-day equities so the leaderboard is reconstructable.

### Step 6: launch script

`scripts/v4_run_sweep.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/dgonier/ecology_experiment/trophic

# Pre-flight
curl -sf http://localhost:8001/v1/models > /dev/null || {
  echo "FATAL: vLLM port 8001 unreachable"; exit 1
}
aws sts get-caller-identity > /dev/null 2>&1 || {
  echo "FATAL: aws cli not configured for Bedrock"; exit 1
}

LOG=/tmp/pm_sweep_v4.log
RUN_LABEL="run_$(date +%Y-%m-%d)_pm_v4"
ARCHIVE="data/firehose_eval/runs/${RUN_LABEL}"
mkdir -p "${ARCHIVE}/passes"

.venv/bin/python -u scripts/run_firehose_loop.py \
    --path all-9 \
    --universe dow30 \
    --apex-pm \
    --apex-via-vllm \
    --vllm-base http://localhost:8001/v1 \
    --max-tokens 4096 \
    --use-interrogator \
    --use-strategy-committee \
    --use-tax-aware-validators \
    --philosophy-weights tasks_v4/philosophy_weights.yaml \
    --debate-predators 4 \
    --run-label "${RUN_LABEL}" \
    2>&1 | tee "$LOG"

cp "$LOG" "${ARCHIVE}/sweep.log"
LATEST_EVAL=$(ls -t data/firehose_eval/eval_*.jsonl | head -1)
cp "$LATEST_EVAL" "${ARCHIVE}/eval.jsonl"
if [[ -f data/firehose_eval/debate_transcripts/latest.jsonl ]]; then
    cp data/firehose_eval/debate_transcripts/latest.jsonl "${ARCHIVE}/debate_transcripts.jsonl"
fi

echo "v4 sweep archived to ${ARCHIVE}"
```

---

## Acceptance Criteria

- [ ] `--path all-9` enumerates the 9 v4 paths (not v3.3's 9-with-ORACLE-SONNET).
- [ ] ECO-DEBATE-QWEN path runs end-to-end on a 1-day mini-test.
- [ ] InterrogatorHerbivore is invoked on every ECO-* day; an interrogator synthesis broadcast reaches the apex.
- [ ] ORACLE-SONNET is dropped from `all-9` default (still accessible via explicit `--path oracle-sonnet`).
- [ ] Dow 30 universe is the default for `--universe`.
- [ ] `scripts/v4_run_sweep.sh` exists and is executable.
- [ ] Existing tests stay green.

---

## Testing Conditions

### 1. Path enumeration

```bash
.venv/bin/python -c "
from scripts.run_firehose_loop import ALL_PATHS_V4
assert 'eco-debate-qwen' in ALL_PATHS_V4, 'debate path missing'
assert 'oracle-sonnet' not in ALL_PATHS_V4, 'oracle-sonnet should be dropped from v4 default'
assert len(ALL_PATHS_V4) == 9, f'expected 9 paths, got {len(ALL_PATHS_V4)}'
print('paths OK:', ALL_PATHS_V4)
"
```
**Expected**: paths OK with eco-debate-qwen present, oracle-sonnet absent, total = 9.

### 2. Interrogator wiring (offline check)

```bash
grep -E "InterrogatorHerbivore\|interrogator\\.predict_only" scripts/run_firehose_loop.py | wc -l
```
**Expected**: `>= 2`.

### 3. Debate path 1-day smoke (uses vLLM)

```bash
.venv/bin/python -u scripts/run_firehose_loop.py \
    --path eco-debate-qwen \
    --universe dow30 \
    --apex-pm --apex-via-vllm \
    --vllm-base http://localhost:8001/v1 \
    --max-tokens 4096 \
    --max-days 1 \
    --use-interrogator \
    --output /tmp/v4_debate_smoke.log
```
**Expected**: exit 0; log contains 1 `PM:` line with predator_breakdown showing 4 predators.

### 4. Dow 30 default verified

```bash
.venv/bin/python -c "
import subprocess, sys
r = subprocess.run(['.venv/bin/python', 'scripts/run_firehose_loop.py', '--help'],
                   capture_output=True, text=True)
assert '--universe' in r.stdout
assert 'dow30' in r.stdout
print('universe flag OK')
"
```
**Expected**: `universe flag OK`.

### 5. No regressions

```bash
.venv/bin/python -m pytest tests/ -x -q
```
**Expected**: green.

---

## Coordination

- **05f (smoke gate)** runs after this. Do not attempt the full sweep in
  this mission; only the 1-day debate path smoke.
- The InterrogatorHerbivore's exact constructor signature may need
  adapting — its training-time use takes `agent_id`, a Channel, etc.
  Wrap with sensible PM-mode defaults.
- If you discover the interrogator's `MathHost` requires a separate vLLM
  endpoint (Qwen2.5-Math-1.5B), confirm with @all in MESSAGES before
  proceeding; we may need to spin up a second vLLM on a different port
  before the sweep.
- The debate path's per-day LM call count is ~4 predators × 3 rounds =
  12 calls/day. With 66 days × 12 calls + 66 days × 8 other paths × 1
  call = ~1320 LM calls total per sweep. At ~5s/call on Qwen-4B vLLM,
  that's ~110 min. Bedrock paths fit within their existing rate limits.

---

## When Done

```bash
grep -E "@all|@phase3" tasks_v4/scratchpad.md | tail -30
sed -i 's/phase3-A-05e:RUNNING/phase3-A-05e:DONE/' tasks_v4/scratchpad.md
sed -i 's/phase3-A-05f:BLOCKED.*$/phase3-A-05f:PENDING/' tasks_v4/scratchpad.md

cat >> tasks_v4/scratchpad.md <<'EOF'

[YYYY-MM-DD HH:MM] phase3-A-05e > @phase3-A-05f,05g: runner wired. 9 v4 paths enumerated (ORACLE-SONNET dropped). InterrogatorHerbivore plumbed into ECO-* paths. ECO-DEBATE-QWEN path with 4-predator orchestration ready. scripts/v4_run_sweep.sh launch wrapper exists. 05f can now smoke-test the new path.
EOF

mv tasks_v4/phase3-A-05e-runner-wiring.md tasks_v4/completed/
```
