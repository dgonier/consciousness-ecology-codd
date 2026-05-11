# Mission 2.5: smoke-gate (contract verification + 3-day mini-sweep)

**Handle**: phase2_5 (any free agent, sequential after all of phase 2)
**Phase**: 2.5 (gate between phase 2 and phase 3)
**Mission file**: phase2_5-smoke-gate.md
**Dependencies**: phase2-B:02, phase2-C:03, phase2-D:04
**Blocks**: phase3-A:05

---

## Before You Start

```bash
grep -E "@all|@phase2|@phase2_5" tasks_v4/scratchpad.md | tail -50
```

```bash
sed -i 's/phase2_5:smoke:BLOCKED.*$/phase2_5:smoke:RUNNING/' tasks_v4/scratchpad.md
```

This mission exists because phase 3 launches a 66-day, 9-portfolio GPU
sweep. The cost of phase 3 failing on day 2 because (say) the committee
output doesn't deserialize through `_order_to_dict` is high. Smoke gate
catches that in 3 minutes instead.

---

## Goal

Two-part gate:

**Part A — contract verification (no GPU, no apex calls)**: a single
deterministic script constructs synthetic apex outputs, runs them through
the committee → validators → portfolio path, and verifies the entire
pipeline returns sensible state without exception.

**Part B — 3-day mini-sweep (uses local vLLM, no Bedrock)**: run the
firehose loop for **3 trading days**, **ECO-QWEN only**, with all phase-2
features enabled. Watch order count and rejection rate.

Pass criteria are explicit numeric thresholds. Fail criteria halt the
gate.

---

## Files to Create / Modify

**Create:**
- `scripts/v4_smoke_part_a.py` — the deterministic contract verifier.
- `scripts/v4_smoke_part_b.sh` — wrapper that invokes `run_firehose_loop.py`
  with the right flags for a 3-day ECO-QWEN smoke.
- `tasks_v4/SMOKE_RESULTS.md` — the report you write after running.

**No modifications** to phase 1 / phase 2 files. This gate observes; it
doesn't change.

---

## Implementation Steps

1. **Part A: contract verifier.**
   `scripts/v4_smoke_part_a.py` must:
   - Construct one synthetic `TickerView` for each horizon (h1/h5/h20/h60).
   - For each TickerView, construct a matching `Order` with realistic
     fields.
   - Run each (TickerView, Order) through:
     - `aggregate_votes` of all 3 strategies → produces an aggregated Order
       (or None for split cases).
     - Each of the 4 phase2-D validators → check accept/reject as expected.
   - Try a few **adversarial** cases:
     - Order with `primary_horizon='h20'` but `size_pct=50` → should
       reject on HorizonSizingMatch.
     - Order with `side='BUY'` but `forecasts.h5 = -0.02` → reject on
       ForecastConsistency.
     - Order with `primary_horizon='h5'` and `expected_alpha_bps=50` (below
       the 162 floor) → reject on edge floor.
     - Same expected_alpha=50 but `primary_horizon='h60'` → accept (waived).
   - Print a single-line summary at the end: `SMOKE_PART_A: PASS` or
     `SMOKE_PART_A: FAIL <reason>`.

   The script exits 0 on PASS, 1 on FAIL.

2. **Part B: 3-day mini-sweep.**
   `scripts/v4_smoke_part_b.sh` runs:
   ```bash
   .venv/bin/python -u scripts/run_firehose_loop.py \
       --path ecology-qwen \
       --apex-pm \
       --apex-via-vllm \
       --vllm-base http://localhost:8001/v1 \
       --max-tokens 4096 \
       --max-days 3 \
       --no-lm-cache \
       --output /tmp/v4_smoke_b.log
   ```
   (Adapt flags to whatever the runner actually exposes after phase 2-B
   modifications. The `--max-days 3` flag may need adding if it doesn't
   already exist; you can hardcode a 3-day slice via existing date filters.)

3. **Part B pass criteria:**
   - Run completes without an unhandled exception.
   - **Order count over 3 days**: ≤ 12 orders total (4/day expected ceiling;
     v3.3 ECO-QWEN averaged ~4.5/day, so v4 should hold roughly flat or
     drop given the new edge floor).
   - **Rejection rate**: < 50% of attempted orders rejected. (v3.3 was
     5-15%; v4 will be higher because of new validators, but >50% means
     the model can't comply with the new constraints.)
   - **At least one Order with `primary_horizon` ∈ {h20, h60}** — proves
     the multi-horizon machinery is firing, not just emitting h1 for
     everything.

4. **Failure modes (halt + flag):**
   - **`SMOKE_PART_A: FAIL`** → phase 2 contracts are broken. Post @all in
     MESSAGES with the failure reason and STOP. Phase 3 cannot start.
   - **Part B exception** → same as above.
   - **Order count > 12 in 3 days** → the apex is over-trading despite the
     new prompt. Post in MESSAGES to @phase2-B (prompt issue) and @phase2-D
     (validator issue) but you may continue to phase 3 with a flag — this
     is a soft-fail, not a hard-fail.
   - **Rejection rate > 50%** → hard-fail. Post @all, halt.
   - **No h20/h60 orders in 3 days** → soft-fail. Post @phase2-B (oracle
     prompt) and @phase2-C (strategy committee). You may continue if the
     other criteria pass; v3.3-grade behavior is the floor.

5. **Write `SMOKE_RESULTS.md`** with:
   - PART A result + any failures.
   - PART B order count, rejection rate, horizon distribution, sample of
     5 orders (ticker, side, size_pct, primary_horizon, expected_alpha_bps).
   - Pass/fail verdict overall.
   - Any soft-fail items flagged for phase 3.

---

## Acceptance Criteria

- [ ] `scripts/v4_smoke_part_a.py` exists, runs without crashing, exits 0
      on the contract path.
- [ ] `scripts/v4_smoke_part_b.sh` exists and is executable.
- [ ] `tasks_v4/SMOKE_RESULTS.md` exists with both parts' results.
- [ ] Part A passes (or its failure is documented and posted in MESSAGES).
- [ ] Part B completes (3 days run end-to-end; soft-fails are documented).

---

## Testing Conditions

### 1. Part A runs and exits 0

```bash
cd /home/dgonier/ecology_experiment/trophic && .venv/bin/python scripts/v4_smoke_part_a.py
echo "exit code: $?"
```
**Expected**: prints `SMOKE_PART_A: PASS` and exits 0.

### 2. Part B runs to completion

```bash
cd /home/dgonier/ecology_experiment/trophic && bash scripts/v4_smoke_part_b.sh
```
**Expected**: 3-day sweep completes; final summary line printed; log at `/tmp/v4_smoke_b.log` has 3 `[N/3]` day markers and 3 `PM:` portfolio reports.

### 3. Report exists and has both sections

```bash
test -f tasks_v4/SMOKE_RESULTS.md && head -50 tasks_v4/SMOKE_RESULTS.md
```
**Expected**: file exists with PART A and PART B sections, verdict line at top.

---

## Coordination

- **You are the gate.** If you mark this DONE, phase 3 launches the full
  sweep. Be honest in the report.
- If any phase-2 mission needs revision based on smoke results, post in
  MESSAGES and set the relevant `phase2-*:DONE` line back to `RUNNING`
  with a note. Then re-run this gate after they fix.

---

## When Done

```bash
grep -E "@all|@phase2|@phase2_5|@phase3" tasks_v4/scratchpad.md | tail -30
sed -i 's/phase2_5:smoke:RUNNING/phase2_5:smoke:DONE/' tasks_v4/scratchpad.md
sed -i 's/phase3-A:05:BLOCKED.*$/phase3-A:05:PENDING/' tasks_v4/scratchpad.md

cat >> tasks_v4/scratchpad.md <<'EOF'

[YYYY-MM-DD HH:MM] phase2_5 > @phase3-A: smoke gate complete. See tasks_v4/SMOKE_RESULTS.md. Phase 3 unblocked. <Add 1 sentence summary of soft-fails or open concerns here.>
EOF

git mv tasks_v4/phase2_5-smoke-gate.md tasks_v4/completed/
```
