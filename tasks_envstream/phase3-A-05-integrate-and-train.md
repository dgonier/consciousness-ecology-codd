# Mission 05: integrate-and-train

**Handle**: `phase3-A`
**Phase**: 3 (sequential, runs alone after all of phase 2)
**Mission file**: `phase3-A-05-integrate-and-train.md`
**Dependencies**: `phase1-A:01:DONE`, `phase2-B:02:DONE`, `phase2-C:03:DONE`, `phase2-D:04:DONE`
**Blocks**: nothing (final integration)

---

## Before You Start

```bash
cd /home/dgonier/ecology_experiment/trophic && \
  grep -nE '@(all|phase3-A|phase3)' tasks_envstream/scratchpad.md
```

Confirm ALL phase 1 + phase 2 STATUS lines are DONE. Then atomically flip:
```
phase3-A:05:PENDING → phase3-A:05:RUNNING
```

---

## Goal

Three things, in order:

1. **Wire EnvironmentStream into `runner.py`** so all adapter outputs deposit into the stream and producers attend over wavelength-filtered subsets.
2. **Refactor `stocknet_loader.py`** to route through the new adapters instead of inline parsing. Remove the phase1-A TODO markers.
3. **Train `sft_seed9` + `ipo_seed9`** with the new architecture live, then run the full eval cascade (dev + held-out + StockNet smoke). Decision criterion: StockNet test MCC moves off zero.

This is the validation gate. If MCC stays at zero, the architecture pivot didn't help and we revisit (issue #8 stays open). If MCC moves, the EnvironmentStream rebuild was the right call.

---

## Files to Modify
- `trophic/runner.py` — instantiate EnvironmentStream; route adapter outputs in; producer attend uses wavelength filter
- `trophic/training/stocknet_loader.py` — replace inline parsing with adapter calls; remove TODO markers
- `trophic/training/scenarios.py` — replace any inline format-conversion remaining after phase1-A's TODO markers

## Files to Create
- `scripts/diagnostics/diag_envstream_smoke.py` — small end-to-end smoke that deposits a few RawInputs through adapters → EnvStream → producers and prints the resulting broadcasts
- (training artifacts will appear under `checkpoints/` and `logs/` during the run)

---

## Implementation Steps

### Step 1: Read everything phase 2 produced
- `trophic/environment_stream.py` from phase2-B
- `trophic/adapters/stocknet/*.py` from phase2-C
- `trophic/agents/social_signal.py` + the producer.py refactor from phase2-D
- All MESSAGES posted by phase2 agents — they may have flagged design choices that affect integration

### Step 2: Wire EnvironmentStream into `runner.py`

Find the place where producers currently receive `RawInput` directly from scenarios. Insert EnvironmentStream as the intermediary:

```python
# Before:
for inp in sc.inputs:
    for prod in producers:
        if prod.attracts(inp):
            br = await prod.produce(inp, tick=tick, host=host)
            ...

# After:
env_stream = EnvironmentStream(hidden_size=host.hidden_size, n_slots=64, ...)
for inp in sc.inputs:
    # Embed once via Qwen (or whatever adapter→embedding policy you choose);
    # phase2-B's deposit() takes (RawInput, embedding).
    emb = host.text_to_hidden(_render_minimal(inp), pool="mean")
    env_stream.deposit(inp, emb)

# Producers attend with their wavelength filter:
for prod in producers:
    out = env_stream.attend(
        query=prod.role_q(),
        wavelength_filter=prod.WAVELENGTHS,
        tau=tau,
    )
    # producer's `produce` becomes a thin wrapper around the attend output:
    br = prod.produce_from_stream(out, tick=tick, host=host)
```

The exact API depends on phase2-B's choices. If `produce_from_stream` doesn't exist, you may need to add it on `Producer` — that's an integration change. Coordinate via MESSAGES if scope expands.

### Step 3: Refactor `stocknet_loader.py`

The existing loader does HTTP fetch + cache + inline parsing. Replace the parsing with adapter calls:

```python
# old: inline tab-split + float() loop
# new:
from trophic.adapters.stocknet import OhlcvNormalizedAdapter, TokenizedTweetAdapter

def _load_ohlcv(ticker: str) -> dict[str, dict]:
    text = _http_get(price_url, cache_dir)
    if text is None: raise FileNotFoundError(...)
    rows = {}
    for ri in OhlcvNormalizedAdapter(ticker).adapt(text):
        rows[ri.payload["date"]] = ri.payload
    return rows
```

HTTP fetch + caching stay in the loader (they're not adapter responsibilities). Only the format-conversion logic moves.

### Step 4: Smoke test

Write `scripts/diagnostics/diag_envstream_smoke.py` that:
1. Creates an EnvironmentStream
2. Uses adapters to load 5 days of AAPL OHLCV + tweets from cache
3. Deposits all into the stream
4. Has each producer (TickDelta, Disclosure, Anomaly, QuantitativeProducer, SocialSignal) attend
5. Prints which slots each producer attended to most strongly

This validates the cortical-column fan-out — TickDelta should pull mostly OHLCV slots; SocialSignal should pull mostly tweets slots; QuantitativeProducer should pull none (no quote_series in the deposit).

### Step 5: Mock SFT smoke

```bash
TROPHIC_MOCK_MODELS=1 TROPHIC_MOCK_PREDATOR=1 TROPHIC_STEPS=10 \
  .venv/bin/python scripts/train_sft.py 2>&1 | tail -20
```
Confirms EnvStream wires don't break the synthetic-scenario training path.

### Step 6: Real training — `sft_seed9`

```bash
TROPHIC_SEED=9 TROPHIC_STEPS=600 TROPHIC_LR=5e-4 \
  TROPHIC_LOG_EVERY=10 TROPHIC_EVAL_EVERY=50 \
  nohup .venv/bin/python -u scripts/train_sft.py \
    > logs/sft_seed9_envstream.log 2>&1 &
echo $! > logs/sft_seed9_envstream.pid
```
~85 min on the 4090. Watch eval-loss curve. Compare to seed 8's pre-fix baseline (best 0.192).

### Step 7: Real training — `ipo_seed9`

After SFT exits:
```bash
TROPHIC_SEED=9 TROPHIC_IPO_STEPS=800 TROPHIC_IPO_LR=1e-4 TROPHIC_IPO_BETA=0.1 \
  TROPHIC_LOG_EVERY=25 TROPHIC_EVAL_EVERY=100 \
  nohup .venv/bin/python -u scripts/train_ipo.py \
    > logs/ipo_seed9_envstream.log 2>&1 &
echo $! > logs/ipo_seed9_envstream.pid
```
Apply same early-stop policy: 2 consecutive non-improvements → kill.

### Step 8: Eval cascade

After IPO finishes:
```bash
# Dev
TROPHIC_CKPT=checkpoints/ipo_seed9_best.pt TROPHIC_SEED=9 \
  .venv/bin/python -u scripts/eval_xml_checkpoint.py > logs/eval_dev_ipo_seed9.log 2>&1

# Held-out
TROPHIC_CKPT=checkpoints/ipo_seed9_best.pt TROPHIC_LABEL=ipo_seed9 TROPHIC_SEED=9 \
  .venv/bin/python -u scripts/diagnostics/eval_holdout_general.py > logs/eval_holdout_ipo_seed9.log 2>&1

# StockNet
TROPHIC_CKPT=checkpoints/ipo_seed9_best.pt TROPHIC_SEED=9 STOCKNET_MAX_PER_TICKER=10 \
  .venv/bin/python -u scripts/eval_stocknet.py > logs/stocknet_smoke_ipo_seed9.log 2>&1
```

### Step 9: Update issues #8 and #9

- **#9**: status:done if everything trained and eval'd cleanly. Comment with file paths and final test counts.
- **#8**: comment with the StockNet MCC result. If MCC > 0.05, status:done with verdict "EnvironmentStream rebuild solved direction collapse." If MCC ≤ 0.05, leave open with verdict "EnvironmentStream landed cleanly but did not move MCC; investigation continues."

### Step 10: Update CHANGELOG.md and EXPERIMENTS.md

Append the new run's row to EXPERIMENTS.md. Mark EnvironmentStream as Added in the Unreleased section of CHANGELOG.md.

---

## Acceptance Criteria

- [ ] `runner.py` instantiates EnvironmentStream and routes adapter outputs through it
- [ ] `stocknet_loader.py` uses adapters; no inline parsing remains; phase1-A TODO markers removed
- [ ] `scripts/diagnostics/diag_envstream_smoke.py` exists and prints per-producer attention distributions
- [ ] Mock SFT smoke completes (exit 0)
- [ ] Real SFT seed 9 trains to step 600 (or early-stops naturally) without NaN/OOM
- [ ] Real IPO seed 9 trains and produces `ipo_seed9_best.pt`
- [ ] Eval cascade completes (dev + held-out + StockNet)
- [ ] All test counts maintained (65+ tests passing)
- [ ] Decision: StockNet test MCC reported. If > 0.05, fix #8 closes; if ≤ 0.05, follow-up needed

---

## Testing Conditions (exit verification)

1. **Whole test suite green**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     .venv/bin/python -m pytest tests/ --tb=line -q 2>&1 | tail -5
   ```
   **Expected**: 65+ passed, 0 failed.

2. **Mock SFT smoke**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     TROPHIC_MOCK_MODELS=1 TROPHIC_MOCK_PREDATOR=1 TROPHIC_STEPS=10 \
     timeout 300 .venv/bin/python scripts/train_sft.py 2>&1 | tail -10
   ```
   **Expected**: exit 0, no exceptions.

3. **Smoke diag prints sensible per-producer attention**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     .venv/bin/python scripts/diagnostics/diag_envstream_smoke.py 2>&1 | tail -30
   ```
   **Expected**: TickDelta highest weight on `ohlcv` slots; SocialSignal highest on `tweets` slots. Specialist producers should NOT have uniform attention across slot types.

4. **Real SFT exits with checkpoint written**
   After `sft_seed9` finishes:
   ```bash
   ls -la checkpoints/sft_seed9_best.pt
   grep "best eval" logs/sft_seed9_envstream.log | tail -3
   ```
   **Expected**: best eval-loss line printed; file exists at ~2.2GB.

5. **Real IPO exits with checkpoint written**
   ```bash
   ls -la checkpoints/ipo_seed9_best.pt
   grep "ckpt\|EVAL_MEAN_SCORE" logs/ipo_seed9_envstream.log | tail -8
   ```
   **Expected**: best ckpt at some step.

6. **Eval cascade headlines captured**
   ```bash
   grep "MEAN PREDATOR REWARD" logs/eval_dev_ipo_seed9.log
   grep "MEAN" logs/eval_holdout_ipo_seed9.log
   grep -E "ACCURACY|MCC|TP=" logs/stocknet_smoke_ipo_seed9.log
   ```
   **Expected**: three numbers. The StockNet MCC is the decisive one.

If any test fails, post `@all` MESSAGES; do not flip DONE.

---

## Coordination

- This is the final mission. Verify all phase-1 and phase-2 missions are DONE before starting.
- Apply checkpoint hygiene per the saved policy: keep best + previous-best, delete `_final.pt` after eval.
- Disk discipline: each ckpt is ~2.2GB; the `sft_seed9_*` and `ipo_seed9_*` artifacts will add ~9GB. Confirm `df -h /` shows enough headroom before launching real training.

---

## When Done

1. Re-run inbox grep.
2. Update STATUS:
   - `phase3-A:05:RUNNING` → `phase3-A:05:DONE`
3. Update `00-README.md` Mission Status table — mark all five rows as `COMPLETE` with handle + date.
4. Update CHANGELOG.md and EXPERIMENTS.md.
5. Update issues #8 and #9 per Step 9 above.
6. Append MESSAGES with the final headline numbers:
   ```
   - [<YYYY-MM-DD HH:MM>] phase3-A > @all: ENVSTREAM project complete.
       Final SFT seed 9 eval-loss: <V>
       Final IPO seed 9 best judge-eval: <V>
       Dev rule-based reward: <V>
       Held-out rule-based reward: <V>
       StockNet smoke: ACCURACY=<V> MCC=<V>
       Verdict: <fix worked / fix landed but MCC unchanged / fix introduced regression>.
       Ready for git review.
   ```
