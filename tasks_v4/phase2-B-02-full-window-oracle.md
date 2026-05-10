# Mission 02: full-window-oracle

**Handle**: phase2-B
**Phase**: 2 (parallel, after phase1-A:01)
**Mission file**: phase2-B-02-full-window-oracle.md
**Dependencies**: phase1-A:01
**Blocks**: phase2_5:smoke, phase3-A:05

---

## Before You Start

```bash
grep -E "@all|@phase2|@phase2-B" tasks_v4/scratchpad.md | tail -30
```

```bash
sed -i 's/phase2-B:02:PENDING/phase2-B:02:RUNNING/' tasks_v4/scratchpad.md
```

Confirm `HorizonForecast`, `TickerView`, and extended `Order` are present:
```bash
.venv/bin/python -c "from trophic.beliefs.apex_signatures import HorizonForecast, TickerView, Order; print('phase1-A contract present')"
```
**Expected**: `phase1-A contract present`.

If this fails, do not proceed — phase1-A:01 hasn't actually landed despite STATUS claim. Post @all in MESSAGES and stop.

---

## Goal

Replace the v3.3 next-day Oracle with a **full-window Oracle** that sees
the entire 66-day forward-return matrix at the start of the window and is
prompted to commit to a buy-low-sell-high trajectory with **infrequent
trades**.

This is the v3.3 ORACLE postmortem in code form. v3.3's Oracle saw
1-day-forward returns and reacted every bar → 384–721 trades, tax ate all
alpha. The fix is structural: change what the apex sees, change the prompt,
and reward commitment to a trajectory over reactive trading.

---

## Files to Create / Modify

**Modify:**
- `trophic/beliefs/apex_signatures.py` — replace `WatchlistOracle` with `WatchlistOracleFullWindow`. Old signature stays for back-compat but is marked deprecated.
- `scripts/run_firehose_loop.py` — the oracle pipeline must build `future_returns_matrix` from the 66-day price slice and pass it to the new signature.
- `trophic/agents/apex_portfolio.py` — verify state surfaces don't need oracle-specific changes (likely none needed since the apex sees the matrix in the *prompt*, not portfolio state).

**Create:**
- `tests/test_full_window_oracle_signature.py` — see Testing Conditions.

---

## Implementation Steps

1. **Define `WatchlistOracleFullWindow` signature.**
   It must accept these inputs:
   - `today: str` (ISO date)
   - `days_remaining: int` (how many days left in the window — important for the apex's trajectory-planning math)
   - `portfolio_state: PortfolioState`
   - `positions: list[PositionSnapshot]`
   - `watchlist: list[str]` (tickers)
   - `future_returns_matrix: dict[str, list[float]]` — per ticker, the daily forward returns from `today` to `today + days_remaining`. Indexed in trading-day order.

2. **Oracle prompt rewrite.** The new prompt must convey, explicitly:
   - You have foresight of every ticker's daily return for the rest of the window.
   - Your job is NOT to react to each day. Your job is to commit now to a trajectory.
   - **Strategy directive**: prefer infrequent, larger, longer-held trades. Buy near local minima, sell near local maxima.
   - **Tax explanation block** (inherited from phase1-A:01 must be present).
   - Each `Order` you emit must commit to `primary_horizon`. Use `h60` for buy-and-hold of names trending up the whole window; `h20` for swing trades inside the window; `h5` for very short-term mispricings (rare); `h1` should be a last resort.
   - Explicit anti-pattern: "If you see a temporary dip in a name that recovers within 5 days, the correct action is buy-and-hold-through-dip, NOT sell-on-dip + rebuy."
   - One concrete worked example in the prompt: "Ticker X has forward returns of [-0.5%, -0.3%, +0.4%, +1.1%, +2.0%, ...]. The dip is 0.8% over 2 days; the recovery is +3.5% over the next 3 days. Net: +2.7%. Correct action: BUY today with primary_horizon=h5, hold through the dip. Wrong action: BUY-then-SELL-on-day-2-then-REBUY."
   - End with: "Output the minimum number of orders that achieves your trajectory. Conservatively, expect <= 0.5 orders per day on average over the window."

3. **Build `future_returns_matrix` in the runner.**
   In `scripts/run_firehose_loop.py`, the oracle dispatch path must:
   - Slice the price dataframe from `today` to window-end.
   - Compute daily returns (`close[t+1] / close[t] - 1`).
   - Pass per-ticker arrays to the signature input.
   - Cap matrix length to `days_remaining` to avoid leaking beyond the window.

4. **Token-budget the matrix.**
   66 days × 20 tickers × 7-character-floats ≈ 9K tokens just for numbers.
   That's fine for 16K context but watch for blow-up if watchlist grows.
   Truncate to 2 decimal places (`f"{r:.4f}"`) and keep as JSON array per ticker. If total > 12K tokens, prune to the top-N watchlist names by volatility and add `pruned_tickers: list[str]` to the prompt.

5. **Wire the new signature into the oracle paths only.**
   v3.3 had `ORACLE-QWEN`, `ORACLE-SONNET`, `ORACLE-OPUS`. All three should
   now use `WatchlistOracleFullWindow`. BARE and ECO paths still use
   `WatchlistFromObservationsPM` / `WatchlistFromFirehosePM`.

6. **Deprecate the old oracle path** — leave the code in place but tag with
   `# DEPRECATED in v4 — replaced by WatchlistOracleFullWindow` and remove
   the `--path oracle*` CLI flags that targeted it. The runner's path
   selection logic should reject `oracle_qwen_v3` style strings.

---

## Acceptance Criteria

- [ ] `WatchlistOracleFullWindow` exists and accepts the full-window matrix.
- [ ] Prompt contains all the elements in step 2 — strategy directive, tax block, worked example, anti-pattern statement, "minimum number of orders" instruction.
- [ ] Runner builds `future_returns_matrix` correctly (verify with a deterministic test).
- [ ] Oracle output schema: `views: list[TickerView]` + `orders: list[Order]`, all orders have `primary_horizon` and `expected_alpha_bps`.
- [ ] Existing v3.3 Oracle code is marked deprecated, not deleted.
- [ ] Phase1-A signatures untouched.

---

## Testing Conditions (exit verification)

### 1. Signature builds and accepts a synthetic full-window matrix

```bash
.venv/bin/python -c "
from trophic.beliefs.apex_signatures import WatchlistOracleFullWindow, PortfolioState
import dspy
# Construct the signature class and inspect its input fields
sig = WatchlistOracleFullWindow
inputs = sig.signature.input_fields if hasattr(sig, 'signature') else sig.input_fields
names = list(inputs.keys()) if isinstance(inputs, dict) else [f.name for f in inputs]
assert 'future_returns_matrix' in names, f'missing future_returns_matrix; have {names}'
assert 'days_remaining' in names, f'missing days_remaining; have {names}'
print('full-window oracle signature has the expected inputs')
"
```
**Expected**: `full-window oracle signature has the expected inputs`.

### 2. Future-returns matrix builder is correct

Create or extend `tests/test_full_window_oracle_signature.py` with a deterministic
test: feed a known synthetic price series, verify the resulting matrix matches
hand-computed returns. Then run:

```bash
.venv/bin/python -m pytest tests/test_full_window_oracle_signature.py -x -v
```
**Expected**: all tests pass.

### 3. Prompt contains required strategy directives

```bash
.venv/bin/python -c "
from trophic.beliefs.apex_signatures import WatchlistOracleFullWindow
text = (WatchlistOracleFullWindow.__doc__ or '')
needed = ['buy near local minima', 'sell near local maxima', '0.5 orders per day', 'do not sell-on-dip', '37%', 'h60', 'foresight']
missing = [n for n in needed if n.lower() not in text.lower()]
assert not missing, f'oracle prompt missing required directives: {missing}'
print('oracle prompt contains all required directives')
"
```
**Expected**: `oracle prompt contains all required directives`.

### 4. Runner can dispatch the new oracle signature end-to-end (offline, no GPU)

```bash
cd /home/dgonier/ecology_experiment/trophic && .venv/bin/python -c "
# Minimal smoke: import the runner module and check that the dispatch table maps the oracle paths to the new signature.
import importlib
m = importlib.import_module('scripts.run_firehose_loop')
src = open('scripts/run_firehose_loop.py').read()
assert 'WatchlistOracleFullWindow' in src, 'runner does not reference the new signature'
print('runner wires WatchlistOracleFullWindow')
"
```
**Expected**: `runner wires WatchlistOracleFullWindow`.

### 5. No regressions

```bash
.venv/bin/python -m pytest tests/ -x -q
```
**Expected**: green.

---

## Coordination

- **Parallel with phase2-C and phase2-D**. You all consume the phase 1
  contract; you don't touch each other's files. If you find yourself
  editing `validators.py` or `strategy_committee.py`, stop — that's phase
  2-D/C territory.
- The `future_returns_matrix` is a **leak vector**. The matrix must be
  capped at `days_remaining` and never extend past window-end. Add an
  assertion in the runner.
- If you discover the apex consistently ignores the buy-low-sell-high
  directive even with the new prompt (e.g., still produces 100+ orders on a
  3-day smoke), post in MESSAGES to @phase3-A — they need to know before
  the full sweep.

---

## When Done

```bash
grep -E "@all|@phase2|@phase2-B" tasks_v4/scratchpad.md | tail -30
sed -i 's/phase2-B:02:RUNNING/phase2-B:02:DONE/' tasks_v4/scratchpad.md

cat >> tasks_v4/scratchpad.md <<'EOF'

[YYYY-MM-DD HH:MM] phase2-B > @phase2_5,phase3-A: full-window oracle landed. `WatchlistOracleFullWindow` accepts a per-ticker forward-returns matrix, and the prompt enforces buy-low-sell-high + h20/h60 preference + tax framing. Old per-day oracle is deprecated. If you see >0.5 orders/day on a smoke sweep, the prompt directives aren't sticking — flag in MESSAGES.
EOF

git mv tasks_v4/phase2-B-02-full-window-oracle.md tasks_v4/completed/
```
