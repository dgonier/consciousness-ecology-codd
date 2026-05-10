# Mission 01: multi-horizon-signatures

**Handle**: phase1-A
**Phase**: 1 (sequential, foundation)
**Mission file**: phase1-A-01-multi-horizon-signatures.md
**Dependencies**: none
**Blocks**: phase2-B:02, phase2-C:03, phase2-D:04

---

## Before You Start

```bash
# Inbox grep — your handle is phase1-A
grep -E "@all|@phase1|@phase1-A" tasks_v4/scratchpad.md | tail -30
```

Set your STATUS line:
```bash
sed -i 's/phase1-A:01:PENDING/phase1-A:01:RUNNING/' tasks_v4/scratchpad.md
```

---

## Goal

Add the multi-horizon forecast types, extend the Order schema, and inject
the tax-explanation block into the apex prompt. After this mission, every
downstream phase has the types and contract it needs.

This is type plumbing + prompt surgery. No new models, no new dependencies.

---

## Files to Create / Modify

**Modify:**
- `trophic/beliefs/apex_signatures.py` — add `HorizonForecast`, `TickerView`; extend `Order` with `primary_horizon` and `expected_alpha_bps`; add tax-framing block to apex prompt strings.
- `trophic/agents/apex_portfolio.py` — `state_for_apex` now surfaces `recent_realized_pnl_5d`, `tax_owed_accrued`, and a new `profit_definition` string field on `PortfolioState` explaining what counts as net profit.
- `trophic/beliefs/portfolio_validators.py` (or wherever current Order validation lives) — add minimal field validation that `primary_horizon` is one of {h1, h5, h20, h60} and `expected_alpha_bps` ∈ [-10000, 10000]. Do NOT add the tax-aware validators yet — that's phase2-D.

**Create:**
- `tests/test_multi_horizon_signatures.py` — see Testing Conditions below.

---

## Implementation Steps

1. **Add the new Pydantic models to `apex_signatures.py`.**
   Insert after the existing `Order` class definition:
   ```python
   class HorizonForecast(BaseModel):
       h1:  float = Field(..., description="Expected return (decimal, not bps) over 1 trading day.")
       h5:  float = Field(..., description="Expected return over 5 trading days.")
       h20: float = Field(..., description="Expected return over 20 trading days.")
       h60: float = Field(..., description="Expected return over ~3 months (window-end).")
       confidence_h1:  Literal["low", "med", "high"]
       confidence_h5:  Literal["low", "med", "high"]
       confidence_h20: Literal["low", "med", "high"]
       confidence_h60: Literal["low", "med", "high"]
       regime_note: str = Field(..., min_length=1, description="One-sentence regime label: trend|mean_revert|breakout|range|event_driven|unclear.")

   class TickerView(BaseModel):
       ticker: str = Field(..., min_length=1, max_length=10)
       forecasts: HorizonForecast
       primary_horizon: Literal["h1", "h5", "h20", "h60"]
       rationale: str = Field(..., min_length=1, description="Why primary_horizon was chosen and what edge is expected.")
   ```

2. **Extend `Order`.**
   Add two required fields:
   ```python
   primary_horizon: Literal["h1", "h5", "h20", "h60"]
   expected_alpha_bps: float = Field(..., ge=-10000, le=10000, description="Expected return in basis points over the committed horizon, after slippage. Negative for shorts/sells.")
   ```
   Do not change existing field types. Keep `populate_by_name = True`.

3. **Update each watchlist signature** (`WatchlistFromFirehose`, `WatchlistFromObservations`, `WatchlistFromFirehosePM`, `WatchlistFromObservationsPM`, `WatchlistOracle`) so the apex output now includes:
   - `views: list[TickerView]` — one per watchlist ticker the apex has an opinion on
   - `orders: list[Order]` — same as before but with the new required fields
   - The `views` and `orders` lists must reference the same tickers; an Order must point to a ticker that has a TickerView in the same response. Add a `validator` (Pydantic root_validator / model_validator) enforcing this consistency.

4. **Tax explanation block in the apex prompt.**
   Add a new prompt section to all PM signatures' `docstring` (or wherever the system prompt is composed). It must include:
   - Short-term cap gains: 37% on realized gains.
   - Slippage: 5bps per trade leg (10bps round-trip).
   - **Net profit definition**: `realized P&L − tax_owed − slippage + cash_yield`.
   - **What this means for strategy**: a trade needs ~162bps of expected edge to clear costs on a typical 1-day flip. A trade held for `h20+` defers realization and softens the tax bite. A trade held to `h60` is most tax-efficient.
   - One-line example: "If you expect a stock to dip 2% then recover 5% over h20, the right move is buy-and-hold-through-dip, not sell-on-dip."

5. **Update `state_for_apex` in `apex_portfolio.py`.**
   Add a new field on `PortfolioState`:
   ```python
   profit_definition: str = Field(default="net_profit = realized_pnl - tax_owed - slippage + cash_yield")
   ```
   And ensure `recent_realized_pnl_5d` and `tax_owed_accrued` continue to populate (they already do in v3.3; verify).

6. **Field validation for `primary_horizon` and `expected_alpha_bps`** lives on the `Order` Pydantic model itself via `Field` constraints. No separate validator needed at this stage. (Tax-aware rule-based validators are phase 2-D.)

7. **Write the test file** (see Testing Conditions).

---

## Acceptance Criteria

- [ ] `HorizonForecast` and `TickerView` exist in `apex_signatures.py` and round-trip through `model_dump()` / `model_validate()`.
- [ ] `Order` has `primary_horizon` and `expected_alpha_bps` as required fields.
- [ ] All 5 watchlist signatures emit both `views: list[TickerView]` and `orders: list[Order]`.
- [ ] A response where `orders[i].ticker` doesn't appear in `views[*].ticker` fails Pydantic validation.
- [ ] Apex prompt strings contain the tax/slippage/net-profit/horizon-strategy block.
- [ ] `state_for_apex()` returns a `PortfolioState` with `profit_definition` populated.
- [ ] Existing tests in `tests/` still pass.

---

## Testing Conditions (exit verification)

Run each and confirm expected output before flipping DONE.

### 1. Pydantic models import and validate

```bash
.venv/bin/python -c "
from trophic.beliefs.apex_signatures import HorizonForecast, TickerView, Order
hf = HorizonForecast(h1=0.001, h5=0.005, h20=0.02, h60=0.05,
    confidence_h1='med', confidence_h5='med', confidence_h20='high', confidence_h60='med',
    regime_note='range')
tv = TickerView(ticker='AAPL', forecasts=hf, primary_horizon='h20', rationale='earnings tailwind h20')
print('TickerView OK:', tv.ticker, tv.primary_horizon)
o = Order(side='BUY', ticker='AAPL', size_pct=10.0, reasoning='r',
    primary_horizon='h20', expected_alpha_bps=180)
print('Order OK:', o.ticker, o.primary_horizon, o.expected_alpha_bps)
"
```
**Expected output**: two `OK:` lines, no exception.

### 2. Order rejects out-of-range expected_alpha_bps

```bash
.venv/bin/python -c "
from trophic.beliefs.apex_signatures import Order
try:
    Order(side='BUY', ticker='X', size_pct=10.0, reasoning='r',
        primary_horizon='h5', expected_alpha_bps=999999)
    print('FAIL: should have raised')
except Exception as e:
    print('OK: rejected oversized expected_alpha_bps')
"
```
**Expected**: `OK: rejected oversized expected_alpha_bps`.

### 3. Order ↔ TickerView consistency check (in a watchlist signature)

```bash
.venv/bin/python -c "
from trophic.beliefs.apex_signatures import WatchlistFromObservationsPM, TickerView, HorizonForecast, Order
# Build response with mismatched ticker
hf = HorizonForecast(h1=0,h5=0,h20=0,h60=0,
    confidence_h1='low',confidence_h5='low',confidence_h20='low',confidence_h60='low',
    regime_note='range')
tv = TickerView(ticker='AAPL', forecasts=hf, primary_horizon='h5', rationale='r')
o = Order(side='BUY', ticker='MSFT', size_pct=10.0, reasoning='r',
    primary_horizon='h5', expected_alpha_bps=200)
# The signature/model should reject this composition.
# Implementation detail: this is enforced by a model_validator on the output type.
print('test composition built; check that the relevant model raises ValidationError when both passed together')
"
```
**Expected**: implementation-dependent — when the apex output model is validated with `views=[tv], orders=[o]`, it raises `ValidationError` mentioning ticker mismatch. Verify by writing this as a `pytest` test in `tests/test_multi_horizon_signatures.py` and running:

```bash
.venv/bin/python -m pytest tests/test_multi_horizon_signatures.py -x -v
```
**Expected**: all tests in the new file pass.

### 4. Tax-framing block is present in prompts

```bash
.venv/bin/python -c "
from trophic.beliefs import apex_signatures as A
text = (A.WatchlistFromObservationsPM.__doc__ or '') + \\
       (A.WatchlistFromFirehosePM.__doc__ or '') + \\
       (A.WatchlistOracle.__doc__ or '')
needed = ['37%', '5bps', 'net_profit', 'h20', 'h60']
missing = [n for n in needed if n not in text]
assert not missing, f'missing tokens in apex prompts: {missing}'
print('tax-framing block present in all 3 PM signatures')
"
```
**Expected**: `tax-framing block present in all 3 PM signatures`.

### 5. Existing tests still pass

```bash
.venv/bin/python -m pytest tests/ -x -q
```
**Expected**: green, no regressions.

---

## Coordination

- This mission is **strictly sequential** — phase 2 cannot start until you finish.
- If you discover that an existing `Order` consumer (e.g., `_order_to_dict` in
  `scripts/run_firehose_loop.py`) breaks because of the new required fields,
  fix that consumer here. Phase 2 should not have to patch around legacy
  Order construction sites.
- The `min_length`/`max_length` on `rationale` is your call — v3.3 user
  preference was "no upper limit on Opus reasoning"; match that here.

---

## When Done

```bash
# 1. Inbox grep one more time
grep -E "@all|@phase1|@phase1-A" tasks_v4/scratchpad.md | tail -30

# 2. STATUS flip
sed -i 's/phase1-A:01:RUNNING/phase1-A:01:DONE/' tasks_v4/scratchpad.md
# Flip the three phase-2 missions from BLOCKED to PENDING
sed -i 's/phase2-B:02:BLOCKED.*$/phase2-B:02:PENDING/' tasks_v4/scratchpad.md
sed -i 's/phase2-C:03:BLOCKED.*$/phase2-C:03:PENDING/' tasks_v4/scratchpad.md
sed -i 's/phase2-D:04:BLOCKED.*$/phase2-D:04:PENDING/' tasks_v4/scratchpad.md

# 3. MESSAGES entry
cat >> tasks_v4/scratchpad.md <<'EOF'

[YYYY-MM-DD HH:MM] phase1-A > @phase2-B,C,D: signatures landed. `HorizonForecast`, `TickerView`, extended `Order` (with `primary_horizon` + `expected_alpha_bps`) all available in `trophic/beliefs/apex_signatures.py`. Tax block is in all PM prompts. Phase 2 unblocked.
EOF

# 4. Move this mission file
git mv tasks_v4/phase1-A-01-multi-horizon-signatures.md tasks_v4/completed/
```
