# Mission 04: tax-aware-validators

**Handle**: phase2-D
**Phase**: 2 (parallel, after phase1-A:01)
**Mission file**: phase2-D-04-tax-aware-validators.md
**Dependencies**: phase1-A:01
**Blocks**: phase2_5:smoke, phase3-A:05

---

## Before You Start

```bash
grep -E "@all|@phase2|@phase2-D" tasks_v4/scratchpad.md | tail -30
```

```bash
sed -i 's/phase2-D:04:PENDING/phase2-D:04:RUNNING/' tasks_v4/scratchpad.md
```

Confirm contracts:
```bash
.venv/bin/python -c "from trophic.beliefs.apex_signatures import Order; o = Order(side='SELL', ticker='X', size_pct=10.0, reasoning='r', primary_horizon='h20', expected_alpha_bps=-200); print('contract OK')"
```

---

## Goal

Three new validators that enforce the v3.3 postmortem lessons:

1. **`MinHoldByHorizon`**: a position bought at `primary_horizon=h20` cannot
   be sold within 20 trading days unless the regime_note explicitly
   invalidates (regime_note ∈ {"breakout_failure", "event_driven_invalidation"}).
2. **`ForecastConsistency`**: an Order's side must agree with the sign of
   the forecast at primary_horizon. BUY requires forecasts[primary_horizon] > 0;
   SELL requires forecasts[primary_horizon] < 0.
3. **`HorizonSizingMatch`**: order's `size_pct` must be within ±5pp of the
   tier-base for its `primary_horizon`. Reject otherwise.

Plus the **edge floor**: orders with `abs(expected_alpha_bps) < floor` are
rejected, where `floor = 2 * 5 + 37% * |expected_holding_return| * 10_000`
unless `primary_horizon == "h60"`.

These compose with the existing v3.3 validators (`UnitNotional`, no-shorts,
sufficient-cash, etc.). Don't remove existing validators.

---

## Files to Create / Modify

**Create or extend:**
- `trophic/beliefs/validators.py` (or wherever current PM validators live —
  grep first) — add the three new validators + the edge floor.
- `tests/test_tax_aware_validators.py` — see Testing Conditions.

**Modify:**
- `scripts/run_firehose_loop.py` — register the new validators in the PM
  validator chain. They run AFTER the existing chain (so existing
  validators reject first; tax/horizon validators are the second layer).
- `trophic/agents/apex_portfolio.py` — must track `primary_horizon` per
  position (it's stored when buying; consulted when selling).

---

## Implementation Steps

1. **Locate existing validators.**
   ```bash
   grep -rln "class.*Validator\|def validate_orders\|def validate_pm" trophic/ scripts/ --include='*.py' | grep -v __pycache__
   ```
   Likely in `trophic/beliefs/portfolio_validators.py` or
   `trophic/beliefs/validators.py`. Read the existing chain before adding
   new ones.

2. **Position must remember its `primary_horizon`.**
   In `apex_portfolio.py`, the position dataclass/dict that tracks
   `(ticker, shares, avg_cost, ...)` must gain a `primary_horizon` field
   set at buy time and a `bought_at_date` (already exists for tax purposes).
   `MinHoldByHorizon` uses `bought_at_date + horizon_in_days` to enforce.

   Horizon-to-days mapping:
   ```python
   HORIZON_DAYS = {"h1": 1, "h5": 5, "h20": 20, "h60": 60}
   ```

3. **`MinHoldByHorizon` validator:**
   ```python
   def validate_min_hold(order: Order, positions, today: str) -> ValidatorResult:
       if order.side != "SELL":
           return ValidatorResult(accepted=True, reason="")
       pos = next((p for p in positions if p.ticker == order.ticker), None)
       if pos is None:
           return ValidatorResult(accepted=True, reason="")  # not our problem; other validators
       held_days = trading_days_between(pos.bought_at_date, today)
       required = HORIZON_DAYS[pos.primary_horizon]
       if held_days >= required:
           return ValidatorResult(accepted=True, reason="")
       # Allowed only on explicit invalidation regimes
       view = order_view_lookup(order)  # however views are passed through
       if view and view.forecasts.regime_note in ("breakout_failure", "event_driven_invalidation"):
           return ValidatorResult(accepted=True, reason="early_sell_allowed_by_regime")
       return ValidatorResult(
           accepted=False,
           reason=f"min_hold_violated: held {held_days}d, needs {required}d for {pos.primary_horizon}",
           suggested_horizon=pos.primary_horizon,
       )
   ```

4. **`ForecastConsistency` validator:**
   ```python
   def validate_forecast_consistency(order: Order, view: TickerView) -> ValidatorResult:
       fcst = getattr(view.forecasts, order.primary_horizon)
       if order.side == "BUY" and fcst <= 0:
           return ValidatorResult(accepted=False,
               reason=f"buy_with_nonpositive_forecast: {order.primary_horizon}={fcst:.4f}")
       if order.side == "SELL" and fcst >= 0:
           return ValidatorResult(accepted=False,
               reason=f"sell_with_nonnegative_forecast: {order.primary_horizon}={fcst:.4f}")
       return ValidatorResult(accepted=True, reason="")
   ```

5. **`HorizonSizingMatch` validator:**
   ```python
   HORIZON_BASE_SIZE = {"h1": 3.0, "h5": 10.0, "h20": 30.0, "h60": 50.0}
   SIZE_TOLERANCE_PP = 5.0  # +/- percentage points
   def validate_horizon_sizing(order: Order) -> ValidatorResult:
       base = HORIZON_BASE_SIZE[order.primary_horizon]
       if abs(order.size_pct - base) > SIZE_TOLERANCE_PP:
           return ValidatorResult(accepted=False,
               reason=f"sizing_mismatch: size={order.size_pct:.1f}% expected ~{base:.0f}±{SIZE_TOLERANCE_PP:.0f} for {order.primary_horizon}")
       return ValidatorResult(accepted=True, reason="")
   ```

6. **Edge floor:**
   ```python
   SLIPPAGE_BPS = 5.0
   TAX_FRACTION = 0.37
   def validate_edge_floor(order: Order, view: TickerView) -> ValidatorResult:
       if order.primary_horizon == "h60":
           return ValidatorResult(accepted=True, reason="floor_waived_for_h60")
       fcst = getattr(view.forecasts, order.primary_horizon)
       tax_drag_bps = TAX_FRACTION * 10_000 * abs(fcst)  # 37% of expected realized
       floor_bps = 2 * SLIPPAGE_BPS + tax_drag_bps
       if abs(order.expected_alpha_bps) < floor_bps:
           return ValidatorResult(accepted=False,
               reason=f"edge_below_floor: alpha={order.expected_alpha_bps:.1f}bps < floor={floor_bps:.1f}bps")
       return ValidatorResult(accepted=True, reason="")
   ```

7. **Register in the runner.**
   The PM dispatch path in `scripts/run_firehose_loop.py` must call:
   ```python
   from trophic.beliefs.validators import (
       validate_min_hold, validate_forecast_consistency,
       validate_horizon_sizing, validate_edge_floor,
   )
   ```
   And chain them after the existing validators. On rejection, increment
   the rejected counter (consistent with v3.3 behavior) and feed the
   rejection reason back to the apex for the retry loop.

8. **Wire `primary_horizon` into position tracking.**
   In `apex_portfolio.py`, the `_buy` (or whatever the method is) must
   accept and store `primary_horizon` on the position. When selling, the
   position's stored `primary_horizon` is read by `validate_min_hold`.

---

## Acceptance Criteria

- [ ] Four validator functions exist in the validators module.
- [ ] `MinHoldByHorizon` rejects early sells unless regime_note invalidates.
- [ ] `ForecastConsistency` rejects side ↔ forecast-sign mismatches.
- [ ] `HorizonSizingMatch` enforces ±5pp tolerance from base size.
- [ ] Edge floor waives for h60, rejects below 2×slippage + tax_drag otherwise.
- [ ] Positions track their `primary_horizon` for the lifetime of the position.
- [ ] Runner registers all four validators in the PM chain.
- [ ] Existing tests still pass.

---

## Testing Conditions

### 1. MinHoldByHorizon happy path + rejection path

```bash
.venv/bin/python -m pytest tests/test_tax_aware_validators.py::test_min_hold_rejects_early_sell -x -v
.venv/bin/python -m pytest tests/test_tax_aware_validators.py::test_min_hold_allows_after_horizon -x -v
.venv/bin/python -m pytest tests/test_tax_aware_validators.py::test_min_hold_allows_regime_invalidation -x -v
```
**Expected**: all three pass.

### 2. ForecastConsistency

```bash
.venv/bin/python -m pytest tests/test_tax_aware_validators.py::test_forecast_consistency -x -v
```
**Expected**: pass. Test should cover BUY+negative-forecast → reject, BUY+positive → accept, SELL+positive → reject.

### 3. HorizonSizingMatch

```bash
.venv/bin/python -m pytest tests/test_tax_aware_validators.py::test_horizon_sizing_match -x -v
```
**Expected**: pass. h5 with size 9% → accept (within 5pp of 10); h5 with size 17% → reject; h20 with size 35% → accept (within 5pp of 30); h20 with size 50% → reject.

### 4. Edge floor

```bash
.venv/bin/python -m pytest tests/test_tax_aware_validators.py::test_edge_floor -x -v
```
**Expected**: pass. h20 expected_alpha=100bps with forecast 0.01 → floor = 10 + 370 = 380bps; alpha < floor → reject. h60 same alpha → accept (waived). h20 alpha=500bps → accept.

### 5. Runner registers the new validators

```bash
grep -E "validate_min_hold|validate_forecast_consistency|validate_horizon_sizing|validate_edge_floor" scripts/run_firehose_loop.py | wc -l
```
**Expected**: `>= 4` (one import + per-call references).

### 6. Position tracking

```bash
.venv/bin/python -c "
from trophic.agents.apex_portfolio import ApexPortfolio
pf = ApexPortfolio(starting_cash=100_000)
# This is illustrative; the actual API may differ. Adapt to current signature.
# Verify positions can store and report primary_horizon.
import inspect
src = inspect.getsource(ApexPortfolio)
assert 'primary_horizon' in src, 'positions do not track primary_horizon'
print('positions track primary_horizon')
"
```
**Expected**: `positions track primary_horizon`.

### 7. No regressions

```bash
.venv/bin/python -m pytest tests/ -x -q
```
**Expected**: green.

---

## Coordination

- **No side-effects on `apex_signatures.py` or `strategy_committee.py`** —
  those are phase 1 / phase 2-C respectively.
- **The `regime_note` invalidation set is intentionally small** —
  `{"breakout_failure", "event_driven_invalidation"}`. If the apex
  hallucinates a third invalidation reason, the validator rejects. That's
  by design; we want the model to commit unless something genuinely changed.
- **Edge floor uses the `expected_alpha_bps` field on the Order**, NOT the
  forecast directly. The apex commits to a number; we hold it to that number.
- If your test suite catches a case where MinHoldByHorizon would reject
  every sell in the back-test (e.g., position re-buys reset the
  bought_at_date and stack), post in MESSAGES with the scenario.

---

## When Done

```bash
grep -E "@all|@phase2|@phase2-D" tasks_v4/scratchpad.md | tail -30
sed -i 's/phase2-D:04:RUNNING/phase2-D:04:DONE/' tasks_v4/scratchpad.md

cat >> tasks_v4/scratchpad.md <<'EOF'

[YYYY-MM-DD HH:MM] phase2-D > @phase2_5,phase3-A: tax-aware validators landed. MinHoldByHorizon, ForecastConsistency, HorizonSizingMatch, edge floor. Positions now persist their primary_horizon. Heads-up: if smoke shows >40% rejection rate, the model is fighting the validators — flag in MESSAGES.
EOF

git mv tasks_v4/phase2-D-04-tax-aware-validators.md tasks_v4/completed/
```
