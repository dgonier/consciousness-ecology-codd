# Mission 03: strategy-committee

**Handle**: phase2-C
**Phase**: 2 (parallel, after phase1-A:01)
**Mission file**: phase2-C-03-strategy-committee.md
**Dependencies**: phase1-A:01
**Blocks**: phase2_5:smoke, phase3-A:05

---

## Before You Start

```bash
grep -E "@all|@phase2|@phase2-C" tasks_v4/scratchpad.md | tail -30
```

```bash
sed -i 's/phase2-C:03:PENDING/phase2-C:03:RUNNING/' tasks_v4/scratchpad.md
```

Verify phase1 contracts:
```bash
.venv/bin/python -c "from trophic.beliefs.apex_signatures import TickerView, Order; o = Order(side='BUY', ticker='X', size_pct=10.0, reasoning='r', primary_horizon='h20', expected_alpha_bps=200); print('contract OK', o.primary_horizon)"
```

---

## Goal

Build a three-strategy committee that each look at the same TickerView and
each propose a sizing for the trade. Vote via weighted-average sizing,
where weights come from `philosophy_weights.yaml`. The aggregated Order is
what gets validated and (if accepted) executed.

Three strategies must coexist:
1. **Conviction-weighted (continuous)**: size = base_size × conviction²,
   where conviction ∈ [0, 1] is derived from the TickerView's
   confidence_at(primary_horizon) plus the magnitude of the forecast.
2. **Tiered discrete**: size ∈ {3%, 10%, 30%, 50%} chosen by mapping
   `primary_horizon` to the tier (h1→3, h5→10, h20→30, h60→50).
3. **Kelly-edge**: size = clamp(0, 50, kelly_fraction * 100) where
   kelly_fraction = expected_alpha / variance_estimate. Variance comes from
   the herd/anomaly tier in ECO, or a default 0.02 stdev assumption in BARE.

These three are NOT mutually exclusive in the aggregator. Each strategy
produces an order; the aggregator combines them.

---

## Files to Create / Modify

**Create:**
- `trophic/beliefs/strategy_committee.py` — `StrategyVote`, three strategy functions, `aggregate_votes`.
- `tasks_v4/philosophy_weights.yaml` — the runtime config.
- `tests/test_strategy_committee.py` — see Testing Conditions.

**Modify:**
- `scripts/run_firehose_loop.py` — after apex emits `views` and `orders`, wrap each Order through the committee aggregator.

---

## Implementation Steps

1. **Define the types and the strategy functions.**
   ```python
   # trophic/beliefs/strategy_committee.py
   from pydantic import BaseModel, Field
   from typing import Literal
   from .apex_signatures import Order, TickerView, PortfolioState

   class StrategyVote(BaseModel):
       strategy: Literal["conviction_weighted", "tiered_discrete", "kelly_edge"]
       order: Order
       conviction: float = Field(ge=0.0, le=1.0)

   HORIZON_BASE_SIZE = {"h1": 3.0, "h5": 10.0, "h20": 30.0, "h60": 50.0}
   ```

2. **Conviction-weighted strategy:**
   - Read confidence_at(primary_horizon) → one of low/med/high, map to {0.33, 0.66, 1.0}.
   - Combine with abs(forecast) at primary_horizon: conviction = min(1, 0.5 * conf_value + 0.5 * tanh(20 * abs(forecast))).
   - Size = HORIZON_BASE_SIZE[primary_horizon] * conviction**2.
   - Side: BUY if forecast > 0, SELL if forecast < 0, HOLD if |forecast| < 0.001.

3. **Tiered discrete:**
   - Size = HORIZON_BASE_SIZE[primary_horizon] (no scaling).
   - Side: same rule as #2.
   - Conviction: confidence_at(primary_horizon) → {0.33, 0.66, 1.0}.

4. **Kelly-edge:**
   - kelly_fraction = expected_alpha_bps / 10_000 / variance_estimate.
   - If `variance_estimate` is not provided in the TickerView (it won't be in v4 round 1), default to 0.02² = 0.0004.
   - Size = clamp(0, 50, 100 * kelly_fraction).
   - **Edge floor**: if abs(expected_alpha_bps) < tax_slip_floor (162 default, but waive for primary_horizon == "h60"), this strategy votes HOLD with conviction 0.
   - Side: same rule as #2.

5. **Aggregator** (`aggregate_votes`):
   ```python
   def aggregate_votes(
       votes: list[StrategyVote],
       philosophy_weights: dict[str, float],  # keys: conviction_weighted, tiered_discrete, kelly_edge
       conviction_floor: float = 0.20,
   ) -> Order | None:
       # Filter to votes whose conviction >= conviction_floor
       active = [v for v in votes if v.conviction >= conviction_floor]
       if not active:
           return None  # all strategies abstain
       # Resolve side: majority wins; if tie or split, return None
       sides = [v.order.side for v in active]
       majority_side = max(set(sides), key=sides.count)
       if sides.count(majority_side) < (len(active) + 1) // 2 + 1:  # strict majority
           return None
       # Active votes that agree with majority side
       agreeing = [v for v in active if v.order.side == majority_side]
       # Weighted average of size_pct
       total_w = sum(philosophy_weights.get(v.strategy, 0.0) * v.conviction for v in agreeing)
       if total_w == 0:
           return None
       avg_size = sum(philosophy_weights.get(v.strategy, 0.0) * v.conviction * v.order.size_pct for v in agreeing) / total_w
       # primary_horizon: majority among agreeing votes; ties → longest horizon
       horizons = [v.order.primary_horizon for v in agreeing]
       horizon_order = ["h1", "h5", "h20", "h60"]
       majority_horizon = max(horizons, key=lambda h: (horizons.count(h), horizon_order.index(h)))
       # Build aggregated Order
       template = agreeing[0].order
       return Order(
           side=majority_side,
           ticker=template.ticker,
           size_pct=max(0.0, min(100.0, avg_size)),
           reasoning=f"committee[{','.join(v.strategy for v in agreeing)}] avg_size={avg_size:.1f}%",
           primary_horizon=majority_horizon,
           expected_alpha_bps=sum(v.order.expected_alpha_bps for v in agreeing) / len(agreeing),
       )
   ```

6. **`tasks_v4/philosophy_weights.yaml`:**
   ```yaml
   # Strategy committee weighting. Higher weights → more influence in size_pct average.
   # All three must sum to a positive number.
   # Defaults: equal weight across all three.
   conviction_weighted: 0.34
   tiered_discrete:     0.33
   kelly_edge:          0.33

   # Trading philosophy bias (additive on top of strategy weights).
   # Defaults: balanced. Tune in v4.1 once we have committee-level signal.
   philosophy_bias:
     momentum:    0.0   # if > 0, boost conviction_weighted for tickers with positive h1+h5 forecasts
     mean_revert: 0.0   # if > 0, boost tiered_discrete for tickers with negative h1, positive h20
     value:       0.0   # if > 0, boost kelly_edge with longer-horizon emphasis
   ```

7. **Wire the committee into the runner.**
   In `scripts/run_firehose_loop.py`, after the apex returns `views` and `orders`:
   ```python
   from trophic.beliefs.strategy_committee import (
       conviction_weighted_vote, tiered_discrete_vote, kelly_edge_vote, aggregate_votes
   )
   import yaml
   weights = yaml.safe_load(open("tasks_v4/philosophy_weights.yaml"))
   committee_orders = []
   for order in apex_response.orders:
       view = next((v for v in apex_response.views if v.ticker == order.ticker), None)
       if view is None:
           continue
       votes = [
           conviction_weighted_vote(view, order),
           tiered_discrete_vote(view, order),
           kelly_edge_vote(view, order),
       ]
       agg = aggregate_votes(votes, weights)
       if agg is not None:
           committee_orders.append(agg)
   ```
   Replace the apex's raw `orders` with `committee_orders` before the validation step.

---

## Acceptance Criteria

- [ ] `strategy_committee.py` exports `StrategyVote`, three strategy functions, and `aggregate_votes`.
- [ ] All three strategy functions accept `(TickerView, Order)` and return `StrategyVote`.
- [ ] Aggregator returns `None` when strategies disagree on side without strict majority.
- [ ] Aggregator returns a fully-formed `Order` with averaged `size_pct` and the agreeing-majority `primary_horizon`.
- [ ] `philosophy_weights.yaml` is loadable and validates (all keys present, weights non-negative).
- [ ] Runner uses committee output (not raw apex orders) for validation.
- [ ] Existing tests pass.

---

## Testing Conditions

### 1. Strategy functions produce sane votes on a known TickerView

```bash
.venv/bin/python -m pytest tests/test_strategy_committee.py::test_conviction_weighted_basic -x -v
```
**Expected**: pass. The test should construct a high-confidence h20 TickerView with forecast h20 = +5%, verify conviction_weighted returns BUY with size ≈ 30% (HORIZON_BASE_SIZE['h20']) × (some conviction ≈ 0.85²) ≈ 21%.

### 2. Aggregator rejects split sides

```bash
.venv/bin/python -m pytest tests/test_strategy_committee.py::test_aggregator_rejects_split_sides -x -v
```
**Expected**: pass. Test fakes 2 BUY votes and 1 SELL vote with similar conviction — strict majority requires 2-of-3 but the test verifies the side-tie/disagreement path.

### 3. Aggregator averages size_pct weighted by philosophy

```bash
.venv/bin/python -m pytest tests/test_strategy_committee.py::test_aggregator_weighted_average -x -v
```
**Expected**: pass. Test gives three votes BUY with sizes {10%, 30%, 20%}, equal weights and convictions → average = 20%. Then swap to non-equal weights and verify the math.

### 4. Edge floor on Kelly-edge waives for h60

```bash
.venv/bin/python -m pytest tests/test_strategy_committee.py::test_kelly_waives_floor_for_h60 -x -v
```
**Expected**: pass. expected_alpha_bps = 100 (below 162 floor); h60 → vote returns BUY; h5 → vote returns HOLD with 0 conviction.

### 5. YAML config loads

```bash
.venv/bin/python -c "
import yaml
d = yaml.safe_load(open('tasks_v4/philosophy_weights.yaml'))
assert {'conviction_weighted','tiered_discrete','kelly_edge'} <= set(d), f'missing strategy keys: {d.keys()}'
assert 'philosophy_bias' in d
print('philosophy_weights.yaml loads with expected keys')
"
```
**Expected**: `philosophy_weights.yaml loads with expected keys`.

### 6. Runner imports the committee

```bash
grep -c "from trophic.beliefs.strategy_committee" scripts/run_firehose_loop.py
```
**Expected**: `>= 1`.

### 7. No regressions

```bash
.venv/bin/python -m pytest tests/ -x -q
```
**Expected**: green.

---

## Coordination

- **No side-effects on `validators.py`** — that's phase2-D's territory. The
  committee outputs raw `Order`s; validators (whether v3.3 or phase2-D)
  consume them afterward.
- **No side-effects on `apex_signatures.py`** — that's phase1-A territory
  and is already DONE. If you need to change it, post @all in MESSAGES first.
- **If you discover the apex's TickerView omits something the strategies
  need** (e.g., volatility for Kelly), default-fill in your strategy
  function and post a MESSAGES note to @phase1-A explaining what would be
  cleaner. Don't block on schema changes — defaults are fine for v4 round 1.

---

## When Done

```bash
grep -E "@all|@phase2|@phase2-C" tasks_v4/scratchpad.md | tail -30
sed -i 's/phase2-C:03:RUNNING/phase2-C:03:DONE/' tasks_v4/scratchpad.md

cat >> tasks_v4/scratchpad.md <<'EOF'

[YYYY-MM-DD HH:MM] phase2-C > @phase2_5,phase3-A: strategy committee landed. Three strategies, weighted-average aggregator, philosophy_weights.yaml config. Runner wraps apex orders through `aggregate_votes`. If aggregator returns None (disagreement or all-abstain), order is dropped.
EOF

git mv tasks_v4/phase2-C-03-strategy-committee.md tasks_v4/completed/
```
