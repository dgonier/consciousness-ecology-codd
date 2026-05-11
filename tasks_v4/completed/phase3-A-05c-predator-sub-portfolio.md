# Mission 05c: predator-sub-portfolio (per-predator capital + book)

**Handle**: phase3-A-05c
**Phase**: 3 (sequential, after 05b)
**Mission file**: phase3-A-05c-predator-sub-portfolio.md
**Dependencies**: phase3-A-05b (InvestmentThesis + ThesisBook)
**Blocks**: phase3-A-05d (debate), phase3-A-05e (runner integration)

---

## Before You Start

```bash
grep -E "@all|@phase3|@phase3-A|@phase3-A-05c" tasks_v4/scratchpad.md | tail -30
```

Confirm 05b's contract:
```bash
.venv/bin/python -c "from trophic.beliefs.investment_thesis import InvestmentThesis, ThesisBook; print('05b contract present')"
```
If this fails, halt and post @all.

```bash
sed -i 's/phase3-A-05c:PENDING/phase3-A-05c:RUNNING/' tasks_v4/scratchpad.md 2>/dev/null || true
```

---

## Goal

Introduce `PredatorSubPortfolio`: each apex predator in a DEBATE path
owns its own slice of capital, its own positions, its own tax_owed_accrued,
its own recent_realized_pnl, and its own `ThesisBook`. The overall
`ApexPortfolio` becomes a thin shell that aggregates N sub-portfolios.

Round 1 of the DEBATE mechanism (next mission, 05d) will read these
sub-portfolios; round 3 will mutate them. Validators will run **per
sub-portfolio**, not on the aggregated whole.

---

## Files to Create / Modify

**Create:**
- `trophic/agents/predator_sub_portfolio.py` — `PredatorSubPortfolio` class.
- `tests/test_predator_sub_portfolio.py`.

**Modify:**
- `trophic/agents/apex_portfolio.py` — `ApexPortfolio` gains an optional
  `sub_portfolios: dict[str, PredatorSubPortfolio]` field (None for v3.3/v4
  non-debate paths; populated for DEBATE paths). The legacy single-book
  positions / cash / tax_owed_accrued fields remain for non-debate paths.
  When `sub_portfolios` is populated, the aggregate equity / cash / inv_pct
  are computed by summing across predators.

---

## Implementation Steps

1. **Define `PredatorSubPortfolio`.**

   ```python
   # trophic/agents/predator_sub_portfolio.py
   from __future__ import annotations
   from dataclasses import dataclass, field
   from trophic.beliefs.investment_thesis import ThesisBook, InvestmentThesis

   @dataclass
   class PredatorSubPortfolio:
       predator_id: str                  # "momentum" | "value" | "mean_revert" | "event_driven"
       philosophy: str                   # one of PredatorPhilosophy values
       starting_cash: float              # e.g., $25K for a 4-predator split of $100K
       cash: float = 0.0                 # current cash
       positions: list = field(default_factory=list)   # _Position-shaped (reuse apex_portfolio's _Position dataclass)
       tax_owed_accrued: float = 0.0
       realized_pnl_history: list[tuple[str, float]] = field(default_factory=list)  # (date, pnl)
       thesis_book: ThesisBook = field(default_factory=lambda: ThesisBook(predator_id=""))

       def __post_init__(self):
           if not self.thesis_book.predator_id:
               self.thesis_book.predator_id = self.predator_id
           if self.cash == 0.0 and self.starting_cash > 0:
               self.cash = self.starting_cash

       def equity(self, prices: dict[str, float]) -> float: ...
       def invested_pct(self, prices: dict[str, float]) -> float: ...
       def recent_realized_pnl_5d(self, today: str) -> float: ...
       # Buy/sell/rotate methods scoped to this sub-portfolio only.
       def buy(self, ticker: str, dollars: float, slip_bps: float, today: str,
               primary_horizon: str, thesis_id: str | None = None) -> None: ...
       def sell(self, ticker: str, fraction_of_position: float, slip_bps: float,
                tax_fraction: float, today: str) -> tuple[float, float]:
           """Returns (proceeds, realized_pnl)."""
       def rotate(self, from_ticker: str, to_ticker: str, fraction: float, ...): ...
   ```

   **Key invariant**: the sub-portfolio's `cash + Σ(position_value) == equity` always.
   Aggregate equity across all predators is the total portfolio equity.

2. **Reuse `_Position` from `apex_portfolio.py`** (don't duplicate the
   dataclass). Import it. The `_Position` already carries `primary_horizon`
   and `bought_at_date` per the phase2-D contract. Add an optional
   `thesis_id: str = ""` field so positions can reference theses.

3. **Extend `ApexPortfolio`** in `apex_portfolio.py`:

   ```python
   @dataclass
   class ApexPortfolio:
       # ... existing fields ...
       sub_portfolios: Optional[Dict[str, PredatorSubPortfolio]] = None

       def is_debate_mode(self) -> bool:
           return self.sub_portfolios is not None and len(self.sub_portfolios) > 0

       def equity(self, prices) -> float:
           if self.is_debate_mode():
               return sum(s.equity(prices) for s in self.sub_portfolios.values())
           # ... existing single-book equity computation ...

       # Similar pattern for: cash, invested_pct, tax_owed_accrued (sum across sub-portfolios)
       # When is_debate_mode(), positions becomes a flat list of (predator_id, _Position) tuples.
   ```

4. **`state_for_apex` in DEBATE mode** returns per-predator state. Add a
   new method:

   ```python
   def state_for_predator(self, predator_id: str, today: str,
                          prices: dict[str, float]) -> PortfolioState:
       """Returns a PortfolioState scoped to ONE predator's view of its book.
       Used by the debate mechanism to feed each predator its own slice."""
   ```

   The legacy `state_for_apex` still works for non-debate paths.

5. **Factory helper:**

   ```python
   def make_debate_portfolio(total_starting_cash: float = 100_000,
                             predators: tuple[tuple[str, str], ...] = (
                                 ("momentum",    "momentum"),
                                 ("value",       "value"),
                                 ("mean_revert", "mean_revert"),
                                 ("event_driven","event_driven"),
                             )) -> ApexPortfolio:
       """Create an ApexPortfolio in debate mode with N predators each
       starting with total_starting_cash / N."""
   ```

6. **Critical detail**: `tax_owed_accrued` is per-predator. Each predator
   pays its own taxes from its own slice. Aggregate `tax_owed_accrued` =
   sum across predators. This means: a momentum predator that does well
   pays more tax (from its slice); a value predator that holds (no
   realizations) pays no tax. **Per-predator taxes reinforce the
   philosophy pressure.**

---

## Acceptance Criteria

- [ ] `PredatorSubPortfolio` exists and round-trips through its dataclass shape.
- [ ] `ApexPortfolio.is_debate_mode()` returns True iff `sub_portfolios` populated.
- [ ] Equity / cash / invested_pct / tax_owed_accrued aggregate correctly across sub-portfolios.
- [ ] `state_for_predator()` returns a PortfolioState scoped to one predator only.
- [ ] `make_debate_portfolio` constructs a 4-predator setup with $25K each.
- [ ] `_Position` gains optional `thesis_id` field; existing v3.3/v4 non-debate code unaffected.
- [ ] All existing tests still pass.

---

## Testing Conditions

### 1. Construction

```bash
.venv/bin/python -c "
from trophic.agents.apex_portfolio import make_debate_portfolio
p = make_debate_portfolio(total_starting_cash=100_000)
assert p.is_debate_mode()
assert len(p.sub_portfolios) == 4
assert all(s.starting_cash == 25_000 for s in p.sub_portfolios.values())
print('debate portfolio OK:', list(p.sub_portfolios.keys()))
"
```
**Expected**: `debate portfolio OK: ['momentum', 'value', 'mean_revert', 'event_driven']`.

### 2. Equity aggregation

```bash
.venv/bin/python -m pytest tests/test_predator_sub_portfolio.py::test_equity_aggregates -x -v
```
**Expected**: pass. 4 sub-portfolios with $25K cash each → aggregate equity = $100K; one buys AAPL @ $200; aggregate equity unchanged (cash → position); price ticks to $210 → aggregate equity = $100K + (position appreciation in that sub).

### 3. Per-predator tax isolation

```bash
.venv/bin/python -m pytest tests/test_predator_sub_portfolio.py::test_per_predator_tax_isolation -x -v
```
**Expected**: pass. Predator A realizes $1000 PnL → A's tax_owed_accrued goes up by $370; B/C/D's tax_owed_accrued unchanged. Aggregate tax_owed = $370.

### 4. state_for_predator

```bash
.venv/bin/python -m pytest tests/test_predator_sub_portfolio.py::test_state_for_predator_scoped -x -v
```
**Expected**: pass. Returns a PortfolioState with cash/equity/positions from one predator only, not aggregate.

### 5. Backward compat

```bash
.venv/bin/python -m pytest tests/test_predator_sub_portfolio.py::test_non_debate_legacy_unchanged -x -v
```
**Expected**: pass. An `ApexPortfolio(...)` without `sub_portfolios` set behaves identically to v3.3/v4 (existing tests are the proof).

### 6. No regressions

```bash
.venv/bin/python -m pytest tests/ -x -q
```
**Expected**: green.

---

## Coordination

- **No side-effects** on apex signatures, strategy committee, or validators.
- 05d (debate) will *drive* PredatorSubPortfolio via the round 1/2/3
  mechanism.
- 05e (runner) will *construct* PredatorSubPortfolios at the start of a
  DEBATE path.
- Validators (phase2-D) need to run per-predator in debate mode. They
  already operate on `PortfolioState` + `positions: list`, which
  `state_for_predator` provides. No validator changes needed here.

---

## When Done

```bash
grep -E "@all|@phase3" tasks_v4/scratchpad.md | tail -30
sed -i 's/phase3-A-05c:RUNNING/phase3-A-05c:DONE/' tasks_v4/scratchpad.md
sed -i 's/phase3-A-05d:BLOCKED.*$/phase3-A-05d:PENDING/' tasks_v4/scratchpad.md

cat >> tasks_v4/scratchpad.md <<'EOF'

[YYYY-MM-DD HH:MM] phase3-A-05c > @phase3-A-05d,05e: PredatorSubPortfolio + ApexPortfolio.is_debate_mode() + state_for_predator() + make_debate_portfolio() all landed. Per-predator tax isolation tested. Validators run per-predator unchanged.
EOF

mv tasks_v4/phase3-A-05c-predator-sub-portfolio.md tasks_v4/completed/
```
