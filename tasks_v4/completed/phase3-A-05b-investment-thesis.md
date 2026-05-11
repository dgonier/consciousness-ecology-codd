# Mission 05b: investment-thesis (Pydantic type + lifecycle)

**Handle**: phase3-A-05b
**Phase**: 3 (sub-phase 3b — parallel start with 05a)
**Mission file**: phase3-A-05b-investment-thesis.md
**Dependencies**: phase2_5:smoke (DONE)
**Blocks**: phase3-A-05c (sub-portfolio), phase3-A-05d (debate)

---

## Before You Start

```bash
grep -E "@all|@phase3|@phase3-A|@phase3-A-05b" tasks_v4/scratchpad.md | tail -30
```

```bash
sed -i 's/phase3-A-05b:PENDING/phase3-A-05b:RUNNING/' tasks_v4/scratchpad.md 2>/dev/null || true
```

---

## Goal

Introduce `InvestmentThesis` as a first-class Pydantic artifact and define
its lifecycle. **A thesis is what each apex predator carries privately**
in its book — it's what gives predators distinct identity beyond a prompt
template.

This is type plumbing + lifecycle helpers + tests. No predator code yet
(that's 05c); no debate code (that's 05d).

---

## Files to Create / Modify

**Create:**
- `trophic/beliefs/investment_thesis.py` — the Pydantic model + lifecycle helpers.
- `tests/test_investment_thesis.py`.

**Do not modify** any existing apex / portfolio / committee / validator
code in this mission. That happens in 05c onward.

---

## Implementation Steps

1. **Define `InvestmentThesis`.**

   ```python
   # trophic/beliefs/investment_thesis.py
   from __future__ import annotations
   import uuid
   from typing import Literal, Optional
   from pydantic import BaseModel, Field

   ThesisStatus = Literal["active", "matured", "invalidated", "expired"]
   ThesisDirection = Literal["long", "short", "neutral"]
   PredatorPhilosophy = Literal["momentum", "value", "mean_revert", "event_driven"]

   class InvestmentThesis(BaseModel):
       thesis_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
       predator_id: str = Field(..., description="Which predator owns this thesis.")
       philosophy: PredatorPhilosophy
       ticker: str = Field(..., min_length=1, max_length=10)
       opened_at_date: str
       direction: ThesisDirection
       primary_horizon: Literal["h1", "h5", "h20", "h60"]

       catalysts: list[str] = Field(default_factory=list, description="Named events / signals that motivated the thesis.")
       invalidation_triggers: list[str] = Field(default_factory=list, description="Conditions under which the thesis should close early.")
       target_price: Optional[float] = None
       expected_alpha_bps: float = Field(..., ge=-10000, le=10000)
       confidence: Literal["low", "med", "high"]

       status: ThesisStatus = "active"
       closed_at_date: Optional[str] = None
       close_reason: Optional[str] = None  # "matured" / "invalidated:<trigger>" / "expired" / "manual"

       # Computed properties (set at close time)
       realized_pnl: Optional[float] = None
       realized_return_pct: Optional[float] = None

       class Config:
           populate_by_name = True
   ```

2. **Horizon-to-days expiry**: a thesis with `primary_horizon=h5` opened
   on day N expires (status → "expired") on day N+5 if not closed manually.
   Reuse `HORIZON_DAYS` from `trophic/beliefs/validators.py`.

3. **Lifecycle helpers.**

   ```python
   def open_thesis(predator_id: str, philosophy: PredatorPhilosophy,
                   ticker: str, opened_at_date: str, direction: ThesisDirection,
                   primary_horizon: str, catalysts: list[str],
                   invalidation_triggers: list[str], expected_alpha_bps: float,
                   confidence: str, target_price: Optional[float] = None) -> InvestmentThesis: ...

   def mature_thesis(thesis: InvestmentThesis, closed_at_date: str,
                     realized_pnl: float, realized_return_pct: float) -> InvestmentThesis:
       """Set status='matured' when the primary_horizon elapsed naturally."""

   def invalidate_thesis(thesis: InvestmentThesis, closed_at_date: str, trigger: str,
                         realized_pnl: float, realized_return_pct: float) -> InvestmentThesis:
       """Set status='invalidated', close_reason='invalidated:<trigger>'."""

   def expire_thesis(thesis: InvestmentThesis, closed_at_date: str,
                     realized_pnl: float, realized_return_pct: float) -> InvestmentThesis:
       """Set status='expired' when horizon passed without explicit close."""

   def is_thesis_overdue(thesis: InvestmentThesis, today_idx: int, opened_idx: int) -> bool:
       """True if today_idx - opened_idx >= HORIZON_DAYS[thesis.primary_horizon]
          and status is still 'active'."""
   ```

4. **Validation rules** on the model itself (model_validator or field validators):
   - `closed_at_date` must be None for `status='active'`, set for all other statuses.
   - `close_reason` must be set when `status` is not 'active'.
   - `realized_pnl` and `realized_return_pct` must be set when status is matured/invalidated/expired.

5. **`ThesisBook`** (a thin wrapper around a list of theses, with lookup helpers):

   ```python
   class ThesisBook(BaseModel):
       predator_id: str
       theses: list[InvestmentThesis] = Field(default_factory=list)

       def active_theses(self) -> list[InvestmentThesis]: ...
       def active_for_ticker(self, ticker: str) -> list[InvestmentThesis]: ...
       def closed_theses(self) -> list[InvestmentThesis]: ...
       def add(self, thesis: InvestmentThesis) -> None: ...
       def close(self, thesis_id: str, status: ThesisStatus, ...) -> InvestmentThesis: ...
       def sweep_overdue(self, today_idx: int, opened_idx_by_thesis: dict[str, int]) -> list[InvestmentThesis]:
           """Move overdue active theses to 'expired'. Returns the list of newly expired."""
   ```

---

## Acceptance Criteria

- [ ] `InvestmentThesis` round-trips through `model_dump()` / `model_validate()`.
- [ ] Lifecycle helpers produce theses with consistent `status` / `closed_at_date` / `close_reason` / `realized_*` fields.
- [ ] `ThesisBook` add/close/active_for_ticker/sweep_overdue all behave correctly.
- [ ] Field constraints enforced (expected_alpha_bps bounds, etc.).
- [ ] Existing test suite (currently 196/1 after 05a, or 193/1 if 05a not yet merged) stays green.

---

## Testing Conditions

### 1. Construction + round-trip

```bash
.venv/bin/python -c "
from trophic.beliefs.investment_thesis import InvestmentThesis
t = InvestmentThesis(
    predator_id='momentum',
    philosophy='momentum',
    ticker='AAPL',
    opened_at_date='2026-02-03',
    direction='long',
    primary_horizon='h20',
    catalysts=['earnings_beat'],
    invalidation_triggers=['guidance_cut','breakout_failure'],
    expected_alpha_bps=350,
    confidence='high',
)
dumped = t.model_dump()
t2 = InvestmentThesis.model_validate(dumped)
assert t == t2, 'round-trip mismatch'
print('round-trip OK:', t.thesis_id, t.status)
"
```
**Expected**: `round-trip OK: <12-hex-id> active`.

### 2. Lifecycle helpers

```bash
.venv/bin/python -m pytest tests/test_investment_thesis.py::test_mature_invalidate_expire -x -v
```
**Expected**: pass. Test opens a thesis, matures it, asserts status/close_reason/realized_pnl; opens another, invalidates it with a trigger string, asserts close_reason includes the trigger; opens a third, expires it.

### 3. ThesisBook

```bash
.venv/bin/python -m pytest tests/test_investment_thesis.py::test_thesis_book_lifecycle -x -v
```
**Expected**: pass. Test adds 4 theses to a book, queries active_for_ticker, closes one as matured, asserts active() drops to 3, etc.

### 4. Overdue sweep

```bash
.venv/bin/python -m pytest tests/test_investment_thesis.py::test_sweep_overdue -x -v
```
**Expected**: pass. Open h5 thesis on day 0; sweep on day 3 → still active; sweep on day 5 → expired.

### 5. Validation

```bash
.venv/bin/python -m pytest tests/test_investment_thesis.py::test_validation_rules -x -v
```
**Expected**: pass. Tests:
- expected_alpha_bps out of range rejects.
- status='matured' without close_reason rejects.
- status='active' with closed_at_date set rejects.

### 6. No regressions

```bash
.venv/bin/python -m pytest tests/ -x -q
```
**Expected**: green.

---

## Coordination

- **No side-effects** on existing apex/portfolio/committee/validator code.
- 05c will *use* `ThesisBook` inside `PredatorSubPortfolio`.
- 05d will *use* the catalysts + invalidation_triggers fields when
  predators argue over each other's theses in round 2.
- The `target_price` field is optional — round 1 won't enforce it, but
  decomposer (deferred) will read it.

---

## When Done

```bash
grep -E "@all|@phase3" tasks_v4/scratchpad.md | tail -30
sed -i 's/phase3-A-05b:RUNNING/phase3-A-05b:DONE/' tasks_v4/scratchpad.md
sed -i 's/phase3-A-05c:BLOCKED.*$/phase3-A-05c:PENDING/' tasks_v4/scratchpad.md

cat >> tasks_v4/scratchpad.md <<'EOF'

[YYYY-MM-DD HH:MM] phase3-A-05b > @phase3-A-05c,05d: InvestmentThesis + ThesisBook landed in trophic/beliefs/investment_thesis.py. Lifecycle helpers (open/mature/invalidate/expire), overdue sweep, full validation. 05c can now build PredatorSubPortfolio around ThesisBook.
EOF

mv tasks_v4/phase3-A-05b-investment-thesis.md tasks_v4/completed/
```
