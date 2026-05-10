# v4 — Coordination Scratchpad

**Status legend**: PENDING (not claimed) | RUNNING (claimed by an agent) | DONE | BLOCKED | PARKED

**Coordinator note**: Read this whole file before starting. Update your STATUS line atomically. Add a MESSAGES entry when you finish, hit a blocker, or have a finding worth surfacing to the next phase.

---

## STATUS

```
phase1-A:01:PENDING
phase2-B:02:BLOCKED  (needs phase1-A:01)
phase2-C:03:BLOCKED  (needs phase1-A:01)
phase2-D:04:BLOCKED  (needs phase1-A:01)
phase2_5:smoke:BLOCKED  (needs phase 2 complete)
phase3-A:05:BLOCKED  (needs phase 2 + smoke gate)
phase4:PARKED:PARKED (documentation only)
```

When phase1-A:01 lands → flip 02/03/04 to PENDING.
When phase2-B/C/D all land → flip smoke gate to PENDING.
When smoke gate lands → flip phase3-A:05 to PENDING.

---

## PHASE MAP

```
Phase 1 (sequential, agent A):
    phase1-A → 01-multi-horizon-signatures
    └─> unblocks phase2-B/C/D

Phase 2 (parallel, agents B/C/D):
    phase2-B → 02-full-window-oracle
    phase2-C → 03-strategy-committee
    phase2-D → 04-tax-aware-validators

Phase 2.5 (sequential smoke gate, any free agent):
    phase2_5 → smoke-gate (3-day mini-sweep, contract-only)
    └─> unblocks phase3-A

Phase 3 (sequential, agent A, with explicit test gates):
    phase3-A → 05-wire-interrogator-and-run

Phase 4 (PARKED):
    phase4 → 06-multi-hop-tool-use (no work yet, vision doc only)
```

---

## MISSION DEPENDENCY GRAPH

```
                phase1-A:01-multi-horizon-signatures
                /          |           \
        phase2-B:02   phase2-C:03   phase2-D:04   (parallel)
                \          |          /
                 \         |         /
                  phase2_5:smoke-gate
                            |
                  phase3-A:05-wire-interrogator-and-run
                            |
                  phase4:PARKED-06-multi-hop-tool-use (deferred)
```

---

## INBOX PROTOCOL

Subagents are not long-running listeners — they wake, run, return. The `tail -F | grep` listener via Monitor only fires while that specific agent is running; it does NOT span agent lifetimes. Late messages are caught by the start/end grep on the next agent's run.

### Required: grep on start (every agent runs this)

```bash
grep -E "@all|@phase[0-9]+|@<your-handle>|@<your-letter>" tasks_v4/scratchpad.md | tail -30
```

### Required: grep before flipping DONE

Same command. If there's a new message addressed to you or @all that you haven't acted on, address it before marking DONE.

### Optional: live tail listener while running

```bash
# Background tail; pair with Monitor or a background bash for notifications.
tail -F tasks_v4/scratchpad.md 2>/dev/null | grep --line-buffered -E "@all|@phase[0-9]+|@<your-handle>"
```

### Addressing scheme

- `@all` — every agent reads
- `@phase2` — every phase-2 agent reads
- `@phase2-B,C` — phase 2 agents B and C
- `@phase3-A` — specific agent

### Message format

```
[YYYY-MM-DD HH:MM] <from-handle> > @<to>: <message>
```

---

## SHARED FACTS

- **Repo root**: `/home/dgonier/ecology_experiment/trophic`
- **Working test command**: `cd /home/dgonier/ecology_experiment/trophic && .venv/bin/python -m pytest tests/ -x`
- **Python interpreter**: `.venv/bin/python` (do not activate the venv)
- **v3.3 archive**: `data/firehose_eval/runs/run_2026-05-10_pm_v3_3_9way/`
- **Baseline numbers (66-day window, same data)**:
  - BARE-QWEN: $109,674 net (+9.67%), $113,489 pre-tax (+13.49%) — **the bar**
  - ECO-QWEN: $108,321 net (+8.32%), $113,262 pre-tax (+13.26%) — within 0.23pp of BARE pre-tax
  - ORACLE-OPUS: $100,172 net (+0.17%), $110,170 pre-tax (+10.17%) — best Oracle, tax ate everything
  - ORACLE-SONNET: $93,272 net (-6.73%) — worst path
- **Tax**: 37% short-term cap gains; **5bps slippage** per leg; **4% annual cash yield**; **2 retries** on validator rejection.
- **Apex model matrix**: Qwen3.5-4B (local vLLM port 8001), Bedrock Sonnet 4.6, Bedrock Opus 4.6. Opus 4.7 is AccessDenied — don't change this.
- **Pydantic everywhere in DSPy** — non-negotiable user preference. No `list[dict]`-typed fields; always BaseModel.

---

## INTERFACE CONTRACTS

These are introduced in phase 1 and consumed by everyone after. **Do not change without an @all MESSAGES note.**

### From phase1-A:01 (must exist before phase 2 starts)

```python
# trophic/beliefs/apex_signatures.py

class HorizonForecast(BaseModel):
    h1:  float
    h5:  float
    h20: float
    h60: float
    confidence_h1:  Literal["low", "med", "high"]
    confidence_h5:  Literal["low", "med", "high"]
    confidence_h20: Literal["low", "med", "high"]
    confidence_h60: Literal["low", "med", "high"]
    regime_note: str  # one sentence: trend / mean-revert / breakout / range

class TickerView(BaseModel):
    ticker: str
    forecasts: HorizonForecast
    primary_horizon: Literal["h1", "h5", "h20", "h60"]
    rationale: str

class Order(BaseModel):
    # ... existing fields ...
    primary_horizon: Literal["h1", "h5", "h20", "h60"]  # NEW required field
    expected_alpha_bps: float = Field(ge=-10000, le=10000)  # NEW required, can be negative for SELL
    # primary_horizon must come from a TickerView in the same response
```

### Tier-to-size mapping (used by committee + validators)

```python
HORIZON_BASE_SIZE = {
    "h1":  3.0,   # probe
    "h5":  10.0,  # satellite
    "h20": 30.0,  # core
    "h60": 50.0,  # conviction-core
}
```

### From phase2-C:03 (strategy committee output)

```python
# trophic/beliefs/strategy_committee.py

class StrategyVote(BaseModel):
    strategy: Literal["conviction_weighted", "tiered_discrete", "kelly_edge"]
    order: Order
    conviction: float = Field(ge=0.0, le=1.0)

def aggregate_votes(
    votes: list[StrategyVote],
    philosophy_weights: dict[str, float],
) -> Order:
    """Weighted average of size_pct across strategies that agree on side.
    Returns aggregated Order or raises if strategies don't reach majority on side."""
```

### From phase2-D:04 (validator interface)

```python
# trophic/beliefs/validators.py (extended)

class ValidatorResult(BaseModel):
    accepted: bool
    reason: str  # required for rejections; empty if accepted
    suggested_horizon: Optional[Literal["h1","h5","h20","h60"]] = None

# All validators implement:
def validate(order: Order, portfolio_state: PortfolioState, positions: list[PositionSnapshot]) -> ValidatorResult
```

### Tax + slippage floor (used by Kelly strategy and validators)

```
floor_bps = 2 * SLIPPAGE_BPS + TAX_FRACTION * 10_000 * (expected_holding_return / portfolio_equity)
# For a default trade: 10bps slippage round-trip + 37% × realized → 162bps minimum edge to clear.
# Reject orders where abs(expected_alpha_bps) < floor_bps unless `primary_horizon == "h60"`
# (long-horizon trades waive the floor; the holding period itself reduces effective tax rate via deferral).
```

---

## MESSAGES

(append-only; newest at bottom)

[2026-05-10 16:15] coordinator > @all: scaffold created. Phase 1 is open. Phase 2 unblocks when phase1-A:01 flips DONE. Phase 3 has explicit test gates — do not skip the smoke gate before launching the v4 sweep. v3.3 archive is at `data/firehose_eval/runs/run_2026-05-10_pm_v3_3_9way/` for reference numbers.

[2026-05-10 16:15] coordinator > @phase2-B: full-window oracle changes from per-day reactive to single-trajectory committal. The buy-low-sell-high prompt is the headline; verify it in your prompt eyeball test before testing pipeline end-to-end.

[2026-05-10 16:15] coordinator > @phase2-C: weighted-average size aggregation. Strategies that disagree on `side` (e.g., BUY vs SELL) cause the aggregator to abstain (return HOLD with reason). Don't average opposing sides.

[2026-05-10 16:15] coordinator > @phase2-D: edge floor exempts `primary_horizon == "h60"` per the contract above. Confirm with @all if you want to change that exemption.

[2026-05-10 16:15] coordinator > @phase3-A: phase 3 is the highest-risk mission. Plan for a smoke (3-day) before the 66-day sweep. If smoke shows >40% order rejection, halt and request triage from @all before continuing.
