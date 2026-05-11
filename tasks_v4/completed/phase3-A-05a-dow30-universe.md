# Mission 05a: dow30-universe (blue-chip benchmark swap)

**Handle**: phase3-A-05a
**Phase**: 3 (sub-phase 3a — parallel start with 05b)
**Mission file**: phase3-A-05a-dow30-universe.md
**Dependencies**: phase2_5:smoke (DONE)
**Blocks**: phase3-A-05e (runner integration), phase3-A-05g (full sweep)

---

## Before You Start

```bash
grep -E "@all|@phase3|@phase3-A|@phase3-A-05a" tasks_v4/scratchpad.md | tail -30
```

```bash
sed -i 's/phase3-A-05a:PENDING/phase3-A-05a:RUNNING/' tasks_v4/scratchpad.md 2>/dev/null || true
```

---

## Goal

Swap the firehose universe to the **Dow 30** for the v4 benchmark sweep.
The current universe is mixed quality/volatility; v4's tax-friction story
gets a cleaner test on blue-chip names where 5bps slippage is reasonable
and h60 holds are realistic.

This mission is **data layer only**. No apex prompt changes, no validator
changes, no signature changes.

---

## Files to Create / Modify

**Create:**
- `trophic/data/dow30_universe.py` — canonical Dow 30 ticker list as a Python constant + helper.
- `tests/test_dow30_universe.py`.

**Modify:**
- Wherever the firehose universe is currently defined (grep for it; likely in `trophic/types.py`, `trophic/firehose.py`, or `scripts/run_firehose_loop.py`).
- The watchlist construction site that feeds apex `watchlist: list[str]`.
- The future-returns-matrix builder in `scripts/run_firehose_loop.py` (must respect the new universe).

---

## Implementation Steps

1. **Define the Dow 30 list.** Use the 30 components as of mid-2026 (or
   the most recent fixed snapshot you can confirm). Format:
   ```python
   # trophic/data/dow30_universe.py
   DOW30 = (
       "AAPL","AMGN","AMZN","AXP","BA","CAT","CRM","CSCO","CVX","DIS",
       "DOW","GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD",
       "MMM","MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WMT",
   )
   def is_dow30(ticker: str) -> bool:
       return ticker.upper() in DOW30
   ```
   If the exact membership has rotated (e.g., DOW out, SOMETHING in),
   adapt — the priority is "30 large-cap blue-chips," not historical purity.

2. **Find the current universe definition.**
   ```bash
   grep -rn "universe\b\|WATCHLIST\b\|TICKER_LIST\b" trophic/ scripts/ --include='*.py' | grep -v __pycache__ | head -40
   ```
   The v3.3 universe was implicit in the firehose data. There's likely a
   list of tickers that gets filtered or used to construct watchlists.
   Find it and replace with `DOW30` import.

3. **Watchlist construction.** When the apex receives `watchlist: list[str]`,
   it must only contain Dow 30 tickers. If watchlists were previously
   constructed from price/news activity (top-N by mention count etc.),
   the construction site stays the same but with the input universe
   filtered to `DOW30` first.

4. **News firehose filter.** Producer broadcasts referencing non-Dow30
   tickers must be filtered out before they reach the apex. Find the
   filter site; if there isn't one, add one in the runner's per-day
   pipeline.

5. **Future-returns-matrix builder** (phase2-B's `build_future_returns_matrix`
   in `scripts/run_firehose_loop.py`) must restrict to Dow 30 tickers.
   The matrix sizing budget calculation should also reflect this (30
   tickers × ~65 days × ~7 chars per float ≈ 14K chars; under 16K context).

6. **Add a feature flag**: `--universe dow30` (default for v4 sweep) and
   `--universe legacy` (back-compat for re-running v3.3-style sweeps).
   The default for v4 phase 3 is `dow30`.

---

## Acceptance Criteria

- [ ] `DOW30` constant exists with exactly 30 tickers.
- [ ] Watchlist construction filters to Dow 30 when `--universe dow30`.
- [ ] Producer broadcasts referencing non-Dow30 tickers are filtered out before apex sees them.
- [ ] `build_future_returns_matrix` respects the universe filter.
- [ ] `--universe legacy` flag preserves v3.3 behavior end-to-end.
- [ ] Existing tests still pass (currently 193/1 baseline).

---

## Testing Conditions

### 1. DOW30 constant has 30 unique tickers

```bash
.venv/bin/python -c "
from trophic.data.dow30_universe import DOW30, is_dow30
assert len(DOW30) == 30, f'expected 30, got {len(DOW30)}'
assert len(set(DOW30)) == 30, 'duplicates in DOW30'
assert is_dow30('AAPL') and is_dow30('aapl'), 'is_dow30 case-insensitive'
assert not is_dow30('TSLA'), 'is_dow30 should reject non-members'
print('DOW30 OK:', len(DOW30), 'unique blue-chip tickers')
"
```
**Expected**: `DOW30 OK: 30 unique blue-chip tickers`.

### 2. Watchlist filter

```bash
.venv/bin/python -m pytest tests/test_dow30_universe.py::test_watchlist_filters_to_dow30 -x -v
```
**Expected**: pass. Test seeds a mixed-universe watchlist (Dow 30 + non-Dow30 names), invokes the construction code, asserts output ⊆ DOW30.

### 3. Producer filter

```bash
.venv/bin/python -m pytest tests/test_dow30_universe.py::test_producer_broadcasts_filtered -x -v
```
**Expected**: pass. Synthetic broadcasts referencing 5 Dow30 names + 5 non-Dow30 names; only the 5 Dow30 broadcasts reach the apex input.

### 4. Future-returns matrix filter

```bash
.venv/bin/python -m pytest tests/test_dow30_universe.py::test_matrix_universe_filter -x -v
```
**Expected**: pass.

### 5. Feature flag works

```bash
grep -E "universe.*dow30|universe.*legacy" scripts/run_firehose_loop.py | wc -l
```
**Expected**: `>= 2` (one each for the dow30/legacy choices).

### 6. No regressions

```bash
.venv/bin/python -m pytest tests/ -x -q
```
**Expected**: green at 196/1 or higher (3 new tests added).

---

## Coordination

- **No side-effects** on apex signatures, strategy committee, or validators.
- **No side-effects** on `phase3-A-05b` (thesis model) — different file.
- If you find news data referencing tickers outside DOW30 that you can't
  cleanly filter (e.g., embedded mention of "TSLA" inside an AAPL-tagged
  article), strip-or-pass is a judgment call: prefer pass (don't tamper
  with article body), but ensure the article is only dispatched to
  in-universe apex contexts.

---

## When Done

```bash
grep -E "@all|@phase3" tasks_v4/scratchpad.md | tail -30
sed -i 's/phase3-A-05a:RUNNING/phase3-A-05a:DONE/' tasks_v4/scratchpad.md

cat >> tasks_v4/scratchpad.md <<'EOF'

[YYYY-MM-DD HH:MM] phase3-A-05a > @phase3-A-05e,05g: Dow 30 universe landed. `from trophic.data.dow30_universe import DOW30` for the ticker tuple. Default `--universe dow30` for v4 sweep; `--universe legacy` preserves v3.3 behavior.
EOF

mv tasks_v4/phase3-A-05a-dow30-universe.md tasks_v4/completed/
```
