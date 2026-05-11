"""Phase 2-B (v4): full-window Oracle signature contract + matrix builder.

Tests the v4 replacement for v3.3's reactive per-day Oracle. Covers:

  - `WatchlistOracleFullWindow` signature has the required input fields
    (`today`, `days_remaining`, `portfolio_state`, `positions`, `watchlist`,
    `future_returns_matrix`, `pruned_tickers`, `previous_attempts`).
  - The prompt docstring contains every directive the mission lists:
    foresight, buy-low/sell-high, anti-sell-on-dip, the 0.5-orders-per-day
    cap, tax framing (37% / h60), and horizon-selection guidance.
  - `build_future_returns_matrix` in the runner correctly slices the
    per-day dataset, never extends past window-end, and prunes by
    realized volatility when the token budget is exceeded.
  - `TickerForwardReturns` is a Pydantic BaseModel (matches the user's
    "Pydantic everywhere in DSPy" preference).
"""
from __future__ import annotations

from pydantic import BaseModel

from trophic.beliefs.apex_signatures import (
    TickerForwardReturns,
    WatchlistOracle,           # DEPRECATED; ensure it's still importable
    WatchlistOracleFullWindow,
)


# ── Signature contract ──────────────────────────────────────────────────

def test_signature_input_fields_present():
    fields = WatchlistOracleFullWindow.input_fields
    names = set(fields.keys())
    required = {
        "today",
        "days_remaining",
        "portfolio_state",
        "positions",
        "watchlist",
        "future_returns_matrix",
        "pruned_tickers",
        "previous_attempts",
    }
    missing = required - names
    assert not missing, f"signature missing input fields: {missing}"


def test_signature_output_fields_match_phase1_pattern():
    fields = WatchlistOracleFullWindow.output_fields
    names = set(fields.keys())
    # Phase 1-A adaptation: `views` and `orders` are SEPARATE OutputFields
    # so the runner can keep `pred.orders` / `pred.views` consumer pattern.
    assert {"views", "orders", "rationale"}.issubset(names), (
        f"signature missing required output fields; have {names}"
    )


def test_legacy_oracle_signature_is_still_importable_but_marked_deprecated():
    # Importability ensures we didn't accidentally break the back-compat
    # path; the source comment marker proves the deprecation was added.
    assert WatchlistOracle is not None
    import inspect
    import trophic.beliefs.apex_signatures as A
    src = inspect.getsource(A)
    assert "DEPRECATED in v4" in src, (
        "expected `DEPRECATED in v4` comment on the legacy WatchlistOracle"
    )


# ── Prompt content ──────────────────────────────────────────────────────

def test_prompt_contains_all_required_directives():
    text = (WatchlistOracleFullWindow.__doc__ or "").lower()
    required_snippets = [
        # Foresight + buy-low-sell-high
        "you have foresight of every ticker's daily return for the rest of the window",
        "buy near local minima",
        "sell near local maxima",
        # Anti-pattern
        "do not sell-on-dip",
        # Minimum orders
        "minimum number of orders",
        "0.5 orders per day",
        # Tax framing (inherited from phase 1-A)
        "37%",
        "5bps",
        "net_profit",
        "h20",
        "h60",
        # Horizon-selection guidance
        "h1",
        "h5",
        "last resort",
        # Dip-recover example numbers should be present (from worked example)
        "+2.7%",
    ]
    missing = [s for s in required_snippets if s not in text]
    assert not missing, f"oracle prompt missing required directives: {missing}"


def test_prompt_carries_dip_and_recover_worked_example():
    text = WatchlistOracleFullWindow.__doc__ or ""
    # Both the worked example (in the strategy block) and the dip-recover
    # example from the inherited tax block should be present.
    assert "dip-and-recover" in text.lower() or "dip and recover" in text.lower()
    # The strategy block's concrete vector
    assert "forward_returns" in text


# ── Pydantic-everywhere contract ────────────────────────────────────────

def test_ticker_forward_returns_is_pydantic():
    assert issubclass(TickerForwardReturns, BaseModel), (
        "TickerForwardReturns must be a Pydantic BaseModel "
        "(user preference: Pydantic everywhere in DSPy)."
    )
    instance = TickerForwardReturns(
        ticker="AAPL",
        forward_returns=[0.001, -0.002, 0.0034],
    )
    assert instance.ticker == "AAPL"
    assert instance.forward_returns == [0.001, -0.002, 0.0034]


# ── Matrix builder math (deterministic) ─────────────────────────────────

def _make_row(date: str, actual_returns: dict[str, float]) -> dict:
    """Synthesize a per-day dataset row with the minimal shape the builder
    expects: `date`, `targets[].ticker`, `targets[].actual_return`.
    """
    return {
        "date": date,
        "targets": [
            {"ticker": tk, "actual_return": r}
            for tk, r in actual_returns.items()
        ],
    }


def test_matrix_builder_basic_math():
    """For today_idx=0, days_remaining=4 in a 4-day window, the matrix should
    cover rows[1..3] (3 future days) and never include today's row itself.
    """
    from scripts.run_firehose_loop import build_future_returns_matrix

    rows = [
        _make_row("2026-02-03", {"AAPL": 0.010, "MSFT": -0.005}),
        _make_row("2026-02-04", {"AAPL": 0.020, "MSFT": -0.010}),
        _make_row("2026-02-05", {"AAPL": -0.005, "MSFT": 0.015}),
        _make_row("2026-02-06", {"AAPL": 0.003, "MSFT": 0.002}),
    ]
    matrix, pruned = build_future_returns_matrix(
        today_idx=0,
        rows=rows,
        universe=["AAPL", "MSFT"],
        days_remaining=4,  # today (idx 0) + 3 forward days
    )

    by_tk = {m.ticker: m.forward_returns for m in matrix}
    # The forward array for AAPL: rows[1..3].targets[AAPL].actual_return
    assert by_tk["AAPL"] == [0.02, -0.005, 0.003], by_tk["AAPL"]
    assert by_tk["MSFT"] == [-0.01, 0.015, 0.002], by_tk["MSFT"]
    # 4 days remaining → at most 3 forward returns per ticker
    for tk, rs in by_tk.items():
        assert len(rs) <= 3, f"{tk} has {len(rs)} forward returns; expected ≤ 3"
    # No ticker pruned in this small case (no token-budget pressure).
    assert pruned == []


def test_matrix_builder_truncates_floats_to_four_decimals():
    """Verifies the token-budget mitigation: floats are rounded to 4 decimals."""
    from scripts.run_firehose_loop import build_future_returns_matrix

    rows = [
        _make_row("d0", {"AAPL": 0.0}),
        _make_row("d1", {"AAPL": 0.123456789}),
        _make_row("d2", {"AAPL": -0.987654321}),
    ]
    matrix, _ = build_future_returns_matrix(
        today_idx=0, rows=rows, universe=["AAPL"], days_remaining=3,
    )
    aapl = matrix[0].forward_returns
    # All values must equal round(orig, 4).
    assert aapl == [0.1235, -0.9877], aapl


def test_matrix_builder_never_leaks_past_window_end():
    """HARD INVARIANT: with days_remaining=N, no ticker has more than N-1
    forward returns even if the dataset has additional rows after window-end.
    """
    from scripts.run_firehose_loop import build_future_returns_matrix

    # 10 dataset rows but window-end is at today_idx + days_remaining - 1 = 2
    rows = [
        _make_row(f"d{i}", {"AAPL": 0.001 * i})
        for i in range(10)
    ]
    matrix, _ = build_future_returns_matrix(
        today_idx=0, rows=rows, universe=["AAPL"], days_remaining=3,
    )
    aapl = matrix[0].forward_returns
    assert len(aapl) == 2, f"matrix leaked past window-end: got {len(aapl)} returns"
    assert aapl == [0.001, 0.002], aapl  # rows[1].AAPL, rows[2].AAPL


def test_matrix_builder_prunes_under_token_budget():
    """When the estimated payload exceeds the budget, the lowest-volatility
    tickers are dropped from the matrix and surfaced in `pruned_tickers`.
    """
    from scripts.run_firehose_loop import build_future_returns_matrix

    # Build a 60-day, 30-ticker setup. Half are high-vol, half are dead-flat
    # constants. With a tiny budget, only the high-vol ones should survive.
    high_vol = [f"V{i:02d}" for i in range(15)]
    low_vol = [f"F{i:02d}" for i in range(15)]
    universe = high_vol + low_vol

    def _ret(tk: str, day_idx: int) -> float:
        if tk in high_vol:
            # Sinusoidal-ish pattern → big stdev
            return 0.05 if day_idx % 3 == 0 else (-0.03 if day_idx % 3 == 1 else 0.04)
        # F* tickers: dead flat (stdev = 0)
        return 0.0

    rows = [
        _make_row(f"d{i}", {tk: _ret(tk, i) for tk in universe})
        for i in range(60)
    ]

    # Pick a budget that fits all 15 high-vol but no low-vol.
    # Each ticker ≈ 59 returns * 8 chars + ~12 chars header ≈ 484 chars
    # → ~121 tokens. 15 high-vol = ~1815 tokens; the 16th would overflow
    # at ~1936 tokens. budget=1900 admits exactly the 15 high-vol set.
    matrix, pruned = build_future_returns_matrix(
        today_idx=0,
        rows=rows,
        universe=universe,
        days_remaining=60,
        token_budget=1900,
    )

    kept_tk = {m.ticker for m in matrix}
    # Every kept ticker should be high-vol (since high-vol entries were
    # sorted to the front by the builder).
    assert kept_tk.issubset(set(high_vol)), (
        f"a low-vol ticker survived but a high-vol one was pruned; kept={kept_tk}"
    )
    # All low-vol tickers must be in pruned (no edge in flat returns).
    assert set(low_vol).issubset(set(pruned)), (
        f"low-vol tickers were not all pruned; pruned={set(pruned)}"
    )
    # At least one high-vol ticker should survive.
    assert any(tk in high_vol for tk in kept_tk), (
        f"pruning was too aggressive; no high-vol survivors. matrix={kept_tk}"
    )
    # Kept + pruned = universe.
    assert kept_tk | set(pruned) == set(universe), (
        f"matrix accounting drift: kept|pruned={(kept_tk | set(pruned))} "
        f"vs universe={set(universe)}"
    )


def test_matrix_builder_handles_today_is_last_day():
    """At window-end (days_remaining=1), there are no forward days. The
    builder should return an empty matrix with all tickers pruned.
    """
    from scripts.run_firehose_loop import build_future_returns_matrix

    rows = [_make_row("d0", {"AAPL": 0.01, "MSFT": 0.02})]
    matrix, pruned = build_future_returns_matrix(
        today_idx=0, rows=rows, universe=["AAPL", "MSFT"], days_remaining=1,
    )
    assert matrix == []
    assert set(pruned) == {"AAPL", "MSFT"}


def test_matrix_builder_ticker_missing_from_universe_is_ignored():
    """A ticker present in the dataset rows but NOT in `universe` should be
    excluded; conversely, a universe ticker with no data is pruned.
    """
    from scripts.run_firehose_loop import build_future_returns_matrix

    rows = [
        _make_row("d0", {"AAPL": 0.0, "GME": 0.99}),  # GME not in universe
        _make_row("d1", {"AAPL": 0.01, "GME": -0.99}),
        _make_row("d2", {"AAPL": 0.02, "GME": 0.99}),
    ]
    matrix, pruned = build_future_returns_matrix(
        today_idx=0,
        rows=rows,
        universe=["AAPL", "NVDA"],  # NVDA has zero data
        days_remaining=3,
    )
    kept = {m.ticker for m in matrix}
    assert kept == {"AAPL"}, kept
    assert pruned == ["NVDA"], pruned
