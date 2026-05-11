"""Phase 2-D (v4): tax-aware validators.

Tests the four new validators that compose with the v3.3 chain:
  - validate_min_hold (reads POSITION's stored primary_horizon, not order's)
  - validate_forecast_consistency (side ↔ forecast-sign agreement)
  - validate_horizon_sizing (size_pct within ±10pp of HORIZON_BASE_SIZE)
  - validate_edge_floor (alpha clears 2×slippage + tax_drag; h60 waived)

Position-tracking behaviour is also smoke-checked: a BUY through the
ApexPortfolio.execute() path must persist primary_horizon on the
_Position, so a later SELL goes through the min-hold validator
correctly.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from trophic.agents.apex_portfolio import ApexPortfolio, _Position
from trophic.beliefs.apex_signatures import (
    HORIZON_BASE_SIZE,
    HorizonForecast,
    Order,
    TickerView,
)
from trophic.beliefs.validators import (
    HORIZON_DAYS,
    SIZE_TOLERANCE_PP,
    ValidatorResult,
    run_tax_aware_chain,
    validate_edge_floor,
    validate_forecast_consistency,
    validate_horizon_sizing,
    validate_in_universe,
    validate_min_hold,
)


# ── Fixtures ────────────────────────────────────────────────────────

def _make_forecast(
    h1=0.001, h5=0.005, h20=0.02, h60=0.05,
    regime_note="range",
):
    return HorizonForecast(
        h1=h1, h5=h5, h20=h20, h60=h60,
        confidence_h1="med", confidence_h5="med",
        confidence_h20="high", confidence_h60="med",
        regime_note=regime_note,
    )


def _make_view(ticker="AAPL", primary_horizon="h20", **forecast_kwargs):
    return TickerView(
        ticker=ticker,
        forecasts=_make_forecast(**forecast_kwargs),
        primary_horizon=primary_horizon,
        rationale="test rationale",
    )


def _make_order(
    side="BUY", ticker="AAPL", size_pct=None,
    primary_horizon="h20", expected_alpha_bps=200.0, reasoning="test",
):
    if size_pct is None:
        size_pct = HORIZON_BASE_SIZE[primary_horizon]
    return Order(
        side=side, ticker=ticker, size_pct=size_pct, reasoning=reasoning,
        primary_horizon=primary_horizon, expected_alpha_bps=expected_alpha_bps,
    )


def _make_position(ticker="AAPL", primary_horizon="h20", opened_on="2026-01-01"):
    return _Position(
        shares=10.0, cost_basis=1000.0,
        opened_on=opened_on, bought_at_date=opened_on,
        primary_horizon=primary_horizon, ticker=ticker,
    )


def _make_calendar(start="2026-01-01", n_days=60):
    """Generate a sequential YYYY-MM-DD calendar of `n_days` trading days."""
    import datetime
    d0 = datetime.date.fromisoformat(start)
    out = []
    for i in range(n_days):
        out.append((d0 + datetime.timedelta(days=i)).isoformat())
    return out


# ── 1. Min-hold ─────────────────────────────────────────────────────

def test_min_hold_rejects_early_sell():
    """h20 position sold after 5 days → reject."""
    cal = _make_calendar(n_days=30)
    pos = _make_position(primary_horizon="h20", opened_on=cal[0])
    today = cal[5]  # 5 trading days later
    order = _make_order(side="SELL", expected_alpha_bps=-200.0)
    view = _make_view()  # default regime "range"
    result = validate_min_hold(order, [pos], today, cal, [view])
    assert not result.accepted, result.reason
    assert "min_hold_violated" in result.reason
    assert result.suggested_horizon == "h20"


def test_min_hold_allows_after_horizon():
    """h5 position sold after 6 days → accept."""
    cal = _make_calendar(n_days=30)
    pos = _make_position(primary_horizon="h5", opened_on=cal[0])
    today = cal[6]  # past the 5-day horizon
    order = _make_order(
        side="SELL", primary_horizon="h5", size_pct=10.0,
        expected_alpha_bps=-200.0,
    )
    view = _make_view(primary_horizon="h5", h5=-0.01)
    result = validate_min_hold(order, [pos], today, cal, [view])
    assert result.accepted, result.reason


def test_min_hold_allows_regime_invalidation():
    """h20 sold after 5 days BUT regime_note=event_driven_invalidation → accept."""
    cal = _make_calendar(n_days=30)
    pos = _make_position(primary_horizon="h20", opened_on=cal[0])
    today = cal[5]
    order = _make_order(side="SELL", expected_alpha_bps=-200.0)
    view = _make_view(regime_note="event_driven_invalidation")
    result = validate_min_hold(order, [pos], today, cal, [view])
    assert result.accepted, result.reason
    assert "regime_invalidation" in (result.reason or "")


def test_min_hold_allows_regime_breakout_failure():
    """h20 sold after 3 days with regime_note=breakout_failure → accept."""
    cal = _make_calendar(n_days=30)
    pos = _make_position(primary_horizon="h20", opened_on=cal[0])
    today = cal[3]
    order = _make_order(side="SELL", expected_alpha_bps=-200.0)
    view = _make_view(regime_note="breakout_failure")
    result = validate_min_hold(order, [pos], today, cal, [view])
    assert result.accepted, result.reason


def test_min_hold_buy_passthrough():
    """BUY orders are never blocked by min_hold."""
    cal = _make_calendar(n_days=30)
    pos = _make_position(primary_horizon="h20", opened_on=cal[0])
    today = cal[1]
    order = _make_order(side="BUY", expected_alpha_bps=200.0)
    view = _make_view()
    result = validate_min_hold(order, [pos], today, cal, [view])
    assert result.accepted


def test_min_hold_no_matching_position():
    """SELL on unheld → accept here; v3.3 'SELL on unheld' fires upstream."""
    cal = _make_calendar(n_days=30)
    today = cal[1]
    order = _make_order(side="SELL", ticker="MSFT", expected_alpha_bps=-200.0)
    result = validate_min_hold(order, [], today, cal, [])
    assert result.accepted


# ── 2. Forecast consistency ─────────────────────────────────────────

def test_forecast_consistency():
    """BUY+negative → reject; BUY+positive → accept; SELL+positive → reject; SELL+negative → accept."""
    # BUY + negative forecast
    o_buy_neg = _make_order(side="BUY", primary_horizon="h20", expected_alpha_bps=200.0)
    v_neg = _make_view(h20=-0.02)
    r = validate_forecast_consistency(o_buy_neg, v_neg)
    assert not r.accepted
    assert "forecast_inconsistency" in r.reason

    # BUY + positive (canonical happy path)
    o_buy_pos = _make_order(side="BUY", primary_horizon="h20", expected_alpha_bps=200.0)
    v_pos = _make_view(h20=0.02)
    r = validate_forecast_consistency(o_buy_pos, v_pos)
    assert r.accepted

    # SELL + positive
    o_sell_pos = _make_order(side="SELL", primary_horizon="h20", expected_alpha_bps=-200.0)
    r = validate_forecast_consistency(o_sell_pos, v_pos)
    assert not r.accepted
    assert "forecast_inconsistency" in r.reason

    # SELL + negative
    r = validate_forecast_consistency(o_sell_pos, v_neg)
    assert r.accepted

    # BUY + exactly zero → reject (treated as non-positive)
    v_zero = _make_view(h20=0.0)
    r = validate_forecast_consistency(o_buy_pos, v_zero)
    assert not r.accepted


# ── 3. Horizon sizing ───────────────────────────────────────────────

def test_horizon_sizing_match():
    """h5/9% accept, h5/17% accept, h5/21% reject, h20/35% accept, h20/50% reject."""
    # h5 base = 10.0; tolerance ±10pp → [0, 20]
    r = validate_horizon_sizing(_make_order(primary_horizon="h5", size_pct=9.0))
    assert r.accepted
    # h5/17% now accepts under widened ±10pp tolerance (was reject at ±5pp)
    r = validate_horizon_sizing(_make_order(primary_horizon="h5", size_pct=17.0))
    assert r.accepted
    # h5/21% is the new reject case (just past the +10pp boundary)
    r = validate_horizon_sizing(_make_order(primary_horizon="h5", size_pct=21.0))
    assert not r.accepted
    assert "sizing_mismatch" in r.reason
    # h20 base = 30.0; tolerance ±10pp → [20, 40]
    r = validate_horizon_sizing(_make_order(primary_horizon="h20", size_pct=35.0))
    assert r.accepted
    # h20/50% remains a reject case (10pp past the +10pp upper boundary of 40)
    r = validate_horizon_sizing(_make_order(primary_horizon="h20", size_pct=50.0))
    assert not r.accepted
    assert "sizing_mismatch" in r.reason
    # boundaries
    # h5 at exactly 20.0 should be accepted (delta = 10.0 == tolerance, not >)
    r = validate_horizon_sizing(_make_order(primary_horizon="h5", size_pct=20.0))
    assert r.accepted
    # h5 at exactly 0.0 should be accepted (lower bound: 10 - 10 = 0)
    r = validate_horizon_sizing(_make_order(primary_horizon="h5", size_pct=0.0))
    assert r.accepted
    # h5 at 20.01 should reject (just over tolerance)
    r = validate_horizon_sizing(_make_order(primary_horizon="h5", size_pct=20.01))
    assert not r.accepted


def test_horizon_sizing_with_slice_fraction():
    """For debate-mode predators with $25K slice of $100K apex (frac=0.25):
    h20 base 30% × 0.25 = 7.5% slice-base. ±10pp tolerance applies.
    """
    # h20 BUY at 7.5% with slice_fraction=0.25 → accept (delta=0)
    r = validate_horizon_sizing(
        _make_order(primary_horizon="h20", size_pct=7.5),
        slice_fraction=0.25,
    )
    assert r.accepted

    # h20 BUY at 15% with slice_fraction=0.25 → accept (delta=7.5pp ≤ 10)
    r = validate_horizon_sizing(
        _make_order(primary_horizon="h20", size_pct=15.0),
        slice_fraction=0.25,
    )
    assert r.accepted

    # h20 BUY at 20% with slice_fraction=0.25 → reject (delta=12.5pp > 10)
    r = validate_horizon_sizing(
        _make_order(primary_horizon="h20", size_pct=20.0),
        slice_fraction=0.25,
    )
    assert not r.accepted
    assert "sizing_mismatch" in r.reason

    # Backwards compat: default slice_fraction=1.0 still tier-based.
    # h20 at 30% with no slice arg → accept (whole-portfolio mode)
    r = validate_horizon_sizing(
        _make_order(primary_horizon="h20", size_pct=30.0),
    )
    assert r.accepted

    # h60 base 50% × 0.25 slice = 12.5%. h60 BUY at 12% accepts.
    r = validate_horizon_sizing(
        _make_order(primary_horizon="h60", size_pct=12.0),
        slice_fraction=0.25,
    )
    assert r.accepted


# ── 4. Edge floor ───────────────────────────────────────────────────

def test_edge_floor():
    """h20/100bps with forecast 0.01 → floor=380, reject; h60 waived; h20/500bps → accept."""
    # forecast 0.01 → tax_drag = 0.37 * 10_000 * 0.01 = 37; floor = 10 + 37 = 47
    # Test from mission file: forecast 0.01, but expected floor=380 → that's
    # 0.10 forecast. Re-derive: 2*5 + 0.37*10000*0.01 = 10 + 37 = 47.
    # The mission's "floor=380" assumes forecast 0.10 (10% return at h20).
    # Use 0.10 to match the mission's stated number.
    v_h20 = _make_view(primary_horizon="h20", h20=0.10)
    # alpha=100bps < floor=380 → reject
    o_low = _make_order(
        side="BUY", primary_horizon="h20", size_pct=30.0,
        expected_alpha_bps=100.0,
    )
    r = validate_edge_floor(o_low, v_h20)
    assert not r.accepted
    assert "edge_below_floor" in r.reason

    # Same numbers but primary_horizon=h60 → waived. Need a view + order at h60.
    v_h60 = _make_view(primary_horizon="h60", h60=0.10)
    o_h60 = _make_order(
        side="BUY", primary_horizon="h60", size_pct=50.0,
        expected_alpha_bps=100.0,
    )
    r = validate_edge_floor(o_h60, v_h60)
    assert r.accepted
    assert "h60" in (r.reason or "")

    # h20 alpha=500bps > 380 floor → accept
    o_hi = _make_order(
        side="BUY", primary_horizon="h20", size_pct=30.0,
        expected_alpha_bps=500.0,
    )
    r = validate_edge_floor(o_hi, v_h20)
    assert r.accepted

    # Sanity: alpha at exactly floor should accept (strict <)
    # floor = 10 + 0.37*10000*0.10 = 380
    o_boundary = _make_order(
        side="BUY", primary_horizon="h20", size_pct=30.0,
        expected_alpha_bps=380.0,
    )
    r = validate_edge_floor(o_boundary, v_h20)
    assert r.accepted


def test_edge_floor_smaller_forecast():
    """Mission-file numbers with forecast=0.01: floor = 10 + 37 = 47bps."""
    v = _make_view(primary_horizon="h20", h20=0.01)
    o_below = _make_order(
        side="BUY", primary_horizon="h20", size_pct=30.0,
        expected_alpha_bps=40.0,
    )
    r = validate_edge_floor(o_below, v)
    assert not r.accepted

    o_above = _make_order(
        side="BUY", primary_horizon="h20", size_pct=30.0,
        expected_alpha_bps=50.0,
    )
    r = validate_edge_floor(o_above, v)
    assert r.accepted


# ── 5. Chain composition + ValidatorResult shape ────────────────────

def test_validator_result_shape():
    r = ValidatorResult(accepted=False, reason="reason", suggested_horizon="h20")
    blob = r.model_dump()
    assert blob == {"accepted": False, "reason": "reason", "suggested_horizon": "h20"}
    assert ValidatorResult.model_validate(blob).suggested_horizon == "h20"
    # Defaults
    r2 = ValidatorResult(accepted=True)
    assert r2.reason == ""
    assert r2.suggested_horizon is None


def test_chain_short_circuits_on_first_rejection():
    """An order failing min-hold should not also report sizing or floor."""
    cal = _make_calendar(n_days=30)
    pos = _make_position(primary_horizon="h20", opened_on=cal[0])
    today = cal[2]
    # Bad on every axis: SELL too early, wrong sign forecast, bad size, bad alpha.
    bad = _make_order(
        side="SELL", primary_horizon="h20", size_pct=50.0,
        expected_alpha_bps=10.0,
    )
    v_bad = _make_view(h20=0.05)  # positive forecast contradicts SELL
    result = run_tax_aware_chain(bad, [pos], today, cal, [v_bad])
    assert not result.accepted
    # min_hold is first in the chain
    assert "min_hold_violated" in result.reason


# ── 6. Position-horizon tracking through ApexPortfolio.execute ──────

def test_position_stores_primary_horizon_after_buy():
    """BUY through execute() must persist primary_horizon on the position."""
    pf = ApexPortfolio(label="t", starting_cash=100_000.0)
    pf.advance_prices("2026-01-01", [{"ticker": "AAPL", "actual_return": 0.0}])
    pf.execute("2026-01-01", [{
        "ticker": "AAPL", "side": "BUY",
        "size_pct": 30.0, "dollars_intent": 30_000.0,
        "reasoning": "test", "primary_horizon": "h20",
        "expected_alpha_bps": 250.0,
    }], {"AAPL": 100.0})
    pos = pf.positions.get("AAPL")
    assert pos is not None
    assert pos.primary_horizon == "h20"
    assert pos.bought_at_date == "2026-01-01"
    assert pos.opened_on == "2026-01-01"
    assert pos.ticker == "AAPL"


def test_rebuy_does_not_reset_primary_horizon():
    """A second BUY on an existing position must NOT overwrite the original
    horizon — otherwise the apex could game min_hold by refreshing the clock.
    """
    pf = ApexPortfolio(label="t", starting_cash=100_000.0)
    pf.advance_prices("2026-01-01", [{"ticker": "AAPL", "actual_return": 0.0}])
    pf.execute("2026-01-01", [{
        "ticker": "AAPL", "side": "BUY",
        "size_pct": 30.0, "dollars_intent": 30_000.0,
        "reasoning": "test", "primary_horizon": "h20",
        "expected_alpha_bps": 250.0,
    }], {"AAPL": 100.0})
    pf.advance_prices("2026-01-02", [{"ticker": "AAPL", "actual_return": 0.0}])
    pf.execute("2026-01-02", [{
        "ticker": "AAPL", "side": "BUY",
        "size_pct": 3.0, "dollars_intent": 3_000.0,
        "reasoning": "add probe", "primary_horizon": "h1",
        "expected_alpha_bps": 50.0,
    }], {"AAPL": 100.0})
    pos = pf.positions["AAPL"]
    # Original h20 commitment persists despite the h1 add-on.
    assert pos.primary_horizon == "h20"
    assert pos.bought_at_date == "2026-01-01"


def test_filter_tax_aware_round_trip():
    """ApexPortfolio.filter_tax_aware reads its own positions' horizons
    and rejects a too-early SELL via the run_firehose_loop integration path.
    """
    pf = ApexPortfolio(label="t", starting_cash=100_000.0)
    pf.advance_prices("2026-01-01", [{"ticker": "AAPL", "actual_return": 0.0}])
    pf.execute("2026-01-01", [{
        "ticker": "AAPL", "side": "BUY",
        "size_pct": 30.0, "dollars_intent": 30_000.0,
        "reasoning": "open core", "primary_horizon": "h20",
        "expected_alpha_bps": 250.0,
    }], {"AAPL": 100.0})
    # Tick forward 3 days
    for d in ("2026-01-02", "2026-01-03", "2026-01-04"):
        pf.advance_prices(d, [{"ticker": "AAPL", "actual_return": 0.0}])

    # Validated SELL the runner would normally feed in
    validated = [{
        "ticker": "AAPL", "side": "SELL",
        "size_pct": 100.0, "dollars_intent": 30_000.0,
        "reasoning": "early exit",
        "primary_horizon": "h20",
        "expected_alpha_bps": -250.0,
    }]
    v = _make_view(ticker="AAPL", primary_horizon="h20", h20=-0.02)
    accepted, rejections = pf.filter_tax_aware(validated, "2026-01-04", [v])
    assert accepted == []
    assert len(rejections) == 1
    assert "min_hold_violated" in rejections[0]


def test_filter_tax_aware_accepts_regime_invalidation():
    """The same too-early SELL is accepted when the view's regime_note
    explicitly invalidates."""
    pf = ApexPortfolio(label="t", starting_cash=100_000.0)
    pf.advance_prices("2026-01-01", [{"ticker": "AAPL", "actual_return": 0.0}])
    pf.execute("2026-01-01", [{
        "ticker": "AAPL", "side": "BUY",
        "size_pct": 30.0, "dollars_intent": 30_000.0,
        "reasoning": "open core", "primary_horizon": "h20",
        "expected_alpha_bps": 250.0,
    }], {"AAPL": 100.0})
    for d in ("2026-01-02", "2026-01-03", "2026-01-04"):
        pf.advance_prices(d, [{"ticker": "AAPL", "actual_return": 0.0}])
    validated = [{
        "ticker": "AAPL", "side": "SELL",
        "size_pct": 100.0, "dollars_intent": 30_000.0,
        "reasoning": "thesis broken",
        "primary_horizon": "h20",
        "expected_alpha_bps": -250.0,
    }]
    v = _make_view(
        ticker="AAPL", primary_horizon="h20", h20=-0.02,
        regime_note="event_driven_invalidation",
    )
    accepted, rejections = pf.filter_tax_aware(validated, "2026-01-04", [v])
    assert len(accepted) == 1
    assert rejections == []


# ── 7. Universe membership (defensive layer over phase-05a filter) ──

# A small Dow-30-ish universe for these tests — the validator only cares
# about the membership semantics, not the specific tickers.
_DOW30_FIXTURE: tuple[str, ...] = (
    "AAPL", "MSFT", "JPM", "JNJ", "V", "PG", "HD", "CVX", "MRK", "KO",
    "WMT", "DIS", "MCD", "CSCO", "VZ", "INTC", "BA", "CAT", "GS", "AXP",
    "IBM", "NKE", "MMM", "TRV", "UNH", "CRM", "HON", "AMGN", "WBA", "DOW",
)


def test_validate_in_universe_accepts_in_universe():
    """BUY AAPL with DOW30 universe → accept."""
    o = _make_order(side="BUY", ticker="AAPL", primary_horizon="h20")
    r = validate_in_universe(o, _DOW30_FIXTURE)
    assert r.accepted, r.reason


def test_validate_in_universe_rejects_off_universe():
    """BUY XOM with DOW30 universe → reject with ticker_not_in_universe."""
    o = _make_order(side="BUY", ticker="XOM", primary_horizon="h20")
    r = validate_in_universe(o, _DOW30_FIXTURE)
    assert not r.accepted
    assert "ticker_not_in_universe" in r.reason
    assert "XOM" in r.reason


def test_validate_in_universe_case_insensitive():
    """BUY 'aapl' (lowercase) with DOW30 universe → accept."""
    o = _make_order(side="BUY", ticker="aapl", primary_horizon="h20")
    r = validate_in_universe(o, _DOW30_FIXTURE)
    assert r.accepted, r.reason


def test_validate_in_universe_hold_passes():
    """HOLD on any ticker (in OR out of universe) → accept; HOLD doesn't trade."""
    o_in = _make_order(side="HOLD", ticker="AAPL", primary_horizon="h20")
    o_out = _make_order(side="HOLD", ticker="XOM", primary_horizon="h20")
    assert validate_in_universe(o_in, _DOW30_FIXTURE).accepted
    assert validate_in_universe(o_out, _DOW30_FIXTURE).accepted


def test_validate_in_universe_rotate_both_legs():
    """ROTATE with in-universe from_ticker but off-universe to_ticker → reject."""
    o = Order(
        side="ROTATE",
        from_ticker="AAPL",
        to_ticker="XOM",
        size_pct=30.0,
        reasoning="rotate AAPL → XOM",
        primary_horizon="h20",
        expected_alpha_bps=200.0,
    )
    r = validate_in_universe(o, _DOW30_FIXTURE)
    assert not r.accepted
    assert "ticker_not_in_universe" in r.reason
    assert "XOM" in r.reason

    # And vice-versa: in-universe to_ticker with off-universe from_ticker.
    o2 = Order(
        side="ROTATE",
        from_ticker="XOM",
        to_ticker="AAPL",
        size_pct=30.0,
        reasoning="rotate XOM → AAPL",
        primary_horizon="h20",
        expected_alpha_bps=200.0,
    )
    r2 = validate_in_universe(o2, _DOW30_FIXTURE)
    assert not r2.accepted
    assert "ticker_not_in_universe" in r2.reason
    assert "XOM" in r2.reason

    # Happy path: ROTATE AAPL → MSFT (both in universe).
    o3 = Order(
        side="ROTATE",
        from_ticker="AAPL",
        to_ticker="MSFT",
        size_pct=30.0,
        reasoning="rotate AAPL → MSFT",
        primary_horizon="h20",
        expected_alpha_bps=200.0,
    )
    assert validate_in_universe(o3, _DOW30_FIXTURE).accepted


def test_validate_in_universe_universe_none_skipped():
    """Passing universe=None → validator is skipped (backward-compat)."""
    o = _make_order(side="BUY", ticker="XOM", primary_horizon="h20")
    r = validate_in_universe(o, None)
    assert r.accepted
    assert r.reason == ""


def test_chain_universe_first():
    """When run_tax_aware_chain is called with universe set, the first
    rejection on an off-universe ticker has reason starting with
    'ticker_not_in_universe' — proving the universe validator runs FIRST.

    Construct an order that is bad on multiple axes (off-universe AND
    sized off-tier AND with a contradicting forecast). The chain must
    short-circuit on the universe check.
    """
    cal = _make_calendar(n_days=30)
    today = cal[2]
    # Off-universe, bad sizing (h5 base 10, this is 50), bad forecast sign.
    bad = _make_order(
        side="BUY", ticker="XOM", primary_horizon="h5",
        size_pct=50.0, expected_alpha_bps=5.0,
    )
    v_neg = _make_view(ticker="XOM", primary_horizon="h5", h5=-0.05)
    result = run_tax_aware_chain(
        bad, [], today, cal, [v_neg], universe=_DOW30_FIXTURE,
    )
    assert not result.accepted
    assert result.reason.startswith("ticker_not_in_universe")


def test_filter_tax_aware_universe_blocks_off_universe():
    """ApexPortfolio.filter_tax_aware with universe set rejects off-universe BUY."""
    pf = ApexPortfolio(label="t", starting_cash=100_000.0)
    pf.advance_prices("2026-01-01", [{"ticker": "XOM", "actual_return": 0.0}])
    validated = [{
        "ticker": "XOM", "side": "BUY",
        "size_pct": 30.0, "dollars_intent": 30_000.0,
        "reasoning": "off-universe hallucination",
        "primary_horizon": "h20",
        "expected_alpha_bps": 250.0,
    }]
    v = _make_view(ticker="XOM", primary_horizon="h20", h20=0.02)
    accepted, rejections = pf.filter_tax_aware(
        validated, "2026-01-01", [v], universe=_DOW30_FIXTURE,
    )
    assert accepted == []
    assert len(rejections) == 1
    assert "ticker_not_in_universe" in rejections[0]
    assert "XOM" in rejections[0]


def test_filter_tax_aware_universe_none_is_legacy():
    """ApexPortfolio.filter_tax_aware with universe=None preserves legacy
    behaviour — off-universe tickers pass the universe layer (other
    validators still apply).
    """
    pf = ApexPortfolio(label="t", starting_cash=100_000.0)
    pf.advance_prices("2026-01-01", [{"ticker": "XOM", "actual_return": 0.0}])
    validated = [{
        "ticker": "XOM", "side": "BUY",
        "size_pct": 30.0, "dollars_intent": 30_000.0,
        "reasoning": "legacy path",
        "primary_horizon": "h20",
        "expected_alpha_bps": 250.0,
    }]
    v = _make_view(ticker="XOM", primary_horizon="h20", h20=0.02)
    accepted, rejections = pf.filter_tax_aware(
        validated, "2026-01-01", [v],  # no universe arg
    )
    # Universe check skipped → other validators decide; with this happy
    # config (forecast positive, alpha well over floor) it accepts.
    assert len(accepted) == 1
    assert rejections == []
