"""Phase 3-A-05c (v4): PredatorSubPortfolio + ApexPortfolio debate mode.

Tests the per-predator capital + book layering on top of ApexPortfolio:
  - construction via `make_debate_portfolio` (4 predators @ $25K each)
  - aggregate equity / cash / invested_pct / tax_owed sum across slices
  - per-predator tax isolation (the load-bearing invariant)
  - `state_for_predator()` scoped to one predator's view
  - legacy non-debate ApexPortfolio is byte-identical (regression guard)
"""
from __future__ import annotations

import pytest

from trophic.agents.apex_portfolio import (
    ApexPortfolio,
    DEFAULT_DEBATE_PREDATORS,
    _Position,
    make_debate_portfolio,
)
from trophic.agents.predator_sub_portfolio import PredatorSubPortfolio
from trophic.beliefs.apex_signatures import PortfolioState, PositionSnapshot
from trophic.beliefs.investment_thesis import InvestmentThesis, ThesisBook


# ── 1. Construction ────────────────────────────────────────────────────


def test_make_debate_portfolio_four_predators_25k_each():
    p = make_debate_portfolio(total_starting_cash=100_000)
    assert p.is_debate_mode()
    assert len(p.sub_portfolios) == 4
    assert list(p.sub_portfolios.keys()) == [
        "momentum", "value", "mean_revert", "event_driven",
    ]
    for sub in p.sub_portfolios.values():
        assert sub.starting_cash == 25_000.0
        assert sub.cash == 25_000.0
        assert sub.tax_owed_accrued == 0.0
        assert sub.positions == {}
        assert isinstance(sub.thesis_book, ThesisBook)
        assert sub.thesis_book.predator_id == sub.predator_id


def test_predator_sub_portfolio_thesis_book_predator_id_autowires():
    sub = PredatorSubPortfolio(
        predator_id="value", philosophy="value", starting_cash=25_000.0,
    )
    assert sub.thesis_book.predator_id == "value"
    assert sub.cash == 25_000.0


def test_predator_sub_portfolio_thesis_book_rejects_cross_predator():
    """ThesisBook enforces predator_id match on add()."""
    sub = PredatorSubPortfolio(
        predator_id="momentum", philosophy="momentum", starting_cash=25_000,
    )
    foreign = InvestmentThesis(
        predator_id="value",  # wrong owner
        philosophy="value",
        ticker="AAPL",
        opened_at_date="2026-01-01",
        direction="long",
        primary_horizon="h20",
        expected_alpha_bps=500.0,
        confidence="med",
    )
    with pytest.raises(ValueError, match="predator_id"):
        sub.thesis_book.add(foreign)


# ── 2. Equity aggregation ─────────────────────────────────────────────


def test_equity_aggregates():
    """4 subs @ $25K cash → aggregate $100K. After AAPL @$200 BUY in one sub,
    aggregate equity unchanged. After AAPL → $210, aggregate equity rises by
    the position's appreciation (only in that sub)."""
    p = make_debate_portfolio(total_starting_cash=100_000)
    prices = {"AAPL": 200.0}
    assert p.equity(prices) == pytest.approx(100_000.0)

    mom = p.sub_portfolios["momentum"]
    # Buy 50 shares of AAPL at $200 = $10K position. Slippage=0 to keep
    # the cash-conservation arithmetic clean for this test.
    mom.buy(
        ticker="AAPL", dollars=10_000.0, slip_bps=0.0, today="2026-01-02",
        primary_horizon="h20", price=200.0,
    )
    # Sub equity: cash $15K + position $10K = $25K. Aggregate still $100K.
    assert mom.cash == pytest.approx(15_000.0)
    assert p.equity(prices) == pytest.approx(100_000.0)

    # Tick price → $210. Only momentum's slice benefits.
    prices2 = {"AAPL": 210.0}
    # 50 shares * $210 = $10,500 → momentum slice +$500.
    assert mom.equity(prices2) == pytest.approx(25_500.0)
    assert p.equity(prices2) == pytest.approx(100_500.0)


def test_invested_pct_aggregates():
    p = make_debate_portfolio(total_starting_cash=100_000)
    prices = {"AAPL": 100.0, "MSFT": 100.0}
    # Two predators each go ~$10K into one ticker.
    p.sub_portfolios["momentum"].buy(
        "AAPL", 10_000.0, 0.0, "2026-01-02", primary_horizon="h20", price=100.0,
    )
    p.sub_portfolios["value"].buy(
        "MSFT", 10_000.0, 0.0, "2026-01-02", primary_horizon="h60", price=100.0,
    )
    # Held = $20K, aggregate eq = $100K → invested = 20%.
    assert p.invested_pct(prices) == pytest.approx(20.0)


def test_total_cash_and_tax_owed_aggregate():
    p = make_debate_portfolio(total_starting_cash=100_000)
    assert p.total_cash == pytest.approx(100_000.0)
    assert p.total_tax_owed_accrued == 0.0
    # Manually credit tax to a couple slices.
    p.sub_portfolios["momentum"].tax_owed_accrued = 370.0
    p.sub_portfolios["event_driven"].tax_owed_accrued = 130.0
    assert p.total_tax_owed_accrued == pytest.approx(500.0)


# ── 3. Per-predator tax isolation (THE load-bearing invariant) ─────────


def test_per_predator_tax_isolation():
    """Predator A realizes $1000 gain → A's tax_owed_accrued goes up by $370;
    B/C/D unchanged. Aggregate tax = $370."""
    p = make_debate_portfolio(total_starting_cash=100_000)
    mom = p.sub_portfolios["momentum"]
    val = p.sub_portfolios["value"]
    mr = p.sub_portfolios["mean_revert"]
    ev = p.sub_portfolios["event_driven"]

    # Buy AAPL @ $100, 100 shares = $10K position in momentum (no slip).
    mom.buy("AAPL", 10_000.0, 0.0, "2026-01-02",
            primary_horizon="h20", price=100.0)
    assert mom.cash == pytest.approx(15_000.0)
    # Price doubles → sell entire position at $200. Realized = $20K - $10K = $10K.
    # But mission spec exemplar is +$1000 realized → +$370 tax.
    # Use a smaller move so realized lands at exactly $1000.
    # Position: 100 shares @ cost $10,000. Sell at $110 → proceeds $11,000,
    # realized = $1,000.
    proceeds_net, pnl = mom.sell(
        "AAPL", fraction_of_position=1.0, slip_bps=0.0,
        tax_fraction=0.37, today="2026-01-03", price=110.0,
    )
    assert pnl == pytest.approx(1_000.0)
    assert mom.tax_owed_accrued == pytest.approx(370.0)

    # Isolation: only momentum's tax moved.
    assert val.tax_owed_accrued == 0.0
    assert mr.tax_owed_accrued == 0.0
    assert ev.tax_owed_accrued == 0.0

    # Aggregate = $370.
    assert p.total_tax_owed_accrued == pytest.approx(370.0)


def test_sell_no_gain_no_tax():
    """SELL with no realized gain accrues no tax (mirrors ApexPortfolio.execute)."""
    p = make_debate_portfolio(total_starting_cash=100_000)
    mom = p.sub_portfolios["momentum"]
    mom.buy("AAPL", 10_000.0, 0.0, "2026-01-02",
            primary_horizon="h20", price=100.0)
    # Sell at same price → realized ≈ 0.
    _, pnl = mom.sell(
        "AAPL", 1.0, slip_bps=0.0, tax_fraction=0.37,
        today="2026-01-03", price=100.0,
    )
    assert pnl == pytest.approx(0.0)
    assert mom.tax_owed_accrued == 0.0
    # SELL at a loss → realized < 0, no tax credit (matches ApexPortfolio).
    mom.buy("AAPL", 5_000.0, 0.0, "2026-01-04",
            primary_horizon="h20", price=100.0)
    _, pnl2 = mom.sell(
        "AAPL", 1.0, slip_bps=0.0, tax_fraction=0.37,
        today="2026-01-05", price=80.0,
    )
    assert pnl2 < 0
    assert mom.tax_owed_accrued == 0.0


# ── 4. state_for_predator ─────────────────────────────────────────────


def test_state_for_predator_scoped():
    """state_for_predator returns a PortfolioState reflecting ONE predator only."""
    p = make_debate_portfolio(total_starting_cash=100_000)
    p.sub_portfolios["momentum"].buy(
        "AAPL", 5_000.0, 0.0, "2026-01-02",
        primary_horizon="h20", price=100.0,
    )
    p.sub_portfolios["value"].buy(
        "MSFT", 10_000.0, 0.0, "2026-01-02",
        primary_horizon="h60", price=100.0,
    )
    prices = {"AAPL": 100.0, "MSFT": 100.0}
    # Register the date in the parent calendar so days_held works.
    p._dates_seen.append("2026-01-02")

    state_mom, pos_mom = p.state_for_predator("momentum", "2026-01-02", prices)
    state_val, pos_val = p.state_for_predator("value", "2026-01-02", prices)

    assert isinstance(state_mom, PortfolioState)
    # Momentum slice: cash $20K + AAPL $5K = $25K equity, invested 20%.
    assert state_mom.equity == pytest.approx(25_000.0)
    assert state_mom.cash == pytest.approx(20_000.0)
    assert state_mom.invested_pct == pytest.approx(20.0)
    assert len(pos_mom) == 1
    assert pos_mom[0].ticker == "AAPL"
    assert isinstance(pos_mom[0], PositionSnapshot)

    # Value slice: cash $15K + MSFT $10K = $25K equity, invested 40%.
    assert state_val.equity == pytest.approx(25_000.0)
    assert state_val.cash == pytest.approx(15_000.0)
    assert state_val.invested_pct == pytest.approx(40.0)
    assert len(pos_val) == 1
    assert pos_val[0].ticker == "MSFT"

    # Each predator does NOT see other predators' positions.
    assert all(p_.ticker != "MSFT" for p_ in pos_mom)
    assert all(p_.ticker != "AAPL" for p_ in pos_val)


def test_state_for_predator_rejects_non_debate_mode():
    legacy = ApexPortfolio(label="legacy")
    with pytest.raises(ValueError, match="debate mode"):
        legacy.state_for_predator("momentum", "2026-01-02", {})


def test_state_for_predator_rejects_unknown_predator():
    p = make_debate_portfolio(total_starting_cash=100_000)
    with pytest.raises(ValueError, match="unknown predator_id"):
        p.state_for_predator("nonexistent", "2026-01-02", {})


# ── 5. Backward compat (regression guard) ─────────────────────────────


def test_non_debate_legacy_unchanged():
    """An ApexPortfolio without sub_portfolios behaves identically to v3.3/v4.

    The 188+ pre-existing tests are the real proof; this is a smoke
    check that the constructor / equity / state_for_apex shape didn't
    drift."""
    legacy = ApexPortfolio(label="legacy", starting_cash=50_000)
    assert not legacy.is_debate_mode()
    assert legacy.sub_portfolios is None
    assert legacy.cash == 50_000.0
    assert legacy.total_cash == 50_000.0
    assert legacy.equity({}) == 50_000.0
    assert legacy.invested_pct({}) == 0.0
    # state_for_apex still works.
    legacy._dates_seen.append("2026-01-02")
    state, positions = legacy.state_for_apex("2026-01-02", {})
    assert isinstance(state, PortfolioState)
    assert state.equity == 50_000.0
    assert positions == []


def test_position_thesis_id_optional_default_empty():
    """_Position.thesis_id defaults to empty string for legacy positions."""
    pos = _Position(shares=10.0, cost_basis=1000.0, ticker="AAPL")
    assert pos.thesis_id == ""
    pos2 = _Position(
        shares=10.0, cost_basis=1000.0, ticker="AAPL", thesis_id="abc123",
    )
    assert pos2.thesis_id == "abc123"


def test_predator_buy_does_not_reset_horizon_on_rebuy():
    """Gameability guard: a re-buy on an open position must NOT reset
    primary_horizon or bought_at_date (mirrors phase2-D contract on
    ApexPortfolio.execute)."""
    sub = PredatorSubPortfolio(
        predator_id="momentum", philosophy="momentum", starting_cash=25_000,
    )
    sub.buy("AAPL", 5_000.0, 0.0, "2026-01-02",
            primary_horizon="h60", price=100.0)
    assert sub.positions["AAPL"].primary_horizon == "h60"
    assert sub.positions["AAPL"].bought_at_date == "2026-01-02"

    # Re-buy with a SHORTER horizon — must not stick.
    sub.buy("AAPL", 2_000.0, 0.0, "2026-01-10",
            primary_horizon="h5", price=110.0)
    assert sub.positions["AAPL"].primary_horizon == "h60"
    assert sub.positions["AAPL"].bought_at_date == "2026-01-02"


def test_default_debate_predators_constant_shape():
    """The factory's default predator tuple is the 4-predator setup."""
    assert len(DEFAULT_DEBATE_PREDATORS) == 4
    ids = [pid for pid, _ in DEFAULT_DEBATE_PREDATORS]
    assert ids == ["momentum", "value", "mean_revert", "event_driven"]


def test_recent_realized_pnl_5d_windowed():
    sub = PredatorSubPortfolio(
        predator_id="momentum", philosophy="momentum", starting_cash=25_000,
    )
    # No history → 0.
    assert sub.recent_realized_pnl_5d("2026-01-10") == 0.0
    # Append 7 days; only last 5 count toward the window.
    for i, v in enumerate([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]):
        sub.realized_pnl_history.append((f"2026-01-{i+1:02d}", v))
    # Sum of last 5: 30+40+50+60+70 = 250.
    assert sub.recent_realized_pnl_5d("2026-01-07") == pytest.approx(250.0)


def test_rotate_atomic_within_sub():
    """Rotate within one sub: SELL leg credits cash + tax, BUY leg consumes it.
    No other sub is touched."""
    p = make_debate_portfolio(total_starting_cash=100_000)
    mom = p.sub_portfolios["momentum"]
    val = p.sub_portfolios["value"]
    mom.buy("AAPL", 10_000.0, 0.0, "2026-01-02",
            primary_horizon="h20", price=100.0)
    # Price doubles, then rotate AAPL → MSFT fully.
    proceeds, pnl = mom.rotate(
        from_ticker="AAPL", to_ticker="MSFT", fraction_of_from=1.0,
        slip_bps=0.0, tax_fraction=0.37, today="2026-01-03",
        primary_horizon="h60",
        from_price=200.0, to_price=100.0,
    )
    assert pnl == pytest.approx(10_000.0)
    assert mom.tax_owed_accrued == pytest.approx(3700.0)
    assert "AAPL" not in mom.positions
    assert "MSFT" in mom.positions
    # Other sub untouched.
    assert val.cash == 25_000.0
    assert val.tax_owed_accrued == 0.0
    assert val.positions == {}
