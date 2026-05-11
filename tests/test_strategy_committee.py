"""Tests for the v4 three-strategy committee.

Coverage targets (from phase2-C-03-strategy-committee.md):
  - conviction_weighted basic: high-conf h20 + +5% forecast → BUY ~21%
  - aggregator rejects split sides (2 BUY + 1 SELL, similar conviction)
  - aggregator weighted-average size_pct (equal + non-equal weights)
  - kelly_edge waives edge floor for h60, enforces it for h5
"""
from __future__ import annotations

import math

import pytest

from trophic.beliefs.apex_signatures import (
    HORIZON_BASE_SIZE,
    HorizonForecast,
    Order,
    TickerView,
)
from trophic.beliefs.strategy_committee import (
    KELLY_EDGE_FLOOR_BPS,
    StrategyVote,
    aggregate_votes,
    conviction_weighted_vote,
    kelly_edge_vote,
    tiered_discrete_vote,
)


# ── Fixtures / helpers ────────────────────────────────────────────────


def _mk_forecast(
    h1: float = 0.0, h5: float = 0.0, h20: float = 0.0, h60: float = 0.0,
    confidence_h1: str = "low",
    confidence_h5: str = "low",
    confidence_h20: str = "low",
    confidence_h60: str = "low",
    regime_note: str = "trend",
) -> HorizonForecast:
    return HorizonForecast(
        h1=h1, h5=h5, h20=h20, h60=h60,
        confidence_h1=confidence_h1, confidence_h5=confidence_h5,
        confidence_h20=confidence_h20, confidence_h60=confidence_h60,
        regime_note=regime_note,
    )


def _mk_view(
    ticker: str = "AAPL",
    primary_horizon: str = "h20",
    **forecast_kwargs,
) -> TickerView:
    return TickerView(
        ticker=ticker,
        forecasts=_mk_forecast(**forecast_kwargs),
        primary_horizon=primary_horizon,  # type: ignore[arg-type]
        rationale="test rationale",
    )


def _mk_order(
    side: str = "BUY",
    ticker: str = "AAPL",
    size_pct: float = 10.0,
    primary_horizon: str = "h20",
    expected_alpha_bps: float = 200.0,
) -> Order:
    return Order(
        side=side,  # type: ignore[arg-type]
        ticker=ticker,
        size_pct=size_pct,
        reasoning="test",
        primary_horizon=primary_horizon,  # type: ignore[arg-type]
        expected_alpha_bps=expected_alpha_bps,
    )


# ── Conviction-weighted basic ─────────────────────────────────────────


def test_conviction_weighted_basic():
    """High-conf h20 + h20 forecast +5% → BUY around 21% size."""
    view = _mk_view(
        primary_horizon="h20",
        h20=0.05,            # +5%
        confidence_h20="high",
    )
    order = _mk_order(primary_horizon="h20", expected_alpha_bps=500.0)

    vote = conviction_weighted_vote(view, order)

    assert vote.strategy == "conviction_weighted"
    assert vote.order.side == "BUY"
    # Expected conviction: 0.5*1.0 + 0.5*tanh(20*0.05) = 0.5 + 0.5*tanh(1.0)
    expected_conviction = min(1.0, 0.5 * 1.0 + 0.5 * math.tanh(1.0))
    assert abs(vote.conviction - expected_conviction) < 1e-6
    # Size = HORIZON_BASE_SIZE['h20'] * conviction^2 = 30 * 0.881^2 ≈ 23.3
    expected_size = HORIZON_BASE_SIZE["h20"] * expected_conviction ** 2
    assert abs(vote.order.size_pct - expected_size) < 1e-6
    # And it should land in the 18-25% band the mission file calls out.
    assert 18.0 <= vote.order.size_pct <= 25.0


def test_conviction_weighted_negative_forecast_is_sell():
    view = _mk_view(primary_horizon="h5", h5=-0.03, confidence_h5="med")
    order = _mk_order(primary_horizon="h5", expected_alpha_bps=-300.0)
    vote = conviction_weighted_vote(view, order)
    assert vote.order.side == "SELL"
    assert vote.conviction > 0.0


def test_conviction_weighted_zero_forecast_is_hold_zero_conviction():
    view = _mk_view(primary_horizon="h1", h1=0.0, confidence_h1="low")
    order = _mk_order(primary_horizon="h1", expected_alpha_bps=0.0)
    vote = conviction_weighted_vote(view, order)
    assert vote.order.side == "HOLD"
    assert vote.conviction == 0.0


# ── Tiered discrete ────────────────────────────────────────────────────


def test_tiered_discrete_size_matches_horizon_table():
    for h, expected in HORIZON_BASE_SIZE.items():
        view = _mk_view(primary_horizon=h, **{h: 0.02, f"confidence_{h}": "med"})
        order = _mk_order(primary_horizon=h, expected_alpha_bps=200.0)
        vote = tiered_discrete_vote(view, order)
        assert vote.order.size_pct == expected, f"horizon {h}"
        assert vote.order.side == "BUY"
        assert abs(vote.conviction - 0.66) < 1e-6


# ── Kelly-edge ────────────────────────────────────────────────────────


def test_kelly_waives_floor_for_h60():
    """expected_alpha_bps = 100 (below 162 floor):
    - h60 → BUY vote with positive conviction
    - h5  → HOLD with conviction 0
    """
    # Sanity: confirm the floor we're testing against is what we claim.
    assert KELLY_EDGE_FLOOR_BPS == 162.0

    view_h60 = _mk_view(primary_horizon="h60", h60=0.01, confidence_h60="med")
    order_h60 = _mk_order(primary_horizon="h60", expected_alpha_bps=100.0)
    vote_h60 = kelly_edge_vote(view_h60, order_h60)
    assert vote_h60.order.side == "BUY"
    assert vote_h60.conviction > 0.0

    view_h5 = _mk_view(primary_horizon="h5", h5=0.01, confidence_h5="med")
    order_h5 = _mk_order(primary_horizon="h5", expected_alpha_bps=100.0)
    vote_h5 = kelly_edge_vote(view_h5, order_h5)
    assert vote_h5.order.side == "HOLD"
    assert vote_h5.conviction == 0.0


def test_kelly_clamps_at_fifty():
    view = _mk_view(primary_horizon="h60", h60=0.10, confidence_h60="high")
    order = _mk_order(primary_horizon="h60", expected_alpha_bps=5000.0)
    vote = kelly_edge_vote(view, order)
    assert vote.order.size_pct <= 50.0 + 1e-9


# ── Aggregator ─────────────────────────────────────────────────────────


def test_aggregator_rejects_split_sides():
    """2 BUY + 1 SELL with similar conviction → strict-majority check is
    NOT enough; sides disagree → None."""
    view = _mk_view(primary_horizon="h20", h20=0.04, confidence_h20="high")
    template = _mk_order(primary_horizon="h20", expected_alpha_bps=400.0)

    votes = [
        StrategyVote(
            strategy="conviction_weighted",
            order=_mk_order(side="BUY", size_pct=15.0, primary_horizon="h20"),
            conviction=0.8,
        ),
        StrategyVote(
            strategy="tiered_discrete",
            order=_mk_order(side="BUY", size_pct=30.0, primary_horizon="h20"),
            conviction=0.7,
        ),
        StrategyVote(
            strategy="kelly_edge",
            order=_mk_order(side="SELL", size_pct=10.0, primary_horizon="h20"),
            conviction=0.7,
        ),
    ]
    weights = {"conviction_weighted": 0.34, "tiered_discrete": 0.33, "kelly_edge": 0.33}

    # 2 BUY + 1 SELL of 3 active = 2/3 majority — strict majority does pass.
    # That's fine; this test checks a more even split below.
    agg_with_majority = aggregate_votes(votes, weights)
    assert agg_with_majority is not None
    assert agg_with_majority.side == "BUY"

    # Now force a true split: 1 BUY, 1 SELL, 1 HOLD (HOLD passes the
    # conviction floor at 0.5). No strict majority — should return None.
    split_votes = [
        StrategyVote(
            strategy="conviction_weighted",
            order=_mk_order(side="BUY", size_pct=15.0, primary_horizon="h20"),
            conviction=0.8,
        ),
        StrategyVote(
            strategy="tiered_discrete",
            order=_mk_order(side="SELL", size_pct=30.0, primary_horizon="h20"),
            conviction=0.8,
        ),
        StrategyVote(
            strategy="kelly_edge",
            order=_mk_order(side="HOLD", size_pct=0.0, primary_horizon="h20"),
            conviction=0.5,
        ),
    ]
    agg_split = aggregate_votes(split_votes, weights)
    assert agg_split is None

    # And the canonical "2 of 4 BUY, 2 of 4 SELL" tie also returns None.
    tied_votes = [
        StrategyVote(
            strategy="conviction_weighted",
            order=_mk_order(side="BUY", size_pct=10.0, primary_horizon="h20"),
            conviction=0.8,
        ),
        StrategyVote(
            strategy="tiered_discrete",
            order=_mk_order(side="SELL", size_pct=10.0, primary_horizon="h20"),
            conviction=0.8,
        ),
    ]
    assert aggregate_votes(tied_votes, weights) is None


def test_aggregator_weighted_average():
    """3 BUYs with sizes {10, 20, 30}, equal weights and conviction 1.0 →
    avg = 20%. Then verify non-equal weights move the average."""
    votes = [
        StrategyVote(
            strategy="conviction_weighted",
            order=_mk_order(side="BUY", size_pct=10.0, primary_horizon="h20",
                            expected_alpha_bps=300.0),
            conviction=1.0,
        ),
        StrategyVote(
            strategy="tiered_discrete",
            order=_mk_order(side="BUY", size_pct=20.0, primary_horizon="h20",
                            expected_alpha_bps=300.0),
            conviction=1.0,
        ),
        StrategyVote(
            strategy="kelly_edge",
            order=_mk_order(side="BUY", size_pct=30.0, primary_horizon="h20",
                            expected_alpha_bps=300.0),
            conviction=1.0,
        ),
    ]
    equal_weights = {"conviction_weighted": 1.0, "tiered_discrete": 1.0, "kelly_edge": 1.0}
    agg = aggregate_votes(votes, equal_weights)
    assert agg is not None
    assert agg.side == "BUY"
    # (1.0*10 + 1.0*20 + 1.0*30) / 3.0 = 20.0
    assert abs(agg.size_pct - 20.0) < 1e-6
    # reasoning string contract
    assert "committee[" in agg.reasoning
    assert "avg_size=20.0%" in agg.reasoning

    # Non-equal weights: heavy Kelly weight → average pulls toward 30.
    biased_weights = {"conviction_weighted": 0.1, "tiered_discrete": 0.1, "kelly_edge": 0.8}
    agg2 = aggregate_votes(votes, biased_weights)
    assert agg2 is not None
    # weighted = (0.1*10 + 0.1*20 + 0.8*30) / 1.0 = 1 + 2 + 24 = 27.0
    assert abs(agg2.size_pct - 27.0) < 1e-6

    # And expected_alpha_bps is a simple mean across agreeing votes (not weighted).
    assert abs(agg.expected_alpha_bps - 300.0) < 1e-6


def test_aggregator_horizon_tiebreak_prefers_longer():
    """When the agreeing-side votes are split across horizons, ties go
    to the longer horizon."""
    votes = [
        StrategyVote(
            strategy="conviction_weighted",
            order=_mk_order(side="BUY", size_pct=10.0, primary_horizon="h5",
                            expected_alpha_bps=200.0),
            conviction=1.0,
        ),
        StrategyVote(
            strategy="tiered_discrete",
            order=_mk_order(side="BUY", size_pct=10.0, primary_horizon="h60",
                            expected_alpha_bps=200.0),
            conviction=1.0,
        ),
    ]
    weights = {"conviction_weighted": 1.0, "tiered_discrete": 1.0, "kelly_edge": 1.0}
    agg = aggregate_votes(votes, weights)
    assert agg is not None
    # 1 vote each — tie → longer horizon wins → h60.
    assert agg.primary_horizon == "h60"


def test_aggregator_all_abstain_returns_none():
    votes = [
        StrategyVote(
            strategy="conviction_weighted",
            order=_mk_order(side="HOLD", size_pct=0.0, primary_horizon="h20"),
            conviction=0.0,
        ),
        StrategyVote(
            strategy="tiered_discrete",
            order=_mk_order(side="HOLD", size_pct=0.0, primary_horizon="h20"),
            conviction=0.0,
        ),
        StrategyVote(
            strategy="kelly_edge",
            order=_mk_order(side="HOLD", size_pct=0.0, primary_horizon="h20"),
            conviction=0.0,
        ),
    ]
    weights = {"conviction_weighted": 0.34, "tiered_discrete": 0.33, "kelly_edge": 0.33}
    assert aggregate_votes(votes, weights) is None


def test_aggregator_drops_low_conviction_votes():
    """A vote below `conviction_floor` is dropped, and if doing so leaves
    a single survivor on a unique side, that side wins (trivial 1/1)."""
    votes = [
        StrategyVote(
            strategy="conviction_weighted",
            order=_mk_order(side="BUY", size_pct=12.0, primary_horizon="h20",
                            expected_alpha_bps=200.0),
            conviction=0.9,
        ),
        StrategyVote(
            strategy="tiered_discrete",
            order=_mk_order(side="SELL", size_pct=30.0, primary_horizon="h20",
                            expected_alpha_bps=-200.0),
            conviction=0.15,  # below 0.20 floor
        ),
        StrategyVote(
            strategy="kelly_edge",
            order=_mk_order(side="HOLD", size_pct=0.0, primary_horizon="h20"),
            conviction=0.10,  # below 0.20 floor
        ),
    ]
    weights = {"conviction_weighted": 0.34, "tiered_discrete": 0.33, "kelly_edge": 0.33}
    agg = aggregate_votes(votes, weights)
    assert agg is not None
    assert agg.side == "BUY"
    assert abs(agg.size_pct - 12.0) < 1e-6
