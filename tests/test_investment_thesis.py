"""Tests for trophic.beliefs.investment_thesis.

Covers: construction round-trip, lifecycle helpers, ThesisBook ops,
overdue-sweep semantics, and the model-validator invariants.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from trophic.beliefs.investment_thesis import (
    InvestmentThesis,
    ThesisBook,
    expire_thesis,
    invalidate_thesis,
    is_thesis_overdue,
    mature_thesis,
    open_thesis,
)


# ── 1. Construction + round-trip ─────────────────────────────────────

def test_construction_round_trip():
    t = InvestmentThesis(
        predator_id="momentum",
        philosophy="momentum",
        ticker="AAPL",
        opened_at_date="2026-02-03",
        direction="long",
        primary_horizon="h20",
        catalysts=["earnings_beat"],
        invalidation_triggers=["guidance_cut", "breakout_failure"],
        expected_alpha_bps=350,
        confidence="high",
    )
    dumped = t.model_dump()
    t2 = InvestmentThesis.model_validate(dumped)
    assert t == t2
    assert t.status == "active"
    assert t.closed_at_date is None
    assert len(t.thesis_id) == 12
    # uuid hex is in [0-9a-f]
    assert all(c in "0123456789abcdef" for c in t.thesis_id)


def test_open_thesis_helper():
    t = open_thesis(
        predator_id="value",
        philosophy="value",
        ticker="JPM",
        opened_at_date="2026-02-10",
        direction="long",
        primary_horizon="h60",
        catalysts=["p_b_below_1"],
        invalidation_triggers=["earnings_miss"],
        expected_alpha_bps=800,
        confidence="med",
        target_price=180.0,
    )
    assert t.status == "active"
    assert t.target_price == 180.0
    assert t.realized_pnl is None


# ── 2. Lifecycle (mature / invalidate / expire) ──────────────────────

def test_mature_invalidate_expire():
    base = open_thesis(
        predator_id="momentum",
        philosophy="momentum",
        ticker="NVDA",
        opened_at_date="2026-02-01",
        direction="long",
        primary_horizon="h5",
        catalysts=["20d_breakout"],
        invalidation_triggers=["breakout_failure"],
        expected_alpha_bps=400,
        confidence="high",
    )

    # mature
    m = mature_thesis(base, closed_at_date="2026-02-08", realized_pnl=1234.56, realized_return_pct=4.2)
    assert m.status == "matured"
    assert m.close_reason == "matured"
    assert m.closed_at_date == "2026-02-08"
    assert m.realized_pnl == 1234.56
    assert m.realized_return_pct == 4.2
    # base unchanged (immutable update)
    assert base.status == "active"

    # invalidate (with trigger)
    inv = invalidate_thesis(
        base, closed_at_date="2026-02-04", trigger="breakout_failure",
        realized_pnl=-200.0, realized_return_pct=-0.7,
    )
    assert inv.status == "invalidated"
    assert inv.close_reason == "invalidated:breakout_failure"
    assert "breakout_failure" in inv.close_reason

    # expire
    ex = expire_thesis(base, closed_at_date="2026-02-06", realized_pnl=50.0, realized_return_pct=0.17)
    assert ex.status == "expired"
    assert ex.close_reason == "expired"


# ── 3. ThesisBook lifecycle ──────────────────────────────────────────

def _make_thesis(predator_id="momentum", ticker="AAPL", horizon="h20", alpha=250.0):
    return open_thesis(
        predator_id=predator_id,
        philosophy="momentum",
        ticker=ticker,
        opened_at_date="2026-02-03",
        direction="long",
        primary_horizon=horizon,
        catalysts=["c1"],
        invalidation_triggers=["t1"],
        expected_alpha_bps=alpha,
        confidence="med",
    )


def test_thesis_book_lifecycle():
    book = ThesisBook(predator_id="momentum")
    t1 = _make_thesis(ticker="AAPL")
    t2 = _make_thesis(ticker="MSFT")
    t3 = _make_thesis(ticker="AAPL", horizon="h60", alpha=600)
    t4 = _make_thesis(ticker="NVDA")
    for t in (t1, t2, t3, t4):
        book.add(t)

    assert len(book.active_theses()) == 4
    aapl_active = book.active_for_ticker("AAPL")
    assert len(aapl_active) == 2
    assert {t.thesis_id for t in aapl_active} == {t1.thesis_id, t3.thesis_id}
    # case-insensitive ticker query
    assert len(book.active_for_ticker("aapl")) == 2

    # Close t1 as matured
    closed = book.close(
        t1.thesis_id, status="matured", closed_at_date="2026-02-23",
        realized_pnl=500.0, realized_return_pct=1.8,
    )
    assert closed.status == "matured"
    assert len(book.active_theses()) == 3
    assert len(book.closed_theses()) == 1
    # AAPL active drops to 1
    assert len(book.active_for_ticker("AAPL")) == 1

    # Close t4 as invalidated
    book.close(
        t4.thesis_id, status="invalidated", closed_at_date="2026-02-12",
        realized_pnl=-100.0, realized_return_pct=-0.4, trigger="guidance_cut",
    )
    assert len(book.closed_theses()) == 2
    nvda_closed = [t for t in book.closed_theses() if t.ticker == "NVDA"][0]
    assert nvda_closed.close_reason == "invalidated:guidance_cut"

    # Double-close errors
    with pytest.raises(ValueError):
        book.close(
            t1.thesis_id, status="matured", closed_at_date="2026-02-24",
            realized_pnl=0.0, realized_return_pct=0.0,
        )

    # Missing thesis errors
    with pytest.raises(KeyError):
        book.close(
            "deadbeef" + "0" * 4, status="matured", closed_at_date="2026-02-24",
            realized_pnl=0.0, realized_return_pct=0.0,
        )

    # Invalidated requires trigger
    t5 = _make_thesis(ticker="GOOG")
    book.add(t5)
    with pytest.raises(ValueError):
        book.close(
            t5.thesis_id, status="invalidated", closed_at_date="2026-02-13",
            realized_pnl=0.0, realized_return_pct=0.0,  # no trigger
        )


def test_thesis_book_rejects_wrong_predator():
    book = ThesisBook(predator_id="momentum")
    other = _make_thesis(predator_id="value")
    with pytest.raises(ValueError):
        book.add(other)


# ── 4. Overdue sweep ─────────────────────────────────────────────────

def test_sweep_overdue():
    book = ThesisBook(predator_id="momentum")
    h5 = _make_thesis(ticker="AAPL", horizon="h5")  # day-0 open
    h20 = _make_thesis(ticker="MSFT", horizon="h20")  # day-0 open
    book.add(h5)
    book.add(h20)

    opened_idx = {h5.thesis_id: 0, h20.thesis_id: 0}

    # is_thesis_overdue: directly verify the predicate
    assert is_thesis_overdue(h5, today_idx=3, opened_idx=0) is False
    assert is_thesis_overdue(h5, today_idx=5, opened_idx=0) is True
    assert is_thesis_overdue(h5, today_idx=6, opened_idx=0) is True
    assert is_thesis_overdue(h20, today_idx=5, opened_idx=0) is False

    # Day 3 sweep — nothing expires.
    expired_day3 = book.sweep_overdue(today_idx=3, opened_idx_by_thesis=opened_idx)
    assert expired_day3 == []
    assert len(book.active_theses()) == 2

    # Day 5 sweep — h5 expires, h20 still active.
    expired_day5 = book.sweep_overdue(today_idx=5, opened_idx_by_thesis=opened_idx)
    assert len(expired_day5) == 1
    assert expired_day5[0].thesis_id == h5.thesis_id
    assert expired_day5[0].status == "expired"
    assert expired_day5[0].close_reason == "expired"
    assert len(book.active_theses()) == 1

    # Second sweep at day 5 — already expired, doesn't re-fire.
    expired_again = book.sweep_overdue(today_idx=5, opened_idx_by_thesis=opened_idx)
    assert expired_again == []

    # Day 20 sweep — h20 expires.
    expired_day20 = book.sweep_overdue(today_idx=20, opened_idx_by_thesis=opened_idx)
    assert len(expired_day20) == 1
    assert expired_day20[0].thesis_id == h20.thesis_id

    # is_thesis_overdue returns False for non-active.
    assert is_thesis_overdue(expired_day5[0], today_idx=100, opened_idx=0) is False


def test_sweep_overdue_skips_unmapped():
    book = ThesisBook(predator_id="momentum")
    t = _make_thesis(ticker="AAPL", horizon="h1")
    book.add(t)
    # opened_idx_by_thesis missing this id → skipped, not expired
    expired = book.sweep_overdue(today_idx=10, opened_idx_by_thesis={})
    assert expired == []
    assert book.active_theses()[0].thesis_id == t.thesis_id


# ── 5. Validation rules ──────────────────────────────────────────────

def test_validation_rules():
    # expected_alpha_bps out of range (above max)
    with pytest.raises(ValidationError):
        InvestmentThesis(
            predator_id="momentum",
            philosophy="momentum",
            ticker="AAPL",
            opened_at_date="2026-02-03",
            direction="long",
            primary_horizon="h20",
            expected_alpha_bps=20000,  # > 10000
            confidence="high",
        )

    # expected_alpha_bps below min
    with pytest.raises(ValidationError):
        InvestmentThesis(
            predator_id="momentum",
            philosophy="momentum",
            ticker="AAPL",
            opened_at_date="2026-02-03",
            direction="long",
            primary_horizon="h20",
            expected_alpha_bps=-20000,
            confidence="high",
        )

    # status='matured' without close_reason / closed_at_date — invalid
    with pytest.raises(ValidationError):
        InvestmentThesis(
            predator_id="momentum",
            philosophy="momentum",
            ticker="AAPL",
            opened_at_date="2026-02-03",
            direction="long",
            primary_horizon="h20",
            expected_alpha_bps=300,
            confidence="high",
            status="matured",
            # missing closed_at_date, close_reason, realized_*
        )

    # status='active' with closed_at_date set — invalid
    with pytest.raises(ValidationError):
        InvestmentThesis(
            predator_id="momentum",
            philosophy="momentum",
            ticker="AAPL",
            opened_at_date="2026-02-03",
            direction="long",
            primary_horizon="h20",
            expected_alpha_bps=300,
            confidence="high",
            status="active",
            closed_at_date="2026-02-23",
        )

    # status='active' with close_reason set — invalid
    with pytest.raises(ValidationError):
        InvestmentThesis(
            predator_id="momentum",
            philosophy="momentum",
            ticker="AAPL",
            opened_at_date="2026-02-03",
            direction="long",
            primary_horizon="h20",
            expected_alpha_bps=300,
            confidence="high",
            status="active",
            close_reason="manual",
        )

    # status='invalidated' missing realized_pnl — invalid
    with pytest.raises(ValidationError):
        InvestmentThesis(
            predator_id="momentum",
            philosophy="momentum",
            ticker="AAPL",
            opened_at_date="2026-02-03",
            direction="long",
            primary_horizon="h20",
            expected_alpha_bps=300,
            confidence="high",
            status="invalidated",
            closed_at_date="2026-02-04",
            close_reason="invalidated:breakout_failure",
            # missing realized_pnl, realized_return_pct
        )

    # ticker too long
    with pytest.raises(ValidationError):
        InvestmentThesis(
            predator_id="momentum",
            philosophy="momentum",
            ticker="TOOLONGTICKER",
            opened_at_date="2026-02-03",
            direction="long",
            primary_horizon="h20",
            expected_alpha_bps=300,
            confidence="high",
        )

    # ticker empty
    with pytest.raises(ValidationError):
        InvestmentThesis(
            predator_id="momentum",
            philosophy="momentum",
            ticker="",
            opened_at_date="2026-02-03",
            direction="long",
            primary_horizon="h20",
            expected_alpha_bps=300,
            confidence="high",
        )


def test_validation_boundaries_accepted():
    # expected_alpha_bps at exact boundaries is OK
    t_lo = InvestmentThesis(
        predator_id="momentum",
        philosophy="momentum",
        ticker="AAPL",
        opened_at_date="2026-02-03",
        direction="long",
        primary_horizon="h20",
        expected_alpha_bps=-10000,
        confidence="high",
    )
    assert t_lo.expected_alpha_bps == -10000
    t_hi = InvestmentThesis(
        predator_id="momentum",
        philosophy="momentum",
        ticker="AAPL",
        opened_at_date="2026-02-03",
        direction="long",
        primary_horizon="h20",
        expected_alpha_bps=10000,
        confidence="high",
    )
    assert t_hi.expected_alpha_bps == 10000
