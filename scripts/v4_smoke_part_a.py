"""v4 phase-2.5 smoke gate — Part A: deterministic contract verifier.

Constructs synthetic TickerViews and Orders, runs them through the strategy
committee + tax-aware validators, and verifies expected accept/reject
outcomes. No GPU, no apex calls, no network — pure unit-style assertions
over the phase-2 contracts.

Exit code 0 on `SMOKE_PART_A: PASS`, 1 on `SMOKE_PART_A: FAIL <reason>`.

Run:
  .venv/bin/python scripts/v4_smoke_part_a.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trophic.agents.apex_portfolio import ApexPortfolio, _Position
from trophic.beliefs.apex_signatures import (
    HORIZON_BASE_SIZE,
    HorizonForecast,
    Order,
    TickerView,
)
from trophic.beliefs.strategy_committee import (
    StrategyVote,
    aggregate_votes,
    conviction_weighted_vote,
    kelly_edge_vote,
    tiered_discrete_vote,
)
from trophic.beliefs.validators import (
    run_tax_aware_chain,
    validate_edge_floor,
    validate_forecast_consistency,
    validate_horizon_sizing,
    validate_min_hold,
)


WEIGHTS = {
    "conviction_weighted": 0.34,
    "tiered_discrete": 0.33,
    "kelly_edge": 0.33,
}


# ── Builders ─────────────────────────────────────────────────────────────


def mk_view(
    ticker: str,
    primary_horizon: str,
    *,
    h1: float = 0.0,
    h5: float = 0.0,
    h20: float = 0.0,
    h60: float = 0.0,
    conf_h1: str = "med",
    conf_h5: str = "med",
    conf_h20: str = "med",
    conf_h60: str = "med",
    regime: str = "trend",
) -> TickerView:
    fcst = HorizonForecast(
        h1=h1, h5=h5, h20=h20, h60=h60,
        confidence_h1=conf_h1, confidence_h5=conf_h5,
        confidence_h20=conf_h20, confidence_h60=conf_h60,
        regime_note=regime,
    )
    return TickerView(
        ticker=ticker,
        forecasts=fcst,
        primary_horizon=primary_horizon,
        rationale=f"test-view-{ticker}-{primary_horizon}",
    )


def mk_order(
    *,
    side: str,
    ticker: str,
    size_pct: float,
    primary_horizon: str,
    expected_alpha_bps: float,
    reasoning: str = "test",
) -> Order:
    return Order(
        side=side,
        ticker=ticker,
        size_pct=size_pct,
        reasoning=reasoning,
        primary_horizon=primary_horizon,
        expected_alpha_bps=expected_alpha_bps,
    )


def mk_vote(
    strategy: str,
    *,
    side: str,
    conviction: float,
    ticker: str = "AAPL",
    size_pct: float = 20.0,
    primary_horizon: str = "h20",
    expected_alpha_bps: float = 400.0,
) -> StrategyVote:
    o = Order(
        side=side,
        ticker=ticker,
        size_pct=size_pct,
        reasoning="vote-stub",
        primary_horizon=primary_horizon,
        expected_alpha_bps=expected_alpha_bps,
    )
    return StrategyVote(strategy=strategy, order=o, conviction=conviction)


# ── Assertion harness ───────────────────────────────────────────────────


class AssertLog:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, cond: bool, note: str = "") -> None:
        self.results.append((name, bool(cond), note))
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}" + (f" — {note}" if note else ""))

    def all_passed(self) -> bool:
        return all(r[1] for r in self.results)

    def first_failure(self) -> str:
        for name, ok, note in self.results:
            if not ok:
                return f"{name}: {note}" if note else name
        return ""


# ── Test cases ──────────────────────────────────────────────────────────


def test_happy_path_buy_h20(log: AssertLog) -> None:
    view = mk_view(
        "AAPL", "h20",
        h1=0.005, h5=0.012, h20=0.04, h60=0.07,
        conf_h1="low", conf_h5="med", conf_h20="high", conf_h60="med",
    )
    order = mk_order(
        side="BUY", ticker="AAPL",
        size_pct=HORIZON_BASE_SIZE["h20"],  # 30
        primary_horizon="h20",
        expected_alpha_bps=400.0,
    )
    votes = [
        conviction_weighted_vote(view, order),
        tiered_discrete_vote(view, order),
        kelly_edge_vote(view, order),
    ]
    sides = [v.order.side for v in votes]
    log.check(
        "happy_h20_buy: all 3 strategies vote BUY",
        sides == ["BUY", "BUY", "BUY"],
        f"sides={sides}",
    )
    agg = aggregate_votes(votes, WEIGHTS)
    log.check(
        "happy_h20_buy: aggregator returns BUY",
        agg is not None and agg.side == "BUY",
        f"agg={agg}",
    )

    # Run all 4 validators on the aggregated order; expect all accept.
    fc = validate_forecast_consistency(order, view)
    sz = validate_horizon_sizing(order)
    ef = validate_edge_floor(order, view)
    mh = validate_min_hold(
        order, [], "2026-02-05", ["2026-02-03", "2026-02-04", "2026-02-05"], [view],
    )
    log.check(
        "happy_h20_buy: all 4 validators accept",
        all(r.accepted for r in [fc, sz, ef, mh]),
        f"fc={fc.accepted} sz={sz.accepted} ef={ef.accepted} mh={mh.accepted}",
    )


def test_happy_path_sell_h20(log: AssertLog) -> None:
    """SELL h20 with negative forecast at primary horizon → all 4 validators
    accept. (horizon_sizing skipped on SELL per phase2-D note.)"""
    view = mk_view(
        "AAPL", "h20",
        h1=-0.005, h5=-0.012, h20=-0.04, h60=-0.02,
        conf_h20="high",
    )
    order = mk_order(
        side="SELL", ticker="AAPL",
        size_pct=100.0,  # full close
        primary_horizon="h20",
        expected_alpha_bps=-400.0,
    )
    fc = validate_forecast_consistency(order, view)
    sz = validate_horizon_sizing(order)   # skips SELL
    ef = validate_edge_floor(order, view)
    mh = validate_min_hold(
        order, [], "2026-02-05", ["2026-02-03", "2026-02-04", "2026-02-05"], [view],
    )
    log.check(
        "happy_h20_sell: validators accept (sizing skipped on SELL)",
        all(r.accepted for r in [fc, sz, ef, mh]),
        f"fc={fc.accepted} sz={sz.accepted} ef={ef.accepted} mh={mh.accepted}",
    )


def test_adversarial_horizon_sizing_reject(log: AssertLog) -> None:
    """BUY h20 with size_pct=50 (h20 base is 30, tol ±5) → reject sizing."""
    view = mk_view("MSFT", "h20", h20=0.04, conf_h20="high")
    order = mk_order(
        side="BUY", ticker="MSFT",
        size_pct=50.0,  # 20pp above base of 30
        primary_horizon="h20",
        expected_alpha_bps=400.0,
    )
    sz = validate_horizon_sizing(order)
    log.check(
        "adv_sizing: h20 BUY size=50 rejects",
        not sz.accepted and "sizing_mismatch" in sz.reason,
        f"sz={sz.accepted} reason={sz.reason[:80]}",
    )


def test_adversarial_forecast_consistency_reject(log: AssertLog) -> None:
    """BUY committed to h5 but forecast.h5 is negative → reject."""
    view = mk_view(
        "MSFT", "h5",
        h1=0.005, h5=-0.02, h20=0.01, h60=0.04,
        conf_h5="med",
    )
    order = mk_order(
        side="BUY", ticker="MSFT",
        size_pct=HORIZON_BASE_SIZE["h5"],
        primary_horizon="h5",
        expected_alpha_bps=200.0,
    )
    fc = validate_forecast_consistency(order, view)
    log.check(
        "adv_fc: BUY h5 with negative forecast rejects",
        not fc.accepted and "forecast_inconsistency" in fc.reason,
        f"fc={fc.accepted} reason={fc.reason[:80]}",
    )


def test_adversarial_edge_floor_reject_h5(log: AssertLog) -> None:
    """BUY h5 with low expected_alpha and meaningful forecast magnitude →
    reject on edge floor."""
    view = mk_view(
        "GOOG", "h5",
        h1=0.005, h5=0.02, h20=0.01, h60=0.01,  # forecast 2% → floor=10+74=84bps
        conf_h5="med",
    )
    order = mk_order(
        side="BUY", ticker="GOOG",
        size_pct=HORIZON_BASE_SIZE["h5"],
        primary_horizon="h5",
        expected_alpha_bps=50.0,  # below 84bps floor
    )
    ef = validate_edge_floor(order, view)
    log.check(
        "adv_edge: h5 BUY alpha=50 (below dynamic floor) rejects",
        not ef.accepted and "edge_below_floor" in ef.reason,
        f"ef={ef.accepted} reason={ef.reason[:80]}",
    )


def test_h60_edge_floor_waived(log: AssertLog) -> None:
    """Same expected_alpha=50 but primary_horizon=h60 → accept (waived)."""
    view = mk_view(
        "GOOG", "h60",
        h1=0.001, h5=0.005, h20=0.01, h60=0.06,
        conf_h60="high",
    )
    order = mk_order(
        side="BUY", ticker="GOOG",
        size_pct=HORIZON_BASE_SIZE["h60"],
        primary_horizon="h60",
        expected_alpha_bps=50.0,
    )
    ef = validate_edge_floor(order, view)
    log.check(
        "h60_waiver: h60 BUY alpha=50 accepts (floor waived)",
        ef.accepted and "h60" in ef.reason,
        f"ef={ef.accepted} reason={ef.reason[:80]}",
    )


def test_rebuy_preserves_min_hold(log: AssertLog) -> None:
    """Simulate ApexPortfolio: BUY h60 of TICKER1, BUY h1 of TICKER1, advance
    3 trading days, attempt SELL — min_hold rejects (still committed to h60).
    """
    p = ApexPortfolio(label="rebuy-test", starting_cash=100_000.0)
    # advance 3 days of prices and dates
    dates = ["2026-02-03", "2026-02-04", "2026-02-05", "2026-02-06"]
    for d in dates[:3]:
        p.advance_prices(d, [{"ticker": "AAPL", "actual_return": 0.0}])
    prices = {"AAPL": 100.0}

    # Day 0 BUY h60
    buy_h60 = [{
        "ticker": "AAPL", "side": "BUY", "size_pct": 50.0,
        "dollars_intent": 50_000.0,
        "primary_horizon": "h60", "expected_alpha_bps": 600.0,
        "reasoning": "long-conviction BUY",
    }]
    p.execute(dates[0], buy_h60, prices)
    pos_h60 = p.positions.get("AAPL")
    log.check(
        "rebuy: initial BUY h60 records primary_horizon",
        pos_h60 is not None and pos_h60.primary_horizon == "h60",
        f"primary_horizon={getattr(pos_h60, 'primary_horizon', None)}",
    )

    # Day 1 BUY h1 — should NOT reset the horizon (gameability fix)
    buy_h1 = [{
        "ticker": "AAPL", "side": "BUY", "size_pct": 3.0,
        "dollars_intent": 3_000.0,
        "primary_horizon": "h1", "expected_alpha_bps": 200.0,
        "reasoning": "tiny probe BUY",
    }]
    p.execute(dates[1], buy_h1, prices)
    pos_after = p.positions.get("AAPL")
    log.check(
        "rebuy: re-BUY h1 preserves original h60 horizon",
        pos_after is not None and pos_after.primary_horizon == "h60",
        f"primary_horizon={getattr(pos_after, 'primary_horizon', None)}",
    )

    # Advance to day 3, try SELL → min_hold rejects (3d held, need 60d)
    p.advance_prices(dates[3], [{"ticker": "AAPL", "actual_return": 0.0}])
    view = mk_view(
        "AAPL", "h20",
        h1=-0.005, h5=-0.012, h20=-0.04, h60=-0.02,
        conf_h20="high",
    )
    sell_order = mk_order(
        side="SELL", ticker="AAPL", size_pct=100.0,
        primary_horizon="h20", expected_alpha_bps=-400.0,
    )
    mh = validate_min_hold(
        sell_order, list(p.positions.values()), dates[3], p._dates_seen, [view],
    )
    log.check(
        "rebuy: SELL after 3d with h60 commitment is rejected by min_hold",
        not mh.accepted and "min_hold_violated" in mh.reason and "h60" in mh.reason,
        f"mh={mh.accepted} reason={mh.reason[:120]}",
    )


def test_committee_2buy_1sell_strict_majority(log: AssertLog) -> None:
    """2 BUY + 1 SELL with similar conviction → aggregator returns BUY."""
    votes = [
        mk_vote("conviction_weighted", side="BUY", conviction=0.6),
        mk_vote("tiered_discrete", side="BUY", conviction=0.6),
        mk_vote("kelly_edge", side="SELL", conviction=0.5, expected_alpha_bps=-400.0),
    ]
    agg = aggregate_votes(votes, WEIGHTS)
    log.check(
        "committee: 2BUY+1SELL aggregates to BUY",
        agg is not None and agg.side == "BUY",
        f"agg={agg}",
    )


def test_committee_mixed_1b_1s_1b_buy_wins(log: AssertLog) -> None:
    """1 BUY + 1 SELL + 1 BUY with all-equal conviction → aggregator returns BUY."""
    votes = [
        mk_vote("conviction_weighted", side="BUY", conviction=0.5),
        mk_vote("tiered_discrete", side="SELL", conviction=0.5, expected_alpha_bps=-400.0),
        mk_vote("kelly_edge", side="BUY", conviction=0.5),
    ]
    agg = aggregate_votes(votes, WEIGHTS)
    log.check(
        "committee: 1B+1S+1B aggregates to BUY",
        agg is not None and agg.side == "BUY",
        f"agg={agg}",
    )


def test_committee_all_low_conviction_none(log: AssertLog) -> None:
    """All votes below conviction floor (0.20) → aggregator returns None."""
    votes = [
        mk_vote("conviction_weighted", side="BUY", conviction=0.10),
        mk_vote("tiered_discrete", side="BUY", conviction=0.10),
        mk_vote("kelly_edge", side="BUY", conviction=0.10),
    ]
    agg = aggregate_votes(votes, WEIGHTS)
    log.check(
        "committee: all conviction<0.20 returns None",
        agg is None,
        f"agg={agg}",
    )


def test_committee_all_hold_none(log: AssertLog) -> None:
    """All votes are HOLD → aggregator returns None.

    Realistic HOLD votes come from the strategies themselves when forecasts
    are below their edge floor / sign threshold. We synthesize three HOLD
    votes directly with conviction 0 (matching strategies' actual HOLD
    output) — aggregator's conviction floor drops them all → None.
    """
    votes = [
        mk_vote("conviction_weighted", side="HOLD", conviction=0.0,
                expected_alpha_bps=0.0),
        mk_vote("tiered_discrete", side="HOLD", conviction=0.0,
                expected_alpha_bps=0.0),
        mk_vote("kelly_edge", side="HOLD", conviction=0.0,
                expected_alpha_bps=0.0),
    ]
    agg = aggregate_votes(votes, WEIGHTS)
    log.check(
        "committee: all HOLD/abstain returns None",
        agg is None,
        f"agg={agg}",
    )


def test_kelly_edge_floor_drives_hold(log: AssertLog) -> None:
    """End-to-end committee with a low-alpha h5 forecast: kelly votes HOLD
    on the edge-floor, conviction_weighted + tiered_discrete still vote BUY.
    Strict majority (2/3) for BUY → aggregator returns BUY.

    This verifies the strategies and aggregator agree on a realistic
    weak-signal scenario."""
    view = mk_view(
        "JPM", "h5",
        h1=0.005, h5=0.02, h20=0.01, h60=0.01,
        conf_h5="med",
    )
    order = mk_order(
        side="BUY", ticker="JPM",
        size_pct=HORIZON_BASE_SIZE["h5"],
        primary_horizon="h5",
        expected_alpha_bps=50.0,  # below dynamic floor → kelly HOLDs
    )
    v1 = conviction_weighted_vote(view, order)
    v2 = tiered_discrete_vote(view, order)
    v3 = kelly_edge_vote(view, order)
    sides = [v1.order.side, v2.order.side, v3.order.side]
    log.check(
        "kelly_edge_floor: kelly HOLDs on low alpha; others BUY",
        sides == ["BUY", "BUY", "HOLD"],
        f"sides={sides}",
    )
    agg = aggregate_votes([v1, v2, v3], WEIGHTS)
    log.check(
        "kelly_edge_floor: 2/3 BUY → strict majority BUY",
        agg is not None and agg.side == "BUY",
        f"agg={agg}",
    )


def test_full_chain_happy(log: AssertLog) -> None:
    """End-to-end run_tax_aware_chain on a clean h20 BUY → accepted."""
    view = mk_view(
        "WMT", "h20",
        h1=0.003, h5=0.01, h20=0.03, h60=0.06,
        conf_h20="high",
    )
    order = mk_order(
        side="BUY", ticker="WMT",
        size_pct=HORIZON_BASE_SIZE["h20"],
        primary_horizon="h20",
        expected_alpha_bps=300.0,
    )
    result = run_tax_aware_chain(
        order, [], "2026-02-05",
        ["2026-02-03", "2026-02-04", "2026-02-05"], [view],
    )
    log.check(
        "full_chain: clean h20 BUY accepts end-to-end",
        result.accepted,
        f"result={result}",
    )


# ── Driver ──────────────────────────────────────────────────────────────


def main() -> int:
    log = AssertLog()
    print("SMOKE PART A — phase-2.5 contract verifier")
    print("=" * 72)
    tests = [
        ("happy path BUY h20", test_happy_path_buy_h20),
        ("happy path SELL h20", test_happy_path_sell_h20),
        ("adversarial: horizon sizing reject", test_adversarial_horizon_sizing_reject),
        ("adversarial: forecast consistency reject",
         test_adversarial_forecast_consistency_reject),
        ("adversarial: h5 edge floor reject", test_adversarial_edge_floor_reject_h5),
        ("h60 edge floor waived", test_h60_edge_floor_waived),
        ("re-buy preserves min_hold horizon", test_rebuy_preserves_min_hold),
        ("committee: 2BUY+1SELL → BUY", test_committee_2buy_1sell_strict_majority),
        ("committee: 1B+1S+1B equal-conv → BUY", test_committee_mixed_1b_1s_1b_buy_wins),
        ("committee: all <0.20 conv → None", test_committee_all_low_conviction_none),
        ("committee: all HOLD → None", test_committee_all_hold_none),
        ("kelly edge-floor → 2/3 BUY", test_kelly_edge_floor_drives_hold),
        ("full tax-aware chain clean BUY", test_full_chain_happy),
    ]
    for name, fn in tests:
        print(f"\n— {name} —")
        try:
            fn(log)
        except Exception as e:
            log.check(name, False, f"EXCEPTION: {e!r}")
            traceback.print_exc()

    print("\n" + "=" * 72)
    n_pass = sum(1 for _, ok, _ in log.results if ok)
    n_fail = sum(1 for _, ok, _ in log.results if not ok)
    print(f"Assertions: {n_pass} passed, {n_fail} failed (total {len(log.results)})")
    if log.all_passed():
        print("SMOKE_PART_A: PASS")
        return 0
    else:
        first = log.first_failure()
        print(f"SMOKE_PART_A: FAIL {first}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
