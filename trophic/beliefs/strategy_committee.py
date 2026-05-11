"""Three-strategy committee for v4 sizing.

Each strategy looks at one (TickerView, Order) pair from the apex output
and casts a `StrategyVote` carrying its own re-sized order and a conviction
in [0, 1]. The aggregator combines the votes using `philosophy_weights`
and a `conviction_floor`, returning either a single aggregated `Order`
or `None` when the strategies fail to reach strict majority agreement
on `side` (or all abstain).

Contracts:
  - Strategies do not mutate the input `Order`.
  - Aggregator returns `None` for "no consensus / no order" — the runner
    must drop that line, NOT raise.
  - Side disagreements without a strict majority → `None` (don't average
    opposing sides).
  - Aggregator's `size_pct` is a weighted average using
    `philosophy_weights[strategy] * conviction` as the weight per vote.

See `tasks_v4/phase2-C-03-strategy-committee.md` and the
INTERFACE CONTRACTS block in `tasks_v4/scratchpad.md` for the
contract this module implements.
"""
from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .apex_signatures import HORIZON_BASE_SIZE, Order, TickerView


__all__ = [
    "StrategyVote",
    "HORIZON_BASE_SIZE",
    "conviction_weighted_vote",
    "tiered_discrete_vote",
    "kelly_edge_vote",
    "aggregate_votes",
]


# Confidence label → numeric multiplier. The high tier saturates at 1.0
# so a "high"-confidence trade has no extra magnitude headroom from the
# label alone; magnitude headroom comes from the forecast itself.
_CONFIDENCE_VALUE: dict[str, float] = {
    "low": 0.33,
    "med": 0.66,
    "high": 1.0,
}

_HORIZON_ORDER: list[str] = ["h1", "h5", "h20", "h60"]

# Default Kelly variance estimate (sigma=2% per period). With a 162bps
# edge and 0.0004 variance, kelly_fraction = 0.0162 / 0.0004 = 40.5 →
# 40.5% size. That's the right order of magnitude for a confident trade.
DEFAULT_KELLY_VARIANCE: float = 0.02 ** 2

# Tax + slippage floor in bps (10bps slippage round-trip + tax buffer).
# Waived for h60 trades (see contract in scratchpad.md INTERFACE CONTRACTS).
KELLY_EDGE_FLOOR_BPS: float = 162.0

# Sides we will vote on. HOLD is a valid abstain side; ROTATE is not a
# committee-voted side (the aggregator passes ROTATE through unchanged
# by construction — see the runner wiring).
_VOTING_SIDES: tuple[str, ...] = ("BUY", "SELL", "HOLD")

# Minimum size_pct for a real BUY/SELL — anything smaller is tax/slippage
# drag and should be treated as HOLD by the strategy.
_MIN_REAL_SIZE_PCT: float = 0.01


class StrategyVote(BaseModel):
    """One strategy's proposed (Order, conviction) for a single ticker.

    `order.size_pct` is the strategy's own sizing, not the apex's. The
    aggregator weights `size_pct` by `philosophy_weights[strategy] *
    conviction` across the agreeing-side votes.
    """
    strategy: Literal["conviction_weighted", "tiered_discrete", "kelly_edge"]
    order: Order
    conviction: float = Field(ge=0.0, le=1.0)


# ── Helpers ────────────────────────────────────────────────────────────


def _forecast_at(view: TickerView, horizon: str) -> float:
    return float(getattr(view.forecasts, horizon))


def _confidence_at(view: TickerView, horizon: str) -> str:
    return getattr(view.forecasts, f"confidence_{horizon}")


def _conf_value(label: str) -> float:
    return _CONFIDENCE_VALUE.get(label, 0.33)


def _side_from_forecast(forecast: float) -> str:
    """BUY / SELL / HOLD by sign + magnitude. |forecast| < 0.001 → HOLD."""
    if abs(forecast) < 0.001:
        return "HOLD"
    return "BUY" if forecast > 0 else "SELL"


def _make_order(
    template: Order,
    *,
    side: str,
    size_pct: float,
    primary_horizon: str,
    reasoning: str,
    expected_alpha_bps: Optional[float] = None,
) -> Order:
    """Construct a new Order from an existing template, replacing the
    fields we vary per-strategy. We keep `ticker` / `from_ticker` /
    `to_ticker` exactly as on the template — strategies never reroute a
    ticker."""
    alpha = (
        expected_alpha_bps
        if expected_alpha_bps is not None
        else template.expected_alpha_bps
    )
    # Clamp size + alpha to the schema bounds defensively. The apex can
    # emit alpha at ±10000bps but the committee's Kelly clamp + size
    # arithmetic can drift past the bounds without this.
    size_pct = max(0.0, min(100.0, size_pct))
    alpha = max(-10000.0, min(10000.0, alpha))
    return Order(
        side=side,  # type: ignore[arg-type]
        ticker=template.ticker,
        from_ticker=template.from_ticker,
        to_ticker=template.to_ticker,
        size_pct=size_pct,
        reasoning=reasoning,
        primary_horizon=primary_horizon,  # type: ignore[arg-type]
        expected_alpha_bps=alpha,
    )


# ── Strategies ────────────────────────────────────────────────────────


def conviction_weighted_vote(view: TickerView, order: Order) -> StrategyVote:
    """Continuous: size = base × conviction²; conviction combines the
    confidence label at `primary_horizon` with the forecast magnitude.

    conviction = min(1, 0.5 * conf_value + 0.5 * tanh(20 * |forecast|))
    """
    horizon = view.primary_horizon
    forecast = _forecast_at(view, horizon)
    conf_label = _confidence_at(view, horizon)
    conf_value = _conf_value(conf_label)
    mag = math.tanh(20.0 * abs(forecast))
    conviction = min(1.0, 0.5 * conf_value + 0.5 * mag)
    base_size = HORIZON_BASE_SIZE[horizon]
    size_pct = base_size * (conviction ** 2)
    side = _side_from_forecast(forecast)
    if side == "HOLD" or size_pct < _MIN_REAL_SIZE_PCT:
        # Force a HOLD vote with conviction 0 — aggregator's floor will
        # drop it. We don't want to hand the aggregator a 0-size BUY.
        return StrategyVote(
            strategy="conviction_weighted",
            order=_make_order(
                order, side="HOLD", size_pct=0.0,
                primary_horizon=horizon,
                reasoning="conviction_weighted: |forecast|<0.001 or size<eps",
            ),
            conviction=0.0,
        )
    return StrategyVote(
        strategy="conviction_weighted",
        order=_make_order(
            order, side=side, size_pct=size_pct,
            primary_horizon=horizon,
            reasoning=(
                f"conviction_weighted: f={forecast:+.3f} conf={conf_label} "
                f"→ conviction={conviction:.2f} → size={size_pct:.1f}%"
            ),
        ),
        conviction=conviction,
    )


def tiered_discrete_vote(view: TickerView, order: Order) -> StrategyVote:
    """Discrete: size = HORIZON_BASE_SIZE[primary_horizon] (no scaling).
    Conviction = confidence label value at primary_horizon.
    """
    horizon = view.primary_horizon
    forecast = _forecast_at(view, horizon)
    conf_label = _confidence_at(view, horizon)
    conviction = _conf_value(conf_label)
    side = _side_from_forecast(forecast)
    size_pct = HORIZON_BASE_SIZE[horizon]
    if side == "HOLD":
        return StrategyVote(
            strategy="tiered_discrete",
            order=_make_order(
                order, side="HOLD", size_pct=0.0,
                primary_horizon=horizon,
                reasoning="tiered_discrete: |forecast|<0.001",
            ),
            conviction=0.0,
        )
    return StrategyVote(
        strategy="tiered_discrete",
        order=_make_order(
            order, side=side, size_pct=size_pct,
            primary_horizon=horizon,
            reasoning=(
                f"tiered_discrete: horizon={horizon} → size={size_pct:.1f}% "
                f"(conf={conf_label})"
            ),
        ),
        conviction=conviction,
    )


def kelly_edge_vote(
    view: TickerView,
    order: Order,
    variance_estimate: float = DEFAULT_KELLY_VARIANCE,
    edge_floor_bps: float = KELLY_EDGE_FLOOR_BPS,
) -> StrategyVote:
    """Kelly-edge: size = clamp(0, 50, 100 * (alpha/10000) / variance).
    Edge floor of `edge_floor_bps` — if |alpha| < floor for non-h60
    horizons, vote HOLD with conviction 0. h60 waives the floor.
    """
    horizon = view.primary_horizon
    alpha_bps = float(order.expected_alpha_bps)
    forecast = _forecast_at(view, horizon)
    side = _side_from_forecast(forecast)

    # Edge floor — waived for h60.
    if horizon != "h60" and abs(alpha_bps) < edge_floor_bps:
        return StrategyVote(
            strategy="kelly_edge",
            order=_make_order(
                order, side="HOLD", size_pct=0.0,
                primary_horizon=horizon,
                reasoning=(
                    f"kelly_edge: |alpha|={abs(alpha_bps):.0f}bps < "
                    f"floor={edge_floor_bps:.0f}bps (h{horizon[1:]} non-h60)"
                ),
            ),
            conviction=0.0,
        )

    if variance_estimate <= 0.0:
        variance_estimate = DEFAULT_KELLY_VARIANCE

    # Kelly fraction; use abs because we'll set side from the forecast sign.
    kelly_fraction = (abs(alpha_bps) / 10_000.0) / variance_estimate
    size_pct = max(0.0, min(50.0, 100.0 * kelly_fraction))

    if side == "HOLD" or size_pct < _MIN_REAL_SIZE_PCT:
        return StrategyVote(
            strategy="kelly_edge",
            order=_make_order(
                order, side="HOLD", size_pct=0.0,
                primary_horizon=horizon,
                reasoning="kelly_edge: HOLD (sign 0 or size eps)",
            ),
            conviction=0.0,
        )

    # Map size_pct ∈ [0, 50] → conviction ∈ [0, 1].
    conviction = min(1.0, size_pct / 50.0)
    return StrategyVote(
        strategy="kelly_edge",
        order=_make_order(
            order, side=side, size_pct=size_pct,
            primary_horizon=horizon,
            reasoning=(
                f"kelly_edge: alpha={alpha_bps:+.0f}bps var={variance_estimate:.4f} "
                f"→ kelly={kelly_fraction:.3f} → size={size_pct:.1f}%"
            ),
        ),
        conviction=conviction,
    )


# ── Aggregator ────────────────────────────────────────────────────────


def aggregate_votes(
    votes: list[StrategyVote],
    philosophy_weights: dict[str, float],
    conviction_floor: float = 0.20,
) -> Optional[Order]:
    """Combine the three votes for one ticker into one Order, or None.

    Steps:
      1. Drop votes with `conviction < conviction_floor`.
      2. Tally `side` among the survivors; require a STRICT majority
         (> half of active votes). Tie → None.
      3. Take the agreeing-side subset; compute size_pct as a
         weighted average using `philosophy_weights[strategy] * conviction`.
      4. primary_horizon = majority among agreeing votes; tiebreak by
         the LONGER horizon (h60 wins over h20 wins over h5 wins over h1).
      5. expected_alpha_bps = simple mean of the agreeing votes' alpha.

    Returns None if no active votes, weights collapse to 0, or no strict
    majority on side.
    """
    # 1. Filter by conviction floor.
    active = [v for v in votes if v.conviction >= conviction_floor]
    if not active:
        return None

    # 2. Strict majority on side. "Strict" = more than half (not just plurality).
    sides = [v.order.side for v in active]
    side_counts: dict[str, int] = {}
    for s in sides:
        side_counts[s] = side_counts.get(s, 0) + 1
    majority_side, majority_count = max(
        side_counts.items(), key=lambda kv: kv[1]
    )
    if majority_count * 2 <= len(active):
        # Need strictly more than half. Ties + plurality without majority → None.
        return None
    # If the majority is HOLD, there's nothing to execute.
    if majority_side == "HOLD":
        return None

    # 3. Weighted average size_pct across agreeing votes.
    agreeing = [v for v in active if v.order.side == majority_side]
    weights_per_vote = [
        philosophy_weights.get(v.strategy, 0.0) * v.conviction for v in agreeing
    ]
    total_w = sum(weights_per_vote)
    if total_w <= 0.0:
        return None
    avg_size = (
        sum(w * v.order.size_pct for w, v in zip(weights_per_vote, agreeing))
        / total_w
    )

    # 4. primary_horizon: majority among agreeing; tiebreak = longer horizon.
    horizons = [v.order.primary_horizon for v in agreeing]
    horizon_counts: dict[str, int] = {}
    for h in horizons:
        horizon_counts[h] = horizon_counts.get(h, 0) + 1

    def _horizon_key(h: str) -> tuple[int, int]:
        # Higher count wins; ties → longer horizon (later in _HORIZON_ORDER).
        return (horizon_counts[h], _HORIZON_ORDER.index(h))

    majority_horizon = max(horizon_counts.keys(), key=_horizon_key)

    # 5. expected_alpha_bps = mean across agreeing votes.
    mean_alpha = sum(v.order.expected_alpha_bps for v in agreeing) / len(agreeing)

    template = agreeing[0].order
    strategies = ",".join(v.strategy for v in agreeing)
    reasoning = f"committee[{strategies}] avg_size={avg_size:.1f}%"

    return _make_order(
        template,
        side=majority_side,
        size_pct=max(0.0, min(100.0, avg_size)),
        primary_horizon=majority_horizon,
        reasoning=reasoning,
        expected_alpha_bps=mean_alpha,
    )
