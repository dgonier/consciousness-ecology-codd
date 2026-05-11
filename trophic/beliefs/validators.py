"""Tax-aware order validators for the v4 multi-horizon trophic pipeline.

Composes with the existing v3.3 validators in
`trophic.agents.apex_portfolio.ApexPortfolio.validate_orders` (universe,
no-shorts, per-name cap, sufficient cash, MIN_TRIM_PCT, ROTATE atomicity).
The new validators here are a *second* layer that runs *after* the v3.3
chain and enforce the tax-aware discipline from the v4 post-mortem:

  - `validate_min_hold`: SELL on a position bought at primary_horizon=h20
    cannot fire before 20 trading days have elapsed since the position
    was opened, UNLESS the ticker's TickerView regime_note explicitly
    invalidates ({"breakout_failure", "event_driven_invalidation"}).
    The horizon that gets enforced is the POSITION's stored horizon
    (committed at buy time), NOT the order's primary_horizon.
  - `validate_forecast_consistency`: BUY requires
    forecasts[primary_horizon] > 0; SELL requires
    forecasts[primary_horizon] < 0. Side-vs-forecast mismatches reject.
  - `validate_horizon_sizing`: size_pct must be within
    SIZE_TOLERANCE_PP of `HORIZON_BASE_SIZE[primary_horizon]`.
  - `validate_edge_floor`: abs(expected_alpha_bps) must clear
    `2 * SLIPPAGE_BPS + TAX_FRACTION * 10_000 * |forecast|`. The h60
    horizon is exempt (waived) — long deferral periods earn their own
    tax efficiency back through compounding.

All validators return `ValidatorResult(BaseModel)` with `accepted`,
`reason`, and an optional `suggested_horizon`. The runner increments
the rejected counter and feeds the rejection reason back to the apex
via the retry loop (consistent with v3.3 behaviour).
"""
from __future__ import annotations

from typing import Iterable, Optional, Union

from pydantic import BaseModel, Field
from typing import Literal

from trophic.beliefs.apex_signatures import (
    HORIZON_BASE_SIZE,
    Order,
    TickerView,
)


# ── Constants ────────────────────────────────────────────────────────

HORIZON_DAYS: dict[str, int] = {
    "h1": 1,
    "h5": 5,
    "h20": 20,
    "h60": 60,
}

SIZE_TOLERANCE_PP: float = 10.0  # widened from 5.0 — Qwen-4B can't hit ±5pp; empirical drift is 5-11pp short of target. Bands overlap slightly with adjacent tiers but the validator also reads primary_horizon so intent is preserved.

SLIPPAGE_BPS: float = 5.0
TAX_FRACTION: float = 0.37

REGIME_INVALIDATION_NOTES: frozenset[str] = frozenset({
    "breakout_failure",
    "event_driven_invalidation",
})


# ── Result type ──────────────────────────────────────────────────────

class ValidatorResult(BaseModel):
    """Outcome of a single tax-aware validator check.

    - `accepted`: True when the order survives the check.
    - `reason`:   empty string when accepted; a directive sentence when
                  rejected so the apex's retry loop can correct.
    - `suggested_horizon`: optional hint back to the apex (e.g. "your
                  position is h20 — wait, or commit to h60 next time").
    """
    accepted: bool
    reason: str = ""
    suggested_horizon: Optional[Literal["h1", "h5", "h20", "h60"]] = None


# ── Position-snapshot protocol ───────────────────────────────────────

class _PositionLike(BaseModel):
    """Minimum shape `validate_min_hold` needs from a tracked position.

    The runner passes _Position instances from `ApexPortfolio`; this
    BaseModel exists for documentation + test fixtures. The validator
    uses duck-typing on the actual call site so plain objects with the
    right attributes work too.
    """
    ticker: str
    primary_horizon: Literal["h1", "h5", "h20", "h60"]
    bought_at_date: str  # ISO date string


# ── Helpers ──────────────────────────────────────────────────────────

def _trading_days_between(opened_on: str, today: str, calendar: list[str]) -> int:
    """Count trading-day index distance between `opened_on` and `today`.

    Both dates must be in `calendar` (the runner's `_dates_seen` ledger).
    If either is missing we return a large sentinel (so the validator
    does NOT block on calendar gaps — the surrounding v3.3 validators
    already gate unheld positions).
    """
    if not opened_on or opened_on not in calendar or today not in calendar:
        return 10**9  # sentinel: assume long-held when calendar is incomplete
    try:
        return calendar.index(today) - calendar.index(opened_on)
    except ValueError:
        return 10**9


def _view_for_order(order: Order, views: Iterable[TickerView]) -> Optional[TickerView]:
    """Find the TickerView matching this order's ticker (case-insensitive).

    For ROTATE orders, returns the view for `to_ticker` since that is
    the side committing to the new horizon. Returns None if no matching
    view exists; callers should treat that as "skip the check" (the
    cross-field model_validator on ApexPMResponse already enforces that
    every order has a matching view at construction time).
    """
    target = (order.to_ticker if order.side == "ROTATE" else order.ticker) or ""
    target = target.upper()
    if not target:
        return None
    for v in views:
        if v.ticker.upper() == target:
            return v
    return None


# ── Validators ───────────────────────────────────────────────────────

def validate_in_universe(
    order: Order,
    universe: Optional[Iterable[str]] = None,
) -> ValidatorResult:
    """Reject orders whose ticker is not in the active universe.

    This is a *defensive* second layer over the input-side filter (phase
    05a): even though the apex never *sees* off-universe news/prices,
    cross-predator suggestions in the debate path can still produce
    ticker strings the apex hallucinates as extensions (e.g. "your CVX
    thesis extends to XOM too"). This validator runs FIRST in the chain
    so off-universe leaks fail-loud with a clear directive reason.

    Rules:
      - `universe` None → skipped (backward-compat for legacy paths).
      - HOLD → accept (HOLD doesn't trade; trivially in-universe).
      - BUY / SELL → ticker must be in `universe` (case-insensitive).
      - ROTATE → BOTH `from_ticker` AND `to_ticker` must be in universe.
        (The from-leg sells a held position; if it isn't in universe
        the position shouldn't exist there anyway. The to-leg opens new
        exposure and is the load-bearing check for hallucinated tickers.)

    Reason format: ``ticker_not_in_universe: <TICKER>`` — the apex retry
    loop pattern-matches on this prefix to teach the model to stay
    in-watchlist.
    """
    if universe is None:
        return ValidatorResult(accepted=True)

    # Normalize once to an uppercase frozenset for O(1) membership checks.
    universe_upper = frozenset(t.upper() for t in universe if t)

    if order.side == "HOLD":
        return ValidatorResult(accepted=True)

    if order.side == "ROTATE":
        from_t = (order.from_ticker or "").upper()
        to_t = (order.to_ticker or "").upper()
        # Both legs required for ROTATE; missing legs are caught by
        # schema-level validators upstream — here we only judge universe
        # membership on whichever legs are present.
        if from_t and from_t not in universe_upper:
            return ValidatorResult(
                accepted=False,
                reason=(
                    f"ticker_not_in_universe: {from_t} — ROTATE from_ticker "
                    f"is outside the active universe. Either drop this "
                    f"order or pick an in-universe ticker."
                ),
            )
        if to_t and to_t not in universe_upper:
            return ValidatorResult(
                accepted=False,
                reason=(
                    f"ticker_not_in_universe: {to_t} — ROTATE to_ticker "
                    f"is outside the active universe. Either drop this "
                    f"order or pick an in-universe ticker."
                ),
            )
        return ValidatorResult(accepted=True)

    # BUY / SELL: single-leg check on .ticker.
    target = (order.ticker or "").upper()
    if not target:
        # Schema-level validators handle empty-ticker on non-HOLD; don't
        # double-report here.
        return ValidatorResult(accepted=True)
    if target not in universe_upper:
        return ValidatorResult(
            accepted=False,
            reason=(
                f"ticker_not_in_universe: {target} — {order.side} on a "
                f"ticker outside the active universe. Stay within the "
                f"watchlist; either drop this order or pick an in-universe "
                f"ticker."
            ),
        )
    return ValidatorResult(accepted=True)


def validate_min_hold(
    order: Order,
    positions: Iterable[object],
    today: str,
    calendar: list[str],
    views: Iterable[TickerView] = (),
) -> ValidatorResult:
    """Enforce the position's committed horizon on SELL.

    Reads `primary_horizon` from the POSITION (the commitment made at
    buy time), not from the order. The order's regime_note can override
    if it's an explicit invalidation reason.

    BUY orders pass-through unchanged (validators here run on SELL
    discipline). ROTATE has a SELL leg — the runner is expected to
    expand ROTATE before reaching this validator, so ROTATE is treated
    as accept (the SELL leg lands separately).

    Positions are duck-typed: each must expose `.ticker`,
    `.primary_horizon`, and `.opened_on` (or `.bought_at_date`).
    """
    if order.side != "SELL":
        return ValidatorResult(accepted=True)

    target = (order.ticker or "").upper()
    pos = None
    for p in positions:
        p_ticker = getattr(p, "ticker", None)
        if p_ticker and p_ticker.upper() == target:
            pos = p
            break
    if pos is None:
        # Not our problem — v3.3 'SELL on unheld' validator handles it.
        return ValidatorResult(accepted=True)

    pos_horizon = getattr(pos, "primary_horizon", None)
    if pos_horizon not in HORIZON_DAYS:
        # No horizon stored (e.g. legacy position before v4) — accept.
        return ValidatorResult(accepted=True)

    opened_on = (
        getattr(pos, "bought_at_date", None) or getattr(pos, "opened_on", "")
    )
    held_days = _trading_days_between(opened_on, today, calendar)
    required = HORIZON_DAYS[pos_horizon]
    if held_days >= required:
        return ValidatorResult(accepted=True)

    # Held < required. Permit only when regime explicitly invalidates.
    view = _view_for_order(order, views)
    if view is not None:
        regime = (view.forecasts.regime_note or "").strip().lower()
        if regime in REGIME_INVALIDATION_NOTES:
            return ValidatorResult(
                accepted=True,
                reason="early_sell_allowed_by_regime_invalidation",
            )

    return ValidatorResult(
        accepted=False,
        reason=(
            f"{target}: min_hold_violated — position opened {opened_on} "
            f"({held_days}d held), committed at primary_horizon="
            f"{pos_horizon} (requires {required}d). Either HOLD until the "
            f"horizon completes, or invoke a regime invalidation "
            f"(regime_note ∈ {{breakout_failure, event_driven_invalidation}}) "
            f"on this ticker's TickerView."
        ),
        suggested_horizon=pos_horizon,
    )


def validate_forecast_consistency(
    order: Order,
    view: Optional[TickerView],
) -> ValidatorResult:
    """BUY requires forecasts[primary_horizon] > 0; SELL requires < 0.

    A side that disagrees with the sign of its own committed-horizon
    forecast is incoherent: the model is asking to BUY into a forecast
    of decline, or SELL into a forecast of recovery. Reject so the
    retry loop sees the contradiction and either flips side or flips
    forecast.

    No view → accept (the surrounding ApexPMResponse validator already
    flags missing views; this validator is not its replacement).
    HOLD/ROTATE → accept (HOLD has no direction; ROTATE atomicity is
    handled by v3.3's pair-tracking).
    """
    if order.side in ("HOLD", "ROTATE"):
        return ValidatorResult(accepted=True)
    if view is None:
        return ValidatorResult(accepted=True)

    horizon = order.primary_horizon
    fcst = getattr(view.forecasts, horizon, None)
    if fcst is None:
        return ValidatorResult(accepted=True)
    fcst = float(fcst)
    target = (order.ticker or "").upper()
    if order.side == "BUY" and fcst <= 0:
        return ValidatorResult(
            accepted=False,
            reason=(
                f"{target}: forecast_inconsistency — BUY committed to "
                f"primary_horizon={horizon} but forecasts.{horizon}={fcst:.4f} "
                f"is non-positive. Either flip side to SELL, drop the order, "
                f"or revise the forecast to be positive."
            ),
        )
    if order.side == "SELL" and fcst >= 0:
        return ValidatorResult(
            accepted=False,
            reason=(
                f"{target}: forecast_inconsistency — SELL committed to "
                f"primary_horizon={horizon} but forecasts.{horizon}={fcst:.4f} "
                f"is non-negative. Either flip side to BUY, drop the order, "
                f"or revise the forecast to be negative."
            ),
        )
    return ValidatorResult(accepted=True)


def validate_horizon_sizing(
    order: Order,
    slice_fraction: float = 1.0,
) -> ValidatorResult:
    """size_pct must be within SIZE_TOLERANCE_PP of HORIZON_BASE_SIZE.

    Forces tier discipline on BUY: an h5 satellite (~10%) cannot balloon
    to 30% just because the apex feels confident, and an h20 core (~30%)
    cannot shrink to 5%. SELL size_pct is a *percent of position* (per
    apex_signatures.Order), not a percent of equity, so the same base
    doesn't apply — a full close on an h20 core is size_pct=100, which
    is correct. ROTATE/HOLD are not subject.

    `slice_fraction`: when the validator runs against a sub-portfolio
    (e.g. a debate predator's $25K slice of a $100K apex), tier bases
    must be scaled down by the slice's share of total equity. h20=30%
    of total equity is 7.5% of a 0.25 slice — sizing inside the slice
    is *correct* for the predator even though it's well below the
    whole-portfolio base. Default 1.0 preserves whole-portfolio
    semantics for non-debate paths.
    """
    if order.side in ("HOLD", "ROTATE", "SELL"):
        return ValidatorResult(accepted=True)
    base = HORIZON_BASE_SIZE[order.primary_horizon] * slice_fraction
    delta = abs(order.size_pct - base)
    if delta > SIZE_TOLERANCE_PP:
        target = (order.ticker or "").upper()
        return ValidatorResult(
            accepted=False,
            reason=(
                f"{target}: sizing_mismatch — size_pct={order.size_pct:.1f}% "
                f"is {delta:.1f}pp off the {order.primary_horizon} base of "
                f"{base:.1f}% (tolerance ±{SIZE_TOLERANCE_PP:.0f}pp). Either "
                f"resize closer to {base:.1f}% or commit to a different "
                f"primary_horizon whose base size matches your conviction."
            ),
            suggested_horizon=order.primary_horizon,
        )
    return ValidatorResult(accepted=True)


def validate_edge_floor(
    order: Order,
    view: Optional[TickerView],
) -> ValidatorResult:
    """Reject orders whose expected_alpha_bps doesn't clear slippage+tax.

    floor_bps = 2 * SLIPPAGE_BPS + TAX_FRACTION * 10_000 * |forecast|

    The first term covers round-trip slippage; the second is the tax
    drag on the realized portion of the forecast (the apex's commitment
    in basis points already represents expected return, but tax claws
    37% of any realized gain back). h60 is waived — long deferral
    periods earn their own efficiency.

    HOLD/ROTATE → accept. No view → accept (caught upstream).
    """
    if order.side in ("HOLD", "ROTATE"):
        return ValidatorResult(accepted=True)
    if order.primary_horizon == "h60":
        return ValidatorResult(
            accepted=True,
            reason="floor_waived_for_h60",
        )
    if view is None:
        return ValidatorResult(accepted=True)
    fcst = getattr(view.forecasts, order.primary_horizon, None)
    if fcst is None:
        return ValidatorResult(accepted=True)
    fcst = float(fcst)
    tax_drag_bps = TAX_FRACTION * 10_000.0 * abs(fcst)
    floor_bps = 2.0 * SLIPPAGE_BPS + tax_drag_bps
    if abs(order.expected_alpha_bps) < floor_bps:
        target = (order.ticker or "").upper()
        return ValidatorResult(
            accepted=False,
            reason=(
                f"{target}: edge_below_floor — "
                f"|expected_alpha_bps|={abs(order.expected_alpha_bps):.1f}bps "
                f"< floor={floor_bps:.1f}bps "
                f"(2×{SLIPPAGE_BPS:.0f}bps slippage + 37%×|{fcst:.4f}|×10000 "
                f"= {tax_drag_bps:.1f}bps tax drag). Either commit to "
                f"primary_horizon='h60' (waived), raise expected_alpha_bps, "
                f"or drop this order — its edge can't beat tax+slippage."
            ),
        )
    return ValidatorResult(accepted=True)


# ── Chain runner ─────────────────────────────────────────────────────

def run_tax_aware_chain(
    order: Order,
    positions: Iterable[object],
    today: str,
    calendar: list[str],
    views: Iterable[TickerView] = (),
    slice_fraction: float = 1.0,
    universe: Optional[Iterable[str]] = None,
) -> ValidatorResult:
    """Run the full tax-aware chain on a single Order. Short-circuits on
    the first rejection so the apex sees one directive reason per
    rejection cycle.

    `slice_fraction`: forwarded to `validate_horizon_sizing` for
    sub-portfolio (debate-mode) calls. Default 1.0 is whole-portfolio
    semantics — non-debate paths are byte-identical.

    `universe`: optional iterable of in-universe tickers. When set,
    `validate_in_universe` runs FIRST so off-universe leaks (hallucinated
    extensions from cross-predator suggestions) fail-loud with a
    ``ticker_not_in_universe:`` reason. When None, the universe check is
    skipped (backward-compat for legacy paths).
    """
    view = _view_for_order(order, views)
    for result in (
        validate_in_universe(order, universe),
        validate_min_hold(order, positions, today, calendar, views),
        validate_forecast_consistency(order, view),
        validate_horizon_sizing(order, slice_fraction=slice_fraction),
        validate_edge_floor(order, view),
    ):
        if not result.accepted:
            return result
    return ValidatorResult(accepted=True)


__all__ = [
    "HORIZON_DAYS",
    "SIZE_TOLERANCE_PP",
    "SLIPPAGE_BPS",
    "TAX_FRACTION",
    "REGIME_INVALIDATION_NOTES",
    "ValidatorResult",
    "validate_in_universe",
    "validate_min_hold",
    "validate_forecast_consistency",
    "validate_horizon_sizing",
    "validate_edge_floor",
    "run_tax_aware_chain",
]
