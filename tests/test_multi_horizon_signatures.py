"""Phase 1-A (v4): multi-horizon signature contracts.

Tests the new types introduced for the v4 multi-horizon trophic upgrade:
  - HorizonForecast, TickerView, ApexPMResponse construction & round-trip
  - Order requires primary_horizon + expected_alpha_bps
  - expected_alpha_bps is bounded [-10000, 10000]
  - ApexPMResponse.model_validator rejects orders that reference a ticker
    not present in `views`
  - PM signatures' docstrings carry the tax-framing block
  - state_for_apex emits profit_definition on PortfolioState
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from trophic.beliefs import apex_signatures as A
from trophic.beliefs.apex_signatures import (
    ApexPMResponse,
    HorizonForecast,
    HORIZON_BASE_SIZE,
    Order,
    PortfolioState,
    TickerView,
)


# ── Type construction & round-trip ─────────────────────────────────────

def _make_forecast(**overrides):
    base = dict(
        h1=0.001, h5=0.005, h20=0.02, h60=0.05,
        confidence_h1="med", confidence_h5="med",
        confidence_h20="high", confidence_h60="med",
        regime_note="range",
    )
    base.update(overrides)
    return HorizonForecast(**base)


def test_horizon_forecast_flattens_nested_dict_from_qwen():
    """Qwen-4B sometimes emits the full forecast dict inside one horizon's
    value: forecasts.h1 = {"h1": 0.001, "h5": 0.005, "h20": 0.02, "h60": 0.05}.
    The normalizer should pull the matching key out and copy siblings up
    so the flat shape validates.
    """
    nested = {
        "h1": {"h1": 0.001, "h5": 0.005, "h20": 0.02, "h60": 0.05},
        "h5": None,
        "h20": None,
        "h60": None,
        "confidence_h1": "med",
        "confidence_h5": "med",
        "confidence_h20": "med",
        "confidence_h60": "med",
        "regime_note": "trend",
    }
    hf = HorizonForecast.model_validate(nested)
    assert hf.h1 == pytest.approx(0.001)
    assert hf.h5 == pytest.approx(0.005)
    assert hf.h20 == pytest.approx(0.02)
    assert hf.h60 == pytest.approx(0.05)


def test_horizon_forecast_unchanged_when_already_flat():
    """The normalizer must not touch a well-shaped forecast."""
    flat = {
        "h1": 0.001, "h5": 0.005, "h20": 0.02, "h60": 0.05,
        "confidence_h1": "med", "confidence_h5": "med",
        "confidence_h20": "med", "confidence_h60": "med",
        "regime_note": "trend",
    }
    hf = HorizonForecast.model_validate(flat)
    assert hf.h1 == pytest.approx(0.001)
    assert hf.h60 == pytest.approx(0.05)


def test_horizon_forecast_round_trip():
    hf = _make_forecast()
    blob = hf.model_dump()
    rebuilt = HorizonForecast.model_validate(blob)
    assert rebuilt.h20 == pytest.approx(0.02)
    assert rebuilt.confidence_h20 == "high"
    assert rebuilt.regime_note == "range"


def test_horizon_forecast_regime_note_defaults_to_unspecified():
    """Phase 2-fix: regime_note has a default so the apex omitting it
    does not nuke the whole TickerView (the smoke gate caught this:
    apex dropped 5 forecasts on day-2 because regime_note was absent).
    Default is 'unspecified' — accepted but uninformative."""
    hf = HorizonForecast(
        h1=0.001, h5=0.005, h20=0.02, h60=0.05,
        confidence_h1="med", confidence_h5="med",
        confidence_h20="high", confidence_h60="med",
        # regime_note intentionally omitted
    )
    assert hf.regime_note == "unspecified"


def test_horizon_forecast_fills_missing_confidence_fields_from_qwen():
    """Run3 day-3 sweep fail: Qwen-4B emitted `forecasts = {h1, h5, h20,
    h60}` (all numeric horizons) but omitted ALL FOUR `confidence_h*`
    fields. Pre-fix this raised 5 ValidationErrors and dropped the entire
    debate day to no-trade fallback. We now fill missing confidence
    fields with 'med' so the view survives — the numeric forecasts carry
    the actual signal; confidence is advisory.
    """
    hf = HorizonForecast.model_validate({
        "h1": 0.001, "h5": 0.002, "h20": 0.005, "h60": 0.01,
        # All four confidence_h* fields intentionally omitted (the
        # exact shape Qwen-4B emitted on 2026-02-05).
    })
    assert hf.h60 == pytest.approx(0.01)
    assert hf.confidence_h1 == "med"
    assert hf.confidence_h5 == "med"
    assert hf.confidence_h20 == "med"
    assert hf.confidence_h60 == "med"
    assert hf.regime_note == "unspecified"


def test_horizon_forecast_does_not_overwrite_explicit_confidence():
    """Sanity: when the model DOES emit confidence_h*, the filler must
    not silently downgrade 'high'/'low' to 'med'.
    """
    hf = HorizonForecast.model_validate({
        "h1": 0.001, "h5": 0.002, "h20": 0.005, "h60": 0.01,
        "confidence_h1": "low",
        "confidence_h5": "high",
        "confidence_h20": "high",
        "confidence_h60": "low",
    })
    assert hf.confidence_h1 == "low"
    assert hf.confidence_h5 == "high"
    assert hf.confidence_h20 == "high"
    assert hf.confidence_h60 == "low"


def test_ticker_view_rationale_defaults_when_missing():
    """Run5 day-9 sweep fail (2026-02-13): Qwen-4B emitted views[0]
    without a `rationale` field at all. Pre-fix this raised
    `views.0.rationale: Field required` and dropped the entire Phase-1
    proposal across 3 errors. We now default rationale to 'unspecified'
    so the numerical signal (forecasts + primary_horizon) survives.
    """
    from trophic.beliefs.apex_signatures import TickerView
    v = TickerView.model_validate({
        "ticker": "V",
        "forecasts": {
            "h1": 0.001, "h5": 0.005, "h20": 0.02, "h60": 0.05,
            "confidence_h1": "med", "confidence_h5": "med",
            "confidence_h20": "high", "confidence_h60": "high",
        },
        "primary_horizon": "h20",
        # rationale intentionally omitted
    })
    assert v.rationale == "unspecified"
    assert v.ticker == "V"
    assert v.forecasts.h20 == pytest.approx(0.02)


def test_ticker_view_rationale_preserved_when_provided():
    """Sanity: when the model DOES supply a rationale, the default
    doesn't silently overwrite it.
    """
    from trophic.beliefs.apex_signatures import TickerView
    v = TickerView.model_validate({
        "ticker": "V",
        "forecasts": {
            "h1": 0.001, "h5": 0.005, "h20": 0.02, "h60": 0.05,
            "confidence_h1": "med", "confidence_h5": "med",
            "confidence_h20": "high", "confidence_h60": "high",
        },
        "primary_horizon": "h20",
        "rationale": "h20 momentum off rate-normalization narrative",
    })
    assert v.rationale == "h20 momentum off rate-normalization narrative"


def test_horizon_forecast_invalid_confidence_string_is_replaced():
    """Day-?: Qwen-4B emitted `confidence_h1='medium'` (full word) when
    the schema requires 'med'. The filler treats anything outside the
    {low, med, high} literal set as missing and replaces it with 'med'.
    Avoids the alternative of failing the entire view.
    """
    hf = HorizonForecast.model_validate({
        "h1": 0.001, "h5": 0.002, "h20": 0.005, "h60": 0.01,
        "confidence_h1": "medium",   # invalid literal
        "confidence_h5": "med",
        "confidence_h20": None,       # explicit null
        "confidence_h60": "high",
    })
    assert hf.confidence_h1 == "med"   # 'medium' → 'med'
    assert hf.confidence_h5 == "med"   # unchanged
    assert hf.confidence_h20 == "med"  # None → 'med'
    assert hf.confidence_h60 == "high" # unchanged


def test_pm_views_desc_still_asks_for_concrete_regime_note():
    """Even though regime_note has a default, the prompt should still
    encourage the apex to fill it in with a concrete tag."""
    desc = A._PM_VIEWS_DESC
    assert "regime_note" in desc
    assert "unspecified" in desc  # acknowledges default
    # And mentions concrete tags so the apex knows what to emit
    assert "breakout_failure" in desc or "event_driven_invalidation" in desc


def test_pm_output_desc_has_strengthened_hard_cap_directive():
    """Phase 2-fix: the smoke gate caught the apex emitting >100%
    allocations every day. The strengthened directive must include an
    emphatic hard-cap line and a worked example."""
    desc = A._PM_OUTPUT_DESC
    # The emphatic cap line
    assert "HARD CAP" in desc
    # The budget formula (with current_invested_pct)
    assert "current_invested_pct" in desc
    # A worked numeric example so the apex can pattern-match
    assert "30+30+30" in desc or "20+15+5" in desc
    # SELL-to-free-capacity reminder
    assert "SELL" in desc and "free capacity" in desc


def test_pm_output_desc_has_horizon_tier_size_anchors_not_legacy_20_cap():
    """Phase 2-fix-2: the v3.3 directive "size_pct ≤ 20 per name"
    conflicted with v4 horizon-tier bases (h20=30%, h60=50%) — the apex
    averaged the two anchors and landed in the 33-49% band, failing the
    ±5pp sizing validator on every BUY. Fix: delete the conflicting line
    and add EXPLICIT horizon-base anchors. Also: SELL-only-from-held
    reminder (smoke saw SELL on AMZN/UNH when not in positions)."""
    desc = A._PM_OUTPUT_DESC
    # The legacy conflicting directive must be gone.
    assert "≤ 20 per name" not in desc
    assert "<= 20 per name" not in desc
    assert "20 per name" not in desc
    # All four explicit tier-base anchors must be present.
    assert "h1" in desc and "3%" in desc
    assert "h5" in desc and "10%" in desc
    assert "h20" in desc and "30%" in desc
    assert "h60" in desc and "50%" in desc
    # A guardrail mentioning the ±10pp tolerance / between-tier rejection.
    # (widened from ±5pp in phase2-fix-3 because Qwen-4B couldn't hit ±5pp precision.)
    assert "10pp" in desc or "±10pp" in desc
    # The "SIZE BY HORIZON" block label so it's visually distinct in the prompt.
    assert "SIZE BY HORIZON" in desc
    # SELL-only-from-held reminder near the top of the order spec.
    # Two forms accepted: the new explicit reminder OR the existing
    # "never SELL a name not in open_positions" in the rules section.
    assert (
        "SELL only positions you currently hold" in desc
        or "SELL a name not in open_positions" in desc
    )
    # Stronger check: BOTH the new reminder line AND its placement
    # rationale should be present (waste-of-retry framing).
    assert "SELL only positions you currently hold" in desc


def test_order_size_field_description_uses_horizon_tier_bases():
    """Phase 2-fix-2: the Field description on Order.size_pct also
    carried the conflicting "≤20 per name" anchor. Update it to point at
    the horizon-tier bases so it agrees with _PM_OUTPUT_DESC."""
    # Pull the Field info from the Pydantic model.
    field = Order.model_fields["size_pct"]
    assert "≤20 per name" not in field.description
    assert "20 per name" not in field.description
    # Mentions at least one tier base (representative check).
    assert "h20=30" in field.description or "h20 → 30" in field.description


def test_ticker_view_round_trip():
    hf = _make_forecast()
    tv = TickerView(
        ticker="AAPL", forecasts=hf,
        primary_horizon="h20", rationale="earnings tailwind through h20",
    )
    blob = tv.model_dump()
    rebuilt = TickerView.model_validate(blob)
    assert rebuilt.ticker == "AAPL"
    assert rebuilt.primary_horizon == "h20"
    assert rebuilt.forecasts.h60 == pytest.approx(0.05)


def test_order_with_new_required_fields_ok():
    o = Order(
        side="BUY", ticker="AAPL", size_pct=10.0,
        reasoning="bullish on h20 earnings",
        primary_horizon="h20", expected_alpha_bps=180.0,
    )
    assert o.primary_horizon == "h20"
    assert o.expected_alpha_bps == pytest.approx(180.0)


def test_horizon_base_size_constants_present():
    """The committee + validators consume HORIZON_BASE_SIZE; make sure
    the canonical values match the contract."""
    assert HORIZON_BASE_SIZE == {"h1": 3.0, "h5": 10.0, "h20": 30.0, "h60": 50.0}


# ── Order field constraints ────────────────────────────────────────────

def test_order_rejects_missing_primary_horizon():
    with pytest.raises(ValidationError):
        Order(
            side="BUY", ticker="X", size_pct=10.0, reasoning="r",
            expected_alpha_bps=100.0,
        )


def test_order_defaults_missing_expected_alpha_bps_to_zero():
    """Updated 2026-05-10 (run6 day-18): missing alpha now defaults to
    0.0 for ANY side (previously raised for non-HOLD). Downstream
    `validate_edge_floor` will still reject any BUY/ROTATE that
    genuinely lacks conviction (floor > 0bps).
    """
    o = Order(
        side="BUY", ticker="X", size_pct=10.0, reasoning="r",
        primary_horizon="h5",
    )
    assert o.expected_alpha_bps == 0.0


def test_order_rejects_bad_primary_horizon():
    with pytest.raises(ValidationError):
        Order(
            side="BUY", ticker="X", size_pct=10.0, reasoning="r",
            primary_horizon="h7",  # not in {h1,h5,h20,h60}
            expected_alpha_bps=100.0,
        )


def test_order_rejects_expected_alpha_bps_above_ceiling():
    with pytest.raises(ValidationError):
        Order(
            side="BUY", ticker="X", size_pct=10.0, reasoning="r",
            primary_horizon="h5",
            expected_alpha_bps=10_001.0,  # > 10_000
        )


def test_order_rejects_expected_alpha_bps_below_floor():
    with pytest.raises(ValidationError):
        Order(
            side="SELL", ticker="X", size_pct=10.0, reasoning="r",
            primary_horizon="h5",
            expected_alpha_bps=-10_001.0,  # < -10_000
        )


def test_order_accepts_negative_expected_alpha_bps_for_sell():
    o = Order(
        side="SELL", ticker="X", size_pct=10.0, reasoning="r",
        primary_horizon="h5", expected_alpha_bps=-200.0,
    )
    assert o.expected_alpha_bps == pytest.approx(-200.0)


def test_order_defaults_expected_alpha_to_zero_for_any_side():
    """Qwen-4B routinely emits orders without expected_alpha_bps. The Order
    schema defaults missing/None alpha to 0.0 for ANY side so a single
    truncated order doesn't poison the whole Phase-1/3/4 proposal.

    Updated 2026-05-10 (run6 day-18 sweep fix): the original defaulter
    only covered HOLD; a SELL on revised_orders[0] missing alpha killed
    the day. We now default for all sides — semantics preserved because
    the validator chain (`validate_edge_floor`, et al.) will reject any
    BUY/ROTATE that genuinely lacks forward conviction (floor > 0bps).
    SELL is exempt from edge_floor by validator design, so 0.0 is a
    fine default for a position-closing leg.
    """
    o_hold = Order(
        side="HOLD", ticker="X", size_pct=0.0, reasoning="hold",
        primary_horizon="h5",
    )
    assert o_hold.expected_alpha_bps == 0.0

    # SELL without expected_alpha_bps now defaults (was: raised) — the
    # canonical run6 day-18 failure mode.
    o_sell = Order(
        side="SELL", ticker="X", size_pct=100.0, reasoning="r",
        primary_horizon="h20",
    )
    assert o_sell.expected_alpha_bps == 0.0

    # BUY without expected_alpha_bps also defaults (was: raised). The
    # downstream validator chain will reject it via edge_floor; better
    # to let one order drop than to fail the whole proposal.
    o_buy = Order(
        side="BUY", ticker="X", size_pct=10.0, reasoning="r",
        primary_horizon="h5",
    )
    assert o_buy.expected_alpha_bps == 0.0

    # Explicit None for any side still triggers the default
    o_none = Order(
        side="HOLD", ticker="X", size_pct=0.0, reasoning="hold",
        primary_horizon="h5", expected_alpha_bps=None,
    )
    assert o_none.expected_alpha_bps == 0.0


# ── ApexPMResponse cross-field validation ──────────────────────────────

def test_apex_response_accepts_matching_views_and_orders():
    hf = _make_forecast()
    tv = TickerView(
        ticker="AAPL", forecasts=hf, primary_horizon="h20",
        rationale="r",
    )
    o = Order(
        side="BUY", ticker="AAPL", size_pct=10.0, reasoning="r",
        primary_horizon="h20", expected_alpha_bps=180.0,
    )
    resp = ApexPMResponse(views=[tv], orders=[o], rationale="thesis")
    assert resp.orders[0].ticker == "AAPL"


def test_apex_response_rejects_order_ticker_not_in_views():
    """Order on MSFT but views only covers AAPL → ValidationError."""
    hf = _make_forecast()
    tv = TickerView(
        ticker="AAPL", forecasts=hf, primary_horizon="h5",
        rationale="r",
    )
    o = Order(
        side="BUY", ticker="MSFT", size_pct=10.0, reasoning="r",
        primary_horizon="h5", expected_alpha_bps=200.0,
    )
    with pytest.raises(ValidationError) as excinfo:
        ApexPMResponse(views=[tv], orders=[o], rationale="")
    msg = str(excinfo.value)
    assert "MSFT" in msg
    assert "TickerView" in msg or "views" in msg


def test_apex_response_rejects_rotate_with_missing_to_ticker_view():
    hf = _make_forecast()
    tv_aapl = TickerView(
        ticker="AAPL", forecasts=hf, primary_horizon="h20", rationale="r",
    )
    # ROTATE AAPL -> MSFT but MSFT has no view
    o = Order(
        side="ROTATE", from_ticker="AAPL", to_ticker="MSFT", size_pct=10.0,
        reasoning="rotate into MSFT", primary_horizon="h20",
        expected_alpha_bps=150.0,
    )
    with pytest.raises(ValidationError) as excinfo:
        ApexPMResponse(views=[tv_aapl], orders=[o])
    assert "MSFT" in str(excinfo.value)


def test_apex_response_allows_hold_without_ticker():
    hf = _make_forecast()
    tv = TickerView(ticker="AAPL", forecasts=hf, primary_horizon="h5", rationale="r")
    o_hold = Order(
        side="HOLD", size_pct=0.0, reasoning="sit on hands",
        primary_horizon="h5", expected_alpha_bps=0.0,
    )
    # No exception: HOLD with no ticker is a no-op.
    resp = ApexPMResponse(views=[tv], orders=[o_hold])
    assert resp.orders[0].side == "HOLD"


def test_apex_response_empty_lists_valid():
    """No views, no orders: trivially consistent."""
    resp = ApexPMResponse(views=[], orders=[], rationale="hold cash")
    assert resp.orders == []
    assert resp.views == []


# ── Tax-framing tokens present in PM prompts ───────────────────────────

REQUIRED_TAX_TOKENS = ["37%", "5bps", "net_profit", "h20", "h60"]


@pytest.mark.parametrize("sig_name", [
    "WatchlistFromObservationsPM",
    "WatchlistFromFirehosePM",
    "WatchlistOracle",
])
def test_pm_signature_docstring_has_tax_framing(sig_name):
    sig = getattr(A, sig_name)
    text = sig.__doc__ or ""
    missing = [t for t in REQUIRED_TAX_TOKENS if t not in text]
    assert not missing, (
        f"{sig_name} docstring missing tax-framing tokens: {missing}\n"
        f"Got docstring head: {text[:200]!r}"
    )


def test_pm_signature_has_dip_and_recover_example():
    """The worked dip-and-recover example is the headline pedagogy of the
    tax block — make sure it survives prompt edits."""
    for sig_name in (
        "WatchlistFromObservationsPM",
        "WatchlistFromFirehosePM",
        "WatchlistOracle",
    ):
        sig = getattr(A, sig_name)
        text = sig.__doc__ or ""
        assert "dip" in text.lower() or "recover" in text.lower(), (
            f"{sig_name} missing dip-and-recover worked example"
        )


# ── PM signature output fields ─────────────────────────────────────────

@pytest.mark.parametrize("sig_name", [
    "WatchlistFromFirehose",
    "WatchlistFromObservations",
    "WatchlistFromFirehosePM",
    "WatchlistFromObservationsPM",
    "WatchlistOracle",
])
def test_all_five_signatures_emit_views_and_orders(sig_name):
    sig = getattr(A, sig_name)
    fields = sig.output_fields
    assert "views" in fields, f"{sig_name} missing `views` OutputField"
    assert "orders" in fields, f"{sig_name} missing `orders` OutputField"


# ── PortfolioState profit_definition ───────────────────────────────────

def test_portfolio_state_has_profit_definition_default():
    ps = PortfolioState(equity=100_000.0, cash=50_000.0, invested_pct=50.0)
    assert "net_profit" in ps.profit_definition
    assert "tax_owed" in ps.profit_definition


def test_state_for_apex_populates_profit_definition():
    """ApexPortfolio.state_for_apex must emit profit_definition on the
    PortfolioState it returns."""
    from trophic.agents.apex_portfolio import ApexPortfolio
    p = ApexPortfolio(label="test", starting_cash=100_000.0)
    p.advance_prices("2024-01-02", [{"ticker": "AAPL", "actual_return": 0.0}])
    ps, _positions = p.state_for_apex(
        "2024-01-02", {"AAPL": 100.0},
    )
    assert ps.profit_definition  # non-empty
    assert "net_profit" in ps.profit_definition
    # Recent realized PnL + tax accrual should be present (default zero).
    assert ps.recent_realized_pnl_5d == pytest.approx(0.0)
    assert ps.tax_owed_accrued == pytest.approx(0.0)
