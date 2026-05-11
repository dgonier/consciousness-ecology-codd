"""DSPy signatures for the apex tier — used by BOTH the bare-Qwen baseline
and the ecology pipeline. Same input fields where possible, same output
schema, so scoring is symmetric.

The bare path inputs the raw firehose. The ecology path inputs the
carnivore's pre-ranked Observation set. Both emit a Watchlist (legacy)
or Orders (PM mode).

DSPy + Pydantic best practice: declare typed output fields with
Pydantic BaseModels so DSPy's JSONAdapter generates a *strict* JSON
schema with required fields. Without this, frontier models (Opus etc.)
take 'list[dict]' literally and return empty objects — Sonnet and
smaller models tend to ignore the loose schema and infer structure
from natural-language docstrings, which masks the issue.
"""
from __future__ import annotations

from typing import Literal, Optional

import dspy
from pydantic import BaseModel, Field, model_validator


# ── Multi-horizon constants (v4) ───────────────────────────────────────

HORIZON_BASE_SIZE: dict[str, float] = {
    "h1":  3.0,   # probe
    "h5":  10.0,  # satellite
    "h20": 30.0,  # core
    "h60": 50.0,  # conviction-core
}


# ── Shared output schema ──────────────────────────────────────────────

class WatchlistEntry(BaseModel):
    """Legacy non-PM watchlist entry — used by WatchlistFromFirehose /
    WatchlistFromObservations."""
    ticker: str = Field(..., description="Ticker symbol; must be in the universe.")
    action: Literal["BUY", "SELL", "WATCH"] = Field(
        ..., description="BUY when bullish, SELL when bearish, WATCH otherwise.",
    )
    p_up: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability the stock closes higher tomorrow vs today.",
    )
    reason: str = Field(
        default="",
        description="One-sentence rationale (≤200 chars).",
    )


class Order(BaseModel):
    """Portfolio-manager order. side discriminates the shape:
      - BUY:    requires ticker, size_pct
      - SELL:   requires ticker, size_pct
      - HOLD:   ticker optional; size_pct ignored
      - ROTATE: requires from_ticker, to_ticker, size_pct (NOT ticker)
    Reasoning is always required.
    """
    side: Literal["BUY", "SELL", "HOLD", "ROTATE"] = Field(
        ..., description="Order type. BUY/SELL operate on `ticker`; ROTATE "
                         "swaps `from_ticker` → `to_ticker`; HOLD is a no-op.",
    )
    ticker: Optional[str] = Field(
        default=None,
        description="Required for BUY/SELL/HOLD. Must be in universe. "
                    "Leave null for ROTATE (use from_ticker/to_ticker instead).",
    )
    from_ticker: Optional[str] = Field(
        default=None, alias="from",
        description="ROTATE only: ticker to SELL. Must be in open_positions.",
    )
    to_ticker: Optional[str] = Field(
        default=None, alias="to",
        description="ROTATE only: ticker to BUY. Must be in universe.",
    )
    size_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percent of CURRENT EQUITY for this order. "
                    "BUY: % of equity to allocate; use the EXACT horizon-tier base "
                    "(h1=3, h5=10, h20=30, h60=50) ±10pp. "
                    "SELL: % of EXISTING POSITION to close (use 100 to fully exit; "
                    "trims must be ≥5%). "
                    "ROTATE: % of equity to swap. "
                    "HOLD: ignored, set 0.",
    )
    reasoning: str = Field(
        default="unspecified", min_length=1,
        description="Why this order makes sense given today's signals + portfolio. "
                    "Keep it concise (≤200 chars ideal) — long reasoning eats tokens "
                    "and slows retries — but no hard cap. Default 'unspecified' is "
                    "allowed but uninformative — the prompt should still ask for a "
                    "concrete rationale; this default exists only so a Qwen-4B "
                    "truncation that drops the field can't poison the whole proposal "
                    "(run7 day-33 sweep fail: revised_orders[4].reasoning missing on "
                    "a HOLD order).",
    )
    primary_horizon: Literal["h1", "h5", "h20", "h60"] = Field(
        ...,
        description="Committed holding horizon for this order. "
                    "h1=1 day (probe), h5=5 days (satellite), h20=20 days (core), "
                    "h60=~3 months (conviction-core). Tax-aware validators use this "
                    "to enforce a minimum holding period and waive the edge floor "
                    "for h60 trades.",
    )
    expected_alpha_bps: float = Field(
        ..., ge=-10000.0, le=10000.0,
        description="Expected return in basis points over the committed primary_horizon, "
                    "net of slippage. Positive for BUY when bullish, negative for SELL "
                    "when bearish. Must clear ~162bps on h1 flips to overcome tax + "
                    "slippage; h60 trades waive that floor.",
    )

    class Config:
        populate_by_name = True   # accepts both 'from' and 'from_ticker'

    @model_validator(mode="before")
    @classmethod
    def _default_alpha_on_missing(cls, data):
        """Qwen-4B routinely emits orders without `expected_alpha_bps`.
        Pre-fix this raised a strict `Field required` error and dropped
        the entire proposal. We now default to 0.0 for any side that
        doesn't strictly need a forward-alpha target.

        Per-side rationale:
          - HOLD: no alpha target by definition (no capital moves).
            Always default to 0.0 (the original fix).
          - SELL: closing-leg has no forward alpha; the past conviction
            is already realized. `validate_edge_floor` ignores SELL,
            so 0.0 is downstream-safe. (Run6 day-18 sweep fail:
            `revised_orders[0].expected_alpha_bps: Field required` on
            a SELL.)
          - BUY / ROTATE: a missing alpha here IS a missing signal.
            Default to 0.0 anyway — `validate_edge_floor` will reject
            the order (floor >0), so the order silently drops without
            inventing a forward view, but the rest of the proposal
            survives. This is preferable to poisoning the whole
            proposal with a strict ValidationError.
        """
        if isinstance(data, dict):
            if "expected_alpha_bps" not in data or data.get("expected_alpha_bps") is None:
                data["expected_alpha_bps"] = 0.0
        return data


# ── Multi-horizon forecasting (v4) ─────────────────────────────────────

class HorizonForecast(BaseModel):
    """Per-ticker forward-return forecast across four committed horizons.

    The apex commits a single ticker-level forecast to FOUR horizons
    (h1/h5/h20/h60) plus a confidence label per horizon. The chosen
    `primary_horizon` on the corresponding TickerView determines which
    one drives sizing.
    """
    h1:  float = Field(..., description="Expected return (decimal, not bps) over 1 trading day.")
    h5:  float = Field(..., description="Expected return over 5 trading days.")
    h20: float = Field(..., description="Expected return over 20 trading days.")
    h60: float = Field(..., description="Expected return over ~3 months (window-end horizon).")
    confidence_h1:  Literal["low", "med", "high"]
    confidence_h5:  Literal["low", "med", "high"]
    confidence_h20: Literal["low", "med", "high"]
    confidence_h60: Literal["low", "med", "high"]
    regime_note: str = Field(
        default="unspecified", min_length=1,
        description="One-sentence regime label: trend | mean_revert | breakout | range | event_driven | unclear. "
                    "Default 'unspecified' is allowed but uninformative — prefer a concrete tag so the "
                    "min-hold validator can detect breakout_failure / event_driven_invalidation overrides.",
    )

    @model_validator(mode="before")
    @classmethod
    def _flatten_nested_forecasts(cls, data):
        """Qwen-4B occasionally emits the entire forecast dict nested
        inside ONE horizon's value — e.g. `h1 = {"h1": 0.001, "h5":
        0.005, "h20": 0.02, "h60": 0.05}` instead of `h1 = 0.001`. When
        we detect a horizon field holding a dict, pull out the matching
        key and copy any sibling horizons up so the flat-shape passes.
        Mirrors the Phase-2 list-shape normalization in
        `trophic.beliefs.debate` (also fix #f for Qwen-4B output drift).
        """
        if not isinstance(data, dict):
            return data
        for h in ("h1", "h5", "h20", "h60"):
            v = data.get(h)
            if isinstance(v, dict):
                inner = v.get(h)
                if isinstance(inner, (int, float)):
                    data[h] = float(inner)
                # Copy up sibling horizons that the nested dict carries,
                # but only if the outer slot is missing / not a number.
                for other_h in ("h1", "h5", "h20", "h60"):
                    if other_h == h:
                        continue
                    if other_h in v and isinstance(v[other_h], (int, float)):
                        if (
                            data.get(other_h) is None
                            or not isinstance(data.get(other_h), (int, float))
                        ):
                            data[other_h] = float(v[other_h])
        return data

    @model_validator(mode="before")
    @classmethod
    def _fill_missing_confidences(cls, data):
        """Qwen-4B run3 day-3 sweep drift: forecasts dict emitted with all
        four numeric horizons (h1/h5/h20/h60) but missing the four
        `confidence_h*` fields entirely. Pre-fix this poisoned the whole
        Phase-1 proposal at `views.<i>.forecasts.confidence_h1: Field
        required`.

        Filling with "med" is the sensible-conservative default:
          - "low" understates the predator's view (it DID emit numbers).
          - "high" overstates it (Qwen-4B may have just truncated).
          - "med" is the middle, used downstream as an advisory weight.

        Already-correct dicts pass through unchanged. Non-dict input is
        returned untouched so strict validation can raise its standard
        error.
        """
        if not isinstance(data, dict):
            return data
        for h in ("h1", "h5", "h20", "h60"):
            key = f"confidence_{h}"
            if key not in data or data[key] not in ("low", "med", "high"):
                data[key] = "med"
        return data


class TickerView(BaseModel):
    """The apex's committed view for one ticker: forecasts at every horizon,
    a chosen primary horizon (which the Order references), and a free-text
    rationale tying the chosen horizon to the regime + expected edge.
    """
    ticker: str = Field(..., min_length=1, max_length=10)
    forecasts: HorizonForecast
    primary_horizon: Literal["h1", "h5", "h20", "h60"]
    rationale: str = Field(
        default="unspecified", min_length=1,
        description="Why primary_horizon was chosen and what edge is expected over it. "
                    "Default 'unspecified' is allowed but uninformative — the prompt "
                    "should still ask for a concrete rationale; this default exists only "
                    "so a Qwen-4B truncation that drops the field can't poison the entire "
                    "Phase-1 proposal (run5 day-9 sweep fail: views[0].rationale missing).",
    )


class ApexPMResponse(BaseModel):
    """Wrapper composing the apex's portfolio-manager response.

    Carries both the per-ticker views and the concrete orders. A
    `model_validator` enforces that every Order's ticker has a matching
    TickerView (an Order's primary_horizon must come from a ticker the
    apex actually has an opinion on in the same response). ROTATE orders
    are checked on both legs (from_ticker AND to_ticker).
    """
    views: list[TickerView] = Field(default_factory=list)
    orders: list["Order"] = Field(default_factory=list)
    rationale: str = Field(default="", description="High-level allocation thesis (≤300 chars).")

    @model_validator(mode="after")
    def _every_order_ticker_has_a_view(self) -> "ApexPMResponse":
        view_tickers = {v.ticker.upper() for v in self.views}
        for o in self.orders:
            # HOLD orders without a ticker are no-ops — skip.
            if o.side == "HOLD" and not o.ticker:
                continue
            if o.side == "ROTATE":
                needed = []
                if o.from_ticker:
                    needed.append(o.from_ticker.upper())
                if o.to_ticker:
                    needed.append(o.to_ticker.upper())
                for t in needed:
                    if t not in view_tickers:
                        raise ValueError(
                            f"Order ROTATE references ticker {t!r} but no TickerView "
                            f"in `views` covers it; every order must commit to a "
                            f"horizon-forecast view in the same response. "
                            f"Available views: {sorted(view_tickers)}"
                        )
                continue
            t = (o.ticker or "").upper()
            if not t:
                raise ValueError(
                    f"Order side={o.side!r} missing ticker; every non-ROTATE order "
                    f"must name a ticker present in `views`."
                )
            if t not in view_tickers:
                raise ValueError(
                    f"Order {o.side} on ticker {t!r} has no matching TickerView; "
                    f"every order's ticker must appear in `views`. "
                    f"Available views: {sorted(view_tickers)}"
                )
        return self


class CausalChainLink(BaseModel):
    """One belief→outcome contribution from the carnivore's chain."""
    parent_id: Optional[str] = None
    p_now: Optional[float] = None
    p_prior: Optional[float] = None
    delta: Optional[float] = None
    link_dir: Optional[str] = None
    link_strength: Optional[float] = None
    contribution_log_odds: Optional[float] = None
    species: Optional[str] = None


class ObservationInput(BaseModel):
    """Carnivore observation (input shape — model only reads these)."""
    ticker: str
    action: Optional[str] = None
    p_up: Optional[float] = None
    salience: Optional[float] = None
    confidence: Optional[str] = None
    reasoning_summary: Optional[str] = None
    causal_chain: Optional[list[CausalChainLink]] = None
    conflicting_signals: Optional[list[CausalChainLink]] = None


class PortfolioState(BaseModel):
    """Snapshot of the portfolio the apex sees before deciding."""
    equity: float = Field(..., description="Total mark-to-market in dollars.")
    cash: float = Field(..., description="Idle cash in dollars (earns yield).")
    invested_pct: float = Field(..., ge=0.0, le=100.0)
    tax_owed_accrued: float = 0.0
    recent_realized_pnl_5d: float = 0.0
    cash_yield_annual_pct: float = 0.0
    cash_yield_today: float = 0.0
    cash_yield_ytd: float = 0.0
    profit_definition: str = Field(
        default="net_profit = realized_pnl - tax_owed - slippage + cash_yield",
        description="What 'profit' means for scoring this portfolio. Realized "
                    "gains incur 37% short-term cap gains tax + 5bps slippage "
                    "per leg; cash yield (cash_yield_annual_pct) accrues "
                    "risk-free; the apex should optimize NET, not gross. "
                    "Longer-horizon trades (h20/h60) defer tax realization and "
                    "softens the per-trade slippage drag.",
    )


class PositionSnapshot(BaseModel):
    ticker: str
    weight_pct: float = Field(..., ge=0.0, le=100.0)
    unrealized_pnl_pct: float = 0.0
    days_held: int = 0


class TomorrowReturn(BaseModel):
    """ORACLE-mode foresight per ticker (single-day; DEPRECATED in v4).

    The v4 oracle uses TickerForwardReturns (full-window matrix) instead.
    """
    ticker: str
    actual_return: float = Field(
        ..., description="Tomorrow's close-to-close return as a decimal "
                         "(e.g. 0.0123 = +1.23%).",
    )


class TickerForwardReturns(BaseModel):
    """One row of the full-window forward-returns matrix: a ticker plus the
    sequence of close-to-close returns from `today` (exclusive) through
    `today + days_remaining` (inclusive). Indexed in trading-day order.

    Used by `WatchlistOracleFullWindow` to give the apex foresight of the
    entire remaining window at once, so it can commit to an infrequent
    buy-low-sell-high trajectory instead of reacting per day.
    """
    ticker: str = Field(..., min_length=1, max_length=10)
    forward_returns: list[float] = Field(
        ..., min_length=1,
        description="Daily forward returns as decimals (e.g. 0.0123 = +1.23%) "
                    "from `today + 1 trading day` through window-end. Truncated "
                    "to 4 decimal places. Length must equal `days_remaining` "
                    "(or fewer if the ticker has missing days at window-end — "
                    "but the runner asserts length ≤ days_remaining).",
    )


class PreviousAttempt(BaseModel):
    """One prior attempt the validator rejected; included on retries."""
    orders: list[Order] = Field(
        default_factory=list,
        description="What you proposed on the previous attempt.",
    )
    rejections: list[str] = Field(
        default_factory=list,
        description="Validator's directive feedback. Read carefully and revise.",
    )


# ── Bare baseline: Qwen reads firehose directly ────────────────────────

class WatchlistFromFirehose(dspy.Signature):
    """You are an analyst. Given today's market firehose (news headlines and
    yesterday's bars for the watchlist universe), produce a ranked watchlist
    of stocks worth attention for the next trading session.

    For each watchlist entry: ticker, action (BUY/SELL/WATCH), p_up
    (your estimated probability that the stock closes higher tomorrow vs
    yesterday's close, in [0,1]), and a 1-sentence reason.

    BIDIRECTIONAL: BUY when bullish evidence dominates (p_up > 0.55), SELL
    when bearish evidence dominates (p_up < 0.45), WATCH when mixed or
    inconclusive. Negative news (earnings miss, lawsuits, guidance cuts,
    declining margins) → SELL. Don't default to BUY.

    Output ONLY the most actionable subset — not all 20 universe tickers.
    Stocks with no meaningful signal should be omitted.

    If `focus_tickers` is non-empty, you MUST include those tickers in the
    output even if signal is weak — they are user-prioritized; emit
    your best directional read for each focus ticker.
    """
    universe: list[str] = dspy.InputField(desc="Watchlist universe (~20 tickers)")
    focus_tickers: list[str] = dspy.InputField(
        desc="User-prioritized tickers — must appear in output regardless of signal strength"
    )
    news_headlines: list[str] = dspy.InputField(
        desc="Today's news firehose (headlines + descriptions, may include multi-ticker articles)"
    )
    prev_bars_summary: str = dspy.InputField(
        desc="Compact summary of yesterday's bars (close, volume, range) for each universe ticker"
    )

    watchlist: list[WatchlistEntry] = dspy.OutputField(
        desc="Ranked watchlist of actionable tickers (highest conviction first). "
             "Omit tickers with no signal unless they are in focus_tickers.",
    )
    views: list[TickerView] = dspy.OutputField(
        desc="Per-ticker multi-horizon (h1/h5/h20/h60) forecasts for every "
             "ticker in `watchlist`. Use the same ticker symbols.",
    )
    orders: list[Order] = dspy.OutputField(
        desc="Concrete orders (BUY/SELL/HOLD/ROTATE) sized as % of equity. "
             "Each must include primary_horizon and expected_alpha_bps and "
             "reference a ticker present in `views`.",
    )


# ── Ecology path: Qwen reads carnivore Observations ────────────────────

class WatchlistFromObservations(dspy.Signature):
    """You are an analyst reviewing pre-synthesized Observations from the
    belief network. Each Observation already has a ticker, action,
    estimated p_up, salience score, and causal_chain showing which beliefs
    drove the prediction.

    Your job is to RECONCILE the Observations into a final watchlist:
      - Drop observations where the causal chain is weak or all conflicting
      - Keep observations with strong, multi-driver, low-conflict chains
      - Adjust p_up if the causal chain doesn't actually support the value
      - Always include focus_tickers in output (best read even if no
        salient observation exists for them)

    BIDIRECTIONAL: respect the carnivore's action (BUY when p_up > 0.55,
    SELL when p_up < 0.45). The carnivore already discriminated — do not
    flip every SELL to BUY. If the causal chain shows bearish drivers
    (earnings_miss, guidance_cut, ceo_departure, lawsuit, support_broken),
    keep it SELL. Don't default to BUY.

    The Observations are already filtered by salience; you are NOT
    expected to invent new tickers, only to curate and reconcile.
    """
    universe: list[str] = dspy.InputField(desc="Watchlist universe (~20 tickers)")
    focus_tickers: list[str] = dspy.InputField(
        desc="User-prioritized tickers — must appear in output regardless of signal strength"
    )
    observations: list[ObservationInput] = dspy.InputField(
        desc="Pre-ranked Observations from the carnivore — already filtered by salience.",
    )

    watchlist: list[WatchlistEntry] = dspy.OutputField(
        desc="Ranked watchlist of actionable tickers (highest conviction first).",
    )
    views: list[TickerView] = dspy.OutputField(
        desc="Per-ticker multi-horizon (h1/h5/h20/h60) forecasts for every "
             "ticker in `watchlist`. Use the same ticker symbols.",
    )
    orders: list[Order] = dspy.OutputField(
        desc="Concrete orders (BUY/SELL/HOLD/ROTATE) sized as % of equity. "
             "Each must include primary_horizon and expected_alpha_bps and "
             "reference a ticker present in `views`.",
    )


# ── Portfolio-manager variants (apex sizes orders, not just flags) ─────

_PM_TAX_FRAMING_BLOCK = """\

TAX & SLIPPAGE — WHAT 'NET PROFIT' MEANS

  net_profit = realized_pnl - tax_owed - slippage + cash_yield

  - Short-term cap gains: realized gains pay 37% tax (you only see 63%
    of the gross dollar move).
  - Slippage: 5bps per trade leg (10bps round-trip on a flip). On a
    typical day-trade you need ~162bps of edge after slippage just to
    break even AFTER tax.
  - Cash yield: idle cash earns ~4% annualized risk-free. A new BUY
    only beats cash if the annualized expected return clears that.

HOW THIS SHAPES STRATEGY

  - h1 flips (1-day holds) are the WORST tax-wise. Every realized gain
    pays 37%. Reserve h1 for high-edge, time-sensitive setups.
  - h20 trades (20-day holds) defer realization and amortize slippage
    across a bigger move. Better tax economics.
  - h60 trades (~3 months) are most tax-efficient. The tax-aware
    validator waives the 162bps edge floor for h60 orders — long
    holding periods themselves earn the cost back through deferral.

WORKED EXAMPLE: dip-and-recover

  You expect AAPL to dip -2% over the next 3 days then recover +5% over
  the rest of the month — h20 view: +3% net move.
    - DAY-TRADER PATH: SELL on the dip (-2%, realize loss → no tax
      benefit immediately), BUY back lower, ride to +5%. Net realized
      flow: ~+3% pre-tax, but ~+1.9% after 37% tax on the recovery leg
      AND ~20bps slippage from the round-trip flip.
    - HOLD-THROUGH PATH (primary_horizon='h20'): sit through the dip,
      let the +3% net compound, realize once at h20 exit. Tax bill on
      one realized gain instead of two; slippage paid once instead of
      twice. Net keeps closer to the +3% gross.
  CONCLUSION: when you forecast a multi-day recovery, commit
  primary_horizon='h20' or 'h60' and HOLD through the dip — don't
  day-trade the noise.

"""


_PM_OUTPUT_DESC = (
    "List of orders. Each entry has these shapes: "
    "  BUY:    {side: 'BUY', ticker, size_pct, reasoning} "
    "  SELL:   {side: 'SELL', ticker, size_pct, reasoning} "
    "  HOLD:   {side: 'HOLD', ticker, reasoning} "
    "  ROTATE: {side: 'ROTATE', from: TICKER_TO_SELL, to: TICKER_TO_BUY, "
    "          size_pct, reasoning} — atomic SELL+BUY of the same %. "
    "          Use ROTATE when you want to swap position A for position B "
    "          without risking the BUY getting clamped while the SELL "
    "          settles. The validator processes ROTATE as a unit. "
    "size_pct: number 0-100, percent of CURRENT EQUITY for the order. "
    "\n\nSELL only positions you currently hold. The `positions` input "
    "lists what you own. Proposing SELL on a name NOT in `positions` is "
    "invalid and will be rejected — waste of a retry slot. "
    "\n\n*** HARD CAP — READ TWICE *** "
    "The sum of all BUY size_pct values across this response MUST be "
    "≤ (100 − current_invested_pct). This is the cash budget available "
    "for deployment today. Orders that violate this cap will be "
    "REJECTED EN MASSE by the budget scaler and you will lose the "
    "entire turn. There is no partial fill — exceed the cap, lose "
    "the trade. "
    "WORKED EXAMPLE: if portfolio.invested_pct = 60, you have 40 "
    "percentage points of cash to deploy. If you want to BUY three "
    "names, valid sizings include 20+15+5 or 20+20+0(=drop) — NOT "
    "30+30+30 (that's 90 pp on a 40 pp budget; all three rejected). "
    "If your conviction on a fourth name exceeds the remaining "
    "budget, the correct move is to add a SELL on the weakest "
    "currently-held name FIRST to free capacity, then the BUY. "
    "Use ROTATE when the SELL and BUY are paired on the same thesis. "
    "*** END HARD CAP *** \n\n"
    "*** SIZE BY HORIZON *** "
    "Set size_pct to the EXACT tier base for the chosen primary_horizon, "
    "NOT halfway between tiers: "
    "  - h1  → 3% "
    "  - h5  → 10% "
    "  - h20 → 30% "
    "  - h60 → 50% "
    "Values within ±10pp of these bases are acceptable; values that land "
    "between tiers (e.g., 20%, 40%) will be rejected as 'sizing_mismatch'. "
    "*** END SIZE BY HORIZON *** \n\n"
    "Other hard rules: never recommend BUY if cash_available is too low "
    "(use ROTATE or add a SELL); never SELL a name not in open_positions; "
    "SELL trims must be ≥5% (smaller is tax/slippage drag). "
    "Keep total orders ≤ 8 — small, decisive moves beat lots of tiny ones. "
    "EVERY order MUST include primary_horizon ∈ {h1,h5,h20,h60} and "
    "expected_alpha_bps (basis points of expected return over that horizon, "
    "after slippage; negative for SELL). The ticker on every order MUST "
    "appear in the `views` output."
)


_PM_VIEWS_DESC = (
    "Per-ticker forecasts for every name you have an opinion on (NOT all 20 — "
    "only names you actually evaluated). Each TickerView has: "
    "  ticker (string, must match an order's ticker), "
    "  forecasts (h1/h5/h20/h60 decimal returns + low/med/high confidence per "
    "  horizon + one-sentence regime_note), "
    "  primary_horizon (which horizon drives the order — pick the one your "
    "  edge is strongest on), "
    "  rationale (why primary_horizon and what the expected edge is). "
    "EVERY ticker that appears in `orders` must have a corresponding entry "
    "here. If you only have an opinion at h60, still emit forecasts for "
    "h1/h5/h20 — they signal that the shorter horizons are NOT where you see "
    "edge (use 0.0 with confidence_h1='low'). The committee + validators use "
    "the full forecast curve. "
    "Always set regime_note on every HorizonForecast; default 'unspecified' "
    "is accepted but uninformative — prefer a concrete tag (trend, "
    "mean_revert, breakout, range, event_driven, breakout_failure, "
    "event_driven_invalidation, unclear) so the min-hold validator can "
    "detect explicit invalidations on early sells."
)

_PM_PORTFOLIO_DESC = (
    "Current portfolio state (BLIND to the sibling pipeline's portfolio): "
    "{equity (float, $), cash (float, $), invested_pct (float 0-100), "
    "tax_owed_accrued (float, $ — already netted from equity), "
    "recent_realized_pnl_5d (float, $), "
    "cash_yield_annual_pct (float — what idle cash earns annualized), "
    "cash_yield_today (float, $ — yield credited overnight), "
    "cash_yield_ytd (float, $ — cumulative yield earned since start)}. "
    "CASH IS A YIELDING POSITION, not a dead bucket. It earns "
    "cash_yield_annual_pct risk-free. A new BUY only beats cash if your "
    "conviction implies >cash_yield_annual_pct annualized expected return. "
    "When evidence is weak or mixed, HOLDING cash is genuinely "
    "positive-EV, not a wasted slot. invested_pct + sum(new BUYs) ≤ 100. "
    "Going to 100% invested means giving up the risk-free yield — only do "
    "it when you have at least 5 high-conviction names."
)

_PM_POSITIONS_DESC = (
    "Open positions, one entry per held name: "
    "{ticker, weight_pct (% of equity), unrealized_pnl_pct, days_held}. "
    "Empty list means flat (all cash)."
)

_PM_PRIOR_ATTEMPTS_DESC = (
    "Previous attempts at sizing today's orders that the validator rejected. "
    "Empty list on the first attempt. On a retry, each entry is "
    "{orders: [...what you proposed...], rejections: ['reason 1', ...]}. "
    "Read the rejections CAREFULLY. They are directive — the validator "
    "names specific tickers and gives you concrete remedies. Examples: "
    "  - 'BUY total 70% blocked: portfolio is 100% invested with only "
    "0.0% budget. To free capacity, add a SELL on one of: AAPL (w=15%, "
    "unreal=-2.1%), MSFT (w=10%, unreal=+0.4%)' — pick the weakest one "
    "and add a SELL to your revised orders, then keep the BUY. "
    "  - 'AAPL: BUY blocked — already at per-name cap (20% of equity). "
    "Either drop this BUY or SELL part of AAPL first.' — drop the order "
    "or first SELL part of AAPL. "
    "  - 'MSFT: SELL on unheld name' — switch to HOLD or BUY. "
    "Stock-to-stock trades go through cash; you must SELL first to free "
    "cash, then BUY. Do NOT repeat a rejected order verbatim — fix the "
    "stated reason or drop it."
)


class WatchlistFromFirehosePM(dspy.Signature):
    __doc__ = """You are a portfolio manager (not just an analyst). Given today's
    market firehose, the universe, and your CURRENT portfolio state, emit
    concrete buy/sell/hold orders sized as percentages of equity, EACH
    committed to a primary_horizon ∈ {h1, h5, h20, h60}.

    Your job changes from 'flag interesting tickers' to 'allocate capital'.
    That means:
      - Opportunity cost matters. BUYing X means not BUYing Y. Be picky.
      - Don't add to a position that's already at its per-name cap (20%).
      - If invested_pct is already high (>80%), prefer HOLD or SELL the
        weakest existing names before adding new ones.
      - Bearish-decisive evidence on a held name → SELL (size_pct=100 to
        fully exit, smaller to trim).
      - When uncertain, HOLD. The default action is to do nothing.
      - Never go to margin (sum of new BUY size_pct must respect cash).

    BIDIRECTIONAL: BUY when bullish evidence dominates (p_up > 0.55), SELL
    when bearish evidence dominates (p_up < 0.45) AND the name is held,
    HOLD otherwise. Don't default to BUY.

    DO NOT mention or assume anything about a sibling/competing portfolio —
    you only see your own state.
    """ + _PM_TAX_FRAMING_BLOCK
    universe: list[str] = dspy.InputField(desc="Watchlist universe (~20 tickers)")
    focus_tickers: list[str] = dspy.InputField(
        desc="User-prioritized tickers — emit your best read for each"
    )
    news_headlines: list[str] = dspy.InputField(
        desc="Today's news firehose (headlines + descriptions)"
    )
    prev_bars_summary: str = dspy.InputField(
        desc="Compact summary of yesterday's bars per universe ticker"
    )
    portfolio_state: PortfolioState = dspy.InputField(desc=_PM_PORTFOLIO_DESC)
    open_positions: list[PositionSnapshot] = dspy.InputField(desc=_PM_POSITIONS_DESC)
    previous_attempts: list[PreviousAttempt] = dspy.InputField(desc=_PM_PRIOR_ATTEMPTS_DESC)

    views: list[TickerView] = dspy.OutputField(desc=_PM_VIEWS_DESC)
    orders: list[Order] = dspy.OutputField(desc=_PM_OUTPUT_DESC)
    rationale: str = dspy.OutputField(
        desc="≤300 chars: high-level allocation thesis for today (what you "
             "are net rotating into/out of, and why, including which "
             "horizon (h1/h5/h20/h60) you are skewing toward."
    )


_PM_TOMORROW_RETURNS_DESC = (
    "PERFECT FORESIGHT — actual close-to-close returns for tomorrow. "
    "List of {ticker, actual_return} for every name in the universe. "
    "Use this to allocate optimally: BUY names with positive return, "
    "SELL held names with negative return. Per-name cap (20%) and total "
    "exposure cap (100%) still apply, so even with the future you must "
    "rank candidates and pick the best ~5. SELL anything in your current "
    "open_positions whose tomorrow return is negative — that's a free "
    "loss avoided. Do not BUY a name with negative tomorrow return; even "
    "small negative names are worse than holding cash (which earns "
    "cash_yield_today risk-free)."
)


# DEPRECATED in v4 — replaced by WatchlistOracleFullWindow.
# Retained for back-compat / contrast tests; the runner no longer dispatches
# the per-day Oracle. The v4 sweep uses WatchlistOracleFullWindow exclusively
# because per-day foresight produced reactive trading and tax wiped out the
# realized alpha (v3.3 postmortem: ORACLE-OPUS finished +0.17% net after
# +10.17% pre-tax). See tasks_v4/00-README.md and the mission file at
# tasks_v4/completed/phase2-B-02-full-window-oracle.md for the rationale.
class WatchlistOracle(dspy.Signature):
    __doc__ = """ORACLE PORTFOLIO MANAGER — you can see tomorrow's actual returns.

    Your input includes `tomorrow_returns` — the realized close-to-close
    move for every ticker in the universe. Your job is to convert that
    perfect foresight into optimally sized orders, respecting the same
    rules as the non-oracle PM:
      - Per-name cap: 20% of equity max.
      - Total exposure: ≤100% (no margin).
      - Must SELL existing positions before BUYing if you're already at
        100% invested — even with the future known.
      - Cash earns cash_yield_today risk-free; only BUY names whose
        tomorrow return is positive AND larger than today's daily yield.
      - SELL all held names with negative tomorrow_return (avoid losses).

    Greedy strategy: rank universe by tomorrow_return, BUY top names up
    to per-name cap until 100% invested or until tomorrow_return drops
    below the cash yield threshold. Trim/SELL anything in open_positions
    whose tomorrow_return is negative.

    For per-day Oracle (this signature), each order's primary_horizon
    will typically be 'h1' (you only see one day ahead) and
    expected_alpha_bps should be 10_000 * tomorrow_return for that
    ticker. Forecasts at h5/h20/h60 can mirror h1 or be 0.0 with
    confidence_h5='low' etc. — you only have h1 foresight.
    """ + _PM_TAX_FRAMING_BLOCK
    universe: list[str] = dspy.InputField(desc="Watchlist universe (~20 tickers)")
    focus_tickers: list[str] = dspy.InputField(
        desc="User-prioritized tickers — sized using foresight."
    )
    tomorrow_returns: list[TomorrowReturn] = dspy.InputField(desc=_PM_TOMORROW_RETURNS_DESC)
    portfolio_state: PortfolioState = dspy.InputField(desc=_PM_PORTFOLIO_DESC)
    open_positions: list[PositionSnapshot] = dspy.InputField(desc=_PM_POSITIONS_DESC)
    previous_attempts: list[PreviousAttempt] = dspy.InputField(desc=_PM_PRIOR_ATTEMPTS_DESC)

    views: list[TickerView] = dspy.OutputField(desc=_PM_VIEWS_DESC)
    orders: list[Order] = dspy.OutputField(desc=_PM_OUTPUT_DESC)
    rationale: str = dspy.OutputField(
        desc="≤300 chars: how you ranked tomorrow's returns and which "
             "names you picked / dropped."
    )


# ── Full-window Oracle (v4) ────────────────────────────────────────────
#
# v3.3 postmortem: a per-day Oracle (above) trades reactively — 384-721
# trades over 66 days, tax + slippage ate every dollar of realized alpha
# (ORACLE-OPUS: +10.17% pre-tax → +0.17% net). The fix is structural:
# show the apex EVERY ticker's daily forward return for the rest of the
# window at once and tell it to commit to a trajectory of infrequent
# trades, not react bar-by-bar.

_PM_FUTURE_RETURNS_MATRIX_DESC = (
    "FULL-WINDOW FORESIGHT — for every ticker in `watchlist`, the daily "
    "close-to-close return from today+1 through today+days_remaining. "
    "Shape: list of {ticker, forward_returns: [r_t+1, r_t+2, ..., r_window_end]}. "
    "Returns are decimals truncated to 4 places (e.g. 0.0123 = +1.23%). "
    "Each `forward_returns` array has length ≤ days_remaining (never longer — "
    "the matrix is hard-capped at window-end to avoid leaking past your task). "
    "USE THE MATRIX to plan a trajectory of FEW, INFREQUENT, LARGER trades "
    "across the entire remaining window. Do not react day-by-day."
)


_PM_PRUNED_TICKERS_DESC = (
    "Tickers that the runner dropped from `future_returns_matrix` to keep the "
    "prompt under the token budget. Empty list when nothing was pruned. The "
    "runner keeps the top-N most volatile names because that's where the "
    "buy-low-sell-high alpha is; the dropped tickers tend to be quiet drifters. "
    "If a pruned ticker is in `open_positions` you can still HOLD it — you "
    "just lack the per-day curve to time a trim."
)


_PM_ORACLE_FULL_WINDOW_STRATEGY_BLOCK = """\

FULL-WINDOW ORACLE — STRATEGY

You are a PORTFOLIO MANAGER with PERFECT FORESIGHT of the remaining window.
For every ticker in `watchlist`, you have foresight of every ticker's daily return for the rest of the window — `future_returns_matrix` carries the
exact close-to-close return for each ticker for each trading day from
tomorrow through window-end (length = `days_remaining`).

YOUR JOB IS NOT TO REACT TO EACH DAY. Your job is to commit TODAY to a
trajectory of infrequent, larger, longer-held positions that captures the
multi-day shape of each ticker's path.

CORE DIRECTIVES

  - Buy near local minima and sell near local maxima of each ticker's
    forward-returns trajectory. The cumulative product of forward_returns
    is the price-path forecast; find its troughs and peaks.
  - Prefer FEW, LARGE, LONG-HELD trades over MANY small flips. Tax + slippage
    crush short-horizon flips — see the tax-framing block below for the
    quantitative reason.
  - Output the MINIMUM NUMBER OF ORDERS that achieves your trajectory.
    Conservatively, expect <= 0.5 orders per day on average over the
    window (so for a 60-day window: aim for ~30 orders TOTAL across the
    sweep, not per day).

HORIZON SELECTION — pick the LONGEST horizon that fits the move

  - h60 (~3 months): use for buy-and-hold of any ticker whose cumulative
    forward-return trajectory trends up monotonically through window-end.
    h60 trades are the MOST tax-efficient — the validator waives the edge
    floor for them. This should be your DEFAULT horizon when you see a
    clean window-long trend.
  - h20 (20 days): use for swing trades INSIDE the window where you plan
    to buy at a trough, hold for ~a month, and sell at a peak. Good when
    the trajectory has one major swing.
  - h5 (5 days): use sparingly — only when you see a short-term
    mispricing the matrix shows recovering within a week.
  - h1 (1 day): LAST RESORT. Every h1 flip pays full 37% tax. Use only
    when the move is huge and isolated (e.g. > 5% in a single day with
    no follow-through).

ANTI-PATTERN: DO NOT SELL-ON-DIP

If you see a temporary dip in a name that recovers within 5 days, the
correct action is BUY-AND-HOLD-THROUGH-DIP, NOT sell-on-dip + rebuy.

  WORKED EXAMPLE — dip-and-recover
  Ticker X has forward_returns of [-0.005, -0.003, +0.004, +0.011, +0.020, ...].
  The dip is -0.8% over the first 2 days; the recovery is +3.5% over the
  next 3 days. Net 5-day move: +2.7%.
    - CORRECT: BUY today with primary_horizon='h5', hold through the dip,
      realize +2.7% at h5 exit (or h20 if you can ride further). One
      realized gain, one slippage round-trip.
    - WRONG: BUY today, SELL on day 2 to "avoid" the dip, REBUY day 3
      lower, SELL day 5. Realizes TWO short-term gains (each taxed 37%)
      and pays TWO slippage round-trips (~20bps). After tax + slippage
      this nets ~+1.5% — almost half the buy-and-hold path.

PORTFOLIO MECHANICS (unchanged from non-oracle PM)

  - Per-name cap: 20% of equity max.
  - Total exposure: ≤100% (no margin).
  - Must SELL existing positions before BUYing if you're at 100% invested.
  - Cash earns cash_yield_today risk-free; only BUY names whose forward
    trajectory beats cash on a risk-adjusted basis over their committed
    horizon.
  - Every order MUST commit to primary_horizon ∈ {h1, h5, h20, h60} and
    expected_alpha_bps = 10_000 * forecast_return_over_that_horizon.

"""


class WatchlistOracleFullWindow(dspy.Signature):
    __doc__ = """FULL-WINDOW ORACLE PORTFOLIO MANAGER. SUMMARY: you have foresight of every ticker's daily return for the rest of the window.

    You receive `future_returns_matrix`: per ticker, the exact daily
    close-to-close return from today+1 through today+days_remaining. Use
    it to commit NOW to a single buy-low-sell-high trajectory of
    infrequent, longer-held trades — not to react bar-by-bar.

    The full strategy directive (buy near local minima / sell near local
    maxima, horizon-selection rubric, anti-sell-on-dip rule with worked
    example, and the minimum-orders / ≤ 0.5 orders per day target) lives
    in the strategy block below. The tax-framing block (37% / 5bps /
    net_profit / h20 / h60 / dip-and-recover) is inherited verbatim from
    the non-oracle PM signatures.
    """ + _PM_ORACLE_FULL_WINDOW_STRATEGY_BLOCK + _PM_TAX_FRAMING_BLOCK

    today: str = dspy.InputField(
        desc="Today's date as ISO string (YYYY-MM-DD). The first entry of "
             "each ticker's forward_returns array is the close-to-close "
             "return from today's close to tomorrow's close."
    )
    days_remaining: int = dspy.InputField(
        desc="Number of trading days remaining in the window, INCLUDING today. "
             "Each ticker's forward_returns has length ≤ days_remaining. "
             "Use this to scope your trajectory math — at days_remaining=60 "
             "an h60 trade fills the whole window; at days_remaining=10 the "
             "longest useful horizon is h5/h20."
    )
    portfolio_state: PortfolioState = dspy.InputField(desc=_PM_PORTFOLIO_DESC)
    positions: list[PositionSnapshot] = dspy.InputField(desc=_PM_POSITIONS_DESC)
    watchlist: list[str] = dspy.InputField(
        desc="Tickers covered by `future_returns_matrix`. Subset of the universe; "
             "may be pruned to top-N most volatile names if the full matrix "
             "exceeded the token budget."
    )
    future_returns_matrix: list[TickerForwardReturns] = dspy.InputField(
        desc=_PM_FUTURE_RETURNS_MATRIX_DESC
    )
    pruned_tickers: list[str] = dspy.InputField(desc=_PM_PRUNED_TICKERS_DESC)
    previous_attempts: list[PreviousAttempt] = dspy.InputField(desc=_PM_PRIOR_ATTEMPTS_DESC)

    views: list[TickerView] = dspy.OutputField(desc=_PM_VIEWS_DESC)
    orders: list[Order] = dspy.OutputField(desc=_PM_OUTPUT_DESC)
    rationale: str = dspy.OutputField(
        desc="≤300 chars: trajectory thesis. Which names are window-long h60 "
             "holds vs. h20 swings, which dips you are holding through, and "
             "the approximate total-order count you are committing to."
    )


class WatchlistFromObservationsPM(dspy.Signature):
    __doc__ = """You are a portfolio manager reviewing pre-synthesized Observations
    from the belief network and your CURRENT portfolio state. Emit concrete
    buy/sell/hold orders sized as percentages of equity, EACH committed to
    a primary_horizon ∈ {h1, h5, h20, h60}.

    Each Observation already carries a ticker, suggested action, p_up,
    salience, confidence, and causal_chain. Your job is to convert those
    into capital allocation:
      - Where the causal chain is strong AND the ticker is not yet held
        at cap, BUY with size_pct proportional to conviction.
      - Where a held ticker has bearish-decisive observations or a weak
        causal chain, SELL (size_pct=100 to exit, smaller to trim).
      - Drop observations whose causal chain is weak or all-conflicting —
        do not act on them at all (no order needed; HOLD is implicit).
      - Never go to margin. Respect the per-name 20% cap.
      - When in doubt, HOLD. Default action is to do nothing.

    DO NOT invent tickers outside the observations + open_positions. You
    are sizing decisions, not generating new picks.

    BIDIRECTIONAL: respect the carnivore's action sign. BUY when p_up > 0.55,
    SELL on held names when p_up < 0.45, HOLD otherwise. Don't default to BUY.
    """ + _PM_TAX_FRAMING_BLOCK
    universe: list[str] = dspy.InputField(desc="Watchlist universe (~20 tickers)")
    focus_tickers: list[str] = dspy.InputField(
        desc="User-prioritized tickers — emit your best read for each"
    )
    observations: list[ObservationInput] = dspy.InputField(
        desc="Pre-ranked Observations from the carnivore."
    )
    portfolio_state: PortfolioState = dspy.InputField(desc=_PM_PORTFOLIO_DESC)
    open_positions: list[PositionSnapshot] = dspy.InputField(desc=_PM_POSITIONS_DESC)
    previous_attempts: list[PreviousAttempt] = dspy.InputField(desc=_PM_PRIOR_ATTEMPTS_DESC)

    views: list[TickerView] = dspy.OutputField(desc=_PM_VIEWS_DESC)
    orders: list[Order] = dspy.OutputField(desc=_PM_OUTPUT_DESC)
    rationale: str = dspy.OutputField(
        desc="≤300 chars: high-level allocation thesis for today."
    )
