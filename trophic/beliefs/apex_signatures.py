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
from pydantic import BaseModel, Field


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
                    "BUY: % of equity to allocate (≤20 per name). "
                    "SELL: % of EXISTING POSITION to close (use 100 to fully exit; "
                    "trims must be ≥5%). "
                    "ROTATE: % of equity to swap. "
                    "HOLD: ignored, set 0.",
    )
    reasoning: str = Field(
        ..., min_length=1,
        description="Why this order makes sense given today's signals + portfolio. "
                    "Keep it concise (≤200 chars ideal) — long reasoning eats tokens "
                    "and slows retries — but no hard cap.",
    )

    class Config:
        populate_by_name = True   # accepts both 'from' and 'from_ticker'


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


class PositionSnapshot(BaseModel):
    ticker: str
    weight_pct: float = Field(..., ge=0.0, le=100.0)
    unrealized_pnl_pct: float = 0.0
    days_held: int = 0


class TomorrowReturn(BaseModel):
    """ORACLE-mode foresight per ticker."""
    ticker: str
    actual_return: float = Field(
        ..., description="Tomorrow's close-to-close return as a decimal "
                         "(e.g. 0.0123 = +1.23%).",
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


# ── Portfolio-manager variants (apex sizes orders, not just flags) ─────

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
    "Hard rules: size_pct ≤ 20 per name; sum of new BUY size_pct ≤ "
    "(100 − current_invested_pct); never recommend BUY if cash_available "
    "is too low (use ROTATE instead); never SELL a name not in "
    "open_positions; SELL trims must be ≥5% (smaller is tax/slippage drag). "
    "Keep total orders ≤ 8 — small, decisive moves beat lots of tiny ones."
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
    """You are a portfolio manager (not just an analyst). Given today's
    market firehose, the universe, and your CURRENT portfolio state, emit
    concrete buy/sell/hold orders sized as percentages of equity.

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
    """
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

    orders: list[Order] = dspy.OutputField(desc=_PM_OUTPUT_DESC)
    rationale: str = dspy.OutputField(
        desc="≤300 chars: high-level allocation thesis for today (what you "
             "are net rotating into/out of, and why)."
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


class WatchlistOracle(dspy.Signature):
    """ORACLE PORTFOLIO MANAGER — you can see tomorrow's actual returns.

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
    """
    universe: list[str] = dspy.InputField(desc="Watchlist universe (~20 tickers)")
    focus_tickers: list[str] = dspy.InputField(
        desc="User-prioritized tickers — sized using foresight."
    )
    tomorrow_returns: list[TomorrowReturn] = dspy.InputField(desc=_PM_TOMORROW_RETURNS_DESC)
    portfolio_state: PortfolioState = dspy.InputField(desc=_PM_PORTFOLIO_DESC)
    open_positions: list[PositionSnapshot] = dspy.InputField(desc=_PM_POSITIONS_DESC)
    previous_attempts: list[PreviousAttempt] = dspy.InputField(desc=_PM_PRIOR_ATTEMPTS_DESC)

    orders: list[Order] = dspy.OutputField(desc=_PM_OUTPUT_DESC)
    rationale: str = dspy.OutputField(
        desc="≤300 chars: how you ranked tomorrow's returns and which "
             "names you picked / dropped."
    )


class WatchlistFromObservationsPM(dspy.Signature):
    """You are a portfolio manager reviewing pre-synthesized Observations
    from the belief network and your CURRENT portfolio state. Emit concrete
    buy/sell/hold orders sized as percentages of equity.

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
    """
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

    orders: list[Order] = dspy.OutputField(desc=_PM_OUTPUT_DESC)
    rationale: str = dspy.OutputField(
        desc="≤300 chars: high-level allocation thesis for today."
    )
