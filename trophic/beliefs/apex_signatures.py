"""DSPy signatures for the apex tier — used by BOTH the bare-Qwen baseline
and the ecology pipeline. Same input fields where possible, same output
schema, so scoring is symmetric.

The bare path inputs the raw firehose. The ecology path inputs the
carnivore's pre-ranked Observation set. Both emit a Watchlist.
"""
from __future__ import annotations

import dspy


# ── Shared output schema ──────────────────────────────────────────────

class WatchlistEntry(dspy.Signature):
    """Schema reference (informational — DSPy uses field types not subclasses)."""
    pass


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

    watchlist: list[dict] = dspy.OutputField(
        desc=(
            "Ranked watchlist of actionable tickers. Each entry has keys: "
            "ticker (str), action (BUY|SELL|WATCH), p_up (float 0-1), "
            "reason (≤200 chars). Order: highest conviction first. "
            "Omit tickers with no signal unless they are in focus_tickers."
        )
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
    observations: list[dict] = dspy.InputField(
        desc=(
            "Pre-ranked Observations from the carnivore. Each has ticker, action, "
            "p_up, salience, confidence, reasoning_summary, causal_chain (list of "
            "contributing belief→outcome edges), conflicting_signals."
        )
    )

    watchlist: list[dict] = dspy.OutputField(
        desc=(
            "Ranked watchlist of actionable tickers. Each entry has keys: "
            "ticker (str), action (BUY|SELL|WATCH), p_up (float 0-1), "
            "reason (≤200 chars). Order: highest conviction first."
        )
    )
