"""Render a Scenario + herb broadcasts into an EvidencePacket.

The evidence packet is the trophic stack's *output* to the apex tier —
all the prepared analysis from producers and herbivores rolled into a
single readable artifact, plus machine-readable metadata that the
aggregator can use for confidence weighting.

The text format is designed so any frontier model (Gemini, GPT, Claude,
Qwen) can read it without per-model prompt engineering. The structured
metadata lives in EvidencePacket.metadata for aggregation logic.
"""
from __future__ import annotations

import math
from typing import Iterable

from .base import EvidencePacket


SYSTEM = (
    "You are a short-horizon market direction predictor. Read the evidence"
    " below — recent OHLCV bars, same-day news/tweets, and a numeric"
    " forecast snapshot — and predict whether the stock's NEXT trading-day"
    " close will be UP or DOWN. Be specific and use the ticker shown."
    "\n\n"
    "Respond with EXACTLY this two-section format and NOTHING ELSE:"
    "\n\n"
    "REASONING: <2-4 sentences citing specific evidence: which OHLCV bars,"
    " which tweets, which forecast features informed your decision.>"
    "\n\n"
    "<prediction><ticker>SYMBOL</ticker>"
    "<direction>up|down</direction>"
    "<horizon_min>1440</horizon_min>"
    "<confidence>0.55</confidence></prediction>"
)


def _render_ohlcv_block(scenario, ticker: str) -> str:
    parts: list[str] = []
    for inp in scenario.inputs:
        if inp.source != "ohlcv":
            continue
        bars = inp.payload.get("bars", [])
        if not bars:
            continue
        parts.append(f"Recent OHLCV history for {ticker} ({len(bars)} bars):")
        for b in bars[-8:]:  # last 8 bars
            parts.append(
                f"  open={b.get('open', 0):.6f}  "
                f"high={b.get('high', 0):.6f}  "
                f"low={b.get('low', 0):.6f}  "
                f"close={b.get('close', 0):.6f}  "
                f"vol={int(b.get('volume', 0))}"
            )
    return "\n".join(parts)


def _render_press_block(scenario, ticker: str) -> str:
    parts: list[str] = []
    for inp in scenario.inputs:
        if inp.source != "press":
            continue
        body = (inp.payload.get("body") or "").strip()
        if not body:
            continue
        n_label = inp.payload.get("headline", "")
        parts.append(f"\nSame-day news/tweets for {ticker} ({n_label}):")
        parts.append(body[:2000])  # cap
    return "\n".join(parts)


def _render_quant_signals_block(scenario) -> str:
    """Compute and render deterministic technical indicators from the
    OHLCV bars. Gives the technical analyst something to interpret
    that ISN'T already in the raw bars or Chronos block — multi-
    timeframe direction agreement, OBV slope, 2σ-anomaly flags.

    All purely deterministic; cheap (one pass over the bars).
    """
    bars = None
    for inp in scenario.inputs:
        if inp.source == "ohlcv":
            bars = inp.payload.get("bars", [])
            if bars:
                break
    if not bars or len(bars) < 3:
        return ""

    closes = [float(b.get("close", 0.0)) for b in bars]
    opens = [float(b.get("open", 0.0)) for b in bars]
    highs = [float(b.get("high", 0.0)) for b in bars]
    lows = [float(b.get("low", 0.0)) for b in bars]
    vols = [float(b.get("volume", 0.0)) for b in bars]
    n = len(closes)

    def _direction_of(window: list[float]) -> str:
        if len(window) < 2 or window[0] == 0:
            return "flat"
        delta = (window[-1] - window[0]) / max(abs(window[0]), 1e-9)
        if delta > 0.005:
            return "up"
        if delta < -0.005:
            return "down"
        return "flat"

    # Multi-timeframe direction:
    short_dir = _direction_of(closes[-2:]) if n >= 2 else "flat"
    medium_dir = _direction_of(closes[-3:]) if n >= 3 else "flat"
    full_dir = _direction_of(closes)
    agreement = (
        "all aligned" if short_dir == medium_dir == full_dir and short_dir != "flat"
        else "mixed" if len({short_dir, medium_dir, full_dir}) > 1
        else "flat-leaning"
    )

    # On-balance volume slope: cumulative signed volume over the window.
    # Positive slope = accumulation, negative = distribution.
    obv = 0.0
    obv_series = [0.0]
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            obv += vols[i]
        elif closes[i] < closes[i - 1]:
            obv -= vols[i]
        obv_series.append(obv)
    if len(obv_series) >= 2 and abs(obv_series[-1]) > 1e-9:
        # Linear-fit slope, normalized
        x_mean = (n - 1) / 2.0
        y_mean = sum(obv_series) / n
        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(obv_series))
        den = sum((i - x_mean) ** 2 for i in range(n)) or 1.0
        obv_slope = num / den
        obv_dir = "accumulation" if obv_slope > 0 else "distribution" if obv_slope < 0 else "flat"
    else:
        obv_slope = 0.0
        obv_dir = "flat"

    # 2σ anomaly: was the last bar's range or volume an outlier vs prior?
    anomalies = []
    if n >= 5:
        prior_ranges = [highs[i] - lows[i] for i in range(n - 1)]
        last_range = highs[-1] - lows[-1]
        mean_r = sum(prior_ranges) / len(prior_ranges)
        std_r = (sum((r - mean_r) ** 2 for r in prior_ranges) / len(prior_ranges)) ** 0.5
        if std_r > 0 and abs(last_range - mean_r) > 2 * std_r:
            anomalies.append(f"last-bar range {last_range:.2f} > 2σ vs prior")
        prior_vols = vols[:-1]
        last_vol = vols[-1]
        if prior_vols:
            mean_v = sum(prior_vols) / len(prior_vols)
            std_v = (sum((v - mean_v) ** 2 for v in prior_vols) / len(prior_vols)) ** 0.5
            if std_v > 0 and (last_vol - mean_v) > 2 * std_v:
                anomalies.append(f"last-bar volume {last_vol:.0f} > 2σ vs prior (high)")
            elif std_v > 0 and (last_vol - mean_v) < -2 * std_v:
                anomalies.append(f"last-bar volume {last_vol:.0f} < 2σ vs prior (low)")

    # Last-bar body direction + size
    last_body = closes[-1] - opens[-1]
    body_pct = last_body / max(opens[-1], 1e-9) * 100 if opens[-1] else 0.0
    body_kind = "bullish" if body_pct > 0.2 else "bearish" if body_pct < -0.2 else "doji"

    parts = [
        "\nQUANT SIGNALS (computed from OHLCV bars; the technical analyst should interpret these):",
        f"  multi-timeframe direction: short={short_dir} mid={medium_dir} full={full_dir} → {agreement}",
        f"  on-balance volume: slope={obv_slope:+.0f} → {obv_dir}",
        f"  last bar: {body_kind} body ({body_pct:+.2f}% close vs open)",
    ]
    if anomalies:
        parts.append(f"  anomalies: {'; '.join(anomalies)}")
    else:
        parts.append("  anomalies: none (last bar within 2σ of prior)")
    return "\n".join(parts)


def _render_forecast_block(scenario) -> str:
    """Render Chronos numeric features into a human-readable snapshot."""
    feats = getattr(scenario, "forecaster_features", None)
    if not feats or len(feats) < 32:
        return ""
    # Layout from forecast_features.py:
    # [0..7] traj, [8..15] spread_traj, [16] drift, [17] sign_drift,
    # [18] abs_drift, [19] avg_spread, [20] final_spread, [21] snr,
    # ... [29] spread_growth, [30] confidence_weighted_dir, ...
    drift = feats[16]
    avg_spread = feats[19]
    snr = feats[21]
    monotonicity = feats[26] if len(feats) > 26 else 0.0
    parts = [
        "\nNumeric forecast snapshot (Chronos-Bolt, 12-step horizon):",
        f"  predicted drift: {drift:+.4f} (relative return)",
        f"  uncertainty (avg q10/q90 spread): {avg_spread:.4f}",
        f"  signal-to-noise: {snr:+.3f}",
        f"  monotonicity: {monotonicity:+.3f} (-1 down throughout, +1 up throughout)",
    ]
    return "\n".join(parts)


def _render_herb_summaries(herb_broadcasts: Iterable) -> tuple[str, list[dict]]:
    """Each herb broadcast is a vector + tag. We don't render the vector
    (apex voters can't read it directly), but we can summarize the
    `decoded_text` if the herbivore was a real agent (versus oracle)."""
    rendered: list[str] = []
    meta: list[dict] = []
    for b in herb_broadcasts:
        kind = getattr(b, "agent_kind", "?")
        text = getattr(b, "decoded_text", None)
        diet = list(getattr(b, "diet_tags", []) or [])
        if text:
            rendered.append(f"  [{kind}] {text[:300]}")
        meta.append({
            "kind": kind,
            "diet": diet,
            "abstained": getattr(b, "abstained", False),
            "has_text": bool(text),
        })
    if not rendered:
        return "", meta
    return "\nHerbivore syntheses:\n" + "\n".join(rendered), meta


def build_evidence_packet(
    scenario,
    herb_broadcasts: Iterable | None = None,
    agent_feedback=None,
) -> EvidencePacket:
    """Render scenario + (optional) herb broadcasts into an EvidencePacket.

    If herb_broadcasts is None, only producer-level evidence (OHLCV,
    tweets) and the forecast snapshot are rendered — that's the "no
    trophic stack" baseline form.

    `agent_feedback` (an AgentFeedback) is the decomposer's per-agent
    modulation. When present its prompt_modulation block is prepended to
    the user evidence so this specific voter sees its own history-derived
    hints. Different voters see different feedback (Hexis: per-agent,
    not global).
    """
    ticker = scenario.name.split("_")[2] if "_" in scenario.name else "?"
    blocks = [SYSTEM, ""]
    blocks.append(_render_ohlcv_block(scenario, ticker))
    blocks.append(_render_press_block(scenario, ticker))
    qb = _render_quant_signals_block(scenario)
    if qb:
        blocks.append(qb)
    fb = _render_forecast_block(scenario)
    if fb:
        blocks.append(fb)
    herb_text = ""
    herb_meta: list[dict] = []
    if herb_broadcasts:
        herb_text, herb_meta = _render_herb_summaries(herb_broadcasts)
        if herb_text:
            blocks.append(herb_text)
    if agent_feedback is not None and getattr(agent_feedback, "prompt_modulation", ""):
        # Prepend per-agent decomposer feedback right after the SYSTEM
        # block so the voter sees its own history-derived hints before
        # the new evidence. Different voters get different feedback.
        blocks.insert(1, "\n" + agent_feedback.prompt_modulation)
    text = "\n".join([b for b in blocks if b])
    return EvidencePacket(
        scenario_name=scenario.name,
        ticker=ticker,
        text=text,
        metadata={
            "has_forecast": bool(fb),
            "has_herb_text": bool(herb_text),
            "herbs": herb_meta,
            "has_agent_feedback": agent_feedback is not None,
        },
        agent_feedback=agent_feedback,
    )
