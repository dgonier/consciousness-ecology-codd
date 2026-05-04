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
