"""Deterministic target generator (the "fake signal").

For each producer broadcast → target what an ideal herbivore synthesis
*should* say given that input. For each set of herbivore broadcasts → target
what the predator *should* predict.

The synthetic feed has full ground truth (it generated the input itself), so
we can write principled, deterministic templates per (producer_kind,
herbivore_kind) and per herbivore_kind → predator. These template strings
become the supervised labels — we tokenize them, forward Qwen on them, and
the resulting hidden states are the regression targets for the Channels.

Templates are intentionally simple. The point of v2 training is not to teach
Qwen to do good fintech analysis — it's to verify that the Channels can
learn to project producer hidden states into a region of Qwen's
input-embedding space such that the herbivore's downstream hidden state
matches a target. Once that mechanism works, sophistication in the targets
becomes a tunable.
"""
from __future__ import annotations

from typing import Iterable

from .types import Broadcast, RawInput


# ---------- target text templates ----------

def _direction(payload: dict) -> str:
    """Best-effort direction inference from a payload."""
    if "open" in payload and "close" in payload:
        try:
            o, c = float(payload["open"]), float(payload["close"])
            if c > o:
                return "up"
            if c < o:
                return "down"
            return "flat"
        except (TypeError, ValueError):
            return "flat"
    if payload.get("side") in ("B", "C"):
        return "up"
    if payload.get("side") in ("S", "P"):
        return "down"
    return "flat"


def _ticker(payload: dict) -> str:
    return str(payload.get("ticker", "UNK"))


def technical_target_for_input(inp: RawInput) -> str:
    t = _ticker(inp.payload)
    d = _direction(inp.payload)
    if inp.source == "ohlcv":
        vol = inp.payload.get("volume", 0)
        return (
            f"SYNTHESIS: {t} 1m bar shows {d} pressure with volume {vol};"
            f" near-term technical bias {d}. CONFIDENCE: 0.7"
        )
    if inp.source == "trades":
        side = inp.payload.get("side", "?")
        return (
            f"SYNTHESIS: {t} prints {side}-side aggressive flow; tape skew"
            f" {d}. CONFIDENCE: 0.6"
        )
    if inp.source == "book":
        return (
            f"SYNTHESIS: {t} order book balanced; no immediate microstructure"
            f" signal. CONFIDENCE: 0.4"
        )
    if inp.source in ("options", "halt"):
        return (
            f"SYNTHESIS: {t} {inp.source} event implies elevated volatility"
            f" risk; {d} bias. CONFIDENCE: 0.6"
        )
    return f"SYNTHESIS: {t} mixed technical signal. CONFIDENCE: 0.4"


def fundamental_target_for_input(inp: RawInput) -> str:
    t = _ticker(inp.payload)
    if inp.source == "filing":
        ftype = inp.payload.get("filing_type", "?")
        summary = inp.payload.get("summary", "")[:60]
        return (
            f"SYNTHESIS: {t} files {ftype} — {summary} Material event; revise"
            f" valuation accordingly. CONFIDENCE: 0.8"
        )
    if inp.source == "press":
        head = inp.payload.get("headline", "")[:60]
        return (
            f"SYNTHESIS: {t} press release: {head}. Event-driven re-pricing"
            f" likely. CONFIDENCE: 0.7"
        )
    if inp.source in ("options", "halt"):
        return (
            f"SYNTHESIS: {t} {inp.source} suggests information asymmetry,"
            f" possibly anticipating disclosure. CONFIDENCE: 0.6"
        )
    return f"SYNTHESIS: {t} no event-driven signal. CONFIDENCE: 0.3"


def herbivore_target(kind: str, inputs: Iterable[RawInput]) -> str:
    """Aggregate target across all inputs in a meal.

    For supervision we pool by *averaging hidden states* (done elsewhere),
    so any individual sentence is a valid target. We pick the first input as
    the target source for simplicity; the averaging in hidden space is what
    actually learns the aggregation.
    """
    inputs = list(inputs)
    if not inputs:
        return f"SYNTHESIS: <no substrate>. CONFIDENCE: 0.0"
    if kind == "technical":
        return technical_target_for_input(inputs[0])
    if kind == "fundamental":
        return fundamental_target_for_input(inputs[0])
    raise ValueError(kind)


def predator_target(herb_targets: Iterable[str]) -> str:
    """Templated short-horizon prediction grounded in herb syntheses.

    Picks the first non-trivial target and lifts it to a prediction.
    """
    for t in herb_targets:
        if "no substrate" in t:
            continue
        # Crude but deterministic.
        return (
            "PREDICTION: short-horizon directional bias supported by upstream"
            " syntheses; recommend monitoring named ticker for next 60 minutes."
            " CONFIDENCE: 0.6"
        )
    return "PREDICTION: insufficient signal; abstain. CONFIDENCE: 0.1"
