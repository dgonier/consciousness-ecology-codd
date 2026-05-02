"""XML schema for trophic agent outputs + parser + rule-based reward.

Why XML: structurally isolates content tokens (ticker, direction, magnitude,
σ, confidence) from template tokens. A reward function can extract exact
field values and compute field-wise correctness without ambiguity. Also
sets up the interface for HEXIS-style ecological state writes — every
parsed field becomes a candidate state entry keyed by (species, agent_id,
field_name).

Schema (compact; designed to tokenize cleanly with Qwen tokenizer):

  Herbivore synthesis (technical / fundamental):
    <synthesis kind="technical|fundamental">
      <ticker>AAPL</ticker>
      <bias>up|down|flat|none</bias>
      <pct_move>+1.50</pct_move>            optional
      <signal>strong|moderate|weak</signal>  optional
      <confidence>0.70</confidence>
      <abstain>true|false</abstain>
    </synthesis>

  Forecaster synthesis:
    <synthesis kind="forecaster">
      <ticker>AAPL</ticker>
      <bias>up|down|flat</bias>
      <pct_move>+0.15</pct_move>
      <sigma_pct>1.72</sigma_pct>
      <confidence>0.70</confidence>
    </synthesis>

  Predator prediction (single-source):
    <prediction>
      <ticker>AAPL</ticker>
      <direction>up|down|flat</direction>
      <pct_move>+1.50</pct_move>            optional
      <horizon_min>60</horizon_min>
      <sigma_pct>1.72</sigma_pct>           optional
      <confidence>0.65</confidence>
      <evidence>
        <herbivore kind="technical">tape consistent</herbivore>
        ...
      </evidence>
    </prediction>

  Predator abstain:
    <prediction>
      <abstain>true</abstain>
      <confidence>0.0</confidence>
    </prediction>
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------- structured records ----------

@dataclass
class Synthesis:
    """Parsed herbivore synthesis."""
    kind: str                                  # technical | fundamental | forecaster
    ticker: str | None = None
    bias: str | None = None                    # up | down | flat | none
    pct_move: float | None = None
    sigma_pct: float | None = None
    signal: str | None = None
    confidence: float | None = None
    abstain: bool = False
    raw: str = ""                              # original text for debug


@dataclass
class EvidenceItem:
    kind: str                                  # which herbivore species
    text: str                                  # short justification


@dataclass
class Prediction:
    """Parsed predator prediction."""
    ticker: str | None = None
    direction: str | None = None
    pct_move: float | None = None
    horizon_min: int | None = None
    sigma_pct: float | None = None
    confidence: float | None = None
    abstain: bool = False
    evidence: list[EvidenceItem] = field(default_factory=list)
    raw: str = ""


# ---------- emitters (target generators) ----------

def emit_synthesis(
    kind: str,
    ticker: str | None = None,
    bias: str | None = None,
    pct_move: float | None = None,
    sigma_pct: float | None = None,
    signal: str | None = None,
    confidence: float | None = None,
    abstain: bool = False,
) -> str:
    if abstain:
        return (
            f'<synthesis kind="{kind}">'
            f'<abstain>true</abstain>'
            f'<confidence>0.00</confidence>'
            f'</synthesis>'
        )
    parts = [f'<synthesis kind="{kind}">']
    if ticker:
        parts.append(f'<ticker>{ticker}</ticker>')
    if bias:
        parts.append(f'<bias>{bias}</bias>')
    if pct_move is not None:
        sign = "+" if pct_move >= 0 else ""
        parts.append(f'<pct_move>{sign}{pct_move:.2f}</pct_move>')
    if sigma_pct is not None:
        parts.append(f'<sigma_pct>{sigma_pct:.2f}</sigma_pct>')
    if signal:
        parts.append(f'<signal>{signal}</signal>')
    if confidence is not None:
        parts.append(f'<confidence>{confidence:.2f}</confidence>')
    parts.append(f'<abstain>false</abstain>')
    parts.append('</synthesis>')
    return ''.join(parts)


def emit_prediction(
    ticker: str | None = None,
    direction: str | None = None,
    pct_move: float | None = None,
    horizon_min: int | None = None,
    sigma_pct: float | None = None,
    confidence: float | None = None,
    abstain: bool = False,
    evidence: list[tuple[str, str]] | None = None,
) -> str:
    """Build a predator prediction. evidence = list of (kind, justification)."""
    if abstain:
        return (
            f'<prediction>'
            f'<abstain>true</abstain>'
            f'<confidence>0.00</confidence>'
            f'</prediction>'
        )
    parts = ['<prediction>']
    if ticker:
        parts.append(f'<ticker>{ticker}</ticker>')
    if direction:
        parts.append(f'<direction>{direction}</direction>')
    if pct_move is not None:
        sign = "+" if pct_move >= 0 else ""
        parts.append(f'<pct_move>{sign}{pct_move:.2f}</pct_move>')
    if horizon_min is not None:
        parts.append(f'<horizon_min>{horizon_min}</horizon_min>')
    if sigma_pct is not None:
        parts.append(f'<sigma_pct>{sigma_pct:.2f}</sigma_pct>')
    if confidence is not None:
        parts.append(f'<confidence>{confidence:.2f}</confidence>')
    parts.append('<abstain>false</abstain>')
    if evidence:
        parts.append('<evidence>')
        for kind, text in evidence:
            # Escape XML special chars in text (cheap minimal escape)
            clean = (text.replace('&', '&amp;').replace('<', '&lt;')
                          .replace('>', '&gt;'))
            parts.append(f'<herbivore kind="{kind}">{clean}</herbivore>')
        parts.append('</evidence>')
    parts.append('</prediction>')
    return ''.join(parts)


# ---------- robust parser ----------

# Regexes are *forgiving* — accept content even if some tags are missing
# or malformed. Real model output during training will be partial XML.

_TAG_RE = re.compile(
    r'<(?P<tag>\w+)(?P<attrs>[^>]*)>(?P<inner>.*?)</(?P=tag)>',
    re.DOTALL,
)
_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def _parse_float(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip().rstrip('%').replace('+', '')
    try:
        return float(s)
    except ValueError:
        return None


def _parse_int(s: str | None) -> int | None:
    if s is None:
        return None
    try:
        return int(s.strip())
    except ValueError:
        # try float-then-int
        f = _parse_float(s)
        return int(f) if f is not None else None


def _parse_bool(s: str | None) -> bool:
    if s is None:
        return False
    return s.strip().lower() in ("true", "yes", "1")


def _extract_inner(text: str, tag: str) -> str | None:
    """Find the first <tag>...</tag> block and return inner text.

    Tolerant of:
      - Extra whitespace
      - Attributes on the open tag
      - Mismatched close tags (typo close like </pct_change> for <pct_move>):
        falls back to closing at the next '<' if the matching close isn't found
    """
    # Try strict match first.
    pat = re.compile(
        rf'<{tag}\b[^>]*>(.*?)</{tag}>',
        re.DOTALL,
    )
    m = pat.search(text)
    if m:
        return m.group(1).strip()

    # Fallback: find the open tag and read inner text until the next '<'
    # (stopping at any tag-like token, even a typo'd close).
    open_pat = re.compile(rf'<{tag}\b[^>]*>', re.DOTALL)
    om = open_pat.search(text)
    if om:
        start = om.end()
        # Read until the next '<' which (in this context) is most likely a
        # close tag, possibly typo'd.
        next_lt = text.find('<', start)
        if next_lt > start:
            inner = text[start:next_lt].strip()
            if inner:
                return inner
    return None


def _extract_attr(open_tag_text: str, attr: str) -> str | None:
    m = re.search(rf'\b{attr}="([^"]*)"', open_tag_text)
    return m.group(1) if m else None


def parse_synthesis(text: str) -> Synthesis:
    """Parse a herbivore synthesis. Returns a Synthesis object with whatever
    fields could be extracted; missing fields are None.
    """
    # Find the kind from the opening <synthesis kind="..."> tag.
    open_match = re.search(r'<synthesis\b([^>]*)>', text)
    kind = "unknown"
    if open_match:
        k = _extract_attr(open_match.group(0), "kind")
        if k:
            kind = k

    return Synthesis(
        kind=kind,
        ticker=_extract_inner(text, "ticker"),
        bias=_extract_inner(text, "bias"),
        pct_move=_parse_float(_extract_inner(text, "pct_move")),
        sigma_pct=_parse_float(_extract_inner(text, "sigma_pct")),
        signal=_extract_inner(text, "signal"),
        confidence=_parse_float(_extract_inner(text, "confidence")),
        abstain=_parse_bool(_extract_inner(text, "abstain")),
        raw=text,
    )


def parse_prediction(text: str) -> Prediction:
    """Parse a predator prediction. Tolerant of missing/extra tags.

    Issue #15 P1: also matches loose direction signals (e.g. "direction is up",
    ":up:" colons, "I predict UP") since trained models sometimes emit
    coherent natural-language predictions outside the strict XML schema.
    Strict XML wins if present; loose match is a fallback.
    """
    direction = _extract_inner(text, "direction")
    if not direction:
        # Fallback: scan for clear up/down direction markers in plain text.
        # Prefer explicit phrasing over single-word matches to avoid noise.
        # Issue #99: strip the "schema enumeration" prefix the model often
        # parrots ("PREDICTION is either :up: or :down:.") because both
        # tokens appear there as schema description, not prediction.
        lower = (text or "").lower()
        # Drop everything up to the first sentence after schema-enumeration phrases.
        for echo in [
            "either :up: or :down:",
            "either up or down",
            "up|down",
            "(up|down|flat)",
        ]:
            idx = lower.find(echo)
            if idx >= 0:
                lower = lower[idx + len(echo):]

        # 2026-05-02: when the apex prompt ends with "DIRECTION:" (force-
        # prefix), the model response is just " up\n..." or " down\n...".
        # Match this leading-token shape — it's the canonical format under
        # the new prompt.
        leading = (text or "").lstrip()
        if leading[:5].lower().startswith("up") and (
            len(leading) <= 2 or not leading[2:3].isalpha()
        ):
            direction = "up"
        elif leading[:5].lower().startswith("down") and (
            len(leading) <= 4 or not leading[4:5].isalpha()
        ):
            direction = "down"

        # NOTE: previously had loose patterns like `direction\s*[:=is]+\s*up`
        # and `\bbullish\b` as natural-language fallbacks. Those patterns
        # over-fire on the trained model's schema-echo output ("DIRECTION
        # is up or down", "the price will go up or down", "bullish or
        # bearish"), turning every abstain into a false-positive 'up'.
        # On seed35 this drove MCC from 0 (correct: abstains everywhere)
        # to -0.12 (35 'up' false-positives out of 36 decisions). Removed.

    pred = Prediction(
        ticker=_extract_inner(text, "ticker"),
        direction=direction,
        pct_move=_parse_float(_extract_inner(text, "pct_move")),
        horizon_min=_parse_int(_extract_inner(text, "horizon_min")),
        sigma_pct=_parse_float(_extract_inner(text, "sigma_pct")),
        confidence=_parse_float(_extract_inner(text, "confidence")),
        abstain=_parse_bool(_extract_inner(text, "abstain")),
        raw=text,
    )
    # Pull evidence items if present.
    ev_block = _extract_inner(text, "evidence")
    if ev_block:
        for m in re.finditer(
            r'<herbivore\b([^>]*)>(.*?)</herbivore>', ev_block, re.DOTALL
        ):
            attrs, inner = m.group(1), m.group(2).strip()
            kind = _extract_attr(attrs, "kind") or "?"
            pred.evidence.append(EvidenceItem(kind=kind, text=inner))
    return pred


# ---------- rule-based reward ----------

@dataclass
class RewardConfig:
    """Weights for predator-prediction reward.

    Sums to 1.0 by convention but doesn't have to. The trainer treats this
    as the per-completion scalar reward.
    """
    ticker_w: float = 0.40
    direction_w: float = 0.25
    magnitude_w: float = 0.15
    horizon_w: float = 0.10
    sigma_w: float = 0.05
    confidence_w: float = 0.05
    parse_floor: float = 0.05    # awarded for at least producing valid XML
    magnitude_tolerance: float = 0.5  # ±0.5% pct counts as exact match
    sigma_tolerance: float = 0.5      # ±0.5% σ counts as exact


def reward_prediction(
    parsed: Prediction,
    target: Prediction,
    cfg: RewardConfig | None = None,
) -> dict:
    """Field-wise reward for a parsed prediction against an oracle target.

    Returns {'total': float, 'breakdown': {field: subreward}}. Total is
    bounded in [0, sum(weights)]. Use total directly as the RL reward.
    """
    cfg = cfg or RewardConfig()
    breakdown: dict[str, float] = {}
    total = 0.0

    # parse floor — got something parseable
    if parsed.ticker or parsed.direction or parsed.abstain:
        total += cfg.parse_floor
        breakdown["parse_floor"] = cfg.parse_floor

    # abstain agreement: if the target is abstain and the prediction is
    # too, that's effectively perfect for this task.
    if target.abstain:
        if parsed.abstain:
            return {"total": 1.0, "breakdown": {"abstain_match": 1.0}}
        # Predicted something when oracle expected abstention — penalize.
        return {"total": cfg.parse_floor, "breakdown": breakdown}

    # ticker
    if parsed.ticker and target.ticker and parsed.ticker.upper() == target.ticker.upper():
        total += cfg.ticker_w
        breakdown["ticker"] = cfg.ticker_w
    else:
        breakdown["ticker"] = 0.0

    # direction
    if parsed.direction and target.direction and parsed.direction.lower() == target.direction.lower():
        total += cfg.direction_w
        breakdown["direction"] = cfg.direction_w
    else:
        breakdown["direction"] = 0.0

    # magnitude — soft within tolerance, exponential falloff outside
    if parsed.pct_move is not None and target.pct_move is not None:
        diff = abs(parsed.pct_move - target.pct_move)
        if diff <= cfg.magnitude_tolerance:
            sub = cfg.magnitude_w
        else:
            # exp decay; at 2× tolerance, ~13% remaining
            import math
            sub = cfg.magnitude_w * math.exp(-(diff - cfg.magnitude_tolerance))
        total += sub
        breakdown["magnitude"] = sub
    elif target.pct_move is None:
        # Not required by target; don't penalize.
        breakdown["magnitude"] = 0.0
    else:
        # Target had a magnitude, model omitted it.
        breakdown["magnitude"] = 0.0

    # horizon
    if parsed.horizon_min is not None and target.horizon_min is not None:
        if parsed.horizon_min == target.horizon_min:
            total += cfg.horizon_w
            breakdown["horizon"] = cfg.horizon_w
        else:
            breakdown["horizon"] = 0.0
    else:
        breakdown["horizon"] = 0.0

    # sigma
    if parsed.sigma_pct is not None and target.sigma_pct is not None:
        diff = abs(parsed.sigma_pct - target.sigma_pct)
        if diff <= cfg.sigma_tolerance:
            sub = cfg.sigma_w
        else:
            import math
            sub = cfg.sigma_w * math.exp(-(diff - cfg.sigma_tolerance))
        total += sub
        breakdown["sigma"] = sub
    elif target.sigma_pct is None:
        breakdown["sigma"] = 0.0
    else:
        breakdown["sigma"] = 0.0

    # confidence — soft, penalty for being wildly off
    if parsed.confidence is not None and target.confidence is not None:
        diff = abs(parsed.confidence - target.confidence)
        if diff <= 0.2:
            sub = cfg.confidence_w * (1.0 - diff / 0.2)
            total += sub
            breakdown["confidence"] = sub
        else:
            breakdown["confidence"] = 0.0
    else:
        breakdown["confidence"] = 0.0

    return {"total": total, "breakdown": breakdown}


def reward_synthesis(
    parsed: Synthesis,
    target: Synthesis,
    cfg: RewardConfig | None = None,
) -> dict:
    """Field-wise reward for herbivore synthesis. Same shape as prediction
    but with synthesis-specific fields (no horizon).
    """
    cfg = cfg or RewardConfig()
    breakdown: dict[str, float] = {}
    total = 0.0

    if parsed.ticker or parsed.bias or parsed.abstain:
        total += cfg.parse_floor
        breakdown["parse_floor"] = cfg.parse_floor

    if target.abstain:
        if parsed.abstain:
            return {"total": 1.0, "breakdown": {"abstain_match": 1.0}}
        return {"total": cfg.parse_floor, "breakdown": breakdown}

    # Reuse the same field weights but redistribute horizon's weight to
    # ticker since synthesis has no horizon.
    if parsed.ticker and target.ticker and parsed.ticker.upper() == target.ticker.upper():
        total += cfg.ticker_w + cfg.horizon_w
        breakdown["ticker"] = cfg.ticker_w + cfg.horizon_w
    else:
        breakdown["ticker"] = 0.0

    if parsed.bias and target.bias and parsed.bias.lower() == target.bias.lower():
        total += cfg.direction_w
        breakdown["bias"] = cfg.direction_w
    else:
        breakdown["bias"] = 0.0

    if parsed.pct_move is not None and target.pct_move is not None:
        diff = abs(parsed.pct_move - target.pct_move)
        if diff <= cfg.magnitude_tolerance:
            sub = cfg.magnitude_w
        else:
            import math
            sub = cfg.magnitude_w * math.exp(-(diff - cfg.magnitude_tolerance))
        total += sub
        breakdown["magnitude"] = sub
    else:
        breakdown["magnitude"] = 0.0

    if parsed.sigma_pct is not None and target.sigma_pct is not None:
        diff = abs(parsed.sigma_pct - target.sigma_pct)
        sub = cfg.sigma_w * (1.0 if diff <= cfg.sigma_tolerance else 0.0)
        total += sub
        breakdown["sigma"] = sub
    else:
        breakdown["sigma"] = 0.0

    if parsed.confidence is not None and target.confidence is not None:
        diff = abs(parsed.confidence - target.confidence)
        if diff <= 0.2:
            sub = cfg.confidence_w * (1.0 - diff / 0.2)
            total += sub
            breakdown["confidence"] = sub
        else:
            breakdown["confidence"] = 0.0
    else:
        breakdown["confidence"] = 0.0

    return {"total": total, "breakdown": breakdown}
