"""Carnivore tier — synthesize propagated belief network state into a
ranked Observation set for the apex.

Inputs:
  - state_beliefs: dict[id → StateBelief] (post-propagation, with current_p
    reflecting the pass's evidence)
  - outcome_beliefs: dict[id → OutcomeBelief] (post-propagation, p_up moved
    by chains from contributing leaves)
  - links: list[InternalLink] (the network edges)
  - triggering_events: list[Event] (the events this pass that started the
    cascade — for traceback in the Observation.triggering_events field)
  - watchlist_focus: list[str] | None (optional user-supplied bias toward
    these tickers; lowers their salience threshold)

Output: list[Observation], ranked by salience descending. The apex reads
the top-K.

Pure function — no I/O, no LLM. Deterministic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

from .schema import (
    Event,
    InternalLink,
    OutcomeBelief,
    StateBelief,
)


# ── Output dataclasses ────────────────────────────────────────────────

@dataclass
class CausalLink:
    """One edge of an Observation's causal chain."""
    parent_belief_id: str
    parent_statement: str
    parent_current_p: float
    parent_prior_p: float
    parent_deviation: float           # signed (current_p - prior_p)
    link_id: str
    link_direction: Literal["positive", "negative"]
    link_strength: float
    contribution_log_odds: float      # signed log-odds shift this link added to outcome
    species_id_of_origin: str         # which herb originally activated this leaf (best-effort)


@dataclass
class Observation:
    """One ticker-level synthesis the carnivore presents to the apex."""
    ticker: str
    action: Literal["BUY", "SELL", "WATCH"]
    p_up: float
    salience: float                   # 0 to inf — top-N by this for apex consumption
    horizon_min: int
    causal_chain: list[CausalLink]
    conflicting_signals: list[CausalLink]
    triggering_events: list[str]      # event ids
    confidence: Literal["low", "medium", "high"]
    reasoning_summary: str            # deterministic, ≤200 chars
    in_focus: bool = False            # bumped by watchlist_focus

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "action": self.action,
            "p_up": round(self.p_up, 4),
            "salience": round(self.salience, 4),
            "horizon_min": self.horizon_min,
            "in_focus": self.in_focus,
            "confidence": self.confidence,
            "reasoning_summary": self.reasoning_summary,
            "causal_chain": [
                {
                    "parent": c.parent_belief_id,
                    "statement": c.parent_statement,
                    "p_now": round(c.parent_current_p, 3),
                    "p_prior": round(c.parent_prior_p, 3),
                    "delta": round(c.parent_deviation, 3),
                    "link_dir": c.link_direction,
                    "link_strength": round(c.link_strength, 2),
                    "contribution_log_odds": round(c.contribution_log_odds, 3),
                    "species": c.species_id_of_origin,
                } for c in self.causal_chain
            ],
            "conflicting_signals": [
                {
                    "parent": c.parent_belief_id,
                    "statement": c.parent_statement,
                    "p_now": round(c.parent_current_p, 3),
                    "delta": round(c.parent_deviation, 3),
                    "contribution_log_odds": round(c.contribution_log_odds, 3),
                } for c in self.conflicting_signals
            ],
            "triggering_events": list(self.triggering_events),
        }


# ── Helpers ────────────────────────────────────────────────────────────

def _log_odds(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1 - p))


def _action_from_p_up(p_up: float, threshold_buy: float = 0.58,
                      threshold_sell: float = 0.42) -> str:
    if p_up >= threshold_buy:
        return "BUY"
    if p_up <= threshold_sell:
        return "SELL"
    return "WATCH"


def _confidence_from_chain(top_contrib_abs_log_odds: float,
                            n_contributors: int,
                            n_conflicting: int) -> str:
    """Confidence is high when there's strong, multi-contributor, mostly-
    aligned evidence; low when sparse or conflicting."""
    if n_contributors == 0:
        return "low"
    # Penalize high conflicting count
    conflict_ratio = n_conflicting / max(1, n_contributors + n_conflicting)
    score = top_contrib_abs_log_odds * math.sqrt(n_contributors) * (1 - conflict_ratio)
    if score >= 1.5:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def _summarize_chain(
    ticker: str,
    p_up: float,
    causal: list[CausalLink],
    conflicting: list[CausalLink],
    action: str,
) -> str:
    """Deterministic 1-sentence summary from the chain."""
    parts = [f"{ticker} {action} p_up={p_up:.2f}"]
    if causal:
        # Take top-2 drivers
        for c in causal[:2]:
            tag = c.parent_belief_id.replace("belief.company.", "").replace("belief.market.", "mkt.")
            tag = tag.split("__")[0]
            sign = "↑" if c.parent_deviation > 0 else "↓"
            parts.append(f"{sign}{tag}({c.parent_deviation:+.2f}×{c.link_strength:.2f})")
    if conflicting:
        parts.append(f"+{len(conflicting)} conflict")
    return "; ".join(parts)[:200]


# ── Main aggregator ────────────────────────────────────────────────────

def aggregate(
    state_beliefs: dict[str, StateBelief],
    outcome_beliefs: dict[str, OutcomeBelief],
    links: Iterable[InternalLink],
    *,
    triggering_events: Optional[list[Event]] = None,
    watchlist_focus: Optional[list[str]] = None,
    species_origin: Optional[dict[str, str]] = None,
    salience_floor: float = 0.02,
    focus_salience_floor: float = 0.005,
    top_chain_k: int = 7,
    conflict_threshold_log_odds: float = 0.05,
    horizon_min_default: int = 24 * 60,
) -> list[Observation]:
    """Build one Observation per outcome that has meaningful deviation.

    Args:
      species_origin: optional map belief_id → species_id of the herb that
        originally activated it. Threaded through to the CausalLink for
        audit.
      salience_floor: outcomes whose salience falls below this are dropped.
      focus_salience_floor: lower floor for tickers in watchlist_focus.
      top_chain_k: max length of causal_chain (rest go to "rest").
      conflict_threshold_log_odds: a contributing link counts as
        "conflicting" if its sign is opposite the net outcome direction
        AND |contribution| ≥ this threshold.
    """
    species_origin = species_origin or {}
    triggering_events = triggering_events or []
    focus_set = set(watchlist_focus or [])

    # Index links by conclusion (target) for fast outcome reverse-lookup
    incoming: dict[str, list[InternalLink]] = {}
    for l in links:
        incoming.setdefault(l.conclusion_belief_id, []).append(l)

    # Derive triggering event-id set (events to attach to all outcomes —
    # carnivore doesn't currently distinguish per-ticker triggers; refine
    # later via species_origin tracebacks).
    trigger_ids = [e.id for e in triggering_events]

    out: list[Observation] = []
    for outcome_id, outcome in outcome_beliefs.items():
        in_links = incoming.get(outcome_id, [])
        if not in_links:
            continue
        # Build CausalLinks for each contributing parent
        all_contribs: list[CausalLink] = []
        for link in in_links:
            parent = state_beliefs.get(link.premise_belief_id)
            if parent is None:
                continue
            dev = parent.current_p - parent.prior_p
            if abs(dev) < 1e-6:
                continue
            sign = +1.0 if link.direction == "positive" else -1.0
            lo_delta = _log_odds(parent.current_p) - _log_odds(parent.prior_p)
            contribution = link.strength_posterior * sign * lo_delta
            all_contribs.append(CausalLink(
                parent_belief_id=parent.id,
                parent_statement=parent.statement_template,
                parent_current_p=parent.current_p,
                parent_prior_p=parent.prior_p,
                parent_deviation=dev,
                link_id=link.id,
                link_direction=link.direction,  # type: ignore[arg-type]
                link_strength=link.strength_posterior,
                contribution_log_odds=contribution,
                species_id_of_origin=species_origin.get(parent.id, ""),
            ))

        if not all_contribs:
            continue

        # Net outcome direction from p_up
        net_up = outcome.p_up >= 0.5

        # Split into causal (aligned with net direction) vs conflicting
        causal: list[CausalLink] = []
        conflicting: list[CausalLink] = []
        for c in all_contribs:
            aligned = (c.contribution_log_odds > 0) == net_up
            if aligned:
                causal.append(c)
            elif abs(c.contribution_log_odds) >= conflict_threshold_log_odds:
                conflicting.append(c)
            else:
                # Tiny opposing — fold into causal as neutral ballast
                causal.append(c)

        # Rank causal by |contribution|
        causal.sort(key=lambda c: -abs(c.contribution_log_odds))
        conflicting.sort(key=lambda c: -abs(c.contribution_log_odds))

        # Salience: |p_up - 0.5| × top-chain log-odds magnitude
        p_dev = abs(outcome.p_up - 0.5)
        top_abs = abs(causal[0].contribution_log_odds) if causal else 0.0
        salience = p_dev * (1.0 + top_abs)

        in_focus = outcome.ticker in focus_set
        floor = focus_salience_floor if in_focus else salience_floor
        if salience < floor:
            continue

        action = _action_from_p_up(outcome.p_up)

        # Force WATCH when conflicting evidence is comparable to causal:
        # total |conflicting contribution| ≥ 0.5 × total |causal contribution|.
        # This prevents the apex from confidently emitting BUY/SELL when the
        # network is internally divided.
        causal_mag = sum(abs(c.contribution_log_odds) for c in causal)
        conflict_mag = sum(abs(c.contribution_log_odds) for c in conflicting)
        if causal_mag > 0 and conflict_mag >= 0.5 * causal_mag:
            action = "WATCH"

        confidence = _confidence_from_chain(
            top_abs, len(causal), len(conflicting),
        )
        reasoning = _summarize_chain(
            outcome.ticker, outcome.p_up, causal, conflicting, action,
        )

        out.append(Observation(
            ticker=outcome.ticker,
            action=action,  # type: ignore[arg-type]
            p_up=outcome.p_up,
            salience=salience,
            horizon_min=outcome.horizon_min or horizon_min_default,
            causal_chain=causal[:top_chain_k],
            conflicting_signals=conflicting[:top_chain_k],
            triggering_events=trigger_ids,
            confidence=confidence,  # type: ignore[arg-type]
            reasoning_summary=reasoning,
            in_focus=in_focus,
        ))

    # Sort observations by salience desc, with focus tickers slightly bumped
    out.sort(key=lambda o: (
        -1 if o.in_focus else 0,
        -o.salience,
    ))
    return out
