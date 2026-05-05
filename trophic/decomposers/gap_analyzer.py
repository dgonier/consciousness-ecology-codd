"""GapAnalyzer — propose new species that fill gaps in the panel.

Reads recent Observations (where the panel was wrong) and the current
species roster. Calls Opus 4.7 to reason: "what kind of species would
have helped on these losses, given that we already have these voters?"

Produces one new Species record per call, parent-less (generation 0
for the new lineage), ready to be written to the registry.

This is the meta-LLM "ecology designer" role. Hardest single
reasoning task in the project — fully open-ended.
"""
from __future__ import annotations

import time
import uuid
from collections import Counter
from typing import Sequence

from .opus_judge import call_opus_for_json
from .observation import Observation
from .species import Species


GAP_SYSTEM = (
    "You are the ecology designer for an evolutionary multi-LLM voting"
    " panel. The panel predicts next-day stock direction (UP/DOWN). It"
    " ensembles K voters across model families (OpenAI, Anthropic"
    " Bedrock, Gemini, local Qwen)."
    "\n\n"
    "You receive:"
    "\n  - the CURRENT panel: one species record per voter"
    "\n  - a sample of RECENT FAILURES: scenarios where the panel got"
    "    the answer wrong"
    "\n  - aggregate panel statistics"
    "\n\n"
    "Your job: identify ONE specific GAP in the panel's analytical"
    " coverage and design ONE new species to fill it. Examples of gaps:"
    "\n  - the panel never reads bond-market context for financial"
    "    sector tickers"
    "\n  - all voters anchor to recent OHLCV without ever doing a"
    "    fundamental valuation lens"
    "\n  - on tweet-heavy scenarios, no voter is filtering for"
    "    sentiment rotation patterns"
    "\n  - the panel is bearish-biased; no voter takes a contrarian"
    "    momentum-reversal stance"
    "\n\n"
    "The new species can use any model_id from the existing roster"
    " (run on the same substrate but with a sharply different prompt"
    " template / diet / role) OR a different model the user has"
    " confirmed access to. Default to reusing the strongest existing"
    " model_id."
    "\n\n"
    "Return ONLY a JSON object with keys: species_id (auto if omitted),"
    " model_id, role, prompt_template, diet_tags, hyperparams,"
    " kg_context_hint, note, gap_addressed (string explaining what gap"
    " this species fills). role must be 'apex_voter'. NO prose, NO code"
    " fences, just JSON."
)


def _failure_brief(obs: Observation) -> dict:
    return {
        "scenario": obs.scenario_name,
        "ticker": obs.ticker,
        "target": obs.target_direction,
        "ensemble": obs.ensemble.get("direction"),
        "voters": [
            {
                "id": v.get("voter_id"),
                "direction": v.get("direction"),
                "reasoning": (v.get("reasoning") or "")[:200],
            }
            for v in obs.voter_responses
        ],
    }


def _panel_summary(observations: Sequence[Observation]) -> dict:
    n = len(observations)
    n_correct = sum(
        1 for o in observations
        if o.target_direction
        and o.ensemble.get("direction") == o.target_direction
    )
    target_counts = Counter(
        o.target_direction for o in observations if o.target_direction
    )
    pred_counts = Counter(
        o.ensemble.get("direction") for o in observations if o.ensemble.get("direction")
    )
    ticker_counts = Counter(o.ticker for o in observations)
    return {
        "n_observations": n,
        "ensemble_correctness": round(n_correct / max(n, 1), 3),
        "target_distribution": dict(target_counts),
        "prediction_distribution": dict(pred_counts),
        "tickers_seen": dict(ticker_counts),
    }


def propose_gap_species(
    panel: Sequence[Species],
    observations: Sequence[Observation],
    *,
    max_failures: int = 8,
) -> Species | None:
    """Identify ONE gap and propose one species to fill it. Returns
    None on Opus failure."""
    failures = [
        o for o in observations
        if o.target_direction and o.ensemble.get("direction")
        and o.target_direction != o.ensemble.get("direction")
    ]
    if not failures:
        return None
    # Sample up to max_failures recent ones
    recent_failures = failures[-max_failures:]
    summary = _panel_summary(observations)
    panel_brief = [
        {
            "species_id": s.species_id,
            "model_id": s.model_id,
            "role": s.role,
            "prompt_template": s.prompt_template or "(default SYSTEM)",
            "diet_tags": s.diet_tags,
            "generation": s.generation,
        }
        for s in panel
    ]
    user = (
        f"CURRENT PANEL ({len(panel_brief)} voters):\n"
        f"{panel_brief}\n\n"
        f"PANEL STATS:\n{summary}\n\n"
        f"RECENT FAILURES ({len(recent_failures)} of {len(failures)} total):\n"
        f"{[_failure_brief(f) for f in recent_failures]}\n\n"
        "Identify ONE gap and design ONE new species JSON now."
    )
    obj = call_opus_for_json(GAP_SYSTEM, user, max_tokens=2500, temperature=0.6)
    if obj is None:
        return None
    if "model_id" not in obj:
        return None
    # Coerce into Species, drop unknown keys (gap_addressed goes into note)
    if "gap_addressed" in obj:
        gap = obj.pop("gap_addressed")
        obj["note"] = (obj.get("note") or "") + f" | gap: {gap}"
    obj.setdefault("role", "apex_voter")
    obj.setdefault("prompt_template", "")
    obj.setdefault("diet_tags", [])
    obj.setdefault("hyperparams", {})
    obj.setdefault("parent_ids", [])
    obj.setdefault("generation", 0)
    obj.setdefault("kg_context_hint", [])
    obj.setdefault("alive", True)
    obj.setdefault("species_id", f"sp.gap.{uuid.uuid4().hex[:8]}")
    obj.setdefault("note", "gap-fill candidate")
    obj.setdefault("created_at", time.time())
    allowed = {
        "species_id", "model_id", "role", "prompt_template", "diet_tags",
        "hyperparams", "alive", "parent_ids", "generation",
        "kg_context_hint", "created_at", "note",
    }
    cleaned = {k: v for k, v in obj.items() if k in allowed}
    try:
        return Species(**cleaned)
    except TypeError as e:
        print(f"[gap_analyzer] failed to coerce Opus response: {e}")
        return None
