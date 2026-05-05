"""Reproducer — recombine two parent species into a child species.

Speculative by definition: the child has to earn its spot via the same
fitness loop. The reproducer just produces candidates.

Calls Opus 4.7 (via Bedrock) with both parents' configs + their recent
fitness traces, asks for ONE child species JSON that combines the best
components. Opus is a hard-reasoning model — small task, but it's the
hardest single task in the system because it's an open-ended design
problem ("what's the best of these two and what gets discarded").

Output is a Species record ready for SpeciesRegistry.write().
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from typing import Sequence

from .opus_judge import call_opus_for_json
from .species import Species
from .fitness import AgentFitness


REPRODUCE_SYSTEM = (
    "You are the reproducer in an evolutionary multi-agent ecosystem."
    " You receive two PARENT species configs that have proven useful in"
    " a multi-LLM voting panel for next-day stock direction prediction."
    "\n\n"
    "Your job: produce ONE CHILD species config that combines the best"
    " components of both parents. The child should:"
    "\n  - inherit the model_id of the parent with stronger marginal"
    "    contribution (the one that more often flipped the panel correctly)"
    "\n  - synthesize a NEW prompt_template that draws specific framings,"
    "    instructions, or analytical lenses from each parent"
    "\n  - merge diet_tags from both parents (deduplicated)"
    "\n  - average numeric hyperparams where both have them; otherwise"
    "    take the value from whichever parent had it"
    "\n  - set parent_ids = [parent_a.species_id, parent_b.species_id]"
    "\n  - set generation = max(parents' generation) + 1"
    "\n\n"
    "DO NOT just concatenate the parents' prompts. Synthesize. The"
    " child should be different from either parent, with a focused angle"
    " that addresses the panel's weaknesses identified in the fitness data."
    "\n\n"
    "Return ONLY a JSON object with these keys: species_id (auto if"
    " omitted), model_id, role, prompt_template, diet_tags,"
    " hyperparams, parent_ids, generation, kg_context_hint, note."
    " role must be 'apex_voter'. NO prose, NO code fences, just JSON."
)


def _parent_brief(species: Species, fitness: AgentFitness | None) -> dict:
    """Compact dict describing one parent for the reproducer prompt."""
    out = {
        "species_id": species.species_id,
        "model_id": species.model_id,
        "role": species.role,
        "prompt_template": species.prompt_template or "(default SYSTEM)",
        "diet_tags": species.diet_tags,
        "hyperparams": species.hyperparams,
        "generation": species.generation,
        "note": species.note,
    }
    if fitness is not None:
        out["fitness"] = {
            "n_seen": fitness.n_seen,
            "solo_accuracy": round(fitness.solo_accuracy, 3),
            "marginal_contribution": round(fitness.marginal_contribution, 3),
            "freerider_score": round(fitness.freerider_score, 3),
        }
    return out


def reproduce(
    parent_a: Species,
    parent_b: Species,
    fitness_a: AgentFitness | None = None,
    fitness_b: AgentFitness | None = None,
    panel_summary: str = "",
) -> Species | None:
    """Mate two parents → one child species. None on Opus failure."""
    user = (
        "PARENT A:\n"
        f"{_parent_brief(parent_a, fitness_a)}\n\n"
        "PARENT B:\n"
        f"{_parent_brief(parent_b, fitness_b)}\n\n"
        f"PANEL CONTEXT:\n{panel_summary or '(no panel context provided)'}\n\n"
        "Produce ONE child species JSON now."
    )
    obj = call_opus_for_json(REPRODUCE_SYSTEM, user, max_tokens=2048, temperature=0.5)
    if obj is None:
        return None
    # Coerce into a Species. Required: model_id. Auto-fill species_id +
    # parent_ids + generation if Opus didn't include them.
    if "model_id" not in obj:
        return None
    obj.setdefault("role", "apex_voter")
    obj.setdefault("prompt_template", "")
    obj.setdefault("diet_tags", [])
    obj.setdefault("hyperparams", {})
    obj.setdefault("parent_ids", [parent_a.species_id, parent_b.species_id])
    obj.setdefault(
        "generation",
        max(parent_a.generation, parent_b.generation) + 1,
    )
    obj.setdefault("kg_context_hint", [])
    obj.setdefault("alive", True)
    obj.setdefault(
        "species_id",
        f"sp.child.{uuid.uuid4().hex[:8]}",
    )
    obj.setdefault(
        "note",
        f"reproduced from {parent_a.species_id} + {parent_b.species_id}",
    )
    obj.setdefault("created_at", time.time())
    # Drop any unknown keys that Opus might have added.
    allowed = {
        "species_id", "model_id", "role", "prompt_template", "diet_tags",
        "hyperparams", "alive", "parent_ids", "generation",
        "kg_context_hint", "created_at", "note",
    }
    cleaned = {k: v for k, v in obj.items() if k in allowed}
    try:
        return Species(**cleaned)
    except TypeError as e:
        print(f"[reproducer] failed to coerce Opus response into Species: {e}")
        return None


def select_top_pair(
    species: Sequence[Species],
    fitness: dict[str, AgentFitness],
    by: str = "combined",
) -> tuple[Species, Species] | None:
    """Pick the two top species by fitness signal for mating.

    `by` ∈ {'combined', 'marginal'}.
      combined = solo_accuracy + marginal_contribution
      marginal = marginal_contribution alone (rewards moving the panel)

    Returns None if fewer than 2 species have fitness data.
    """
    scored: list[tuple[float, Species]] = []
    for sp in species:
        # Look up by full voter_id (model_namespace::species_id) since
        # that's how the eval orchestrator records fitness today.
        fit = None
        for fid, af in fitness.items():
            if fid.endswith(f"::{sp.species_id}"):
                fit = af
                break
        if fit is None or fit.n_seen == 0:
            continue
        if by == "marginal":
            score = fit.marginal_contribution
        else:
            score = fit.solo_accuracy + fit.marginal_contribution
        scored.append((score, sp))
    if len(scored) < 2:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1], scored[1][1]
