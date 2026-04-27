"""Appetite function E — what makes a substrate item attractive to a herbivore.

v1: structured, hand-initialized, no gradients. The interface accepts a
substrate item and a herbivore state (with current tick, query embedding,
diet preferences, and meal-so-far for diversity penalties) and returns a
scalar score. v2 will swap this for a learnable function — the contract
stays the same.

The features computed here (age, relevance, diversity, reputation, diet
match) are not the system's only features. They are the v1 *defaults* — the
prior. A future learnable E may discover that producer-3's output is only
useful every 7 ticks, and that's fine; nothing here forecloses on that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .config import AppetiteWeights, DEFAULT_CONFIG
from .embeddings import cosine
from .types import ProducedSubstrate


@dataclass
class HerbivoreState:
    herbivore_id: str
    diet: list[str]
    current_tick: int
    query_embedding: list[float]                # retrieval-space query
    meal_so_far: list[ProducedSubstrate] = field(default_factory=list)
    producer_reputation: dict[str, float] = field(default_factory=dict)


def _max_similarity_to_meal(
    item: ProducedSubstrate, meal: Iterable[ProducedSubstrate]
) -> float:
    sims = [cosine(item.retrieval_embedding, m.retrieval_embedding) for m in meal]
    return max(sims) if sims else 0.0


def appetite(
    item: ProducedSubstrate,
    state: HerbivoreState,
    weights: AppetiteWeights | None = None,
) -> float:
    """Score in roughly [-inf, +inf]; > weights.min_appetite means edible."""
    w = weights or DEFAULT_CONFIG.appetite

    # Diet check (hard). The query_diet pre-filter usually catches this, but
    # we re-check here in case appetite is called outside the consumption path.
    diet_ok = any(t in state.diet for t in item.diet_tags)
    if not diet_ok:
        return float("-inf")

    age = max(0, state.current_tick - item.created_tick)
    relevance = cosine(item.retrieval_embedding, state.query_embedding)
    redundancy = _max_similarity_to_meal(item, state.meal_so_far)
    rep = state.producer_reputation.get(item.producer_id, 0.0)

    score = (
        w.age_weight * age
        + w.relevance_weight * relevance
        - w.diversity_weight * redundancy
        + w.reputation_weight * rep
        + w.diet_match_weight * 1.0
    )
    return float(score)
