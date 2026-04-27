"""EcologyCoupling — greedy diversity-weighted per-herbivore selection.

Herbivores eat in randomized order each tick. Each one greedy-picks until
capacity is hit, scoring each candidate with the live appetite function
(which already includes the diversity-vs-meal-so-far penalty). Items
selected by the first herbivore are not visible to the second — first eater
wins.

This is the brief's original consumption ranker, lifted into the C
interface.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..appetite import appetite
from ..config import AppetiteWeights, DEFAULT_CONFIG
from .base import Bid, Coupling, MealAssignment


@dataclass
class EcologyCoupling(Coupling):
    name: str = "ecology"
    seed: int | None = None

    def select(
        self,
        bids: list[Bid],
        weights: AppetiteWeights = DEFAULT_CONFIG.appetite,
    ) -> list[MealAssignment]:
        rng = random.Random(self.seed)
        order = list(range(len(bids)))
        rng.shuffle(order)

        claimed: set[str] = set()
        assignments: dict[str, MealAssignment] = {
            b.herbivore_id: MealAssignment(herbivore_id=b.herbivore_id) for b in bids
        }

        for i in order:
            bid = bids[i]
            assn = assignments[bid.herbivore_id]
            available = [c for c in bid.candidates if c.id not in claimed]
            for _ in range(bid.capacity):
                if not available:
                    break
                # Re-score live so diversity-vs-meal-so-far updates as we eat.
                bid.state.meal_so_far = list(assn.selected)
                scored = [(appetite(c, bid.state, weights), c) for c in available]
                scored.sort(key=lambda x: x[0], reverse=True)
                top_score, top = scored[0]
                if top_score < weights.min_appetite:
                    break
                assn.selected.append(top)
                claimed.add(top.id)
                available = [c for c in available if c.id != top.id]

            # Anything we didn't pick that was edible counts as rejected.
            assn.rejected = [
                c for c in bid.candidates
                if c.id not in claimed and c.id not in {s.id for s in assn.selected}
            ]
        return list(assignments.values())
