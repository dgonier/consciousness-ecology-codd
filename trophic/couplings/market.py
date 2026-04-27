"""MarketCoupling — simultaneous bid clearing.

All hungry herbivores submit appetite-ranked bids simultaneously. The
clearing engine resolves contention item-by-item: highest appetite wins,
ties broken randomly. After each item is awarded, losers re-rank their
remaining candidates (diversity vs new meal-so-far updates).

Closer to a real market than EcologyCoupling because no herbivore gets
strict first-mover advantage — both compete for the same item with their
revealed preferences.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..appetite import appetite
from ..config import AppetiteWeights, DEFAULT_CONFIG
from .base import Bid, Coupling, MealAssignment


@dataclass
class MarketCoupling(Coupling):
    name: str = "market"
    seed: int | None = None

    def select(
        self,
        bids: list[Bid],
        weights: AppetiteWeights = DEFAULT_CONFIG.appetite,
    ) -> list[MealAssignment]:
        rng = random.Random(self.seed)
        assignments: dict[str, MealAssignment] = {
            b.herbivore_id: MealAssignment(herbivore_id=b.herbivore_id) for b in bids
        }
        remaining_capacity: dict[str, int] = {b.herbivore_id: b.capacity for b in bids}
        # Per-herbivore live candidate pool (mutates as items are claimed elsewhere).
        candidates: dict[str, list] = {b.herbivore_id: list(b.candidates) for b in bids}
        states = {b.herbivore_id: b.state for b in bids}

        # Pool of items still uncllaimed across the whole market.
        all_items: dict[str, object] = {}
        for b in bids:
            for c in b.candidates:
                all_items[c.id] = c

        while True:
            # Each herbivore proposes its single best item right now.
            proposals: list[tuple[str, str, float]] = []  # (item_id, herb_id, score)
            for hid, b in zip(remaining_capacity, bids):
                if remaining_capacity[hid] <= 0:
                    continue
                state = states[hid]
                state.meal_so_far = list(assignments[hid].selected)
                cand = [c for c in candidates[hid] if c.id in all_items]
                if not cand:
                    continue
                scored = [(appetite(c, state, weights), c) for c in cand]
                scored.sort(key=lambda x: x[0], reverse=True)
                top_score, top = scored[0]
                if top_score < weights.min_appetite:
                    continue
                proposals.append((top.id, hid, top_score))

            if not proposals:
                break

            # Group by item, resolve highest bidder per contested item.
            by_item: dict[str, list[tuple[str, float]]] = {}
            for iid, hid, sc in proposals:
                by_item.setdefault(iid, []).append((hid, sc))

            awarded_this_round: set[str] = set()
            for iid, bidders in by_item.items():
                if iid in awarded_this_round:
                    continue
                bidders.sort(key=lambda x: (x[1], rng.random()), reverse=True)
                winner_hid, _ = bidders[0]
                item = all_items.pop(iid)
                assignments[winner_hid].selected.append(item)
                remaining_capacity[winner_hid] -= 1
                awarded_this_round.add(iid)
                # Losers will simply re-propose next round from their remaining pool.

        # Rejected = diet-filtered originals not in selected.
        for b in bids:
            sel_ids = {s.id for s in assignments[b.herbivore_id].selected}
            assignments[b.herbivore_id].rejected = [
                c for c in b.candidates if c.id not in sel_ids
            ]
        return list(assignments.values())
