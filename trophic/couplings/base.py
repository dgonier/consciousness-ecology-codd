"""Coupling C — how appetite scores resolve into actual consumption.

Different C's correspond to different coordination regimes (market,
democracy, ecology, hierarchy, community). All operate on the same inputs
(scored substrate per herbivore) and produce per-herbivore meal assignments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..appetite import HerbivoreState
from ..config import AppetiteWeights, DEFAULT_CONFIG
from ..types import ProducedSubstrate


@dataclass
class Bid:
    herbivore_id: str
    state: HerbivoreState
    candidates: list[ProducedSubstrate]    # diet-filtered, retrieval-K
    capacity: int                          # stomach size remaining


@dataclass
class MealAssignment:
    herbivore_id: str
    selected: list[ProducedSubstrate] = field(default_factory=list)
    rejected: list[ProducedSubstrate] = field(default_factory=list)


class Coupling(Protocol):
    name: str

    def select(
        self,
        bids: list[Bid],
        weights: AppetiteWeights = DEFAULT_CONFIG.appetite,
    ) -> list[MealAssignment]:
        """Resolve all simultaneous bids into meal assignments.

        Implementations must respect:
          - per-bid capacity (hard cap)
          - that the same item cannot be assigned to two herbivores
          - rejected = diet-filtered candidates that weren't selected (so
            decomposers can attribute waste)
        """
        ...
