"""Herbivore fitness — different shape than apex voter fitness.

Apex voters cast directional votes; we compute marginal contribution
via leave-one-out drop tests on the panel decision. Herbivores emit
synthesis paragraphs, not directions. Their contribution is
"did the apex panel get more accurate when this herb's synthesis was
present vs absent?"

Per-scenario leave-one-herb-out is too expensive (each herb-drop = 1
extra full apex pass). We use a cheaper sampled-dropout: every Nth
scenario, randomly drop ONE herb's synthesis from the packet; over a
rolling window we compare apex correctness with-vs-without that herb.

Fields:
  n_seen           : scenarios where this herb was either present or dropped
  n_present        : scenarios where this herb's synthesis was in the packet
  n_present_correct: of those, n where apex was correct
  n_dropped        : scenarios where this herb was dropped (sampling)
  n_dropped_correct: of those, n where apex was still correct without it

  contribution     = (correct_rate_when_present - correct_rate_when_dropped)
                     positive → herb adds value; negative → herb hurts.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class HerbFitness:
    herb_id: str
    species_id: str = ""
    n_seen: int = 0
    n_present: int = 0
    n_present_correct: int = 0
    n_dropped: int = 0
    n_dropped_correct: int = 0

    @property
    def correct_rate_when_present(self) -> float:
        if self.n_present == 0:
            return 0.0
        return self.n_present_correct / self.n_present

    @property
    def correct_rate_when_dropped(self) -> float:
        if self.n_dropped == 0:
            return 0.0
        return self.n_dropped_correct / self.n_dropped

    @property
    def contribution(self) -> float:
        """Difference in apex correctness rate between present/dropped.
        Positive = herb adds value; negative = herb hurts.
        Returns 0.0 when either side has insufficient data."""
        if self.n_present == 0 or self.n_dropped == 0:
            return 0.0
        return self.correct_rate_when_present - self.correct_rate_when_dropped


@dataclass
class HerbFitnessTracker:
    """Rolling-window per-herb fitness tracker."""
    window: int = 200
    _by_herb: dict[str, HerbFitness] = field(default_factory=dict)
    _recent: dict[str, deque] = field(default_factory=dict)
    # _recent[herb_id]: deque of (was_present, apex_correct)

    def record(
        self,
        *,
        herb_id: str,
        species_id: str = "",
        was_present: bool,
        apex_correct: bool,
    ) -> None:
        af = self._by_herb.setdefault(
            herb_id, HerbFitness(herb_id=herb_id, species_id=species_id),
        )
        if not af.species_id and species_id:
            af.species_id = species_id
        af.n_seen += 1
        if was_present:
            af.n_present += 1
            if apex_correct:
                af.n_present_correct += 1
        else:
            af.n_dropped += 1
            if apex_correct:
                af.n_dropped_correct += 1
        rq = self._recent.setdefault(herb_id, deque(maxlen=self.window))
        rq.append((was_present, apex_correct))

    def summary(self) -> dict[str, HerbFitness]:
        return dict(self._by_herb)

    def render_summary(self) -> str:
        lines = ["=== HERB-VOTER FITNESS ==="]
        for hid, f in self._by_herb.items():
            lines.append(
                f"  {hid:60s} seen={f.n_seen} "
                f"present={f.n_present} (correct={f.n_present_correct}) "
                f"dropped={f.n_dropped} (correct={f.n_dropped_correct}) "
                f"contribution={f.contribution:+.3f}"
            )
        return "\n".join(lines)
