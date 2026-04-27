"""Per-tick instrumentation. Just collected dicts for v1, no plotting.

The point of v1 is making the dynamics legible enough that we can see
clearing/hunger/waste happen. This is a flat list of dicts that can be
inspected from a notebook or pretty-printed by demo.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TickMetrics:
    tick: int = 0
    n_inputs: int = 0
    n_substrate_added: int = 0
    n_eaten: int = 0
    n_rotted: int = 0
    n_syntheses: int = 0
    n_judgments: int = 0
    avg_judgment: float = 0.0
    avg_confidence: float = 0.0
    population: dict = field(default_factory=dict)


@dataclass
class MetricsCollector:
    rows: list[TickMetrics] = field(default_factory=list)

    def record(self, m: TickMetrics) -> None:
        self.rows.append(m)

    def latest(self) -> TickMetrics | None:
        return self.rows[-1] if self.rows else None
