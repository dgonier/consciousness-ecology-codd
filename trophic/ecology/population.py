"""Agent registry + energy bookkeeping.

Producers and herbivores live and die here. Energy ticks down each step
(c_exist) and is replenished by being consumed (producers) or eating
(herbivores), plus bonuses from decomposer attribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..agents.forecaster_herbivore import ForecasterHerbivore
from ..agents.herbivore import Herbivore
from ..agents.predator import Predator
from ..agents.producer import Producer
from ..agents.quant_producer import QuantitativeProducer
from ..config import PopulationConfig, DEFAULT_CONFIG
from ..types import AgentState


ProducerLike = Producer | QuantitativeProducer
HerbivoreLike = Herbivore | ForecasterHerbivore


@dataclass
class Population:
    cfg: PopulationConfig = field(default_factory=lambda: DEFAULT_CONFIG.population)
    # Both Qwen-text and Quantitative producers live here; same Agent shape.
    producers: dict[str, ProducerLike] = field(default_factory=dict)
    # Both Qwen-text and Forecaster (Chronos) herbivores live here.
    herbivores: dict[str, HerbivoreLike] = field(default_factory=dict)
    predators: dict[str, Predator] = field(default_factory=dict)
    # Per-agent reputation tracker (separate from energy so culling and
    # weighting are not coupled).
    reputation: dict[str, float] = field(default_factory=dict)

    def add_producer(self, p: ProducerLike) -> None:
        self.producers[p.id] = p
        p.state.energy = self.cfg.energy_baseline
        self.reputation[p.id] = 0.0

    def add_herbivore(self, h: HerbivoreLike) -> None:
        self.herbivores[h.id] = h
        h.state.energy = self.cfg.energy_baseline
        self.reputation[h.id] = 0.0

    def add_predator(self, p: Predator) -> None:
        self.predators[p.id] = p
        p.state.energy = self.cfg.energy_baseline
        self.reputation[p.id] = 0.0

    def alive_producers(self) -> list[ProducerLike]:
        return [p for p in self.producers.values() if p.state.alive]

    def alive_herbivores(self) -> list[HerbivoreLike]:
        return [h for h in self.herbivores.values() if h.state.alive]

    def alive_predators(self) -> list[Predator]:
        return [p for p in self.predators.values() if p.state.alive]

    # ---------- energy events ----------

    def _agent(self, agent_id: str):
        return (
            self.producers.get(agent_id)
            or self.herbivores.get(agent_id)
            or self.predators.get(agent_id)
        )

    def tick_existence_costs(self, tick: int) -> None:
        for p in self.alive_producers():
            p.state.energy -= self.cfg.cost_exist_per_tick
        for h in self.alive_herbivores():
            h.state.energy -= self.cfg.cost_exist_per_tick
        for pr in self.alive_predators():
            pr.state.energy -= self.cfg.cost_exist_per_tick

    def credit_production(self, producer_id: str, n_items: int) -> None:
        a = self._agent(producer_id)
        if a is not None:
            a.state.energy -= n_items * self.cfg.cost_produce_per_item

    def credit_eaten(self, agent_id: str, n_items: int) -> None:
        a = self._agent(agent_id)
        if a is not None:
            a.state.energy += n_items * self.cfg.reward_eaten_per_item

    def credit_intake(self, agent_id: str, fill_ratio: float, n_items: int) -> None:
        a = self._agent(agent_id)
        if a is not None:
            a.state.energy += n_items * self.cfg.reward_intake_per_item * fill_ratio

    def credit_judgment(self, agent_id: str, score: float) -> None:
        a = self._agent(agent_id)
        if a is not None:
            a.state.energy += score * self.cfg.reward_judgment_scale

    def apply_decomposer(
        self,
        producer_energy: dict[str, float],
        producer_reputation: dict[str, float],
        herbivore_energy: dict[str, float],
    ) -> None:
        for pid, dE in producer_energy.items():
            a = self._agent(pid)
            if a is not None:
                a.state.energy += dE
        for pid, dR in producer_reputation.items():
            self.reputation[pid] = self.reputation.get(pid, 0.0) + dR
            a = self._agent(pid)
            if a is not None:
                a.state.reputation = self.reputation[pid]
        for hid, dE in herbivore_energy.items():
            a = self._agent(hid)
            if a is not None:
                a.state.energy += dE

    # ---------- census ----------

    def census(self, tick: int) -> dict:
        for p in self.alive_producers():
            if p.state.energy < self.cfg.energy_cull_threshold:
                p.state.alive = False
                p.state.died_tick = tick
        for h in self.alive_herbivores():
            if h.state.energy < self.cfg.energy_cull_threshold:
                h.state.alive = False
                h.state.died_tick = tick
        for pr in self.alive_predators():
            if pr.state.energy < self.cfg.energy_cull_threshold:
                pr.state.alive = False
                pr.state.died_tick = tick
        return self.snapshot()

    def snapshot(self) -> dict:
        return {
            "producers": {
                pid: {
                    "kind": p.kind,
                    "alive": p.state.alive,
                    "energy": round(p.state.energy, 4),
                    "reputation": round(p.state.reputation, 4),
                }
                for pid, p in self.producers.items()
            },
            "herbivores": {
                hid: {
                    "kind": h.kind,
                    "alive": h.state.alive,
                    "energy": round(h.state.energy, 4),
                }
                for hid, h in self.herbivores.items()
            },
            "predators": {
                pid: {
                    "kind": p.kind,
                    "alive": p.state.alive,
                    "energy": round(p.state.energy, 4),
                }
                for pid, p in self.predators.items()
            },
        }
