"""Decomposer — credit-assignment engine.

Given (synthesis, judgment), pushes credit back through the consumption
chain to the producers whose substrate was eaten and to the herbivore that
synthesized. Emits DecomposerRecord rows describing the attribution.

v1: simple proportional credit. Each consumed substrate item gets an equal
share of the synthesis's judgment score. Non-eaten (rejected) items the
herbivore *could* have eaten get a small "wasted" record. Eventually this
becomes the gradient signal for E.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Iterable

from ..types import ConsumedSynthesis, DecomposerRecord, PredatorJudgment


# Threshold above which a judgment is considered "useful"; below "misleading".
USEFUL_THRESHOLD = 0.6
MISLEADING_THRESHOLD = 0.3


@dataclass
class DecomposerOutput:
    records: list[DecomposerRecord] = field(default_factory=list)
    # Accumulated deltas the population layer will apply.
    producer_energy: dict[str, float] = field(default_factory=dict)
    producer_reputation: dict[str, float] = field(default_factory=dict)
    herbivore_energy: dict[str, float] = field(default_factory=dict)


@dataclass
class Decomposer:
    id: str = "decomposer.v1"

    def attribute(
        self,
        syntheses: list[ConsumedSynthesis],
        judgments: dict[str, PredatorJudgment],
        substrate_to_producer: dict[str, str],
        tick: int,
        useful_bonus: float = 0.15,
        wasted_penalty: float = 0.02,
    ) -> DecomposerOutput:
        out = DecomposerOutput()
        for s in syntheses:
            j = judgments.get(s.id)
            if j is None:
                continue
            score = j.score
            n = len(s.consumed_substrate_ids)
            per_item = (score - 0.5) * 2.0  # map [0,1] → [-1,+1]

            # ---- credit to consumed substrate (and through to producers) ----
            for sid in s.consumed_substrate_ids:
                pid = substrate_to_producer.get(sid)
                if pid is None:
                    continue
                # Producer reputation moves toward judgment.
                out.producer_reputation[pid] = (
                    out.producer_reputation.get(pid, 0.0) + per_item / max(n, 1)
                )
                if score >= USEFUL_THRESHOLD:
                    out.producer_energy[pid] = (
                        out.producer_energy.get(pid, 0.0) + useful_bonus / max(n, 1)
                    )
                    out.records.append(self._record(
                        "useful", sid, "substrate", tick,
                        f"Cited in synthesis {s.id[:8]} judged {score:.2f}",
                    ))
                elif score <= MISLEADING_THRESHOLD:
                    out.records.append(self._record(
                        "misleading", sid, "substrate", tick,
                        f"Cited in synthesis {s.id[:8]} judged {score:.2f}",
                    ))

            # ---- credit to the herbivore ----
            out.herbivore_energy[s.herbivore_id] = (
                out.herbivore_energy.get(s.herbivore_id, 0.0) + per_item * 0.1
            )

            # ---- waste records for rejected-but-edible substrate ----
            for rid in s.rejected_substrate_ids:
                out.records.append(self._record(
                    "wasted", rid, "substrate", tick,
                    f"In diet of {s.herbivore_id[:18]} but not selected",
                ))
                pid = substrate_to_producer.get(rid)
                if pid is not None:
                    out.producer_energy[pid] = (
                        out.producer_energy.get(pid, 0.0) - wasted_penalty
                    )
        return out

    @staticmethod
    def _record(kind: str, target_id: str, target_kind: str, tick: int, rationale: str) -> DecomposerRecord:
        return DecomposerRecord(
            id=str(uuid.uuid4()),
            record_type=kind,  # type: ignore[arg-type]
            target_id=target_id,
            target_kind=target_kind,  # type: ignore[arg-type]
            rationale=rationale,
            weight=1.0,
            created_tick=tick,
        )
