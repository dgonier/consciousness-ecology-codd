"""Unified per-scenario observation: every signal between every node,
plus the apex voters' reasoning and the ensemble outcome.

The decomposer reads this artifact (not the apex slice alone) so its
KG records and evolutionary signals are grounded in the *full* trophic
chain, not just the final answer.

Schema:

  Observation
    ├─ scenario_name, ticker, target_direction, timestamp
    ├─ inter_tier_signals: list of NodeSignal
    │     each NodeSignal: {tier, node_id, parents, agent_kind,
    │                        norm, finite, logit_lens, raw_text,
    │                        diet_tags}
    ├─ voter_responses: list of VoterResponse-like dicts with reasoning,
    │     citations, provider_meta
    ├─ ensemble: {direction, method, confidence, weight_up, weight_down}
    └─ decomposer_judgments: list of per-voter
          {voter_id, correct, decisive, marginal_flip, marginal_correct}

JSONL serializer writes one line per Observation, fully reviewable.

The inter_tier_signals capture is optional — the orchestrator can pass
None if it doesn't have an inspector hooked up — but when present, the
decomposer can reason about cases like "voter X was wrong AND the
producer.disclosure broadcast was input-uniformative AND herb.technical
output was high-perplexity" — i.e., chain-of-blame.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field


@dataclass
class NodeSignal:
    """One node's outbound signal at one tier."""
    tier: int               # 0 scenario, 1 producer, 2 trough, 3 herb, 4 apex-pre, 5 apex-out
    node_id: str            # e.g. "producer.tickdelta", "herb.technical"
    parents: list[str] = field(default_factory=list)
    agent_kind: str = ""
    norm: float | None = None
    finite: bool = True
    logit_lens: list[dict] = field(default_factory=list)  # [{token, prob}]
    raw_text: str = ""
    diet_tags: list[str] = field(default_factory=list)


@dataclass
class DecomposerJudgment:
    voter_id: str
    correct: bool | None
    decisive: bool
    marginal_flip: bool
    marginal_correct: bool


@dataclass
class Observation:
    """Everything the decomposer needs to know about one scenario."""
    timestamp: float
    scenario_name: str
    ticker: str
    target_direction: str | None
    inter_tier_signals: list[NodeSignal] = field(default_factory=list)
    voter_responses: list[dict] = field(default_factory=list)  # serialized VoterResponse
    ensemble: dict = field(default_factory=dict)
    decomposer_judgments: list[DecomposerJudgment] = field(default_factory=list)
    signature: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_voter_responses(
        cls,
        scenario_name: str,
        ticker: str,
        target_direction: str | None,
        voters,           # list[VoterResponse]
        ensemble,         # EnsembleDecision
        inter_tier: list[NodeSignal] | None = None,
        signature: dict | None = None,
    ) -> "Observation":
        return cls(
            timestamp=time.time(),
            scenario_name=scenario_name,
            ticker=ticker,
            target_direction=target_direction,
            inter_tier_signals=inter_tier or [],
            voter_responses=[
                {
                    "voter_id": v.voter_id,
                    "direction": v.direction,
                    "confidence": v.confidence,
                    "perplexity": v.perplexity,
                    "reasoning": getattr(v, "reasoning", ""),
                    "evidence_citations": list(getattr(v, "evidence_citations", []) or []),
                    "provider_meta": dict(getattr(v, "provider_meta", {}) or {}),
                    "raw_text": v.raw_text,
                }
                for v in voters
            ],
            ensemble={
                "direction": ensemble.direction,
                "method": ensemble.method,
                "confidence": ensemble.confidence,
                "weight_up": ensemble.weight_up,
                "weight_down": ensemble.weight_down,
                "n_voters": ensemble.n_voters,
                "n_decisive": ensemble.n_decisive,
            },
            signature=signature or {},
        )


class ObservationWriter:
    """Append observations to a JSONL file. Replaces the apex-slice-only
    KGWriter as the canonical record format when full inter-tier signals
    are available."""

    def __init__(self, path):
        from pathlib import Path
        self.path = Path(path) if not hasattr(path, "open") else path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def open(self):
        if self._fh is None:
            self._fh = self.path.open("a")
        return self

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def write(self, obs: Observation) -> None:
        if self._fh is None:
            self.open()
        self._fh.write(obs.to_json() + "\n")
        self._fh.flush()

    @classmethod
    def read_all(cls, path) -> list[Observation]:
        from pathlib import Path
        p = Path(path) if not hasattr(path, "open") else path
        if not p.exists():
            return []
        out: list[Observation] = []
        for ln in p.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            d = json.loads(ln)
            d["inter_tier_signals"] = [NodeSignal(**s) for s in d.get("inter_tier_signals", [])]
            d["decomposer_judgments"] = [DecomposerJudgment(**j) for j in d.get("decomposer_judgments", [])]
            out.append(Observation(**d))
        return out
