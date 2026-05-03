"""Hexis-style KG writer.

For each scenario evaluated, write a structured record to a JSONL
knowledge graph file capturing:
  - scenario signature (ticker, date, simple feature digest)
  - what each voter said (direction, confidence, perplexity)
  - ensemble decision and ground truth
  - per-voter outcome (correct? abstain? misled?)

Future runs query the KG for "what worked when the input looked like X":
e.g., "on JPM scenarios with high realized vol, which voters tend to be
right?" That informs evolutionary policy.

JSONL is the simplest queryable format. Upgrade path: load into a graph
DB or vector store keyed by signature embeddings.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class KGRecord:
    """One observation of the ecosystem at one scenario."""
    timestamp: float
    scenario_name: str
    ticker: str
    target_direction: str  # ground truth
    ensemble_direction: str | None
    ensemble_method: str
    ensemble_confidence: float
    voters: list[dict] = field(default_factory=list)
    # Each voter dict: {voter_id, direction, confidence, perplexity,
    # correct: bool, abstained: bool}
    signature: dict = field(default_factory=dict)
    # Optional simple features the user can query against (price drift,
    # tweet count, etc.) — populated by the orchestrator.

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class KGWriter:
    path: Path = Path("external/decomposer_kg/kg.jsonl")
    _fh: object = None

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def open(self):
        if self._fh is None:
            self._fh = self.path.open("a")
        return self

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def write(self, record: KGRecord) -> None:
        if self._fh is None:
            self.open()
        self._fh.write(record.to_json() + "\n")  # type: ignore[union-attr]
        self._fh.flush()  # type: ignore[union-attr]

    def write_observation(
        self,
        *,
        scenario_name: str,
        ticker: str,
        target_direction: str,
        ensemble_decision,  # EnsembleDecision
        signature: dict | None = None,
    ) -> KGRecord:
        """Convenience: package an EnsembleDecision into a KG record."""
        voter_records = []
        for v in ensemble_decision.voters:
            correct = (v.direction == target_direction) if v.direction else False
            voter_records.append({
                "voter_id": v.voter_id,
                "direction": v.direction,
                "confidence": v.confidence,
                "perplexity": v.perplexity,
                "correct": correct,
                "abstained": v.direction is None,
            })
        rec = KGRecord(
            timestamp=time.time(),
            scenario_name=scenario_name,
            ticker=ticker,
            target_direction=target_direction,
            ensemble_direction=ensemble_decision.direction,
            ensemble_method=ensemble_decision.method,
            ensemble_confidence=ensemble_decision.confidence,
            voters=voter_records,
            signature=signature or {},
        )
        self.write(rec)
        return rec

    @classmethod
    def read_all(cls, path: Path | str = "external/decomposer_kg/kg.jsonl") -> list[KGRecord]:
        """Read the entire KG into memory. JSON-line format."""
        path = Path(path) if isinstance(path, str) else path
        if not path.exists():
            return []
        out: list[KGRecord] = []
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            d = json.loads(ln)
            out.append(KGRecord(**d))
        return out
