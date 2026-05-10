"""Belief network storage.

Adapter pattern:
  - BeliefStore: ABC defining the interface.
  - JSONLBeliefStore: dev impl using append-only JSONL files. Read-mostly
    structures (links, state beliefs) are loaded once into memory.
  - Neo4jBeliefStore: future impl. Same interface.

Layout under root/:
  links.jsonl              — InternalLink records (rewritten on update)
  state_beliefs.jsonl      — StateBelief records (rewritten on update)
  outcome_beliefs.jsonl    — OutcomeBelief records (append-only)
  events.jsonl             — Event + activations (append-only)
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

from .schema import (
    BeliefActivation,
    Event,
    InternalLink,
    OutcomeBelief,
    Scope,
    StateBelief,
)


# ───── Abstract store ─────

class BeliefStore(ABC):
    """Common interface; JSONL or Neo4j-backed."""

    @abstractmethod
    def upsert_link(self, link: InternalLink) -> None: ...

    @abstractmethod
    def get_link(self, link_id: str) -> Optional[InternalLink]: ...

    @abstractmethod
    def all_links(self) -> list[InternalLink]: ...

    @abstractmethod
    def links_into(self, conclusion_belief_id: str) -> list[InternalLink]:
        """All links whose conclusion is the given belief id."""
        ...

    @abstractmethod
    def upsert_state_belief(self, belief: StateBelief) -> None: ...

    @abstractmethod
    def get_state_belief(self, belief_id: str) -> Optional[StateBelief]: ...

    @abstractmethod
    def all_state_beliefs(self) -> list[StateBelief]: ...

    @abstractmethod
    def state_beliefs_in_scope(self, scope: Scope) -> list[StateBelief]: ...

    @abstractmethod
    def append_outcome_belief(self, outcome: OutcomeBelief) -> None: ...

    @abstractmethod
    def append_event(self, event: Event) -> None: ...


# ───── JSONL implementation ─────

def _to_jsonable(obj) -> dict:
    """asdict + ensure all values are JSON-serializable."""
    return asdict(obj)


def _state_belief_from_dict(d: dict) -> StateBelief:
    return StateBelief(**d)


def _link_from_dict(d: dict) -> InternalLink:
    return InternalLink(**d)


def _outcome_from_dict(d: dict) -> OutcomeBelief:
    return OutcomeBelief(**d)


def _event_from_dict(d: dict) -> Event:
    activations = [BeliefActivation(**a) for a in d.get("activations", [])]
    payload = {k: v for k, v in d.items() if k != "activations"}
    payload["activations"] = activations
    return Event(**payload)


class JSONLBeliefStore(BeliefStore):
    """Append-only JSONL store. Links and state-beliefs use a "rewrite on
    update" pattern (small N, dozens to hundreds of records). Outcomes and
    events are pure append-only.

    Mutable in-memory caches for links and state-beliefs avoid re-reading
    the file every lookup.
    """

    LINKS_FILE = "links.jsonl"
    STATE_BELIEFS_FILE = "state_beliefs.jsonl"
    OUTCOME_BELIEFS_FILE = "outcome_beliefs.jsonl"
    EVENTS_FILE = "events.jsonl"

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._links: dict[str, InternalLink] = {}
        self._state_beliefs: dict[str, StateBelief] = {}
        self._load()

    def _load(self) -> None:
        for p in (self.root / self.LINKS_FILE,):
            if p.exists():
                for line in p.open():
                    line = line.strip()
                    if line:
                        link = _link_from_dict(json.loads(line))
                        self._links[link.id] = link
        for p in (self.root / self.STATE_BELIEFS_FILE,):
            if p.exists():
                for line in p.open():
                    line = line.strip()
                    if line:
                        b = _state_belief_from_dict(json.loads(line))
                        self._state_beliefs[b.id] = b

    def _rewrite(self, filename: str, records: Iterable[dict]) -> None:
        path = self.root / filename
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            for r in records:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        tmp.replace(path)

    def _append(self, filename: str, record: dict) -> None:
        with (self.root / filename).open("a") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    # ── Links ──

    def upsert_link(self, link: InternalLink) -> None:
        self._links[link.id] = link
        self._rewrite(
            self.LINKS_FILE,
            (_to_jsonable(l) for l in self._links.values()),
        )

    def get_link(self, link_id: str) -> Optional[InternalLink]:
        return self._links.get(link_id)

    def all_links(self) -> list[InternalLink]:
        return list(self._links.values())

    def links_into(self, conclusion_belief_id: str) -> list[InternalLink]:
        return [l for l in self._links.values() if l.conclusion_belief_id == conclusion_belief_id]

    # ── State beliefs ──

    def upsert_state_belief(self, belief: StateBelief) -> None:
        self._state_beliefs[belief.id] = belief
        self._rewrite(
            self.STATE_BELIEFS_FILE,
            (_to_jsonable(b) for b in self._state_beliefs.values()),
        )

    def get_state_belief(self, belief_id: str) -> Optional[StateBelief]:
        return self._state_beliefs.get(belief_id)

    def all_state_beliefs(self) -> list[StateBelief]:
        return list(self._state_beliefs.values())

    def state_beliefs_in_scope(self, scope: Scope) -> list[StateBelief]:
        return [b for b in self._state_beliefs.values() if b.scope == scope]

    # ── Outcomes (append-only) ──

    def append_outcome_belief(self, outcome: OutcomeBelief) -> None:
        self._append(self.OUTCOME_BELIEFS_FILE, _to_jsonable(outcome))

    # ── Events (append-only) ──

    def append_event(self, event: Event) -> None:
        d = _to_jsonable(event)
        self._append(self.EVENTS_FILE, d)
