"""Shared agent scaffolding."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from ..types import AgentState


def new_agent_id(role: str, kind: str) -> str:
    return f"{role}.{kind}.{uuid.uuid4().hex[:8]}"


@dataclass
class BaseAgent:
    id: str
    kind: str
    role: str
    state: AgentState = field(init=False)

    def __post_init__(self):
        self.state = AgentState(id=self.id, role=self.role, kind=self.kind)
