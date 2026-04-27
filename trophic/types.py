"""Pydantic message types passed between agents and into substrate.

The system is hidden-state-native. Each broadcast carries:
  - channel_embedding: the agent's pooled last-layer hidden state
        (the "meat" — what gets attended to by hunters)
  - decoded_text:      the optional token decoding, for legibility only
        (never read back into another agent's forward pass in v1.5)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------- raw input ----------

class RawInput(BaseModel):
    """A single observation from the outside world (synthetic feed in v1)."""
    id: str
    source: str  # "ohlcv" | "trades" | "book" | "filing" | "press" | "options" | "halt" | ...
    payload: dict[str, Any]
    created_at: datetime = Field(default_factory=_now)


# ---------- producer output ----------

DietTag = Literal[
    "is_price_event",
    "is_disclosure_event",
    "is_anomaly",
    "is_options_flow",
]


# Tier of the broadcast pool the item lives in. Producers emit "substrate";
# herbivores emit "herbivore_broadcast"; predators emit "predator_broadcast".
BroadcastTier = Literal["substrate", "herbivore_broadcast", "predator_broadcast"]


class Broadcast(BaseModel):
    """A unified hidden-state broadcast at any trophic tier.

    Replaces the prior ProducedSubstrate / ConsumedSynthesis split so the
    same Channel mechanics work at every edge. The diet_tags at higher
    tiers identify the agent's *kind* (e.g. 'from_technical_herbivore'),
    so hunters can filter by source kind in addition to (or instead of)
    semantic content.
    """
    id: str
    tier: BroadcastTier
    agent_id: str            # who broadcast it
    agent_kind: str          # producer kind / herbivore kind / predator kind
    parent_input_ids: list[str] = Field(default_factory=list)
        # For tier='substrate': the source RawInput.id.
        # For higher tiers: the IDs of broadcasts this agent consumed.
    diet_tags: list[str]     # what kinds of hunters this should match
    payload: dict[str, Any] = Field(default_factory=dict)
    decoded_text: str | None = None  # legibility-only token decoding
    channel_embedding: list[float]   # the actual hidden-state broadcast
    created_at: datetime = Field(default_factory=_now)
    created_tick: int = 0
    abstained: bool = False  # True iff hunter chose to STOP (no real broadcast)


# Aliases so older code reads naturally.
ProducedSubstrate = Broadcast
ConsumedSynthesis = Broadcast


# ---------- judgment from the top-of-stack judge ----------

class PredatorJudgment(BaseModel):
    """A scalar judgment of an upstream broadcast.

    Name kept for backward compat (substrate.py, decomposer.py reference
    it). In v1.5 this is what the apex judge emits about predator
    broadcasts. The `synthesis_id` field is the id of whatever broadcast
    is being judged (predator broadcast in v1.5).
    """
    id: str
    synthesis_id: str  # target broadcast id
    score: float  # [0,1]
    rationale: str
    predicted_action: str | None = None
    created_at: datetime = Field(default_factory=_now)
    created_tick: int = 0


# ---------- decomposer output ----------

DecomposerRecordType = Literal["useful", "misleading", "falsified", "unreliable_source", "wasted"]


class DecomposerRecord(BaseModel):
    id: str
    record_type: DecomposerRecordType
    target_id: str  # substrate_id, producer_id, herbivore_id, or synthesis_id
    target_kind: Literal["substrate", "producer", "herbivore", "synthesis"]
    rationale: str
    weight: float = 1.0  # magnitude of credit assignment
    created_at: datetime = Field(default_factory=_now)
    created_tick: int = 0


# ---------- upper-tier (stubs) ----------

class IntegratedAnalysis(BaseModel):
    """Tier-2 consumer output. STUB. Token-mediated handoff to apex."""
    id: str
    integrated_synthesis_ids: list[str]
    content: str
    confidence: float
    escalate_to_apex: bool


class ApexDecision(BaseModel):
    """Tier-3 apex output. STUB. Final recommendation."""
    id: str
    integrated_analysis_id: str
    content: str
    reasoning: str


# ---------- agent state ----------

class AgentState(BaseModel):
    id: str
    role: Literal["producer", "herbivore", "predator", "decomposer"]
    kind: str  # producer kind ("tickdelta") / herbivore kind / predator kind
    alive: bool = True
    energy: float = 1.0
    born_tick: int = 0
    died_tick: int | None = None
    reputation: float = 0.0  # running score from decomposer
    params: dict[str, Any] = Field(default_factory=dict)
