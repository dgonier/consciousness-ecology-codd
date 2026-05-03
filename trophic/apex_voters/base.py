"""Apex voter interface — common shape for OpenAI / Anthropic / Gemini /
Qwen-via-OpenRouter / local Qwen voters.

Each voter ingests an EvidencePacket and emits a VoterResponse with:
  - direction: 'up' | 'down' | None (abstain)
  - confidence: 0..1, voter's self-reported confidence
  - perplexity: token-level log-perplexity of the answer span (lower =
    more confident the model is in its own emission). For voters that
    don't expose token logprobs, use float('inf') as a sentinel.
  - raw_text: the model's full response, for debugging / decomposer
    feedback.

Aggregation across voters then weights by 1/perplexity (or by
self-confidence) to produce the ensemble decision.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class EvidencePacket:
    """The trophic stack's contribution to the apex voter.

    Each voter reads `text` (the rendered evidence) and `metadata`
    (machine-readable signals like herb confidences, perplexity).
    Voters can choose to use either or both.
    """
    scenario_name: str
    ticker: str
    text: str  # full prompt body for the voter (system + user)
    metadata: dict = field(default_factory=dict)


@dataclass
class VoterResponse:
    voter_id: str  # e.g. "local_qwen", "anthropic_sonnet", "openai_gpt5"
    direction: str | None  # 'up' | 'down' | None (abstain)
    confidence: float | None  # voter's self-reported confidence in [0, 1]
    perplexity: float  # log-perplexity of answer span (lower = more confident)
    raw_text: str  # full response text
    # 2026-05-03: structured reasoning. The voter's natural-language
    # justification for its answer, separate from the XML answer span.
    # Reviewable by humans, visible to the decomposer for KG traces, and
    # available as round-2 deliberation context.
    reasoning: str = ""
    # Pointers into the EvidencePacket that the voter found most
    # informative. Free-form strings the voter can use to cite specific
    # bars / tweets / forecast features. Optional; voters that don't
    # support structured output leave this empty.
    evidence_citations: list[str] = field(default_factory=list)
    # Provider-side debug info (model id, latency, tokens used) for
    # auditability + cost tracking.
    provider_meta: dict = field(default_factory=dict)


class ApexVoter(ABC):
    """One voter in the apex ensemble. Different concrete subclasses for
    each model family. Stateless after construction."""

    voter_id: str

    @abstractmethod
    def vote(self, evidence: EvidencePacket) -> VoterResponse:
        """Synchronous vote. Subclasses may make this an async wrapper."""
        ...

    def is_available(self) -> bool:
        """True if the voter can actually be called right now (e.g. API
        key present). Caller skips unavailable voters."""
        return True
