"""Apex voter ecosystem.

The trophic stack's job is to prepare a rich evidence packet (producer
broadcasts + herb syntheses + numeric features); apex voters then read
that packet and emit a directional prediction with confidence and
self-perplexity. Aggregation across voters is rank-vote /
perplexity-weighted soft-vote.

Designed for multi-provider voting (OpenAI, Anthropic, Gemini, Qwen
via OpenRouter, plus local Qwen). The local voter works today; the API
voters are stubs to be wired when keys are configured.
"""
from .base import ApexVoter, VoterResponse, EvidencePacket
from .evidence import build_evidence_packet
from .local_qwen import LocalQwenVoter
from .api_voters import OpenAIVoter, AnthropicVoter, GeminiVoter, OpenRouterVoter
from .aggregate import (
    plurality, confidence_weighted, perplexity_weighted, rank_vote_borda,
    deliberation_packet, EnsembleDecision,
)

__all__ = [
    "ApexVoter", "VoterResponse", "EvidencePacket",
    "build_evidence_packet",
    "LocalQwenVoter",
    "OpenAIVoter", "AnthropicVoter", "GeminiVoter", "OpenRouterVoter",
    "plurality", "confidence_weighted", "perplexity_weighted",
    "rank_vote_borda", "deliberation_packet", "EnsembleDecision",
]
