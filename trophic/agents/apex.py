"""Apex predator — STUB.

Final recommendation tier. Will be backed by a frontier model (Bedrock
Claude Opus, GPT-4.1, etc.) and emit final actions/recommendations. Not
built in v1; predator (Claude on Bedrock) plays the optimization-signal
role for now.
"""
from __future__ import annotations

from ..types import ApexDecision, IntegratedAnalysis


class ApexAgent:
    async def decide(self, analysis: IntegratedAnalysis) -> ApexDecision:
        raise NotImplementedError(
            "Apex tier is stubbed in v1. The predator (Claude on Bedrock)"
            " currently provides the optimization signal that flows back"
            " through the ecology."
        )
