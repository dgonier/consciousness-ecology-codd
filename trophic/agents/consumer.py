"""Tier-2 consumer — STUB.

Will integrate multiple herbivore syntheses into a single token-mediated
analysis for the apex tier. Embedding-in, token-out (this is the
embedding/token boundary the architecture commits to). Not built in v1.
"""
from __future__ import annotations

from typing import Iterable

from ..types import ConsumedSynthesis, IntegratedAnalysis


class Tier2Consumer:
    async def integrate(self, syntheses: Iterable[ConsumedSynthesis]) -> IntegratedAnalysis:
        raise NotImplementedError(
            "Tier-2 consumer is stubbed in v1. It will integrate herbivore"
            " syntheses with embedding-level input and token-level output to"
            " apex. Likely backed by a medium model (Qwen3-30B-A3B class)."
        )
