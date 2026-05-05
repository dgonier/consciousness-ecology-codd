"""Herbivore voter interface — synthesis-paragraph emitter.

Reads scenario evidence text (same shape as the apex EvidencePacket)
+ optional inter-tier signals from upstream producers, returns a
short analytical synthesis paragraph that downstream apex voters can
read as additional context.

Differs from ApexVoter in payload: HerbivoreSynthesis is a free-text
string + a diet_tag and confidence, not a direction. The trough
attention metaphor is preserved at the conceptual level — different
herb species apply different analytical lenses (technical, fundamental,
event-driven, contrarian) — but at this layer they're all just
"give the apex a paragraph that frames the evidence in your way".
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class HerbivoreSynthesis:
    """One herbivore's contribution to the apex's evidence packet."""
    herb_id: str             # e.g. "openai_gpt-5::herb.technical.gen0"
    species_id: str          # e.g. "herb.technical.gen0"
    diet_tag: str            # 'technical' | 'fundamental' | 'macro' | etc.
    synthesis: str           # 1-3 sentence analytical paragraph
    confidence: float | None # voter's self-reported confidence in [0, 1]
    raw_text: str            # full LLM response, for debugging
    provider_meta: dict = field(default_factory=dict)
    # Provenance
    direction_hint: str | None = None  # 'up' | 'down' | None (optional)


class HerbivoreVoter(ABC):
    """One herbivore in the panel. Stateless after construction."""

    herb_id: str
    species_template: str | None = None

    @abstractmethod
    def synthesize(self, scenario_text: str) -> HerbivoreSynthesis:
        """Read raw scenario evidence, emit a synthesis paragraph."""
        ...

    def is_available(self) -> bool:
        return True
