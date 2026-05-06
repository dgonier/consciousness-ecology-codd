"""Herbivore voters — synthesis agents that pre-digest evidence for the apex.

In Phase B the herbivore tier is reframed as a panel of *synthesis*
agents that read the raw scenario evidence (OHLCV + tweets + Chronos
features) and produce a short analytical synthesis paragraph. The
apex panel reads these syntheses *alongside* the raw evidence, so each
voter benefits from the herbivore's framing.

Like apex voters, herbivore voters are KG-driven species. They have
the same lifecycle: bootstrap → fitness tracking → reproduction /
death / gap-fill via Opus.

Different from apex voters in two ways:
  - role='herbivore' on their Species record
  - they emit a `synthesis` string + diet_tag, not a direction. No
    perplexity / confidence parsing.

The eval orchestrator runs herbivore voters BEFORE apex voters, then
folds their syntheses into each apex EvidencePacket so every voter
sees the same digested context.
"""
from .base import HerbivoreSynthesis, HerbivoreVoter
from .api_voters import (
    OpenAIHerbivore, AnthropicBedrockHerbivore, GeminiHerbivore,
)
from .local_qwen import LocalQwenHerbivore
from .factory import build_herbivore_panel_from_registry, herbivore_from_species

__all__ = [
    "HerbivoreSynthesis", "HerbivoreVoter",
    "OpenAIHerbivore", "AnthropicBedrockHerbivore", "GeminiHerbivore",
    "LocalQwenHerbivore",
    "build_herbivore_panel_from_registry", "herbivore_from_species",
]
