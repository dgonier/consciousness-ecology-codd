"""Build a herbivore panel from KG species records."""
from __future__ import annotations

from .base import HerbivoreVoter
from .api_voters import (
    OpenAIHerbivore, AnthropicBedrockHerbivore, GeminiHerbivore,
)
from .local_qwen import LocalQwenHerbivore


def _resolve_class_for_model(model_id: str) -> type[HerbivoreVoter]:
    m = (model_id or "").lower()
    if m.startswith("local") or m.startswith("qwen3") or "local" in m:
        return LocalQwenHerbivore
    if m.startswith("gpt") or "openai" in m:
        return OpenAIHerbivore
    if "claude" in m or "anthropic" in m or m.startswith("us.anthropic"):
        return AnthropicBedrockHerbivore
    if m.startswith("gemini") or "google" in m:
        return GeminiHerbivore
    raise ValueError(f"herbivore_factory: no class for model_id={model_id!r}")


def _research_tool_for_species(species):
    """Attach a research tool when the species's diet implies it needs
    external lookup (e.g. fundamental / news lenses)."""
    diet = set(getattr(species, "diet_tags", []) or [])
    if not (diet & {"fundamental", "news", "macro"}):
        return None
    # Prefer OpenAI web search if available; else mock for tests.
    try:
        from ..research_tools import OpenAIWebSearchTool
        tool = OpenAIWebSearchTool()
        if tool.is_available():
            return tool
    except Exception:
        pass
    return None


def herbivore_from_species(species) -> HerbivoreVoter:
    """Build a HerbivoreVoter from a Species record (role='herbivore')."""
    cls = _resolve_class_for_model(species.model_id)
    research_tool = _research_tool_for_species(species)
    if cls is LocalQwenHerbivore:
        h = LocalQwenHerbivore(
            species_template=species.prompt_template or None,
            research_tool=research_tool,
        )
    else:
        h = cls(
            model=species.model_id,
            species_template=species.prompt_template or None,
        )
    # Brand the herb_id with species so fitness records are species-keyed
    h.herb_id = f"{h.herb_id}::{species.species_id}"
    h.species = species
    return h


def build_herbivore_panel_from_registry(registry) -> list[HerbivoreVoter]:
    """Read alive role='herbivore' species from the registry; build
    one voter per species. Skip unavailable ones (no API key)."""
    panel = []
    for sp in registry.alive():
        if sp.role != "herbivore":
            continue
        try:
            h = herbivore_from_species(sp)
        except Exception as e:
            print(f"[herbivore_factory] skipped species {sp.species_id}: {e}")
            continue
        if not h.is_available():
            print(f"[herbivore_factory] species {sp.species_id} unavailable "
                  f"(no creds for {sp.model_id})")
            continue
        panel.append(h)
    return panel
