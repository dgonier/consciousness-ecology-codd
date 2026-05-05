"""Voter factory: instantiate a concrete ApexVoter from a Species record.

The Species is the config artifact (read from the KG); the voter is
the runtime object (a class instance). One species → one voter. Two
species with the same model_id but different prompts produce two
distinct voter instances.

Maps model_id → concrete class. Override prompt_template on the
species to differentiate; voter's `species_template` field is read at
vote time and prepended to the EvidencePacket text.
"""
from __future__ import annotations

from .base import ApexVoter
from .api_voters import (
    AnthropicVoter, GeminiVoter, OpenAIVoter, OpenRouterVoter,
)
from .local_qwen import LocalQwenVoter


# Identifies the concrete class per model_id keyword.
def _resolve_class_for_model(model_id: str) -> type[ApexVoter]:
    m = (model_id or "").lower()
    if m.startswith("local") or m.startswith("qwen3") or "local" in m:
        return LocalQwenVoter
    if m.startswith("gpt") or "openai" in m:
        return OpenAIVoter
    if "claude" in m or "anthropic" in m or m.startswith("us.anthropic"):
        return AnthropicVoter  # routed through Bedrock
    if m.startswith("gemini") or "google" in m:
        return GeminiVoter
    if "openrouter" in m or "/" in m:  # provider/model slug
        return OpenRouterVoter
    raise ValueError(f"voter_factory: no concrete class for model_id={model_id!r}")


def voter_from_species(species) -> ApexVoter:
    """Build a concrete ApexVoter from a Species record. Returns an
    instance whose voter_id is suffixed with the species_id so two
    species running on the same substrate produce distinguishable
    fitness signals.

    The Species's prompt_template (if non-empty) overrides the default
    SYSTEM block at vote time via ApexVoter.species_template — the
    EvidencePacket builder checks for this and prepends it.
    """
    cls = _resolve_class_for_model(species.model_id)
    # Each concrete class accepts model= as the first kwarg; for local
    # Qwen the singleton ModelHost is shared, so model= is a no-op.
    if cls is LocalQwenVoter:
        v = LocalQwenVoter()
    else:
        v = cls(model=species.model_id)
    # Brand the voter with its species so fitness records are
    # species-keyed (multiple species can share a model_id).
    v.voter_id = f"{v.voter_id}::{species.species_id}"
    v.species = species
    # Custom prompt template (Species-defined) overrides the default
    # SYSTEM block. None / empty falls back to evidence.SYSTEM.
    v.species_template = species.prompt_template or None
    return v


def build_panel_from_registry(registry) -> list[ApexVoter]:
    """Read all alive apex_voter species from the registry and build a
    voter for each. Skips species whose voter is_available()=False
    (e.g. missing API key)."""
    panel = []
    for sp in registry.alive():
        if sp.role != "apex_voter":
            continue
        try:
            v = voter_from_species(sp)
        except Exception as e:
            print(f"[factory] skipped species {sp.species_id}: {e}")
            continue
        if not v.is_available():
            print(f"[factory] species {sp.species_id} unavailable (no creds for {sp.model_id})")
            continue
        panel.append(v)
    return panel
