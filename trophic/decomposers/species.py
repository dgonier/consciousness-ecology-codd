"""KG-backed species registry.

A *species* is a config artifact: {model_id, role, prompt_template,
diet_tags, hyperparams, alive, parent_ids, generation,
kg_context_hint}. The decomposer writes species records to the KG; the
voter factory reads them and instantiates the right concrete voter
class with the species's config.

Storage: external/decomposer_kg/species.jsonl — append-only log with
the latest record per species_id winning at compile time. Mutations
(alive flips, prompt edits) are written as new lines, not in-place
edits, so the KG is auditable.

A *cycle* is N passes, where each pass reads the SAME compiled species
roster. Between cycles, the decomposer can write new species records
(or flip alive=False) and the next cycle re-compiles.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable


# Roles parallel the trophic tiers + the multi-family apex panel.
SpeciesRole = str  # 'producer' | 'herbivore' | 'apex_voter' | 'apex_aggregator' | 'decomposer'


@dataclass
class Species:
    """KG-backed config for one agent.

    Two species can share the same model_id but differ in prompt /
    diet / hyperparams — they're different species running on the same
    substrate. Reproduction operates here.
    """
    species_id: str
    model_id: str          # e.g. 'gpt-5', 'us.anthropic.claude-sonnet-4-6',
                           # 'gemini-2.5-flash', 'local_qwen3_4b'
    role: SpeciesRole = "apex_voter"
    prompt_template: str = ""              # rendered SYSTEM block for this species
    diet_tags: list[str] = field(default_factory=list)
    hyperparams: dict = field(default_factory=dict)
    # Lifecycle
    alive: bool = True
    parent_ids: list[str] = field(default_factory=list)
    generation: int = 0
    # Hexis-side: KG record ids this species wants to attend over.
    # Local trained agents fold these into M-tensor input; API voters
    # render them into the prompt as additional context.
    kg_context_hint: list[str] = field(default_factory=list)
    # Provenance
    created_at: float = field(default_factory=time.time)
    note: str = ""

    @classmethod
    def new(
        cls,
        model_id: str,
        role: SpeciesRole,
        prompt_template: str = "",
        **kwargs,
    ) -> "Species":
        sp_id = kwargs.pop("species_id", None) or f"sp.{role}.{uuid.uuid4().hex[:8]}"
        return cls(
            species_id=sp_id,
            model_id=model_id,
            role=role,
            prompt_template=prompt_template,
            **kwargs,
        )


@dataclass
class SpeciesRegistry:
    """Append-only JSONL registry of species records.

    To get the current alive roster: read all records, group by
    species_id, take the latest record per id, filter to alive=True.
    """
    path: Path = Path("external/decomposer_kg/species.jsonl")

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, species: Species) -> None:
        """Append a species record. Use to register a new species, flip
        alive, or edit prompt — the latest record per id wins on read."""
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(species)) + "\n")

    def write_all(self, species_list: Iterable[Species]) -> None:
        for s in species_list:
            self.write(s)

    def read_all(self) -> list[Species]:
        """Return the latest record per species_id, in insertion order
        of first appearance."""
        if not self.path.exists():
            return []
        latest: dict[str, Species] = {}
        order: list[str] = []
        for ln in self.path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            d = json.loads(ln)
            sp = Species(**d)
            if sp.species_id not in latest:
                order.append(sp.species_id)
            latest[sp.species_id] = sp
        return [latest[sid] for sid in order]

    def alive(self) -> list[Species]:
        """Current alive roster — what the next cycle should compile."""
        return [s for s in self.read_all() if s.alive]

    def by_role(self, role: SpeciesRole) -> list[Species]:
        return [s for s in self.alive() if s.role == role]

    def get(self, species_id: str) -> Species | None:
        for s in self.read_all():
            if s.species_id == species_id:
                return s
        return None

    def kill(self, species_id: str, note: str = "") -> Species | None:
        """Mark species as dead by writing a new record with alive=False.
        The original record stays in the log for audit; only the latest
        wins on read."""
        sp = self.get(species_id)
        if sp is None:
            return None
        sp.alive = False
        sp.note = note or f"killed at {time.time():.0f}"
        self.write(sp)
        return sp


def bootstrap_default_panel(registry: SpeciesRegistry) -> list[Species]:
    """If the registry is COMPLETELY EMPTY, seed it with the default
    apex+herb panel. If the registry has any records (even custom
    ones), this is a no-op — do not pollute custom rosters with the
    default seeds.

    Each default species's prompt_template is empty so concrete voter
    classes fall back to evidence.SYSTEM. Once the decomposer mutates
    a species, its template becomes non-empty and overrides default.
    """
    if registry.read_all():
        return []  # registry has content; don't pollute it
    HERB_TECHNICAL_PROMPT = (
        "You are a TECHNICAL-analysis herbivore in a multi-species panel."
        " Read the evidence and produce a 2-3 sentence synthesis from a"
        " technical / price-action lens (momentum, mean-reversion,"
        " volume divergence, support/resistance). Do NOT make a final"
        " UP/DOWN trade decision; the apex panel does that. Reply on three"
        " lines:\nDIET: technical\nSYNTHESIS: <2-3 sentences>"
        "\nDIRECTION_HINT: <up | down | none>"
    )
    HERB_FUNDAMENTAL_PROMPT = (
        "You are a FUNDAMENTAL-analysis herbivore in a multi-species panel."
        " Read the evidence and produce a 2-3 sentence synthesis from a"
        " fundamental / event-driven lens (news flow, earnings, sector"
        " catalysts, sentiment shifts in tweets). Do NOT make a final"
        " UP/DOWN trade decision. Reply on three lines:\nDIET: fundamental"
        "\nSYNTHESIS: <2-3 sentences>"
        "\nDIRECTION_HINT: <up | down | none>"
    )
    seeds = [
        # Apex voters
        Species(
            species_id="bootstrap.openai.gpt-5",
            model_id="gpt-5",
            role="apex_voter",
            note="bootstrap",
        ),
        Species(
            species_id="bootstrap.bedrock.sonnet-4-6",
            model_id="us.anthropic.claude-sonnet-4-6",
            role="apex_voter",
            note="bootstrap; via bedrock",
        ),
        Species(
            species_id="bootstrap.gemini.flash",
            model_id="gemini-2.5-flash",
            role="apex_voter",
            note="bootstrap; thinking disabled",
        ),
        Species(
            species_id="bootstrap.local.qwen3_4b",
            model_id="local_qwen3_4b",
            role="apex_voter",
            note="bootstrap; HEXIS substrate (M-tensor channel available)",
        ),
        # Herbivores (Phase B): synthesis agents that pre-digest evidence
        # for the apex panel. Different lenses = different species.
        Species(
            species_id="bootstrap.herb.technical.gemini",
            model_id="gemini-2.5-flash",
            role="herbivore",
            prompt_template=HERB_TECHNICAL_PROMPT,
            diet_tags=["technical"],
            note="bootstrap; technical lens on flash",
        ),
        Species(
            species_id="bootstrap.herb.fundamental.gemini",
            model_id="gemini-2.5-flash",
            role="herbivore",
            prompt_template=HERB_FUNDAMENTAL_PROMPT,
            diet_tags=["fundamental"],
            note="bootstrap; fundamental lens on flash",
        ),
    ]
    existing_ids = {s.species_id for s in registry.read_all()}
    new_seeds: list[Species] = []
    for sp in seeds:
        if sp.species_id in existing_ids:
            continue
        registry.write(sp)
        new_seeds.append(sp)
    return new_seeds
