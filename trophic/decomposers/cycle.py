"""Evolutionary cycle orchestrator.

Glossary:
  pass    one end-to-end scenario eval
  cycle   N passes constituting a generation; decomposer fires
          evolution updates between cycles

After each cycle:
  1. Read fitness from this cycle's tracker.
  2. PopulationManager.decide() → list of EvolutionDecision (keep/die/
     reproduce/thin_mutate).
  3. For each DIE: registry.kill(species_id).
  4. For top-2 species (if reproduce-eligible): reproducer.reproduce()
     → write child species.
  5. If panel_correctness < threshold: gap_analyzer.propose_gap_species()
     → write new species filling identified gap.
  6. Next cycle re-compiles the panel from the registry.

This module exposes pure functions; the eval orchestrator (apex_vote_eval)
calls them at cycle boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .agent_feedback import AgentFeedback
from .fitness import FitnessTracker, AgentFitness
from .gap_analyzer import propose_gap_species
from .herb_fitness import HerbFitness, HerbFitnessTracker
from .observation import Observation
from .opus_judge import is_opus_available
from .population_manager import PopulationManager, EvolutionDecision
from .reproducer import reproduce, select_top_pair
from .species import Species, SpeciesRegistry


@dataclass
class CycleReport:
    cycle_index: int
    n_passes: int
    panel_size_before: int
    panel_size_after: int
    deaths: list[str] = field(default_factory=list)
    reproductions: list[str] = field(default_factory=list)
    gap_fills: list[str] = field(default_factory=list)
    # Phase B: separate buckets for herb-tier evolution actions
    herb_deaths: list[str] = field(default_factory=list)
    herb_reproductions: list[str] = field(default_factory=list)
    decisions: list[EvolutionDecision] = field(default_factory=list)
    panel_correctness: float = 0.0


def run_evolutionary_update(
    *,
    cycle_index: int,
    registry: SpeciesRegistry,
    fitness: FitnessTracker,
    observations: Sequence[Observation],
    panel: Sequence[Species],
    n_passes: int,
    enable_reproduction: bool = True,
    enable_gap_analysis: bool = True,
    gap_correctness_threshold: float = 0.6,
    population_manager: PopulationManager | None = None,
    herb_fitness: HerbFitnessTracker | None = None,
    herb_panel: Sequence[Species] | None = None,
    herb_min_seen: int = 8,
    herb_kill_contribution: float = -0.10,
    herb_reproduce_contribution: float = 0.20,
) -> CycleReport:
    """One evolutionary update at cycle boundary. Mutates the registry
    via writes; returns a CycleReport with what changed.

    Reproduction and gap-analysis call Opus 4.7 via Bedrock — slow and
    costs tokens. Disable via the enable_* flags for cheap dry runs."""
    population_manager = population_manager or PopulationManager()
    by_voter_id = fitness.summary()

    # 1. PopulationManager decisions
    decisions = population_manager.decide(fitness)
    deaths: list[str] = []
    for d in decisions:
        if d.action == "die":
            # voter_id format: "<class>::<species_id>"; recover species id
            sp_id = d.voter_id.split("::", 1)[-1]
            killed = registry.kill(sp_id, note=d.reason)
            if killed is not None:
                deaths.append(sp_id)

    # 2. Reproduction: top-2 species → child
    reproductions: list[str] = []
    panel_correctness = _panel_correctness(observations)
    if (
        enable_reproduction
        and is_opus_available()
        and len(panel) >= 2
    ):
        pair = select_top_pair(panel, by_voter_id, by="combined")
        if pair is not None:
            a, b = pair
            # Find their fitness records
            fit_a = _fit_for_species(by_voter_id, a)
            fit_b = _fit_for_species(by_voter_id, b)
            child = reproduce(
                a, b, fit_a, fit_b,
                panel_summary=(
                    f"panel correctness this cycle: {panel_correctness:.2%}"
                    f" over {n_passes} passes"
                ),
            )
            if child is not None:
                registry.write(child)
                reproductions.append(child.species_id)

    # 3. Gap analysis: only if panel is underperforming
    gap_fills: list[str] = []
    if (
        enable_gap_analysis
        and is_opus_available()
        and panel_correctness < gap_correctness_threshold
        and observations
    ):
        gap = propose_gap_species(panel, observations)
        if gap is not None:
            registry.write(gap)
            gap_fills.append(gap.species_id)

    # 4. HERB-tier evolution. Different fitness signal (sampled-dropout
    # contribution) so we don't reuse PopulationManager. Cheap thresholds:
    #   contribution < herb_kill_contribution → die
    #   contribution > herb_reproduce_contribution AND another herb also
    #     above threshold → mate top 2 herbs.
    herb_deaths: list[str] = []
    herb_reproductions: list[str] = []
    if herb_fitness is not None and herb_panel:
        by_herb = herb_fitness.summary()
        # Map each herb species to its fitness record
        herb_pairs: list[tuple[Species, "HerbFitness"]] = []
        for sp in herb_panel:
            for hid, hf in by_herb.items():
                if hid.endswith(f"::{sp.species_id}"):
                    herb_pairs.append((sp, hf))
                    break
        # Death decisions
        for sp, hf in herb_pairs:
            if hf.n_seen < herb_min_seen:
                continue
            if hf.contribution < herb_kill_contribution:
                killed = registry.kill(
                    sp.species_id,
                    note=(
                        f"herb contribution {hf.contribution:+.3f} < "
                        f"{herb_kill_contribution} after {hf.n_seen} obs"
                    ),
                )
                if killed is not None:
                    herb_deaths.append(sp.species_id)
        # Reproduction: top-2 herbs both above threshold
        if (
            enable_reproduction
            and is_opus_available()
            and len(herb_pairs) >= 2
        ):
            sorted_pairs = sorted(
                herb_pairs, key=lambda p: p[1].contribution, reverse=True,
            )
            if (
                sorted_pairs[0][1].contribution >= herb_reproduce_contribution
                and sorted_pairs[1][1].contribution >= 0  # second-best at least non-negative
                and sorted_pairs[0][1].n_seen >= herb_min_seen
                and sorted_pairs[1][1].n_seen >= herb_min_seen
            ):
                a_sp, a_fit = sorted_pairs[0]
                b_sp, b_fit = sorted_pairs[1]
                child = reproduce(
                    a_sp, b_sp, None, None,
                    panel_summary=(
                        f"herb tier: {a_sp.species_id} contribution {a_fit.contribution:+.3f}, "
                        f"{b_sp.species_id} contribution {b_fit.contribution:+.3f}. "
                        f"Mate them at the herbivore tier — child should "
                        f"emit a synthesis paragraph (not a directional "
                        f"vote), with role='herbivore'."
                    ),
                )
                if child is not None:
                    # Force herbivore role on child even if Opus drifted
                    child.role = "herbivore"
                    registry.write(child)
                    herb_reproductions.append(child.species_id)

    panel_after = registry.alive()
    return CycleReport(
        cycle_index=cycle_index,
        n_passes=n_passes,
        panel_size_before=len(panel),
        panel_size_after=len(panel_after),
        deaths=deaths,
        reproductions=reproductions,
        gap_fills=gap_fills,
        herb_deaths=herb_deaths,
        herb_reproductions=herb_reproductions,
        decisions=decisions,
        panel_correctness=panel_correctness,
    )


def _panel_correctness(observations: Sequence[Observation]) -> float:
    if not observations:
        return 0.0
    n = 0
    n_correct = 0
    for o in observations:
        if not o.target_direction:
            continue
        if not o.ensemble.get("direction"):
            continue
        n += 1
        if o.ensemble["direction"] == o.target_direction:
            n_correct += 1
    return n_correct / max(n, 1)


def _fit_for_species(
    fitness: dict[str, AgentFitness],
    species: Species,
) -> AgentFitness | None:
    for fid, af in fitness.items():
        if fid.endswith(f"::{species.species_id}"):
            return af
    return None


def render_cycle_report(report: CycleReport) -> str:
    lines = [
        "",
        f"=== CYCLE {report.cycle_index} EVOLUTION REPORT ===",
        f"  passes:             {report.n_passes}",
        f"  panel correctness:  {report.panel_correctness:.2%}",
        f"  panel size:         {report.panel_size_before} → {report.panel_size_after}",
        f"  apex deaths:        {report.deaths or '(none)'}",
        f"  apex reproductions: {report.reproductions or '(none)'}",
        f"  apex gap_fills:     {report.gap_fills or '(none)'}",
        f"  herb deaths:        {report.herb_deaths or '(none)'}",
        f"  herb reproductions: {report.herb_reproductions or '(none)'}",
    ]
    return "\n".join(lines)
