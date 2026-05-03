"""PopulationManager — evolutionary policy over a voter panel.

Reads FitnessTracker summaries and emits EvolutionDecision suggestions
that the orchestrator chooses to act on (typically at session
boundaries; not mid-eval). The manager produces *suggestions*, not
mutations — keeps the policy auditable and reversible.

Policies (start conservative):

  - DEATH: a voter that has been seen >= min_seen times AND has
    solo_accuracy < death_acc_threshold AND marginal_contribution <
    death_marg_threshold should be retired. We don't kill voters that
    haven't been observed enough yet (insufficient data).

  - REPRODUCTION: a voter that has been seen >= min_seen times AND has
    solo_accuracy + marginal_contribution > reproduce_combined_threshold
    is a good candidate to clone with mutation. The manager suggests a
    set of CLONE_PROPOSALS; the orchestrator decides which to actually
    instantiate (because instantiating a real voter may incur API cost).

  - FREE-RIDER: a voter with high solo accuracy but zero marginal
    contribution is consuming budget without changing outcomes. We
    suggest a "thin" mutation rather than full clone (e.g., raise
    temperature) so the variant has a chance to disagree with consensus.

Mutation primitives (apply at orchestrator-level when reproducing):
  - temperature_shift:     ±0.1 from parent's effective temperature
  - prompt_template_swap:  use one of N prebuilt prompt variants
  - model_variant:         within-family upgrade (sonnet → opus, etc.)

The manager is provider-agnostic — it just emits clone-with-mutation
intentions; the orchestrator maps them to concrete voter constructions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .fitness import AgentFitness, FitnessTracker


MutationKind = Literal["temperature_shift", "prompt_template_swap", "model_variant"]


@dataclass
class EvolutionDecision:
    voter_id: str
    action: Literal["keep", "die", "reproduce", "thin_mutate"]
    reason: str
    suggested_mutation: MutationKind | None = None


@dataclass
class PopulationManager:
    # Don't make any decision until at least this many observations.
    min_seen: int = 20
    # DEATH thresholds
    death_acc_threshold: float = 0.40         # below random on a 50% prior
    death_marg_threshold: float = 0.30        # almost never moves the panel
    # REPRODUCTION threshold
    reproduce_combined_threshold: float = 1.20  # solo + marginal both meaningful
    # FREE-RIDER detection
    freerider_threshold: float = 0.40         # solo - marginal > this

    def decide(self, tracker: FitnessTracker) -> list[EvolutionDecision]:
        decisions: list[EvolutionDecision] = []
        for voter_id, af in tracker.summary().items():
            decisions.append(self._decide_one(af))
        return decisions

    def _decide_one(self, af: AgentFitness) -> EvolutionDecision:
        if af.n_seen < self.min_seen:
            return EvolutionDecision(
                voter_id=af.voter_id, action="keep",
                reason=f"insufficient data ({af.n_seen}/{self.min_seen})",
            )
        # DEATH: low solo accuracy AND low marginal contribution
        if (
            af.solo_accuracy < self.death_acc_threshold
            and af.marginal_contribution < self.death_marg_threshold
        ):
            return EvolutionDecision(
                voter_id=af.voter_id, action="die",
                reason=(
                    f"acc={af.solo_accuracy:.2f}<{self.death_acc_threshold} "
                    f"AND marginal={af.marginal_contribution:.2f}<{self.death_marg_threshold}"
                ),
            )
        # REPRODUCTION: combined score above threshold
        combined = af.solo_accuracy + af.marginal_contribution
        if combined >= self.reproduce_combined_threshold:
            return EvolutionDecision(
                voter_id=af.voter_id, action="reproduce",
                reason=f"combined acc+marginal={combined:.2f} >= {self.reproduce_combined_threshold}",
                suggested_mutation="prompt_template_swap",
            )
        # FREE-RIDER: high accuracy but never moves the panel
        if af.freerider_score > self.freerider_threshold:
            return EvolutionDecision(
                voter_id=af.voter_id, action="thin_mutate",
                reason=(
                    f"freerider: solo={af.solo_accuracy:.2f} but "
                    f"marginal={af.marginal_contribution:.2f}"
                ),
                suggested_mutation="temperature_shift",
            )
        return EvolutionDecision(
            voter_id=af.voter_id, action="keep",
            reason=(
                f"acc={af.solo_accuracy:.2f} marginal={af.marginal_contribution:.2f} "
                f"combined={combined:.2f}"
            ),
        )

    def render_report(self, decisions: list[EvolutionDecision]) -> str:
        rows = ["EVOLUTION REPORT", "=" * 60]
        by_action: dict[str, list[EvolutionDecision]] = {}
        for d in decisions:
            by_action.setdefault(d.action, []).append(d)
        for action in ("keep", "thin_mutate", "reproduce", "die"):
            if action not in by_action:
                continue
            rows.append(f"\n{action.upper()} ({len(by_action[action])}):")
            for d in by_action[action]:
                line = f"  {d.voter_id:30s}  {d.reason}"
                if d.suggested_mutation:
                    line += f"  [mutation: {d.suggested_mutation}]"
                rows.append(line)
        return "\n".join(rows)
