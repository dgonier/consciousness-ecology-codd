"""Per-agent rolling fitness tracker.

For each voter (or, more generally, any agent the orchestrator wants to
track), maintain a rolling window of recent (correct?, abstained?,
contributed-to-ensemble?) records. Expose summary statistics that the
PopulationManager uses to decide reproduction / death.

Two distinct signals:

  - SOLO accuracy: when this agent emitted a non-abstain vote, was it
    correct? Captures whether the agent's own outputs are good.

  - MARGINAL contribution: how often was this agent's vote DECISIVE in
    the ensemble — i.e., would the ensemble's answer have flipped if the
    voter were excluded? An agent that always agrees with the consensus
    gets high solo accuracy but ZERO marginal contribution; we want to
    notice that and select against it.

The marginal signal requires either:
  (a) tracking ensemble-without-this-voter outcomes per scenario
      (cheap if voter list is small), or
  (b) periodic full drop-tests where the panel re-evaluates with each
      voter excluded over a held-out batch.

This module provides the data structures; the orchestrator computes
both.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class AgentFitness:
    voter_id: str
    n_seen: int = 0
    n_decisive: int = 0       # emitted a non-abstain vote
    n_correct: int = 0        # of the decisive votes, n correct
    n_marginal_flip: int = 0  # n times this voter's exclusion would have flipped panel
    n_marginal_correct: int = 0  # of those flips, n where ensemble was right WITH voter

    @property
    def solo_accuracy(self) -> float:
        if self.n_decisive == 0:
            return 0.0
        return self.n_correct / self.n_decisive

    @property
    def decision_rate(self) -> float:
        if self.n_seen == 0:
            return 0.0
        return self.n_decisive / self.n_seen

    @property
    def marginal_contribution(self) -> float:
        """Of times this voter was decisive (excluding it would have
        flipped the panel), how often did its vote make the panel
        correct? Higher is better. Range [0, 1]."""
        if self.n_marginal_flip == 0:
            return 0.0
        return self.n_marginal_correct / self.n_marginal_flip

    @property
    def freerider_score(self) -> float:
        """Solo accuracy minus marginal contribution. High freerider
        score = agent is right but never moves the panel = redundant."""
        return self.solo_accuracy - self.marginal_contribution


@dataclass
class FitnessTracker:
    """Tracks fitness across all voters with a rolling window."""
    window: int = 200
    _by_voter: dict[str, AgentFitness] = field(default_factory=dict)
    _recent: dict[str, deque] = field(default_factory=dict)
    # _recent[voter_id]: deque of (correct: bool|None, decisive: bool, marginal_flip: bool, marginal_correct: bool)

    def record(
        self,
        *,
        voter_id: str,
        correct: bool | None,
        decisive: bool,
        marginal_flip: bool = False,
        marginal_correct: bool = False,
    ) -> None:
        af = self._by_voter.setdefault(voter_id, AgentFitness(voter_id=voter_id))
        af.n_seen += 1
        if decisive:
            af.n_decisive += 1
            if correct:
                af.n_correct += 1
        if marginal_flip:
            af.n_marginal_flip += 1
            if marginal_correct:
                af.n_marginal_correct += 1
        # Rolling window
        rq = self._recent.setdefault(voter_id, deque(maxlen=self.window))
        rq.append((correct, decisive, marginal_flip, marginal_correct))

    def summary(self) -> dict[str, AgentFitness]:
        return dict(self._by_voter)

    def rolling_summary(self) -> dict[str, AgentFitness]:
        """Recompute AgentFitness from the rolling window only."""
        out: dict[str, AgentFitness] = {}
        for voter_id, rq in self._recent.items():
            af = AgentFitness(voter_id=voter_id, n_seen=len(rq))
            for correct, decisive, mflip, mcor in rq:
                if decisive:
                    af.n_decisive += 1
                    if correct:
                        af.n_correct += 1
                if mflip:
                    af.n_marginal_flip += 1
                    if mcor:
                        af.n_marginal_correct += 1
            out[voter_id] = af
        return out
