"""Aggregation across apex voters.

Three strategies, all over a list[VoterResponse]:

  1. plurality: each voter casts 1 vote for its direction; majority wins;
     ties → None (abstain). Simple sanity baseline.
  2. confidence_weighted: sum each voter's self-reported confidence on
     its chosen direction; argmax. Confidence acts as a weight.
  3. perplexity_weighted: weight each vote by 1/perplexity (lower
     perplexity = higher weight). The "rank-vote" you described.

Aggregation also returns a metadata dict capturing which voters were
seen, their answers, and the winning weight.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .base import VoterResponse


@dataclass
class EnsembleDecision:
    direction: str | None  # 'up' | 'down' | None
    confidence: float  # 0..1, derived from winning weight share
    weight_up: float
    weight_down: float
    n_voters: int
    n_decisive: int  # voters that emitted up or down (not None)
    method: str
    voters: list[VoterResponse]


def _split_weights(votes: list[VoterResponse], weight_fn) -> tuple[float, float]:
    w_up = 0.0
    w_down = 0.0
    for v in votes:
        if v.direction == "up":
            w_up += weight_fn(v)
        elif v.direction == "down":
            w_down += weight_fn(v)
    return w_up, w_down


def plurality(votes: list[VoterResponse]) -> EnsembleDecision:
    w_up, w_down = _split_weights(votes, lambda v: 1.0)
    return _decide(votes, w_up, w_down, method="plurality")


def confidence_weighted(votes: list[VoterResponse]) -> EnsembleDecision:
    w_up, w_down = _split_weights(
        votes,
        lambda v: max(v.confidence or 0.0, 0.01),
    )
    return _decide(votes, w_up, w_down, method="confidence_weighted")


def perplexity_weighted(votes: list[VoterResponse]) -> EnsembleDecision:
    """Weight by 1 / log-perplexity. Lower perplexity → higher weight.

    A voter that emitted high-perplexity (uncertain) text contributes
    little; one that emitted a confident answer contributes a lot.
    """
    def w(v: VoterResponse) -> float:
        # Treat perplexity ≤ 1 as "max confidence". Add small floor so we
        # don't divide by zero and don't let one voter dominate.
        p = max(v.perplexity, 1.001)
        return 1.0 / math.log(p)
    w_up, w_down = _split_weights(votes, w)
    return _decide(votes, w_up, w_down, method="perplexity_weighted")


def rank_vote_borda(votes: list[VoterResponse]) -> EnsembleDecision:
    """Borda-style rank: each voter ranks {up, down}; rank 1 = its
    chosen direction (it 'prefers' up over down or vice versa). With 2
    candidates, this reduces to 1·chosen + 0·other, which is plurality
    in disguise. Kept for symmetry with what `rank voting` typically
    means; provides no lift over plurality on binary choice."""
    return plurality(votes)


def _decide(
    votes: list[VoterResponse], w_up: float, w_down: float, method: str,
) -> EnsembleDecision:
    n_decisive = sum(1 for v in votes if v.direction in ("up", "down"))
    if w_up == 0 and w_down == 0:
        return EnsembleDecision(
            direction=None, confidence=0.0,
            weight_up=0.0, weight_down=0.0,
            n_voters=len(votes), n_decisive=n_decisive,
            method=method, voters=votes,
        )
    if w_up > w_down:
        d = "up"
        conf = w_up / (w_up + w_down)
    elif w_down > w_up:
        d = "down"
        conf = w_down / (w_up + w_down)
    else:
        d = None
        conf = 0.5
    return EnsembleDecision(
        direction=d, confidence=conf,
        weight_up=w_up, weight_down=w_down,
        n_voters=len(votes), n_decisive=n_decisive,
        method=method, voters=votes,
    )


# 2nd-pass deliberation: each voter sees peers' first-round answers in a
# follow-up prompt. For text-providers this means we re-call them with an
# extended prompt that includes peer answers. The aggregator then re-runs
# on second-round responses.

def deliberation_packet(round1: list[VoterResponse]) -> str:
    """Build the text snippet that's appended to round-2 prompts so each
    voter sees what its peers said in round 1."""
    lines = ["\n\nPEER OPINIONS FROM ROUND 1 (you may revise):"]
    for r in round1:
        d = r.direction or "abstain"
        c = r.confidence if r.confidence is not None else 0.5
        lines.append(f"  {r.voter_id}: {d} (confidence {c:.2f})")
    lines.append(
        "Now produce your FINAL answer in the same XML format. You may"
        " keep your previous answer or change it."
    )
    return "\n".join(lines)
