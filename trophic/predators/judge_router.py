"""JudgeRouter — wraps SelfJudge + ApexJudge, escalates on demand.

Implements the cheap-judge-first principle: while the policy outputs
nothing parseable, score with SelfJudge (Qwen-4B in-process). When
SelfJudge's self-consistency variance grows beyond a threshold, the
cheap judge can no longer reliably rank the policy's outputs — escalate
to ApexJudge (Bedrock).

The router exposes:
  - judge(broadcast, tick) → PredatorJudgment from current judge
  - judge_batch(broadcasts, tick) → list
  - probe(broadcast, tick, k=3) → ConsistencyResult, also feeds the
    rolling window that drives the escalation decision
  - current_judge_name → "self" | "apex"
"""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field

from ..types import Broadcast, PredatorJudgment
from .apex_judge import ApexJudge
from .self_judge import ConsistencyResult, SelfJudge


@dataclass
class JudgeRouter:
    self_judge: SelfJudge
    apex_judge: ApexJudge | None = None
    # Rolling window of self-consistency variances (len == window_size).
    # When mean(window) > escalation_threshold for window_size measurements,
    # we flip to apex.
    window_size: int = 5
    escalation_threshold: float = 0.05
    _variance_window: deque = field(default_factory=lambda: deque(maxlen=5))
    _mode: str = "self"  # "self" | "apex"

    def __post_init__(self):
        # Make the deque size match the configured window.
        self._variance_window = deque(maxlen=self.window_size)

    @property
    def current_judge_name(self) -> str:
        return self._mode

    async def judge(
        self, broadcast: Broadcast, tick: int, context: str | None = None
    ) -> PredatorJudgment:
        if self._mode == "apex" and self.apex_judge is not None:
            # ApexJudge doesn't yet take context; pass through plain.
            return await self.apex_judge.judge(broadcast, tick)
        return await self.self_judge.judge(broadcast, tick, context=context)

    async def judge_batch(
        self, broadcasts: list[Broadcast], tick: int, context: str | None = None
    ) -> list[PredatorJudgment]:
        return [await self.judge(b, tick, context=context) for b in broadcasts]

    async def probe(
        self, broadcast: Broadcast, tick: int, k: int = 3
    ) -> ConsistencyResult:
        """Self-consistency probe + record into the rolling window.

        Always uses SelfJudge (no point probing apex — it's the answer to
        the question "is the cheap judge enough?").
        """
        result = await self.self_judge.judge_with_consistency(
            broadcast, tick, k=k
        )
        self._variance_window.append(result.variance)
        if (
            self._mode == "self"
            and len(self._variance_window) == self.window_size
            and self.apex_judge is not None
        ):
            window_mean = statistics.fmean(self._variance_window)
            if window_mean > self.escalation_threshold:
                self._mode = "apex"
        return result

    def force_mode(self, mode: str) -> None:
        """Manual override for tests/debug."""
        assert mode in ("self", "apex")
        self._mode = mode
