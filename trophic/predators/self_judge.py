"""SelfJudge — Qwen-4B with a judge prompt.

Same interface as ApexJudge. Used during early SFT/GRPO when outputs are
noisy enough that even the small model can rank them. The runner switches
to ApexJudge (Bedrock) when self-consistency variance grows beyond a
threshold.
"""
from __future__ import annotations

import json
import re
import statistics
import uuid
from dataclasses import dataclass, field

from ..config import DEFAULT_CONFIG, ModelConfig
from ..model_host import ModelHost
from ..types import Broadcast, PredatorJudgment


SYSTEM_PROMPT = (
    "You are a judge. Do NOT think out loud. Do NOT use <think> tags. Output"
    " ONLY a JSON object on the first line, no preamble, no reasoning.\n\n"
    "A tier-2 predator agent has emitted a short-horizon market prediction"
    " conditioned on upstream herbivore analyses about a specific ticker."
    " You will be shown both the prediction AND a short summary of what the"
    " upstream context was about. Score 0.0-1.0:\n"
    "  - 0.0: pure noise (no ticker, no direction, malformed)\n"
    "  - 0.3: parseable but not specific, OR mentions wrong ticker\n"
    "  - 0.6: specific (correct ticker + direction + horizon) and grounded\n"
    "  - 0.8: specific, correct ticker, direction matches upstream context\n"
    "  - 1.0: excellent, falsifiable, references upstream context appropriately\n\n"
    'OUTPUT EXACTLY: {"score": <float>, "rationale": "<one short phrase>"}\n'
    "Nothing else. No explanation. JSON only."
)


@dataclass
class ConsistencyResult:
    scores: list[float]
    variance: float
    rationales: list[str]


_JSON_RE = re.compile(r"\{[^{}]*?\"score\"\s*:\s*([0-9]*\.?[0-9]+)[^{}]*?\}", re.DOTALL)
_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9]*\.?[0-9]+)')


def _parse(raw: str, target_id: str, tick: int) -> PredatorJudgment:
    """Parse a judge response.

    Robust to thinking-model output: strip <think>...</think>, then try
    to find a JSON object with "score". If JSON parse fails but a numeric
    score is present anywhere, fall back to that.
    """
    # Strip out any <think>...</think> blocks (Qwen-3 reasoning models).
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = raw

    # Try strict JSON match first.
    m = _JSON_RE.search(cleaned)
    if m:
        try:
            obj = json.loads(m.group(0))
            score = max(0.0, min(1.0, float(obj.get("score", 0.5))))
            rationale = str(obj.get("rationale", "")).strip() or cleaned[:160]
            return PredatorJudgment(
                id=str(uuid.uuid4()),
                synthesis_id=target_id,
                score=score,
                rationale=rationale,
                created_tick=tick,
            )
        except (ValueError, json.JSONDecodeError):
            pass

    # Fallback: pluck the first "score": <num> from anywhere in the text.
    m2 = _SCORE_RE.search(cleaned)
    if m2:
        try:
            score = max(0.0, min(1.0, float(m2.group(1))))
            return PredatorJudgment(
                id=str(uuid.uuid4()),
                synthesis_id=target_id,
                score=score,
                rationale=f"[fuzzy-parsed] {cleaned[:160]}",
                created_tick=tick,
            )
        except ValueError:
            pass

    return PredatorJudgment(
        id=str(uuid.uuid4()),
        synthesis_id=target_id,
        score=0.0,
        rationale=f"<unparseable> {cleaned[:160]}",
        created_tick=tick,
    )


@dataclass
class SelfJudge:
    host: ModelHost = field(default=None)  # type: ignore[assignment]
    name: str = "self_judge"

    def __post_init__(self):
        if self.host is None:
            self.host = ModelHost.get()

    async def judge(
        self,
        broadcast: Broadcast,
        tick: int,
        temperature: float = 0.0,
        context: str | None = None,
    ) -> PredatorJudgment:
        """Score the broadcast. If `context` is provided, the judge sees a
        short summary of upstream content (e.g. the scenario's expected
        ticker + direction) so it can downgrade wrong-ticker outputs.
        """
        if broadcast.abstained:
            return PredatorJudgment(
                id=str(uuid.uuid4()),
                synthesis_id=broadcast.id,
                score=0.0,
                rationale="[self-judge] abstained — no prediction emitted",
                created_tick=tick,
            )
        ctx_part = f"Upstream context: {context}\n\n" if context else ""
        user = (
            f"{ctx_part}"
            f"Predator output: {broadcast.decoded_text or '<none>'}\n\n"
            "Return your JSON judgment now."
        )
        raw = self.host.generate_chat(
            system=SYSTEM_PROMPT, user=user, max_new_tokens=512,
            temperature=temperature,
        )
        return _parse(raw, broadcast.id, tick)

    async def judge_batch(
        self, broadcasts: list[Broadcast], tick: int
    ) -> list[PredatorJudgment]:
        return [await self.judge(b, tick) for b in broadcasts]

    async def judge_with_consistency(
        self, broadcast: Broadcast, tick: int, k: int = 3,
        temperature: float = 0.3,
    ) -> ConsistencyResult:
        """Run k independent judgments at non-zero temperature to measure
        judge stability.

        Variance > threshold = the judge can't reliably distinguish the
        broadcast at this difficulty level → escalate to a stronger judge.
        """
        scores: list[float] = []
        rats: list[str] = []
        for _ in range(k):
            j = await self.judge(broadcast, tick, temperature=temperature)
            scores.append(j.score)
            rats.append(j.rationale)
        var = statistics.pvariance(scores) if len(scores) > 1 else 0.0
        return ConsistencyResult(scores=scores, variance=var, rationales=rats)
