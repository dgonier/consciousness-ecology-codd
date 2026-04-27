"""Apex judge — Bedrock Claude evaluates predator broadcasts.

This is the v1.5 optimization signal. The full apex tier isn't built yet;
we use a frontier model (via Bedrock) as a stand-in oracle that scores
predator predictions for "would this be useful to a downstream apex tier
making real recommendations." The score cascades back through energy and
reputation deltas to all upstream agents.

The bedrock dependency is temporary scaffolding (see architecture.md
"bootstrap-then-distill" plan).
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import BedrockConfig, DEFAULT_CONFIG
from ..types import Broadcast, PredatorJudgment


SYSTEM_PROMPT = (
    "You are the apex evaluator in a multi-tier fintech analysis system."
    " A tier-2 predator agent has emitted a short-horizon market prediction"
    " based on syntheses from tier-1 herbivores, which themselves consumed"
    " raw market substrate from tier-0 producers.\n\n"
    "Your job: score the prediction for usefulness to a downstream"
    " recommendation tier (apex). Specifically:\n"
    "  (a) Is it specific (named ticker, direction, time horizon)?\n"
    "  (b) Is it falsifiable (could you tell after the fact whether it was right)?\n"
    "  (c) Does it surface a non-obvious signal worth acting on, vs."
    " restating priors?\n\n"
    "Score 0.0..1.0 (0 = useless, 1 = excellent). The system is in early"
    " training; expect outputs to be noisy or malformed — score them on"
    " whatever signal you can extract, including 0 for total noise.\n\n"
    "Output ONLY a single JSON object on one line:\n"
    '  {"score": float, "rationale": "short", "predicted_action": "what apex would do, or null"}\n'
    "Do not include any other text."
)


def _build_user(broadcast: Broadcast) -> str:
    abstained_note = " [ABSTAINED — no prediction emitted]" if broadcast.abstained else ""
    return (
        f"Predator broadcast (id={broadcast.id[:8]}, tier={broadcast.tier},"
        f" tick={broadcast.created_tick}){abstained_note}:\n\n"
        f"{broadcast.decoded_text or '<no decoded text>'}\n\n"
        f"Parent broadcasts consumed: "
        f"{[p[:8] for p in broadcast.parent_input_ids]}\n\n"
        "Return your JSON judgment now."
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(raw: str, target_id: str, tick: int) -> PredatorJudgment:
    m = _JSON_RE.search(raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            score = float(obj.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            rationale = str(obj.get("rationale", "")).strip()
            pa = obj.get("predicted_action")
            return PredatorJudgment(
                id=str(uuid.uuid4()),
                synthesis_id=target_id,
                score=score,
                rationale=rationale,
                predicted_action=str(pa) if pa else None,
                created_tick=tick,
            )
        except (ValueError, json.JSONDecodeError):
            pass
    return PredatorJudgment(
        id=str(uuid.uuid4()),
        synthesis_id=target_id,
        score=0.5,
        rationale=f"<unparseable> {raw[:200]}",
        predicted_action=None,
        created_tick=tick,
    )


@dataclass
class ApexJudge:
    cfg: BedrockConfig

    @classmethod
    def from_config(cls, cfg: BedrockConfig | None = None) -> "ApexJudge":
        return cls(cfg=cfg or DEFAULT_CONFIG.bedrock)

    async def judge(self, broadcast: Broadcast, tick: int) -> PredatorJudgment:
        if self.cfg.mock or broadcast.abstained:
            return self._mock_judge(broadcast, tick)
        raw = await self._invoke(broadcast)
        return _parse(raw, broadcast.id, tick)

    async def judge_batch(self, broadcasts: list[Broadcast], tick: int) -> list[PredatorJudgment]:
        return [await self.judge(b, tick) for b in broadcasts]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=4))
    async def _invoke(self, broadcast: Broadcast) -> str:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._invoke_sync, broadcast)

    def _invoke_sync(self, broadcast: Broadcast) -> str:
        import boto3

        client = boto3.client("bedrock-runtime", region_name=self.cfg.region)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": _build_user(broadcast)}],
        }
        resp = client.invoke_model(
            modelId=self.cfg.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(resp["body"].read())
        for p in payload.get("content", []):
            if p.get("type") == "text":
                return p.get("text", "")
        return ""

    @staticmethod
    def _mock_judge(broadcast: Broadcast, tick: int) -> PredatorJudgment:
        if broadcast.abstained:
            return PredatorJudgment(
                id=str(uuid.uuid4()),
                synthesis_id=broadcast.id,
                score=0.0,
                rationale="[mock] predator abstained — no prediction to judge",
                created_tick=tick,
            )
        # Deterministic mock: score scales with how many parents the predator
        # consumed (richer meals → higher score in mock).
        n_parents = len(broadcast.parent_input_ids)
        score = max(0.0, min(1.0, 0.2 + 0.15 * n_parents))
        return PredatorJudgment(
            id=str(uuid.uuid4()),
            synthesis_id=broadcast.id,
            score=score,
            rationale=f"[mock] consumed {n_parents} herbivore broadcasts",
            created_tick=tick,
        )
