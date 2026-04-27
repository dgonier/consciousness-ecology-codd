"""ForecasterHerbivore — Chronos-backed quantitative herbivore.

Hunts QuantitativeProducer broadcasts (diet=`from_quant_series`). Bypasses
the Channel cross-attention on the producer→herb edge for v1 — payloads
are read directly. Aggregation = "take the most recent broadcast per
ticker" (simple deterministic rule; replaceable with a learned aggregator
later).

Pipeline per call:
  1. Filter candidates by diet + capacity. Pick most recent N by created_tick.
  2. Extract numeric histories from payloads.
  3. For each, call ChronosHost.forecast → ForecastResult.
  4. Render forecast(s) to short text.
  5. Concatenate rendered text → run Qwen forward via existing
     forward_with_prefix path → pooled hidden state.
  6. That hidden state is the herbivore's broadcast (predator-readable shape).

Abstention: fires when no diet-matching candidates available, like other
herbivores. No null gate (no Channel attention here yet).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..forecaster_host import ForecasterHost, ForecastResult
from ..model_host import ModelHost
from ..types import Broadcast
from .base import BaseAgent, new_agent_id


FORECASTER_KINDS = ("forecaster",)


DIETS = {
    "forecaster": ["from_quant_series"],
}


ROLE_PROMPTS = {
    "forecaster": (
        "You are a forecaster herbivore. The text that follows summarizes"
        " probabilistic point forecasts produced by a time-series model"
        " (mean, quantile spread, percent-change) for one or more tickers."
        " Synthesize the trend signal these forecasts collectively support."
        " Format: SYNTHESIS: <text> CONFIDENCE: <num>"
    ),
}

QUERIES = {
    "forecaster": "Now produce the SYNTHESIS and CONFIDENCE.",
}


@dataclass
class ForecasterHerbivore(BaseAgent):
    role: str = "herbivore"
    capacity: int = 4
    role_prefix: Optional[torch.Tensor] = None
    _diet_tags: list[str] = field(default_factory=list)

    @classmethod
    def make(cls, kind: str = "forecaster", capacity: int = 4) -> "ForecasterHerbivore":
        assert kind in FORECASTER_KINDS, kind
        h = cls(id=new_agent_id("herbivore", kind), kind=kind, capacity=capacity)
        h._diet_tags = list(DIETS[kind])
        return h

    def ensure_initialized(self, host: ModelHost, seed_base: int | None = None) -> None:
        if self.role_prefix is None:
            self.role_prefix = host.encode_role_prefix(ROLE_PROMPTS[self.kind])

    @property
    def diet_tags(self) -> list[str]:
        return self._diet_tags

    async def hunt_and_synthesize(
        self,
        candidates: list[Broadcast],
        tick: int,
        host: ModelHost,
        forecaster: ForecasterHost | None = None,
    ) -> tuple[Broadcast, list[Broadcast], list[Broadcast], dict]:
        self.ensure_initialized(host)
        forecaster = forecaster or ForecasterHost.get()

        # Diet pre-filter (cheap re-check; runner already passed filtered set)
        edible = [c for c in candidates if any(t in self._diet_tags for t in c.diet_tags)]
        # Keep most recent N per ticker; cap to capacity.
        edible.sort(key=lambda c: c.created_tick, reverse=True)
        seen_tickers: set[str] = set()
        meal: list[Broadcast] = []
        rejected: list[Broadcast] = []
        for c in edible:
            ticker = str(c.payload.get("ticker", "?"))
            if ticker in seen_tickers:
                rejected.append(c)
                continue
            if len(meal) >= self.capacity:
                rejected.append(c)
                continue
            seen_tickers.add(ticker)
            meal.append(c)

        if not meal:
            null_emb = torch.zeros(host.hidden_size, dtype=torch.float32).tolist()
            br = Broadcast(
                id=str(uuid.uuid4()),
                tier="herbivore_broadcast",
                agent_id=self.id,
                agent_kind=self.kind,
                parent_input_ids=[],
                diet_tags=[f"from_{self.kind}_herbivore"],
                payload={"abstain": True, "reason": "no diet-matching candidates"},
                decoded_text="[ABSTAIN no quant candidates]",
                channel_embedding=null_emb,
                created_tick=tick,
                abstained=True,
            )
            return br, [], rejected, {"abstain": True, "n_meal": 0, "avg_null_prob": 1.0}

        # Run Chronos on each meal item; render forecasts.
        rendered_lines: list[str] = []
        forecasts: list[ForecastResult] = []
        for item in meal:
            history = item.payload.get("history") or []
            horizon = int(item.payload.get("horizon", 12))
            ticker = str(item.payload.get("ticker", "?"))
            window = str(item.payload.get("window", "1m"))
            if len(history) < 4:
                # Too short to forecast; skip.
                continue
            fr = forecaster.forecast(history, horizon=horizon)
            forecasts.append(fr)
            rendered_lines.append(fr.render_text(ticker=ticker, window=window))

        if not rendered_lines:
            null_emb = torch.zeros(host.hidden_size, dtype=torch.float32).tolist()
            br = Broadcast(
                id=str(uuid.uuid4()),
                tier="herbivore_broadcast",
                agent_id=self.id,
                agent_kind=self.kind,
                parent_input_ids=[c.id for c in meal],
                diet_tags=[f"from_{self.kind}_herbivore"],
                payload={"abstain": True, "reason": "histories too short"},
                decoded_text="[ABSTAIN insufficient history]",
                channel_embedding=null_emb,
                created_tick=tick,
                abstained=True,
            )
            return br, meal, rejected, {"abstain": True, "n_meal": len(meal), "avg_null_prob": 1.0}

        # Forecast text → Qwen forward → broadcast hidden state.
        rendered = "\n".join(rendered_lines)
        # Use Qwen's text→hidden as the producer-side adaptor: tokenize the
        # rendered forecast text, get a pooled hidden state. Then run
        # forward_with_prefix to integrate role prefix + this content.
        # Simplest path: just text_to_hidden on (role_prompt_user-style text)
        # then use forward_with_prefix to decode for legibility.
        content_hidden = host.text_to_hidden(rendered, pool=None)
        # content_hidden is [seq, hidden]; use it as the "channel_output" prefix.
        result = host.forward_with_prefix(
            role_prefix=self.role_prefix,
            channel_output=content_hidden,
            query_text=QUERIES[self.kind],
            decode=True,
            pool="mean",
        )

        emb = result.last_hidden.detach().cpu().tolist()
        br = Broadcast(
            id=str(uuid.uuid4()),
            tier="herbivore_broadcast",
            agent_id=self.id,
            agent_kind=self.kind,
            parent_input_ids=[c.id for c in meal],
            diet_tags=[f"from_{self.kind}_herbivore"],
            payload={
                "abstain": False,
                "n_forecasts": len(forecasts),
                "rendered": rendered,
            },
            decoded_text=result.decoded_text,
            channel_embedding=emb,
            created_tick=tick,
        )
        return br, meal, rejected, {"abstain": False, "n_meal": len(meal), "avg_null_prob": 0.0}
