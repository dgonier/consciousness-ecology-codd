"""Producers — forward-pass-and-pool only.

A producer takes a RawInput, renders it to a short text string, runs it
through Qwen, pools the last hidden state, and broadcasts. There is no
system prompt and no Channel — producers are the bottom of the trophic
stack and have no upstream tier to hunt.

Differentiation between producer kinds:
  - attraction filter (which raw inputs they ingest)
  - the rendering template that turns structured payload → tokenizable text
  - the diet_tags they stamp on their broadcast (so hunters can filter)
  - the pooling strategy (mean / last)

That's it. No prompt engineering. Information is in the hidden state.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import torch

from ..model_host import ModelHost
from ..types import Broadcast, RawInput
from .base import BaseAgent, new_agent_id


PRODUCER_KINDS = ("tickdelta", "disclosure", "anomaly")


# ---------- attraction filters ----------

def attracts_tickdelta(inp: RawInput) -> bool:
    return inp.source in {"ohlcv", "trades", "book"}


def attracts_disclosure(inp: RawInput) -> bool:
    return inp.source in {"filing", "press"}


def attracts_anomaly(inp: RawInput) -> bool:
    return inp.source in {"ohlcv", "options", "halt"}


ATTRACTION = {
    "tickdelta": attracts_tickdelta,
    "disclosure": attracts_disclosure,
    "anomaly": attracts_anomaly,
}


# ---------- diet tags emitted ----------

def diet_tags_for(kind: str, src: str) -> list[str]:
    if kind == "tickdelta":
        return ["is_price_event", "from_tickdelta"]
    if kind == "disclosure":
        return ["is_disclosure_event", "from_disclosure"]
    if kind == "anomaly":
        tags = ["is_anomaly", "from_anomaly"]
        if src == "options":
            tags.append("is_options_flow")
        return tags
    return []


# ---------- rendering ----------

RENDER_PREAMBLE = {
    "tickdelta": "MARKET MICRO ",
    "disclosure": "CORP DISCLOSURE ",
    "anomaly": "ANOMALY OBS ",
}


def _render(kind: str, inp: RawInput) -> str:
    bits = [f"src={inp.source}"]
    for k, v in inp.payload.items():
        bits.append(f"{k}={v}")
    return RENDER_PREAMBLE.get(kind, "") + "; ".join(bits)


# ---------- producer ----------

@dataclass
class Producer(BaseAgent):
    role: str = "producer"
    pool: str = "mean"  # mean | last

    @classmethod
    def make(cls, kind: str, pool: str = "mean") -> "Producer":
        assert kind in PRODUCER_KINDS, kind
        return cls(id=new_agent_id("producer", kind), kind=kind, pool=pool)

    def attracts(self, inp: RawInput) -> bool:
        return ATTRACTION[self.kind](inp)

    async def produce(
        self,
        inp: RawInput,
        tick: int,
        host: ModelHost | None = None,
    ) -> Broadcast | None:
        if not self.attracts(inp):
            return None
        host = host or ModelHost.get()

        text = _render(self.kind, inp)
        # Forward through Qwen and pool. This IS the broadcast; tokens are
        # only the input to the model, not the output to downstream agents.
        pooled = host.text_to_hidden(text, pool=self.pool)
        emb = pooled.detach().cpu().tolist()

        return Broadcast(
            id=str(uuid.uuid4()),
            tier="substrate",
            agent_id=self.id,
            agent_kind=self.kind,
            parent_input_ids=[inp.id],
            diet_tags=diet_tags_for(self.kind, inp.source),
            payload=inp.payload,
            decoded_text=text,  # the rendered input — useful for debug only
            channel_embedding=emb,
            created_tick=tick,
        )
