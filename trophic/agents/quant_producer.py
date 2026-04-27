"""QuantitativeProducer — emits raw numeric history for forecaster herbivores.

Distinct from the Qwen-text producer family. Does NOT run an LM. Does NOT
produce hidden states. Its broadcast carries the numeric payload directly
in the `payload` dict; the channel_embedding is a small zero/marker vector
since the forecaster path bypasses Channel attention on the producer→herb
edge (the herbivore reads payloads directly).

Diet tag: `from_quant_series`. Forecaster herbivore eats this.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from ..types import Broadcast, RawInput
from .base import BaseAgent, new_agent_id


QUANT_PRODUCER_KINDS = ("quote_series",)


def attracts_quote_series(inp: RawInput) -> bool:
    return inp.source == "quote_series"


ATTRACTION = {
    "quote_series": attracts_quote_series,
}


@dataclass
class QuantitativeProducer(BaseAgent):
    role: str = "producer"

    @classmethod
    def make(cls, kind: str) -> "QuantitativeProducer":
        assert kind in QUANT_PRODUCER_KINDS, kind
        return cls(id=new_agent_id("producer", kind), kind=kind)

    def attracts(self, inp: RawInput) -> bool:
        return ATTRACTION[self.kind](inp)

    async def produce(self, inp: RawInput, tick: int, host=None) -> Broadcast | None:
        if not self.attracts(inp):
            return None
        # No LM forward. Just package the payload.
        # channel_embedding is a zero-vector placeholder; downstream forecaster
        # consumes payload directly.
        # Use a small fixed dim so the substrate pool's BLOB column stays small.
        emb = [0.0] * 8
        return Broadcast(
            id=str(uuid.uuid4()),
            tier="substrate",
            agent_id=self.id,
            agent_kind=self.kind,
            parent_input_ids=[inp.id],
            diet_tags=["from_quant_series", "is_quantitative"],
            payload=inp.payload,
            decoded_text=(
                f"QUANT {inp.payload.get('ticker','?')} "
                f"window={inp.payload.get('window','?')} "
                f"history_len={len(inp.payload.get('history', []))}"
            ),
            channel_embedding=emb,
            created_tick=tick,
        )
