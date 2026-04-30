"""SocialSignal — producer for tweet-shaped social/text streams.

Wavelength: {"tweets"}. Future-proofed for {"reddit", "news_headlines"}
once those wavelengths land in the registry.

Same forward-pass-and-pool pattern as the other text producers in
`producer.py`. Reads tweet-shaped payloads (the tweets adapter aggregates
a day's tweets into payload["body"]) and emits a substrate broadcast
whose channel_embedding is the pooled hidden state from Qwen.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import ClassVar

import torch

from ..model_host import ModelHost
from ..types import Broadcast, RawInput
from .base import new_agent_id
from .producer import Producer


@dataclass
class SocialSignal(Producer):
    role: str = "producer"
    pool: str = "mean"

    WAVELENGTHS: ClassVar[set[str]] = {"tweets"}
    KIND: ClassVar[str] = "social_signal"

    @classmethod
    def make(cls, pool: str = "mean") -> "SocialSignal":
        return cls(
            id=new_agent_id("producer", cls.KIND),
            kind=cls.KIND,
            pool=pool,
        )

    async def produce(
        self,
        inp: RawInput,
        tick: int,
        host: ModelHost | None = None,
    ) -> Broadcast | None:
        if not self.attracts(inp):
            return None
        host = host or ModelHost.get()

        # Render tweets to a short text string for Qwen. The tweet adapter
        # has already aggregated a day's tweets into payload["body"]; we
        # pass it with a minimal preamble.
        body = inp.payload.get("body", "") or ""
        ticker = inp.payload.get("ticker", "")
        text = f"SOCIAL_SIGNAL ticker={ticker}; {body}"
        pooled = host.text_to_hidden(text, pool=self.pool)
        emb = pooled.detach().cpu().tolist()

        return Broadcast(
            id=str(uuid.uuid4()),
            tier="substrate",
            agent_id=self.id,
            agent_kind=self.kind,
            parent_input_ids=[inp.id],
            diet_tags=["is_social_text", "from_social_signal"],
            payload=inp.payload,
            decoded_text=text,
            channel_embedding=emb,
            created_tick=tick,
        )

    def produce_from_stream(
        self,
        attend_output,
        tick: int,
        host: ModelHost | None = None,
        parent_input_ids: list[str] | None = None,
    ) -> Broadcast | None:
        """Override of Producer.produce_from_stream to stamp social diet tags."""
        if attend_output is None:
            return None
        attn = attend_output.per_slot_attention
        if attn is None or float(attn.sum().item()) <= 1e-6:
            return None
        out_seq = attend_output.output
        pooled = out_seq.mean(dim=0) if out_seq.dim() == 2 else out_seq
        emb = pooled.detach().float().cpu().tolist()

        topk = min(3, attn.shape[0])
        top_vals, top_idx = torch.topk(attn, topk)
        debug_summary = (
            f"ENVSTREAM kind={self.KIND} "
            + " ".join(
                f"slot{int(top_idx[i].item())}={float(top_vals[i].item()):.3f}"
                for i in range(topk)
            )
        )

        return Broadcast(
            id=str(uuid.uuid4()),
            tier="substrate",
            agent_id=self.id,
            agent_kind=self.kind,
            parent_input_ids=parent_input_ids or [],
            diet_tags=["is_social_text", "from_social_signal"],
            payload={"from_envstream": True, "kind": self.KIND},
            decoded_text=debug_summary,
            channel_embedding=emb,
            created_tick=tick,
        )
