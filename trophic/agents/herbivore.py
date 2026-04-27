"""Herbivores — hunt producer broadcasts via Channel + abstain.

A herbivore:
  1. Pulls diet-filtered candidate broadcasts from the substrate pool.
  2. Computes its own hunter state (its frozen role-prefix pooled, plus
     any meal-so-far context — empty in v1.5).
  3. Runs Channel(hunter_state, prey_states) → (channel_output, null_prob,
     selected indices).
  4. If null_prob > abstain_threshold → STOP. No broadcast emitted.
  5. Else: forward Qwen with [role_prefix ⊕ channel_output ⊕ query_text],
     pool last hidden state, that's the herbivore's broadcast.
  6. Decode tokens for legibility.

Each herbivore kind has one Channel per producer kind (3 Channels per
herbivore). At hunt time we run each Channel against its kind's
candidates, concatenate the outputs into a single channel-sequence, then
take top-K across all of them by attention weight to enforce capacity.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..channel import Channel, ChannelOutput
from ..model_host import ModelHost
from ..types import Broadcast
from .base import BaseAgent, new_agent_id


HERBIVORE_KINDS = ("technical", "fundamental")


DIETS = {
    # Per-kind: list of (producer_kind, diet_tags_to_match) so the herbivore
    # knows which Channel to apply to which subset of candidates.
    "technical": [
        ("tickdelta", ["from_tickdelta"]),
        ("anomaly",   ["from_anomaly"]),
    ],
    "fundamental": [
        ("disclosure", ["from_disclosure"]),
        ("anomaly",    ["from_anomaly"]),
    ],
}


ROLE_PROMPTS = {
    "technical": (
        "You are a technical-analysis primary consumer. The vectors that"
        " follow encode market-microstructure substrate. Synthesize a"
        " short-horizon signal from them and report a confidence in [0,1]."
        " Format: SYNTHESIS: <text> CONFIDENCE: <num>"
    ),
    "fundamental": (
        "You are a fundamental-analysis primary consumer. The vectors that"
        " follow encode disclosure / event substrate. Synthesize an"
        " event-driven thesis from them and report a confidence in [0,1]."
        " Format: SYNTHESIS: <text> CONFIDENCE: <num>"
    ),
}

QUERIES = {
    "technical": "Now produce the SYNTHESIS and CONFIDENCE.",
    "fundamental": "Now produce the SYNTHESIS and CONFIDENCE.",
}


def _stable_hash(*parts) -> int:
    """Deterministic 32-bit int from string parts. Used to derive per-Channel seeds."""
    import hashlib
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


@dataclass
class Herbivore(BaseAgent):
    role: str = "herbivore"
    capacity: int = 6
    abstain_threshold: float = 0.6  # null_prob above this → STOP
    # Per-producer-kind channels. Keyed by producer kind.
    channels: dict[str, Channel] = field(default_factory=dict)
    role_prefix: Optional[torch.Tensor] = None  # cached frozen prefix
    _diet_tags: list[str] = field(default_factory=list)

    @classmethod
    def make(cls, kind: str, capacity: int = 6, abstain_threshold: float = 0.6) -> "Herbivore":
        assert kind in HERBIVORE_KINDS, kind
        h = cls(id=new_agent_id("herbivore", kind), kind=kind, capacity=capacity,
                abstain_threshold=abstain_threshold)
        h._diet_tags = [t for _pk, tags in DIETS[kind] for t in tags]
        return h

    def ensure_initialized(self, host: ModelHost, seed_base: int | None = None) -> None:
        """Lazy init of role prefix + per-edge Channels (needs host's hidden_size).

        If `seed_base` is provided, each Channel gets a deterministic seed
        derived from (seed_base, herbivore_kind, producer_kind) so re-runs
        with the same seed_base produce bit-identical Channel inits. This is
        the reproducibility hook for multi-seed comparison runs.
        """
        if self.role_prefix is None:
            self.role_prefix = host.encode_role_prefix(ROLE_PROMPTS[self.kind])
        if not self.channels:
            tn = host.input_embedding_norm()
            for producer_kind, _tags in DIETS[self.kind]:
                ch_seed = None
                if seed_base is not None:
                    ch_seed = _stable_hash(seed_base, "herb", self.kind, producer_kind)
                self.channels[producer_kind] = Channel(
                    hidden_size=host.hidden_size,
                    n_heads=8,
                    out_seq_len=8,
                    capacity=self.capacity,
                    target_norm=tn,
                    seed=ch_seed,
                )

    @property
    def diet_tags(self) -> list[str]:
        return self._diet_tags

    async def hunt_and_synthesize(
        self,
        candidates: list[Broadcast],
        tick: int,
        host: ModelHost,
    ) -> tuple[Broadcast, list[Broadcast], list[Broadcast], dict]:
        """Returns (broadcast, claimed, rejected, stats).

        Mission 03 (phase2-C): per-source-kind branching is gone. All
        edible candidates (already filtered by `query_pool` against
        `self.diet_tags`) feed into a single Channel call. The Channel's
        multi-head attention is responsible for routing — heads
        specialize in different upstream sources during training rather
        than us hardcoding the partition. The per-kind Channel objects
        survive (one is selected as the "primary" routing channel) so
        SFT/IPO/GRPO trainers that train each Channel separately keep
        working through the migration; runtime no longer fans out.
        """
        self.ensure_initialized(host)

        # Hunter state = pooled role prefix. (Future: also include meal-so-far.)
        hunter_state = self.role_prefix.mean(dim=0)  # [hidden]

        # Single Channel call — multi-head attention does the routing.
        # Pick the first per-kind Channel as the runtime channel; with
        # SFT updating all Channel weights similarly, the choice is
        # interchangeable for the agent's runtime path. (When phase2-D's
        # shared trough lands, this becomes a single TroughAttention
        # call.)
        primary_kind = next(iter(self.channels))
        ch = self.channels[primary_kind]
        ch_device = ch.W_Q.weight.device
        ch_dtype = ch.W_Q.weight.dtype

        all_claimed: list[Broadcast] = []
        all_rejected: list[Broadcast] = []
        null_prob = 1.0
        if candidates:
            prey_t = torch.tensor(
                [p.channel_embedding for p in candidates],
                dtype=ch_dtype, device=ch_device,
            )
        else:
            prey_t = torch.zeros(
                0, host.hidden_size, dtype=ch_dtype, device=ch_device,
            )
        hs = hunter_state.to(device=ch_device, dtype=ch_dtype)
        out: ChannelOutput = ch(hs, prey_t)
        null_prob = out.null_prob
        for i in out.selected_indices:
            all_claimed.append(candidates[i])
        for i in out.rejected_indices:
            all_rejected.append(candidates[i])

        # Single null gate now (no fanned-out Channels to aggregate over).
        avg_null = null_prob if candidates else 1.0
        null_probs = [null_prob]
        abstain = avg_null >= self.abstain_threshold or not candidates

        if abstain:
            # No real broadcast. Emit a null broadcast for bookkeeping.
            null_emb = torch.zeros(host.hidden_size, dtype=torch.float32).tolist()
            br = Broadcast(
                id=str(uuid.uuid4()),
                tier="herbivore_broadcast",
                agent_id=self.id,
                agent_kind=self.kind,
                parent_input_ids=[c.id for c in all_claimed],
                diet_tags=[f"from_{self.kind}_herbivore"],
                payload={"abstain": True, "avg_null_prob": avg_null},
                decoded_text=f"[ABSTAIN avg_null={avg_null:.3f}]",
                channel_embedding=null_emb,
                created_tick=tick,
                abstained=True,
            )
            return br, all_claimed, all_rejected, {
                "abstain": True,
                "avg_null_prob": avg_null,
                "null_probs": null_probs,
            }

        # Single channel sequence directly from the (only) Channel call.
        channel_seq = out.output  # [out_seq_len, hidden]

        # Forward Qwen on [role_prefix ⊕ channel_seq ⊕ query_text].
        result = host.forward_with_prefix(
            role_prefix=self.role_prefix,
            channel_output=channel_seq,
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
            parent_input_ids=[c.id for c in all_claimed],
            diet_tags=[f"from_{self.kind}_herbivore"],
            payload={"avg_null_prob": avg_null, "null_probs": null_probs},
            decoded_text=result.decoded_text,
            channel_embedding=emb,
            created_tick=tick,
        )
        return br, all_claimed, all_rejected, {
            "abstain": False,
            "avg_null_prob": avg_null,
            "null_probs": null_probs,
        }
