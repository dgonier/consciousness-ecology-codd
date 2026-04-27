"""Predator (tier 2) — same architecture as Herbivore, one tier up.

Hunts herbivore broadcasts via Channels (one per herbivore kind),
abstains if null gate fires, otherwise forwards Qwen with [role_prefix ⊕
channel_output ⊕ query] and broadcasts its own pooled hidden state.

The toy v1.5 has one predator kind: ShortHorizonPredictor. Its job is to
emit a brief tradable prediction. The apex judge (Bedrock) scores its
decoded text for "would this prediction have been useful for downstream
apex action."
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from ..channel import Channel, ChannelOutput
from ..model_host import ModelHost
from ..trough_attention import TroughAttention
from ..types import Broadcast
from .base import BaseAgent, new_agent_id


PREDATOR_KINDS = ("short_horizon",)


# Predators eat from any herbivore (their diet at this tier is across both
# herbivore kinds — a real ecology might have specialist predators too).
DIETS = {
    "short_horizon": [
        ("technical",    ["from_technical_herbivore"]),
        ("fundamental",  ["from_fundamental_herbivore"]),
        ("forecaster",   ["from_forecaster_herbivore"]),
        ("interrogator", ["from_interrogator_herbivore"]),
    ],
}


ROLE_PROMPTS = {
    "short_horizon": (
        "You are a short-horizon market predictor (tier-2 consumer)."
        " The vectors that follow encode four kinds of upstream signal:"
        " (1) technical-herbivore synthesis of price/volume microstructure,"
        " (2) fundamental-herbivore synthesis of disclosures and events,"
        " (3) forecaster-herbivore quantitative trend forecast (μ, spread),"
        " (4) interrogator-herbivore math-grounded analysis (exact numerics)."
        " Integrate them into ONE concrete short-horizon prediction that"
        " references both the quantitative trend and the qualitative reasoning."
        " Be specific (ticker, direction, horizon)."
        " Format: PREDICTION: <text> CONFIDENCE: <num>"
    ),
}

QUERIES = {
    "short_horizon": "Now produce the PREDICTION and CONFIDENCE.",
}


def _stable_hash(*parts) -> int:
    """Deterministic 32-bit int from string parts. Used to derive per-Channel seeds."""
    import hashlib
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


@dataclass
class Predator(BaseAgent):
    role: str = "predator"
    capacity: int = 4
    abstain_threshold: float = 0.6
    channels: dict[str, Channel] = field(default_factory=dict)
    role_prefix: Optional[torch.Tensor] = None
    _diet_tags: list[str] = field(default_factory=list)
    # Mission 06 (phase3-A): cross-tier skip connection.
    # `skip_weight` is a learnable scalar gating the residual path from
    # the producer trough into the predator's combined input. It is held
    # as a `nn.Parameter` (wrapped in a tiny holder module so it can move
    # to the host's device/dtype with the rest of the predator's
    # trainable params). At init: skip_weight = 0.1 → sigmoid(0.1) ≈
    # 0.525, so the producer-tier residual contributes ~52% of the
    # combined input from step 0 — small enough that the herbivore-side
    # Channel still dominates (it has explicit gradient on it from the
    # CE loss), large enough that the gradient through the skip is
    # non-zero from step 0.
    _skip_holder: Optional["_SkipWeightHolder"] = None
    skip_weight_init: float = 0.1

    @classmethod
    def make(cls, kind: str, capacity: int = 4, abstain_threshold: float = 0.6) -> "Predator":
        assert kind in PREDATOR_KINDS, kind
        p = cls(id=new_agent_id("predator", kind), kind=kind, capacity=capacity,
                abstain_threshold=abstain_threshold)
        p._diet_tags = [t for _hk, tags in DIETS[kind] for t in tags]
        return p

    @property
    def skip_weight(self) -> torch.nn.Parameter:
        """The learnable scalar α in (0, 1) (post-sigmoid)."""
        if self._skip_holder is None:
            self._skip_holder = _SkipWeightHolder(self.skip_weight_init)
        return self._skip_holder.skip_weight

    def alpha(self) -> torch.Tensor:
        """Sigmoid-clipped skip weight, used to combine main and skip paths."""
        return torch.sigmoid(self.skip_weight)

    def ensure_initialized(self, host: ModelHost, seed_base: int | None = None) -> None:
        if self.role_prefix is None:
            self.role_prefix = host.encode_role_prefix(ROLE_PROMPTS[self.kind])
        if not self.channels:
            tn = host.input_embedding_norm()
            for herb_kind, _tags in DIETS[self.kind]:
                ch_seed = None
                if seed_base is not None:
                    ch_seed = _stable_hash(seed_base, "pred", self.kind, herb_kind)
                self.channels[herb_kind] = Channel(
                    hidden_size=host.hidden_size,
                    n_heads=8,
                    out_seq_len=8,
                    capacity=self.capacity,
                    target_norm=tn,
                    seed=ch_seed,
                )
        # Materialise the skip holder so its parameter is included by
        # any `parameters()` enumeration the trainer does at attach time.
        if self._skip_holder is None:
            self._skip_holder = _SkipWeightHolder(self.skip_weight_init)

    @property
    def diet_tags(self) -> list[str]:
        return self._diet_tags

    async def hunt_and_predict(
        self,
        candidates: list[Broadcast],
        tick: int,
        host: ModelHost,
        producer_trough: Optional[TroughAttention] = None,
        tau: float = 1.0,
    ) -> tuple[Broadcast, list[Broadcast], list[Broadcast], dict]:
        """Mission 03 (phase2-C): per-herbivore-kind branching removed.
        All edible candidates (already filtered by `query_pool` against
        `self.diet_tags`) flow through a single Channel call;
        multi-head attention does the routing rather than agent_kind
        string matching. Per-kind Channel objects survive temporarily
        for SFT/IPO/GRPO compat; runtime path no longer fans out.

        Mission 06 (phase3-A): cross-tier skip connection. If a
        `producer_trough` is supplied, we additionally call
        `producer_trough.attend(hunter_state, tau=tau)` to read producer-
        tier broadcasts directly, and combine the two paths via the
        learnable scalar α = sigmoid(skip_weight):

            combined = (1 - α) * channel_seq + α * skip_seq

        Both paths produce shape `[out_seq_len, hidden]`, so the
        combined sequence has the same shape and downstream
        `forward_with_prefix` is unaffected. `tau` is also forwarded
        into the skip-path attend so the τ schedule (Mission 06)
        anneals both paths in lockstep.
        """
        self.ensure_initialized(host)

        hunter_state = self.role_prefix.mean(dim=0)

        # Single Channel call — multi-head attention routes; we pick the
        # first per-kind Channel as the runtime channel.
        primary_kind = next(iter(self.channels))
        ch = self.channels[primary_kind]
        ch_device = ch.W_Q.weight.device
        ch_dtype = ch.W_Q.weight.dtype

        all_claimed: list[Broadcast] = []
        all_rejected: list[Broadcast] = []
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
        for i in out.selected_indices:
            all_claimed.append(candidates[i])
        for i in out.rejected_indices:
            all_rejected.append(candidates[i])

        avg_null = out.null_prob if candidates else 1.0
        null_probs = [out.null_prob]
        abstain = avg_null >= self.abstain_threshold or not candidates

        if abstain:
            null_emb = torch.zeros(host.hidden_size, dtype=torch.float32).tolist()
            br = Broadcast(
                id=str(uuid.uuid4()),
                tier="predator_broadcast",
                agent_id=self.id,
                agent_kind=self.kind,
                parent_input_ids=[c.id for c in all_claimed],
                diet_tags=[f"from_{self.kind}_predator"],
                payload={"abstain": True, "avg_null_prob": avg_null},
                decoded_text=f"[ABSTAIN avg_null={avg_null:.3f}]",
                channel_embedding=null_emb,
                created_tick=tick,
                abstained=True,
            )
            return br, all_claimed, all_rejected, {
                "abstain": True, "avg_null_prob": avg_null, "null_probs": null_probs,
            }

        channel_seq = out.output  # [out_seq_len, hidden]

        # Mission 06: cross-tier skip path. If a producer trough is
        # available AND has at least one alive slot, run a parallel
        # attend on it from the same hunter state and combine via α.
        # When the producer trough has no alive slots (lazy creation
        # may not have happened yet), the attend would be all-null;
        # we still compute it because the gradient must flow into
        # `skip_weight` from the very first step.
        skip_seq: Optional[torch.Tensor] = None
        if producer_trough is not None:
            try:
                skip_out = producer_trough.attend(hs, tau=tau)
                skip_seq = skip_out.output.to(device=ch_device, dtype=ch_dtype)
            except Exception:
                skip_seq = None

        if skip_seq is not None and skip_seq.shape == channel_seq.shape:
            alpha = torch.sigmoid(self.skip_weight.to(
                device=ch_device, dtype=ch_dtype
            ))
            channel_seq = (1.0 - alpha) * channel_seq + alpha * skip_seq

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
            tier="predator_broadcast",
            agent_id=self.id,
            agent_kind=self.kind,
            parent_input_ids=[c.id for c in all_claimed],
            diet_tags=[f"from_{self.kind}_predator"],
            payload={"avg_null_prob": avg_null, "null_probs": null_probs},
            decoded_text=result.decoded_text,
            channel_embedding=emb,
            created_tick=tick,
        )
        return br, all_claimed, all_rejected, {
            "abstain": False, "avg_null_prob": avg_null, "null_probs": null_probs,
        }


class _SkipWeightHolder(nn.Module):
    """Tiny `nn.Module` that owns the learnable skip-connection scalar.

    Wrapping the scalar in a Module (rather than a free `nn.Parameter`
    on the dataclass) gives us:

      - A standard `parameters()` enumeration the optimizer can pick up.
      - Easy `.to(device, dtype)` movement alongside the Channels.
      - State dict round-trip, so checkpoints can save/restore α.
    """

    def __init__(self, init_value: float = 0.1):
        super().__init__()
        self.skip_weight = nn.Parameter(torch.tensor(float(init_value)))
