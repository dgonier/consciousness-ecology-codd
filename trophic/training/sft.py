"""SFT — teacher-forcing CE on target tokens, only Channels learn.

For each scenario:
  1. Run producers on the scenario's raw inputs → cached producer broadcasts.
  2. For each herbivore kind, group producer broadcasts by source kind, run
     each Channel against that group, concatenate channel outputs.
  3. Compute CE loss against the herbivore's target text using
     ModelHost.teacher_forcing_loss (gradients flow through Channels).
  4. Same for the predator: run herbivore Channels in eval mode against the
     producer broadcasts to get herbivore *channel-driven* outputs (live,
     so the predator's gradient also flows back to herbivore Channels for
     end-to-end coupling), then run predator Channel + CE on predator target.
  5. Sum losses, backward, step optimizer.

Producer broadcasts are computed once and cached for the run — they don't
change because producers aren't being trained.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import torch

import os

from ..agents.herbivore import DIETS as HERB_DIETS, Herbivore
from ..agents.predator import DIETS as PRED_DIETS, Predator
from ..agents.producer import Producer
from ..model_host import ModelHost
from ..trough_attention import TroughAttention
from ..types import Broadcast


def _tier_transport() -> str:
    """Phase 1 of #8 fix: env-var gate for which inter-tier transport
    mechanism is active. "channel" (default) = legacy per-Channel
    point-to-point; "trough" = shared K/V trough at each tier with E_in
    projection + diet-mask attend + gated residual output.
    """
    return os.environ.get("TROPHIC_TIER_TRANSPORT", "channel").lower()
from .channel_trainer import ChannelTrainer
from .scenarios import Scenario
from .tau_schedule import cosine_tau


@dataclass
class SFTConfig:
    lr: float = 5e-4
    grad_clip: float = 1.0
    steps: int = 200
    log_every: int = 10
    eval_every: int = 25
    seed: int = 7
    # Multiplier on CE loss at content tokens (ticker, direction, %, σ,
    # horizon). 1.0 → flat, 5.0 → 5× weight on grounding tokens.
    content_token_weight: float = 1.0
    # Mission 06 (phase3-A): cross-attention softmax temperature schedule.
    # Cosine anneal from `tau_start` (warm: explore) to `tau_end` (cool:
    # exploit). Drives the τ argument passed to the trough's attend()
    # call on the predator's skip-connection path each step.
    tau_start: float = 2.0
    tau_end: float = 0.5


@dataclass
class SFTRunner:
    cfg: SFTConfig
    host: ModelHost
    producers: list[Producer]
    herbivores: list[Herbivore]
    predator: Predator
    train: list[Scenario]
    eval_: list[Scenario]
    trainer: ChannelTrainer = field(default=None)  # type: ignore[assignment]
    _producer_cache: dict[str, list[Broadcast]] = field(default_factory=dict)
    # Mission 06 (phase3-A): per-scenario producer trough used by the
    # predator's skip-connection path. Populated lazily on the first
    # _pred_loss call for each scenario from the cached producer
    # broadcasts. Kept on the SFT runner (not the predator) because
    # scenarios share producers but the trough's V_store is content-
    # addressed per scenario.
    _producer_trough: TroughAttention | None = None
    # Phase 1 of #8 fix: herb-trough. New tier substrate that holds
    # herbivore broadcasts; predator attends it per-diet in trough mode.
    # Built lazily on first _pred_loss call. Only used when
    # TROPHIC_TIER_TRANSPORT=trough (else None and predator uses per-Channel
    # point-to-point as before — backward compat with seed1..seed20).
    _herb_trough: TroughAttention | None = None
    # τ used for the most recent step — exposed for diagnostic logging.
    _last_tau: float = 1.0

    def __post_init__(self):
        self.host.freeze_base_model()
        # Materialize Channels + role prefixes with reproducible seeds.
        seed_base = self.cfg.seed
        for h in self.herbivores:
            h.ensure_initialized(self.host, seed_base=seed_base)
        self.predator.ensure_initialized(self.host, seed_base=seed_base)
        if self.trainer is None:
            self.trainer = ChannelTrainer(lr=self.cfg.lr, grad_clip=self.cfg.grad_clip)
        n_params = self.trainer.attach(self.herbivores, [self.predator])
        # Move Channels onto the model device, with model dtype, for fast matmul.
        for h in self.herbivores:
            for ch in h.channels.values():
                ch.to(device=self.host.device, dtype=self.host.dtype)
        for ch in self.predator.channels.values():
            ch.to(device=self.host.device, dtype=self.host.dtype)
        # Mission 06: move skip-weight onto host device too.
        if self.predator._skip_holder is not None:
            self.predator._skip_holder.to(
                device=self.host.device, dtype=self.host.dtype
            )
        # Re-attach (parameters() now refers to the moved tensors).
        self.trainer.attach(self.herbivores, [self.predator])
        print(f"[sft] Channel parameter tensors: {n_params}")
        print(
            f"[sft] tau schedule: cosine {self.cfg.tau_start} → {self.cfg.tau_end}"
            f" over {self.cfg.steps} steps"
        )

    # ---------- #8 phase-1 trough mode helpers ----------

    def _trough_extras(self) -> list:
        """Modules whose parameters should join the optimizer in trough mode."""
        extras = []
        if self._producer_trough is not None:
            extras.append(self._producer_trough)
        if self._herb_trough is not None:
            extras.append(self._herb_trough)
        return extras

    def _ensure_herb_trough(self, herb_broadcasts: list[Broadcast]) -> TroughAttention | None:
        """Phase 1 of #8 fix: lazily build the herb→pred trough.

        Reset-and-redeposit each scenario, same lifecycle as producer-trough.
        Active only when TROPHIC_TIER_TRANSPORT=trough.
        """
        if _tier_transport() != "trough" or not herb_broadcasts:
            return None
        if self._herb_trough is None:
            n_slots = max(32, len(herb_broadcasts) * 4)
            self._herb_trough = TroughAttention(
                hidden_size=self.host.hidden_size,
                n_slots=n_slots,
                n_heads=8 if self.host.hidden_size % 8 == 0 else 4,
                out_seq_len=8,
                seed=self.cfg.seed + 9002,
                use_E_in=True,
                gated_residual=True,
            ).to(device=self.host.device, dtype=self.host.dtype)
            self.trainer.attach(
                self.herbivores, [self.predator],
                extra_modules=self._trough_extras(),
            )
        trough = self._herb_trough
        alive_ids = trough.alive.nonzero(as_tuple=False).flatten().tolist()
        if alive_ids:
            trough.evict(alive_ids)
        deposit_n = min(len(herb_broadcasts), trough.n_slots)
        trough.deposit(herb_broadcasts[:deposit_n])
        return trough

    def _herb_output(self, herb: Herbivore, candidates: list[Broadcast]):
        """Dispatch herb's cross-tier consumption: per-Channel (channel mode)
        vs producer-trough attend (trough mode). Returns (ch_out, nulls)."""
        if _tier_transport() == "trough":
            trough = self._ensure_producer_trough(candidates)
            if trough is not None:
                from ..agents.base import ROLE_Q_REF_NORM
                _rq = self._role_prefix_mean.to(self.host.device, self.host.dtype)
                role_q = _rq * (ROLE_Q_REF_NORM / (_rq.norm() + 1e-6))
                return self._trough_output_for(
                    herb.kind, HERB_DIETS[herb.kind], trough, role_q
                )
        return self._channel_output_for(
            herb.kind, HERB_DIETS[herb.kind], herb.channels, candidates
        )

    def _pred_output(self, herb_for_pred: list[Broadcast]):
        """Dispatch predator's cross-tier consumption."""
        if _tier_transport() == "trough":
            trough = self._ensure_herb_trough(herb_for_pred)
            if trough is not None:
                from ..agents.base import ROLE_Q_REF_NORM
                _rq = self._role_prefix_mean.to(self.host.device, self.host.dtype)
                role_q = _rq * (ROLE_Q_REF_NORM / (_rq.norm() + 1e-6))
                return self._trough_output_for(
                    self.predator.kind, PRED_DIETS[self.predator.kind],
                    trough, role_q,
                )
        return self._channel_output_for(
            self.predator.kind, PRED_DIETS[self.predator.kind],
            self.predator.channels, herb_for_pred
        )

    def _trough_output_for(
        self,
        agent_kind: str,
        diet_map: list[tuple[str, list[str]]],
        trough: TroughAttention,
        role_q: torch.Tensor,
    ) -> tuple[torch.Tensor, list[float]]:
        """Trough-mode analog of `_channel_output_for`. For each (source_kind,
        diet_tags) entry in diet_map, query the shared trough with
        allowed_tag_strs=diet_tags. Concat outputs in the same shape the
        per-Channel path produces so downstream code is unchanged.
        """
        outs = []
        nulls = []
        for source_kind, tags in diet_map:
            out = trough.attend(
                role_q,
                tau=self._last_tau,
                allowed_tag_strs=tags,
            )
            outs.append(out.output)
            nulls.append(out.null_prob)
        return torch.cat(outs, dim=0), nulls

    # ---------- Mission 06: producer trough + skip path ----------

    def _ensure_producer_trough(self, candidates: list[Broadcast]) -> TroughAttention | None:
        """Lazily build a producer trough sized to the host hidden dim and
        deposit the current scenario's producer broadcasts into it.

        Returns `None` if there are no candidate broadcasts (skip path is
        a no-op for that scenario; predator falls back to channel-only).
        """
        if not candidates:
            return None
        if self._producer_trough is None:
            n_slots = max(64, len(candidates) * 4)
            # Trough-mode (#8 phase 1): identity-init E_in + gated residual
            # (α=0 + zero-init readout). Day-0 output is zero so the
            # consumer's baseline behavior is unchanged; SFT opens the gate.
            _trough_mode = _tier_transport() == "trough"
            self._producer_trough = TroughAttention(
                hidden_size=self.host.hidden_size,
                n_slots=n_slots,
                n_heads=8 if self.host.hidden_size % 8 == 0 else 4,
                out_seq_len=8,
                seed=self.cfg.seed + 9001,
                use_E_in=_trough_mode,
                gated_residual=_trough_mode,
            ).to(device=self.host.device, dtype=self.host.dtype)
            if _trough_mode:
                # Re-attach trainer with the trough as an extra trainable module.
                self.trainer.attach(
                    self.herbivores, [self.predator],
                    extra_modules=self._trough_extras(),
                )
        # Reset the trough each scenario — evict everything, re-deposit.
        trough = self._producer_trough
        alive_ids = trough.alive.nonzero(as_tuple=False).flatten().tolist()
        if alive_ids:
            trough.evict(alive_ids)
        # Truncate to slots available.
        deposit_n = min(len(candidates), trough.n_slots)
        trough.deposit(candidates[:deposit_n])
        return trough

    # ---------- producer cache ----------

    async def _cache_producer_broadcasts(self) -> None:
        for sc in self.train + self.eval_:
            items: list[Broadcast] = []
            for inp in sc.inputs:
                for prod in self.producers:
                    if prod.attracts(inp):
                        b = await prod.produce(inp, tick=0, host=self.host)
                        if b is not None:
                            items.append(b)
            self._producer_cache[sc.name] = items

    # ---------- one scenario forward+loss ----------

    def _channel_output_for(
        self,
        agent_kind: str,                   # herbivore or predator kind
        diet_map: list[tuple[str, list[str]]],
        channels: dict,
        candidates: list[Broadcast],
    ) -> tuple[torch.Tensor, list[float]]:
        """Run each per-source-kind Channel and concat outputs. Returns (channel_seq, null_probs).

        Issue #6 fix: hunter_state must depend on input. Q is built from
        `self._role_prefix_mean` (set by callers to the agent's role prefix
        pooled mean) + the pooled mean of the candidate broadcasts going
        into the channel. The role prefix is fixed per agent; the input
        pool changes per scenario, which restores input conditioning of
        the cross-attention. Without this, Q is constant across all
        scenarios for a given agent and the channel output collapses.
        """
        by_kind: dict[str, list[Broadcast]] = {pk: [] for pk, _ in diet_map}
        for c in candidates:
            if c.agent_kind in by_kind:
                by_kind[c.agent_kind].append(c)
        outs = []
        nulls = []
        # Issue #12 fix: normalize role_q to a fixed reference norm (1.0) so the
        # cross-attention Q geometry doesn't shift when role_prefix length
        # changes (e.g., a longer prompt at inference). Trained Channel weights
        # learned against the original norm; without this normalization any
        # prompt edit at inference time produces gibberish (#11 surfaced this).
        from ..agents.base import ROLE_Q_REF_NORM
        _role_q = self._role_prefix_mean.to(self.host.device, self.host.dtype)
        role_q = _role_q * (ROLE_Q_REF_NORM / (_role_q.norm() + 1e-6))
        for source_kind, _tags in diet_map:
            ch = channels[source_kind]
            prey = by_kind[source_kind]
            if prey:
                prey_t = torch.tensor(
                    [p.channel_embedding for p in prey],
                    dtype=self.host.dtype,
                    device=self.host.device,
                )
                input_q = prey_t.mean(dim=0)
                hs = role_q + input_q
            else:
                prey_t = torch.zeros(0, self.host.hidden_size,
                                     dtype=self.host.dtype, device=self.host.device)
                hs = role_q
            out = ch(hs, prey_t)
            outs.append(out.output)
            nulls.append(out.null_prob)
        return torch.cat(outs, dim=0), nulls

    def _herb_loss(self, herb: Herbivore, sc: Scenario, candidates: list[Broadcast]) -> torch.Tensor | None:
        target = sc.technical_target if herb.kind == "technical" else sc.fundamental_target
        if not target:
            return None
        # set hunter state for _channel_output_for / trough dispatch
        self._role_prefix_mean = herb.role_prefix.mean(dim=0)
        ch_out, _nulls = self._herb_output(herb, candidates)
        loss = self.host.teacher_forcing_loss(
            role_prefix=herb.role_prefix,
            channel_output=ch_out,
            query_text="Now produce the SYNTHESIS and CONFIDENCE.",
            target_text=target,
            content_token_weight=self.cfg.content_token_weight,
        )
        return loss

    def _pred_loss(
        self,
        sc: Scenario,
        herb_outputs_for_predator: list[Broadcast],
        producer_candidates: list[Broadcast] | None = None,
        tau: float = 1.0,
    ) -> torch.Tensor | None:
        if not sc.predator_target:
            return None
        self._role_prefix_mean = self.predator.role_prefix.mean(dim=0)
        ch_out, _nulls = self._pred_output(herb_outputs_for_predator)

        # Mission 06 (phase3-A): cross-tier skip from the producer trough.
        # The herbivore-side path produces ch_out of shape
        # `[N_kinds * out_seq_len, hidden]`. The producer-trough attend
        # produces `[out_seq_len, hidden]`. To combine in matching
        # shape we tile the skip across the N_kinds blocks and mix via
        # `α = sigmoid(skip_weight)`.
        if producer_candidates:
            trough = self._ensure_producer_trough(producer_candidates)
            if trough is not None:
                # Issue #12 fix: normalize role-prefix-derived Q (same fix as
                # in _channel_output_for above).
                from ..agents.base import ROLE_Q_REF_NORM
                _hs = self._role_prefix_mean.to(
                    device=self.host.device, dtype=self.host.dtype
                )
                hs = _hs * (ROLE_Q_REF_NORM / (_hs.norm() + 1e-6))
                skip_out = trough.attend(hs, tau=tau)
                skip_seq = skip_out.output  # [out_seq_len, hidden]
                # ch_out length is N_kinds * out_seq_len; tile skip to match.
                if ch_out.shape[0] % skip_seq.shape[0] == 0 and ch_out.shape[1] == skip_seq.shape[1]:
                    n_tiles = ch_out.shape[0] // skip_seq.shape[0]
                    skip_tiled = skip_seq.repeat(n_tiles, 1)
                    alpha = torch.sigmoid(self.predator.skip_weight.to(
                        device=ch_out.device, dtype=ch_out.dtype
                    ))
                    ch_out = (1.0 - alpha) * ch_out + alpha * skip_tiled

        loss = self.host.teacher_forcing_loss(
            role_prefix=self.predator.role_prefix,
            channel_output=ch_out,
            query_text="Now produce the PREDICTION and CONFIDENCE.",
            target_text=sc.predator_target,
            content_token_weight=self.cfg.content_token_weight,
        )
        return loss

    def _herb_broadcasts_for_predator(self, sc: Scenario, candidates: list[Broadcast]) -> list[Broadcast]:
        """For the predator's training, we use *target-derived* herbivore
        broadcasts (computed by Qwen on the target text) rather than the
        herbivore's actual current output. This decouples predator training
        from herbivore Channel quality early on.

        Includes:
          - technical_target → from_technical_herbivore broadcast
          - fundamental_target → from_fundamental_herbivore broadcast
          - forecaster_target → from_forecaster_herbivore broadcast
            (only for multi-modal scenarios that have one)
          - interrogator_target → from_interrogator_herbivore broadcast
            (only for scenarios with quantitative inputs)

        Abstain targets still produce a broadcast but a degenerate one —
        the predator's null gate decides whether to attend to it.
        """
        out = []
        # Textual herbivores (technical, fundamental)
        for h in self.herbivores:
            target = sc.technical_target if h.kind == "technical" else sc.fundamental_target
            if not target:
                continue
            pooled = self.host.text_to_hidden(target, pool="mean")
            out.append(Broadcast(
                id=f"oracle.{sc.name}.{h.kind}",
                tier="herbivore_broadcast",
                agent_id=h.id,
                agent_kind=h.kind,
                diet_tags=[f"from_{h.kind}_herbivore"],
                channel_embedding=pooled.detach().cpu().tolist(),
            ))
        # Forecaster herbivore (multi-modal scenarios only)
        if sc.forecaster_target:
            pooled = self.host.text_to_hidden(sc.forecaster_target, pool="mean")
            out.append(Broadcast(
                id=f"oracle.{sc.name}.forecaster",
                tier="herbivore_broadcast",
                agent_id="oracle.forecaster",
                agent_kind="forecaster",
                diet_tags=["from_forecaster_herbivore"],
                channel_embedding=pooled.detach().cpu().tolist(),
            ))
        # Interrogator herbivore (any scenario with quant inputs)
        if sc.interrogator_target:
            pooled = self.host.text_to_hidden(sc.interrogator_target, pool="mean")
            out.append(Broadcast(
                id=f"oracle.{sc.name}.interrogator",
                tier="herbivore_broadcast",
                agent_id="oracle.interrogator",
                agent_kind="interrogator",
                diet_tags=["from_interrogator_herbivore"],
                channel_embedding=pooled.detach().cpu().tolist(),
            ))
        return out

    # ---------- step ----------

    def step(self, sc: Scenario, train_step: int = 0) -> dict:
        candidates = self._producer_cache.get(sc.name, [])
        # Mission 06 (phase3-A): cosine τ schedule.
        tau = cosine_tau(
            train_step,
            max(1, self.cfg.steps),
            tau_start=self.cfg.tau_start,
            tau_end=self.cfg.tau_end,
        )
        self._last_tau = tau
        # Herbivore losses
        total = None
        per_loss = {}
        for h in self.herbivores:
            l = self._herb_loss(h, sc, candidates)
            if l is not None:
                per_loss[f"herb.{h.kind}"] = float(l.detach().item())
                total = l if total is None else total + l
        # Predator loss (oracle herb broadcasts, decoupled)
        herb_for_pred = self._herb_broadcasts_for_predator(sc, candidates)
        l = self._pred_loss(sc, herb_for_pred, producer_candidates=candidates, tau=tau)
        if l is not None:
            per_loss["pred.short_horizon"] = float(l.detach().item())
            total = l if total is None else total + l
        if total is None:
            return {"loss": 0.0, "per_loss": per_loss, "n": 0, "tau": tau}
        self.trainer.add_loss(total)
        return {
            "loss": float(total.detach().item()),
            "per_loss": per_loss,
            "n": 1,
            "tau": tau,
        }

    # ---------- Mission 06 diagnostics: per-epoch ecology snapshot ----------

    def ecology_snapshot(self) -> dict:
        """Diagnostic snapshot of skip / τ / trough state.

        Intended for per-epoch logging by the SFT script. Cheap — reads
        buffers and detached parameters; no forward pass.
        """
        trough = self._producer_trough
        snap: dict = {
            "tau": self._last_tau,
            "skip_weight_raw": float(self.predator.skip_weight.detach().item()),
            "skip_weight_alpha": float(
                torch.sigmoid(self.predator.skip_weight).detach().item()
            ),
        }
        if trough is not None:
            spec = trough.slot_specialization()
            head_per_slot = spec["head_per_slot"]
            alive_count = int(trough.alive.sum().item())
            # Distribution: count head dominance per alive slot
            from collections import Counter
            alive_mask = trough.alive.detach().cpu().tolist()
            dominant_per_alive = [
                h for h, alv in zip(head_per_slot, alive_mask) if alv
            ]
            head_distribution = dict(Counter(dominant_per_alive))
            # Specialization "entropy" — Shannon entropy of head usage
            # over alive slots, normalised to [0, 1]. Low → strong
            # specialization (heads have partitioned the niche space);
            # high → heads are interchangeable.
            import math as _math
            n_alive = max(1, len(dominant_per_alive))
            probs = [c / n_alive for c in head_distribution.values()]
            ent = -sum(p * _math.log(p + 1e-12) for p in probs)
            max_ent = _math.log(max(1, trough.n_heads))
            ent_norm = ent / max_ent if max_ent > 0 else 0.0
            snap["trough_n_alive"] = alive_count
            snap["trough_n_killed_total"] = int(trough.dead.sum().item())
            snap["head_distribution"] = head_distribution
            snap["head_specialization_entropy"] = float(ent_norm)
        return snap

    def maybe_step(self, tick: int) -> float | None:
        return self.trainer.maybe_step(tick)

    def eval_loss(self) -> dict:
        """Compute mean teacher-forcing CE on the held-out eval set.

        Returns {'mean': float, 'per_kind': {...}}. Uses no_grad so it
        doesn't contaminate the training gradient or step the optimizer.
        """
        per_kind_totals: dict[str, list[float]] = {}
        with torch.no_grad():
            for sc in self.eval_:
                candidates = self._producer_cache.get(sc.name, [])
                # Per-herb
                for h in self.herbivores:
                    target = sc.technical_target if h.kind == "technical" else sc.fundamental_target
                    if not target:
                        continue
                    self._role_prefix_mean = h.role_prefix.mean(dim=0)
                    ch_out, _ = self._herb_output(h, candidates)
                    loss = self.host.teacher_forcing_loss(
                        role_prefix=h.role_prefix,
                        channel_output=ch_out,
                        query_text="Now produce the SYNTHESIS and CONFIDENCE.",
                        target_text=target,
                    )
                    per_kind_totals.setdefault(f"herb.{h.kind}", []).append(float(loss.item()))
                # Predator (oracle herb hiddens)
                if sc.predator_target:
                    herb_for_pred = self._herb_broadcasts_for_predator(sc, candidates)
                    self._role_prefix_mean = self.predator.role_prefix.mean(dim=0)
                    ch_out, _ = self._pred_output(herb_for_pred)
                    loss = self.host.teacher_forcing_loss(
                        role_prefix=self.predator.role_prefix,
                        channel_output=ch_out,
                        query_text="Now produce the PREDICTION and CONFIDENCE.",
                        target_text=sc.predator_target,
                    )
                    per_kind_totals.setdefault("pred.short_horizon", []).append(float(loss.item()))
        per_kind = {k: round(sum(v) / len(v), 4) for k, v in per_kind_totals.items()}
        flat = [vv for v in per_kind_totals.values() for vv in v]
        mean = sum(flat) / max(1, len(flat))
        return {"mean": round(mean, 4), "per_kind": per_kind}

    # ---------- eval (no_grad decoded sample) ----------

    def eval_decode(self, sc: Scenario, herb_kind: str | None = None,
                    max_new_tokens: int | None = None) -> dict:
        """Run a single scenario in inference mode, return decoded outputs.

        Issue #8 / phase3-A finding: greedy decoding on undertrained checkpoints
        burns the full max_new_tokens budget on gibberish, making in-loop eval
        the wall-time bottleneck (7-15 min per eval × every-50-steps killed
        SFT seed 9 twice). The TROPHIC_EVAL_MAX_TOKENS env var caps in-loop
        eval generation; default 96 (matches IPO target_max_tokens). Set higher
        only for end-of-training eval where verbose output matters.
        """
        import os as _os
        if max_new_tokens is None:
            max_new_tokens = int(_os.environ.get("TROPHIC_EVAL_MAX_TOKENS", "96"))
        candidates = self._producer_cache.get(sc.name, [])
        out: dict = {"name": sc.name}
        with torch.no_grad():
            for h in self.herbivores:
                if herb_kind and h.kind != herb_kind:
                    continue
                self._role_prefix_mean = h.role_prefix.mean(dim=0)
                ch_out, nulls = self._herb_output(h, candidates)
                fr = self.host.forward_with_prefix(
                    role_prefix=h.role_prefix,
                    channel_output=ch_out,
                    query_text="Now produce the SYNTHESIS and CONFIDENCE.",
                    decode=True,
                    max_new_tokens=max_new_tokens,
                )
                out[f"herb.{h.kind}"] = (fr.decoded_text or "").strip()[:400]
                out[f"herb.{h.kind}.null"] = round(sum(nulls) / max(1, len(nulls)), 3)
            # Predator on oracle herb hiddens
            herb_for_pred = self._herb_broadcasts_for_predator(sc, candidates)
            self._role_prefix_mean = self.predator.role_prefix.mean(dim=0)
            ch_out, nulls = self._pred_output(herb_for_pred)
            fr = self.host.forward_with_prefix(
                role_prefix=self.predator.role_prefix,
                channel_output=ch_out,
                query_text="Now produce the PREDICTION and CONFIDENCE.",
                decode=True,
                max_new_tokens=max_new_tokens,
            )
            out["pred.short_horizon"] = (fr.decoded_text or "").strip()[:400]
        return out
