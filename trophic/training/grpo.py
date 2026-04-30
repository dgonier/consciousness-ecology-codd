"""GRPO trainer for the predator's Channels.

Group-Relative Policy Optimization without a learned value model. For each
scenario:
  1. Sample G completions from the predator (same prefix, sampling temperature)
  2. Score each with JudgeRouter → rewards
  3. Advantage per completion = (r_i - mean(rewards)) / (std(rewards)+eps)
  4. Loss per token = -advantage * logprob(token) + β * KL(π_live || π_ref)
  5. Backprop through predator's 3 Channels only

For v1 of GRPO we train *only the predator's Channels*:
  - The herbivore broadcasts come from the oracle path (target text → Qwen),
    same as in SFT. Decouples GRPO from herbivore quality.
  - The forecaster Channel + technical Channel + fundamental Channel of the
    predator are the trainable surface.
  - All other parameters (Qwen weights, role prefixes, herbivore Channels)
    are frozen.

KL anchor: a frozen snapshot of the post-SFT predator Channels lives in
parallel. Each forward pass also runs through this snapshot to get the
reference logprobs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn.functional as F

from ..agents.herbivore import DIETS as HERB_DIETS, Herbivore
from ..agents.predator import DIETS as PRED_DIETS, Predator
from ..channel import Channel
from ..model_host import ModelHost
from ..types import Broadcast, PredatorJudgment
from ..predators.judge_router import JudgeRouter
from .scenarios import Scenario


@dataclass
class GRPOConfig:
    lr: float = 1e-4                 # smaller than SFT — fine-tuning
    grad_clip: float = 1.0
    group_size: int = 4              # G
    sampling_temperature: float = 0.8
    sampling_top_p: float = 0.9
    max_new_tokens: int = 96
    kl_beta: float = 0.02
    steps: int = 200
    log_every: int = 10
    eval_every: int = 25
    consistency_probe_every: int = 10
    consistency_k: int = 3
    seed: int = 7


# ---------- helper: deep-copy + freeze a Channel ----------

def _freeze_clone_channel(ch: Channel) -> Channel:
    clone = copy.deepcopy(ch)
    for p in clone.parameters():
        p.requires_grad = False
    clone.eval()
    return clone


@dataclass
class _PredatorChannelStack:
    """Reference vs live Channels, indexed by source kind."""
    live: dict[str, Channel]
    ref: dict[str, Channel]


@dataclass
class GRPORunner:
    cfg: GRPOConfig
    host: ModelHost
    herbivores: list[Herbivore]   # frozen, used for oracle herb broadcasts
    predator: Predator            # live Channels are inside .channels
    train: list[Scenario]
    eval_: list[Scenario]
    judge: JudgeRouter
    optimizer: torch.optim.Optimizer | None = None
    pred_stack: _PredatorChannelStack | None = None
    losses: list[float] = field(default_factory=list)
    rewards_history: list[float] = field(default_factory=list)
    abstain_count: int = 0

    def __post_init__(self):
        self.host.freeze_base_model()
        # Make sure the predator's Channels are initialized.
        self.predator.ensure_initialized(self.host, seed_base=self.cfg.seed)
        # Build ref snapshot before any training step.
        ref_channels = {
            src: _freeze_clone_channel(ch)
            for src, ch in self.predator.channels.items()
        }
        self.pred_stack = _PredatorChannelStack(
            live=dict(self.predator.channels), ref=ref_channels
        )
        # Move ref Channels onto same device/dtype as live ones.
        for ch in self.pred_stack.ref.values():
            ch.to(device=self.host.device, dtype=self.host.dtype)

        # Optimizer over live Channel params.
        params: list[torch.nn.Parameter] = []
        for ch in self.pred_stack.live.values():
            params.extend(p for p in ch.parameters() if p.requires_grad)
        self.optimizer = torch.optim.Adam(params, lr=self.cfg.lr)

    # ---------- oracle herbivore broadcasts (mirror of SFT path) ----------

    def _oracle_herb_broadcasts(self, sc: Scenario) -> list[Broadcast]:
        out = []
        for h in self.herbivores:
            target = (
                sc.technical_target if h.kind == "technical"
                else sc.fundamental_target
            )
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

    # ---------- channel output (live or ref) ----------

    def _channel_output(
        self, channels: dict[str, Channel], candidates: list[Broadcast]
    ) -> torch.Tensor:
        """Run the diet-mapped Channels and concat their outputs.

        Mirrors SFTRunner._channel_output_for, but takes a stack-side selector.
        """
        diet_map = PRED_DIETS[self.predator.kind]
        by_kind: dict[str, list[Broadcast]] = {pk: [] for pk, _ in diet_map}
        for c in candidates:
            if c.agent_kind in by_kind:
                by_kind[c.agent_kind].append(c)
        outs = []
        # Issue #6 fix: hunter_state must depend on input.
        # Issue #12 fix: normalize role_q to fixed reference norm.
        from ..agents.base import normalize_role_q
        role_q = normalize_role_q(self.predator.role_prefix).to(
            device=self.host.device, dtype=self.host.dtype
        )
        for source_kind, _tags in diet_map:
            ch = channels[source_kind]
            prey = by_kind[source_kind]
            if prey:
                prey_t = torch.tensor(
                    [p.channel_embedding for p in prey],
                    dtype=self.host.dtype, device=self.host.device,
                )
                input_q = prey_t.mean(dim=0)
                hunter_state = role_q + input_q
            else:
                prey_t = torch.zeros(
                    0, self.host.hidden_size,
                    dtype=self.host.dtype, device=self.host.device,
                )
                hunter_state = role_q
            out = ch(hunter_state, prey_t)
            outs.append(out.output)
        return torch.cat(outs, dim=0)

    # ---------- group sample completions ----------

    def _sample_completion(
        self,
        prefix_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        """Generate a token sequence from the predator. Returns just the
        generated portion (does NOT include the prefix tokens).
        """
        if self.host.cfg.mock:
            # Mock path: deterministic-ish per-call random token ids of fixed length.
            import os as _os
            seed_bytes = _os.urandom(4)
            g = torch.Generator().manual_seed(int.from_bytes(seed_bytes, "big"))
            return torch.randint(
                low=0, high=32000, size=(min(max_new_tokens, 16),),
                dtype=torch.long, device=self.host.device, generator=g,
            )
        with torch.no_grad():
            gen = self.host._model.generate(
                inputs_embeds=prefix_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.3,
                pad_token_id=self.host._tok.eos_token_id,
            )
        # When inputs_embeds is used, generate returns only new tokens in [0, T).
        return gen[0]  # [T]

    def _build_prefix(
        self,
        channel_output: torch.Tensor,
        query_text: str = "Now produce the PREDICTION and CONFIDENCE.",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct inputs_embeds = [role_prefix ⊕ channel_output ⊕ query_embeds].

        channel_output is a live tensor with grad. role_prefix and query_embeds
        are detached.
        """
        device = self.host.device
        dtype = self.host.dtype
        rp = self.predator.role_prefix.detach().to(device=device, dtype=dtype)
        co = channel_output.to(device=device, dtype=dtype)
        if self.host.cfg.mock:
            # Mock query: 4 tokens of zero-vector embedding so shapes are valid.
            q_emb = torch.zeros(4, self.host.hidden_size, device=device, dtype=dtype)
        else:
            q_ids = self.host._tok(
                query_text, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            q_emb = self.host._embed_layer(q_ids)[0].detach()
        inputs_embeds = torch.cat([rp, co, q_emb], dim=0).unsqueeze(0)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=device
        )
        return inputs_embeds, attention_mask

    def _logprobs_under(
        self,
        channels: dict[str, Channel],
        candidates: list[Broadcast],
        gen_tokens: torch.Tensor,
        compute_grad: bool,
    ) -> torch.Tensor:
        """Per-token logprobs of `gen_tokens` under the policy defined by
        the given Channels (with role_prefix + channel_output + query as the
        prefix). Returns [T] tensor on device.
        """
        ctx = torch.enable_grad() if compute_grad else torch.no_grad()
        with ctx:
            if self.host.cfg.mock:
                # Mock: a synthetic logprob signal that has gradient flow
                # through `channels` so the optimizer mechanics can be
                # smoke-tested. We compute a scalar that depends on the
                # channel output norm; lp ≈ -||ch_out||² broadcast to [T].
                ch_out = self._channel_output(channels, candidates)
                T = gen_tokens.shape[0]
                # Single scalar with grad, then expanded to [T]; not real
                # logprobs, but gradient does flow.
                scalar = -ch_out.float().pow(2).mean()
                return scalar.expand(T)
            ch_out = self._channel_output(channels, candidates)
            prefix_embeds, _ = self._build_prefix(ch_out)
            # Append generated tokens' embeddings (detached — we only need
            # logprobs at those positions, not gradients through the
            # token IDs).
            gen_emb = self.host._embed_layer(gen_tokens.unsqueeze(0)).detach()
            full_embeds = torch.cat([prefix_embeds, gen_emb], dim=1)
            attn_mask = torch.ones(
                full_embeds.shape[:2], dtype=torch.long, device=self.host.device
            )
            out = self.host._model(
                inputs_embeds=full_embeds,
                attention_mask=attn_mask,
                use_cache=False,
            )
            logits = out.logits[0]  # [P+T, vocab]
            P = prefix_embeds.shape[1]
            T = gen_tokens.shape[0]
            # logits at position P+t-1 predict token t (0-indexed in gen).
            # Position P-1 predicts gen[0]; position P+T-2 predicts gen[T-1].
            pred_logits = logits[P - 1 : P - 1 + T]  # [T, vocab]
            log_probs = F.log_softmax(pred_logits, dim=-1)  # [T, vocab]
            chosen = log_probs.gather(1, gen_tokens.unsqueeze(1)).squeeze(1)  # [T]
        return chosen

    # ---------- one GRPO step over one scenario ----------

    async def step(self, sc: Scenario, tick: int) -> dict:
        candidates = self._oracle_herb_broadcasts(sc)

        # 1. Build prefix embeds (one shared prefix per group, computed once)
        with torch.no_grad():
            ch_out_for_sample = self._channel_output(self.pred_stack.live, candidates)
            prefix_embeds, attn_mask = self._build_prefix(ch_out_for_sample)

        # 2. Sample G completions
        completions: list[torch.Tensor] = []
        for _ in range(self.cfg.group_size):
            tokens = self._sample_completion(
                prefix_embeds, attn_mask,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.sampling_temperature,
                top_p=self.cfg.sampling_top_p,
            )
            completions.append(tokens)

        # 3. Build broadcasts for judging + score
        broadcasts: list[Broadcast] = []
        for tokens in completions:
            if self.host.cfg.mock:
                # Synthetic decoded text in mock mode for the judge to score.
                text = f"PREDICTION: mock_token_seq[len={tokens.shape[0]}]"
            else:
                text = self.host._tok.decode(tokens, skip_special_tokens=True).strip()
            broadcasts.append(Broadcast(
                id=f"grpo.{sc.name}.{len(broadcasts)}",
                tier="predator_broadcast",
                agent_id=self.predator.id,
                agent_kind=self.predator.kind,
                parent_input_ids=[c.id for c in candidates],
                diet_tags=[f"from_{self.predator.kind}_predator"],
                payload={},
                decoded_text=text,
                channel_embedding=[],
                created_tick=tick,
            ))
        # Build context from the scenario for the judge — gives it a way to
        # detect wrong-ticker / wrong-direction outputs.
        context = _scenario_context(sc)
        judgments = await self.judge.judge_batch(broadcasts, tick, context=context)
        rewards = torch.tensor(
            [j.score for j in judgments], dtype=torch.float32
        )
        self.rewards_history.extend(rewards.tolist())

        # 4. Within-group advantage
        if rewards.std().item() < 1e-6:
            advantages = torch.zeros_like(rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        # 5. Compute loss with KL anchor
        total_loss = torch.zeros(1, device=self.host.device, dtype=self.host.dtype)
        for i, tokens in enumerate(completions):
            adv = float(advantages[i].item())
            # Skip if advantage is exactly 0 (no signal).
            if abs(adv) < 1e-6:
                continue
            live_lp = self._logprobs_under(
                self.pred_stack.live, candidates, tokens, compute_grad=True
            )
            # Avg per-token logprob ⇒ scalar; multiply by advantage.
            policy_term = -adv * live_lp.mean()
            ref_lp = self._logprobs_under(
                self.pred_stack.ref, candidates, tokens, compute_grad=False
            )
            kl_term = (live_lp.exp() * (live_lp - ref_lp)).mean()
            total_loss = total_loss + policy_term + self.cfg.kl_beta * kl_term

        # 6. Step
        if total_loss.requires_grad:
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            params = []
            for ch in self.pred_stack.live.values():
                params.extend(ch.parameters())
            torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
            self.optimizer.step()

        loss_val = float(total_loss.detach().item())
        self.losses.append(loss_val)
        return {
            "loss": loss_val,
            "rewards": rewards.tolist(),
            "rewards_mean": float(rewards.mean().item()),
            "rewards_std": float(rewards.std().item()),
            "advantages": advantages.tolist(),
            "judge": self.judge.current_judge_name,
        }

    # ---------- eval (greedy decode + score) ----------

    async def eval_decode(self, sc: Scenario, tick: int) -> dict:
        candidates = self._oracle_herb_broadcasts(sc)
        with torch.no_grad():
            ch_out = self._channel_output(self.pred_stack.live, candidates)
        result = self.host.forward_with_prefix(
            role_prefix=self.predator.role_prefix,
            channel_output=ch_out,
            query_text="Now produce the PREDICTION and CONFIDENCE.",
            decode=True,
            max_new_tokens=192,
            pool="mean",
        )
        text = (result.decoded_text or "").strip()
        br = Broadcast(
            id=f"eval.{sc.name}",
            tier="predator_broadcast",
            agent_id=self.predator.id,
            agent_kind=self.predator.kind,
            diet_tags=[f"from_{self.predator.kind}_predator"],
            decoded_text=text,
            channel_embedding=[],
            created_tick=tick,
        )
        ctx = _scenario_context(sc)
        j = await self.judge.judge(br, tick, context=ctx)
        return {
            "name": sc.name,
            "decoded": text[:300],
            "score": j.score,
            "rationale": j.rationale[:160],
            "judge": self.judge.current_judge_name,
        }


# ---------- module-level helper ----------

def _scenario_context(sc: Scenario) -> str:
    """Short summary of what a scenario's upstream content is about.

    Extracted from the scenario's own targets so the judge can detect
    grounding failures (wrong ticker, wrong direction) without seeing the
    full target text.
    """
    import re
    target = sc.predator_target or ""
    m = re.search(r"\b([A-Z]{2,5})\b", target)
    ticker = m.group(1) if m else "?"
    direction = "?"
    for d in ("up", "down", "flat"):
        if d in target.lower():
            direction = d
            break
    return f"about ticker {ticker}, expected direction {direction}"
