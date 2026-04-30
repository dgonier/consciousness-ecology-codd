"""Reward-driven RL trainer using XML parser + rule-based reward.

For each scenario:
  1. Build the predator's prefix (channel output + role + query) as in
     SFT/IPO.
  2. Sample G completions per scenario, with optional Gaussian noise on
     channel_output per completion to give RL real prefix-level diversity.
  3. Parse each completion's XML and compute rule-based reward against
     the oracle target.
  4. Within-group advantage = (r_i - mean) / (std + eps).
  5. Loss per completion = -advantage * mean_logprob(completion) +
     β * KL(π_live || π_ref).
  6. Backprop through Predator's Channels only.

Differences from GRPO:
  - Reward source is rule-based parser, not LLM judge. Faster, exact, free.
  - Channel-side noise is a first-class sampling knob.
  - No JudgeRouter (the parser is the judge).

The KL anchor is still useful: it prevents the policy from drifting away
from valid XML format under the reward pressure (reward pressures only
ticker/direction/magnitude/etc.; format compliance came from SFT and we
want to keep it).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..agents.herbivore import Herbivore
from ..agents.predator import DIETS as PRED_DIETS, Predator
from ..channel import Channel
from ..model_host import ModelHost
from ..types import Broadcast
from .scenarios import Scenario
from .xml_schema import parse_prediction, reward_prediction, RewardConfig


@dataclass
class RewardRLConfig:
    lr: float = 5e-5             # smaller than SFT — fine-tuning
    grad_clip: float = 1.0
    group_size: int = 4
    sampling_temperature: float = 0.9
    sampling_top_p: float = 0.95
    max_new_tokens: int = 96
    kl_beta: float = 0.05
    channel_noise_std: float = 0.05  # Gaussian noise on channel_output per sample
    steps: int = 300
    log_every: int = 10
    eval_every: int = 50
    seed: int = 1
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)


def _freeze_clone_channel(ch: Channel) -> Channel:
    clone = copy.deepcopy(ch)
    for p in clone.parameters():
        p.requires_grad = False
    clone.eval()
    return clone


@dataclass
class _PredatorChannelStack:
    live: dict[str, Channel]
    ref: dict[str, Channel]


@dataclass
class RewardRLRunner:
    cfg: RewardRLConfig
    host: ModelHost
    herbivores: list[Herbivore]
    predator: Predator
    train: list[Scenario]
    eval_: list[Scenario]
    optimizer: torch.optim.Optimizer | None = None
    pred_stack: _PredatorChannelStack | None = None
    losses: list[float] = field(default_factory=list)
    rewards_history: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.host.freeze_base_model()
        self.predator.ensure_initialized(self.host, seed_base=self.cfg.seed)
        ref_channels = {
            src: _freeze_clone_channel(ch)
            for src, ch in self.predator.channels.items()
        }
        self.pred_stack = _PredatorChannelStack(
            live=dict(self.predator.channels), ref=ref_channels
        )
        for ch in self.pred_stack.ref.values():
            ch.to(device=self.host.device, dtype=self.host.dtype)
        params: list[torch.nn.Parameter] = []
        for ch in self.pred_stack.live.values():
            params.extend(p for p in ch.parameters() if p.requires_grad)
        self.optimizer = torch.optim.Adam(params, lr=self.cfg.lr)

    # ---------- oracle herb broadcasts ----------

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

    # ---------- channel output (with optional noise) ----------

    def _channel_output(
        self, channels: dict[str, Channel], candidates: list[Broadcast],
        noise_std: float = 0.0,
    ) -> torch.Tensor:
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
        result = torch.cat(outs, dim=0)
        if noise_std > 0.0:
            noise = torch.randn_like(result) * noise_std
            result = result + noise
        return result

    def _build_prefix(
        self,
        channel_output: torch.Tensor,
        query_text: str = "Now produce the PREDICTION and CONFIDENCE.",
    ) -> torch.Tensor:
        device = self.host.device
        dtype = self.host.dtype
        rp = self.predator.role_prefix.detach().to(device=device, dtype=dtype)
        co = channel_output.to(device=device, dtype=dtype)
        q_ids = self.host._tok(
            query_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        q_emb = self.host._embed_layer(q_ids)[0].detach()
        return torch.cat([rp, co, q_emb], dim=0).unsqueeze(0)

    def _sample_completion(self, prefix_embeds: torch.Tensor) -> torch.Tensor:
        attn_mask = torch.ones(
            prefix_embeds.shape[:2], dtype=torch.long, device=self.host.device
        )
        with torch.no_grad():
            gen = self.host._model.generate(
                inputs_embeds=prefix_embeds,
                attention_mask=attn_mask,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=True,
                temperature=self.cfg.sampling_temperature,
                top_p=self.cfg.sampling_top_p,
                repetition_penalty=1.3,
                pad_token_id=self.host._tok.eos_token_id,
            )
        return gen[0]

    # ---------- per-completion logprob (live or ref) ----------

    def _summed_logprob(
        self,
        channels: dict[str, Channel],
        candidates: list[Broadcast],
        completion_tokens: torch.Tensor,
        compute_grad: bool,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sum of per-token logprobs of `completion_tokens` under the policy
        defined by `channels` plus optional fixed noise on the channel output
        (to match the noise used at sampling time).

        Returns a scalar tensor on device.
        """
        ctx = torch.enable_grad() if compute_grad else torch.no_grad()
        with ctx:
            ch_out = self._channel_output(channels, candidates, noise_std=0.0)
            if noise is not None:
                ch_out = ch_out + noise
            prefix_embeds = self._build_prefix(ch_out)
            gen_emb = self.host._embed_layer(
                completion_tokens.unsqueeze(0)
            ).detach()
            full_embeds = torch.cat([prefix_embeds, gen_emb], dim=1)
            attn_mask = torch.ones(
                full_embeds.shape[:2], dtype=torch.long,
                device=self.host.device,
            )
            out = self.host._model(
                inputs_embeds=full_embeds,
                attention_mask=attn_mask,
                use_cache=False,
            )
            logits = out.logits[0]
            P = prefix_embeds.shape[1]
            T = completion_tokens.shape[0]
            pred_logits = logits[P - 1: P - 1 + T]
            log_probs = F.log_softmax(pred_logits, dim=-1)
            chosen = log_probs.gather(
                1, completion_tokens.unsqueeze(1)
            ).squeeze(1)
            return chosen.sum()

    # ---------- one step ----------

    def step(self, sc: Scenario, tick: int) -> dict:
        candidates = self._oracle_herb_broadcasts(sc)
        target = parse_prediction(sc.predator_target or "")

        # Sample G completions, each with fresh channel-side noise
        completions: list[torch.Tensor] = []
        noises: list[torch.Tensor] = []
        for _ in range(self.cfg.group_size):
            with torch.no_grad():
                ch_out_clean = self._channel_output(
                    self.pred_stack.live, candidates, noise_std=0.0
                )
                if self.cfg.channel_noise_std > 0:
                    noise = torch.randn_like(ch_out_clean) * self.cfg.channel_noise_std
                else:
                    noise = torch.zeros_like(ch_out_clean)
                ch_out_noisy = ch_out_clean + noise
                prefix = self._build_prefix(ch_out_noisy)
            tokens = self._sample_completion(prefix)
            if tokens.numel() == 0:
                continue
            completions.append(tokens)
            noises.append(noise)

        if not completions:
            return {"loss": 0.0, "skipped": True}

        # Score each completion via XML parse + rule-based reward
        decoded = [
            self.host._tok.decode(t, skip_special_tokens=True).strip()
            for t in completions
        ]
        # Truncate at the closing predator tag so reasoning tail doesn't pollute
        truncated = []
        for d in decoded:
            for stop in ("</prediction>", "</synthesis>"):
                idx = d.find(stop)
                if idx >= 0:
                    d = d[: idx + len(stop)]
                    break
            truncated.append(d)
        rewards = torch.tensor(
            [reward_prediction(parse_prediction(d), target, self.cfg.reward_cfg)["total"]
             for d in truncated],
            dtype=torch.float32,
        )
        self.rewards_history.extend(rewards.tolist())

        # Within-group advantage
        if rewards.std().item() < 1e-6:
            advantages = torch.zeros_like(rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        # Loss
        total_loss = torch.zeros(1, device=self.host.device, dtype=self.host.dtype)
        for i, tokens in enumerate(completions):
            adv = float(advantages[i].item())
            if abs(adv) < 1e-6:
                continue
            live_lp = self._summed_logprob(
                self.pred_stack.live, candidates, tokens,
                compute_grad=True, noise=noises[i],
            )
            policy_term = -adv * (live_lp / max(1, tokens.shape[0]))  # mean per-token
            ref_lp = self._summed_logprob(
                self.pred_stack.ref, candidates, tokens,
                compute_grad=False, noise=noises[i],
            )
            kl_term = (live_lp.exp() * (live_lp - ref_lp)).mean()
            total_loss = total_loss + policy_term + self.cfg.kl_beta * kl_term

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
            "rewards_max": float(rewards.max().item()),
            "decoded_best": truncated[int(rewards.argmax().item())][:200],
        }

    # ---------- eval (greedy decode + score, no training) ----------

    def eval_decode(self, sc: Scenario) -> dict:
        candidates = self._oracle_herb_broadcasts(sc)
        with torch.no_grad():
            ch_out = self._channel_output(self.pred_stack.live, candidates, noise_std=0.0)
        result = self.host.forward_with_prefix(
            role_prefix=self.predator.role_prefix,
            channel_output=ch_out,
            query_text="Now produce the PREDICTION and CONFIDENCE.",
            decode=True,
            max_new_tokens=128,
            pool="mean",
        )
        text = (result.decoded_text or "").strip()
        # forward_with_prefix already early-stops at </prediction>
        target = parse_prediction(sc.predator_target or "")
        parsed = parse_prediction(text)
        rew = reward_prediction(parsed, target, self.cfg.reward_cfg)
        return {
            "name": sc.name,
            "decoded": text[:240],
            "reward": rew["total"],
            "breakdown": rew["breakdown"],
            "parsed": parsed,
            "target": target,
        }
