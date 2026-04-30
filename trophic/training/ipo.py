"""IPO — Identity Preference Optimization.

We have oracle targets per scenario, so every (chosen=target, rejected=
sampled-from-policy) pair has *certain* preference. DPO's log-sigmoid loss
becomes unstable in that regime (drives logπ(rejected)→-∞); IPO replaces
it with a squared loss that doesn't.

Loss formulation (from Azar et al., "A General Theoretical Paradigm to
Understand Learning from Human Preferences"):

    h_w(x, y_c, y_r) = log π_θ(y_c|x) - log π_θ(y_r|x)
                     - (log π_ref(y_c|x) - log π_ref(y_r|x))
    L_IPO = (h_w - 1/(2β))²

For each step:
  1. Pick a scenario; chosen = sc.predator_target.
  2. Sample one completion from the live policy → rejected.
  3. Compute log π_θ and log π_ref of both chosen and rejected.
  4. IPO loss above; backprop into predator Channels only.

Like GRPO, we only train the predator's Channels (technical, fundamental,
forecaster). Herbivore Channels stay frozen at their SFT values.
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


@dataclass
class IPOConfig:
    lr: float = 1e-4
    grad_clip: float = 1.0
    beta: float = 0.1               # IPO temperature (smaller = sharper preference)
    sampling_temperature: float = 1.0  # sample diverse rejecteds
    sampling_top_p: float = 0.95
    max_new_tokens: int = 64        # rejected completions only need to be plausibly wrong
    target_max_tokens: int = 96     # chosen target tokens budget
    steps: int = 800
    log_every: int = 25
    eval_every: int = 100
    seed: int = 1


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
class IPORunner:
    cfg: IPOConfig
    host: ModelHost
    herbivores: list[Herbivore]
    predator: Predator
    train: list[Scenario]
    eval_: list[Scenario]
    optimizer: torch.optim.Optimizer | None = None
    pred_stack: _PredatorChannelStack | None = None
    losses: list[float] = field(default_factory=list)
    margins: list[float] = field(default_factory=list)  # h_w per step

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

    # ---------- oracle herbivore broadcasts (same as GRPO) ----------

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

    # ---------- channel output ----------

    def _channel_output(
        self, channels: dict[str, Channel], candidates: list[Broadcast]
    ) -> torch.Tensor:
        diet_map = PRED_DIETS[self.predator.kind]
        by_kind: dict[str, list[Broadcast]] = {pk: [] for pk, _ in diet_map}
        for c in candidates:
            if c.agent_kind in by_kind:
                by_kind[c.agent_kind].append(c)
        outs = []
        # Issue #6 fix: hunter_state must depend on input, not just role prefix.
        # Issue #12 fix: normalize role_q to fixed reference norm so prefix
        # length changes don't shift Q geometry out of the trained distribution.
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
        return torch.cat([rp, co, q_emb], dim=0).unsqueeze(0)  # [1, P, H]

    # ---------- sample rejected ----------

    def _sample_rejected_tokens(
        self, prefix_embeds: torch.Tensor
    ) -> torch.Tensor:
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
        return gen[0]  # [T_rej]

    def _tokenize_target(self, text: str) -> torch.Tensor:
        ids = self.host._tok(
            text, return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=self.cfg.target_max_tokens,
        ).input_ids[0].to(self.host.device)
        eos = torch.tensor([self.host._tok.eos_token_id], device=self.host.device)
        return torch.cat([ids, eos])  # [T_chosen]

    # ---------- logprobs ----------

    def _summed_logprob(
        self,
        channels: dict[str, Channel],
        candidates: list[Broadcast],
        completion_tokens: torch.Tensor,
        compute_grad: bool,
    ) -> torch.Tensor:
        """Sum of per-token logprobs of `completion_tokens` under the policy
        defined by `channels` (with the standard predator prefix). Returns
        scalar tensor on device.
        """
        ctx = torch.enable_grad() if compute_grad else torch.no_grad()
        with ctx:
            ch_out = self._channel_output(channels, candidates)
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
            pred_logits = logits[P - 1: P - 1 + T]                         # [T, V]
            log_probs = F.log_softmax(pred_logits, dim=-1)                  # [T, V]
            chosen = log_probs.gather(
                1, completion_tokens.unsqueeze(1)
            ).squeeze(1)                                                    # [T]
            return chosen.sum()

    # ---------- one IPO step ----------

    def step(self, sc: Scenario, tick: int) -> dict:
        if not sc.predator_target:
            return {"loss": 0.0, "skipped": True}
        candidates = self._oracle_herb_broadcasts(sc)

        # Tokenize chosen target.
        chosen_tokens = self._tokenize_target(sc.predator_target)

        # Sample rejected with temperature on the live policy.
        with torch.no_grad():
            ch_out = self._channel_output(self.pred_stack.live, candidates)
            prefix_embeds = self._build_prefix(ch_out)
        rej_tokens = self._sample_rejected_tokens(prefix_embeds)
        if rej_tokens.numel() == 0:
            return {"loss": 0.0, "skipped": True, "reason": "empty rejected"}

        # Compute the four logprob sums.
        log_pi_c = self._summed_logprob(
            self.pred_stack.live, candidates, chosen_tokens, compute_grad=True
        )
        log_pi_r = self._summed_logprob(
            self.pred_stack.live, candidates, rej_tokens, compute_grad=True
        )
        log_ref_c = self._summed_logprob(
            self.pred_stack.ref, candidates, chosen_tokens, compute_grad=False
        )
        log_ref_r = self._summed_logprob(
            self.pred_stack.ref, candidates, rej_tokens, compute_grad=False
        )

        # h_w = (log π(c) - log π(r)) - (log π_ref(c) - log π_ref(r))
        h_w = (log_pi_c - log_pi_r) - (log_ref_c - log_ref_r)
        target = 1.0 / (2.0 * self.cfg.beta)
        loss = (h_w - target) ** 2

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = []
        for ch in self.pred_stack.live.values():
            params.extend(ch.parameters())
        torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
        self.optimizer.step()

        loss_val = float(loss.detach().item())
        h_val = float(h_w.detach().item())
        self.losses.append(loss_val)
        self.margins.append(h_val)

        # Decode samples for visibility (cheap because tokens already exist).
        rej_text = self.host._tok.decode(rej_tokens, skip_special_tokens=True).strip()
        return {
            "loss": loss_val,
            "h_w": h_val,
            "log_pi_c": float(log_pi_c.detach().item()),
            "log_pi_r": float(log_pi_r.detach().item()),
            "log_ref_c": float(log_ref_c.detach().item()),
            "log_ref_r": float(log_ref_r.detach().item()),
            "rejected_text": rej_text[:160],
        }

    # ---------- eval (greedy decode + judge) ----------

    async def eval_decode(self, sc: Scenario, tick: int, judge=None) -> dict:
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
        out = {"name": sc.name, "decoded": text[:300]}
        if judge is not None:
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
            from .grpo import _scenario_context
            ctx = _scenario_context(sc)
            j = await judge.judge(br, tick, context=ctx)
            out["score"] = j.score
            out["rationale"] = j.rationale[:120]
        return out
