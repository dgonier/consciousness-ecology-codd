"""Isolated training run for E_analyst_math — token-bypass cross-model channel.

What's trained: only the InterrogatorHerbivore's CrossModelChannel (the
two CrossModelProjection MLPs, ~42M params). Everything else stays frozen:
Qwen3-4B, Qwen2.5-Math-1.5B, all herbivore/predator Channels.

Training signal: rule-based reward on the Interrogator's synthesis output.
Specifically magnitude correctness — if the synthesis emits a pct_move
within tolerance of the oracle target's pct_move, reward is high.

Loss shape: REINFORCE with within-group advantage and KL anchor against
a frozen snapshot of the channel at init.

For each scenario:
  1. Build producer broadcasts + render the meal.
  2. Plan phase: Qwen3-4B emits ASK lines. (No grad.)
  3. For each ASK question:
     a. Embed question via Qwen3-4B's tokenizer + embeddings.
     b. Project through E_a→m (LIVE — gradient flows here).
     c. Forward Math model on projected embeds. Get answer logprobs.
        (No grad on Math params; grad does flow through the input embeds.)
     d. Decode answer text.
  4. Synthesize: Qwen3-4B writes the XML synthesis using the answers.
  5. Parse synthesis with the rule-based parser.
  6. Compute reward against the oracle.
  7. Advantage = (reward - mean) / (std + eps); loss = -adv * Σ logprob_math.
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field

import torch

from ..agents.interrogator_herbivore import InterrogatorHerbivore, _extract_questions, _render_inputs
from ..cross_model_channel import CrossModelChannel, CrossModelProjection
from ..math_host import MathHost, _parse_math_response
from ..model_host import ModelHost
from ..types import Broadcast
from .scenarios import Scenario
from .xml_schema import (
    Synthesis, parse_synthesis, reward_synthesis, RewardConfig, emit_synthesis,
)


@dataclass
class CrossModelRLConfig:
    lr: float = 5e-5
    grad_clip: float = 1.0
    group_size: int = 4
    sampling_temperature: float = 0.7   # math model decode temp
    max_new_tokens_math: int = 96
    kl_beta: float = 0.02
    steps: int = 200
    log_every: int = 10
    eval_every: int = 50
    seed: int = 1
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)


def _freeze_clone_projection(p: CrossModelProjection) -> CrossModelProjection:
    clone = copy.deepcopy(p)
    for param in clone.parameters():
        param.requires_grad = False
    clone.eval()
    return clone


@dataclass
class _ChannelStack:
    live: CrossModelChannel
    ref: CrossModelChannel


@dataclass
class CrossModelRLRunner:
    cfg: CrossModelRLConfig
    host: ModelHost
    math_host: MathHost
    interrogator: InterrogatorHerbivore
    train: list[Scenario]
    eval_: list[Scenario]
    optimizer: torch.optim.Optimizer | None = None
    stack: _ChannelStack | None = None
    losses: list[float] = field(default_factory=list)
    rewards_history: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.host.freeze_base_model()
        self.math_host.freeze()
        # Make sure the interrogator has its channel built.
        assert self.interrogator.use_channel, "Interrogator must be channel-mode"
        self.interrogator.ensure_initialized(
            self.host, seed_base=self.cfg.seed, math_host=self.math_host,
        )
        self.interrogator.channel.to(
            device=self.host.device, dtype=self.host.dtype,
        )
        # Snapshot the initial channel weights as the KL reference.
        ref = CrossModelChannel(
            a_to_m=_freeze_clone_projection(self.interrogator.channel.a_to_m),
            m_to_a=_freeze_clone_projection(self.interrogator.channel.m_to_a),
        )
        self.stack = _ChannelStack(live=self.interrogator.channel, ref=ref)

        # Optimizer over live channel params only.
        params = [p for p in self.stack.live.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=self.cfg.lr)

    # ---------- helpers ----------

    def _build_producer_broadcast(self, sc: Scenario) -> list[Broadcast]:
        """Synthetic producer broadcasts derived from scenario inputs.

        We don't run real Producer agents here — for this isolated training
        we just synthesize broadcasts directly from the scenario's raw
        inputs since only the Interrogator's channel is being trained.
        """
        out = []
        for inp in sc.inputs:
            if inp.source == "ohlcv":
                out.append(Broadcast(
                    id=f"prod.{sc.name}.{inp.id}",
                    tier="substrate",
                    agent_id=f"p.tickdelta.0",
                    agent_kind="tickdelta",
                    diet_tags=["from_tickdelta", "is_price_event"],
                    payload=inp.payload,
                    decoded_text=f"OHLCV {inp.payload}",
                    channel_embedding=torch.zeros(self.host.hidden_size).tolist(),
                    created_tick=0,
                ))
            elif inp.source == "quote_series":
                out.append(Broadcast(
                    id=f"prod.{sc.name}.{inp.id}",
                    tier="substrate",
                    agent_id=f"p.quant.0",
                    agent_kind="quote_series",
                    diet_tags=["from_quant_series", "is_quantitative"],
                    payload=inp.payload,
                    decoded_text=f"QUOTE_SERIES {inp.payload.get('ticker')}",
                    channel_embedding=torch.zeros(self.host.hidden_size).tolist(),
                    created_tick=0,
                ))
        return out

    def _solve_with_grad(
        self,
        question: str,
        channel: CrossModelChannel,
        compute_grad: bool,
    ) -> tuple[str, torch.Tensor]:
        """Project question through channel and forward Math. Returns
        (decoded_answer_text, summed_logprob_of_answer).

        The summed logprob is computed by re-feeding the generated tokens
        as labels under the math model with the same channel-projected
        prefix. Gradient flows through `channel` because `prefix_embeds`
        depends on it.
        """
        # Embed question in analyst space
        q_ids = self.host._tok(
            question, return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(self.host.device)
        q_emb_a = self.host._embed_layer(q_ids)[0].detach()  # [Q, hidden_a]

        ctx = torch.enable_grad() if compute_grad else torch.no_grad()
        with ctx:
            q_emb_m = channel.a_to_m(q_emb_a.to(
                device=channel.a_to_m.proj_in.weight.device,
                dtype=channel.a_to_m.proj_in.weight.dtype,
            ))

            # Build math system prefix in math space (text)
            sys_text = "Solve the following exactly. Output: ANSWER: <number>"
            sys_emb_m = self.math_host.text_to_input_embeds(sys_text).to(
                device=q_emb_m.device, dtype=q_emb_m.dtype,
            )
            prefix_embeds = torch.cat([sys_emb_m, q_emb_m], dim=0).unsqueeze(0)
            attn = torch.ones(
                prefix_embeds.shape[:2], dtype=torch.long, device=q_emb_m.device,
            )

            # Step 1: sample completion (no grad) to determine answer text
            with torch.no_grad():
                gen = self.math_host._model.generate(
                    inputs_embeds=prefix_embeds,
                    attention_mask=attn,
                    max_new_tokens=self.cfg.max_new_tokens_math,
                    do_sample=True,
                    temperature=self.cfg.sampling_temperature,
                    top_p=0.95,
                    repetition_penalty=1.05,
                    pad_token_id=self.math_host._tok.eos_token_id,
                )
                gen_tokens = gen[0]
                decoded = self.math_host._tok.decode(
                    gen_tokens, skip_special_tokens=True,
                ).strip()

            if gen_tokens.numel() == 0:
                zero = torch.zeros(1, device=q_emb_m.device, dtype=q_emb_m.dtype)
                if compute_grad:
                    zero = zero + 0.0 * channel.a_to_m.proj_in.weight.sum()
                return decoded, zero.squeeze()

            # Step 2: re-feed tokens with channel-projected prefix to get
            # logprobs that depend on the channel weights.
            gen_emb = self.math_host.embed_layer(gen_tokens.unsqueeze(0)).detach()
            full_embeds = torch.cat([prefix_embeds, gen_emb], dim=1)
            attn_full = torch.ones(
                full_embeds.shape[:2], dtype=torch.long, device=q_emb_m.device,
            )
            out = self.math_host._model(
                inputs_embeds=full_embeds,
                attention_mask=attn_full,
                use_cache=False,
            )
            logits = out.logits[0]
            P = prefix_embeds.shape[1]
            T = gen_tokens.shape[0]
            pred_logits = logits[P - 1: P - 1 + T]
            log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
            chosen = log_probs.gather(1, gen_tokens.unsqueeze(1)).squeeze(1)
            return decoded, chosen.sum()

    def _build_oracle_synthesis_target(self, sc: Scenario) -> Synthesis:
        """Construct the oracle synthesis the Interrogator should have
        produced. Uses scenario's predator_target ticker/direction/pct_move
        as the reference truth.
        """
        from .xml_schema import parse_prediction
        pred = parse_prediction(sc.predator_target or "")
        return Synthesis(
            kind="interrogator",
            ticker=pred.ticker,
            bias=pred.direction,
            pct_move=pred.pct_move,
            sigma_pct=pred.sigma_pct,
            confidence=pred.confidence,
        )

    # ---------- one step ----------

    async def step(self, sc: Scenario, tick: int) -> dict:
        target = self._build_oracle_synthesis_target(sc)
        candidates = self._build_producer_broadcast(sc)
        if not candidates:
            return {"loss": 0.0, "skipped": True, "reason": "no quant inputs"}

        # Phase 1: PLAN (no grad — host.generate_chat goes through
        # generate which is non-differentiable anyway)
        rendered = _render_inputs(candidates)
        try:
            plan = self.host.generate_chat(
                system=(
                    "You are a quantitative-analysis planner. Given the data,"
                    " output one ASK: line per math question. Be concise.\n"
                    "Format: ASK: <question with numbers inline>"
                ),
                user=f"DATA:\n{rendered}",
                max_new_tokens=128,
                temperature=0.0,
            )
        except Exception:
            plan = "ABSTAIN"
        questions = _extract_questions(plan)[:2]  # cap at 2 for training speed
        if not questions:
            return {"loss": 0.0, "skipped": True, "reason": "no questions"}

        # Phase 2: SOLVE (G times, with sampling) — for each completion in
        # the group, solve all questions; aggregate findings; synthesize.
        group_rewards: list[float] = []
        group_logprobs: list[torch.Tensor] = []
        group_decoded: list[str] = []

        for g in range(self.cfg.group_size):
            answers: list[str] = []
            total_lp = torch.zeros(
                1, device=self.host.device, dtype=self.host.dtype,
            )
            for q in questions:
                decoded_ans, lp = self._solve_with_grad(
                    q, self.stack.live, compute_grad=True,
                )
                answers.append(decoded_ans)
                total_lp = total_lp + lp.to(total_lp.device, total_lp.dtype)

            # Phase 3: SYNTHESIZE (no grad — token-based gen)
            # IMPORTANT: during cross-model channel training, we DO NOT
            # show producer data to the synthesizer. Only the math
            # findings (the questions and the math model's answers) are
            # available. This blocks the synthesizer from computing the
            # answer itself by reading producer payloads — which would
            # detach the reward signal from the channel's quality.
            findings_text = "\n".join(
                f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)
            )
            # Extract just the ticker hint from the questions (the math
            # findings should already contain it; this is a small concession
            # so the synthesizer knows what symbol to fill in).
            ticker_hint = "?"
            for q in questions:
                m = re.search(r"\b([A-Z]{2,5})\b", q)
                if m:
                    ticker_hint = m.group(1)
                    break
            try:
                synth_text = self.host.generate_chat(
                    system=(
                        "You are an interrogator synthesizer. You ONLY have"
                        " access to a math specialist's answers. Use the"
                        " math findings to fill in pct_move and bias."
                        " Output XML only.\n"
                        "Format:\n"
                        "<synthesis kind=\"interrogator\"><ticker>...</ticker>"
                        "<bias>up|down|flat</bias><pct_move>+X.XX</pct_move>"
                        "<confidence>0.7</confidence><abstain>false</abstain>"
                        "</synthesis>"
                    ),
                    user=(
                        f"TICKER: {ticker_hint}\n"
                        f"MATH FINDINGS:\n{findings_text}\n\n"
                        "Output the synthesis XML only."
                    ),
                    max_new_tokens=128,
                    temperature=0.0,
                )
            except Exception:
                synth_text = emit_synthesis(kind="interrogator", abstain=True)

            # Truncate at first </synthesis>
            for stop in ("</synthesis>",):
                idx = synth_text.find(stop)
                if idx >= 0:
                    synth_text = synth_text[: idx + len(stop)]
                    break

            parsed = parse_synthesis(synth_text)
            rew = reward_synthesis(parsed, target, self.cfg.reward_cfg)
            group_rewards.append(rew["total"])
            group_logprobs.append(total_lp.squeeze())
            group_decoded.append(synth_text[:200])

        rewards = torch.tensor(group_rewards, dtype=torch.float32)
        self.rewards_history.extend(rewards.tolist())

        if rewards.std().item() < 1e-6:
            advantages = torch.zeros_like(rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        # KL anchor: probably noisy because we'd need ref logprobs too.
        # For v1 we skip the KL term — small lr + grad clip should be
        # enough to keep things stable. The ref snapshot is preserved for
        # eval (we can compare current channel to init).
        total_loss = torch.zeros(
            1, device=self.host.device, dtype=self.host.dtype,
        )
        for i, lp in enumerate(group_logprobs):
            adv = float(advantages[i].item())
            if abs(adv) < 1e-6:
                continue
            T = max(1, lp.shape[0]) if lp.dim() > 0 else 1
            total_loss = total_loss + (-adv * lp / T)

        if total_loss.requires_grad:
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            params = [p for p in self.stack.live.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
            self.optimizer.step()

        return {
            "loss": float(total_loss.detach().item()),
            "rewards": group_rewards,
            "rewards_mean": float(rewards.mean().item()),
            "rewards_std": float(rewards.std().item()),
            "rewards_max": float(rewards.max().item()),
            "decoded_best": group_decoded[int(rewards.argmax().item())],
            "questions": questions,
        }

    async def eval_decode(self, sc: Scenario) -> dict:
        """Single greedy-pass eval; no training."""
        target = self._build_oracle_synthesis_target(sc)
        candidates = self._build_producer_broadcast(sc)
        if not candidates:
            return {"name": sc.name, "reward": 0.0, "decoded": "<no inputs>"}

        rendered = _render_inputs(candidates)
        plan = self.host.generate_chat(
            system="You are a quantitative-analysis planner. Output ASK: lines.",
            user=f"DATA:\n{rendered}", max_new_tokens=128, temperature=0.0,
        )
        questions = _extract_questions(plan)[:2]
        answers = []
        for q in questions:
            decoded_ans, _ = self._solve_with_grad(
                q, self.stack.live, compute_grad=False,
            )
            answers.append(decoded_ans)
        findings_text = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)
        )
        # Block producer-data leak: synthesizer only sees the math findings.
        ticker_hint = "?"
        for q in questions:
            m = re.search(r"\b([A-Z]{2,5})\b", q)
            if m:
                ticker_hint = m.group(1)
                break
        synth_text = self.host.generate_chat(
            system=(
                "You are an interrogator synthesizer. You ONLY have access"
                " to a math specialist's answers. Use the findings to fill"
                " in pct_move and bias. Output XML only."
            ),
            user=(
                f"TICKER: {ticker_hint}\n"
                f"MATH FINDINGS:\n{findings_text}\n\n"
                "Output XML."
            ),
            max_new_tokens=128, temperature=0.0,
        )
        for stop in ("</synthesis>",):
            idx = synth_text.find(stop)
            if idx >= 0:
                synth_text = synth_text[: idx + len(stop)]
                break
        parsed = parse_synthesis(synth_text)
        rew = reward_synthesis(parsed, target, self.cfg.reward_cfg)
        return {
            "name": sc.name,
            "reward": rew["total"],
            "breakdown": rew["breakdown"],
            "decoded": synth_text[:240],
            "questions": questions,
            "answers": answers,
        }
