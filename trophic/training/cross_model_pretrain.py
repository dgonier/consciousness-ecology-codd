"""Pre-train the cross-model channel via embedding alignment.

The cold-start problem: a randomly initialized E_a→m projects analyst
hidden states into Math's space, but the result doesn't decode to any
particular tokens — so the math model produces random math content
rather than answering the actual question.

Fix: pre-train E_a→m against the math model's *natural* embedding for
the same question text. For a corpus of math-shaped questions:
  - Tokenize the question with Qwen3-4B's tokenizer → get analyst embeddings
  - Tokenize the same text with Math's tokenizer → get math embeddings
  - Train E_a→m so that proj(analyst_embed) ≈ math_embed

After this pre-training, the math model's first forward step on the
projected embeddings will produce a token distribution close to what it
would have produced on the *real* tokenized question. From there, RL
fine-tunes for actual reward signal.

This is just supervised regression — fast (no autoregressive sampling),
no gradient through the math model.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..cross_model_channel import CrossModelChannel
from ..math_host import MathHost
from ..model_host import ModelHost


# A diverse seed corpus of math-shaped questions covering the kinds the
# Interrogator will ask. Plus a few generic math sentences for breadth.
PRETRAIN_CORPUS = [
    # Percent change
    "Open price is 100.0 and close price is 102.5. What is the percent change from open to close?",
    "If a stock opens at 250 and closes at 247.5, what is the percentage move?",
    "Price was 50, now is 51.5. Compute the percentage increase.",
    "Initial value 1000, final value 980. Find the percent decrease.",
    # Statistics
    "Given the series [100.1, 100.3, 100.5, 100.7, 100.6, 100.8, 100.9, 101.0], compute the mean.",
    "Compute the standard deviation of [10, 12, 11, 13, 14, 12, 11].",
    "What is the median of [3, 7, 1, 9, 4, 8, 2]?",
    "Find the variance of [5, 5, 6, 7, 5, 8, 6, 7].",
    # Trends and slopes
    "For the time series y=[100, 101, 102, 103, 104] indexed by t=[0,1,2,3,4], what is the linear trend slope?",
    "Given quarterly values [50, 55, 60, 65], what is the average growth rate per quarter?",
    "If a series increases by 0.5 per step over 12 steps starting from 100, where does it end?",
    # Range and extrema
    "What is the range (max minus min) of [42, 38, 45, 41, 39, 46]?",
    "Identify the maximum value in [3.2, 4.1, 2.8, 5.5, 4.9].",
    "What is the difference between the largest and smallest values in [10, 20, 5, 30, 15]?",
    # OHLCV patterns
    "OHLCV bar: open=100, high=102, low=99, close=101. Compute (close - open) / open * 100.",
    "OHLCV bar with open=200, close=196. What is the percent change?",
    "Given high=105, low=95, what is the trading range?",
    # Volume z-scores
    "Recent volume is 5,000,000. Historical mean is 3,000,000 and std is 1,000,000. What is the volume z-score?",
    "Volume of 8M with mean 5M and std 2M — is this an outlier?",
    # Generic
    "Compute the sum of [1, 2, 3, 4, 5].",
    "What is 2.5 plus 1.7?",
    "Multiply 0.05 by 100.",
    "Divide 50 by 4.",
    "Round 1.2345 to two decimal places.",
]


@dataclass
class PretrainConfig:
    lr: float = 1e-3
    grad_clip: float = 1.0
    steps: int = 500
    log_every: int = 25
    seed: int = 1


def pretrain_cross_model_channel(
    channel: CrossModelChannel,
    host: ModelHost,
    math_host: MathHost,
    cfg: PretrainConfig | None = None,
) -> list[float]:
    """Train the channel's a_to_m direction via embedding-alignment MSE.

    The math model and Qwen3-4B both stay frozen. Only channel.a_to_m
    receives gradients.
    """
    cfg = cfg or PretrainConfig()
    rng = random.Random(cfg.seed)

    host.freeze_base_model()
    math_host.freeze()

    # Trainable params: just a_to_m
    params = [p for p in channel.a_to_m.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=cfg.lr)

    losses: list[float] = []

    print(f"[pretrain] {len(PRETRAIN_CORPUS)} sentences × {cfg.steps} steps,"
          f" lr={cfg.lr}")

    for step in range(1, cfg.steps + 1):
        text = rng.choice(PRETRAIN_CORPUS)

        # Analyst tokenization + embedding
        a_ids = host._tok(
            text, return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(host.device)
        a_emb = host._embed_layer(a_ids)[0].detach()  # [Qa, hidden_a]

        # Math tokenization + embedding (the regression target)
        m_ids = math_host._tok(
            text, return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(math_host.device)
        m_emb_target = math_host.embed_layer(m_ids)[0].detach().float()  # [Qm, hidden_m]

        # The two tokenizations may produce different token counts. Resample
        # one to match the other via interpolation along the sequence dim.
        Qa = a_emb.shape[0]
        Qm = m_emb_target.shape[0]
        if Qa != Qm:
            # Linear interpolation on a_emb so its sequence length matches Qm
            a_emb_resampled = F.interpolate(
                a_emb.float().T.unsqueeze(0),  # [1, hidden_a, Qa]
                size=Qm, mode="linear", align_corners=False,
            ).squeeze(0).T  # [Qm, hidden_a]
            a_emb_input = a_emb_resampled
        else:
            a_emb_input = a_emb.float()

        # Project through channel.a_to_m
        a_emb_input = a_emb_input.to(
            device=channel.a_to_m.proj_in.weight.device,
            dtype=channel.a_to_m.proj_in.weight.dtype,
        )
        projected = channel.a_to_m(a_emb_input)  # [Qm, hidden_m]

        # MSE against math's natural embedding
        m_emb_target_dev = m_emb_target.to(
            device=projected.device, dtype=projected.dtype,
        )
        loss = F.mse_loss(projected, m_emb_target_dev)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optim.step()

        loss_val = float(loss.detach().item())
        losses.append(loss_val)

        if step % cfg.log_every == 0:
            recent = sum(losses[-cfg.log_every:]) / cfg.log_every
            print(f"[pretrain step {step:4d}] loss={loss_val:.6f}"
                  f"  recent_avg={recent:.6f}")

    return losses
