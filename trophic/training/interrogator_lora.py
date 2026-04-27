"""LoRA SFT training for the Interrogator's planner + synthesizer.

The Interrogator runs two Qwen3-4B forward passes per scenario:
  1. Plan phase: emit ASK: lines from rendered producer data
  2. Synthesize phase: emit XML from question/answer pairs

Both share the same base model. We attach a single LoRA adapter to
Qwen3-4B targeting attention Q/K/V/O projections (rank 16). Training
signal: the Interrogator's final XML synthesis should match the
scenario's `interrogator_target`.

The LoRA covers both planner *and* synthesizer because they're calls
to the same model — the forward pass at inference time is the same
shape regardless of system prompt. We rely on the system prompt to
condition behavior between phases, and the LoRA adapts the model's
response to those system prompts.

When LoRA is active, *all* Qwen3-4B forwards go through the adapter —
including the herbivore + predator forwards in the trophic stack.
That's a v1 simplification; v2 would use per-species adapters.
"""
from __future__ import annotations

import asyncio
import functools
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class InterrogatorLoRAConfig:
    """LoRA SFT config for the Interrogator."""
    lr: float = 5e-5
    grad_clip: float = 1.0
    steps: int = 400
    log_every: int = 25
    eval_every: int = 100
    seed: int = 1
    # LoRA hyperparams
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    # When True, the planner phase also gets a CE loss against an oracle
    # plan (questions). When False, only the synthesizer is trained.
    train_planner: bool = False


def attach_lora(model, cfg: InterrogatorLoRAConfig):
    """Wrap the model with a LoRA adapter and freeze base weights."""
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        target_modules=list(cfg.target_modules),
        lora_dropout=cfg.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    # Sanity: only LoRA params should be trainable.
    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in peft_model.parameters())
    return peft_model, n_trainable, n_total


def _synthesizer_forward_loss(
    host,
    target_text: str,
    findings_text: str,
    ticker_hint: str,
) -> torch.Tensor:
    """One CE forward for the synthesizer phase.

    Builds the exact same prompt the live interrogator uses for synthesis
    (TICKER + MATH FINDINGS), runs Qwen forward with target tokens
    appended as labels, computes CE on the target portion only.

    Gradients flow through the LoRA params. Base model frozen.
    """
    device = host.device
    dtype = host.dtype
    tok = host._tok

    system = (
        "You are an interrogator synthesizer. You ONLY have access to a"
        " math specialist's answers. Use the findings to fill in pct_move"
        " and bias. Output XML only.\n"
        "Format:\n"
        "<synthesis kind=\"interrogator\"><ticker>...</ticker>"
        "<bias>up|down|flat</bias><pct_move>+X.XX</pct_move>"
        "<confidence>0.7</confidence><abstain>false</abstain>"
        "</synthesis>"
    )
    user = (
        f"TICKER: {ticker_hint}\n"
        f"MATH FINDINGS:\n{findings_text}\n\n"
        "Output the synthesis XML only."
    )

    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    target_ids = tok(target_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    eos = torch.tensor([tok.eos_token_id], device=device)
    target_ids = torch.cat([target_ids, eos])

    full_ids = torch.cat([prompt_ids, target_ids]).unsqueeze(0)
    attn = torch.ones_like(full_ids)
    out = host._model(input_ids=full_ids, attention_mask=attn, use_cache=False)
    logits = out.logits[0]
    P = prompt_ids.shape[0]
    T = target_ids.shape[0]
    pred_logits = logits[P - 1 : P - 1 + T, :]
    return F.cross_entropy(pred_logits, target_ids)


def _build_findings_text(sc) -> tuple[str, str]:
    """Build a *teacher* findings text and ticker hint from the scenario.

    We don't actually run the math node during LoRA training — that would
    be slow and noisy. Instead we synthesize what the math node *should*
    have produced, derived from the scenario's interrogator_target which
    already contains the right pct_move.
    """
    import re
    target = sc.interrogator_target or ""
    m = re.search(r"<ticker>([A-Z]+)</ticker>", target)
    ticker = m.group(1) if m else "?"
    m = re.search(r"<pct_move>([+-]?[0-9.]+)</pct_move>", target)
    pct_str = m.group(1) if m else "0.00"
    m = re.search(r"<bias>(up|down|flat)</bias>", target)
    bias = m.group(1) if m else "flat"

    findings = (
        f"Q: What is the percent change for {ticker}?\n"
        f"A: {pct_str}\n"
        f"Q: What direction is the move?\n"
        f"A: {bias}"
    )
    return findings, ticker


def lora_train_synthesizer(
    host,
    train,
    eval_,
    cfg: InterrogatorLoRAConfig,
    ckpt_path: str,
    print_fn=print,
) -> dict:
    """Run LoRA-SFT on the synthesizer phase.

    Args:
      host: ModelHost with Qwen3-4B already loaded (will be LoRA-wrapped)
      train, eval_: scenario lists
      cfg: LoRA config
      ckpt_path: where to save the LoRA adapter
    """
    # Filter to scenarios with non-abstain interrogator targets — the
    # abstain ones are short and don't teach much; we'll add them back
    # via random selection.
    train_with_target = [
        sc for sc in train
        if sc.interrogator_target and "abstain>true" not in sc.interrogator_target
    ]
    print_fn(f"[lora] training scenarios with interrogator content: {len(train_with_target)}")

    # Attach LoRA
    peft_model, n_trainable, n_total = attach_lora(host._model, cfg)
    host._model = peft_model
    print_fn(f"[lora] trainable params: {n_trainable:,} / {n_total:,}"
             f" ({100*n_trainable/n_total:.2f}%)")

    rng = random.Random(cfg.seed)
    optimizer = torch.optim.Adam(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=cfg.lr,
    )

    losses: list[float] = []
    eval_history: list[float] = []
    best_eval = float("inf")

    print_fn("\n=== Interrogator LoRA SFT ===")
    for step in range(1, cfg.steps + 1):
        sc = rng.choice(train_with_target)
        findings, ticker = _build_findings_text(sc)
        loss = _synthesizer_forward_loss(
            host=host,
            target_text=sc.interrogator_target,
            findings_text=findings,
            ticker_hint=ticker,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in peft_model.parameters() if p.requires_grad],
            cfg.grad_clip,
        )
        optimizer.step()
        loss_val = float(loss.detach().item())
        losses.append(loss_val)

        if step % cfg.log_every == 0:
            recent = sum(losses[-cfg.log_every:]) / cfg.log_every
            print_fn(f"[lora step {step:4d}] loss={loss_val:.3f}"
                     f" recent_avg={recent:.3f}")

        if step % cfg.eval_every == 0:
            eval_loss = _eval_synthesizer(host, eval_, print_fn=print_fn)
            eval_history.append(eval_loss)
            print_fn(f"[lora EVAL @ step {step}] mean_loss={eval_loss:.4f}")
            if eval_loss < best_eval:
                best_eval = eval_loss
                # Save LoRA adapter
                peft_model.save_pretrained(ckpt_path)
                print_fn(f"[lora ckpt] new best ({eval_loss:.4f}) → {ckpt_path}")

    return {
        "best_eval": best_eval,
        "n_trainable": n_trainable,
        "losses": losses,
        "eval_history": eval_history,
    }


def _eval_synthesizer(host, eval_, print_fn=print) -> float:
    """Compute mean teacher-forcing CE on eval scenarios."""
    losses = []
    with torch.no_grad():
        for sc in eval_:
            if not sc.interrogator_target or "abstain>true" in sc.interrogator_target:
                continue
            findings, ticker = _build_findings_text(sc)
            loss = _synthesizer_forward_loss(
                host=host,
                target_text=sc.interrogator_target,
                findings_text=findings,
                ticker_hint=ticker,
            )
            losses.append(float(loss.item()))
    return sum(losses) / max(1, len(losses))
