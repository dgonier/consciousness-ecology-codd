"""LoRA SFT for the Interrogator on the FinCoT dataset (Fino1).

Phase 1 of issue #10. Per the dataset deep-dive, FinCoT is the
highest-leverage dataset for the trophic system's #8 direction-collapse
problem because it installs explicit, verifiable, multi-step conditioning
of an answer on evidence — exactly the reasoning skill the Interrogator
needs.

Dataset: TheFinAI/Fino1_Reasoning_Path_FinQA
  - 5,499 rows, fields: Open-ended Verifiable Question, Ground-True Answer,
    Complex_CoT, Response
  - License: CC-BY 4.0 (redistributable weights)
  - Source: arXiv 2502.08127 (Fino1)

Training shape:
  - LoRA on Qwen3-4B's attention projections (Q/K/V/O), rank 16
  - SFT teacher-forcing on (question + CoT + response) sequence
  - Lossed against assistant turn only (CoT + response)
  - Result: a small adapter (~50-100 MB) that loads on top of base Qwen3-4B

Once trained, the adapter is loaded by the Interrogator herbivore at
runtime so the *same* base Qwen3-4B serves all agents but the Interrogator
gets reasoning-conditioned weights for its plan + synthesize phases.

Env vars:
  TROPHIC_FINCOT_LORA_OUT — output dir for adapter (default checkpoints/interrogator_fincot_lora/seedN)
  TROPHIC_FINCOT_LORA_STEPS — training steps (default 600)
  TROPHIC_FINCOT_LORA_LR — learning rate (default 5e-5; same as existing Interrogator LoRA)
  TROPHIC_FINCOT_LORA_RANK — LoRA rank (default 16)
  TROPHIC_SEED — random seed (default 11; not 9 since that's our EnvStream IPO)
"""
from __future__ import annotations

import functools
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F

print = functools.partial(print, flush=True)


@dataclass
class FinCoTLoRAConfig:
    """LoRA SFT config for FinCoT-on-Interrogator."""
    lr: float = 5e-5
    grad_clip: float = 1.0
    steps: int = 600
    log_every: int = 10
    eval_every: int = 50
    seed: int = 11
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    # NOTE: 2048 hit a long-context PEFT autograd stall ~step 80; 1024 also
    # stalled. Profiled distribution: user-side med=988, p95=1574, max=2577;
    # asst-side med=413, p95=685, max=1199. Set caps just above median so the
    # pathological long-tail rows are pre-filtered (the actual stall trigger),
    # while keeping ~half of the dataset trainable rather than the 7% we got
    # at 768/384.
    max_input_len: int = 1280
    max_target_len: int = 720
    eval_holdout_size: int = 100  # tail of train set held out for eval-loss only
    out_dir: str = ""           # populated from env at runtime
    # Early stopping on dev plateau:
    es_patience: int = 4  # consecutive eval cycles without improvement before stop
    es_min_delta: float = 1e-3  # eval_loss must improve by this much to count


def _build_chat_messages(row: dict) -> tuple[str, str]:
    """Returns (user_text, assistant_text) for the chat template.

    The user turn is the financial question; the assistant turn is the CoT
    followed by the final response. The model learns to produce both —
    reasoning + answer — given a question.
    """
    user = row["Open-ended Verifiable Question"].strip()
    cot = row.get("Complex_CoT", "").strip()
    resp = row.get("Response", "").strip()
    if cot and resp:
        assistant = f"<thinking>\n{cot}\n</thinking>\n\n{resp}"
    elif resp:
        assistant = resp
    else:
        assistant = cot or ""
    return user, assistant


def _encode_chat(tok, user_text: str, assistant_text: str,
                 max_input_len: int, max_target_len: int):
    """Build (input_ids, labels) where loss is masked on the user-side tokens
    and active on assistant tokens only.
    """
    msgs_user_only = [{"role": "user", "content": user_text}]
    user_ids = tok.apply_chat_template(
        msgs_user_only, tokenize=True, add_generation_prompt=True,
    )
    if len(user_ids) > max_input_len:
        # Truncate the user text from the left so we keep the question stem
        # right before the assistant turn marker.
        user_ids = user_ids[-max_input_len:]
    asst_ids = tok(assistant_text, add_special_tokens=False).input_ids[:max_target_len]
    eos = tok.eos_token_id
    if eos is not None and (not asst_ids or asst_ids[-1] != eos):
        asst_ids = asst_ids + [eos]
    full_ids = list(user_ids) + list(asst_ids)
    labels = [-100] * len(user_ids) + list(asst_ids)
    return torch.tensor(full_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def _iter_rows(rows, seed: int) -> Iterator[dict]:
    """Infinite shuffled iterator over a list of row dicts."""
    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    while True:
        rng.shuffle(idxs)
        for i in idxs:
            yield rows[i]


def main():
    # ---- config ----
    cfg = FinCoTLoRAConfig(
        lr=float(os.environ.get("TROPHIC_FINCOT_LORA_LR", "5e-5")),
        steps=int(os.environ.get("TROPHIC_FINCOT_LORA_STEPS", "600")),
        seed=int(os.environ.get("TROPHIC_SEED", "11")),
        rank=int(os.environ.get("TROPHIC_FINCOT_LORA_RANK", "16")),
    )
    out_dir_env = os.environ.get("TROPHIC_FINCOT_LORA_OUT", "")
    cfg.out_dir = out_dir_env or f"checkpoints/interrogator_fincot_lora/seed{cfg.seed}"
    print(f"[fincot-lora] cfg={cfg}")

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # ---- dataset ----
    print(f"[fincot-lora] loading TheFinAI/Fino1_Reasoning_Path_FinQA")
    from datasets import load_dataset
    ds = load_dataset("TheFinAI/Fino1_Reasoning_Path_FinQA")
    rows = list(ds["train"])
    holdout = rows[-cfg.eval_holdout_size:]
    rows = rows[: len(rows) - cfg.eval_holdout_size]
    print(f"[fincot-lora] {len(rows)} train, {len(holdout)} holdout (pre-filter)")

    # ---- model ----
    print(f"[fincot-lora] loading Qwen3-4B + tokenizer")
    from trophic.config import DEFAULT_CONFIG
    from trophic.model_host import ModelHost
    host = ModelHost.get(DEFAULT_CONFIG.model)
    model = host._model
    tok = host._tok
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Pre-filter rows whose tokenized chat-template input would exceed the cap.
    # The PEFT autograd stall we saw at step 50-80 was triggered by a specific
    # long-context row whose forward pass hung silently. Drop those rows
    # entirely so the optimizer only sees lengths we know are safe.
    def _row_fits(row: dict) -> bool:
        u, a = _build_chat_messages(row)
        if not a:
            return False
        try:
            uids = tok.apply_chat_template(
                [{"role": "user", "content": u}],
                tokenize=True, add_generation_prompt=True,
            )
            aids = tok(a, add_special_tokens=False).input_ids
        except Exception:
            return False
        return len(uids) <= cfg.max_input_len and len(aids) <= cfg.max_target_len

    n_before = len(rows)
    rows = [r for r in rows if _row_fits(r)]
    holdout = [r for r in holdout if _row_fits(r)]
    print(f"[fincot-lora] post-filter: {len(rows)} train ({n_before - len(rows)} dropped),"
          f" {len(holdout)} holdout (max_input={cfg.max_input_len},"
          f" max_target={cfg.max_target_len})")

    # ---- LoRA ----
    print(f"[fincot-lora] attaching LoRA (rank={cfg.rank}, target={cfg.target_modules})")
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        target_modules=list(cfg.target_modules),
        lora_dropout=cfg.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[fincot-lora] trainable {n_trainable:,} / total {n_total:,} ({100*n_trainable/n_total:.2f}%)")
    model.train()

    # ---- optimizer ----
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
    )

    # ---- training loop ----
    device = host.device
    dtype = host.dtype
    train_iter = _iter_rows(rows, cfg.seed)

    best_eval_loss = float("inf")
    eval_no_improve = 0  # consecutive eval cycles without improvement → early stop
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    for step in range(1, cfg.steps + 1):
        row = next(train_iter)
        user_text, asst_text = _build_chat_messages(row)
        if not asst_text:
            continue
        input_ids, labels = _encode_chat(
            tok, user_text, asst_text,
            max_input_len=cfg.max_input_len, max_target_len=cfg.max_target_len,
        )
        input_ids = input_ids.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            cfg.grad_clip,
        )
        optim.step()
        optim.zero_grad(set_to_none=True)

        if step % cfg.log_every == 0:
            print(f"[step {step:4d}] train_loss={loss.item():.4f}")

        if step % cfg.eval_every == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for row in holdout[:32]:  # quick eval over 32 rows
                    user_text, asst_text = _build_chat_messages(row)
                    if not asst_text:
                        continue
                    input_ids, labels = _encode_chat(
                        tok, user_text, asst_text,
                        max_input_len=cfg.max_input_len, max_target_len=cfg.max_target_len,
                    )
                    input_ids = input_ids.unsqueeze(0).to(device)
                    labels = labels.unsqueeze(0).to(device)
                    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    losses.append(out.loss.item())
            eval_loss = sum(losses) / max(1, len(losses))
            print(f"  EVAL_LOSS: mean={eval_loss:.4f}  (over {len(losses)} rows)")
            if eval_loss < best_eval_loss - cfg.es_min_delta:
                best_eval_loss = eval_loss
                eval_no_improve = 0
                model.save_pretrained(cfg.out_dir)
                print(f"  [ckpt] new best ({eval_loss:.4f}) → {cfg.out_dir}")
            else:
                eval_no_improve += 1
                print(f"  [es] no improvement ({eval_no_improve}/{cfg.es_patience});"
                      f" best={best_eval_loss:.4f}")
                if eval_no_improve >= cfg.es_patience:
                    print(f"[fincot-lora] early stop at step {step}: "
                          f"{cfg.es_patience} dev-eval cycles without improvement")
                    break
            model.train()

    # final save
    final_dir = cfg.out_dir + "_final"
    model.save_pretrained(final_dir)
    print(f"[fincot-lora] final adapter → {final_dir}; best eval={best_eval_loss:.4f}")


if __name__ == "__main__":
    main()
