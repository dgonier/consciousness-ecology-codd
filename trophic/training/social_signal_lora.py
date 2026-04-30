"""LoRA SFT for the SocialSignal producer on financial-tweet sentiment.

Phase 2 of issue #10. Phase 1 (FinCoT-on-Interrogator) was a clean negative
result: more reasoning training did not break StockNet direction collapse,
because FinQA's reasoning style (calculate ratios from tables) is not what
installs directional priors on noisy stock-related tweets.

This trainer takes the architectural lesson — co-training Channels with a
LoRA-modified base is correct — and applies it to the input modality
StockNet's tweet stream actually carries.

Dataset: TimKoornstra/financial-tweets-sentiment
  - 38,091 rows, fields: tweet (str), sentiment (int: 0=Neutral, 1=Bullish, 2=Bearish)
  - License: MIT (redistributable weights)
  - Tweet-noisy (cashtags, URLs, RT-style)
  - Aggregates zeroshot/twitter-financial-news-sentiment + Kaggle + GitHub + IEEE

Training shape:
  - LoRA on Qwen3-4B's attention projections (Q/K/V/O), rank 16
  - SFT teacher-forcing on (tweet → "Sentiment: <label>") chat-template
  - Loss masked on user-side; active on assistant turn (the 1-token label is
    the load-bearing target; we don't ask for justifications since (a) we
    don't have ground-truth justifications and (b) the goal is to install
    a sentiment prior, not a verbose reasoner)

Lessons applied from FinCoT runs (#10 phase 1):
  - max_input_len=256, max_target_len=64 — tweets are short, label is 1 token.
    No long-tail rows means no PEFT autograd stall (the 4× repro from FinCoT).
  - Length pre-filter still in place defensively — drops zero rows in practice.
  - Early stopping on dev-loss plateau (patience=4 cycles, ~200 steps).

Once trained, the adapter loads on top of base Qwen3-4B at runtime so the
*same* base model serves all agents but the SocialSignal producer sees
sentiment-conditioned weights when reading a day's tweets.

Env vars:
  TROPHIC_SS_LORA_OUT — output dir (default checkpoints/social_signal_lora/seed{N})
  TROPHIC_SS_LORA_STEPS — training steps (default 600)
  TROPHIC_SS_LORA_LR — learning rate (default 5e-5)
  TROPHIC_SS_LORA_RANK — LoRA rank (default 16)
  TROPHIC_SEED — random seed (default 16)
"""
from __future__ import annotations

import functools
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

print = functools.partial(print, flush=True)

LABEL_NAMES = {0: "neutral", 1: "bullish", 2: "bearish"}


@dataclass
class SocialSignalLoRAConfig:
    lr: float = 5e-5
    grad_clip: float = 1.0
    steps: int = 600
    log_every: int = 10
    eval_every: int = 50
    seed: int = 16
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    # Tweets are short; label is 1 token. These caps will retain ~100% of rows.
    max_input_len: int = 256
    max_target_len: int = 64
    eval_holdout_size: int = 500  # tail of train held out for dev-loss only
    out_dir: str = ""             # populated from env at runtime
    es_patience: int = 4
    es_min_delta: float = 1e-3


def _build_chat_messages(row: dict) -> tuple[str, str]:
    """Returns (user_text, assistant_text) for the chat template.

    User: the raw tweet text.
    Assistant: a short, structured "Sentiment: <label>" — the load-bearing
    token is the label. We deliberately do NOT request a verbose justification
    (we don't have ground truth justifications, and the FinCoT phase showed
    that verbose-reasoning LoRAs displace structured output formats).
    """
    tweet = row["tweet"].strip()
    label_idx = int(row["sentiment"])
    label = LABEL_NAMES.get(label_idx, "neutral")
    user = f"Classify the directional sentiment of this stock tweet as bullish, bearish, or neutral.\n\nTweet: {tweet}"
    assistant = f"Sentiment: {label}"
    return user, assistant


def _encode_chat(tok, user_text: str, assistant_text: str,
                 max_input_len: int, max_target_len: int):
    msgs_user_only = [{"role": "user", "content": user_text}]
    user_ids = tok.apply_chat_template(
        msgs_user_only, tokenize=True, add_generation_prompt=True,
    )
    if len(user_ids) > max_input_len:
        user_ids = user_ids[-max_input_len:]
    asst_ids = tok(assistant_text, add_special_tokens=False).input_ids[:max_target_len]
    eos = tok.eos_token_id
    if eos is not None and (not asst_ids or asst_ids[-1] != eos):
        asst_ids = asst_ids + [eos]
    full_ids = list(user_ids) + list(asst_ids)
    labels = [-100] * len(user_ids) + list(asst_ids)
    return torch.tensor(full_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def _iter_rows(rows, seed: int) -> Iterator[dict]:
    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    while True:
        rng.shuffle(idxs)
        for i in idxs:
            yield rows[i]


def main():
    cfg = SocialSignalLoRAConfig(
        lr=float(os.environ.get("TROPHIC_SS_LORA_LR", "5e-5")),
        steps=int(os.environ.get("TROPHIC_SS_LORA_STEPS", "600")),
        seed=int(os.environ.get("TROPHIC_SEED", "16")),
        rank=int(os.environ.get("TROPHIC_SS_LORA_RANK", "16")),
    )
    out_dir_env = os.environ.get("TROPHIC_SS_LORA_OUT", "")
    cfg.out_dir = out_dir_env or f"checkpoints/social_signal_lora/seed{cfg.seed}"
    print(f"[ss-lora] cfg={cfg}")

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    print(f"[ss-lora] loading TimKoornstra/financial-tweets-sentiment")
    from datasets import load_dataset
    ds = load_dataset("TimKoornstra/financial-tweets-sentiment")
    rows = list(ds["train"])
    rng = random.Random(cfg.seed)
    rng.shuffle(rows)  # avoid systematic source-order bias from the aggregation
    holdout = rows[-cfg.eval_holdout_size:]
    rows = rows[: len(rows) - cfg.eval_holdout_size]
    print(f"[ss-lora] {len(rows)} train, {len(holdout)} holdout (pre-filter)")

    print(f"[ss-lora] loading Qwen3-4B + tokenizer")
    from trophic.config import DEFAULT_CONFIG
    from trophic.model_host import ModelHost
    host = ModelHost.get(DEFAULT_CONFIG.model)
    model = host._model
    tok = host._tok
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

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
    print(f"[ss-lora] post-filter: {len(rows)} train ({n_before - len(rows)} dropped),"
          f" {len(holdout)} holdout (max_input={cfg.max_input_len},"
          f" max_target={cfg.max_target_len})")

    print(f"[ss-lora] attaching LoRA (rank={cfg.rank}, target={cfg.target_modules})")
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
    print(f"[ss-lora] trainable {n_trainable:,} / total {n_total:,} ({100*n_trainable/n_total:.2f}%)")
    model.train()

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
    )

    device = host.device
    train_iter = _iter_rows(rows, cfg.seed)

    best_eval_loss = float("inf")
    eval_no_improve = 0
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
                for row in holdout[:64]:  # 64-row dev sample (tweets are short, fast)
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
                    print(f"[ss-lora] early stop at step {step}: "
                          f"{cfg.es_patience} dev-eval cycles without improvement")
                    break
            model.train()

    final_dir = cfg.out_dir + "_final"
    model.save_pretrained(final_dir)
    print(f"[ss-lora] final adapter → {final_dir}; best eval={best_eval_loss:.4f}")


if __name__ == "__main__":
    main()
