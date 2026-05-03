"""Prompt-only Qwen3-4B baseline on StockNet — second-opinion advice.

Both reviewers (GPT-5.2 + Gemini 3.1 Pro) flagged that we have NEVER run a
no-trophic-stack baseline. Without it we can't tell whether the constant-
direction collapse is "the architecture is broken" or "the benchmark is
uninformative."

This script:
  1. Loads frozen Qwen3-4B (no Channels, no LoRA, no trophic anything).
  2. For each StockNet scenario, builds a structured prompt from the same
     raw inputs (OHLCV history + concatenated tweets).
  3. Asks Qwen3-4B for an XML <prediction> directly.
  4. Parses + scores ACC and MCC against the same labels.

Decisive: if MCC > 0 here, our trophic stack is destroying signal. If MCC
= 0, the dataset can't be learned from at this scale, and any architecture
will collapse to majority class.

Env vars:
  STOCKNET_TICKERS — default top-5 (AAPL,GOOG,MSFT,AMZN,JPM)
  STOCKNET_MAX_PER_TICKER — default 10
  TROPHIC_EVAL_MAX_TOKENS — default 96
"""
from __future__ import annotations
import asyncio, builtins, functools, os, sys
from pathlib import Path
builtins.print = functools.partial(builtins.print, flush=True)
print = builtins.print

ROOT = Path("/home/dgonier/ecology_experiment/trophic")
sys.path.insert(0, str(ROOT))

import torch

from scripts.eval_stocknet import matthews_corrcoef
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.training.stocknet_loader import build_stocknet_scenarios
from trophic.training.xml_schema import parse_prediction


SYSTEM = (
    "You are a short-horizon market direction predictor. "
    "Read the evidence below — recent OHLCV bars and same-day tweets — "
    "and output ONLY a single XML prediction tag, no other text, no thinking, no preamble. "
    "Predict whether the stock's next-day close will be UP or DOWN. "
    "Be specific (use the ticker shown in the evidence)."
)

OUTPUT_FMT = (
    "Respond with EXACTLY ONE LINE matching this format and NOTHING else "
    "(no <think>, no reasoning, no explanation):\n"
    "<prediction><ticker>SYMBOL</ticker>"
    "<direction>up|down</direction>"
    "<horizon_min>1440</horizon_min>"
    "<confidence>0.55</confidence></prediction>"
)


def _build_prompt(sc) -> tuple[str, str]:
    """Return (system_text, user_text) for the chat template."""
    parts = []
    ticker = None
    for inp in sc.inputs:
        if inp.source == "ohlcv":
            bars = inp.payload.get("bars", [])
            ticker = ticker or sc.name.split("_")[2]
            parts.append(f"Recent OHLCV history for {ticker} ({len(bars)} bars):")
            for b in bars[-5:]:
                parts.append(
                    f"  open={b.get('open', 0):.6f}  high={b.get('high', 0):.6f}  "
                    f"low={b.get('low', 0):.6f}  close={b.get('close', 0):.6f}  "
                    f"vol={b.get('volume', 0)}"
                )
        elif inp.source == "press":
            t = inp.payload.get("ticker") or sc.name.split("_")[2]
            ticker = ticker or t
            body = (inp.payload.get("body") or "").strip()
            n_tweets = inp.payload.get("headline", "")
            parts.append(f"\nSame-day tweets for {t} ({n_tweets}):")
            parts.append(body[:2000])  # cap

    user = "\n".join(parts) + "\n\n" + OUTPUT_FMT
    return SYSTEM, user


def _decode(host, system: str, user: str, max_new_tokens: int) -> str:
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    tok = host._tok
    # Qwen3-4B has reasoning baked in — disable thinking explicitly.
    try:
        ids = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            enable_thinking=False,
        ).to(host.device)
    except TypeError:
        ids = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        ).to(host.device)
    attn = torch.ones_like(ids)
    with torch.no_grad():
        out = host._model.generate(
            input_ids=ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0, ids.shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


async def main() -> None:
    cfg = DEFAULT_CONFIG
    tickers_env = os.environ.get("STOCKNET_TICKERS", "")
    tickers = [t.strip() for t in tickers_env.split(",") if t.strip()] or ["AAPL", "GOOG", "MSFT", "AMZN", "JPM"]
    max_per = int(os.environ.get("STOCKNET_MAX_PER_TICKER", "10"))
    max_new_tokens = int(os.environ.get("TROPHIC_EVAL_MAX_TOKENS", "256"))

    print(f"[promptonly] tickers={tickers}  max_per_ticker={max_per}  max_new_tokens={max_new_tokens}")
    host = ModelHost.get(cfg.model)
    print(f"[promptonly] base model loaded; hidden={host.hidden_size}")

    scenarios = build_stocknet_scenarios(
        split="test", tickers=tickers, max_per_ticker=max_per,
    )
    print(f"[promptonly] scenarios: {len(scenarios)}")

    tp = tn = fp = fn = 0
    abstained = 0
    parse_failed = 0
    per_ticker: dict[str, dict[str, int]] = {t: {"correct": 0, "total": 0} for t in tickers}
    sample_decodes = []

    for i, sc in enumerate(scenarios):
        system, user = _build_prompt(sc)
        text = _decode(host, system, user, max_new_tokens)
        target = parse_prediction(sc.predator_target or "")
        parsed = parse_prediction(text)
        target_dir = target.direction
        pred_dir = parsed.direction
        sc_ticker = sc.name.split("_")[2]
        per_ticker[sc_ticker]["total"] += 1
        if i < 5:
            sample_decodes.append((sc.name, target_dir, pred_dir, text[:200].replace("\n", " ")))
        # 2026-05-03: log every scenario for ensemble parsing.
        print(f"  [{sc.name}] target={target_dir} pred={pred_dir}")
        if pred_dir is None or pred_dir == "abstain":
            abstained += 1
            continue
        if pred_dir not in ("up", "down"):
            parse_failed += 1
            continue
        if pred_dir == target_dir:
            per_ticker[sc_ticker]["correct"] += 1
        if target_dir == "up" and pred_dir == "up":
            tp += 1
        elif target_dir == "down" and pred_dir == "down":
            tn += 1
        elif target_dir == "down" and pred_dir == "up":
            fp += 1
        elif target_dir == "up" and pred_dir == "down":
            fn += 1

    n = len(scenarios)
    decided = tp + tn + fp + fn
    acc = (tp + tn) / max(decided, 1)
    mcc = matthews_corrcoef(tp, tn, fp, fn)

    print()
    print(f"=== RESULTS [prompt_only_qwen3_4b] ===")
    print(f"  scenarios:       {n}")
    print(f"  decisions:       {decided} ({100*decided/max(n,1):.1f}%)")
    print(f"  abstained:       {abstained}")
    print(f"  parse_failed:    {parse_failed}")
    print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"  ACCURACY:        {acc:.3f}")
    print(f"  MCC:             {mcc:+.3f}")
    print()
    print(f"  per-ticker:")
    for t in tickers:
        c, tot = per_ticker[t]["correct"], per_ticker[t]["total"]
        if tot:
            print(f"    {t}: {c}/{tot} = {c/tot:.3f}")
    print()
    print(f"  sample decodes (first 5):")
    for name, td, pd, txt in sample_decodes:
        print(f"    [{name}] target={td} pred={pd}")
        print(f"      raw: {txt}")


if __name__ == "__main__":
    asyncio.run(main())
