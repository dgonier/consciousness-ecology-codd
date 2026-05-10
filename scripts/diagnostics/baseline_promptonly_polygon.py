"""Bare-Qwen3-4B prompt-only baseline on the Polygon benchmark.

Same prompt as scripts/diagnostics/baseline_promptonly_stocknet.py
(StockNet's bare baseline) — only difference is scenarios come from
build_polygon_scenarios() instead of build_stocknet_scenarios().

This is the apples-to-apples bar: what does ONE Qwen3-4B forward,
prompt-only, score on the same Polygon scenarios that the apex panel
saw?
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path("/home/dgonier/ecology_experiment/trophic")
sys.path.insert(0, str(ROOT))

# Load .env BEFORE importing trophic modules that read env at import time
import re
env = ROOT / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
            continue
        os.environ.setdefault(k, v)

import torch

from scripts.eval_stocknet import matthews_corrcoef
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.training.polygon_loader import build_polygon_scenarios
from trophic.training.xml_schema import parse_prediction


SYSTEM = (
    "You are a short-horizon market direction predictor. "
    "Read the evidence below — recent OHLCV bars and recent news — "
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


def _ticker_from_name(name: str) -> str:
    # polygon_AAPL_2026-04-23 → AAPL
    parts = name.split("_")
    return parts[1] if len(parts) >= 2 else "?"


def _build_prompt(sc) -> tuple[str, str]:
    parts = []
    ticker = _ticker_from_name(sc.name)
    for inp in sc.inputs:
        if inp.source == "ohlcv":
            bars = inp.payload.get("bars", [])
            parts.append(f"Recent OHLCV history for {ticker} ({len(bars)} bars):")
            for b in bars[-5:]:
                parts.append(
                    f"  open={b.get('open', 0):.6f}  high={b.get('high', 0):.6f}  "
                    f"low={b.get('low', 0):.6f}  close={b.get('close', 0):.6f}  "
                    f"vol={b.get('volume', 0)}"
                )
        elif inp.source == "press":
            body = (inp.payload.get("body") or "").strip()
            n = inp.payload.get("headline", "")
            parts.append(f"\nRecent news for {ticker} ({n}):")
            parts.append(body[:8000])

    user = "\n".join(parts) + "\n\n" + OUTPUT_FMT
    return SYSTEM, user


def _decode(host, system: str, user: str, max_new_tokens: int) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    tok = host._tok
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
            input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0, pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0, ids.shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


async def main() -> None:
    cfg = DEFAULT_CONFIG
    tickers_env = os.environ.get("POLY_TICKERS", "")
    tickers = (
        [t.strip() for t in tickers_env.split(",") if t.strip()]
        or ["AAPL", "MSFT", "GOOG", "AMZN", "JPM", "UNH", "V", "PG"]
    )
    target_window = int(os.environ.get("POLY_TARGET_DAYS", "21"))
    lookback = int(os.environ.get("POLY_LOOKBACK_DAYS", "75"))
    max_per = int(os.environ.get("POLY_MAX_PER_TICKER", "5"))
    max_new_tokens = int(os.environ.get("TROPHIC_EVAL_MAX_TOKENS", "256"))

    print(f"[poly_baseline] tickers={tickers}  target_window={target_window}  "
          f"lookback={lookback}  max_per_ticker={max_per}")
    host = ModelHost.get(cfg.model)
    print(f"[poly_baseline] base model loaded; hidden={host.hidden_size}")

    scenarios = build_polygon_scenarios(
        tickers=tickers, target_window_days=target_window,
        observation_lookback_days=lookback, max_per_ticker=max_per,
    )
    print(f"[poly_baseline] scenarios: {len(scenarios)}")

    tp = tn = fp = fn = 0
    abstained = 0
    parse_failed = 0

    for i, sc in enumerate(scenarios):
        system, user = _build_prompt(sc)
        text = _decode(host, system, user, max_new_tokens)
        target = parse_prediction(sc.predator_target or "")
        parsed = parse_prediction(text)
        target_dir = target.direction
        pred_dir = parsed.direction
        print(f"  [{sc.name}] target={target_dir} pred={pred_dir}")
        if pred_dir is None or pred_dir == "abstain":
            abstained += 1
            continue
        if pred_dir not in ("up", "down"):
            parse_failed += 1
            continue
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
    print(f"=== RESULTS [bare_qwen3_4b on Polygon] ===")
    print(f"  scenarios:    {n}")
    print(f"  decisions:    {decided} ({100*decided/max(n,1):.1f}%)")
    print(f"  abstained:    {abstained}")
    print(f"  parse_failed: {parse_failed}")
    print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"  ACCURACY:     {acc:.3f}")
    print(f"  MCC:          {mcc:+.3f}")


if __name__ == "__main__":
    asyncio.run(main())
