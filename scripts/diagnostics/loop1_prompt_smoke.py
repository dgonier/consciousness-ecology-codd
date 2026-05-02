"""Loop 1: prompt + parser smoke (~30s).

Tests prompt design alone with no training. Loads frozen Qwen3-4B, runs
the predator's role+query prompt on 5 StockNet scenarios, parses, reports
direction. If this fails the prompt is wrong before training even starts.

What this catches without burning training cycles:
- Prompt-parroting (model echoes the schema instead of filling it)
- Parser misses on the format the model emits
- Format spec is ambiguous → model picks first option
- Tokenization of the role prompt collapses important tokens

Usage:
    .venv/bin/python -u scripts/diagnostics/loop1_prompt_smoke.py
"""
from __future__ import annotations
import asyncio, functools, sys
from pathlib import Path

print = functools.partial(print, flush=True)
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch

from trophic.agents.predator import ROLE_PROMPTS, QUERIES
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.training.stocknet_loader import build_stocknet_scenarios
from trophic.training.xml_schema import parse_prediction


def render_curated_slot(sc) -> str:
    parts = []
    for inp in sc.inputs:
        if inp.source == "ohlcv":
            bars = inp.payload.get("bars", [])
            ticker = sc.name.split("_")[2]
            parts.append(f"OHLCV {ticker} ({len(bars)} bars):")
            for b in bars[-5:]:
                parts.append(
                    f"  open={b.get('open',0):.4f} high={b.get('high',0):.4f} "
                    f"low={b.get('low',0):.4f} close={b.get('close',0):.4f}"
                )
        elif inp.source == "press":
            t = inp.payload.get("ticker") or sc.name.split("_")[2]
            body = (inp.payload.get("body") or "").strip()[:1200]
            parts.append(f"Tweets {t}: {body}")
    return "\n".join(parts)


async def main():
    cfg = DEFAULT_CONFIG
    host = ModelHost.get(cfg.model)
    role = ROLE_PROMPTS["short_horizon"]
    query = QUERIES["short_horizon"]

    print(f"[loop1] role: {role[:80]}...")
    print(f"[loop1] query (first 150 chars): {query[:150]}")
    print()

    import os as _os
    n_per = int(_os.environ.get("LOOP1_N_PER_TICKER", "10"))
    tickers = _os.environ.get("LOOP1_TICKERS", "AAPL,GOOG,MSFT,AMZN,JPM").split(",")
    scens = build_stocknet_scenarios(split="test", tickers=tickers, max_per_ticker=n_per)
    n_correct = 0
    n_decided = 0
    decoded_set = set()

    for sc in scens:
        target = parse_prediction(sc.predator_target or "")
        curated = render_curated_slot(sc)
        prompt = f"{role}\n\n{curated}\n\n{query}"

        # Use chat template with thinking disabled (Qwen3-4B's <think> mode
        # otherwise produces stream-of-consciousness instead of structured output)
        msgs = [{"role": "user", "content": prompt}]
        try:
            ids = host._tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", enable_thinking=False,
            ).to(host.device)
        except TypeError:
            ids = host._tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_tensors="pt",
            ).to(host.device)
        am = torch.ones_like(ids)
        with torch.no_grad():
            out = host._model.generate(
                input_ids=ids, attention_mask=am, max_new_tokens=256,
                do_sample=False, pad_token_id=host._tok.eos_token_id,
            )
        new = out[0, ids.shape[1]:]
        decoded = host._tok.decode(new, skip_special_tokens=True)
        parsed = parse_prediction(decoded)
        d = parsed.direction
        decoded_set.add(decoded[:60])

        decided = d in ("up", "down")
        correct = d == target.direction if decided else False
        n_decided += int(decided)
        n_correct += int(correct)
        print(f"[{sc.name}] target={target.direction} pred={d}")
        print(f"  decoded[:200]: {decoded[:200]!r}")

    # Compute MCC over decided cases
    tp = tn = fp = fn = 0
    for sc, (tgt, pred) in zip(scens, [(parse_prediction(s.predator_target or "").direction, p)
                                          for s, p in zip(scens, [None]*len(scens))]):
        pass  # placeholder; we already have per-scenario printouts above

    print()
    print(f"[loop1] decided={n_decided}/{len(scens)} correct={n_correct}/{len(scens)} "
          f"unique_decodes={len(decoded_set)}")
    decision_rate = n_decided / max(len(scens), 1)
    if decision_rate < 0.5:
        print(f"[loop1] FAIL: only {decision_rate:.0%} decisions parsed — parser/format broken")
        return 1
    if len(decoded_set) <= 2 and len(scens) > 5:
        print(f"[loop1] FAIL: only {len(decoded_set)} unique decoded strings across {len(scens)} scenarios — constant collapse")
        return 1
    print(f"[loop1] PASS: model produces parseable output across {len(scens)} scenarios")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
