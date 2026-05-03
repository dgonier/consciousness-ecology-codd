"""Loop 1 baseline with constrained decode: bare Qwen + new apex prompt.

Mirrors loop1_prompt_smoke.py but instead of free-generating 256 tokens
and parsing, we do a single forward with the apex-style prompt and read
the next-token logit, restricting to {up, down} ids. Tells us whether
bare Qwen has any directional signal in its first-token distribution
under this prompt format.

Usage:
    .venv/bin/python -u scripts/diagnostics/loop1_constrained.py
"""
from __future__ import annotations
import asyncio, functools, math, os, sys
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

    n_per = int(os.environ.get("LOOP1_N_PER_TICKER", "20"))
    tickers = os.environ.get("LOOP1_TICKERS", "AAPL,GOOG,MSFT,AMZN,JPM").split(",")
    scens = build_stocknet_scenarios(split="test", tickers=tickers, max_per_ticker=n_per)

    tok = host._tok
    cand_strs = [" up", "up", " UP", "UP", " down", "down", " DOWN", "DOWN"]
    up_ids: set[int] = set()
    down_ids: set[int] = set()
    for s in cand_strs:
        enc = tok(s, add_special_tokens=False).input_ids
        if not enc:
            continue
        tid = enc[0]
        if "up" in s.lower():
            up_ids.add(tid)
        else:
            down_ids.add(tid)
    allowed = sorted(up_ids | down_ids)

    n=tp=tn=fp=fn=0
    for sc in scens:
        target = parse_prediction(sc.predator_target or "")
        curated = render_curated_slot(sc)
        prompt = f"{role}\n\n{curated}\n\n{query}"
        msgs = [{"role": "user", "content": prompt}]
        try:
            ids = tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", enable_thinking=False,
            ).to(host.device)
        except TypeError:
            ids = tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_tensors="pt",
            ).to(host.device)
        am = torch.ones_like(ids)
        with torch.no_grad():
            out = host._model(input_ids=ids, attention_mask=am, use_cache=False)
            logits = out.logits[0, -1]
            masked = torch.full_like(logits, float("-inf"))
            masked[allowed] = logits[allowed]
            pick = int(masked.argmax().item())
            d = "up" if pick in up_ids else "down"
            sub = torch.tensor([logits[i].item() for i in allowed], device=logits.device)
            p = torch.softmax(sub.float(), dim=0)
            conf = float(p[allowed.index(pick)])
        t = target.direction
        n += 1
        if d == "up" and t == "up": tp += 1
        elif d == "down" and t == "down": tn += 1
        elif d == "up" and t == "down": fp += 1
        elif d == "down" and t == "up": fn += 1
        print(f"  [{sc.name[:40]:40s}] tgt={t:<5} pred={d:<5} conf={conf:.3f}")

    dec = tp+tn+fp+fn
    mcc = (tp*tn-fp*fn) / max(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)), 1e-9)
    print()
    print(f"[loop1c] n={n}  tp={tp} tn={tn} fp={fp} fn={fn}")
    print(f"  acc={(tp+tn)/n:.2%}  MCC={mcc:+.4f}  classes={{up:{tp+fp},down:{tn+fn}}}")


if __name__ == "__main__":
    asyncio.run(main())
