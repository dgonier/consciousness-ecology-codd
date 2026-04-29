"""A/B test: producer mean-pool vs last-pool variance across StockNet days.

The producer collapse cosine = 0.998 (mean-pool). Hypothesis: long boilerplate
prefix dominates the mean. If we re-pool the same texts with pool='last',
we get the hidden state at the END of the input — which is positioned right
at the most recent OHLCV bar / latest tweet count and SHOULD differ across
days.

Compares:
  - pool='mean' cosine across 5 AAPL days (current architecture)
  - pool='last' cosine across same 5 days (proposed fix)
  - pool='last_n=8' cosine (mean of last 8 tokens — between mean and last)

If pool='last' gives cosine < 0.95, this is the simple fix for #8.
"""
from __future__ import annotations
import asyncio, builtins, functools, os, sys
from pathlib import Path
builtins.print = functools.partial(builtins.print, flush=True)
print = builtins.print

ROOT = Path("/home/dgonier/ecology_experiment/trophic")
sys.path.insert(0, str(ROOT))

import itertools
import torch
import torch.nn.functional as F

from trophic.agents.producer import Producer, _render
from trophic.agents.quant_producer import QuantitativeProducer
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.training.stocknet_loader import build_stocknet_scenarios


def _pairwise_cosines(tensors):
    pairs = []
    for ti, tj in itertools.combinations(tensors, 2):
        c = F.cosine_similarity(
            ti.flatten().unsqueeze(0).float(),
            tj.flatten().unsqueeze(0).float(),
        ).item()
        pairs.append(c)
    return {"mean": sum(pairs)/len(pairs), "min": min(pairs), "max": max(pairs), "n": len(pairs)}


async def main() -> None:
    cfg = DEFAULT_CONFIG
    ticker = os.environ.get("STOCKNET_TICKER", "AAPL")
    n_days = int(os.environ.get("STOCKNET_DAYS", "5"))
    print(f"[diag-pool] ticker={ticker}  n_days={n_days}")
    host = ModelHost.get(cfg.model)

    scenarios = build_stocknet_scenarios(
        split="test", tickers=[ticker], max_per_ticker=n_days,
    )

    # For each producer kind, render text per scenario and forward with each pool
    producers = [Producer.make(k) for k in ("tickdelta", "disclosure", "anomaly")]
    producers.append(QuantitativeProducer.make("quote_series"))

    print(f"\n=== POOL A/B: producer broadcasts across {n_days} days of {ticker} ===\n")
    for prod in producers:
        # Gather rendered texts that this producer would actually emit
        texts = []
        names = []
        for sc in scenarios:
            for inp in sc.inputs:
                if not prod.attracts(inp):
                    continue
                try:
                    text = _render(prod.kind, inp)
                except Exception:
                    continue
                texts.append(text)
                names.append(sc.name)
                break  # one per scenario for fair comparison
        if len(texts) < 2:
            print(f"  {prod.kind:25s}: only {len(texts)} texts; skipping")
            continue
        print(f"  {prod.kind:25s}: {len(texts)} texts")
        # token lengths
        tok = host._tok
        tlens = [len(tok(t)["input_ids"]) for t in texts]
        print(f"    text lengths (tokens): min={min(tlens)} max={max(tlens)} median={sorted(tlens)[len(tlens)//2]}")
        # mean pool
        means = [host.text_to_hidden(t, pool="mean").detach() for t in texts]
        m = _pairwise_cosines(means)
        # last pool
        lasts = [host.text_to_hidden(t, pool="last").detach() for t in texts]
        l = _pairwise_cosines(lasts)
        # last-N=8 pool: mean of last 8 tokens
        seqs = [host.text_to_hidden(t, pool=None).detach() for t in texts]
        last8s = [s[-min(8, s.shape[0]):].mean(dim=0) for s in seqs]
        l8 = _pairwise_cosines(last8s)
        print(f"    mean-pool   cosine: mean={m['mean']:.4f}  min={m['min']:.4f}  max={m['max']:.4f}")
        print(f"    last-pool   cosine: mean={l['mean']:.4f}  min={l['min']:.4f}  max={l['max']:.4f}")
        print(f"    last8-pool  cosine: mean={l8['mean']:.4f}  min={l8['min']:.4f}  max={l8['max']:.4f}")
        print()

    print("=== INTERPRETATION ===")
    print("Lower cosine = more input-responsive.")
    print("If last-pool / last8-pool < 0.95 (and mean-pool > 0.99): mean-pool collapse confirmed,")
    print("  switching producer pool='last' should restore input-responsiveness with no retraining.")


if __name__ == "__main__":
    asyncio.run(main())
