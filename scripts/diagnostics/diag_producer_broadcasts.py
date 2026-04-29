"""Check producer broadcast variance across StockNet days for the same ticker.

The input-responsiveness diagnostic localized the collapse to the herbivore
output (cosine 1.0000 across 5 AAPL days). Two possibilities upstream of
the herbivore Channel:

  A. Producer broadcasts are themselves constant — producers see different
     inputs but their channel_embedding doesn't vary.
  B. Producer broadcasts vary, but the herbivore's Channel cross-attention
     squashes them.

This script answers (A): pull the cached producer broadcasts for 5 AAPL
StockNet days, compute pairwise cosine of each producer's channel_embedding.

If producer broadcasts also collapse → the bug is at the producer level
(content_text rendered via Qwen3 produces constant embeddings — likely the
producer prompt template is not actually varying with the input).
If producer broadcasts vary → the bug is in herbivore Channel.forward
(taught to ignore K/V differences during training).
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

from trophic.agents.herbivore import Herbivore
from trophic.agents.predator import Predator
from trophic.agents.producer import Producer
from trophic.agents.quant_producer import QuantitativeProducer
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.training.stocknet_loader import build_stocknet_scenarios
from trophic.training.sft import SFTConfig, SFTRunner


def _pairwise_cosines(tensors, names):
    if len(tensors) < 2:
        return None
    pairs = []
    for (i, ti), (j, tj) in itertools.combinations(enumerate(tensors), 2):
        c = F.cosine_similarity(
            ti.flatten().unsqueeze(0).float(),
            tj.flatten().unsqueeze(0).float(),
        ).item()
        pairs.append(c)
    return {"mean": sum(pairs)/len(pairs), "min": min(pairs), "max": max(pairs), "n": len(pairs)}


async def main() -> None:
    cfg = DEFAULT_CONFIG
    seed = int(os.environ.get("TROPHIC_SEED", "16"))
    ticker = os.environ.get("STOCKNET_TICKER", "AAPL")
    n_days = int(os.environ.get("STOCKNET_DAYS", "5"))
    lora_dir = os.environ.get("TROPHIC_LORA_DIR", "checkpoints/social_signal_lora/seed16")

    print(f"[diag-prod] seed={seed}  ticker={ticker}  n_days={n_days}")
    host = ModelHost.get(cfg.model)
    if lora_dir and Path(lora_dir).exists():
        from peft import PeftModel
        print(f"[diag-prod] attaching LoRA from {lora_dir}")
        host._model = PeftModel.from_pretrained(host._model, lora_dir, is_trainable=False)
        host._model.eval()

    scenarios = build_stocknet_scenarios(
        split="test", tickers=[ticker], max_per_ticker=n_days,
    )
    print(f"[diag-prod] scenarios: {len(scenarios)}")
    for sc in scenarios:
        print(f"  - {sc.name}  inputs={len(sc.inputs)}  sources={[i.source for i in sc.inputs]}")

    # We need a runner to populate the producer cache
    sft_cfg = SFTConfig(seed=seed)
    herbivores = [Herbivore.make(k, capacity=cfg.population.intake_budget)
                  for k in ("technical", "fundamental")]
    predator = Predator.make("short_horizon")
    for h in herbivores:
        h.ensure_initialized(host, seed_base=seed)
    predator.ensure_initialized(host, seed_base=seed)

    producers = [Producer.make(k) for k in ("tickdelta", "disclosure", "anomaly")]
    producers.append(QuantitativeProducer.make("quote_series"))
    runner = SFTRunner(cfg=sft_cfg, host=host, producers=producers,
                       herbivores=herbivores, predator=predator,
                       train=[], eval_=scenarios)
    print(f"[diag-prod] caching producer broadcasts...")
    await runner._cache_producer_broadcasts()

    # Inspect cache
    cache = runner._producer_cache
    print(f"\n=== PRODUCER BROADCAST INVENTORY ===")
    for sc in scenarios:
        items = cache.get(sc.name, [])
        print(f"\n[{sc.name}] {len(items)} broadcasts:")
        for b in items:
            ce = b.channel_embedding
            ce_norm = float(torch.tensor(ce).norm()) if ce is not None else None
            text_preview = (b.decoded_text or "")[:80].replace("\n", " ")
            print(f"  source={b.agent_kind!r:20s}  emb_dim={len(ce) if ce is not None else 'None'}"
                  f"  norm={ce_norm:.3f}  text={text_preview!r}")

    # Group by source and compute cosine across scenarios for each source
    print(f"\n=== PAIRWISE COSINE BY PRODUCER SOURCE ===")
    by_source: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for sc in scenarios:
        for b in cache.get(sc.name, []):
            ce = b.channel_embedding
            if ce is None:
                continue
            t = torch.tensor(ce, dtype=torch.float32)
            by_source.setdefault(b.agent_kind, []).append((sc.name, t))

    for source, items in by_source.items():
        if len(items) < 2:
            print(f"  {source}: only {len(items)} broadcasts, skipping")
            continue
        # Take first broadcast per scenario for this source (some scenarios have duplicates)
        seen_sc = set()
        firsts = []
        names = []
        for n, t in items:
            if n in seen_sc:
                continue
            seen_sc.add(n)
            firsts.append(t)
            names.append(n)
        r = _pairwise_cosines(firsts, names)
        if r is None:
            print(f"  {source}: not enough unique-scenario broadcasts ({len(firsts)})")
            continue
        flag = "← COLLAPSE" if r["mean"] > 0.99 else "← VARIANCE OK" if r["mean"] < 0.90 else "← borderline"
        print(f"  {source:25s} cosine mean={r['mean']:.4f} min={r['min']:.4f} max={r['max']:.4f} (n={r['n']}) {flag}")

    print(f"\n=== INTERPRETATION ===")
    print("If all producer cosines > 0.99: producers emit constant embeddings regardless of input.")
    print("  → root cause is upstream of the herbivore — producer prompt/template doesn't vary.")
    print("If producer cosines < 0.90: producers DO see input variance.")
    print("  → root cause is herbivore Channel learned to ignore K/V differences during IPO.")


if __name__ == "__main__":
    asyncio.run(main())
