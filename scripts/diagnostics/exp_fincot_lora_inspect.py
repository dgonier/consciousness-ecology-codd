"""Inspect what the FinCoT-LoRA-wrapped model actually emits on StockNet.

Mirrors scripts/diagnostics/exp_A_inspect.py from issue #11 — runs ONE
StockNet scenario, dumps full decoded predator output, so we can see
whether the 50/50 'abstention' result from exp_fincot_lora_stocknet.py
is genuine abstention or gibberish.
"""
from __future__ import annotations
import asyncio, builtins, functools, os, sys
from pathlib import Path
builtins.print = functools.partial(builtins.print, flush=True)
print = builtins.print

ROOT = Path("/home/dgonier/ecology_experiment/trophic")
sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from trophic.agents.herbivore import Herbivore
from trophic.agents.predator import Predator
from trophic.agents.producer import Producer
from trophic.agents.quant_producer import QuantitativeProducer
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.training.checkpoint import load_channels
from trophic.training.stocknet_loader import build_stocknet_scenarios
from trophic.training.sft import SFTConfig, SFTRunner
from trophic.training.xml_schema import parse_prediction


async def main() -> None:
    cfg = DEFAULT_CONFIG
    seed = 8
    host = ModelHost.get(cfg.model)

    # Attach FinCoT LoRA
    lora_dir = "checkpoints/interrogator_fincot_lora/seed11"
    print(f"[insp-lora] attaching adapter from {lora_dir}")
    host._model = PeftModel.from_pretrained(host._model, lora_dir)
    host._model.eval()

    sft_cfg = SFTConfig(seed=seed)
    herbivores = [Herbivore.make(k, capacity=cfg.population.intake_budget)
                  for k in ("technical", "fundamental")]
    predator = Predator.make("short_horizon")
    for h in herbivores:
        h.ensure_initialized(host, seed_base=seed)
    predator.ensure_initialized(host, seed_base=seed)
    for h in herbivores:
        for ch in h.channels.values():
            ch.to(device=host.device, dtype=host.dtype)
    for ch in predator.channels.values():
        ch.to(device=host.device, dtype=host.dtype)
    load_channels("checkpoints/ipo_seed8_best.pt", herbivores=herbivores, predators=[predator])
    print(f"[insp-lora] loaded ipo_seed8_best.pt")

    scenarios = build_stocknet_scenarios(
        split="test", tickers=["AAPL"], max_per_ticker=2,
    )

    producers = [Producer.make(k) for k in ("tickdelta", "disclosure", "anomaly")]
    producers.append(QuantitativeProducer.make("quote_series"))
    runner = SFTRunner(cfg=sft_cfg, host=host, producers=producers,
                       herbivores=herbivores, predator=predator,
                       train=[], eval_=scenarios)
    await runner._cache_producer_broadcasts()

    for sc in scenarios:
        print(f"\n=== {sc.name} ===")
        target = parse_prediction(sc.predator_target or "")
        print(f"  target direction: {target.direction}")
        out = runner.eval_decode(sc, max_new_tokens=400)
        decoded = out.get("pred.short_horizon", "")
        print(f"  raw decode (first 600 chars):")
        print(f"  {'-'*70}")
        print(f"  {decoded[:600]}")
        print(f"  {'-'*70}")
        parsed = parse_prediction(decoded)
        print(f"  parsed: ticker={parsed.ticker} dir={parsed.direction} pct={parsed.pct_move}")


if __name__ == "__main__":
    asyncio.run(main())
