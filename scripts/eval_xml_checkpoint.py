"""Re-eval an existing SFT/IPO checkpoint, parsing XML and computing
field-wise rule-based reward on every eval scenario. No training, just
inspection.
"""
from __future__ import annotations

import asyncio
import functools
import os
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from trophic.agents.herbivore import Herbivore
from trophic.agents.predator import Predator
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.training.checkpoint import load_channels
from trophic.training.scenarios import build_scenarios, split
from trophic.training.sft import SFTConfig, SFTRunner
from trophic.training.xml_schema import (
    parse_prediction, parse_synthesis, reward_prediction, reward_synthesis,
)


async def main() -> None:
    cfg = DEFAULT_CONFIG
    sft_cfg = SFTConfig(
        seed=int(os.environ.get("TROPHIC_SEED", "1")),
    )
    ckpt_path = os.environ.get(
        "TROPHIC_CKPT", str(ROOT / "checkpoints" / f"sft_seed{sft_cfg.seed}_best.pt")
    )
    print(f"[eval] ckpt={ckpt_path}")

    host = ModelHost.get(cfg.model)
    print(f"[eval] hidden_size={host.hidden_size} dtype={host.dtype} device={host.device}")

    herbivores = [Herbivore.make(k, capacity=cfg.population.intake_budget)
                  for k in ("technical", "fundamental")]
    predator = Predator.make("short_horizon")
    for h in herbivores:
        h.ensure_initialized(host, seed_base=sft_cfg.seed)
    predator.ensure_initialized(host, seed_base=sft_cfg.seed)
    for h in herbivores:
        for ch in h.channels.values():
            ch.to(device=host.device, dtype=host.dtype)
    for ch in predator.channels.values():
        ch.to(device=host.device, dtype=host.dtype)

    meta = load_channels(ckpt_path, herbivores=herbivores, predators=[predator])
    print(f"[eval] loaded checkpoint meta: {meta}")

    for ch in predator.channels.values():
        ch.to(device=host.device, dtype=host.dtype)
    for h in herbivores:
        for ch in h.channels.values():
            ch.to(device=host.device, dtype=host.dtype)

    scenarios = build_scenarios()
    train, eval_ = split(scenarios)

    runner = SFTRunner(
        cfg=sft_cfg, host=host,
        producers=[],   # not needed for eval-decode (uses oracle herb path)
        herbivores=herbivores, predator=predator,
        train=train, eval_=eval_,
    )

    # Cache producer broadcasts (eval_decode walks producers, so we need them)
    from trophic.agents.producer import Producer
    from trophic.agents.quant_producer import QuantitativeProducer
    producers = [Producer.make(k) for k in ("tickdelta", "disclosure", "anomaly")]
    producers.append(QuantitativeProducer.make("quote_series"))
    runner.producers = producers
    print("[eval] caching producer broadcasts...")
    await runner._cache_producer_broadcasts()

    # Per-scenario eval
    pred_rewards = []
    print("\n=== EVAL (XML parsed + rule-based reward) ===")
    for sc in eval_:
        out = runner.eval_decode(sc)
        pred_text = out.get("pred.short_horizon", "")
        target_text = sc.predator_target or ""
        parsed = parse_prediction(pred_text)
        target = parse_prediction(target_text)
        rew = reward_prediction(parsed, target)
        pred_rewards.append(rew["total"])
        print(f"\n[{sc.name}] reward={rew['total']:.3f}")
        print(f"  TARGET:   ticker={target.ticker} dir={target.direction} pct={target.pct_move} horizon={target.horizon_min} σ={target.sigma_pct}")
        print(f"  PREDICTED: ticker={parsed.ticker} dir={parsed.direction} pct={parsed.pct_move} horizon={parsed.horizon_min} σ={parsed.sigma_pct}")
        print(f"  RAW: {pred_text[:200]}")
        print(f"  BREAKDOWN: {rew['breakdown']}")

    mean_r = sum(pred_rewards) / max(1, len(pred_rewards))
    print(f"\n=== MEAN PREDATOR REWARD: {mean_r:.3f} (over {len(pred_rewards)} scenarios) ===")


if __name__ == "__main__":
    asyncio.run(main())
