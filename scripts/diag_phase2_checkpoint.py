"""Quick diagnostic: load the post-Phase-2 SFT checkpoint and check greedy
decode output on a few held-out scenarios. Tells us whether the checkpoint
itself produces parseable XML.
"""
from __future__ import annotations

import asyncio
import functools
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
    parse_prediction, parse_synthesis, reward_prediction,
)


async def main() -> None:
    cfg = DEFAULT_CONFIG
    sft_cfg = SFTConfig(seed=1)
    ckpt = ROOT / "checkpoints" / "sft_seed1_best.pt"

    host = ModelHost.get(cfg.model)
    print(f"[init] hidden={host.hidden_size} dtype={host.dtype} device={host.device}")

    herbivores = [Herbivore.make(k) for k in ("technical", "fundamental")]
    predator = Predator.make("short_horizon")
    for h in herbivores:
        h.ensure_initialized(host, seed_base=1)
    predator.ensure_initialized(host, seed_base=1)
    for h in herbivores:
        for ch in h.channels.values():
            ch.to(device=host.device, dtype=host.dtype)
    for ch in predator.channels.values():
        ch.to(device=host.device, dtype=host.dtype)

    meta = load_channels(str(ckpt), herbivores=herbivores, predators=[predator])
    print(f"[init] loaded ckpt: {meta}")
    for h in herbivores:
        for ch in h.channels.values():
            ch.to(device=host.device, dtype=host.dtype)
    for ch in predator.channels.values():
        ch.to(device=host.device, dtype=host.dtype)

    scenarios = build_scenarios()
    train, eval_ = split(scenarios)

    runner = SFTRunner(
        cfg=sft_cfg, host=host,
        producers=[],
        herbivores=herbivores, predator=predator,
        train=train, eval_=eval_,
    )
    from trophic.agents.producer import Producer
    from trophic.agents.quant_producer import QuantitativeProducer
    runner.producers = [Producer.make(k) for k in ("tickdelta", "disclosure", "anomaly")]
    runner.producers.append(QuantitativeProducer.make("quote_series"))
    print("[init] caching producer broadcasts...")
    await runner._cache_producer_broadcasts()

    print("\n=== Greedy-decode eval on held-out scenarios ===")
    rewards = []
    for sc in eval_:
        out = runner.eval_decode(sc)
        pred_text = out.get("pred.short_horizon", "")
        target = parse_prediction(sc.predator_target or "")
        parsed = parse_prediction(pred_text)
        rew = reward_prediction(parsed, target)
        rewards.append(rew["total"])
        print(f"\n[{sc.name}] reward={rew['total']:.3f}")
        print(f"  TARGET:   ticker={target.ticker} dir={target.direction} pct={target.pct_move} horizon={target.horizon_min}")
        print(f"  PREDICTED: ticker={parsed.ticker} dir={parsed.direction} pct={parsed.pct_move} horizon={parsed.horizon_min}")
        print(f"  RAW: {pred_text[:200]}")
        # Also show the interrogator broadcast that was fed in
        print(f"  INTERROG_TARGET: {sc.interrogator_target[:200] if sc.interrogator_target else '<none>'}")

    mean = sum(rewards) / max(1, len(rewards))
    print(f"\n=== MEAN REWARD: {mean:.3f} ===")


if __name__ == "__main__":
    asyncio.run(main())
