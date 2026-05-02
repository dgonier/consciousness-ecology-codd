"""Loop 2: hooks-mode forward smoke (~2 min).

Loads an existing checkpoint, runs end-to-end inference on N scenarios with
the per-tier logit-lens trace. NO TRAINING. Tests the architecture's
*current state*: signal flow, NaN absence, parser hits, decode quality.

Catches without burning a training run:
- OOD prefix vectors (would cause gibberish or NaN at decode)
- phi_mlp magnitude blowup post-checkpoint-load
- Missing trough state, missing E_in, mismatched dtype/device
- Parser misses on the model's actual output format
- Constant-collapse at the eval level

Pass criteria:
- All scenarios produce finite (non-NaN) hidden states across tiers
- Decoded output parses to a direction (up/down) on >=80% of scenarios
- At least 2 unique direction predictions across N scenarios

Usage:
    .venv/bin/python -u scripts/diagnostics/loop2_hooks_smoke.py \\
        --ckpt checkpoints/sft_seed28_stocknet_best.pt
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import os
import sys
from pathlib import Path

print = functools.partial(print, flush=True)
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch

from trophic.agents.herbivore import Herbivore
from trophic.agents.predator import Predator
from trophic.agents.producer import Producer
from trophic.agents.quant_producer import QuantitativeProducer
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.training.checkpoint import load_channels
from trophic.training.sft import SFTConfig, SFTRunner
from trophic.training.stocknet_loader import build_stocknet_scenarios, SMOKE_TICKERS_TOP5
from trophic.training.xml_schema import parse_prediction


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sft_seed28_stocknet_best.pt")
    ap.add_argument("--seed", type=int, default=28)
    ap.add_argument("--n-per-ticker", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()

    os.environ.setdefault("TROPHIC_CONSUMER_INTERFACE", "hooks")
    os.environ.setdefault("TROPHIC_DSTAR_PATH", "checkpoints/dstar_stocknet.pt")
    os.environ.setdefault("TROPHIC_DSTAR_SCALE_OVERRIDE", "1.0")

    cfg = DEFAULT_CONFIG
    host = ModelHost.get(cfg.model)
    sft_cfg = SFTConfig(seed=args.seed)

    herbs = [Herbivore.make(k, capacity=cfg.population.intake_budget)
             for k in ("technical", "fundamental")]
    pred = Predator.make("short_horizon")
    producers = [Producer.make(k) for k in ("tickdelta", "disclosure", "anomaly")]
    producers.append(QuantitativeProducer.make("quote_series"))

    scens = build_stocknet_scenarios(
        split="test", tickers=SMOKE_TICKERS_TOP5, max_per_ticker=args.n_per_ticker,
    )
    print(f"[loop2] {len(scens)} scenarios, ckpt={args.ckpt}")

    runner = SFTRunner(
        cfg=sft_cfg, host=host, producers=producers,
        herbivores=herbs, predator=pred, train=[], eval_=scens,
    )
    load_channels(args.ckpt, herbivores=herbs, predators=[pred], runner=runner)
    if pred.phi_mlp is not None:
        pred.phi_mlp.to(device=host.device, dtype=host.dtype)
    for h in herbs:
        if h.phi_mlp is not None:
            h.phi_mlp.to(device=host.device, dtype=host.dtype)
    if runner._producer_trough is not None:
        runner._producer_trough.to(device=host.device, dtype=host.dtype)
    if runner._herb_trough is not None:
        runner._herb_trough.to(device=host.device, dtype=host.dtype)
    await runner._cache_producer_broadcasts()

    n_decided = 0
    n_correct = 0
    n_finite = 0  # how many scenarios had finite herb broadcasts
    decoded_set = set()
    direction_set = set()

    for sc in scens:
        target = parse_prediction(sc.predator_target or "")

        # Tier check: are herb broadcasts finite?
        candidates = runner._producer_cache.get(sc.name, [])
        herb_for_pred = runner._real_herb_broadcasts_for_predator(sc, candidates)
        any_nan = False
        for b in herb_for_pred:
            emb = torch.tensor(b.channel_embedding)
            if not torch.isfinite(emb).all():
                any_nan = True
                break
        if not any_nan:
            n_finite += 1

        # Apex decode
        out = runner.eval_decode(sc, max_new_tokens=args.max_new_tokens)
        decoded = out.get("pred.short_horizon", "")
        parsed = parse_prediction(decoded)
        d = parsed.direction
        decoded_set.add(decoded[:60])
        if d in ("up", "down"):
            n_decided += 1
            direction_set.add(d)
            if d == target.direction:
                n_correct += 1

        snip = decoded[:120].replace("\n", " ")
        print(f"  [{sc.name[:40]:40s}] tgt={target.direction or '?':<5} pred={d or 'None':<6} "
              f"finite={'Y' if not any_nan else 'N'}  | {snip}")

    n = len(scens)
    print()
    print(f"[loop2] finite_tiers={n_finite}/{n}  decided={n_decided}/{n}  "
          f"correct={n_correct}/{n}  unique_decodes={len(decoded_set)}  "
          f"directions_emitted={direction_set}")

    # Pass criteria
    fail = False
    if n_finite < n:
        print(f"[loop2] FAIL: {n - n_finite} scenarios produced NaN herb broadcasts")
        fail = True
    if n_decided / n < 0.80:
        print(f"[loop2] FAIL: only {100*n_decided/n:.0f}% decided (<80%)")
        fail = True
    if len(direction_set) < 2:
        print(f"[loop2] FAIL: only {len(direction_set)} direction class emitted — constant collapse")
        fail = True

    if fail:
        return 1
    print(f"[loop2] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
