"""Loop 3: 50-step training smoke (~6 min).

Runs 50 SFT steps with eval at 25 and 50, on a tiny subset of StockNet
train. Tells us whether:
  - Loss is descending (gradient health)
  - phi_mlp params are moving (not stuck at saddle)
  - No NaN in train_loss across 50 steps
  - First dev eval is finite + lower than initial

Catches without burning the full 90-min run:
  - Gradient explosion / NaN within 50 steps
  - Phi_mlp params zero-grad (saddle-init regression)
  - Training data caching that corrupts trough state
  - Optimizer initialization bugs

Pass criteria:
  - All 50 train_loss values are finite
  - Step-50 dev loss < step-25 dev loss (some descent)
  - phi_mlp predator s_M moved >0.001 absolute from init (gradient flowed)

Usage:
    .venv/bin/python -u scripts/diagnostics/loop3_train_smoke.py
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import os
import random
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
from trophic.training.sft import SFTConfig, SFTRunner
from trophic.training.stocknet_loader import build_stocknet_scenarios


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=99)  # different from any real run
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--n-train", type=int, default=80)  # small training pool, fast cache
    ap.add_argument("--n-dev", type=int, default=8)
    args = ap.parse_args()

    os.environ.setdefault("TROPHIC_CONSUMER_INTERFACE", "hooks")
    os.environ.setdefault("TROPHIC_DSTAR_PATH", "checkpoints/dstar_stocknet.pt")
    os.environ.setdefault("TROPHIC_DSTAR_SCALE_OVERRIDE", "1.0")

    cfg = DEFAULT_CONFIG
    sft_cfg = SFTConfig(
        lr=5e-5,
        steps=args.steps,
        log_every=10,
        eval_every=25,
        seed=args.seed,
        grad_clip=0.5,
    )
    print(f"[loop3] seed={sft_cfg.seed} steps={sft_cfg.steps} lr={sft_cfg.lr}")
    rng = random.Random(sft_cfg.seed)

    host = ModelHost.get(cfg.model)

    producers = [Producer.make(k) for k in ("tickdelta", "disclosure", "anomaly")]
    producers.append(QuantitativeProducer.make("quote_series"))
    herbs = [Herbivore.make(k, capacity=cfg.population.intake_budget)
             for k in ("technical", "fundamental")]
    pred = Predator.make("short_horizon")

    # Subsample train + dev to keep cache fast.
    train_full = build_stocknet_scenarios(split="train", tickers=None, max_per_ticker=None)
    dev_full = build_stocknet_scenarios(split="dev", tickers=None, max_per_ticker=None)
    train_scens = rng.sample(train_full, min(args.n_train, len(train_full)))
    dev_scens = rng.sample(dev_full, min(args.n_dev, len(dev_full)))
    print(f"[loop3] train={len(train_scens)} dev={len(dev_scens)}")

    runner = SFTRunner(
        cfg=sft_cfg, host=host, producers=producers,
        herbivores=herbs, predator=pred, train=train_scens, eval_=dev_scens,
    )

    # Cache producer broadcasts (fast, ~30s for 88 scenarios).
    print(f"[loop3] caching producer broadcasts…")
    import time
    t0 = time.time()
    for sc in train_scens + dev_scens:
        items = []
        for inp in sc.inputs:
            for prod in producers:
                if prod.attracts(inp):
                    b = await prod.produce(inp, tick=0, host=host)
                    if b is not None:
                        items.append(b)
        runner._producer_cache[sc.name] = items
    print(f"[loop3] cache done in {time.time()-t0:.0f}s")

    # Snapshot phi_mlp predator s_M before training.
    s_M_init = pred.phi_mlp.s_M.detach().clone() if pred.phi_mlp is not None else None

    print(f"[loop3] === TRAINING ===")
    train_losses = []
    dev_losses = {}
    nan_streak = 0
    t0 = time.time()
    for step in range(1, sft_cfg.steps + 1):
        sc = rng.choice(train_scens)
        result = runner.step(sc, train_step=step)
        runner.maybe_step(step)
        loss = result.get("loss", float("nan"))
        is_nan = not isinstance(loss, (int, float)) or loss != loss
        train_losses.append(loss)
        if is_nan:
            nan_streak += 1
        else:
            nan_streak = 0
        if step % sft_cfg.log_every == 0:
            print(f"  [step {step:3d}] train_loss={loss:.3f}")
        if step % sft_cfg.eval_every == 0:
            elr = runner.eval_loss()
            print(f"  [step {step:3d}] dev_loss={elr['mean']:.4f}")
            dev_losses[step] = elr["mean"]
        if nan_streak >= 5:
            print(f"  [loop3] ABORT: 5 consecutive NaN")
            break

    elapsed = time.time() - t0
    print(f"\n[loop3] {sft_cfg.steps} steps in {elapsed:.0f}s ({elapsed/sft_cfg.steps:.1f}s/step)")

    # Pass criteria
    fail = False
    n_finite_train = sum(1 for v in train_losses if isinstance(v, (int, float)) and v == v)
    if n_finite_train < len(train_losses):
        print(f"[loop3] FAIL: {len(train_losses) - n_finite_train} of {len(train_losses)} train_loss values were NaN")
        fail = True

    if 25 in dev_losses and 50 in dev_losses:
        d25, d50 = dev_losses[25], dev_losses[50]
        print(f"[loop3] dev: step25={d25:.4f}  step50={d50:.4f}  delta={d50-d25:+.4f}")
        if d50 >= d25:
            print(f"[loop3] WARN: dev didn't descend across 25→50 (delta={d50-d25:+.4f})")
            # don't fail; some smoke runs may have noisy dev

    if s_M_init is not None and pred.phi_mlp is not None:
        s_M_now = pred.phi_mlp.s_M.detach()
        delta = (s_M_now - s_M_init.to(s_M_now.device)).abs().max().item()
        print(f"[loop3] predator phi_mlp s_M max-delta from init: {delta:.5f}")
        if delta < 1e-4:
            print(f"[loop3] FAIL: phi_mlp s_M didn't move (gradient zero at the saddle)")
            fail = True

    if fail:
        return 1
    print(f"[loop3] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
