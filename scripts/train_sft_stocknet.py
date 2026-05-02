"""SFT trained on StockNet train split (1167 scenarios), eval on dev (143).

Swaps the toy `eval_*`/`holdout_*` scenarios from `trophic.training.scenarios`
for StockNet's official splits. Keeps every other piece of the per-layer
M+E + d* + ORPO architecture intact:

  TRAIN_DATES = 2014-01-01 → 2015-08-01  → 1167 scenarios (53% up, 47% down)
  DEV_DATES   = 2015-08-01 → 2015-10-01  → 143 scenarios  (50/50)
  TEST_DATES  = 2015-10-01 → 2016-01-01  → 196 scenarios  (55/45)
                                           (eval_stocknet.py uses top-5 × 10)

Per the seed26 verdict (best dev 0.38 in-distribution, NaN/abstain on test
distribution-shift), the only intervention this run makes is **alignment of
training distribution with test distribution.** No phi_mlp shape change,
no LR change, no architecture change.

Activates per-layer hooks-mode + d* per `train_sft_perlayer.py`.

Saves checkpoints with phi_mlp + trough state (post-`save_channels` runner
arg).
"""
from __future__ import annotations

import asyncio
import functools
import os
import random
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Activate the per-layer hooks-mode architecture BEFORE any trophic imports.
os.environ.setdefault("TROPHIC_CONSUMER_INTERFACE", "hooks")
os.environ.setdefault("TROPHIC_DSTAR_PATH", str(ROOT / "checkpoints" / "dstar_stocknet.pt"))

import torch

from trophic.agents.herbivore import Herbivore
from trophic.agents.predator import Predator
from trophic.agents.producer import Producer
from trophic.agents.quant_producer import QuantitativeProducer
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.training.checkpoint import save_channels
from trophic.training.sft import SFTConfig, SFTRunner
from trophic.training.stocknet_loader import build_stocknet_scenarios


async def main() -> None:
    cfg = DEFAULT_CONFIG
    sft_cfg = SFTConfig(
        lr=float(os.environ.get("TROPHIC_LR", "5e-4")),
        steps=int(os.environ.get("TROPHIC_STEPS", "200")),
        log_every=int(os.environ.get("TROPHIC_LOG_EVERY", "10")),
        eval_every=int(os.environ.get("TROPHIC_EVAL_EVERY", "25")),
        seed=int(os.environ.get("TROPHIC_SEED", "28")),
        content_token_weight=float(os.environ.get("TROPHIC_CONTENT_W", "1.0")),
        grad_clip=float(os.environ.get("TROPHIC_GRAD_CLIP", "1.0")),
    )
    print(f"[init] seed={sft_cfg.seed} lr={sft_cfg.lr} steps={sft_cfg.steps}"
          f" content_w={sft_cfg.content_token_weight} grad_clip={sft_cfg.grad_clip}")
    rng = random.Random(sft_cfg.seed)

    print(f"[init] loading Qwen3-4B (mock={cfg.model.mock})")
    host = ModelHost.get(cfg.model)
    print(f"[init] hidden_size={host.hidden_size} dtype={host.dtype} device={host.device}")

    # Agents.
    producers = [Producer.make(k) for k in ("tickdelta", "disclosure", "anomaly")]
    producers.append(QuantitativeProducer.make("quote_series"))
    herbivores = [Herbivore.make(k, capacity=cfg.population.intake_budget)
                  for k in ("technical", "fundamental")]
    predator = Predator.make("short_horizon")

    # StockNet splits.
    print(f"[init] loading StockNet train split (1167-ish scenarios)…")
    train_scens = build_stocknet_scenarios(
        split="train",
        tickers=None,  # default top-5
        max_per_ticker=None,
    )
    dev_scens = build_stocknet_scenarios(
        split="dev",
        tickers=None,
        max_per_ticker=None,
    )
    print(f"[init] StockNet scenarios: {len(train_scens)} train, {len(dev_scens)} dev")

    # Cap dev to a manageable in-loop eval size (full 143 × ~30s = 70min per eval).
    dev_eval_cap = int(os.environ.get("TROPHIC_DEV_EVAL_CAP", "20"))
    if len(dev_scens) > dev_eval_cap:
        # Subsample deterministically with the seed.
        rng_dev = random.Random(sft_cfg.seed + 1)
        dev_eval = rng_dev.sample(dev_scens, dev_eval_cap)
        print(f"[init] dev eval capped to {dev_eval_cap} (full dev = {len(dev_scens)})")
    else:
        dev_eval = dev_scens

    runner = SFTRunner(
        cfg=sft_cfg, host=host,
        producers=producers, herbivores=herbivores, predator=predator,
        train=train_scens, eval_=dev_eval,
    )

    # Cache producer broadcasts for both train + dev.
    all_scens = train_scens + dev_eval
    print(f"[init] caching producer broadcasts for {len(all_scens)} scenarios…")
    t0 = time.time()
    done = 0
    for sc in all_scens:
        items = []
        for inp in sc.inputs:
            for prod in producers:
                if prod.attracts(inp):
                    b = await prod.produce(inp, tick=0, host=host)
                    if b is not None:
                        items.append(b)
                    done += 1
                    if done % 100 == 0:
                        print(f"[cache] {done} producer forwards done"
                              f" (elapsed {time.time()-t0:.1f}s)")
        runner._producer_cache[sc.name] = items
    print(f"[cache] DONE: {done} forwards in {time.time()-t0:.1f}s")

    # Checkpoint paths.
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_best = ckpt_dir / f"sft_seed{sft_cfg.seed}_stocknet_best.pt"
    ckpt_final = ckpt_dir / f"sft_seed{sft_cfg.seed}_stocknet_final.pt"
    best_eval = float("inf")

    print("\n=== TRAINING (StockNet train split) ===")
    n = sft_cfg.steps
    for step in range(1, n + 1):
        sc = rng.choice(train_scens)
        result = runner.step(sc, train_step=step)
        loss_after = runner.maybe_step(step)
        if step % sft_cfg.log_every == 0:
            ploss = " ".join(
                f"{k}={v:.3f}" for k, v in result["per_loss"].items()
                if isinstance(v, (int, float))
            )
            tau = result.get("tau", 1.0)
            snap = runner.ecology_snapshot()
            extra = f" tau={tau:.3f} alpha={snap['skip_weight_alpha']:.3f}"
            if "trough_n_alive" in snap:
                extra += (f" trough_alive={snap['trough_n_alive']}"
                          f" head_ent={snap['head_specialization_entropy']:.3f}")
            print(f"[step {step:4d}] lr={runner.trainer.current_lr:.2e}"
                  f" sc={sc.name[:36]:36s} train_loss={result['loss']:.3f}"
                  f"  ({ploss}){extra}")
        if step % sft_cfg.eval_every == 0:
            print(f"\n--- dev eval @ step {step} ---")
            elr = runner.eval_loss()
            cur_lr, _ = runner.trainer.on_eval(elr["mean"])
            pk = " ".join(f"{k}={v:.3f}" for k, v in elr["per_kind"].items())
            print(f"  EVAL_LOSS: mean={elr['mean']:.4f}  ({pk})  lr→{cur_lr:.2e}")
            if elr["mean"] < best_eval:
                best_eval = elr["mean"]
                save_channels(
                    str(ckpt_best),
                    herbivores=herbivores, predators=[predator],
                    meta={"sft_seed": sft_cfg.seed, "step": step,
                          "eval_loss": elr["mean"], "split": "stocknet_train"},
                    runner=runner,
                )
                print(f"  [ckpt] new best ({elr['mean']:.4f}) → {ckpt_best.name}")
            print()

    save_channels(
        str(ckpt_final),
        herbivores=herbivores, predators=[predator],
        meta={"sft_seed": sft_cfg.seed, "step": n,
              "eval_loss": best_eval, "split": "stocknet_train"},
        runner=runner,
    )
    print(f"[ckpt] final → {ckpt_final.name}; best dev={best_eval:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
