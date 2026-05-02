"""SFT trained on StockNet train split + stability fixes (seed30).

Issue #15 (P1): the seed24/26/28 NaN-at-step-150-190 pattern is the
gating problem before any other architectural extension. Three
empirically-targeted fixes:

  1. **d* scale 10.0 → 1.0** via TROPHIC_DSTAR_SCALE_OVERRIDE=1.0.
     d* at scale=10 composes with trained M tensors to push Qwen
     activations to NaN on out-of-distribution inputs at decode time.
  2. **phi_mlp s_M / s_E bounded by tanh × 0.5** in phi_mlp.py.
     The multiplicative path s * (x A) B^T can't push hidden states
     to NaN if s ∈ (-0.5, 0.5).
  3. **Lower phi_mlp LR (5e-5 = 10× lower than 5e-4 default)** via
     TROPHIC_LR=5e-5. Smaller steps so the parameter trajectory
     doesn't overshoot the stable-loss minimum.

Same StockNet train (1167) / dev (143 → 20 in-loop) split as
train_sft_stocknet.py. Trains 400 steps (4× longer to compensate
for the smaller LR), checkpoints every 50.
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

# Stability fixes (env var defaults). Caller can override.
os.environ.setdefault("TROPHIC_CONSUMER_INTERFACE", "hooks")
os.environ.setdefault("TROPHIC_DSTAR_PATH", str(ROOT / "checkpoints" / "dstar_stocknet.pt"))
os.environ.setdefault("TROPHIC_DSTAR_SCALE_OVERRIDE", "1.0")
os.environ.setdefault("TROPHIC_LR", "5e-5")
os.environ.setdefault("TROPHIC_STEPS", "400")
os.environ.setdefault("TROPHIC_EVAL_EVERY", "50")
os.environ.setdefault("TROPHIC_GRAD_CLIP", "0.5")

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
        lr=float(os.environ.get("TROPHIC_LR", "5e-5")),
        steps=int(os.environ.get("TROPHIC_STEPS", "400")),
        log_every=int(os.environ.get("TROPHIC_LOG_EVERY", "10")),
        eval_every=int(os.environ.get("TROPHIC_EVAL_EVERY", "50")),
        seed=int(os.environ.get("TROPHIC_SEED", "30")),
        content_token_weight=float(os.environ.get("TROPHIC_CONTENT_W", "1.0")),
        grad_clip=float(os.environ.get("TROPHIC_GRAD_CLIP", "0.5")),
    )
    print(f"[init] STABLE INFRA seed={sft_cfg.seed} lr={sft_cfg.lr} steps={sft_cfg.steps}"
          f" grad_clip={sft_cfg.grad_clip}")
    print(f"[init] d* scale override: {os.environ.get('TROPHIC_DSTAR_SCALE_OVERRIDE')}")
    rng = random.Random(sft_cfg.seed)

    print(f"[init] loading Qwen3-4B (mock={cfg.model.mock})")
    host = ModelHost.get(cfg.model)
    print(f"[init] hidden_size={host.hidden_size} dtype={host.dtype} device={host.device}")

    producers = [Producer.make(k) for k in ("tickdelta", "disclosure", "anomaly")]
    producers.append(QuantitativeProducer.make("quote_series"))
    herbivores = [Herbivore.make(k, capacity=cfg.population.intake_budget)
                  for k in ("technical", "fundamental")]
    predator = Predator.make("short_horizon")

    print(f"[init] loading StockNet train + dev splits…")
    train_scens = build_stocknet_scenarios(split="train", tickers=None, max_per_ticker=None)
    dev_scens = build_stocknet_scenarios(split="dev", tickers=None, max_per_ticker=None)
    print(f"[init] StockNet: {len(train_scens)} train, {len(dev_scens)} dev")

    dev_eval_cap = int(os.environ.get("TROPHIC_DEV_EVAL_CAP", "20"))
    if len(dev_scens) > dev_eval_cap:
        rng_dev = random.Random(sft_cfg.seed + 1)
        dev_eval = rng_dev.sample(dev_scens, dev_eval_cap)
        print(f"[init] dev eval capped to {dev_eval_cap}")
    else:
        dev_eval = dev_scens

    runner = SFTRunner(
        cfg=sft_cfg, host=host,
        producers=producers, herbivores=herbivores, predator=predator,
        train=train_scens, eval_=dev_eval,
    )

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
                    if done % 200 == 0:
                        print(f"[cache] {done} producer forwards "
                              f"(elapsed {time.time()-t0:.1f}s)")
        runner._producer_cache[sc.name] = items
    print(f"[cache] DONE: {done} forwards in {time.time()-t0:.1f}s")

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_best = ckpt_dir / f"sft_seed{sft_cfg.seed}_stable_best.pt"
    ckpt_final = ckpt_dir / f"sft_seed{sft_cfg.seed}_stable_final.pt"
    best_eval = float("inf")

    print(f"\n=== TRAINING (StockNet train, stable infra) ===")
    n = sft_cfg.steps
    nan_streak = 0
    for step in range(1, n + 1):
        sc = rng.choice(train_scens)
        result = runner.step(sc, train_step=step)
        loss_after = runner.maybe_step(step)

        # NaN guard: if 5 consecutive train steps NaN, abort early.
        if not isinstance(result.get("loss"), (int, float)) or \
           result["loss"] != result["loss"]:  # NaN check
            nan_streak += 1
            if nan_streak >= 5:
                print(f"\n[ABORT] 5 consecutive NaN steps; bailing at step {step}.")
                break
        else:
            nan_streak = 0

        if step % sft_cfg.log_every == 0:
            ploss = " ".join(
                f"{k}={v:.3f}" for k, v in result["per_loss"].items()
                if isinstance(v, (int, float))
            )
            tau = result.get("tau", 1.0)
            snap = runner.ecology_snapshot()
            extra = f" tau={tau:.3f} alpha={snap['skip_weight_alpha']:.3f}"
            if "trough_n_alive" in snap:
                extra += f" trough_alive={snap['trough_n_alive']}"
            print(f"[step {step:4d}] lr={runner.trainer.current_lr:.2e}"
                  f" sc={sc.name[:36]:36s} train_loss={result['loss']:.3f}"
                  f"  ({ploss}){extra}")

        if step % sft_cfg.eval_every == 0:
            print(f"\n--- dev eval @ step {step} ---")
            elr = runner.eval_loss()
            cur_lr, _ = runner.trainer.on_eval(elr["mean"])
            pk = " ".join(f"{k}={v:.3f}" for k, v in elr["per_kind"].items())
            print(f"  EVAL_LOSS: mean={elr['mean']:.4f}  ({pk})  lr→{cur_lr:.2e}")
            if elr["mean"] < best_eval and elr["mean"] == elr["mean"]:  # finite + improvement
                best_eval = elr["mean"]
                save_channels(
                    str(ckpt_best),
                    herbivores=herbivores, predators=[predator],
                    meta={"sft_seed": sft_cfg.seed, "step": step,
                          "eval_loss": elr["mean"], "split": "stocknet_train_stable",
                          "dstar_scale_override": os.environ.get("TROPHIC_DSTAR_SCALE_OVERRIDE", ""),
                          "lr": sft_cfg.lr,
                          "grad_clip": sft_cfg.grad_clip},
                    runner=runner,
                )
                print(f"  [ckpt] new best ({elr['mean']:.4f}) → {ckpt_best.name}")
            print()

    save_channels(
        str(ckpt_final),
        herbivores=herbivores, predators=[predator],
        meta={"sft_seed": sft_cfg.seed, "step": step,
              "eval_loss": best_eval, "split": "stocknet_train_stable"},
        runner=runner,
    )
    print(f"[ckpt] final → {ckpt_final.name}; best dev={best_eval:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
