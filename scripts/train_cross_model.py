"""Train the InterrogatorHerbivore's CrossModelChannel (E_analyst_math).

Isolated training run: only the cross-model projection MLPs train.
Qwen3-4B, Qwen2.5-Math-1.5B, all herbivore/predator Channels stay frozen.

Usage:
  TROPHIC_CMRL_STEPS=200 python scripts/train_cross_model.py
"""
from __future__ import annotations

import asyncio
import functools
import os
import random
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trophic.agents.interrogator_herbivore import InterrogatorHerbivore
from trophic.config import DEFAULT_CONFIG
from trophic.math_host import MathHost
from trophic.model_host import ModelHost
from trophic.training.checkpoint import save_cross_model_channel
from trophic.training.cross_model_pretrain import (
    PretrainConfig, pretrain_cross_model_channel,
)
from trophic.training.cross_model_rl import CrossModelRLConfig, CrossModelRLRunner
from trophic.training.scenarios import build_scenarios, split


async def main() -> None:
    cfg = DEFAULT_CONFIG
    rl_cfg = CrossModelRLConfig(
        lr=float(os.environ.get("TROPHIC_CMRL_LR", "5e-5")),
        steps=int(os.environ.get("TROPHIC_CMRL_STEPS", "200")),
        log_every=int(os.environ.get("TROPHIC_LOG_EVERY", "10")),
        eval_every=int(os.environ.get("TROPHIC_EVAL_EVERY", "50")),
        sampling_temperature=float(os.environ.get("TROPHIC_TEMP", "0.7")),
        group_size=int(os.environ.get("TROPHIC_GROUP_SIZE", "4")),
        kl_beta=float(os.environ.get("TROPHIC_KL_BETA", "0.0")),
        seed=int(os.environ.get("TROPHIC_SEED", "1")),
    )
    print(f"[init] rl_cfg={rl_cfg}")

    rng = random.Random(rl_cfg.seed)

    print(f"[init] loading Qwen3-4B (mock={cfg.model.mock})")
    host = ModelHost.get(cfg.model)
    print(f"[init]   analyst hidden={host.hidden_size} dtype={host.dtype} device={host.device}")

    print(f"[init] loading Qwen2.5-Math-1.5B")
    math_host = MathHost.get(cfg.model)
    print(f"[init]   math hidden={math_host.hidden_size} dtype={math_host.dtype} device={math_host.device}")

    interrogator = InterrogatorHerbivore.make(use_channel=True)
    interrogator.ensure_initialized(host, seed_base=rl_cfg.seed, math_host=math_host)
    interrogator.channel.to(device=host.device, dtype=host.dtype)
    n_params = sum(p.numel() for p in interrogator.channel.parameters())
    print(f"[init] cross-model channel params: {n_params:,}")

    # ------------------------------------------------------------------
    # Pre-training: MSE alignment so the projected embeddings decode to
    # something the math model can recognize. Without this, RL starts
    # from a near-noise channel and rewards are zero everywhere.
    # ------------------------------------------------------------------
    pretrain_steps = int(os.environ.get("TROPHIC_PRETRAIN_STEPS", "300"))
    if pretrain_steps > 0:
        print(f"\n=== Phase 1a: embedding-alignment pretraining ({pretrain_steps} steps) ===")
        pre_cfg = PretrainConfig(
            lr=float(os.environ.get("TROPHIC_PRETRAIN_LR", "1e-3")),
            steps=pretrain_steps,
            log_every=max(10, pretrain_steps // 20),
            seed=rl_cfg.seed,
        )
        pretrain_cross_model_channel(
            channel=interrogator.channel,
            host=host, math_host=math_host,
            cfg=pre_cfg,
        )
        print("[pretrain] complete\n")

    scenarios = build_scenarios()
    train_all, eval_all = split(scenarios)
    # Filter to scenarios with quantitative inputs
    def has_quant(sc):
        return any(inp.source in ("ohlcv", "quote_series") for inp in sc.inputs)
    train = [s for s in train_all if has_quant(s)]
    eval_ = [s for s in eval_all if has_quant(s)]
    print(f"[init] scenarios (quantitative subset): {len(train)} train, {len(eval_)} eval")

    runner = CrossModelRLRunner(
        cfg=rl_cfg, host=host, math_host=math_host,
        interrogator=interrogator,
        train=train, eval_=eval_,
    )

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_best = ckpt_dir / f"cross_model_seed{rl_cfg.seed}_best.pt"
    ckpt_final = ckpt_dir / f"cross_model_seed{rl_cfg.seed}_final.pt"
    best_eval = -1.0

    print("\n=== Cross-model channel training ===")
    n = rl_cfg.steps
    for step in range(1, n + 1):
        sc = rng.choice(train)
        result = await runner.step(sc, tick=step)

        if step % rl_cfg.log_every == 0:
            if result.get("skipped"):
                print(f"[step {step:4d}] skipped sc={sc.name[:32]} reason={result.get('reason')}")
            else:
                print(
                    f"[step {step:4d}] loss={result['loss']:.4f}"
                    f"  rewards={result['rewards_mean']:.3f}±{result['rewards_std']:.3f}"
                    f"  max={result['rewards_max']:.3f}"
                    f"  Q: {result.get('questions', [])}"
                )
                print(f"   best: {result['decoded_best']}")

        if step % rl_cfg.eval_every == 0:
            print(f"\n--- eval @ step {step} ---")
            scores = []
            for sc_eval in eval_[:5]:  # subset for speed
                r = await runner.eval_decode(sc_eval)
                scores.append(r["reward"])
                print(f"  [{r['name'][:36]}] reward={r['reward']:.3f}")
                print(f"     Q: {r.get('questions')}")
                print(f"     A: {r.get('answers')}")
                print(f"     {r['decoded']}")
            mean = sum(scores) / max(1, len(scores))
            print(f"  EVAL_MEAN_REWARD: {mean:.3f}")
            if mean > best_eval:
                best_eval = mean
                save_cross_model_channel(
                    str(ckpt_best),
                    channel=interrogator.channel,
                    meta={"seed": rl_cfg.seed, "step": step, "eval_reward": mean},
                )
                print(f"  [ckpt] new best ({mean:.3f}) → {ckpt_best.name}")
            print()

    save_cross_model_channel(
        str(ckpt_final),
        channel=interrogator.channel,
        meta={"seed": rl_cfg.seed, "step": n, "eval_reward": best_eval},
    )
    print(f"[ckpt] final → {ckpt_final.name}; best eval_reward={best_eval:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
