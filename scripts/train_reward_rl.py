"""Reward-driven RL training. Loads SFT-XML checkpoint, samples G=4
completions per scenario with channel-side noise, scores via rule-based
parser, group-relative advantage, KL-anchored policy gradient on Predator
Channels only.
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

import torch

from trophic.agents.herbivore import Herbivore
from trophic.agents.predator import Predator
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.training.checkpoint import load_channels, save_channels
from trophic.training.reward_rl import RewardRLConfig, RewardRLRunner
from trophic.training.scenarios import build_scenarios, split


async def main() -> None:
    cfg = DEFAULT_CONFIG
    rl_cfg = RewardRLConfig(
        lr=float(os.environ.get("TROPHIC_RL_LR", "5e-5")),
        steps=int(os.environ.get("TROPHIC_RL_STEPS", "300")),
        log_every=int(os.environ.get("TROPHIC_LOG_EVERY", "10")),
        eval_every=int(os.environ.get("TROPHIC_EVAL_EVERY", "50")),
        sampling_temperature=float(os.environ.get("TROPHIC_TEMP", "0.9")),
        channel_noise_std=float(os.environ.get("TROPHIC_NOISE", "0.05")),
        kl_beta=float(os.environ.get("TROPHIC_KL_BETA", "0.05")),
        group_size=int(os.environ.get("TROPHIC_GROUP_SIZE", "4")),
        seed=int(os.environ.get("TROPHIC_SEED", "1")),
    )
    ckpt_in = os.environ.get(
        "TROPHIC_CKPT", str(ROOT / "checkpoints" / f"sft_seed{rl_cfg.seed}_best.pt")
    )
    print(f"[init] rl_cfg={rl_cfg}")
    print(f"[init] ckpt_in={ckpt_in}")

    rng = random.Random(rl_cfg.seed)

    host = ModelHost.get(cfg.model)
    print(f"[init] hidden_size={host.hidden_size} dtype={host.dtype} device={host.device}")

    herbivores = [Herbivore.make(k, capacity=cfg.population.intake_budget)
                  for k in ("technical", "fundamental")]
    predator = Predator.make("short_horizon")
    for h in herbivores:
        h.ensure_initialized(host, seed_base=rl_cfg.seed)
    predator.ensure_initialized(host, seed_base=rl_cfg.seed)
    for h in herbivores:
        for ch in h.channels.values():
            ch.to(device=host.device, dtype=host.dtype)
    for ch in predator.channels.values():
        ch.to(device=host.device, dtype=host.dtype)

    if Path(ckpt_in).exists():
        meta = load_channels(ckpt_in, herbivores=herbivores, predators=[predator])
        print(f"[init] loaded checkpoint meta: {meta}")
    else:
        print(f"[warn] checkpoint not found at {ckpt_in}; starting from random init")

    for ch in predator.channels.values():
        ch.to(device=host.device, dtype=host.dtype)
    for h in herbivores:
        for ch in h.channels.values():
            ch.to(device=host.device, dtype=host.dtype)

    scenarios = build_scenarios()
    train, eval_ = split(scenarios)
    print(f"[init] scenarios: {len(train)} train, {len(eval_)} eval")

    runner = RewardRLRunner(
        cfg=rl_cfg, host=host,
        herbivores=herbivores, predator=predator,
        train=train, eval_=eval_,
    )

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_best = ckpt_dir / f"reward_rl_seed{rl_cfg.seed}_best.pt"
    ckpt_final = ckpt_dir / f"reward_rl_seed{rl_cfg.seed}_final.pt"
    best_eval = -1.0

    print("\n=== Reward-driven RL ===")
    n = rl_cfg.steps
    for step in range(1, n + 1):
        sc = rng.choice(train)
        result = runner.step(sc, tick=step)

        if step % rl_cfg.log_every == 0:
            if result.get("skipped"):
                print(f"[step {step:4d}] skipped sc={sc.name[:32]}")
            else:
                print(
                    f"[step {step:4d}] loss={result['loss']:.4f}"
                    f"  rewards={result['rewards_mean']:.3f}±{result['rewards_std']:.3f}"
                    f"  max={result['rewards_max']:.3f}"
                    f"  sc={sc.name[:32]}"
                )
                print(f"   best: {result['decoded_best']}")

        if step % rl_cfg.eval_every == 0:
            print(f"\n--- eval @ step {step} ---")
            scores = []
            for sc_eval in eval_:
                r = runner.eval_decode(sc_eval)
                scores.append(r["reward"])
                bd = r["breakdown"]
                br_str = " ".join(f"{k}={v:.2f}" for k, v in bd.items() if v > 0)
                print(f"  [{r['name'][:36]}] reward={r['reward']:.3f}  ({br_str})")
                print(f"     {r['decoded']}")
            mean = sum(scores) / max(1, len(scores))
            print(f"  EVAL_MEAN_REWARD: {mean:.3f}")
            if mean > best_eval:
                best_eval = mean
                save_channels(
                    str(ckpt_best),
                    herbivores=herbivores, predators=[predator],
                    meta={"rl_seed": rl_cfg.seed, "step": step, "eval_reward": mean},
                )
                print(f"  [ckpt] new best ({mean:.3f}) → {ckpt_best.name}")
            print()

    save_channels(
        str(ckpt_final),
        herbivores=herbivores, predators=[predator],
        meta={"rl_seed": rl_cfg.seed, "step": n, "eval_reward": best_eval},
    )
    print(f"[ckpt] final → {ckpt_final.name}; best eval_reward={best_eval:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
