"""IPO training: refine post-SFT predator Channels with paired oracle preferences.

Loads sft_seed1_best.pt, runs IPO with (chosen=oracle target, rejected=
predator sample). No judge calls during training; judge is only used for
periodic eval inspection.
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
from trophic.predators.judge_router import JudgeRouter
from trophic.predators.self_judge import SelfJudge
from trophic.training.checkpoint import load_channels, save_channels
from trophic.training.ipo import IPOConfig, IPORunner
from trophic.training.scenarios import build_scenarios, split


async def main() -> None:
    cfg = DEFAULT_CONFIG
    ipo_cfg = IPOConfig(
        lr=float(os.environ.get("TROPHIC_IPO_LR", "1e-4")),
        beta=float(os.environ.get("TROPHIC_IPO_BETA", "0.1")),
        steps=int(os.environ.get("TROPHIC_IPO_STEPS", "800")),
        log_every=int(os.environ.get("TROPHIC_LOG_EVERY", "25")),
        eval_every=int(os.environ.get("TROPHIC_EVAL_EVERY", "100")),
        sampling_temperature=float(os.environ.get("TROPHIC_TEMP", "1.0")),
        seed=int(os.environ.get("TROPHIC_SEED", "1")),
    )
    ckpt_in = os.environ.get(
        "TROPHIC_CKPT", str(ROOT / "checkpoints" / f"sft_seed{ipo_cfg.seed}_best.pt")
    )
    print(f"[init] ipo_cfg={ipo_cfg}")
    print(f"[init] ckpt_in={ckpt_in}")

    rng = random.Random(ipo_cfg.seed)

    print(f"[init] loading Qwen3-4B (mock={cfg.model.mock})")
    host = ModelHost.get(cfg.model)
    print(f"[init] hidden_size={host.hidden_size} dtype={host.dtype} device={host.device}")

    herbivores = [Herbivore.make(k, capacity=cfg.population.intake_budget)
                  for k in ("technical", "fundamental")]
    predator = Predator.make("short_horizon")
    for h in herbivores:
        h.ensure_initialized(host, seed_base=ipo_cfg.seed)
    predator.ensure_initialized(host, seed_base=ipo_cfg.seed)
    for h in herbivores:
        for ch in h.channels.values():
            ch.to(device=host.device, dtype=host.dtype)
    for ch in predator.channels.values():
        ch.to(device=host.device, dtype=host.dtype)

    if Path(ckpt_in).exists():
        meta = load_channels(ckpt_in, herbivores=herbivores, predators=[predator])
        print(f"[init] loaded checkpoint meta: {meta}")
    else:
        print(f"[warn] checkpoint not found at {ckpt_in}; running IPO from random init")

    for ch in predator.channels.values():
        ch.to(device=host.device, dtype=host.dtype)
    for h in herbivores:
        for ch in h.channels.values():
            ch.to(device=host.device, dtype=host.dtype)

    # Judge for eval-only inspection (no training signal)
    judge = JudgeRouter(self_judge=SelfJudge(host=host), apex_judge=None)

    scenarios = build_scenarios()
    train, eval_ = split(scenarios)
    print(f"[init] scenarios: {len(train)} train, {len(eval_)} eval")

    runner = IPORunner(
        cfg=ipo_cfg, host=host,
        herbivores=herbivores, predator=predator,
        train=train, eval_=eval_,
    )

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_best = ckpt_dir / f"ipo_seed{ipo_cfg.seed}_best.pt"
    ckpt_final = ckpt_dir / f"ipo_seed{ipo_cfg.seed}_final.pt"
    best_eval_score = -1.0

    print("\n=== IPO ===")
    n = ipo_cfg.steps
    for step in range(1, n + 1):
        sc = rng.choice(train)
        result = runner.step(sc, tick=step)
        if step % ipo_cfg.log_every == 0:
            if result.get("skipped"):
                print(f"[step {step:4d}] skipped sc={sc.name[:32]} reason={result.get('reason')}")
            else:
                print(
                    f"[step {step:4d}] loss={result['loss']:.4f}  h_w={result['h_w']:+.3f}"
                    f"  Δlogπ_c={result['log_pi_c']-result['log_ref_c']:+.2f}"
                    f"  Δlogπ_r={result['log_pi_r']-result['log_ref_r']:+.2f}"
                    f"  sc={sc.name[:32]}"
                )
                print(f"   rejected: {result['rejected_text']}")

        if step % ipo_cfg.eval_every == 0:
            print(f"\n--- eval @ step {step} ---")
            scores = []
            for sc_eval in eval_:
                r = await runner.eval_decode(sc_eval, tick=step, judge=judge)
                if "score" in r:
                    scores.append(r["score"])
                    print(f"  [{r['name'][:36]}] score={r['score']:.2f}  {r['decoded']}")
                else:
                    print(f"  [{r['name'][:36]}] {r['decoded']}")
            if scores:
                eval_mean = sum(scores) / len(scores)
                print(f"  EVAL_MEAN_SCORE: {eval_mean:.3f}")
                if eval_mean > best_eval_score:
                    best_eval_score = eval_mean
                    save_channels(
                        str(ckpt_best),
                        herbivores=herbivores, predators=[predator],
                        meta={"ipo_seed": ipo_cfg.seed, "step": step,
                              "eval_score": eval_mean},
                    )
                    print(f"  [ckpt] new best ({eval_mean:.3f}) → {ckpt_best.name}")
            print()

    # Final checkpoint
    save_channels(
        str(ckpt_final),
        herbivores=herbivores, predators=[predator],
        meta={"ipo_seed": ipo_cfg.seed, "step": n, "eval_score": best_eval_score},
    )
    print(f"[ckpt] final → {ckpt_final.name}; best eval_score={best_eval_score:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
