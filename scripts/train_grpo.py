"""GRPO training: refine post-SFT predator Channels with reward signal.

Loads the post-SFT Channel checkpoint, freezes a snapshot for KL reference,
trains the live Channels via group-relative policy optimization. Self-judge
(Qwen-4B in-process) scores completions; auto-escalates to Bedrock when
self-judge variance trips.

Usage:
  TROPHIC_CKPT=checkpoints/sft_seed1_best.pt \
  TROPHIC_GRPO_STEPS=200 \
  python scripts/train_grpo.py
"""
from __future__ import annotations

import asyncio
import functools
import os
import random
import sys
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from trophic.agents.herbivore import Herbivore
from trophic.agents.predator import Predator
from trophic.agents.producer import Producer
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.predators.apex_judge import ApexJudge
from trophic.predators.judge_router import JudgeRouter
from trophic.predators.self_judge import SelfJudge
from trophic.training.checkpoint import load_channels
from trophic.training.grpo import GRPOConfig, GRPORunner
from trophic.training.scenarios import build_scenarios, split


async def main() -> None:
    cfg = DEFAULT_CONFIG
    grpo_cfg = GRPOConfig(
        lr=float(os.environ.get("TROPHIC_GRPO_LR", "1e-4")),
        steps=int(os.environ.get("TROPHIC_GRPO_STEPS", "200")),
        log_every=int(os.environ.get("TROPHIC_LOG_EVERY", "5")),
        eval_every=int(os.environ.get("TROPHIC_EVAL_EVERY", "25")),
        consistency_probe_every=int(os.environ.get("TROPHIC_PROBE_EVERY", "10")),
        seed=int(os.environ.get("TROPHIC_SEED", "1")),
        group_size=int(os.environ.get("TROPHIC_GROUP_SIZE", "4")),
        kl_beta=float(os.environ.get("TROPHIC_KL_BETA", "0.02")),
        sampling_temperature=float(os.environ.get("TROPHIC_TEMP", "0.8")),
    )
    ckpt_path = os.environ.get(
        "TROPHIC_CKPT", str(ROOT / "checkpoints" / f"sft_seed{grpo_cfg.seed}_best.pt")
    )
    print(f"[init] grpo_cfg={grpo_cfg}")
    print(f"[init] ckpt={ckpt_path}")

    rng = random.Random(grpo_cfg.seed)

    print(f"[init] loading Qwen3-4B (mock={cfg.model.mock})")
    host = ModelHost.get(cfg.model)
    print(f"[init] hidden_size={host.hidden_size} dtype={host.dtype} device={host.device}")

    # Build agents (same as SFT)
    herbivores = [Herbivore.make(k, capacity=cfg.population.intake_budget)
                  for k in ("technical", "fundamental")]
    predator = Predator.make("short_horizon")

    # Init Channels deterministically (SAME seed_base as SFT — must match)
    for h in herbivores:
        h.ensure_initialized(host, seed_base=grpo_cfg.seed)
    predator.ensure_initialized(host, seed_base=grpo_cfg.seed)

    # Move Channels to device/dtype
    for h in herbivores:
        for ch in h.channels.values():
            ch.to(device=host.device, dtype=host.dtype)
    for ch in predator.channels.values():
        ch.to(device=host.device, dtype=host.dtype)

    # Load SFT checkpoint
    if Path(ckpt_path).exists():
        meta = load_channels(ckpt_path, herbivores=herbivores, predators=[predator])
        print(f"[init] loaded checkpoint meta: {meta}")
    else:
        print(f"[warn] checkpoint not found at {ckpt_path}; running GRPO from scratch")

    # Re-move after loading (state_dict load may put params on cpu)
    for ch in predator.channels.values():
        ch.to(device=host.device, dtype=host.dtype)
    for h in herbivores:
        for ch in h.channels.values():
            ch.to(device=host.device, dtype=host.dtype)

    # Judges
    self_j = SelfJudge(host=host)
    use_bedrock = os.environ.get("TROPHIC_DISABLE_BEDROCK", "") != "1"
    apex_j = ApexJudge.from_config(cfg.bedrock) if use_bedrock else None
    judge = JudgeRouter(self_judge=self_j, apex_judge=apex_j)
    print(f"[init] judge: self_j={self_j.name} apex_j={'bedrock' if apex_j else 'disabled'}")

    # Scenarios
    scenarios = build_scenarios()
    train, eval_ = split(scenarios)
    # GRPO trains on training set only; eval set is for periodic decode checks.
    print(f"[init] scenarios: {len(train)} train, {len(eval_)} eval")

    runner = GRPORunner(
        cfg=grpo_cfg, host=host,
        herbivores=herbivores, predator=predator,
        train=train, eval_=eval_,
        judge=judge,
    )

    # ---------- training loop ----------
    print("\n=== GRPO ===")
    n = grpo_cfg.steps
    for step in range(1, n + 1):
        sc = rng.choice(train)
        result = await runner.step(sc, tick=step)

        if step % grpo_cfg.log_every == 0:
            rmin = min(result["rewards"])
            rmax = max(result["rewards"])
            print(
                f"[step {step:4d}] loss={result['loss']:.4f}"
                f"  rewards={result['rewards_mean']:.3f}±{result['rewards_std']:.3f}"
                f"  range=[{rmin:.2f},{rmax:.2f}]"
                f"  judge={result['judge']}  sc={sc.name[:32]}"
            )

        # Self-consistency probe → maybe escalate
        if step % grpo_cfg.consistency_probe_every == 0:
            # Use a recent eval scenario for the probe — we want a real
            # broadcast not a synthetic one.
            probe_sc = rng.choice(eval_)
            probe_result = await runner.eval_decode(probe_sc, tick=step)
            # Probe variance via the router
            from trophic.types import Broadcast
            br = Broadcast(
                id=f"probe.{step}",
                tier="predator_broadcast",
                agent_id=predator.id,
                agent_kind=predator.kind,
                diet_tags=[f"from_{predator.kind}_predator"],
                decoded_text=probe_result["decoded"],
                channel_embedding=[],
                created_tick=step,
            )
            probe = await judge.probe(br, step)
            print(
                f"[probe @ {step}] sc={probe_sc.name[:24]} variance={probe.variance:.4f}"
                f" scores={probe.scores}  judge_now={judge.current_judge_name}"
            )

        # Periodic full eval (decode + score on held-out)
        if step % grpo_cfg.eval_every == 0:
            print(f"\n--- eval @ step {step} ---")
            scores = []
            for sc_eval in eval_:
                r = await runner.eval_decode(sc_eval, tick=step)
                scores.append(r["score"])
                print(f"  [{r['name'][:36]}] score={r['score']:.2f}  judge={r['judge']}")
                print(f"     {r['decoded']}")
                print(f"     JDG: {r['rationale']}")
            print(f"  EVAL_MEAN_SCORE: {sum(scores)/max(1,len(scores)):.3f}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
