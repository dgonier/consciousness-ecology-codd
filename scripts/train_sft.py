"""Run the SFT phase: teacher-forcing CE on target tokens.

Loads Qwen3-4B locally, builds scenarios + producers + herbivores +
predator, and trains only the Channel parameters until target outputs
become parseable.
"""
from __future__ import annotations

import asyncio
import functools
import os
import random
import sys
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001 — line-buffer to logfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from trophic.agents.herbivore import Herbivore
from trophic.agents.predator import Predator
from trophic.agents.producer import Producer
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.training.checkpoint import save_channels
from trophic.training.scenarios import build_scenarios, split
from trophic.training.sft import SFTConfig, SFTRunner


async def main() -> None:
    cfg = DEFAULT_CONFIG
    sft_cfg = SFTConfig(
        lr=float(os.environ.get("TROPHIC_LR", "5e-4")),
        steps=int(os.environ.get("TROPHIC_STEPS", "200")),
        log_every=int(os.environ.get("TROPHIC_LOG_EVERY", "10")),
        eval_every=int(os.environ.get("TROPHIC_EVAL_EVERY", "25")),
        seed=int(os.environ.get("TROPHIC_SEED", "7")),
        content_token_weight=float(os.environ.get("TROPHIC_CONTENT_W", "1.0")),
    )
    print(f"[init] seed={sft_cfg.seed} lr={sft_cfg.lr} steps={sft_cfg.steps}"
          f" content_w={sft_cfg.content_token_weight}")
    rng = random.Random(sft_cfg.seed)

    print(f"[init] loading Qwen3-4B (mock={cfg.model.mock})")
    host = ModelHost.get(cfg.model)
    print(f"[init] hidden_size={host.hidden_size} dtype={host.dtype} device={host.device}")

    # Issue #10 phase 1: optionally attach a LoRA adapter to the base model
    # BEFORE Channel training, so Channels learn against the LoRA-modified
    # hidden-state distribution (otherwise Channels and base drift apart —
    # see #12, and the schema-collapse finding from the partial integration
    # test). Co-training is the architecturally correct approach.
    _lora_dir = os.environ.get("TROPHIC_LORA_DIR", "")
    if _lora_dir:
        from peft import PeftModel
        print(f"[init] attaching LoRA from {_lora_dir}")
        host._model = PeftModel.from_pretrained(host._model, _lora_dir, is_trainable=False)
        host._model.eval()  # adapter frozen; we're training Channels, not the LoRA
        print(f"[init] LoRA active (frozen for SFT — only Channels train)")

    # Agents — one of each kind
    producers = [Producer.make(k) for k in ("tickdelta", "disclosure", "anomaly")]
    herbivores = [Herbivore.make(k, capacity=cfg.population.intake_budget)
                  for k in ("technical", "fundamental")]
    predator = Predator.make("short_horizon")

    scenarios = build_scenarios()
    train, eval_ = split(scenarios)
    print(f"[init] scenarios: {len(train)} train, {len(eval_)} eval")

    runner = SFTRunner(
        cfg=sft_cfg, host=host,
        producers=producers, herbivores=herbivores, predator=predator,
        train=train, eval_=eval_,
    )

    n_inputs = sum(len(sc.inputs) for sc in scenarios)
    print(f"[init] caching producer broadcasts for {len(scenarios)} scenarios"
          f" ({n_inputs} producer forwards)…")
    import time
    t0 = time.time()
    done = 0
    for sc in scenarios:
        items = []
        for inp in sc.inputs:
            for prod in producers:
                if prod.attracts(inp):
                    b = await prod.produce(inp, tick=0, host=host)
                    if b is not None:
                        items.append(b)
                    done += 1
                    if done % 10 == 0:
                        print(f"[cache] {done} producer forwards done"
                              f" (elapsed {time.time()-t0:.1f}s)")
        runner._producer_cache[sc.name] = items
    print(f"[cache] DONE: {done} forwards in {time.time()-t0:.1f}s")

    # Initial eval — pure baseline before any training
    print("\n=== BASELINE (pre-training) ===")
    for sc in eval_[:2]:
        out = runner.eval_decode(sc)
        print(f"  [{sc.name}]")
        for k, v in out.items():
            if k == "name":
                continue
            print(f"    {k}: {v}")

    # Checkpoint paths
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_best = ckpt_dir / f"sft_seed{sft_cfg.seed}_best.pt"
    ckpt_final = ckpt_dir / f"sft_seed{sft_cfg.seed}_final.pt"
    best_eval = float("inf")

    print("\n=== TRAINING ===")
    n = sft_cfg.steps
    for step in range(1, n + 1):
        sc = rng.choice(train)
        result = runner.step(sc, train_step=step)
        loss_after = runner.maybe_step(step)
        if step % sft_cfg.log_every == 0:
            ploss = " ".join(f"{k}={v:.3f}" for k, v in result["per_loss"].items())
            tau = result.get("tau", 1.0)
            # Mission 06: per-epoch ecology snapshot.
            snap = runner.ecology_snapshot()
            extra = (
                f" tau={tau:.3f}"
                f" alpha={snap['skip_weight_alpha']:.3f}"
            )
            if "trough_n_alive" in snap:
                extra += (
                    f" trough_alive={snap['trough_n_alive']}"
                    f" head_ent={snap['head_specialization_entropy']:.3f}"
                )
            print(f"[step {step:4d}] lr={runner.trainer.current_lr:.2e}"
                  f" sc={sc.name:32s} train_loss={result['loss']:.3f}"
                  f"  ({ploss}){extra}")
        if step % sft_cfg.eval_every == 0:
            print(f"\n--- eval @ step {step} ---")
            elr = runner.eval_loss()
            cur_lr, _ = runner.trainer.on_eval(elr["mean"])
            pk = " ".join(f"{k}={v:.3f}" for k, v in elr["per_kind"].items())
            print(f"  EVAL_LOSS: mean={elr['mean']:.4f}  ({pk})  lr→{cur_lr:.2e}")
            # Save best-eval checkpoint
            if elr["mean"] < best_eval:
                best_eval = elr["mean"]
                save_channels(
                    str(ckpt_best),
                    herbivores=herbivores, predators=[predator],
                    meta={"sft_seed": sft_cfg.seed, "step": step,
                          "eval_loss": elr["mean"]},
                )
                print(f"  [ckpt] new best ({elr['mean']:.4f}) → {ckpt_best.name}")
            for sc_eval in eval_:
                out = runner.eval_decode(sc_eval)
                print(f"  [{sc_eval.name}]")
                for k, v in out.items():
                    if k == "name":
                        continue
                    print(f"    {k}: {v}")
            print()

    # Final checkpoint (whether or not it's best)
    save_channels(
        str(ckpt_final),
        herbivores=herbivores, predators=[predator],
        meta={"sft_seed": sft_cfg.seed, "step": n, "eval_loss": best_eval},
    )
    print(f"[ckpt] final → {ckpt_final.name}; best eval={best_eval:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
