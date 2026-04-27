"""Evaluate the predator with the LoRA-enhanced live Interrogator in the loop.

This is the validation test for the bottom-up LoRA hypothesis. We:
  1. Load the post-IPO predator + herbivore Channels (ipo_seed1_best.pt)
  2. Load the LoRA adapter onto Qwen3-4B
  3. For each held-out scenario, run the FULL live pipeline:
     producers → herbivores (technical/fundamental run forward) →
     interrogator (uses LoRA-enhanced Qwen for plan + math + synthesize) →
     predator (attends to all four herbivore broadcasts including
     the live interrogator broadcast)
  4. Parse the predator's XML and score with the rule-based reward

If the rule-based reward jumps materially above the IPO-only baseline of
0.388, LoRA on the interrogator is doing real work upstream. If it stays
the same, the predator's attention is the bottleneck.
"""
from __future__ import annotations

import asyncio
import functools
import os
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from trophic.agents.herbivore import Herbivore
from trophic.agents.interrogator_herbivore import InterrogatorHerbivore
from trophic.agents.predator import Predator
from trophic.agents.producer import Producer
from trophic.agents.quant_producer import QuantitativeProducer
from trophic.config import DEFAULT_CONFIG
from trophic.math_host import MathHost
from trophic.model_host import ModelHost
from trophic.training.checkpoint import load_channels
from trophic.training.scenarios import build_scenarios, split
from trophic.training.xml_schema import (
    parse_prediction, parse_synthesis, reward_prediction, reward_synthesis,
)
from trophic.types import Broadcast


def load_lora(host: ModelHost, lora_dir: str) -> None:
    """Wrap host._model with the LoRA adapter."""
    from peft import PeftModel
    print(f"[lora] loading adapter from {lora_dir}")
    host._model = PeftModel.from_pretrained(host._model, lora_dir)
    host._model.eval()
    n_lora = sum(p.numel() for p in host._model.parameters() if p.requires_grad)
    print(f"[lora] adapter active. trainable params (eval mode, irrelevant): {n_lora:,}")


async def main() -> None:
    cfg = DEFAULT_CONFIG
    use_lora = os.environ.get("TROPHIC_USE_LORA", "1") == "1"
    ckpt_path = os.environ.get(
        "TROPHIC_CKPT", str(ROOT / "checkpoints" / "ipo_seed1_best.pt")
    )
    lora_dir = os.environ.get(
        "TROPHIC_LORA_DIR", str(ROOT / "checkpoints" / "interrogator_lora" / "seed1")
    )
    print(f"[init] use_lora={use_lora} ckpt={ckpt_path}")

    host = ModelHost.get(cfg.model)
    math_host = MathHost.get(cfg.model)
    print(f"[init] hidden={host.hidden_size} math_hidden={math_host.hidden_size}")

    # Load LoRA *before* building Channels so the channel hidden-state
    # space is the LoRA-enhanced Qwen's space throughout.
    if use_lora:
        load_lora(host, lora_dir)

    # Build agents
    producers = [Producer.make(k) for k in ("tickdelta", "disclosure", "anomaly")]
    producers.append(QuantitativeProducer.make("quote_series"))
    herbivores = [Herbivore.make(k, capacity=cfg.population.intake_budget)
                  for k in ("technical", "fundamental")]
    interrogator = InterrogatorHerbivore.make(use_channel=False)
    predator = Predator.make("short_horizon")

    for h in herbivores:
        h.ensure_initialized(host, seed_base=1)
    interrogator.ensure_initialized(host, seed_base=1)
    predator.ensure_initialized(host, seed_base=1)
    for h in herbivores:
        for ch in h.channels.values():
            ch.to(device=host.device, dtype=host.dtype)
    for ch in predator.channels.values():
        ch.to(device=host.device, dtype=host.dtype)

    # Load post-IPO Channel weights into the herbivore + predator Channels
    meta = load_channels(ckpt_path, herbivores=herbivores, predators=[predator], strict=False)
    print(f"[init] loaded channel ckpt: {meta}")
    for h in herbivores:
        for ch in h.channels.values():
            ch.to(device=host.device, dtype=host.dtype)
    for ch in predator.channels.values():
        ch.to(device=host.device, dtype=host.dtype)

    scenarios = build_scenarios()
    train, eval_ = split(scenarios)
    print(f"[init] eval scenarios: {len(eval_)}")

    print("\n=== EVAL with LIVE interrogator (LoRA-active=" + str(use_lora) + ") ===")
    rewards = []
    interrog_rewards = []  # how well the interrogator did on its own

    for sc in eval_:
        # Step 1: producers fire
        producer_broadcasts: list[Broadcast] = []
        for inp in sc.inputs:
            for prod in producers:
                if prod.attracts(inp):
                    b = await prod.produce(inp, tick=0, host=host)
                    if b is not None:
                        producer_broadcasts.append(b)

        # Step 2: live herbivores forward
        herb_broadcasts: list[Broadcast] = []
        # Technical and fundamental herbs use Channel attention
        for h in herbivores:
            br, _claimed, _rej, _stats = await h.hunt_and_synthesize(
                producer_broadcasts, tick=0, host=host,
            )
            herb_broadcasts.append(br)

        # Interrogator runs full pipeline (plan → math → synthesize)
        i_br, _, _, i_stats = await interrogator.hunt_and_synthesize(
            producer_broadcasts, tick=0, host=host, math_host=math_host,
        )
        herb_broadcasts.append(i_br)

        # Score the interrogator independently against its target
        if sc.interrogator_target:
            i_target = parse_synthesis(sc.interrogator_target)
            i_parsed = parse_synthesis(i_br.decoded_text or "")
            i_rew = reward_synthesis(i_parsed, i_target)
            interrog_rewards.append(i_rew["total"])
        else:
            interrog_rewards.append(None)

        # Step 3: predator forward (attends to all four herb broadcasts)
        pred_br, _claimed, _rej, _pstats = await predator.hunt_and_predict(
            herb_broadcasts, tick=0, host=host,
        )

        # Score predator
        target = parse_prediction(sc.predator_target or "")
        parsed = parse_prediction(pred_br.decoded_text or "")
        rew = reward_prediction(parsed, target)
        rewards.append(rew["total"])

        i_text = (i_br.decoded_text or "")[:200]
        p_text = (pred_br.decoded_text or "")[:200]
        i_str = f"{interrog_rewards[-1]:.2f}" if interrog_rewards[-1] is not None else "n/a"
        print(f"\n[{sc.name}]  pred_reward={rew['total']:.3f}  interrog_reward={i_str}")
        print(f"  TARGET: ticker={target.ticker} dir={target.direction} pct={target.pct_move}")
        print(f"  INTER:  {i_text}")
        print(f"  PRED:   {p_text}")
        print(f"  PARSED: ticker={parsed.ticker} dir={parsed.direction} pct={parsed.pct_move}")

    valid_irew = [r for r in interrog_rewards if r is not None]
    print(f"\n=== MEAN PREDATOR REWARD: {sum(rewards)/len(rewards):.3f} (over {len(rewards)} scenarios) ===")
    if valid_irew:
        print(f"=== MEAN INTERROGATOR REWARD: {sum(valid_irew)/len(valid_irew):.3f} (over {len(valid_irew)} scenarios) ===")


if __name__ == "__main__":
    asyncio.run(main())
