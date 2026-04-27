"""Diagnostic: run one GRPO step manually, print all 4 completions + judge raw output.

Helps figure out whether scores=0.0 is from (a) judge parse failures or (b)
predator producing garbage that judge correctly rates 0.
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
from trophic.agents.predator import Predator
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.predators.apex_judge import ApexJudge
from trophic.predators.judge_router import JudgeRouter
from trophic.predators.self_judge import SelfJudge, SYSTEM_PROMPT as SJ_SYSTEM
from trophic.training.checkpoint import load_channels
from trophic.training.grpo import GRPOConfig, GRPORunner
from trophic.training.scenarios import build_scenarios, split


async def main() -> None:
    cfg = DEFAULT_CONFIG
    grpo_cfg = GRPOConfig(seed=1, group_size=4, sampling_temperature=0.8)
    ckpt_path = str(ROOT / "checkpoints" / "sft_seed1_best.pt")

    host = ModelHost.get(cfg.model)

    herbivores = [Herbivore.make(k) for k in ("technical", "fundamental")]
    predator = Predator.make("short_horizon")
    for h in herbivores:
        h.ensure_initialized(host, seed_base=1)
    predator.ensure_initialized(host, seed_base=1)
    for h in herbivores:
        for ch in h.channels.values():
            ch.to(device=host.device, dtype=host.dtype)
    for ch in predator.channels.values():
        ch.to(device=host.device, dtype=host.dtype)
    load_channels(ckpt_path, herbivores=herbivores, predators=[predator])
    for ch in predator.channels.values():
        ch.to(device=host.device, dtype=host.dtype)
    for h in herbivores:
        for ch in h.channels.values():
            ch.to(device=host.device, dtype=host.dtype)

    self_j = SelfJudge(host=host)
    judge = JudgeRouter(self_judge=self_j, apex_judge=None)
    scenarios = build_scenarios()
    train, eval_ = split(scenarios)

    runner = GRPORunner(
        cfg=grpo_cfg, host=host,
        herbivores=herbivores, predator=predator,
        train=train, eval_=eval_, judge=judge,
    )

    sc = train[0]  # deterministic — the first scenario
    print(f"## SCENARIO: {sc.name}")
    print(f"## TARGET predator: {sc.predator_target}")
    print()

    # Build prefix
    candidates = runner._oracle_herb_broadcasts(sc)
    print(f"## n oracle herb broadcasts: {len(candidates)}")
    for c in candidates:
        print(f"   - kind={c.agent_kind} id={c.id}")
    print()

    with torch.no_grad():
        ch_out = runner._channel_output(runner.pred_stack.live, candidates)
        prefix_embeds, attn_mask = runner._build_prefix(ch_out)
    print(f"## prefix_embeds shape: {prefix_embeds.shape}")
    print()

    # Sample 4 completions and judge each one with full diagnostic output
    for i in range(4):
        tokens = runner._sample_completion(
            prefix_embeds, attn_mask,
            max_new_tokens=grpo_cfg.max_new_tokens,
            temperature=0.8,
            top_p=0.9,
        )
        text = host._tok.decode(tokens, skip_special_tokens=True).strip()
        print(f"== COMPLETION {i}  tokens.shape={tokens.shape} ==")
        print(f"   text: {text[:300]}")
        # Manually call the self-judge to see raw response, with context.
        from trophic.training.grpo import _scenario_context
        ctx = _scenario_context(sc)
        user = (
            f"Upstream context: {ctx}\n\n"
            f"Predator output: {text}\n\n"
            "Return your JSON judgment now."
        )
        raw = host.generate_chat(system=SJ_SYSTEM, user=user, max_new_tokens=512, temperature=0.0)
        print(f"   ctx: {ctx}")
        print(f"   raw judge response: {raw[:300]}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
