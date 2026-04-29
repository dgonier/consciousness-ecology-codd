"""Quick check: does the fixed hunter_state actually differ across scenarios?

Bypasses model loading, just reconstructs the Q computation directly to confirm
the fix introduces input dependence. If Q differs but Channel output doesn't,
the issue is the trained Channel weights (which were trained against constant Q
and learned to ignore Q variations).
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path("/home/dgonier/ecology_experiment/trophic")
sys.path.insert(0, str(ROOT))
import asyncio, builtins, functools
builtins.print = functools.partial(builtins.print, flush=True)
print = builtins.print

import torch
import torch.nn.functional as F
from trophic.agents.herbivore import Herbivore
from trophic.agents.predator import Predator
from trophic.agents.producer import Producer
from trophic.agents.quant_producer import QuantitativeProducer
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.training.checkpoint import load_channels
from trophic.training.scenarios import build_scenarios
from trophic.training.sft import SFTConfig, SFTRunner

CKPT = "checkpoints/ipo_seed7_best.pt"
SEED = 7
TARGETS = ["eval_disclosure_GOOG_settlement", "eval_disclosure_TSLA_recall"]


async def main() -> None:
    cfg = DEFAULT_CONFIG
    host = ModelHost.get(cfg.model)
    sft_cfg = SFTConfig(seed=SEED)
    herbivores = [Herbivore.make(k, capacity=cfg.population.intake_budget)
                  for k in ("technical", "fundamental")]
    predator = Predator.make("short_horizon")
    for h in herbivores: h.ensure_initialized(host, seed_base=SEED)
    predator.ensure_initialized(host, seed_base=SEED)
    for h in herbivores:
        for ch in h.channels.values(): ch.to(device=host.device, dtype=host.dtype)
    for ch in predator.channels.values(): ch.to(device=host.device, dtype=host.dtype)
    load_channels(CKPT, herbivores=herbivores, predators=[predator])

    scenarios = build_scenarios()
    by_name = {s.name: s for s in scenarios}
    targets = [by_name[t] for t in TARGETS]

    producers = [Producer.make(k) for k in ("tickdelta", "disclosure", "anomaly")]
    producers.append(QuantitativeProducer.make("quote_series"))
    runner = SFTRunner(cfg=sft_cfg, host=host, producers=producers,
                       herbivores=herbivores, predator=predator,
                       train=[], eval_=targets)
    await runner._cache_producer_broadcasts()

    # Reconstruct Q directly per scenario for the predator (mirrors the SFT eval path).
    role_q = predator.role_prefix.mean(dim=0).to(device=host.device, dtype=host.dtype)
    print(f"role_q shape: {tuple(role_q.shape)} norm: {role_q.norm().item():.4f}")
    print()

    for s in targets:
        print(f"=== {s.name} ===")
        # Build the herb_for_pred list the same way SFTRunner does at eval time
        candidates = runner._herb_broadcasts_for_predator(s, runner._producer_cache.get(s.name, []))
        print(f"  candidates: {len(candidates)} herb broadcasts")
        if candidates:
            prey_t = torch.tensor(
                [p.channel_embedding for p in candidates],
                dtype=host.dtype, device=host.device,
            )
            input_q = prey_t.mean(dim=0)
            hunter_state_fixed = role_q + input_q
            hunter_state_old = role_q  # what it was before the fix
            print(f"  prey_t shape: {tuple(prey_t.shape)}")
            print(f"  per-candidate norms: {[c.channel_embedding[:3] for c in candidates[:2]]}...")  # peek
            print(f"  input_q norm:    {input_q.norm().item():.4f}")
            print(f"  hunter (OLD/role-only) norm: {hunter_state_old.norm().item():.4f}")
            print(f"  hunter (FIXED) norm:         {hunter_state_fixed.norm().item():.4f}")
            print(f"  cos(role-only, fixed):       {F.cosine_similarity(hunter_state_old.unsqueeze(0), hunter_state_fixed.unsqueeze(0)).item():.4f}")
        print()

    # Compare Q across scenarios (this is the key check)
    print("=== Q COMPARISON ACROSS SCENARIOS ===")
    qs_old = []
    qs_fixed = []
    for s in targets:
        candidates = runner._herb_broadcasts_for_predator(s, runner._producer_cache.get(s.name, []))
        if candidates:
            prey_t = torch.tensor(
                [p.channel_embedding for p in candidates],
                dtype=host.dtype, device=host.device,
            )
            qs_old.append(role_q.clone())
            qs_fixed.append(role_q + prey_t.mean(dim=0))
        else:
            qs_old.append(role_q.clone())
            qs_fixed.append(role_q.clone())

    cos_old = F.cosine_similarity(qs_old[0].unsqueeze(0), qs_old[1].unsqueeze(0)).item()
    cos_fixed = F.cosine_similarity(qs_fixed[0].unsqueeze(0), qs_fixed[1].unsqueeze(0)).item()
    diff_old = (qs_old[0] - qs_old[1]).norm().item()
    diff_fixed = (qs_fixed[0] - qs_fixed[1]).norm().item()
    print(f"  OLD (role-only)  Q cos: {cos_old:.6f}  diff_norm: {diff_old:.4f}")
    print(f"  FIXED (role+prey) Q cos: {cos_fixed:.6f}  diff_norm: {diff_fixed:.4f}")
    print()
    print("If FIXED cos < 1.0 and diff_norm > 0: the fix really does introduce input dependence.")
    print("Then the question is whether the trained Channel weights respond to the differing Q.")


if __name__ == "__main__":
    asyncio.run(main())
