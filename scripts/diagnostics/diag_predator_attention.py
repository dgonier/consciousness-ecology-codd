"""Diagnostic for issue #6: dump predator cross-attention weights, channel
output, and last hidden state for two contrasting eval scenarios. Determine
where the mode collapse occurs along the predator's pipeline.

Compares:
  - eval_disclosure_GOOG_settlement (target: GOOG up, 120min)
  - eval_disclosure_TSLA_recall     (target: TSLA down, 120min)

For each scenario, captures:
  1. Predator Channel's per-K attention weights (one set per herbivore kind)
  2. Channel output sequence (8 vectors, 2560-dim)
  3. Predator's pooled last_hidden_state before decode (via the predator's
     forward path; out-of-band signal that the channel hook picks up).

Verdict logic:
  - If attention cosine(s1, s2) ≈ 1.0: cross-attention is NOT input-conditioned.
    Mode collapse is at the attention layer; Q is likely dominated by role_prefix.
  - If attention differs but channel output cosine ≈ 1.0: collapse is in W_V or proj_out.
  - If both differ but the predator decodes a constant ticker: collapse is downstream
    in the predator's Qwen3 forward (role_prefix dominates channel_seq).
"""
from __future__ import annotations
import asyncio, builtins, functools, os, sys
from pathlib import Path
builtins.print = functools.partial(builtins.print, flush=True)
print = builtins.print

ROOT = Path("/home/dgonier/ecology_experiment/trophic")
sys.path.insert(0, str(ROOT))

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
    print(f"[diag] ckpt={CKPT}  seed={SEED}")
    host = ModelHost.get(cfg.model)
    sft_cfg = SFTConfig(seed=SEED)

    herbivores = [Herbivore.make(k, capacity=cfg.population.intake_budget)
                  for k in ("technical", "fundamental")]
    predator = Predator.make("short_horizon")
    for h in herbivores:
        h.ensure_initialized(host, seed_base=SEED)
    predator.ensure_initialized(host, seed_base=SEED)
    for h in herbivores:
        for ch in h.channels.values():
            ch.to(device=host.device, dtype=host.dtype)
    for ch in predator.channels.values():
        ch.to(device=host.device, dtype=host.dtype)
    meta = load_channels(CKPT, herbivores=herbivores, predators=[predator])
    print(f"[diag] loaded: {meta}")

    scenarios = build_scenarios()
    by_name = {s.name: s for s in scenarios}
    targets = [by_name[t] for t in TARGETS]
    for s in targets:
        print(f"[diag] scenario found: {s.name}")

    producers = [Producer.make(k) for k in ("tickdelta", "disclosure", "anomaly")]
    producers.append(QuantitativeProducer.make("quote_series"))
    runner = SFTRunner(cfg=sft_cfg, host=host, producers=producers,
                       herbivores=herbivores, predator=predator,
                       train=[], eval_=targets)
    await runner._cache_producer_broadcasts()

    captured: dict[str, dict] = {s.name: {} for s in targets}
    current_scenario = {"name": None}

    def make_hook(channel_name: str):
        def hook(module, inputs, output):
            sc_name = current_scenario["name"]
            if sc_name is None:
                return
            entry = captured[sc_name].setdefault(f"channel_{channel_name}", {})
            try:
                entry["output"] = output.output.detach().cpu().float()
                entry["attention"] = (
                    output.attention.detach().cpu().float()
                    if hasattr(output, "attention") and output.attention is not None
                    else None
                )
                entry["null_prob"] = float(output.null_prob)
            except Exception as e:
                entry["error"] = str(e)
        return hook

    hook_handles = []
    for kind, ch in predator.channels.items():
        h = ch.register_forward_hook(make_hook(kind))
        hook_handles.append(h)

    print("\n=== running both scenarios with hooks ===")
    for s in targets:
        current_scenario["name"] = s.name
        out = runner.eval_decode(s)
        decoded = out.get("pred.short_horizon", "")
        captured[s.name]["decoded"] = decoded[:300]
        print(f"\n[{s.name}]")
        print(f"  decoded[:300]: {decoded[:300]}")

    print("\n\n=== DIAGNOSTIC ANALYSIS ===\n")
    s1, s2 = TARGETS[0], TARGETS[1]
    for ch_name in [f"channel_{k}" for k in predator.channels.keys()]:
        c1 = captured[s1].get(ch_name, {})
        c2 = captured[s2].get(ch_name, {})
        if "output" not in c1 or "output" not in c2:
            print(f"  {ch_name}: missing data, skipping")
            continue

        out1, out2 = c1["output"], c2["output"]
        out_cos = F.cosine_similarity(
            out1.flatten().unsqueeze(0), out2.flatten().unsqueeze(0)
        ).item()
        out_diff_norm = (out1 - out2).norm().item()
        out_norm_1 = out1.norm().item()
        out_norm_2 = out2.norm().item()

        att_cos = None
        if c1.get("attention") is not None and c2.get("attention") is not None:
            att1, att2 = c1["attention"], c2["attention"]
            att_cos = F.cosine_similarity(
                att1.flatten().unsqueeze(0), att2.flatten().unsqueeze(0)
            ).item()
            print(f"  {ch_name}:")
            print(f"    attention shape:   {tuple(att1.shape)}")
            print(f"    attention[s1]:     {att1.squeeze().tolist()[:8]}")
            print(f"    attention[s2]:     {att2.squeeze().tolist()[:8]}")
            print(f"    attention cosine:  {att_cos:.4f}")
        else:
            print(f"  {ch_name}: attention not captured (Channel may not return it)")

        np1 = c1.get('null_prob')
        np2 = c2.get('null_prob')
        np1s = f"{np1:.3f}" if isinstance(np1, float) else str(np1)
        np2s = f"{np2:.3f}" if isinstance(np2, float) else str(np2)
        print(f"    null_prob[s1]:     {np1s}")
        print(f"    null_prob[s2]:     {np2s}")
        print(f"    output cosine:     {out_cos:.4f}")
        print(f"    output diff_norm:  {out_diff_norm:.4f}")
        print(f"    output norm[s1]:   {out_norm_1:.4f}")
        print(f"    output norm[s2]:   {out_norm_2:.4f}")
        print()

    print("=== INTERPRETATION ===")
    print()
    print("If attention cosine ≈ 1.0 across scenarios:")
    print("  → attention is NOT input-conditioned. Collapse is at the cross-attention layer.")
    print("  → Q is likely dominated by role_prefix.mean() which is identical per agent.")
    print()
    print("If attention differs but channel output cosine ≈ 1.0:")
    print("  → attention reads input but outputs collapse. Look at W_V or proj_out.")
    print()
    print("If both differ but predator decodes a constant ticker:")
    print("  → attention path works; collapse is in the predator's Qwen3 forward")
    print("    (role_prefix dominates channel_seq, channel context is ignored).")

    for h in hook_handles:
        h.remove()


if __name__ == "__main__":
    asyncio.run(main())
