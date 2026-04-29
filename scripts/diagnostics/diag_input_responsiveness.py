"""Diagnostic for issue #8: Where is the predator's input-responsiveness lost?

After #10 phases A (FinCoT LoRA) and B (SocialSignal LoRA) both failed to
break direction collapse on StockNet (MCC=0.000 across no-LoRA, FinCoT, and
SocialSignal co-training), the bottleneck must be downstream of base-model
pretraining. This diagnostic asks: across multiple StockNet days for the
SAME ticker, where does input variance get squashed?

For 5 different days of AAPL StockNet data, capture:
  1. prey_t (herbivore-output candidate broadcasts the predator sees)
  2. hunter_state (Q construction = role_q + prey_t.mean())
  3. Predator Channel output (Q×K×V cross-attention result)
  4. Final decoded prediction text

Then for each tensor, compute pairwise cosine similarity across the 5 days.
The signal:
  - prey_t cosine ≈ 1.0  → herbivore output collapses upstream; species CPT
    can't fix this no matter what (which matches the path A+B negative result).
  - prey_t differs but hunter_state cosine ≈ 1.0 → input_q.mean() is too
    averaged-out across the candidate set; need a different Q construction.
  - hunter_state differs but Channel output cosine ≈ 1.0 → cross-attention
    weights are saturated or W_V/proj_out collapse; the K/V or projection is
    the actual bottleneck.
  - Channel output differs but decoded text is constant → predator's Qwen3
    forward ignores the channel context (role_prefix dominates).

Run against ipo_seed16_best.pt (latest co-trained) with the SocialSignal LoRA
attached, since that's the most recent architecturally-clean configuration.

Env vars:
  TROPHIC_CKPT — checkpoint to diagnose (default ipo_seed16_best.pt)
  TROPHIC_LORA_DIR — LoRA to attach (default social_signal_lora/seed16)
  TROPHIC_SEED — seed (default 16)
  STOCKNET_TICKER — which ticker to compare days of (default AAPL)
  STOCKNET_DAYS — how many days to compare (default 5)
"""
from __future__ import annotations
import asyncio, builtins, functools, os, sys
from pathlib import Path
builtins.print = functools.partial(builtins.print, flush=True)
print = builtins.print

ROOT = Path("/home/dgonier/ecology_experiment/trophic")
sys.path.insert(0, str(ROOT))

import itertools
import torch
import torch.nn.functional as F

from trophic.agents.herbivore import Herbivore
from trophic.agents.predator import Predator
from trophic.agents.producer import Producer
from trophic.agents.quant_producer import QuantitativeProducer
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.training.checkpoint import load_channels
from trophic.training.stocknet_loader import build_stocknet_scenarios
from trophic.training.sft import SFTConfig, SFTRunner


def _pairwise_cosines(tensors: list[torch.Tensor], names: list[str]) -> dict:
    """Pairwise cosine similarities + summary stats."""
    if not tensors or any(t is None for t in tensors):
        return {"error": "missing tensors"}
    pairs = []
    for (i, ti), (j, tj) in itertools.combinations(enumerate(tensors), 2):
        c = F.cosine_similarity(
            ti.flatten().unsqueeze(0).float(),
            tj.flatten().unsqueeze(0).float(),
        ).item()
        pairs.append({"i": names[i], "j": names[j], "cos": c})
    cosines = [p["cos"] for p in pairs]
    return {
        "pairs": pairs,
        "mean": sum(cosines) / len(cosines),
        "min": min(cosines),
        "max": max(cosines),
        "n_pairs": len(cosines),
    }


async def main() -> None:
    cfg = DEFAULT_CONFIG
    ckpt_path = os.environ.get("TROPHIC_CKPT", "checkpoints/ipo_seed16_best.pt")
    lora_dir = os.environ.get("TROPHIC_LORA_DIR", "checkpoints/social_signal_lora/seed16")
    seed = int(os.environ.get("TROPHIC_SEED", "16"))
    ticker = os.environ.get("STOCKNET_TICKER", "AAPL")
    n_days = int(os.environ.get("STOCKNET_DAYS", "5"))

    print(f"[diag-ir] ckpt={ckpt_path}")
    print(f"[diag-ir] lora_dir={lora_dir}")
    print(f"[diag-ir] seed={seed}  ticker={ticker}  n_days={n_days}")

    host = ModelHost.get(cfg.model)
    if lora_dir and Path(lora_dir).exists():
        from peft import PeftModel
        print(f"[diag-ir] attaching LoRA from {lora_dir}")
        host._model = PeftModel.from_pretrained(host._model, lora_dir, is_trainable=False)
        host._model.eval()
    else:
        print(f"[diag-ir] no LoRA active (path: {lora_dir})")

    sft_cfg = SFTConfig(seed=seed)
    herbivores = [Herbivore.make(k, capacity=cfg.population.intake_budget)
                  for k in ("technical", "fundamental")]
    predator = Predator.make("short_horizon")
    for h in herbivores:
        h.ensure_initialized(host, seed_base=seed)
    predator.ensure_initialized(host, seed_base=seed)
    for h in herbivores:
        for ch in h.channels.values():
            ch.to(device=host.device, dtype=host.dtype)
    for ch in predator.channels.values():
        ch.to(device=host.device, dtype=host.dtype)
    meta = load_channels(ckpt_path, herbivores=herbivores, predators=[predator])
    print(f"[diag-ir] loaded channels: {meta}")

    scenarios = build_stocknet_scenarios(
        split="test", tickers=[ticker], max_per_ticker=n_days,
    )
    print(f"[diag-ir] scenarios: {len(scenarios)}")

    producers = [Producer.make(k) for k in ("tickdelta", "disclosure", "anomaly")]
    producers.append(QuantitativeProducer.make("quote_series"))
    runner = SFTRunner(cfg=sft_cfg, host=host, producers=producers,
                       herbivores=herbivores, predator=predator,
                       train=[], eval_=scenarios)
    await runner._cache_producer_broadcasts()

    # Capture buffers, indexed by scenario name.
    captured: dict[str, dict] = {sc.name: {} for sc in scenarios}
    current = {"name": None}

    # Hook the predator Channel(s) to grab cross-attention output.
    def make_predator_hook(channel_name: str):
        def hook(module, inputs, output):
            sc_name = current["name"]
            if sc_name is None:
                return
            entry = captured[sc_name].setdefault(f"pred_{channel_name}", {})
            try:
                entry["output"] = output.output.detach().cpu().float()
                entry["null_prob"] = float(output.null_prob)
            except Exception as e:
                entry["error"] = str(e)
        return hook

    # Hook the predator Channel.forward to also grab hunter_state (the Q input).
    # We do this by wrapping the Channel.forward to also capture its first arg.
    def make_q_hook(channel_name: str):
        orig = predator.channels[channel_name].forward
        def wrapped(hunter_state, prey_t, *args, **kwargs):
            sc_name = current["name"]
            if sc_name is not None:
                entry = captured[sc_name].setdefault(f"pred_{channel_name}", {})
                entry["hunter_state"] = hunter_state.detach().cpu().float()
                entry["prey_t"] = prey_t.detach().cpu().float()
            return orig(hunter_state, prey_t, *args, **kwargs)
        return wrapped

    handles = []
    for kind, ch in predator.channels.items():
        h = ch.register_forward_hook(make_predator_hook(kind))
        handles.append(h)
        # Patch forward to capture inputs
        ch.forward = make_q_hook(kind)

    # Hook each herbivore's Channel to capture herbivore output (= the broadcasts
    # that become the predator's prey_t candidates).
    def make_herb_hook(herb_kind: str, channel_name: str):
        def hook(module, inputs, output):
            sc_name = current["name"]
            if sc_name is None:
                return
            entry = captured[sc_name].setdefault(f"herb_{herb_kind}_{channel_name}", {})
            try:
                entry["output"] = output.output.detach().cpu().float()
                entry["null_prob"] = float(output.null_prob)
            except Exception:
                pass
        return hook

    for h in herbivores:
        for kind, ch in h.channels.items():
            handle = ch.register_forward_hook(make_herb_hook(h.kind, kind))
            handles.append(handle)

    print("\n=== running scenarios with hooks ===")
    for sc in scenarios:
        current["name"] = sc.name
        out = runner.eval_decode(sc)
        decoded = out.get("pred.short_horizon", "")
        captured[sc.name]["decoded"] = decoded[:200]
        print(f"\n[{sc.name}]")
        print(f"  decoded[:200]: {decoded[:200]}")

    print("\n\n=== INPUT-RESPONSIVENESS DIAGNOSTIC ===\n")
    names = [sc.name for sc in scenarios]

    # 1. prey_t (predator's input candidates pooled) — the *input* to predator's Q construction
    print("LAYER 1: prey_t (predator candidate broadcasts)")
    for kind in predator.channels.keys():
        prey_ts = []
        ok_names = []
        for n in names:
            entry = captured[n].get(f"pred_{kind}", {})
            if "prey_t" in entry:
                pt = entry["prey_t"]
                if pt.numel() > 0:
                    prey_ts.append(pt.mean(dim=0))  # pool to single vector for comparison
                    ok_names.append(n)
        if len(prey_ts) >= 2:
            r = _pairwise_cosines(prey_ts, ok_names)
            print(f"  pred_{kind}: prey_t.mean cosine — mean={r['mean']:.4f} min={r['min']:.4f} max={r['max']:.4f} (n_pairs={r['n_pairs']})")
        else:
            print(f"  pred_{kind}: not enough prey_t captures ({len(prey_ts)})")

    print()
    print("LAYER 2: hunter_state (Q = role_q + prey_t.mean)")
    for kind in predator.channels.keys():
        hs_list = []
        ok_names = []
        for n in names:
            entry = captured[n].get(f"pred_{kind}", {})
            if "hunter_state" in entry:
                hs_list.append(entry["hunter_state"])
                ok_names.append(n)
        if len(hs_list) >= 2:
            r = _pairwise_cosines(hs_list, ok_names)
            print(f"  pred_{kind}: hunter_state cosine — mean={r['mean']:.4f} min={r['min']:.4f} max={r['max']:.4f} (n_pairs={r['n_pairs']})")
        else:
            print(f"  pred_{kind}: not enough hunter_state captures ({len(hs_list)})")

    print()
    print("LAYER 3: predator Channel output (cross-attention result)")
    for kind in predator.channels.keys():
        outs = []
        ok_names = []
        nps = []
        for n in names:
            entry = captured[n].get(f"pred_{kind}", {})
            if "output" in entry:
                outs.append(entry["output"])
                ok_names.append(n)
                nps.append(entry.get("null_prob"))
        if len(outs) >= 2:
            r = _pairwise_cosines(outs, ok_names)
            print(f"  pred_{kind}: channel_output cosine — mean={r['mean']:.4f} min={r['min']:.4f} max={r['max']:.4f}")
            print(f"    null_probs: {[f'{x:.3f}' if x is not None else 'None' for x in nps]}")
        else:
            print(f"  pred_{kind}: not enough output captures ({len(outs)})")

    print()
    print("LAYER 0 (sanity): herbivore Channel outputs (= upstream of predator)")
    for h in herbivores:
        for kind in h.channels.keys():
            outs = []
            ok_names = []
            for n in names:
                entry = captured[n].get(f"herb_{h.kind}_{kind}", {})
                if "output" in entry:
                    outs.append(entry["output"])
                    ok_names.append(n)
            if len(outs) >= 2:
                r = _pairwise_cosines(outs, ok_names)
                print(f"  herb_{h.kind}.{kind}: output cosine — mean={r['mean']:.4f} min={r['min']:.4f} max={r['max']:.4f}")

    print()
    print("LAYER 4: decoded text uniqueness")
    decoded_set = {captured[n]["decoded"] for n in names}
    print(f"  unique decodes: {len(decoded_set)} / {len(names)}")
    for n in names:
        print(f"    [{n}] {captured[n]['decoded'][:100]}")

    print("\n=== INTERPRETATION ===")
    print("Cosine values to interpret each layer:")
    print("  > 0.99  → essentially identical (collapse here)")
    print("  0.90-0.99 → similar but with detectable variance")
    print("  < 0.90  → meaningfully input-responsive")
    print()
    print("Localization:")
    print("  herbivore output ≈ 1.0 + prey_t ≈ 1.0  → upstream herbivore collapse (preceding species)")
    print("  prey_t ≠ 1.0 + hunter_state ≈ 1.0      → averaging in role_q+prey_t.mean() drowns variance")
    print("  hunter_state ≠ 1.0 + ch_output ≈ 1.0   → cross-attn W_V/proj_out collapse")
    print("  ch_output ≠ 1.0 + decoded constant     → Qwen3 forward ignores channel context")

    for h in handles:
        h.remove()


if __name__ == "__main__":
    asyncio.run(main())
