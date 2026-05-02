"""Decode signals at every tier of the trophic stack for one scenario.

Observability tool: turns hidden-state vectors flowing between tiers into
human-readable token approximations via **logit-lens** projection. Free at
runtime (one matmul per tier), zero training cost.

Usage:
    .venv/bin/python -u scripts/diagnostics/inspect_signals.py \\
        --ckpt checkpoints/sft_seed28_stocknet_best.pt \\
        --scenario stocknet_test_AAPL_2015-10-01 \\
        --top-k 8

Each tier reports:
  - hidden-state norm (calibration: what magnitude the signal carries)
  - logit-lens top-k (what tokens the model would emit if it stopped here)
  - tier-specific metadata (alive slots, attention weights, M magnitudes)
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import os
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch

from trophic.agents.herbivore import Herbivore
from trophic.agents.predator import Predator
from trophic.agents.producer import Producer
from trophic.agents.quant_producer import QuantitativeProducer
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.training.checkpoint import load_channels
from trophic.training.sft import SFTConfig, SFTRunner
from trophic.training.stocknet_loader import build_stocknet_scenarios, SMOKE_TICKERS_TOP5
from trophic.training.scenarios import build_scenarios, split as split_toy
from trophic.training.xml_schema import parse_prediction


def logit_lens(host: ModelHost, hidden: torch.Tensor, top_k: int = 8) -> list[tuple[str, float]]:
    """Project hidden vector through lm_head and return top-k (token_str, prob).

    `hidden`: [H] or [N, H]. If multi-row, mean-pools first.
    """
    if hidden.dim() > 1:
        hidden = hidden.mean(dim=0)
    h = hidden.detach().to(device=host.device, dtype=host.dtype)
    with torch.no_grad():
        # Get the LM head — usually at host._model.lm_head, but Qwen wraps it.
        lm_head = None
        if hasattr(host._model, "lm_head"):
            lm_head = host._model.lm_head
        elif hasattr(host._model, "get_output_embeddings"):
            lm_head = host._model.get_output_embeddings()
        if lm_head is None:
            return [("(no lm_head found)", 0.0)]
        logits = lm_head(h.unsqueeze(0)).squeeze(0)  # [vocab]
        probs = torch.softmax(logits.float(), dim=-1)
        topv, topi = probs.topk(min(top_k, probs.shape[0]))
    tok = host._tok
    out = []
    for i, p in zip(topi.tolist(), topv.tolist()):
        try:
            t = tok.decode([i]).strip()
        except Exception:
            t = f"<id={i}>"
        if not t:
            t = f"<id={i}>"
        out.append((t, p))
    return out


def fmt_lens(lens: list[tuple[str, float]]) -> str:
    return "  ".join(f"{t!r}({p*100:.0f}%)" for t, p in lens)


def find_scenario(name_filter: str, prefer_split: str = "test"):
    """Find a scenario across StockNet splits + toy scenarios that matches the filter."""
    candidates = []
    # Try StockNet first
    for split in (prefer_split, "dev", "train"):
        try:
            scens = build_stocknet_scenarios(
                split=split, tickers=SMOKE_TICKERS_TOP5, max_per_ticker=None,
            )
            candidates.extend(scens)
        except Exception:
            pass
    try:
        toy = build_scenarios()
        candidates.extend(toy)
    except Exception:
        pass
    matches = [s for s in candidates if name_filter in s.name]
    if not matches:
        # Substring match on first scenario
        return candidates[0] if candidates else None
    return matches[0]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sft_seed28_stocknet_best.pt")
    ap.add_argument("--scenario", default="stocknet_test_AAPL_2015-10-01",
                    help="substring of scenario name; first match wins")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=28)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    os.environ.setdefault("TROPHIC_CONSUMER_INTERFACE", "hooks")
    os.environ.setdefault("TROPHIC_DSTAR_PATH", "checkpoints/dstar_stocknet.pt")
    os.environ.setdefault("TROPHIC_DSTAR_SCALE_OVERRIDE", "1.0")

    cfg = DEFAULT_CONFIG
    host = ModelHost.get(cfg.model)
    sft_cfg = SFTConfig(seed=args.seed)

    herbs = [Herbivore.make(k, capacity=cfg.population.intake_budget)
             for k in ("technical", "fundamental")]
    pred = Predator.make("short_horizon")
    producers = [Producer.make(k) for k in ("tickdelta", "disclosure", "anomaly")]
    producers.append(QuantitativeProducer.make("quote_series"))

    sc = find_scenario(args.scenario)
    if sc is None:
        print(f"[err] no scenario found matching {args.scenario!r}")
        return
    print(f"\n=== scenario: {sc.name} ===")
    if sc.predator_target:
        target = parse_prediction(sc.predator_target)
        print(f"target: ticker={target.ticker} direction={target.direction}")

    runner = SFTRunner(
        cfg=sft_cfg, host=host, producers=producers,
        herbivores=herbs, predator=pred, train=[], eval_=[sc],
    )
    load_channels(args.ckpt, herbivores=herbs, predators=[pred], runner=runner)
    if pred.phi_mlp is not None:
        pred.phi_mlp.to(device=host.device, dtype=host.dtype)
    for h in herbs:
        if h.phi_mlp is not None:
            h.phi_mlp.to(device=host.device, dtype=host.dtype)
    if runner._producer_trough is not None:
        runner._producer_trough.to(device=host.device, dtype=host.dtype)
    if runner._herb_trough is not None:
        runner._herb_trough.to(device=host.device, dtype=host.dtype)
    await runner._cache_producer_broadcasts()

    candidates = runner._producer_cache.get(sc.name, [])

    # ---- TIER 1: producer broadcasts ----
    print(f"\n--- Tier 1: PRODUCERS ({len(candidates)} broadcasts) ---")
    for b in candidates:
        emb = torch.tensor(b.channel_embedding, dtype=host.dtype, device=host.device)
        lens = logit_lens(host, emb, top_k=args.top_k)
        text = (b.decoded_text or "")[:80].replace("\n", " ")
        print(f"  [{b.agent_kind}] norm={emb.norm().item():.1f} diet={b.diet_tags}")
        print(f"    logit-lens: {fmt_lens(lens)}")
        print(f"    raw render: {text!r}")

    # ---- TIER 3: herbivores attend producer-trough → herb broadcasts ----
    # (Tier 2 trough state is checked AFTER Tier 3 runs, since the deposit
    # only happens when herbs forward through _hooks_compile_M_for_herb.)
    print(f"\n--- Tier 3: HERBIVORES (attend producer-trough, emit herb broadcasts) ---")
    herb_for_pred = runner._real_herb_broadcasts_for_predator(sc, candidates)

    # ---- TIER 2: producer-trough state (POST-deposit) ----
    if runner._producer_trough is not None:
        t = runner._producer_trough
        n_alive = int(t.alive.sum().item())
        cum = t.cumulative_attention[:n_alive].tolist() if n_alive > 0 else []
        print(f"\n--- Tier 2: PRODUCER-TROUGH (after Tier 3 deposit) ---")
        print(f"  alive_slots={n_alive}/{t.n_slots}")
        if cum:
            print(f"  cumulative_attention[:n_alive]={[round(x, 3) for x in cum]}")
        if n_alive > 0:
            # Logit-lens the most-attended slot's V_store
            most_att = t.cumulative_attention[:n_alive].argmax().item() if cum else 0
            v = t.V_store[most_att].to(host.dtype).to(host.device)
            v_lens = logit_lens(host, v, top_k=args.top_k)
            print(f"  most-attended-slot V_store (slot {most_att}) logit-lens: {fmt_lens(v_lens)}")

    # ---- Tier 3 herb broadcasts ----
    print(f"\n--- Tier 3 (continued): HERB BROADCASTS ---")
    for b in herb_for_pred:
        emb = torch.tensor(b.channel_embedding, dtype=host.dtype, device=host.device)
        finite = torch.isfinite(emb).all().item()
        norm = emb.norm().item() if finite else float('nan')
        if finite:
            lens = logit_lens(host, emb, top_k=args.top_k)
            print(f"  [{b.agent_kind}] norm={norm:.1f} finite={finite}")
            print(f"    logit-lens: {fmt_lens(lens)}")
        else:
            print(f"  [{b.agent_kind}] norm=NaN finite={finite}  ← signal corrupted")

    # ---- TIER 4: predator's view (compile M, attended pooled) ----
    print(f"\n--- Tier 4: APEX (compiles M from herb-trough, decodes) ---")
    if pred.phi_mlp is not None:
        # Build herb-trough + pooled, like _hooks_eval_decode_predator does
        from trophic.agents.base import ROLE_Q_REF_NORM
        rp_mean = pred.role_prefix.mean(dim=0).to(host.device, host.dtype)
        role_q = rp_mean * (ROLE_Q_REF_NORM / (rp_mean.norm() + 1e-6))
        # Logit-lens the role_q itself
        rq_lens = logit_lens(host, role_q, top_k=args.top_k)
        print(f"  apex role_q logit-lens: {fmt_lens(rq_lens)}")

        if herb_for_pred and runner._herb_trough is not None:
            trough = runner._herb_trough
            alive_ids = trough.alive.nonzero(as_tuple=False).flatten().tolist()
            if alive_ids:
                trough.evict(alive_ids)
            deposit_n = min(len(herb_for_pred), trough.n_slots)
            trough.deposit(herb_for_pred[:deposit_n])
            pooled = runner._trough_pooled_hidden(trough, role_q)
            finite = torch.isfinite(pooled).all().item()
            norm = pooled.norm().item() if finite else float('nan')
            print(f"  herb-trough attended-pooled: norm={norm:.1f} finite={finite}")
            if finite:
                pooled_lens = logit_lens(host, pooled, top_k=args.top_k)
                print(f"    logit-lens: {fmt_lens(pooled_lens)}")

                # M tensors compiled from this pooled
                m_tensors = pred.phi_mlp(pooled)
                # Inspect first patched layer + last
                first_layer = next(iter(m_tensors))
                last_layer = list(m_tensors.keys())[-1]
                d0 = m_tensors[first_layer]
                dN = m_tensors[last_layer]
                print(f"    M layer {first_layer}: ||M_A||={d0['M_A'].norm().item():.1f} "
                      f"s_M={d0['s_M'].item():+.4f}  ||E_A||={d0['E_A'].norm().item():.1f} "
                      f"s_E={d0['s_E'].item():+.4f}")
                print(f"    M layer {last_layer}: ||M_A||={dN['M_A'].norm().item():.1f} "
                      f"s_M={dN['s_M'].item():+.4f}  ||E_A||={dN['E_A'].norm().item():.1f} "
                      f"s_E={dN['s_E'].item():+.4f}")

    if runner._dstar is not None:
        d = runner._dstar
        n_d = len(d.directions)
        first = next(iter(d.directions.values()))
        d_lens = logit_lens(host, first, top_k=args.top_k)
        print(f"  d* (frozen direction, scale={d.scale}, {n_d} layers)")
        print(f"    layer-0 logit-lens: {fmt_lens(d_lens)}")

    # ---- TIER 5: predator decode ----
    print(f"\n--- Tier 5: APEX OUTPUT ---")
    out = runner.eval_decode(sc, max_new_tokens=args.max_new_tokens)
    decoded = out.get("pred.short_horizon", "")
    parsed = parse_prediction(decoded)
    print(f"  decoded: {decoded[:300]!r}")
    print(f"  parsed: ticker={parsed.ticker} direction={parsed.direction} confidence={parsed.confidence}")
    if sc.predator_target:
        tgt = parse_prediction(sc.predator_target)
        match = parsed.direction == tgt.direction
        print(f"  target: direction={tgt.direction}  →  {'CORRECT' if match else 'WRONG'}")


if __name__ == "__main__":
    asyncio.run(main())
