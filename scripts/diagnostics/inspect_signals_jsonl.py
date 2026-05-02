"""Capture every inter-species signal as JSONL with parent/child edges.

For one scenario, walks the trophic stack and emits one JSON line per node:
producer broadcasts (Tier 1), producer-trough slots (Tier 2), herb broadcasts
(Tier 3), apex inputs role_q + attended-pooled (Tier 4 PRE-@E modulation),
d*, and apex output (Tier 5). Each node carries:
  - id (stable within a scenario)
  - parents (list of node ids feeding it)
  - tier (1..5), agent_kind (e.g. tickdelta, technical, short_horizon)
  - norm, finite
  - logit_lens (top-k token decodes — what the LM head reads)
  - raw_text (where applicable: producer.decoded_text, herb prefix_text, apex output)

Output: logs/signals/<scenario>.jsonl
The viz reads all jsonl files in that directory.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import json
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
from trophic.training.scenarios import build_scenarios
from trophic.training.xml_schema import parse_prediction


def logit_lens(host: ModelHost, hidden: torch.Tensor, top_k: int = 8):
    if hidden.dim() > 1:
        hidden = hidden.mean(dim=0)
    h = hidden.detach().to(device=host.device, dtype=host.dtype)
    with torch.no_grad():
        lm_head = getattr(host._model, "lm_head", None) or host._model.get_output_embeddings()
        logits = lm_head(h.unsqueeze(0)).squeeze(0)
        probs = torch.softmax(logits.float(), dim=-1)
        topv, topi = probs.topk(min(top_k, probs.shape[0]))
    out = []
    for i, p in zip(topi.tolist(), topv.tolist()):
        try:
            t = host._tok.decode([i]).strip()
        except Exception:
            t = f"<id={i}>"
        if not t:
            t = f"<id={i}>"
        out.append({"token": t, "prob": p})
    return out


def find_scenario(name_filter: str):
    candidates = []
    for split in ("test", "dev", "train"):
        try:
            candidates.extend(build_stocknet_scenarios(
                split=split, tickers=SMOKE_TICKERS_TOP5, max_per_ticker=None,
            ))
        except Exception:
            pass
    try:
        candidates.extend(build_scenarios())
    except Exception:
        pass
    matches = [s for s in candidates if name_filter in s.name]
    return matches[0] if matches else (candidates[0] if candidates else None)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sft_seed32_stable_best.pt")
    ap.add_argument("--scenario", default="stocknet_test_AAPL_2015-10-01")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=32)
    ap.add_argument("--out-dir", default="logs/signals")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--bare-herb", action="store_true",
                    help="Skip M-hook installation when running herb forward — emits"
                    " bare-Qwen pooled hidden as the herb broadcast. Use to isolate"
                    " whether the M hooks themselves cause the herb collapse.")
    args = ap.parse_args()

    os.environ.setdefault("TROPHIC_CONSUMER_INTERFACE", "hooks")
    os.environ.setdefault("TROPHIC_DSTAR_PATH", "checkpoints/dstar_stocknet.pt")
    os.environ.setdefault("TROPHIC_DSTAR_SCALE_OVERRIDE", "1.0")
    if args.bare_herb:
        os.environ["TROPHIC_BARE_HERB"] = "1"

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
        print(f"[err] no scenario matching {args.scenario!r}")
        return
    target_dir = parse_prediction(sc.predator_target or "").direction if sc.predator_target else None

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sc.name}.jsonl"
    f = out_path.open("w")

    nodes: list[dict] = []

    def emit(node: dict):
        nodes.append(node)
        f.write(json.dumps(node) + "\n")

    emit({
        "id": "scenario",
        "parents": [],
        "tier": 0,
        "agent_kind": "scenario",
        "label": sc.name,
        "target_direction": target_dir,
    })

    # ---- Tier 1: producers ----
    for b in candidates:
        emb = torch.tensor(b.channel_embedding, dtype=host.dtype, device=host.device)
        finite = bool(torch.isfinite(emb).all().item())
        emit({
            "id": f"producer.{b.agent_kind}",
            "parents": ["scenario"],
            "tier": 1,
            "agent_kind": b.agent_kind,
            "label": f"producer.{b.agent_kind}",
            "norm": float(emb.norm().item()) if finite else None,
            "finite": finite,
            "diet_tags": list(b.diet_tags),
            "logit_lens": logit_lens(host, emb, top_k=args.top_k) if finite else [],
            "raw_text": (b.decoded_text or "")[:240],
        })

    # ---- Tier 3 (run first so trough deposit happens) ----
    herb_for_pred = runner._real_herb_broadcasts_for_predator(sc, candidates)

    # ---- Tier 2: producer-trough state (post-deposit) ----
    if runner._producer_trough is not None:
        t = runner._producer_trough
        n_alive = int(t.alive.sum().item())
        cum = t.cumulative_attention[:n_alive].tolist() if n_alive > 0 else []
        for slot_i in range(n_alive):
            v = t.V_store[slot_i].to(host.dtype).to(host.device)
            finite = bool(torch.isfinite(v).all().item())
            # Slot's parent producer = best guess from order of deposit;
            # producer order ~= candidates list. Map slot_i to candidate kind.
            parent = (
                f"producer.{candidates[slot_i].agent_kind}"
                if slot_i < len(candidates) else "scenario"
            )
            emit({
                "id": f"trough.producer.slot{slot_i}",
                "parents": [parent],
                "tier": 2,
                "agent_kind": "producer_trough_slot",
                "label": f"trough.slot{slot_i}",
                "norm": float(v.norm().item()) if finite else None,
                "finite": finite,
                "cumulative_attention": cum[slot_i] if slot_i < len(cum) else None,
                "logit_lens": logit_lens(host, v, top_k=args.top_k) if finite else [],
            })

    # ---- Tier 3: herbivore broadcasts ----
    # Each herb attends ALL trough slots, so parents = all trough.producer.slot*
    trough_node_ids = [
        f"trough.producer.slot{i}" for i in range(int(runner._producer_trough.alive.sum().item()))
    ] if runner._producer_trough is not None else []
    for b in herb_for_pred:
        emb = torch.tensor(b.channel_embedding, dtype=host.dtype, device=host.device)
        finite = bool(torch.isfinite(emb).all().item())
        emit({
            "id": f"herb.{b.agent_kind}",
            "parents": trough_node_ids if trough_node_ids else ["scenario"],
            "tier": 3,
            "agent_kind": b.agent_kind,
            "label": f"herb.{b.agent_kind}",
            "norm": float(emb.norm().item()) if finite else None,
            "finite": finite,
            "diet_tags": list(b.diet_tags),
            "logit_lens": logit_lens(host, emb, top_k=args.top_k) if finite else [],
        })

    # ---- Tier 4 PRE-@E: apex inputs (role_q, herb-trough attended-pooled, M tensors)
    # role_q
    if pred.phi_mlp is not None:
        from trophic.agents.base import ROLE_Q_REF_NORM
        rp_mean = pred.role_prefix.mean(dim=0).to(host.device, host.dtype)
        role_q = rp_mean * (ROLE_Q_REF_NORM / (rp_mean.norm() + 1e-6))
        emit({
            "id": "apex.role_q",
            "parents": ["scenario"],
            "tier": 4,
            "agent_kind": "apex_role_q",
            "label": "apex.role_q (frozen)",
            "norm": float(role_q.norm().item()),
            "finite": True,
            "logit_lens": logit_lens(host, role_q, top_k=args.top_k),
        })

        # herb-trough attended-pooled
        if herb_for_pred and runner._herb_trough is not None:
            trough = runner._herb_trough
            alive_ids = trough.alive.nonzero(as_tuple=False).flatten().tolist()
            if alive_ids:
                trough.evict(alive_ids)
            deposit_n = min(len(herb_for_pred), trough.n_slots)
            trough.deposit(herb_for_pred[:deposit_n])
            pooled = runner._trough_pooled_hidden(trough, role_q)
            finite = bool(torch.isfinite(pooled).all().item())

            herb_parents = [f"herb.{b.agent_kind}" for b in herb_for_pred[:deposit_n]]
            emit({
                "id": "apex.attended_pooled",
                "parents": herb_parents + ["apex.role_q"],
                "tier": 4,
                "agent_kind": "apex_attended_pooled",
                "label": "apex.attended_pooled (PRE-@E input)",
                "norm": float(pooled.norm().item()) if finite else None,
                "finite": finite,
                "logit_lens": logit_lens(host, pooled, top_k=args.top_k) if finite else [],
            })

            if finite:
                m_tensors = pred.phi_mlp(pooled)
                # Summarize first + last patched layer; record both magnitudes.
                layer_keys = list(m_tensors.keys())
                for lk in (layer_keys[0], layer_keys[-1]):
                    d = m_tensors[lk]
                    emit({
                        "id": f"apex.M_layer{lk}",
                        "parents": ["apex.attended_pooled"],
                        "tier": 4,
                        "agent_kind": "apex_M_tensor",
                        "label": f"apex.M_layer{lk} (φ_mlp output, PRE-@E)",
                        "norm_M_A": float(d["M_A"].norm().item()),
                        "norm_E_A": float(d["E_A"].norm().item()),
                        "s_M": float(d["s_M"].item()),
                        "s_E": float(d["s_E"].item()),
                        "finite": True,
                    })

    # d*
    if runner._dstar is not None:
        d = runner._dstar
        first = next(iter(d.directions.values()))
        emit({
            "id": "apex.dstar",
            "parents": ["scenario"],
            "tier": 4,
            "agent_kind": "apex_dstar",
            "label": f"apex.dstar (frozen, scale={d.scale}, layers={len(d.directions)})",
            "norm": float(first.norm().item()),
            "finite": True,
            "logit_lens": logit_lens(host, first, top_k=args.top_k),
        })

    # ---- Tier 5: predator decode (post-@E, full hooks active) ----
    out = runner.eval_decode(sc, max_new_tokens=args.max_new_tokens)
    decoded = out.get("pred.short_horizon", "")
    parsed = parse_prediction(decoded)
    apex_parents = ["apex.attended_pooled"]
    if pred.phi_mlp is not None:
        apex_parents.append("apex.role_q")
    if runner._dstar is not None:
        apex_parents.append("apex.dstar")
    emit({
        "id": "apex.output",
        "parents": apex_parents,
        "tier": 5,
        "agent_kind": "apex_output",
        "label": "apex.output (POST-@E decode)",
        "raw_text": decoded[:600],
        "parsed_ticker": parsed.ticker,
        "parsed_direction": parsed.direction,
        "parsed_confidence": parsed.confidence,
        "target_direction": target_dir,
        "match": parsed.direction == target_dir if target_dir else None,
        "finite": True,
    })

    f.close()
    print(f"[done] {len(nodes)} nodes → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
