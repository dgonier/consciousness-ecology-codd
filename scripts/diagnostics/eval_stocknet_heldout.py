"""StockNet held-out evals to test for two leakage modes:

1. **Held-out dev**: the StockNet dev split (143 scenarios) MINUS the
   20-scenario in-loop dev sample seed28 used for `_best.pt` selection.
   Catches train→dev model-selection overfitting (we picked the best ckpt
   by loss on those 20; the other 123 are unseen-during-selection).

2. **Held-out tickers**: full StockNet test split for tickers NOT in
   {AAPL, GOOG, MSFT, AMZN, JPM} (the 5 we trained on). Default panel:
   INTC (tech), JNJ (pharma), XOM (energy), WMT (retail), DIS (media).
   Catches cross-ticker leakage (the model may have learned per-ticker
   patterns from training dates that leak to test dates of the same
   tickers).

Reports ACC, MCC, confusion, per-ticker breakdown for each panel.

Env vars:
  TROPHIC_CKPT — checkpoint path
  TROPHIC_SEED — seed used during training (for the dev-sample dedup)
  TROPHIC_LABEL — label for the result table
  TROPHIC_DSTAR_PATH — d* path
  TROPHIC_CONSUMER_INTERFACE=hooks (or prefix)
  HELDOUT_PANEL — comma-sep list (default INTC,JNJ,XOM,WMT,DIS)
  TROPHIC_EVAL_MAX_TOKENS — default 96
"""
from __future__ import annotations

import asyncio
import functools
import os
import random
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch

from scripts.eval_stocknet import matthews_corrcoef
from trophic.agents.herbivore import Herbivore
from trophic.agents.predator import Predator
from trophic.agents.producer import Producer
from trophic.agents.quant_producer import QuantitativeProducer
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.training.checkpoint import load_channels
from trophic.training.sft import SFTConfig, SFTRunner
from trophic.training.stocknet_loader import build_stocknet_scenarios, SMOKE_TICKERS_TOP5
from trophic.training.xml_schema import parse_prediction


def _evaluate(decoded_predictions: list[tuple[str, str]]) -> dict:
    """decoded_predictions: list of (target_dir, predicted_dir)."""
    tp = tn = fp = fn = 0
    abst = 0
    parse_failed = 0
    per_ticker: dict[str, dict[str, int]] = {}
    for sc_name, target_dir, pred_dir, ticker in decoded_predictions:
        per_ticker.setdefault(ticker, {"correct": 0, "total": 0})
        per_ticker[ticker]["total"] += 1
        if pred_dir is None or pred_dir == "abstain":
            abst += 1
            continue
        if pred_dir not in ("up", "down"):
            parse_failed += 1
            continue
        if pred_dir == target_dir:
            per_ticker[ticker]["correct"] += 1
        if target_dir == "up" and pred_dir == "up": tp += 1
        elif target_dir == "down" and pred_dir == "down": tn += 1
        elif target_dir == "down" and pred_dir == "up": fp += 1
        elif target_dir == "up" and pred_dir == "down": fn += 1
    n = len(decoded_predictions)
    decided = tp + tn + fp + fn
    acc = (tp + tn) / max(decided, 1)
    mcc = matthews_corrcoef(tp, tn, fp, fn)
    return {
        "n": n, "decided": decided, "abstained": abst, "parse_failed": parse_failed,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "acc": acc, "mcc": mcc,
        "per_ticker": per_ticker,
    }


def _print_panel(label: str, scens: list, results: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"  scenarios:      {results['n']}")
    print(f"  decisions:      {results['decided']} ({100*results['decided']/max(results['n'],1):.1f}%)")
    print(f"  abstained:      {results['abstained']}")
    print(f"  parse_failed:   {results['parse_failed']}")
    print(f"  TP={results['tp']} TN={results['tn']} FP={results['fp']} FN={results['fn']}")
    print(f"  ACCURACY:       {results['acc']:.3f}")
    print(f"  MCC:            {results['mcc']:+.3f}")
    print(f"  per-ticker:")
    for t, d in results["per_ticker"].items():
        if d["total"]:
            print(f"    {t}: {d['correct']}/{d['total']} = {d['correct']/max(d['total'],1):.3f}")


async def main() -> None:
    cfg = DEFAULT_CONFIG
    seed = int(os.environ.get("TROPHIC_SEED", "28"))
    ckpt_path = os.environ.get("TROPHIC_CKPT", "checkpoints/sft_seed28_stocknet_best.pt")
    label = os.environ.get("TROPHIC_LABEL", f"seed{seed}")
    max_new_tokens = int(os.environ.get("TROPHIC_EVAL_MAX_TOKENS", "96"))
    heldout_panel = os.environ.get(
        "HELDOUT_PANEL", "INTC,JNJ,XOM,WMT,DIS"
    ).split(",")
    dev_eval_cap = int(os.environ.get("TROPHIC_DEV_EVAL_CAP", "20"))

    print(f"[heldout] label={label} ckpt={ckpt_path} seed={seed}")
    print(f"[heldout] held-out tickers: {heldout_panel}")
    print(f"[heldout] in-loop dev cap was: {dev_eval_cap}")

    host = ModelHost.get(cfg.model)
    _lora_dir = os.environ.get("TROPHIC_LORA_DIR", "")
    if _lora_dir:
        from peft import PeftModel
        print(f"[heldout] attaching LoRA from {_lora_dir}")
        host._model = PeftModel.from_pretrained(host._model, _lora_dir, is_trainable=False)
        host._model.eval()

    sft_cfg = SFTConfig(seed=seed)
    herbs = [Herbivore.make(k, capacity=cfg.population.intake_budget)
             for k in ("technical", "fundamental")]
    pred = Predator.make("short_horizon")
    for h in herbs: h.ensure_initialized(host, seed_base=seed)
    pred.ensure_initialized(host, seed_base=seed)
    for h in herbs:
        for ch in h.channels.values():
            ch.to(device=host.device, dtype=host.dtype)
    for ch in pred.channels.values():
        ch.to(device=host.device, dtype=host.dtype)

    # ---- panel 1: held-out dev (StockNet dev MINUS the in-loop sample) ----
    print(f"\n[panel-1] loading StockNet dev split…")
    full_dev = build_stocknet_scenarios(split="dev", tickers=None, max_per_ticker=None)
    rng_dev = random.Random(seed + 1)  # MUST match train_sft_stocknet.py
    in_loop_sample = set(s.name for s in rng_dev.sample(full_dev, min(dev_eval_cap, len(full_dev))))
    heldout_dev = [s for s in full_dev if s.name not in in_loop_sample]
    print(f"[panel-1] heldout-dev: {len(heldout_dev)} scenarios "
          f"(full dev = {len(full_dev)}, in-loop = {len(in_loop_sample)})")

    # ---- panel 2: held-out tickers from the test split ----
    print(f"\n[panel-2] loading test split for held-out tickers {heldout_panel}…")
    heldout_test = build_stocknet_scenarios(
        split="test", tickers=heldout_panel, max_per_ticker=None,
    )
    print(f"[panel-2] heldout-tickers test: {len(heldout_test)} scenarios")

    # Build runner with the union of both panels for cache.
    all_scens = heldout_dev + heldout_test
    producers = [Producer.make(k) for k in ("tickdelta", "disclosure", "anomaly")]
    producers.append(QuantitativeProducer.make("quote_series"))
    runner = SFTRunner(
        cfg=sft_cfg, host=host, producers=producers,
        herbivores=herbs, predator=pred,
        train=[], eval_=all_scens,
    )

    # Load checkpoint AFTER runner so phi_mlp + troughs land properly.
    meta = load_channels(ckpt_path, herbivores=herbs, predators=[pred], runner=runner)
    print(f"[heldout] loaded: {meta}")
    if pred.phi_mlp is not None: pred.phi_mlp.to(device=host.device, dtype=host.dtype)
    for h in herbs:
        if h.phi_mlp is not None: h.phi_mlp.to(device=host.device, dtype=host.dtype)
    if runner._producer_trough is not None: runner._producer_trough.to(device=host.device, dtype=host.dtype)
    if runner._herb_trough is not None: runner._herb_trough.to(device=host.device, dtype=host.dtype)

    print(f"[heldout] caching producer broadcasts for {len(all_scens)} scenarios…")
    await runner._cache_producer_broadcasts()

    # ---- run both panels ----
    def run_panel(scens, panel_name):
        results_in = []
        for sc in scens:
            target = parse_prediction(sc.predator_target or "")
            out = runner.eval_decode(sc, max_new_tokens=max_new_tokens)
            decoded = out.get("pred.short_horizon", "")
            parsed = parse_prediction(decoded)
            ticker = sc.name.split("_")[2]
            results_in.append((sc.name, target.direction, parsed.direction, ticker))
        return _evaluate(results_in)

    print(f"\n[panel-1] running heldout-dev ({len(heldout_dev)} scenarios)…")
    r1 = run_panel(heldout_dev, "heldout-dev")
    print(f"\n[panel-2] running heldout-tickers test ({len(heldout_test)} scenarios)…")
    r2 = run_panel(heldout_test, "heldout-tickers")

    _print_panel(f"PANEL 1: HELD-OUT DEV (123 scenarios, never selected on)", heldout_dev, r1)
    _print_panel(f"PANEL 2: HELD-OUT TICKERS TEST (cross-ticker generalization)", heldout_test, r2)

    print(f"\n=== SUMMARY ===")
    print(f"  in-distribution test (top-5 tickers, eval_stocknet.py): see logs/eval_stocknet_*.log")
    print(f"  held-out dev (panel 1):       ACC {r1['acc']:.3f}  MCC {r1['mcc']:+.3f}  decided {r1['decided']}/{r1['n']}")
    print(f"  held-out tickers test (panel 2): ACC {r2['acc']:.3f}  MCC {r2['mcc']:+.3f}  decided {r2['decided']}/{r2['n']}")
    print(f"  prompt-only Qwen3-4B baseline (top-5 test): ACC 0.460  MCC +0.292  (the bar)")


if __name__ == "__main__":
    asyncio.run(main())
