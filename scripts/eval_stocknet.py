"""Eval a checkpoint on the StockNet (ACL-18) test split.

Metrics: directional accuracy + Matthews Correlation Coefficient (MCC).
These are the two numbers reported by the StockNet leaderboard.

Env vars:
  TROPHIC_CKPT  — path to checkpoint .pt (default: checkpoints/ipo_seed7_best.pt)
  TROPHIC_SEED  — seed for Channel init (default: 7)
  TROPHIC_LABEL — label for the report (default: "ipo_seed7")
  STOCKNET_TICKERS  — comma-separated ticker list (default: top-5)
  STOCKNET_DATE_FROM / STOCKNET_DATE_TO — narrow the date window for smoke runs
                       (defaults: full test split 2015-10-01 → 2016-01-01)
  STOCKNET_MAX_PER_TICKER — cap per ticker (default: 10 for smoke, None=full)
"""
from __future__ import annotations

import asyncio
import builtins
import functools
import os
import sys
from pathlib import Path

builtins.print = functools.partial(builtins.print, flush=True)
print = builtins.print

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from trophic.agents.herbivore import Herbivore
from trophic.agents.predator import Predator
from trophic.agents.producer import Producer
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.training.checkpoint import load_channels
from trophic.training.sft import SFTConfig, SFTRunner
from trophic.training.stocknet_loader import (
    SMOKE_TICKERS_TOP5,
    build_stocknet_scenarios,
)
from trophic.training.xml_schema import parse_prediction


def matthews_corrcoef(tp: int, tn: int, fp: int, fn: int) -> float:
    """MCC for binary classification. Returns 0 on degenerate denominator."""
    num = tp * tn - fp * fn
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if den == 0:
        return 0.0
    return num / den


async def main() -> None:
    cfg = DEFAULT_CONFIG
    seed = int(os.environ.get("TROPHIC_SEED", "7"))
    ckpt_path = os.environ.get("TROPHIC_CKPT", "checkpoints/ipo_seed7_best.pt")
    label = os.environ.get("TROPHIC_LABEL", "ipo_seed7")

    tickers_env = os.environ.get("STOCKNET_TICKERS", "")
    tickers = [t.strip() for t in tickers_env.split(",") if t.strip()] or SMOKE_TICKERS_TOP5

    max_per = os.environ.get("STOCKNET_MAX_PER_TICKER")
    max_per_ticker = int(max_per) if max_per else None

    print(f"[stocknet-eval] label={label} ckpt={ckpt_path} seed={seed} tickers={tickers}"
          f" max_per_ticker={max_per_ticker}")

    host = ModelHost.get(cfg.model)
    print(f"[stocknet-eval] hidden={host.hidden_size} dtype={host.dtype} dev={host.device}")
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
    print(f"[stocknet-eval] loaded: {meta}")
    for ch in predator.channels.values():
        ch.to(device=host.device, dtype=host.dtype)
    for h in herbivores:
        for ch in h.channels.values():
            ch.to(device=host.device, dtype=host.dtype)

    # Load StockNet scenarios.
    print(f"[stocknet-eval] loading scenarios...")
    scenarios = build_stocknet_scenarios(
        split="test",
        tickers=tickers,
        history_days=5,
        max_per_ticker=max_per_ticker,
    )
    print(f"[stocknet-eval] {len(scenarios)} scenarios")

    # Producer cache via SFTRunner.
    from trophic.agents.quant_producer import QuantitativeProducer
    producers = [Producer.make(k) for k in ("tickdelta", "disclosure", "anomaly")]
    producers.append(QuantitativeProducer.make("quote_series"))
    runner = SFTRunner(
        cfg=sft_cfg, host=host, producers=producers,
        herbivores=herbivores, predator=predator,
        train=[], eval_=scenarios,
    )
    await runner._cache_producer_broadcasts()

    # Eval loop: directional accuracy + MCC.
    tp = tn = fp = fn = 0
    abstained = 0
    parse_failed = 0
    per_ticker: dict[str, dict[str, int]] = {t: {"correct": 0, "total": 0} for t in tickers}

    print(f"\n=== STOCKNET EVAL [{label}] ===")
    for sc in scenarios:
        out = runner.eval_decode(sc)
        pred_text = out.get("pred.short_horizon", "")
        target = parse_prediction(sc.predator_target or "")
        parsed = parse_prediction(pred_text)

        target_dir = target.direction
        pred_dir = parsed.direction

        # Extract ticker from scenario name for per-ticker stats.
        # Format: stocknet_test_TICKER_DATE
        sc_ticker = sc.name.split("_")[2]
        per_ticker[sc_ticker]["total"] += 1

        if pred_dir is None or pred_dir == "abstain":
            abstained += 1
            continue
        if pred_dir not in ("up", "down"):
            parse_failed += 1
            continue

        correct = pred_dir == target_dir
        if correct:
            per_ticker[sc_ticker]["correct"] += 1
        if target_dir == "up" and pred_dir == "up":
            tp += 1
        elif target_dir == "down" and pred_dir == "down":
            tn += 1
        elif target_dir == "down" and pred_dir == "up":
            fp += 1
        elif target_dir == "up" and pred_dir == "down":
            fn += 1

    total_decisions = tp + tn + fp + fn
    n_total = len(scenarios)
    accuracy = (tp + tn) / total_decisions if total_decisions > 0 else 0.0
    mcc = matthews_corrcoef(tp, tn, fp, fn)

    print(f"\n=== RESULTS [{label}] ===")
    print(f"  scenarios:       {n_total}")
    print(f"  decisions:       {total_decisions} ({100*total_decisions/max(1,n_total):.1f}%)")
    print(f"  abstained:       {abstained}")
    print(f"  parse_failed:    {parse_failed}")
    print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"  ACCURACY:        {accuracy:.3f}")
    print(f"  MCC:             {mcc:.3f}")
    print(f"\n  per-ticker:")
    for t, stats in per_ticker.items():
        if stats["total"] > 0:
            tac = stats["correct"] / stats["total"]
            print(f"    {t}: {stats['correct']}/{stats['total']} = {tac:.3f}")
    print(f"\n  baselines (reference):")
    print(f"    Adv-ALSTM (Feng+2018):   ~0.57 acc")
    print(f"    MAN-SF (Sawhney+2020):   ~0.58 acc")
    print(f"    StockEmbed/BERT-tuned:   ~0.74 acc")
    print(f"    Random (50/50):           0.50 acc, 0.00 MCC")


if __name__ == "__main__":
    asyncio.run(main())
