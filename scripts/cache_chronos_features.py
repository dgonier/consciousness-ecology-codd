"""Pre-compute Chronos numeric forecast features for all StockNet scenarios.

Writes external/stocknet_cache/forecast_features.json keyed by
"<TICKER>|<DATE>" → list[float] of N_FORECAST_FEATURES floats.

Run once; the train+eval loop then consumes the cache via
TROPHIC_COMPUTE_FORECAST=1 in build_stocknet_scenarios.

Usage:
    .venv/bin/python -u scripts/cache_chronos_features.py
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TROPHIC_COMPUTE_FORECAST", "1")

from trophic.training.stocknet_loader import build_stocknet_scenarios

def main():
    for split in ("test", "dev", "train"):
        t0 = time.time()
        scens = build_stocknet_scenarios(
            split=split, tickers=None, max_per_ticker=None, compute_forecast=True,
        )
        with_feats = sum(1 for s in scens if s.forecaster_features)
        elapsed = time.time() - t0
        print(f"[{split}] {len(scens)} scenarios, {with_feats} with features, {elapsed:.1f}s")
    print("[done] cache written to external/stocknet_cache/forecast_features.json")

if __name__ == "__main__":
    main()
