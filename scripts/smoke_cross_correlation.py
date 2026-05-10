"""Smoke-test the cross-correlation herbivore against the firehose dataset.

Runs the herbivore as-of the latest date in the corpus, prints:
  - the 20d correlation matrix (sanity check: AAPL/MSFT/GOOG should cluster)
  - cluster intra-correlations (tech > defensives etc.)
  - emitted cluster/regime BeliefActivations
  - sympathy broadcast for a fake AAPL earnings_beat activation

No LLM. Pure numpy.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

env = ROOT / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
            continue
        os.environ.setdefault(k, v)

import numpy as np

from trophic.beliefs.herbivores.cross_correlation import (
    CLUSTERS,
    CrossCorrelationHerbivore,
    update_state_from_bars,
    _cluster_intra_corr,
)
from trophic.beliefs.schema import BeliefActivation


CACHE = ROOT / "external" / "polygon_cache" / "polygon"
DATASET = ROOT / "data" / "firehose_dataset.jsonl"
STATE_PATH = ROOT / "data" / "cross_correlation_state.npz"


def load_dataset_bars() -> dict[str, dict[str, dict]]:
    """Reconstruct bars_by_date_ticker from the firehose dataset.

    The dataset stores t-1 bars on row t (firehose_bars), so iterate and
    assemble.
    """
    bars: dict[str, dict[str, dict]] = {}
    with DATASET.open() as fh:
        for line in fh:
            row = json.loads(line)
            for fb in row["firehose_bars"]:
                d = fb["date"]
                bars.setdefault(d, {})[fb["ticker"]] = {
                    "close": fb["close"],
                }
            # Also add the actual close for the row's date from targets
            # (so the latest date has an entry too)
            for tg in row["targets"]:
                d = row["date"]
                bars.setdefault(d, {})[tg["ticker"]] = {
                    "close": tg["actual_close"],
                }
    return bars


def hr() -> None:
    print("─" * 78)


def main() -> None:
    print("Cross-correlation herbivore smoke test")
    hr()

    bars = load_dataset_bars()
    dates = sorted(bars.keys())
    print(f"loaded bars: {len(dates)} dates from {dates[0]} to {dates[-1]}")

    # Universe: all tickers seen in bars
    universe: set[str] = set()
    for d_bars in bars.values():
        universe.update(d_bars.keys())
    universe = sorted(universe)
    print(f"universe: {universe}")

    herb = CrossCorrelationHerbivore.with_state(universe, STATE_PATH)
    as_of = dates[-1]
    print(f"as_of: {as_of}")

    update_state_from_bars(herb.state, bars, as_of, herb.cfg)
    print(f"n_observations used: {herb.state.n_observations}")

    hr()
    print("=== 20d correlation matrix (rolling_20d) ===")
    corr20 = herb.state.rolling_20d
    # Print as a square table
    header = "      " + "  ".join(f"{tk:>5s}" for tk in universe)
    print(header)
    for i, tk_i in enumerate(universe):
        row_str = f"{tk_i:>5s} "
        for j in range(len(universe)):
            v = corr20[i, j]
            if np.isnan(v):
                row_str += "   .   "
            else:
                row_str += f" {v:+.2f}"
        print(row_str)

    hr()
    print("=== cluster intra-correlations (20d) ===")
    for cluster_name, tickers in CLUSTERS.items():
        idxs = [herb.state.idx(t) for t in tickers if herb.state.idx(t) is not None]
        if len(idxs) < 2:
            continue
        c = _cluster_intra_corr(corr20, idxs)
        members = [t for t in tickers if herb.state.idx(t) is not None]
        print(f"  {cluster_name:18s} mean intra-corr = {c:+.3f}  members={members}")

    hr()
    print("=== emitted cluster / regime activations ===")
    cluster_acts = herb.emit_cluster_activations()
    if not cluster_acts:
        print("  (none — no cluster crossed threshold)")
    for a in cluster_acts:
        print(f"  → {a.target_belief_id}  {a.direction_of_effect:9s} {a.magnitude:7s} "
              f"(conf={a.self_rated_confidence})")
        print(f"    reasoning: {a.reasoning}")

    hr()
    print("=== sympathy broadcast (fake AAPL earnings_beat strong) ===")
    fake_src = BeliefActivation(
        target_belief_id="belief.company.earnings_beat__ticker_AAPL",
        direction_of_effect="increases",
        magnitude="strong",
        decay_class="slow",
        self_rated_confidence="very_high",
        reasoning="(synthetic source for smoke test)",
        species_id="event_classifier.test",
    )
    sympathy = herb.broadcast_sympathy([fake_src])
    if not sympathy:
        print("  (no correlated tickers above threshold)")
    for a in sympathy:
        print(f"  → {a.target_belief_id:55s} {a.direction_of_effect:9s} {a.magnitude:7s} "
              f"(conf={a.self_rated_confidence})")
        print(f"    reasoning: {a.reasoning}")

    # Persist state
    herb.state.save(STATE_PATH)
    print()
    print(f"saved correlation state → {STATE_PATH}")


if __name__ == "__main__":
    main()
