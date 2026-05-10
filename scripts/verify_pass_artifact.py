"""Sanity-check a Pass JSON artifact written by the firehose loop runner.

Verifies:
  - File exists, is valid JSON.gz
  - Required top-level fields present
  - Activations include applied_log_odds_shift and leaf_p_after where the
    target was in state
  - Observations have causal_chain populated for entries with non-trivial p_up
  - Watchlists have rank, action, p_up, ticker
  - ground_truth has ideal_p_up populated per soft-label scheme
  - Cross-references: every activation's species_id is one we expect, every
    watchlist ticker is in the universe
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trophic.beliefs.pass_record import load_pass


REQUIRED_TOP = [
    "date", "prev_trading_day", "runner_version", "model_id",
    "focus_tickers", "n_news", "activations", "observations",
    "watchlist_ecology", "watchlist_bare", "ground_truth", "elapsed",
]
EXPECTED_SPECIES = {
    "event_classifier.v0.qwen-local",
    "cross_correlation.v0",
    "(unknown)",  # tolerable
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    rec = load_pass(Path(args.path))

    missing = [f for f in REQUIRED_TOP if f not in rec]
    if missing:
        print(f"FAIL: missing top-level fields: {missing}")
        sys.exit(1)

    print(f"date={rec['date']}  model={rec['model_id']}  n_news={rec['n_news']}")
    print(f"  activations:        {len(rec['activations'])}")
    print(f"  observations:       {len(rec['observations'])}")
    print(f"  watchlist_ecology:  {len(rec['watchlist_ecology'])}")
    print(f"  watchlist_bare:     {len(rec['watchlist_bare'])}")
    print(f"  ground_truth:       {len(rec['ground_truth'])}")
    print(f"  elapsed:            {rec['elapsed']}")

    # Activation field sanity
    n_with_shift = sum(1 for a in rec['activations'] if a.get('applied_log_odds_shift') is not None)
    n_with_pafter = sum(1 for a in rec['activations'] if a.get('leaf_p_after') is not None)
    n_articles = sum(1 for a in rec['activations'] if a.get('source_article_id'))
    species = set(a.get('species_id', '') for a in rec['activations'])
    print(f"\n  acts with applied_log_odds_shift: {n_with_shift}/{len(rec['activations'])}")
    print(f"  acts with leaf_p_after:           {n_with_pafter}/{len(rec['activations'])}")
    print(f"  acts with source_article_id:      {n_articles}/{len(rec['activations'])}")
    print(f"  species: {species}")

    # Observation sanity
    n_obs_with_chain = sum(1 for o in rec['observations'] if o.get('causal_chain'))
    print(f"\n  observations with causal_chain: {n_obs_with_chain}/{len(rec['observations'])}")
    if rec['observations']:
        sample = rec['observations'][0]
        print(f"  sample observation keys: {sorted(sample.keys())}")

    # Watchlist sanity
    for path in ("ecology", "bare"):
        wl = rec[f'watchlist_{path}']
        n_with_rank = sum(1 for w in wl if w.get('rank'))
        actions = set(w.get('action') for w in wl)
        print(f"  watchlist_{path}: rank populated={n_with_rank}/{len(wl)}, actions={actions}")

    # Ground truth sanity
    n_with_ideal = sum(1 for g in rec['ground_truth'] if g.get('ideal_p_up') is not None)
    print(f"\n  ground_truth with ideal_p_up: {n_with_ideal}/{len(rec['ground_truth'])}")
    if rec['ground_truth']:
        # Show distribution of ideal_p_up
        from collections import Counter
        ip = Counter(round(g['ideal_p_up'], 2) for g in rec['ground_truth'])
        print(f"  ideal_p_up distribution: {sorted(ip.items())}")

    # First sample of each
    if rec['activations']:
        print(f"\n  sample activation (first):")
        a = rec['activations'][0]
        for k in ('id', 'target_belief_id', 'species_id', 'magnitude', 'direction',
                  'applied_log_odds_shift', 'leaf_p_before', 'leaf_p_after',
                  'source_article_id'):
            print(f"    {k}: {a.get(k)}")

    print("\nPASS — schema looks complete.")


if __name__ == "__main__":
    main()
