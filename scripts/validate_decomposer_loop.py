"""Validate that decomposer updates close the learning loop.

Sequence:
  1. Snapshot current strength_posterior of a sample of links
  2. Decomposer processes day 2026-02-03 (already in passes/), writes
     Bayesian updates back to Neo4j
  3. Re-snapshot the same links — verify they changed
  4. Print delta table

This is the smallest possible verification that the loop closes. If the
deltas are nonzero, day-N+1's eval will read the updated values when it
loads links from Neo4j at startup.
"""
from __future__ import annotations

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

from trophic.beliefs.decomposer_temporal import Decomposer
from trophic.beliefs.neo4j_store import Neo4jBeliefStore


def snapshot_link_strengths(store: Neo4jBeliefStore) -> dict[str, dict]:
    """Capture (link_id → {strength_posterior, n_validations, n_correct})
    for all links targeting outcomes."""
    out = {}
    links = store.all_links()
    for l in links:
        if l.conclusion_belief_id.startswith("outcome.") and \
           ".next_day_direction" in l.conclusion_belief_id:
            out[l.id] = {
                "strength_posterior": l.strength_posterior,
                "n_validations": l.n_validations,
                "n_correct": l.n_correct,
                "premise": l.premise_belief_id,
                "conclusion": l.conclusion_belief_id,
            }
    return out


def main() -> None:
    print("Decomposer loop-closure validation")
    print("─" * 70)

    store = Neo4jBeliefStore()
    before = snapshot_link_strengths(store)
    print(f"baseline: {len(before)} ticker-outcome-targeting links")
    sample_ids = sorted(before.keys())[:5]
    print("\nbaseline strengths (first 5 links):")
    for lid in sample_ids:
        e = before[lid]
        print(f"  {lid:80s}  s={e['strength_posterior']:.3f}  "
              f"n_val={e['n_validations']}  n_correct={e['n_correct']}")

    print("\nrunning decomposer on unprocessed passes...")
    d = Decomposer(root=ROOT, write_neo4j=True)
    # Reset state to reprocess (idempotent for this validation)
    d.state = {"processed_dates": [], "n_resolved_passes": 0}
    d.d_star = {}
    d.fitness = {}
    results = d.run_batch()
    print(f"  processed {len(results)} passes")
    for r in results:
        print(f"    {r['date']}: links_updated={r.get('neo4j_links_actually_updated')}")

    after = snapshot_link_strengths(store)

    print("\n─" * 70)
    print("CHANGED links:")
    n_changed = 0
    for lid, before_e in before.items():
        after_e = after.get(lid)
        if after_e is None:
            continue
        ds = after_e["strength_posterior"] - before_e["strength_posterior"]
        dn = after_e["n_validations"] - before_e["n_validations"]
        if abs(ds) > 1e-6 or dn != 0:
            n_changed += 1
            if n_changed <= 15:
                print(f"  {lid[:75]:75s}  Δs={ds:+.4f}  Δn_val={dn:+d}  "
                      f"correct={before_e['n_correct']}→{after_e['n_correct']}")
    print(f"\ntotal changed: {n_changed} / {len(before)}")
    if n_changed > 0:
        print("\nLOOP CLOSED ✓ — link strengths updated; next eval pass will pick them up.")
    else:
        print("\nLOOP NOT CLOSED ✗ — no link strengths changed.")


if __name__ == "__main__":
    main()
