"""Find cycles in the InternalLink graph and break each by removing the
weakest-strength edge in the cycle.

Idempotent: run repeatedly until "no cycles found".
"""
from __future__ import annotations

import argparse
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

from trophic.beliefs.neo4j_store import Neo4jBeliefStore
from trophic.beliefs.propagation import find_cycles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report cycles but don't delete edges")
    ap.add_argument("--max-passes", type=int, default=20)
    args = ap.parse_args()

    store = Neo4jBeliefStore()

    total_removed = 0
    for pass_idx in range(args.max_passes):
        links = store.all_links()
        cycles = find_cycles(links, max_cycles=200)
        if not cycles:
            print(f"pass {pass_idx}: no cycles. Done.")
            break

        # Pick weakest edge in each cycle
        link_by_pair: dict[tuple[str, str], list] = {}
        for l in links:
            link_by_pair.setdefault(
                (l.premise_belief_id, l.conclusion_belief_id), []
            ).append(l)

        to_delete: set[str] = set()
        for cyc in cycles:
            edges_in_cycle = []
            for i in range(len(cyc) - 1):
                pair = (cyc[i], cyc[i + 1])
                for l in link_by_pair.get(pair, []):
                    edges_in_cycle.append(l)
            if not edges_in_cycle:
                continue
            weakest = min(edges_in_cycle, key=lambda l: l.strength_posterior)
            to_delete.add(weakest.id)

        print(f"pass {pass_idx}: {len(cycles)} cycles found, "
              f"{len(to_delete)} edges queued for deletion")

        if args.dry_run:
            for lid in list(to_delete)[:10]:
                print(f"  would delete: {lid}")
            print("  (dry-run, stopping after first pass)")
            break

        for lid in to_delete:
            store.delete_link(lid)
        total_removed += len(to_delete)
        print(f"  pass {pass_idx}: removed {len(to_delete)} edges "
              f"(running total: {total_removed})")

    print(f"\nTOTAL edges removed: {total_removed}")
    final_links = len(store.all_links())
    print(f"final link count: {final_links}")


if __name__ == "__main__":
    main()
