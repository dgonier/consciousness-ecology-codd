"""Run the temporal decomposer over all unprocessed Pass artifacts.

  .venv/bin/python -u scripts/run_decomposer.py [--limit N] [--no-neo4j]
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

from trophic.beliefs.decomposer_temporal import Decomposer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-neo4j", action="store_true",
                    help="don't write link strength updates back to Neo4j")
    args = ap.parse_args()

    d = Decomposer(root=ROOT, write_neo4j=not args.no_neo4j)
    print(f"passes_dir: {d.passes_dir}")
    print(f"state.processed_dates: {len(d.state['processed_dates'])}")

    results = d.run_batch(limit=args.limit)
    print(f"\nprocessed {len(results)} passes:")
    for r in results:
        if r.get("skipped"):
            continue
        print(f"  {r['date']}: hints={r['teacher_hints']}  apex_pair={r['apex_pair']}  "
              f"link_updates={r['neo4j_links_actually_updated']}/{r['link_updates_inferred']}  "
              f"species={r['species_observed']}  beliefs_d*={r['beliefs_with_d_star']}")

    print()
    print(d.report())


if __name__ == "__main__":
    main()
