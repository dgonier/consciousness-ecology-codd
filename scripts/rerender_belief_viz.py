"""Re-render the belief network viz from current Neo4j state.

Skips decompose/consolidate — just snapshots whatever is in store and writes
fresh HTML using the updated viz template.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load .env
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
from trophic.beliefs.viz import render_pass_viz, snapshot_pass


def main():
    store = Neo4jBeliefStore()
    print(f"connected; {len(store.all_state_beliefs())} beliefs, "
          f"{len(store.all_links())} links")

    snap = snapshot_pass(
        store=store,
        pass_id="seed_network",
        pass_meta={
            "ticker": "(static seed)",
            "scenario": "post-decompose-consolidate",
            "label": "(rerender)",
            "prediction": None,
        },
        incoming_events=None,
        outcome_predictions=None,
    )
    out_dir = ROOT / "viz_beliefs"
    json_path, html_path = render_pass_viz(snap, out_dir, pass_id="seed_network")
    print(f"wrote: {html_path}")
    print(f"latest: {out_dir / 'latest.html'}")


if __name__ == "__main__":
    main()
