"""One-call smoke for the LLM-driven recursive decomposer.

CALL BUDGET: this script makes EXACTLY ONE Bedrock Sonnet call by setting
max_depth=1 and one root belief. Confirms the prompt + JSON parsing +
schema integration before unleashing the recursive version on all 25 seeds.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

ROOT = Path("/home/dgonier/ecology_experiment/trophic")
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

from trophic.beliefs import (
    InternalLink,
    JSONLBeliefStore,
    StateBelief,
)
from trophic.beliefs.decomposer import BeliefDecomposer
from trophic.beliefs.schema import prior_p_from_categorical


def main() -> int:
    # Use isolated dev store so the smoke doesn't mutate the real Neo4j
    store_root = ROOT / "external" / "beliefs_decomposer_smoke"
    if store_root.exists():
        for p in store_root.glob("*.jsonl"):
            p.unlink()
    store = JSONLBeliefStore(store_root)

    # Seed ONE composite belief — the kind that v0 lists as a single direct
    # outcome parent but actually has multiple assumptions underneath.
    p = prior_p_from_categorical("strong", "leans_true")
    composite = StateBelief(
        id="belief.company.AAPL.earnings_beat_will_lift_stock",
        statement_template="An AAPL earnings beat tomorrow will translate to an up-day for the stock",
        scope="company",
        prior_p=p, current_p=p, decay_class="slow",
        context={"ticker": "AAPL"},
        last_updated=time.time(),
    )
    store.upsert_state_belief(composite)
    print(f"seeded: {composite.id}  (prior_p={p})")
    print(f"  statement: {composite.statement_template}")

    # Hook a fake outcome link to it so decompose_link has something to walk
    fake_link = InternalLink(
        id="link.test.composite→outcome",
        premise_belief_id=composite.id,
        conclusion_belief_id="outcome.AAPL.next_day_up",
        scope="company",
        direction="positive",
        strength_prior=0.65, strength_posterior=0.65,
    )
    store.upsert_link(fake_link)

    # ── Single-call decomposition (max_depth=1 → exactly 1 LLM call) ──
    print()
    print(f"calling Bedrock Sonnet (max_depth=1, max_calls=1)...")
    decomposer = BeliefDecomposer(store=store, max_depth=1, max_calls=1)
    stats = decomposer.decompose_all_seed_premises([fake_link])

    print()
    print(f"=== STATS ===")
    print(f"  LLM calls:           {stats.n_llm_calls}")
    print(f"  terminal leaves:     {stats.n_terminal_leaves}")
    print(f"  intermediate beliefs created: {stats.n_intermediate_beliefs}")
    print(f"  links created:       {stats.n_links_created}")
    print(f"  max depth reached:   {stats.max_depth_reached}")
    print(f"  budget exhausted:    {stats.budget_exhausted}")

    # Show what landed in the store
    print()
    print(f"=== beliefs in store ({len(store.all_state_beliefs())}) ===")
    for b in store.all_state_beliefs():
        print(f"  [{b.scope:7s}] {b.id}")
        print(f"            {b.statement_template[:90]}")

    print()
    print(f"=== links in store ({len(store.all_links())}) ===")
    for l in store.all_links():
        print(f"  {l.premise_belief_id}")
        print(f"    -[{l.direction:8s} {l.strength_posterior:.2f}]→ {l.conclusion_belief_id}")

    if stats.n_llm_calls == 0:
        print()
        print("WARN: no LLM calls made — likely Bedrock unavailable")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
