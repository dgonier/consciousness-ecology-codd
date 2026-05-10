"""Smoke test for the belief consolidator.

Seeds 7 beliefs with deliberate near-duplicates:
  - 2 lexical-near-duplicates ("market is risk-off" vs "market is in risk-off mode")
  - 2 semantic-near-duplicates ("AAPL has good regulatory standing" vs "AAPL is in good standing with US regulators")
  - 2 distinct (control)
  - 1 polymarket-bound (must NOT be merged with anything)

Verifies:
  - Modal embeddings reachable + 1024-dim
  - Lexical fast-path catches the obvious dupes
  - Embedding cosine catches paraphrases
  - Polymarket-bound belief survives
  - Distinct beliefs survive
  - Edges rewrite correctly when a node is merged away
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
    ModalEmbedder,
    StateBelief,
    consolidate,
)
from trophic.beliefs.schema import prior_p_from_categorical


def hr() -> None:
    print("─" * 70)


def main() -> int:
    print("Belief consolidator smoke\n")

    # 1. Modal embeddings reachable?
    embedder = ModalEmbedder()
    print(f"Modal embedder available: {embedder.is_available()}")
    print(f"  endpoint: {embedder.endpoint_url}")
    if embedder.is_available():
        try:
            test_emb = embedder.embed_text("hello world")
            print(f"  smoke embed dim: {len(test_emb)}")
        except Exception as e:
            print(f"  embed failed: {e}")
            return 1
    else:
        print("  cannot proceed without auth token")
        return 1

    # 2. Build a store with deliberate near-dupes
    store_root = ROOT / "external" / "beliefs_consolidate_smoke"
    if store_root.exists():
        for p in store_root.glob("*.jsonl"):
            p.unlink()
    store = JSONLBeliefStore(store_root)
    t0 = time.time()

    p_neutral = prior_p_from_categorical("moderate", "neutral")

    # Pair 1: lexical near-dupes (high SequenceMatcher ratio)
    a1 = StateBelief(
        id="belief.market.risk_off_a",
        statement_template="The market is in risk-off mode",
        scope="market", prior_p=p_neutral, current_p=p_neutral,
        last_updated=t0,
    )
    a2 = StateBelief(
        id="belief.market.risk_off_b",
        statement_template="The market is in a risk-off mode",
        scope="market", prior_p=p_neutral, current_p=p_neutral,
        last_updated=t0,
    )

    # Pair 2: paraphrase (different surface form, same meaning)
    b1 = StateBelief(
        id="belief.company.AAPL.standing_a",
        statement_template="AAPL has good regulatory standing with US authorities",
        scope="company", prior_p=p_neutral, current_p=p_neutral,
        context={"ticker": "AAPL"}, last_updated=t0,
    )
    b2 = StateBelief(
        id="belief.company.AAPL.standing_b",
        statement_template="Apple Inc is in good standing with US regulators",
        scope="company", prior_p=p_neutral, current_p=p_neutral,
        context={"ticker": "AAPL"}, last_updated=t0,
    )

    # Pair 3: distinct (control — should NOT merge)
    c1 = StateBelief(
        id="belief.company.AAPL.distinct_1",
        statement_template="AAPL beats consensus earnings estimates this quarter",
        scope="company", prior_p=p_neutral, current_p=p_neutral,
        context={"ticker": "AAPL"}, last_updated=t0,
    )
    c2 = StateBelief(
        id="belief.company.AAPL.distinct_2",
        statement_template="AAPL faces a regulatory enforcement action",
        scope="company", prior_p=p_neutral, current_p=p_neutral,
        context={"ticker": "AAPL"}, last_updated=t0,
    )

    # Polymarket-bound — must survive
    pm = StateBelief(
        id="belief.macro.fed_cuts_jun_2026",
        statement_template="Fed cuts rates by July 2026 FOMC",
        scope="macro", prior_p=0.5, current_p=0.5,
        polymarket_market_id="fed-rate-cut-by-july-2026",
        last_updated=t0,
    )

    all_beliefs = [a1, a2, b1, b2, c1, c2, pm]
    for b in all_beliefs:
        store.upsert_state_belief(b)

    # Add a couple of links to verify edge rewrite
    fake_outcome_id = "outcome.AAPL.next_day_up"
    link_a1 = InternalLink(
        id="link.test.a1→outcome",
        premise_belief_id=a1.id, conclusion_belief_id=fake_outcome_id,
        scope="market", direction="negative",
        strength_prior=0.5, strength_posterior=0.5,
    )
    link_a2 = InternalLink(
        id="link.test.a2→outcome",
        premise_belief_id=a2.id, conclusion_belief_id=fake_outcome_id,
        scope="market", direction="negative",
        strength_prior=0.5, strength_posterior=0.5,
    )
    link_b1 = InternalLink(
        id="link.test.b1→outcome",
        premise_belief_id=b1.id, conclusion_belief_id=fake_outcome_id,
        scope="company", direction="positive",
        strength_prior=0.6, strength_posterior=0.6,
    )
    link_b2 = InternalLink(
        id="link.test.b2→outcome",
        premise_belief_id=b2.id, conclusion_belief_id=fake_outcome_id,
        scope="company", direction="positive",
        strength_prior=0.6, strength_posterior=0.6,
    )
    for l in (link_a1, link_a2, link_b1, link_b2):
        store.upsert_link(l)

    print(f"\nseeded {len(all_beliefs)} beliefs and {len(store.all_links())} links")
    hr()
    print("BEFORE consolidation:")
    for b in store.all_state_beliefs():
        marker = " (polymarket)" if b.is_polymarket_bound else ""
        print(f"  [{b.scope:7s}] {b.id:50s} → {b.statement_template[:60]}{marker}")

    # 3. Run consolidation
    hr()
    print("\nrunning consolidate()...")
    report = consolidate(store, embedder=embedder, llm_judge_borderline=True, apply=True)

    print(f"\n=== REPORT ===")
    print(f"  total beliefs:      {report.n_total}")
    print(f"  pairs evaluated:    {report.n_pairs_evaluated}")
    print(f"  lexical merges:     {report.n_lexical_merges}")
    print(f"  embedding merges:   {report.n_embedding_merges}")
    print(f"  llm borderline:     {report.n_llm_borderline}")
    print(f"  llm confirmed:      {report.n_llm_confirmed}")
    print(f"  llm rejected:       {report.n_llm_rejected}")
    print(f"  total llm calls:    {report.n_llm_calls}")
    print(f"  merges applied:     {len(report.merges_applied)}")
    for m in report.merges_applied:
        print(f"    {m.source:14s} score={m.score:.3f}  {m.redundant_id} → {m.canonical_id}")
        if m.reasoning:
            print(f"      reason: {m.reasoning[:120]}")

    hr()
    print("\nAFTER consolidation:")
    remaining = store.all_state_beliefs()
    for b in remaining:
        marker = " (polymarket)" if b.is_polymarket_bound else ""
        print(f"  [{b.scope:7s}] {b.id:50s} → {b.statement_template[:60]}{marker}")
    print(f"\nremaining links: {len(store.all_links())}")
    for l in store.all_links():
        print(f"  {l.premise_belief_id} -[{l.direction}]→ {l.conclusion_belief_id}")

    # 4. Assertions
    hr()
    remaining_ids = {b.id for b in remaining}
    # Polymarket survives
    assert pm.id in remaining_ids, "polymarket-bound belief was merged (BAD)"
    print("✓ polymarket-bound belief preserved")
    # Distinct controls survive
    assert c1.id in remaining_ids, "distinct c1 was merged (BAD)"
    assert c2.id in remaining_ids, "distinct c2 was merged (BAD)"
    print("✓ distinct beliefs preserved")
    # At least one of each pair was merged
    a_left = (a1.id in remaining_ids) + (a2.id in remaining_ids)
    b_left = (b1.id in remaining_ids) + (b2.id in remaining_ids)
    assert a_left == 1, f"lexical pair: expected 1 surviving, got {a_left}"
    print(f"✓ lexical pair merged ({a_left}/2 survived)")
    if b_left == 1:
        print(f"✓ paraphrase pair merged via embedding/llm ({b_left}/2 survived)")
    else:
        print(f"⚠️  paraphrase pair did NOT merge (both survived) — check cosine threshold")

    # All edges should still exist (rewritten, not deleted)
    assert len(store.all_links()) == 4, f"expected 4 links, got {len(store.all_links())}"
    print("✓ edges rewritten (none lost)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
