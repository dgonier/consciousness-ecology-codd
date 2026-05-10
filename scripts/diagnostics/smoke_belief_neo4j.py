"""Smoke test for Neo4j-backed belief store.

Same scenario as smoke_belief_network.py but persists to the shared
DB under the :Ecology namespace. Tears its own data down on cleanup.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

ROOT = Path("/home/dgonier/ecology_experiment/trophic")
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

from trophic.beliefs import (
    BeliefActivation,
    Event,
    InternalLink,
    OutcomeBelief,
    StateBelief,
    apply_activation,
    apply_decay,
    propagate_outcome,
)
from trophic.beliefs.schema import prior_p_from_categorical
from trophic.beliefs.neo4j_store import Neo4jBeliefStore


def hr() -> None:
    print("─" * 70)


def cleanup_smoke_data(store: Neo4jBeliefStore) -> None:
    """Delete all :Ecology nodes that contain 'smoke' in their id, plus
    everything created by this run. Bounded cleanup so we don't nuke
    other Ecology data."""
    with store.driver.session() as sess:
        sess.run("""
            MATCH (n:Ecology)
            WHERE n.id CONTAINS 'smoke'
            DETACH DELETE n
        """)


def main() -> int:
    print("Bayesian belief network — Neo4j smoke test\n")

    store = Neo4jBeliefStore()
    print(f"connected to {os.environ['NEO4J_URI']}")
    print(f"namespace label: {store.NAMESPACE_LABEL}\n")

    # Clean any prior smoke data
    cleanup_smoke_data(store)

    t0 = time.time()

    # Three state beliefs with different priors
    p_a = prior_p_from_categorical("well_established", "leans_true")
    a = StateBelief(
        id="belief.smoke.A",
        statement_template="S&P closes up on a given day",
        scope="market",
        prior_p=p_a, current_p=p_a, decay_class="slow",
        last_updated=t0,
    )
    p_b = prior_p_from_categorical("well_established", "leans_false")
    b = StateBelief(
        id="belief.smoke.B",
        statement_template="Company faces SEC enforcement this quarter",
        scope="company",
        prior_p=p_b, current_p=p_b, decay_class="slow",
        context={"ticker": "AAPL"},
        last_updated=t0,
    )

    store.upsert_state_belief(a)
    store.upsert_state_belief(b)
    print(f"upserted: belief.smoke.A (prior_p={p_a}), belief.smoke.B (prior_p={p_b})")

    # Apply activation, persist
    hr()
    act = BeliefActivation(
        target_belief_id="belief.smoke.B",
        direction_of_effect="increases",
        magnitude="decisive",
        decay_class="slow",
        self_rated_confidence="high",
        reasoning="SEC probe announcement",
        species_id="test.smoke.species",
    )
    b = apply_activation(b, act, event_id="event.smoke.1", now_ts=t0 + 60)
    store.upsert_state_belief(b)
    print(f"after activation: belief.smoke.B.current_p = {b.current_p:.4f}")

    # Persist the event
    event = Event(
        id="event.smoke.1",
        timestamp=t0 + 60,
        source="smoke_test",
        raw_content="SEC opens probe into AAPL accounting practices",
        ticker="AAPL",
        sector="tech",
        activations=[act],
    )
    store.append_event(event)

    # Build a link
    link = InternalLink(
        id="link.smoke.B→outcome",
        premise_belief_id="belief.smoke.B",
        conclusion_belief_id="outcome.smoke.AAPL",
        scope="company",
        direction="negative",
        strength_prior=0.7, strength_posterior=0.7,
        citation="smoke",
    )
    # Pre-create the outcome so the LINKS_TO edge MERGE finds a target
    outcome = OutcomeBelief(
        id="outcome.smoke.AAPL",
        ticker="AAPL", horizon_min=1440,
        statement="AAPL closes up tomorrow",
        p_up=0.5,  # placeholder; updated below
        activated_at=t0 + 60,
    )
    store.append_outcome_belief(outcome)
    store.upsert_link(link)
    print(f"upserted link: {link.id}")

    # Compute outcome posterior
    p_up, contrib_b, contrib_l = propagate_outcome(
        outcome_prior_p=0.5,
        state_beliefs={"belief.smoke.A": a, "belief.smoke.B": b},
        links=[link],
    )
    outcome.p_up = p_up
    outcome.contributing_state_beliefs = contrib_b
    outcome.contributing_links = contrib_l
    store.append_outcome_belief(outcome)
    print(f"propagated outcome.p_up = {p_up:.4f}")

    # Reload from Neo4j to verify round-trip
    hr()
    print("reload from Neo4j:")
    a2 = store.get_state_belief("belief.smoke.A")
    b2 = store.get_state_belief("belief.smoke.B")
    link2 = store.get_link("link.smoke.B→outcome")
    print(f"  belief.smoke.A: current_p={a2.current_p:.4f}  scope={a2.scope}")
    print(f"  belief.smoke.B: current_p={b2.current_p:.4f}  context={b2.context}")
    print(f"  link.smoke.B→outcome: strength_posterior={link2.strength_posterior}, "
          f"direction={link2.direction}")

    assert abs(a2.current_p - a.current_p) < 1e-9
    assert abs(b2.current_p - b.current_p) < 1e-9
    assert link2.strength_posterior == link.strength_posterior

    # Also verify the LINKS_TO edge exists
    with store.driver.session() as sess:
        row = sess.run("""
            MATCH (premise:Ecology:StateBelief {id: 'belief.smoke.B'})
                  -[r:LINKS_TO]->(target {id: 'outcome.smoke.AAPL'})
            RETURN r.strength AS s, r.direction AS d
        """).single()
        if row:
            print(f"  edge: belief.smoke.B-[LINKS_TO]->outcome.smoke.AAPL "
                  f"strength={row['s']} direction={row['d']}")
        else:
            print("  WARN: LINKS_TO edge not found")

    # Verify ACTIVATED edge
    with store.driver.session() as sess:
        row = sess.run("""
            MATCH (e:Ecology:TrophicEvent {id: 'event.smoke.1'})
                  -[r:ACTIVATED]->(b:Ecology:StateBelief {id: 'belief.smoke.B'})
            RETURN r.magnitude AS m, r.species_id AS s
        """).single()
        if row:
            print(f"  edge: event.smoke.1-[ACTIVATED]->belief.smoke.B "
                  f"magnitude={row['m']} species={row['s']}")
        else:
            print("  WARN: ACTIVATED edge not found")

    hr()
    print("\n✓ Neo4j round-trip passed\n")

    # Cleanup
    print("cleaning up smoke data...")
    cleanup_smoke_data(store)
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
