"""Smoke-test the carnivore aggregator end-to-end against the populated graph.

Plants synthetic activations on a few AAPL/CVX ticker-suffixed beliefs,
runs propagate_network, then runs aggregate() and prints the Observation
set as JSON.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
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

from trophic.beliefs.carnivore import aggregate
from trophic.beliefs.neo4j_store import Neo4jBeliefStore
from trophic.beliefs.propagation import apply_activation, propagate_network
from trophic.beliefs.schema import (
    BeliefActivation,
    Event,
    OutcomeBelief,
)


def hr() -> None:
    print("─" * 78)


def main() -> None:
    print("Carnivore aggregator smoke test")
    hr()

    store = Neo4jBeliefStore()
    state_list = store.all_state_beliefs()
    state = {b.id: b for b in state_list}
    links = store.all_links()

    # Reset our planted beliefs to prior so reruns are deterministic
    target_ids = [
        "belief.company.earnings_beat__ticker_AAPL",
        "belief.company.major_product_launch_positive_reception__ticker_AAPL",
        "belief.company.guidance_cut__ticker_AAPL",  # conflicting!
        "belief.company.earnings_miss__ticker_CVX",  # bearish CVX
    ]
    for tid in target_ids:
        if tid in state:
            state[tid].current_p = state[tid].prior_p

    # Plant activations
    NOW = time.time()
    fake_event = Event(
        id="smoke.carnivore.event.001",
        timestamp=NOW,
        source="carnivore_smoke",
        raw_content="(synthetic — earnings beat at AAPL with strong launch reception, "
                    "soft guidance bullet; CVX earnings miss)",
        ticker="AAPL",
    )
    activations = [
        BeliefActivation(
            target_belief_id="belief.company.earnings_beat__ticker_AAPL",
            direction_of_effect="increases", magnitude="strong",
            decay_class="slow", self_rated_confidence="very_high",
            reasoning="Q2 EPS beat by 12%",
            species_id="event_classifier.test",
        ),
        BeliefActivation(
            target_belief_id="belief.company.major_product_launch_positive_reception__ticker_AAPL",
            direction_of_effect="increases", magnitude="medium",
            decay_class="normal", self_rated_confidence="high",
            reasoning="iPhone 17 launch strong reception",
            species_id="event_classifier.test",
        ),
        BeliefActivation(
            target_belief_id="belief.company.guidance_cut__ticker_AAPL",
            direction_of_effect="increases", magnitude="weak",
            decay_class="slow", self_rated_confidence="medium",
            reasoning="Slight downside revision in Q3 guidance",
            species_id="event_classifier.test",
        ),
        BeliefActivation(
            target_belief_id="belief.company.earnings_miss__ticker_CVX",
            direction_of_effect="increases", magnitude="medium",
            decay_class="slow", self_rated_confidence="high",
            reasoning="CVX EPS missed",
            species_id="event_classifier.test",
        ),
    ]
    fake_event.activations = activations

    # Apply to state
    for a in activations:
        if a.target_belief_id in state:
            updated = apply_activation(state[a.target_belief_id], a, fake_event.id, NOW)
            state[a.target_belief_id] = updated

    # Pull outcomes from store
    outcomes_dict: dict[str, OutcomeBelief] = {}
    # Use Neo4j cypher to fetch outcomes (no helper exists; minimal hop)
    with store.driver.session() as sess:
        rows = sess.run("MATCH (o:Ecology:OutcomeBelief) RETURN o")
        for r in rows:
            n = r["o"]
            outcomes_dict[n["id"]] = OutcomeBelief(
                id=n["id"],
                ticker=n["ticker"],
                horizon_min=int(n.get("horizon_min", 1440)),
                statement=n.get("statement", ""),
                p_up=0.5,
            )
    print(f"loaded: {len(state)} state beliefs, {len(outcomes_dict)} outcomes, {len(links)} links")

    # Propagate (clamping our activated targets so they aren't overwritten)
    clamped = {a.target_belief_id for a in activations}
    contribs = propagate_network(
        state_beliefs=state,
        outcome_beliefs=outcomes_dict,
        links=links,
        clamped_ids=clamped,
        max_iterations=25,
    )
    hr()

    # Aggregate
    species_origin = {a.target_belief_id: a.species_id for a in activations}
    obs = aggregate(
        state_beliefs=state,
        outcome_beliefs=outcomes_dict,
        links=links,
        triggering_events=[fake_event],
        watchlist_focus=["MSFT"],  # bias toward MSFT to test focus path
        species_origin=species_origin,
        salience_floor=0.02,
    )
    print(f"observations emitted: {len(obs)}")
    hr()

    # Print top 10 by salience
    for o in obs[:10]:
        print(f"\n[{o.action}] {o.ticker}  p_up={o.p_up:.3f}  salience={o.salience:.3f}  "
              f"conf={o.confidence}  in_focus={o.in_focus}")
        print(f"  summary: {o.reasoning_summary}")
        for c in o.causal_chain[:3]:
            tag = c.parent_belief_id[:60]
            print(f"  → {tag}  Δ={c.parent_deviation:+.3f}  contrib_lo={c.contribution_log_odds:+.3f}")
        if o.conflicting_signals:
            print(f"  conflicts:")
            for c in o.conflicting_signals[:2]:
                print(f"    ⚠ {c.parent_belief_id[:60]}  contrib_lo={c.contribution_log_odds:+.3f}")

    hr()
    print("Top observation as JSON:")
    if obs:
        print(json.dumps(obs[0].to_dict(), indent=2))


if __name__ == "__main__":
    main()
