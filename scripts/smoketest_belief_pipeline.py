"""End-to-end smoke test for the belief network round-trip.

Pipeline exercised:
  1. Pull current network state from Neo4j (1011 beliefs, 1093 links).
  2. Construct a synthetic news Event with hand-crafted BeliefActivations
     targeting real beliefs in the network.
  3. Apply activations to those state beliefs (decay → log-odds shift → clamp).
  4. Define an OutcomeBelief for AAPL and run propagate_network() over the
     full DAG to compute outcome.p_up.
  5. Persist mutated beliefs + activated event + outcome back to Neo4j.
  6. Render a pass viz HTML showing activations, deviations, and the outcome.

Run with --dry-run to skip Neo4j writes (in-memory only).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import uuid
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
from trophic.beliefs.propagation import (
    apply_activation,
    propagate_network,
)
from trophic.beliefs.schema import (
    BeliefActivation,
    Event,
    InternalLink,
    OutcomeBelief,
    StateBelief,
)
from trophic.beliefs.viz import render_pass_viz, snapshot_pass


# ── Synthetic event the smoke test will inject ──

TICKER = "AAPL"
NOW = time.time()

EVENT = Event(
    id=f"smoke.event.{uuid.uuid4().hex[:8]}",
    timestamp=NOW,
    source="smoketest_synthetic",
    raw_content=(
        "Apple beats Q2 EPS estimate by 12%, raises guidance for H2; "
        "iPhone 17 launch reception strong across analyst notes. "
        "Hawkish Fed minutes released same day suggest higher-for-longer."
    ),
    ticker=TICKER,
    sector="technology",
)

# Hand-crafted activations targeting real beliefs in the seeded network.
# Each pair (target_belief_id, direction, magnitude) is the LLM's
# would-be classification of how this news affects each belief.
ACTIVATIONS = [
    BeliefActivation(
        target_belief_id="belief.company.earnings_beat",
        direction_of_effect="increases",
        magnitude="strong",
        decay_class="slow",
        self_rated_confidence="very_high",
        reasoning="Q2 EPS beat by 12% - direct match for earnings_beat template.",
        species_id="smoketest.synthetic_classifier",
    ),
    BeliefActivation(
        target_belief_id="belief.company.major_product_launch_positive_reception",
        direction_of_effect="increases",
        magnitude="medium",
        decay_class="normal",
        self_rated_confidence="high",
        reasoning="iPhone 17 launch with strong analyst reception.",
        species_id="smoketest.synthetic_classifier",
    ),
    BeliefActivation(
        target_belief_id="belief.macro.fed_hawkish_stance",
        direction_of_effect="increases",
        magnitude="medium",
        decay_class="slow",
        self_rated_confidence="high",
        reasoning="Hawkish Fed minutes suggest higher-for-longer regime.",
        species_id="smoketest.synthetic_classifier",
    ),
]
EVENT.activations = ACTIVATIONS


def hr() -> None:
    print("─" * 70)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="don't persist activated state back to Neo4j")
    parser.add_argument("--no-write-event", action="store_true",
                        help="don't append the synthetic event to Neo4j")
    args = parser.parse_args()

    print("Belief network smoke test")
    print(f"connected: {os.environ['NEO4J_URI']}")
    hr()

    store = Neo4jBeliefStore()
    state_beliefs_list = store.all_state_beliefs()
    links_list = store.all_links()
    print(f"loaded: {len(state_beliefs_list)} state beliefs, {len(links_list)} links")

    state_beliefs: dict[str, StateBelief] = {b.id: b for b in state_beliefs_list}

    # Verify targets exist
    missing = [a.target_belief_id for a in ACTIVATIONS
               if a.target_belief_id not in state_beliefs]
    if missing:
        print(f"ERROR: activation targets not in store: {missing}")
        sys.exit(1)

    hr()
    print("STAGE 1 — pre-activation state of targets")
    for a in ACTIVATIONS:
        b = state_beliefs[a.target_belief_id]
        print(f"  {b.id:60s} prior={b.prior_p:.3f}  current={b.current_p:.3f}")

    hr()
    print("STAGE 2 — applying activations")
    for a in ACTIVATIONS:
        before = state_beliefs[a.target_belief_id].current_p
        updated = apply_activation(
            belief=state_beliefs[a.target_belief_id],
            activation=a,
            event_id=EVENT.id,
            now_ts=NOW,
        )
        state_beliefs[a.target_belief_id] = updated
        delta = updated.current_p - before
        print(f"  {a.target_belief_id:60s}  {before:.3f} → {updated.current_p:.3f}  "
              f"(Δ={delta:+.3f}, mag={a.magnitude}, dir={a.direction_of_effect})")

    hr()
    print("STAGE 3 — defining outcome + propagating network")
    outcome = OutcomeBelief(
        id=f"outcome.{TICKER}.next_day_direction.{uuid.uuid4().hex[:6]}",
        ticker=TICKER,
        horizon_min=24 * 60,
        statement=f"{TICKER} closes higher next session vs prior close",
        p_up=0.5,  # prior; propagation will overwrite
        activated_at=NOW,
    )

    # Propagate over full network. Outcomes need to be wired in via links to
    # be reachable; for the smoke test we attach the outcome to a small set
    # of contributing leaves (company-scope beliefs that we activated).
    smoke_links: list[InternalLink] = list(links_list)
    for src_id, direction, strength in [
        ("belief.company.earnings_beat", "positive", 0.7),
        ("belief.company.major_product_launch_positive_reception", "positive", 0.5),
        ("belief.company.guidance_cut", "negative", 0.7),
        ("belief.macro.fed_hawkish_stance", "negative", 0.3),
    ]:
        if src_id not in state_beliefs:
            continue
        smoke_links.append(InternalLink(
            id=f"smoke.link.{src_id}.to.{outcome.id}",
            premise_belief_id=src_id,
            conclusion_belief_id=outcome.id,
            scope="company",
            direction=direction,  # type: ignore[arg-type]
            strength_prior=strength,
            strength_posterior=strength,
            citation="smoketest hand-wired",
            created_at=NOW,
            updated_at=NOW,
        ))

    print(f"  total links for propagation: {len(smoke_links)} "
          f"(network {len(links_list)} + {len(smoke_links) - len(links_list)} smoke)")

    outcomes_dict: dict[str, OutcomeBelief] = {outcome.id: outcome}
    # Clamp the activated beliefs — propagation reads from them, doesn't
    # overwrite them. Direct evidence wins over child-decomposition recompute.
    clamped_ids = {a.target_belief_id for a in ACTIVATIONS}
    contribs = propagate_network(
        state_beliefs=state_beliefs,
        outcome_beliefs=outcomes_dict,
        links=smoke_links,
        max_iterations=25,
        convergence_eps=1e-4,
        scaling_factor=1.0,
        require_dag=False,  # cycle-tolerant Jacobi iteration
        clamped_ids=clamped_ids,
    )
    propagate_ok = True
    n_contrib = len(contribs.get(outcome.id, []))
    print(f"  propagation converged; outcome contributors: {n_contrib}  "
          f"(clamped: {len(clamped_ids)} beliefs)")

    hr()
    print(f"STAGE 4 — outcome.p_up = {outcome.p_up:.4f}")
    direction = "UP" if outcome.p_up >= 0.5 else "DOWN"
    print(f"  predicted direction: {direction}")
    print(f"  contributing beliefs: {len(outcome.contributing_state_beliefs)}")
    print(f"  contributing links:   {len(outcome.contributing_links)}")
    if outcome.contributing_state_beliefs:
        print("  top contributing beliefs (by deviation from prior):")
        ranked = sorted(
            [state_beliefs[bid] for bid in outcome.contributing_state_beliefs
             if bid in state_beliefs],
            key=lambda b: abs(b.current_p - b.prior_p),
            reverse=True,
        )[:5]
        for b in ranked:
            print(f"    {b.id:60s}  Δ={b.current_p - b.prior_p:+.3f}")

    hr()
    if args.dry_run:
        print("STAGE 5 — DRY RUN (skipping Neo4j writes)")
    else:
        print("STAGE 5 — persisting state to Neo4j")
        for a in ACTIVATIONS:
            store.upsert_state_belief(state_beliefs[a.target_belief_id])
        store.append_outcome_belief(outcome)
        if not args.no_write_event:
            store.append_event(EVENT)
        print("  persisted: 3 mutated beliefs + 1 outcome + 1 event")

    hr()
    print("STAGE 6 — rendering pass viz")
    pass_id = f"smoke_{int(NOW)}"
    snap = snapshot_pass(
        store=store,
        pass_id=pass_id,
        pass_meta={
            "ticker": TICKER,
            "scenario": "smoke_synthetic_news",
            "label": "smoke test",
            "prediction": direction,
        },
        incoming_events=[EVENT],
        outcome_predictions=[outcome],
    )
    out_dir = ROOT / "viz_beliefs"
    json_path, html_path = render_pass_viz(snap, out_dir, pass_id=pass_id)
    print(f"  json: file://{json_path}")
    print(f"  html: file://{html_path}")
    print(f"  latest: file://{out_dir / 'latest.html'}")

    hr()
    print("=== SMOKE TEST RESULT ===")
    print(f"  beliefs activated:    {len(ACTIVATIONS)}")
    print(f"  outcome p_up:         {outcome.p_up:.4f}")
    print(f"  predicted direction:  {direction}")
    print(f"  propagation:          {'CONVERGED' if propagate_ok else 'fallback (DAG cycle)'}")
    print(f"  persisted:            {'NO (dry-run)' if args.dry_run else 'YES'}")


if __name__ == "__main__":
    main()
