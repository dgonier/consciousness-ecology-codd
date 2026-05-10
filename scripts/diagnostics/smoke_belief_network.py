"""Smoke test for the Bayesian belief network.

Builds a tiny network with three state beliefs and one outcome, applies
a few activations + decays, and prints state at each step. Verifies:
  - decay-toward-prior (not toward 0.5) for non-trivial priors
  - log-odds activation math
  - outcome propagation from contributing beliefs
  - JSONL persistence round-trip
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path("/home/dgonier/ecology_experiment/trophic")
sys.path.insert(0, str(ROOT))

from trophic.beliefs import (
    BeliefActivation,
    Event,
    InternalLink,
    JSONLBeliefStore,
    OutcomeBelief,
    StateBelief,
    apply_activation,
    apply_decay,
    propagate_outcome,
)
from trophic.beliefs.schema import prior_p_from_categorical


def hr() -> None:
    print("─" * 70)


def main() -> int:
    print("Bayesian belief network — smoke test\n")

    # ── 1. Set up store + seed beliefs ──
    store_root = ROOT / "external" / "beliefs_smoke"
    if store_root.exists():
        for p in store_root.glob("*.jsonl"):
            p.unlink()
    store = JSONLBeliefStore(store_root)
    print(f"store: {store_root}")

    t0 = time.time()

    # Belief A: well-established, leans-true (e.g. "S&P closes up on a given day")
    p_a = prior_p_from_categorical("well_established", "leans_true")
    print(f"\nbelief.A — prior_p (well_established, leans_true) = {p_a}")
    a = StateBelief(
        id="belief.A",
        statement_template="S&P closes up on a given day",
        scope="market",
        prior_p=p_a,
        current_p=p_a,
        decay_class="slow",
        last_updated=t0,
    )
    store.upsert_state_belief(a)

    # Belief B: well-established + leans_false (rare event)
    p_b = prior_p_from_categorical("well_established", "leans_false")
    print(f"belief.B — prior_p (well_established, leans_false) = {p_b}")
    b = StateBelief(
        id="belief.B",
        statement_template="Company faces SEC enforcement this quarter",
        scope="company",
        prior_p=p_b,
        current_p=p_b,
        decay_class="slow",
        context={"ticker": "AAPL"},
        last_updated=t0,
    )
    store.upsert_state_belief(b)

    # Belief C: novel — agnostic
    p_c = prior_p_from_categorical("novel", "neutral")
    print(f"belief.C — prior_p (novel, neutral) = {p_c}")
    c = StateBelief(
        id="belief.C",
        statement_template="AAPL announces a fancy new gadget",
        scope="company",
        prior_p=p_c,
        current_p=p_c,
        decay_class="normal",
        context={"ticker": "AAPL"},
        last_updated=t0,
    )
    store.upsert_state_belief(c)

    # ── 2. Apply some activations ──
    hr()
    print("STEP 1: activate belief.B ('SEC probe announced') with decisive negative shift")
    act = BeliefActivation(
        target_belief_id="belief.B",
        direction_of_effect="increases",
        magnitude="decisive",
        decay_class="slow",
        self_rated_confidence="high",
        reasoning="SEC probe news is decisive evidence",
        species_id="test.smoke",
    )
    b = apply_activation(b, act, event_id="event.smoke.1", now_ts=t0 + 60)
    store.upsert_state_belief(b)
    print(f"  belief.B.current_p = {b.current_p:.4f}  (was {p_b:.4f})")
    assert b.current_p > p_b, "credence should rise after positive activation"

    # ── 3. Decay over time ──
    hr()
    print("STEP 2: decay belief.B over time, verify return-to-prior")
    print(f"  prior_p={b.prior_p:.4f}, decay_class={b.decay_class} (24h half-life)")
    for hours in (1, 6, 24, 72, 168):
        snap = StateBelief(**{
            **{k: v for k, v in b.__dict__.items()},
        })
        snap = apply_decay(snap, t0 + 60 + hours * 3600)
        print(f"  +{hours:4d}h → current_p = {snap.current_p:.4f}")
    # Final assertion: after a week, current_p should be very close to prior
    far = StateBelief(**{**b.__dict__})
    far = apply_decay(far, t0 + 60 + 168 * 3600)
    delta_to_prior = abs(far.current_p - b.prior_p)
    print(f"  after 1 week, |current_p − prior_p| = {delta_to_prior:.5f}")
    assert delta_to_prior < 0.01, f"after 1 week, should be near prior; got delta {delta_to_prior}"

    # ── 4. Build a link + outcome ──
    hr()
    print("STEP 3: build link 'belief.B → outcome.AAPL.next_day_up' (negative)")
    link = InternalLink(
        id="link.test.B→outcome",
        premise_belief_id="belief.B",
        conclusion_belief_id="outcome.AAPL.next_day_up",
        scope="company",
        direction="negative",       # SEC probe → less likely up
        strength_prior=0.7,
        strength_posterior=0.7,
        citation="smoke test",
    )
    store.upsert_link(link)
    # Reset belief.B back to its post-activation state for outcome calc
    b_active = StateBelief(**{**b.__dict__})
    b_active.current_p = 0.85  # simulate strongly-elevated probe credence

    p_up, contrib_b, contrib_l = propagate_outcome(
        outcome_prior_p=0.5,
        state_beliefs={
            "belief.A": a, "belief.B": b_active, "belief.C": c,
        },
        links=[link],
    )
    print(f"  outcome.p_up = {p_up:.4f}")
    print(f"  contributing beliefs: {contrib_b}")
    print(f"  contributing links:   {contrib_l}")
    assert p_up < 0.5, "outcome should be down-leaning given negative link with elevated B"

    outcome = OutcomeBelief(
        id="outcome.AAPL.smoke",
        ticker="AAPL",
        horizon_min=1440,
        statement="AAPL closes up tomorrow",
        p_up=p_up,
        contributing_state_beliefs=contrib_b,
        contributing_links=contrib_l,
        activated_at=t0 + 60,
    )
    store.append_outcome_belief(outcome)

    # ── 5. Verify persistence round-trip ──
    hr()
    print("STEP 4: reload store and verify persistence")
    store2 = JSONLBeliefStore(store_root)
    a2 = store2.get_state_belief("belief.A")
    b2 = store2.get_state_belief("belief.B")
    link2 = store2.get_link("link.test.B→outcome")
    assert a2 is not None and a2.current_p == a.current_p
    assert b2 is not None and abs(b2.current_p - b.current_p) < 1e-9
    assert link2 is not None and link2.strength_prior == link.strength_prior
    print(f"  reloaded {len(store2.all_state_beliefs())} beliefs and "
          f"{len(store2.all_links())} links")

    hr()
    print("\n✓ all assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
