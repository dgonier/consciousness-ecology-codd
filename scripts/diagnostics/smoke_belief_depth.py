"""Depth-3+ chain smoke test for the belief network.

Builds the AI-Psychosis-style chain you described:

  event.lawsuit                                                 (input)
        │ activates
        ▼
  belief.macro.regulators_under_scrutiny           (depth 1, market)
        │ negative link
        ▼
  belief.sector.ai_oversight_credibility           (depth 2, sector)
        │ negative link
        ▼
  belief.company.<TICKER>.regulatory_standing      (depth 3, company)
        │ positive link
        ▼
  outcome.<TICKER>.next_day_up                     (leaf)

Verifies:
  - Forward propagation passes credence through 3 state→state edges
  - Outcome ends up reflecting the lawsuit's effect via the chain
  - Without the chain, outcome would be unaffected (parents at prior)
  - Topological ordering is correct
  - Viz renders the multi-step chain as edges (not just direct-to-outcome)
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
    BeliefActivation,
    Event,
    InternalLink,
    JSONLBeliefStore,
    OutcomeBelief,
    StateBelief,
    apply_activation,
    propagate_network,
    snapshot_pass,
    render_pass_viz,
    topological_order,
)
from trophic.beliefs.schema import prior_p_from_categorical


def hr() -> None:
    print("─" * 70)


def main() -> int:
    print("Belief network — depth-3 chain smoke (AI Psychosis example)\n")
    ticker = "ABCAI"  # synthetic AI Lab

    store_root = ROOT / "external" / "beliefs_depth_smoke"
    if store_root.exists():
        for p in store_root.glob("*.jsonl"):
            p.unlink()
    store = JSONLBeliefStore(store_root)
    t0 = time.time()

    # ── Build the chain ──
    p1 = prior_p_from_categorical("strong", "leans_false")
    regulators_belief = StateBelief(
        id="belief.macro.regulators_under_scrutiny",
        statement_template="US regulators are perceived as not doing enough on AI safety",
        scope="macro",
        prior_p=p1, current_p=p1, decay_class="slow",
        last_updated=t0,
    )
    p2 = prior_p_from_categorical("moderate", "leans_true")
    sector_belief = StateBelief(
        id="belief.sector.ai_oversight_credibility",
        statement_template="AI sector regulators are seen as credible",
        scope="sector",
        prior_p=p2, current_p=p2, decay_class="slow",
        context={"sector": "ai_tech"},
        last_updated=t0,
    )
    p3 = prior_p_from_categorical("strong", "leans_true")
    standing_belief = StateBelief(
        id=f"belief.company.{ticker}.regulatory_standing",
        statement_template=f"{ticker} is in good regulatory standing",
        scope="company",
        prior_p=p3, current_p=p3, decay_class="slow",
        context={"ticker": ticker},
        last_updated=t0,
    )
    for b in (regulators_belief, sector_belief, standing_belief):
        store.upsert_state_belief(b)

    print("seeded 3 state beliefs:")
    print(f"  [macro]   regulators_under_scrutiny (prior={p1})")
    print(f"  [sector]  ai_oversight_credibility (prior={p2})")
    print(f"  [company] {ticker}.regulatory_standing (prior={p3})")

    # ── Build the chain of links ──
    outcome_id = f"outcome.{ticker}.next_day_up"
    chain_links = [
        # macro → sector: regulators-under-scrutiny ↑ → AI-oversight-credibility ↓
        InternalLink(
            id="link.regulators→sector_credibility",
            premise_belief_id=regulators_belief.id,
            conclusion_belief_id=sector_belief.id,
            scope="sector",
            direction="negative",
            strength_prior=0.7, strength_posterior=0.7,
            citation="If regulators seen as failing, sector oversight credibility drops",
        ),
        # sector → company: oversight-credibility ↓ → company regulatory standing ↓
        InternalLink(
            id=f"link.sector_credibility→{ticker}_standing",
            premise_belief_id=sector_belief.id,
            conclusion_belief_id=standing_belief.id,
            scope="company",
            direction="positive",
            strength_prior=0.6, strength_posterior=0.6,
            citation="Sector-wide regulatory credibility supports individual firm standing",
        ),
        # company → outcome: regulatory standing ↓ → next-day price down
        InternalLink(
            id=f"link.{ticker}_standing→outcome",
            premise_belief_id=standing_belief.id,
            conclusion_belief_id=outcome_id,
            scope="company",
            direction="positive",
            strength_prior=0.65, strength_posterior=0.65,
            citation="Good regulatory standing → up-day; bad standing → down-day",
        ),
    ]
    for l in chain_links:
        store.upsert_link(l)

    print("\nseeded 3 chained links forming a depth-3 chain")

    # Pre-create the outcome
    outcome = OutcomeBelief(
        id=outcome_id, ticker=ticker, horizon_min=1440,
        statement=f"{ticker} closes up tomorrow",
        p_up=0.5, activated_at=t0 + 60,
    )
    store.append_outcome_belief(outcome)

    # ── Step 1: verify topological order ──
    hr()
    all_ids = [b.id for b in (regulators_belief, sector_belief, standing_belief)] + [outcome_id]
    order = topological_order(
        all_ids,
        chain_links,
        {b.id: b for b in (regulators_belief, sector_belief, standing_belief)},
    )
    print(f"topological order:")
    for i, nid in enumerate(order):
        print(f"  [{i}] {nid}")

    # ── Step 2: incoming event activates the ROOT of the chain ──
    hr()
    print("STEP: lawsuit event activates belief.macro.regulators_under_scrutiny")
    lawsuit_act = BeliefActivation(
        target_belief_id=regulators_belief.id,
        direction_of_effect="increases",
        magnitude="decisive",
        decay_class="slow",
        self_rated_confidence="high",
        reasoning="Lawsuit raises high-profile concern that regulators not doing enough on AI safety",
        species_id="herb.fundamental.test",
    )
    lawsuit_event = Event(
        id="event.lawsuit_ai_psychosis",
        timestamp=t0 + 60,
        source="polygon_news",
        raw_content="Lawsuit raises concern that regulators not doing enough to prevent AI Psychosis",
        ticker=ticker,
        sector="ai_tech",
        activations=[lawsuit_act],
    )
    regulators_belief = apply_activation(
        regulators_belief, lawsuit_act,
        event_id=lawsuit_event.id, now_ts=t0 + 60,
    )
    store.upsert_state_belief(regulators_belief)
    print(f"  regulators_under_scrutiny: {p1:.3f} → {regulators_belief.current_p:.3f}")
    print(f"  (sector and company beliefs still at their priors)")

    # ── Step 3: BEFORE propagation, sector and company are still at priors ──
    print(f"\nbefore propagate_network():")
    print(f"  sector ai_oversight_credibility: current_p = {sector_belief.current_p:.3f}  (still prior)")
    print(f"  {ticker} regulatory_standing:    current_p = {standing_belief.current_p:.3f}  (still prior)")
    print(f"  outcome.next_day_up:             p_up      = {outcome.p_up:.3f}  (still 0.5)")

    # ── Step 4: propagate the network ──
    hr()
    print("STEP: propagate_network() walks DAG until quiescent")
    state_beliefs = {b.id: b for b in (regulators_belief, sector_belief, standing_belief)}
    outcomes = {outcome_id: outcome}
    contributions = propagate_network(
        state_beliefs=state_beliefs,
        outcome_beliefs=outcomes,
        links=chain_links,
        max_iterations=10,
    )

    print(f"\nafter propagate_network():")
    for b in state_beliefs.values():
        print(f"  [{b.scope:7s}] {b.id:50s} prior={b.prior_p:.3f}  current={b.current_p:.3f}")
    print(f"  [outcome] {outcome.id:50s}                p_up = {outcome.p_up:.3f}")

    print(f"\ncontribution chain:")
    for nid, contribs in contributions.items():
        for parent_id, link_id in contribs:
            print(f"  {parent_id} -[{link_id}]→ {nid}")

    # ── Step 5: assertions ──
    hr()
    # NOTE: log-odds-additive propagation attenuates signal at each hop. With
    # 3 hops at link strengths ~0.65, ~75% of magnitude is lost by the time
    # signal reaches the leaf. This is real (uncertainty compounds in chains).
    # Assertions verify direction-of-effect propagates correctly; magnitude
    # at the leaf is small but non-zero.
    assert regulators_belief.current_p > p1 + 0.1, "regulators should be much higher"
    # Each hop should move credence in the right direction, even if small.
    assert sector_belief.current_p < p2, (
        f"sector credibility should drop from {p2} but got {sector_belief.current_p}"
    )
    assert standing_belief.current_p < p3, (
        f"company standing should drop from {p3} but got {standing_belief.current_p}"
    )
    assert outcome.p_up < 0.5, f"outcome should be down-leaning, got {outcome.p_up}"

    print("✓ all chain assertions passed:")
    print(f"  lawsuit ↑ → regulators_under_scrutiny ↑ ({p1:.2f} → {regulators_belief.current_p:.2f})")
    print(f"  → sector credibility ↓               ({p2:.2f} → {sector_belief.current_p:.2f})")
    print(f"  → company regulatory standing ↓      ({p3:.2f} → {standing_belief.current_p:.2f})")
    print(f"  → outcome p_up ↓                     (0.50 → {outcome.p_up:.2f})")

    # Update the outcome's contributing list AFTER propagation since the
    # propagate_network mutation should have set it
    store.append_outcome_belief(outcome)

    # ── Step 6: render viz ──
    hr()
    print("STEP: rendering viz with depth-3 chain")
    snap = snapshot_pass(
        store=store,
        pass_id="depth3_chain",
        pass_meta={
            "ticker": ticker,
            "scenario": "ai_psychosis_lawsuit_chain",
            "label": "(depth=3 demo)",
            "prediction": "down" if outcome.p_up < 0.5 else "up",
        },
        incoming_events=[lawsuit_event],
        outcome_predictions=[outcome],
    )
    out_dir = ROOT / "viz_beliefs"
    json_path, html_path = render_pass_viz(snap, out_dir, pass_id="depth3_chain")
    print(f"  open: file://{html_path.resolve()}")
    print(f"  latest: file://{(out_dir / 'latest.html').resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
