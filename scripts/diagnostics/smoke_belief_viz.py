"""End-to-end smoke for the belief network viz.

Builds a small network (4 state beliefs, 1 outcome, 3 links), runs one
pass with a couple of incoming activations, renders an html snapshot,
prints the path. Open in a browser.
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
    propagate_outcome,
    snapshot_pass,
    render_pass_viz,
)
from trophic.beliefs.schema import prior_p_from_categorical


def main() -> int:
    store_root = ROOT / "external" / "beliefs_viz_smoke"
    if store_root.exists():
        for p in store_root.glob("*.jsonl"):
            p.unlink()
    store = JSONLBeliefStore(store_root)
    t0 = time.time()
    ticker = "AAPL"

    # ── State beliefs at three scopes ──
    fed_p = prior_p_from_categorical("strong", "leans_false")
    fed_belief = StateBelief(
        id="belief.macro.fed_hawkish",
        statement_template="Fed is in hawkish mode this quarter",
        scope="macro",
        prior_p=fed_p, current_p=fed_p, decay_class="glacial",
        last_updated=t0,
    )
    sec_p = prior_p_from_categorical("well_established", "leans_false")
    sec_belief = StateBelief(
        id=f"belief.company.{ticker}.sec_enforcement_q",
        statement_template=f"{ticker} faces SEC enforcement this quarter",
        scope="company",
        prior_p=sec_p, current_p=sec_p, decay_class="slow",
        context={"ticker": ticker},
        last_updated=t0,
    )
    earnings_p = prior_p_from_categorical("strong", "leans_true")
    earnings_belief = StateBelief(
        id=f"belief.company.{ticker}.earnings_beat",
        statement_template=f"{ticker} beats earnings this quarter",
        scope="company",
        prior_p=earnings_p, current_p=earnings_p, decay_class="slow",
        context={"ticker": ticker},
        last_updated=t0,
    )
    sector_p = prior_p_from_categorical("moderate", "neutral")
    sector_belief = StateBelief(
        id="belief.sector.tech.regulatory_scrutiny",
        statement_template="Tech sector is under heightened regulatory scrutiny",
        scope="sector",
        prior_p=sector_p, current_p=sector_p, decay_class="slow",
        context={"sector": "tech"},
        last_updated=t0,
    )
    for b in (fed_belief, sec_belief, earnings_belief, sector_belief):
        store.upsert_state_belief(b)

    # ── Links into the outcome ──
    outcome_id = f"outcome.{ticker}.next_day_up"
    links = [
        InternalLink(
            id=f"link.{ticker}.fed→outcome",
            premise_belief_id=fed_belief.id,
            conclusion_belief_id=outcome_id,
            scope="macro",
            direction="negative",
            strength_prior=0.55, strength_posterior=0.55,
        ),
        InternalLink(
            id=f"link.{ticker}.sec→outcome",
            premise_belief_id=sec_belief.id,
            conclusion_belief_id=outcome_id,
            scope="company",
            direction="negative",
            strength_prior=0.7, strength_posterior=0.7,
        ),
        InternalLink(
            id=f"link.{ticker}.earnings→outcome",
            premise_belief_id=earnings_belief.id,
            conclusion_belief_id=outcome_id,
            scope="company",
            direction="positive",
            strength_prior=0.65, strength_posterior=0.65,
        ),
        InternalLink(
            id=f"link.{ticker}.sector→outcome",
            premise_belief_id=sector_belief.id,
            conclusion_belief_id=outcome_id,
            scope="sector",
            direction="negative",
            strength_prior=0.5, strength_posterior=0.5,
        ),
    ]
    for l in links:
        store.upsert_link(l)

    # ── Incoming events for this pass ──
    earnings_act = BeliefActivation(
        target_belief_id=earnings_belief.id,
        direction_of_effect="increases",
        magnitude="strong",
        decay_class="slow",
        self_rated_confidence="high",
        reasoning=f"Whisper number suggests {ticker} earnings will beat consensus by 8%",
        species_id="herb.fundamental.test",
    )
    sec_act = BeliefActivation(
        target_belief_id=sec_belief.id,
        direction_of_effect="increases",
        magnitude="medium",
        decay_class="slow",
        self_rated_confidence="medium",
        reasoning=f"WSJ reports SEC has opened informal inquiry into {ticker}'s revenue recognition",
        species_id="herb.fundamental.test",
    )

    earnings_event = Event(
        id=f"event.{ticker}.earnings_whisper",
        timestamp=t0 + 30,
        source="polygon_news",
        raw_content=f"Wall Street whisper number for {ticker} suggests +8% earnings beat",
        ticker=ticker,
        sector="tech",
        activations=[earnings_act],
    )
    sec_event = Event(
        id=f"event.{ticker}.sec_inquiry",
        timestamp=t0 + 45,
        source="polygon_news",
        raw_content=f"WSJ: SEC opens informal inquiry into {ticker} revenue recognition",
        ticker=ticker,
        sector="tech",
        activations=[sec_act],
    )

    # Apply activations
    earnings_belief = apply_activation(
        earnings_belief, earnings_act,
        event_id=earnings_event.id, now_ts=t0 + 30,
    )
    sec_belief = apply_activation(
        sec_belief, sec_act,
        event_id=sec_event.id, now_ts=t0 + 45,
    )
    store.upsert_state_belief(earnings_belief)
    store.upsert_state_belief(sec_belief)

    print(f"after activations:")
    print(f"  earnings (prior={earnings_p}): now {earnings_belief.current_p:.3f}")
    print(f"  sec (prior={sec_p}): now {sec_belief.current_p:.3f}")

    # ── Compute outcome posterior ──
    sb = {b.id: b for b in [fed_belief, sec_belief, earnings_belief, sector_belief]}
    p_up, contrib_b, contrib_l = propagate_outcome(
        outcome_prior_p=0.5,
        state_beliefs=sb,
        links=links,
    )
    outcome = OutcomeBelief(
        id=outcome_id,
        ticker=ticker,
        horizon_min=1440,
        statement=f"{ticker} closes up tomorrow",
        p_up=p_up,
        contributing_state_beliefs=contrib_b,
        contributing_links=contrib_l,
        activated_at=t0 + 60,
    )
    print(f"\noutcome.p_up = {p_up:.4f}")
    print(f"  contributing beliefs: {contrib_b}")

    # ── Build snapshot + render html ──
    snap = snapshot_pass(
        store=store,
        pass_id="smoke_001",
        pass_meta={
            "ticker": ticker,
            "scenario": f"{ticker}_2026-05-09",
            "label": "(unknown)",
            "prediction": "up" if p_up >= 0.5 else "down",
        },
        incoming_events=[earnings_event, sec_event],
        outcome_predictions=[outcome],
    )

    out_dir = ROOT / "viz_beliefs"
    json_path, html_path = render_pass_viz(snap, out_dir, pass_id="smoke_001")
    print(f"\n✓ wrote {json_path.name} and {html_path.name}")
    print(f"  open: file://{html_path.resolve()}")
    print(f"  latest: file://{(out_dir / 'latest.html').resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
