"""Pre-populate Neo4j with the ticker watchlist universe.

For each ticker in the universe:
  1. Create 1 OutcomeBelief (next-day direction, p_up=0.5).
  2. Instantiate ticker-suffixed copies of company-scope state-belief templates.
     id pattern: belief.company.{template}__ticker_{TKR}
     statement: same template phrasing with ticker substituted in.
     prior_p: same as parent template (usually 0.5).
  3. Wire links from each ticker-suffixed company belief to that ticker's
     outcome, with direction (positive/negative) per the template's known
     bullishness/bearishness, and strength per template.

Templates we skip:
  - lead_signal_active (placeholder, not ticker-applicable in this form)
  - any template not in TEMPLATE_BIAS

This script is idempotent — uses MERGE everywhere.
"""
from __future__ import annotations

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

from trophic.beliefs.neo4j_store import Neo4jBeliefStore
from trophic.beliefs.schema import (
    InternalLink,
    OutcomeBelief,
    StateBelief,
)


UNIVERSE = [
    "AAPL", "ABBV", "AMZN", "AVGO", "CSCO", "CVX", "GOOG", "HD", "JNJ", "JPM",
    "KO", "MA", "MCD", "MRK", "MSFT", "PEP", "PG", "UNH", "V", "WMT",
]

# (template_short_name, direction, strength) for company-scope beliefs.
# direction: "positive" = belief INCREASING → outcome.p_up INCREASES (bullish)
#            "negative" = belief INCREASING → outcome.p_up DECREASES (bearish)
# strength_prior in [0, 1] — how much the link pulls when activated.
TEMPLATE_BIAS: dict[str, tuple[str, float]] = {
    # Bullish templates — kept moderate; markets often "buy the rumor sell
    # the news" and earnings beats already-priced-in are common.
    "earnings_beat":                          ("positive", 0.55),  # was 0.70
    "regulatory_clearance_received":          ("positive", 0.50),
    "major_product_launch_positive_reception":("positive", 0.50),  # was 0.55
    "activist_investor_takes_stake":          ("positive", 0.45),
    "target_in_announced_ma":                 ("positive", 0.75),
    "dividend_increase_announced":            ("positive", 0.35),  # was 0.40 — slow signal
    "buyback_program_announced":              ("positive", 0.40),  # was 0.45

    # Bearish templates — bumped up. Bad legal/regulatory/governance news
    # tends to dominate good earnings on the day it breaks (per the day-2
    # GOOG antitrust analysis from the eval).
    "earnings_miss":                          ("negative", 0.75),  # was 0.70
    "guidance_cut":                           ("negative", 0.75),  # was 0.65
    # regulatory_action_announced disabled 2026-05-09: 65-day audit
    # showed 29% accurate SELLs (anti-correlated). Strength 0 keeps the
    # link in the graph so decomposer can re-learn it from data, but
    # contributes nothing to outcome propagation today.
    "regulatory_action_announced":            ("negative", 0.0),
    "material_lawsuit_filed":                 ("negative", 0.65),  # was 0.50
    "ceo_departure_unplanned":                ("negative", 0.65),  # was 0.55
    "acting_as_acquirer_in_announced_ma":     ("negative", 0.40),
    # technical_support_broken disabled 2026-05-09: 65-day audit showed
    # 41% accurate SELLs and 10 wrong SELLs (largest single contributor
    # to ecology MCC dragged below zero). event_classifier hallucinates
    # TA signals from news headlines. Strength 0 = decomposer can re-learn.
    "technical_support_broken":               ("negative", 0.0),
    "short_interest_high_and_rising":         ("negative", 0.45),  # was 0.40
}


def hr() -> None:
    print("─" * 70)


def main() -> None:
    print("Ticker-universe instantiation")
    print(f"connected: {os.environ['NEO4J_URI']}")
    hr()

    store = Neo4jBeliefStore()
    NOW = time.time()

    # Resolve template parents from store so we copy their priors / decay
    template_parents: dict[str, StateBelief] = {}
    for tmpl in TEMPLATE_BIAS.keys():
        parent_id = f"belief.company.{tmpl}"
        parent = store.get_state_belief(parent_id)
        if parent is None:
            print(f"  WARN: template not found in store: {parent_id}")
            continue
        template_parents[tmpl] = parent

    print(f"resolved {len(template_parents)}/{len(TEMPLATE_BIAS)} templates")
    hr()

    n_outcomes = 0
    n_beliefs = 0
    n_links = 0

    for tk in UNIVERSE:
        # 1. Outcome
        outcome_id = f"outcome.{tk}.next_day_direction"
        outcome = OutcomeBelief(
            id=outcome_id,
            ticker=tk,
            horizon_min=24 * 60,
            statement=f"{tk} closes higher next session vs prior close",
            p_up=0.5,
            activated_at=NOW,
        )
        store.append_outcome_belief(outcome)
        n_outcomes += 1

        # 2. Ticker-suffixed company state beliefs
        for tmpl, (direction, strength) in TEMPLATE_BIAS.items():
            parent = template_parents.get(tmpl)
            if parent is None:
                continue
            child_id = f"belief.company.{tmpl}__ticker_{tk}"
            child_statement = (
                f"[{tk}] " + parent.statement_template
            )
            child = StateBelief(
                id=child_id,
                statement_template=child_statement,
                scope="company",
                prior_p=parent.prior_p,
                current_p=parent.prior_p,
                decay_class=parent.decay_class,
                context={"ticker": tk, "template": tmpl},
                last_updated=NOW,
            )
            store.upsert_state_belief(child)
            n_beliefs += 1

            # 3. Link from child belief → outcome
            link = InternalLink(
                id=f"link.{tmpl}__ticker_{tk}→outcome",
                premise_belief_id=child_id,
                conclusion_belief_id=outcome_id,
                scope="company",
                direction=direction,  # type: ignore[arg-type]
                strength_prior=strength,
                strength_posterior=strength,
                citation=f"ticker-instantiation seed for {tk}",
                created_at=NOW,
                updated_at=NOW,
            )
            store.upsert_link(link)
            n_links += 1

        # 4. Link the parent template → child (so propagation from
        # decomposed children of the template flows down to the ticker
        # instance). One link per template.
        for tmpl in TEMPLATE_BIAS.keys():
            parent = template_parents.get(tmpl)
            if parent is None:
                continue
            link_id = f"link.template_{tmpl}→ticker_{tk}"
            template_to_ticker = InternalLink(
                id=link_id,
                premise_belief_id=parent.id,
                conclusion_belief_id=f"belief.company.{tmpl}__ticker_{tk}",
                scope="company",
                direction="positive",
                strength_prior=0.40,
                strength_posterior=0.40,
                citation=f"template-to-ticker propagation",
                created_at=NOW,
                updated_at=NOW,
            )
            # Use cycle-safe upsert (the template→ticker is naturally acyclic
            # but let's be defensive)
            inserted = store.upsert_link_if_acyclic(template_to_ticker)
            if inserted:
                n_links += 1

    hr()
    print("=== SUMMARY ===")
    print(f"  outcomes created/updated:   {n_outcomes}")
    print(f"  ticker-suffixed beliefs:    {n_beliefs}")
    print(f"  links created:              {n_links}")
    final_beliefs = len(store.all_state_beliefs())
    final_links = len(store.all_links())
    print(f"  network now: {final_beliefs} beliefs, {final_links} links")


if __name__ == "__main__":
    main()
