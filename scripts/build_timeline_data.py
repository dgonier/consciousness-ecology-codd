"""Build per-day timeline data from pass artifacts + Neo4j catalog.

Output layout (data/firehose_eval/timeline/):

  MANIFEST.json
    - schema_version
    - description
    - dates: ordered list of day stems
    - belief_catalog:
        nodes:
          - { id, scope, statement_template, decay_class, prior_p }
        outcomes:
          - { id, ticker, statement, horizon_min }
        links_summary:
          n_total, n_company_to_outcome, ...

  {date}.json   (per trading day, ~5-10 KB each)
    - date, prev_trading_day, n_news, n_activations, n_observations
    - belief_deltas: [
        { belief_id, p_before, p_after, delta, n_activations,
          species, sample_reasoning }
      ]   (only beliefs whose leaf_p moved this day; ids reference catalog)
    - observations: [
        { ticker, action, p_up, salience, confidence,
          reasoning_summary,
          causal_chain: [ {parent_id, contribution_log_odds, link_strength}, ... ],
          conflicting_signals: [ ... ] }
      ]   (the apex's curated watchlist for this day, with chain refs)
    - watchlist_ecology, watchlist_bare: as written by the runner
    - golden: per-ticker realized 1-day-ahead direction:
        [ { ticker, actual_direction, actual_return,
            magnitude_bucket, ideal_p_up,
            was_mentioned_in_news } ]

Designed so the animation viz can:
  - Load MANIFEST once → know every node id + canonical description
  - Scrub timeline → diff belief states day-to-day
  - Color-code observations: green if p_up direction == actual_direction,
    red if wrong, gray if flat — using the `golden` field on each day
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# .env loader so Neo4j creds populate
env = ROOT / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

PASS_DIR = ROOT / "data" / "firehose_eval" / "passes"
OUT_DIR = ROOT / "data" / "firehose_eval" / "timeline"


def build_manifest() -> dict:
    """Pull the full belief + outcome catalog from Neo4j once."""
    from trophic.beliefs.neo4j_store import Neo4jBeliefStore
    store = Neo4jBeliefStore()
    sb = store.all_state_beliefs()
    links = store.all_links()

    # Outcomes are everything starting with `outcome.` in link conclusions
    outcome_ids: set[str] = set()
    for l in links:
        if l.conclusion_belief_id.startswith("outcome."):
            outcome_ids.add(l.conclusion_belief_id)

    nodes = [
        {
            "id": b.id,
            "scope": b.scope,
            "statement_template": (b.statement_template or "")[:200],
            "decay_class": b.decay_class,
            "prior_p": round(b.prior_p, 4),
        }
        for b in sb
    ]
    outcomes = [
        {
            "id": oid,
            "ticker": oid.split(".")[1] if "." in oid else None,
        }
        for oid in sorted(outcome_ids)
    ]

    # Link summary
    by_scope_kind = {}
    for l in links:
        prem = "company" if "belief.company" in l.premise_belief_id else \
               "macro" if "belief.macro" in l.premise_belief_id else \
               "sector" if "belief.sector" in l.premise_belief_id else \
               "market" if "belief.market" in l.premise_belief_id else "other"
        conc = "outcome" if l.conclusion_belief_id.startswith("outcome.") else "belief"
        key = f"{prem}_to_{conc}"
        by_scope_kind[key] = by_scope_kind.get(key, 0) + 1

    return {
        "schema_version": "2.0",
        "description": (
            "Per-day belief-network timeline for the trophic firehose, "
            "regenerated 2026-05-10. MANIFEST has the full catalog of "
            "belief and outcome node ids; per-day files reference these "
            "ids and only carry deltas + observations + goldens for that day."
        ),
        "n_state_beliefs": len(sb),
        "n_outcome_beliefs": len(outcomes),
        "n_links": len(links),
        "links_by_kind": by_scope_kind,
        "belief_catalog": {
            "nodes": nodes,
            "outcomes": outcomes,
        },
    }


def build_day_record(rec: dict) -> dict:
    """Project a pass record into the timeline shape for one day."""
    from collections import defaultdict

    # Belief deltas: aggregate by target_belief_id
    first_before = {}
    last_after = {}
    n_acts = defaultdict(int)
    sample_reasoning = {}
    species_of = {}

    for a in rec.get("activations", []):
        bid = a.get("target_belief_id")
        if not bid:
            continue
        if bid not in first_before:
            first_before[bid] = a.get("leaf_p_before", 0.5)
            sample_reasoning[bid] = (a.get("reasoning") or "")[:160]
            species_of[bid] = a.get("species_id", "")
        last_after[bid] = a.get("leaf_p_after", first_before[bid])
        n_acts[bid] += 1

    belief_deltas = []
    for bid in first_before:
        p_before = first_before[bid]
        p_after = last_after[bid]
        if abs(p_after - p_before) < 1e-4:
            continue
        belief_deltas.append({
            "belief_id": bid,
            "p_before": round(p_before, 4),
            "p_after": round(p_after, 4),
            "delta": round(p_after - p_before, 4),
            "n_activations": n_acts[bid],
            "species": species_of[bid],
            "sample_reasoning": sample_reasoning[bid],
        })
    belief_deltas.sort(key=lambda x: -abs(x["delta"]))

    # Observations: keep the carnivore's full output but strip statement
    # (the chain references node IDs already in the manifest)
    observations = []
    for o in rec.get("observations", []):
        chain = []
        for c in (o.get("causal_chain") or []):
            chain.append({
                "parent_id": c.get("parent"),
                "p_now": round(c.get("p_now", 0.5), 4),
                "p_prior": round(c.get("p_prior", 0.5), 4),
                "delta": round(c.get("delta", 0), 4),
                "link_dir": c.get("link_dir"),
                "link_strength": round(c.get("link_strength", 0), 4),
                "contribution_log_odds": round(c.get("contribution_log_odds", 0), 4),
                "species": c.get("species"),
            })
        conflicting = []
        for c in (o.get("conflicting_signals") or []):
            conflicting.append({
                "parent_id": c.get("parent"),
                "p_now": round(c.get("p_now", 0.5), 4),
                "delta": round(c.get("delta", 0), 4),
                "contribution_log_odds": round(c.get("contribution_log_odds", 0), 4),
            })
        observations.append({
            "ticker": o.get("ticker"),
            "action": o.get("action"),
            "p_up": round(o.get("p_up", 0.5), 4),
            "salience": round(o.get("salience", 0), 4),
            "confidence": o.get("confidence"),
            "reasoning_summary": o.get("reasoning_summary", ""),
            "causal_chain": chain,
            "conflicting_signals": conflicting,
        })

    # Golden: ground-truth realized direction for *this trading day's*
    # 1-day-ahead outcomes. The pass record's `ground_truth` is the realization
    # used to grade this day's predictions, so we copy it directly.
    golden = []
    for g in rec.get("ground_truth", []):
        golden.append({
            "ticker": g.get("ticker"),
            "actual_direction": g.get("actual_direction"),
            "actual_return": g.get("actual_return"),
            "magnitude_bucket": g.get("magnitude_bucket"),
            "ideal_p_up": g.get("ideal_p_up"),
            "was_mentioned_in_news": g.get("was_mentioned_in_news"),
        })

    return {
        "date": rec["date"],
        "prev_trading_day": rec.get("prev_trading_day"),
        "n_news": rec.get("n_news", 0),
        "n_activations": len(rec.get("activations", [])),
        "n_observations": len(rec.get("observations", [])),
        "belief_deltas": belief_deltas,
        "observations": observations,
        "watchlist_ecology": rec.get("watchlist_ecology", []),
        "watchlist_bare": rec.get("watchlist_bare", []),
        "golden": golden,
        "elapsed": rec.get("elapsed", {}),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build manifest from Neo4j (the catalog)
    print("Building MANIFEST.json (catalog + ordered date index)...")
    manifest = build_manifest()
    print(f"  n_state_beliefs={manifest['n_state_beliefs']}, "
          f"n_outcomes={manifest['n_outcome_beliefs']}, "
          f"n_links={manifest['n_links']}")

    # 2. Run portfolio sim (frictionless + realistic)
    print("Running portfolio sim...")
    from scripts.portfolio_sim import run_full
    sim_realistic = run_full()  # 5 bps slippage + 37% tax (defaults)
    sim_frictionless = run_full(slippage_bps=0.0, tax_rate=0.0)
    print(f"  realistic:    eco=${sim_realistic['ecology']['final']:,.0f}  "
          f"bare=${sim_realistic['bare']['final']:,.0f}  "
          f"buyhold=${sim_realistic['buyhold']['final']:,.0f}")
    print(f"  frictionless: eco=${sim_frictionless['ecology']['final']:,.0f}  "
          f"bare=${sim_frictionless['bare']['final']:,.0f}  "
          f"buyhold=${sim_frictionless['buyhold']['final']:,.0f}")

    # Index per-day states by date for fast merge
    eco_by_date = {s["date"]: s for s in sim_realistic["ecology"]["daily_states"]}
    bare_by_date = {s["date"]: s for s in sim_realistic["bare"]["daily_states"]}
    bh_by_date = dict(sim_realistic["buyhold"]["equity_curve"])
    eco_fl_by_date = {s["date"]: s["equity"]
                      for s in sim_frictionless["ecology"]["daily_states"]}
    bare_fl_by_date = {s["date"]: s["equity"]
                       for s in sim_frictionless["bare"]["daily_states"]}
    bh_fl_by_date = dict(sim_frictionless["buyhold"]["equity_curve"])

    # 3. Per-day files
    paths = sorted(PASS_DIR.glob("*.json.gz"))
    print(f"Processing {len(paths)} pass artifacts...")
    dates = []
    total_belief_deltas = 0
    total_observations = 0
    total_golden = 0
    for path in paths:
        rec = json.load(gzip.open(path, "rt"))
        out = build_day_record(rec)

        d = out["date"]
        out["portfolio"] = {
            "ecology": eco_by_date.get(d),
            "bare": bare_by_date.get(d),
            "buyhold_equity": round(bh_by_date.get(d, 0.0), 2),
            "frictionless": {
                "ecology_equity": round(eco_fl_by_date.get(d, 0.0), 2),
                "bare_equity": round(bare_fl_by_date.get(d, 0.0), 2),
                "buyhold_equity": round(bh_fl_by_date.get(d, 0.0), 2),
            },
        }

        dates.append(d)
        total_belief_deltas += len(out["belief_deltas"])
        total_observations += len(out["observations"])
        total_golden += len(out["golden"])
        out_path = OUT_DIR / f"{d}.json"
        out_path.write_text(json.dumps(out, indent=1))

    # 4. Finalize manifest with date index + portfolio summary
    manifest["dates"] = dates
    manifest["portfolio_summary"] = {
        "starting_cash": sim_realistic["starting_cash"],
        "policy": sim_realistic["policy"],
        "frictions_realistic": sim_realistic["frictions"],
        "final_realistic": {
            "ecology": round(sim_realistic["ecology"]["final"], 2),
            "bare": round(sim_realistic["bare"]["final"], 2),
            "buyhold": round(sim_realistic["buyhold"]["final"], 2),
        },
        "final_frictionless": {
            "ecology": round(sim_frictionless["ecology"]["final"], 2),
            "bare": round(sim_frictionless["bare"]["final"], 2),
            "buyhold": round(sim_frictionless["buyhold"]["final"], 2),
        },
        "trade_counts": {
            "ecology_buys": sim_realistic["ecology"]["n_buys"],
            "ecology_sells": sim_realistic["ecology"]["n_sells"],
            "bare_buys": sim_realistic["bare"]["n_buys"],
            "bare_sells": sim_realistic["bare"]["n_sells"],
        },
    }
    (OUT_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    # Stats
    print(f"Wrote {len(dates)} day files to {OUT_DIR}")
    print(f"  total belief deltas: {total_belief_deltas}")
    print(f"  total observations:  {total_observations}")
    print(f"  total golden tickers: {total_golden}")
    import os as _os
    sizes = [(_os.path.getsize(p)) for p in OUT_DIR.glob("*.json")]
    print(f"  per-file size: min={min(sizes)}B, max={max(sizes)}B, "
          f"total={sum(sizes)/1024:.1f} KB")


if __name__ == "__main__":
    main()
