"""Convert a v3 9-way ablation run into the per-day JSON + MANIFEST shape the
Belief Timeline dashboard expects.

Source: trophic/data/firehose_eval/runs/<run>/{run_meta.json, eval.jsonl}
Target: trophic/dashboard/{<date>.json, MANIFEST.json, run_meta.json}

The dashboard schema is the v2.0 single-pipeline format. The new run is a 9-way
sweep (3 apex models x 3 pipelines: BARE/ECOLOGY/ORACLE). We pick ECOLOGY-QWEN
as the primary pipeline (matches dashboard's `portfolio.ecology` semantics) and
BARE-QWEN as the bare baseline. BUYHOLD is computed by equal-weighting the 20
target tickers from day 1 close prices (no buyhold field exists in v3 source).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RUN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/home/dgonier/ecology_experiment/trophic/data/firehose_eval/runs/run_2026-05-10_pm_v3_3_9way"
)
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    "/home/dgonier/ecology_experiment/trophic/dashboard"
)

PARENT_RE = re.compile(r"^belief\.company\.(.+)__ticker_([A-Z0-9.]+)$")

TICKERS = [
    "AAPL", "AMZN", "AVGO", "CSCO", "GOOG", "MSFT",
    "ABBV", "JNJ", "MRK", "UNH",
    "CVX",
    "JPM", "MA", "V",
    "HD", "KO", "MCD", "PEP", "PG", "WMT",
]

TEMPLATES = [
    "earnings_beat", "earnings_miss", "guidance_raised", "guidance_cut",
    "major_product_launch_positive_reception", "major_product_launch_negative_reception",
    "analyst_upgrade", "analyst_downgrade",
    "regulatory_clearance_received", "regulatory_action_announced",
    "technical_resistance_broken", "technical_support_broken",
    "buyback_program_announced", "dividend_increase_announced",
    "ceo_departure_unplanned", "activist_investor_takes_stake",
    "short_interest_high_and_rising", "material_lawsuit_filed",
    "acting_as_acquirer_in_announced_ma", "target_in_announced_ma",
    "partnership_announced", "supply_chain_disruption",
]


def load_days(run_dir: Path) -> list[dict]:
    with (run_dir / "eval.jsonl").open() as f:
        return [json.loads(line) for line in f if line.strip()]


def derive_belief_deltas(observations: list[dict]) -> list[dict]:
    """Each causal_chain entry on each observation describes one parent belief
    that was active. Collapse to one delta per (template, ticker) keeping the
    largest |delta|. Surface the parent observation's reasoning_summary as the
    sample_reasoning so the heatmap tooltip + leaf-activations panel get
    something to render."""
    out: dict[tuple[str, str], dict] = {}
    for obs in observations or []:
        rs = obs.get("reasoning_summary") or ""
        for c in obs.get("causal_chain") or []:
            m = PARENT_RE.match(c.get("parent", ""))
            if not m:
                continue
            tpl, tkr = m.group(1), m.group(2)
            key = (tpl, tkr)
            delta = c.get("delta", 0.0) or 0.0
            prev = out.get(key)
            if prev is None or abs(delta) > abs(prev["delta"]):
                out[key] = {
                    "belief_id": c["parent"],
                    "p_after": c.get("p_now", 0.5),
                    "p_before": c.get("p_prior", 0.5),
                    "delta": delta,
                    "n_activations": 1,
                    "sample_reasoning": rs,
                    "species": c.get("species", ""),
                }
    return list(out.values())


def derive_golden(targets: list[dict]) -> list[dict]:
    """v2 dashboard expects each golden entry to have an `ideal_p_up`. The v3
    source has actual_direction + actual_return + magnitude_bucket but no
    explicit ideal_p_up — derive a calibrated probability from the realized
    return so the ideal-tick chevron has somewhere to land."""
    out = []
    for t in targets or []:
        ar = t.get("actual_return", 0.0) or 0.0
        # Map realized return to a 0..1 ideal probability via a soft sigmoid
        # bucketing: ~0.5 for flat (|r|<0.5%), saturating to 0.05/0.95 by ~3%.
        x = max(-1.0, min(1.0, ar / 0.03))
        ideal = 0.5 + 0.45 * x
        out.append({
            "ticker": t["ticker"],
            "actual_direction": t.get("actual_direction", "flat"),
            "actual_return": ar,
            "magnitude_bucket": t.get("magnitude_bucket", "flat"),
            "ideal_p_up": round(ideal, 4),
            "was_mentioned_in_news": t.get("was_mentioned_in_news", False),
        })
    return out


def derive_portfolio(day: dict) -> dict:
    """Map ECOLOGY-QWEN and BARE-QWEN snapshots to the v2 portfolio shape."""

    def remap_snapshot(snap: dict, orders: list[dict]) -> dict:
        if not snap:
            return None
        # Build positions list with the field names the holdings bar reads.
        positions = []
        for p in snap.get("open_positions") or []:
            mv = p.get("mv", 0.0)
            cost = p.get("cost_basis", mv)
            upnl = p.get("unrealized_pnl", mv - cost)
            upnl_pct = (upnl / cost) if cost else 0.0
            positions.append({
                "ticker": p["ticker"],
                "shares": p.get("shares", 0.0),
                "cost_basis": cost,
                "mv": mv,
                "unrealized_pnl": upnl,
                "unrealized_pnl_pct": upnl_pct,
                "weight_pct": p.get("weight_pct", 0.0),
                "days_held": p.get("days_held", 0),
            })
        equity = snap.get("equity", 0.0)
        cash = snap.get("cash", 0.0)
        invested = snap.get("invested_pct", 0.0) / 100.0
        return {
            "equity": equity,
            "cash": cash,
            "invested_frac": invested,
            "n_open_positions": len(positions),
            "open_positions": positions,
            "realized_gains_cum": snap.get("realized_gains_cum", 0.0),
            "tax_owed_accrued": snap.get("tax_owed_accrued", 0.0),
            "orders_today": [
                {
                    "ticker": o["ticker"],
                    "side": o["side"],
                    "dollars_intent": o.get("dollars_intent", 0.0),
                    "size_pct": o.get("size_pct", 0.0),
                    "reasoning": o.get("reasoning", ""),
                }
                for o in (orders or [])
            ],
        }

    eco_snap = day["ecology"].get("pm_snapshot_post")
    eco_orders = day["ecology"].get("pm_orders_validated") or []
    bare_snap = day["bare"].get("pm_snapshot_post")
    bare_orders = day["bare"].get("pm_orders_validated") or []
    return {
        "ecology": remap_snapshot(eco_snap, eco_orders),
        "bare": remap_snapshot(bare_snap, bare_orders),
        # frictionless: surface pretax pipelines if present (none here, leave null)
        "frictionless": None,
    }


def buyhold_series(days: list[dict]) -> list[float]:
    """Equal-weighted buyhold of all 20 tickers from day 1 prev_close."""
    seed = 100_000.0
    # Day 0: invest seed/N in each ticker at prev_close on day 0.
    day0 = days[0]
    px0 = {t["ticker"]: t["prev_close"] for t in day0["targets"]}
    n = sum(1 for t in day0["targets"] if t["prev_close"])
    per_tkr_dollars = seed / n
    shares = {tkr: per_tkr_dollars / px for tkr, px in px0.items() if px}
    out = []
    for d in days:
        eq = 0.0
        for t in d["targets"]:
            tkr = t["ticker"]
            if tkr in shares:
                eq += shares[tkr] * t["actual_close"]
        out.append(round(eq, 2))
    return out


def build_manifest() -> dict:
    """Synthesize a manifest in the shape ContextStrip expects:
    `belief_catalog.nodes[]` with {id, scope, statement_template, decay_class,
    prior_p}. The v3 source has no macro/sector beliefs to surface, so we emit
    only the canonical company grid; the macro row will be empty (dashboard
    handles this — `groups.macro.length` = 0)."""
    nodes = []
    for tpl in TEMPLATES:
        for tkr in TICKERS:
            nodes.append({
                "id": f"belief.company.{tpl}__ticker_{tkr}",
                "scope": "company",
                "statement_template": f"[{tkr}] (stub for belief.company.{tpl})",
                "decay_class": "event",
                "prior_p": 0.5,
            })
    outcomes = [
        {
            "id": f"outcome.company.next_day_direction__ticker_{tkr}",
            "scope": "company",
            "ticker": tkr,
        }
        for tkr in TICKERS
    ]
    return {
        "schema_version": "2.0",
        "description": "Synthesized from v3 9-way ablation run; macro/sector beliefs not present in source.",
        "n_state_beliefs": len(nodes),
        "n_outcome_beliefs": len(outcomes),
        "n_links": 0,
        "links_by_kind": {},
        "belief_catalog": {
            "nodes": nodes,
            "outcomes": outcomes,
        },
        "dates": [],
        "portfolio_summary": {},
    }


def remap_run_meta(src_meta: dict) -> dict:
    fe_net = src_meta.get("final_equities_net", {})
    eco = fe_net.get("ECOLOGY-QWEN")
    bare = fe_net.get("BARE-QWEN")
    parts = []
    if eco is not None: parts.append(f"ECOLOGY=${eco:,}")
    if bare is not None: parts.append(f"BARE=${bare:,}")
    final_str = " · ".join(parts)
    return {
        "label": src_meta.get("label", ""),
        "git_sha": src_meta.get("git_sha", ""),
        "feature_flags": src_meta.get("feature_flags", ""),
        "final_equities": final_str,
        "note": src_meta.get("note", ""),
    }


def main() -> None:
    src_meta = json.loads((RUN_DIR / "run_meta.json").read_text())
    days = load_days(RUN_DIR)
    print(f"loaded {len(days)} days from {RUN_DIR}")

    bh = buyhold_series(days)

    # Clean stale per-day JSONs in OUT_DIR before writing the new ones so old
    # dates don't linger.
    for old in OUT_DIR.glob("2026-*.json"):
        old.unlink()

    for d, bh_eq in zip(days, bh):
        date = d["date"]
        eco = d["ecology"]
        observations = eco.get("observations") or []
        deltas = derive_belief_deltas(observations)
        golden = derive_golden(d.get("targets") or [])
        portfolio = derive_portfolio(d)
        portfolio["buyhold_equity"] = bh_eq

        # observations remap: dashboard reads {ticker, action, p_up, salience,
        # confidence, causal_chain, conflicting_signals, reasoning_summary}.
        # The causal_chain field name needs to be `parent_id` (not `parent`)
        # because viz-grid's link snapshot loop reads `c.parent_id`. Same for
        # contribution_log_odds (already-named correctly).
        obs_out = []
        for o in observations:
            cc = []
            for c in o.get("causal_chain") or []:
                cc.append({
                    "parent_id": c.get("parent"),
                    "statement": c.get("statement", ""),
                    "p_now": c.get("p_now", 0.5),
                    "p_prior": c.get("p_prior", 0.5),
                    "delta": c.get("delta", 0.0),
                    "link_dir": c.get("link_dir", "positive"),
                    "link_strength": c.get("link_strength", 0.0),
                    "contribution_log_odds": c.get("contribution_log_odds", 0.0),
                    "species": c.get("species", ""),
                })
            obs_out.append({
                "ticker": o["ticker"],
                "action": o.get("action", "WATCH"),
                "p_up": o.get("p_up", 0.5),
                "salience": o.get("salience", 0.0),
                "confidence": o.get("confidence", "low"),
                "horizon_min": o.get("horizon_min", 1440),
                "reasoning_summary": o.get("reasoning_summary", ""),
                "causal_chain": cc,
                "conflicting_signals": o.get("conflicting_signals") or [],
                "triggering_events": o.get("triggering_events") or [],
            })

        out = {
            "date": date,
            "n_news": d.get("n_news", 0),
            "n_activations": len(deltas),
            "n_observations": len(obs_out),
            "belief_deltas": deltas,
            "observations": obs_out,
            "golden": golden,
            "watchlist_ecology": eco.get("watchlist") or [],
            "watchlist_bare": (d["bare"].get("watchlist") or []),
            "portfolio": portfolio,
            "pm_rationale_ecology": eco.get("pm_rationale", ""),
            "pm_rationale_bare": d["bare"].get("pm_rationale", ""),
        }
        (OUT_DIR / f"{date}.json").write_text(json.dumps(out, separators=(",", ":")))

    (OUT_DIR / "MANIFEST.json").write_text(json.dumps(build_manifest(), separators=(",", ":")))
    (OUT_DIR / "run_meta.json").write_text(json.dumps(remap_run_meta(src_meta), indent=2))

    print(f"wrote {len(days)} day files + MANIFEST.json + run_meta.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
