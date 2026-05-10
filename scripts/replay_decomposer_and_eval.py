"""Replay decomposer with new prior_weight=3 and re-eval at matched commit-rate.

Pure-Python (no Neo4j, no LLM). Reads existing pass artifacts and:
  1. Recomputes link strengths with prior_weight=3 (was 15).
  2. Re-projects each pass's outcome.p_up using the updated link strengths
     (replay the carnivore aggregation with new weights).
  3. Reports MCC for ecology, bare, and matched commit-rate ecology vs bare.

Why pure-Python: we don't want to mutate the live Neo4j store; this is an
ablation. If the result is good, we then re-run the live sweep.
"""
from __future__ import annotations

import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PASS_DIR = ROOT / "data" / "firehose_eval" / "passes"


def _logodds(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _bayes_update(current: float, n_correct: int, n_total: int,
                  prior_weight: float) -> float:
    a = max(0.5, current * prior_weight)
    b = max(0.5, (1 - current) * prior_weight)
    a += n_correct
    b += max(0, n_total - n_correct)
    return a / (a + b)


def _mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den > 0 else 0.0


def score_pairs(pairs: list[tuple[bool, bool]]) -> dict:
    tp = tn = fp = fn = 0
    for pred_up, actual_up in pairs:
        if pred_up and actual_up:
            tp += 1
        elif pred_up and not actual_up:
            fp += 1
        elif not pred_up and not actual_up:
            tn += 1
        else:
            fn += 1
    n = tp + tn + fp + fn
    return {
        "n": n,
        "acc": (tp + tn) / n if n else 0.0,
        "mcc": _mcc(tp, tn, fp, fn),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def main(prior_weight: float = 3.0) -> None:
    paths = sorted(PASS_DIR.glob("*.json.gz"))
    print(f"Loading {len(paths)} pass artifacts from {PASS_DIR}")

    # Bootstrap link strengths from priors observed in earliest pass's chains.
    # We don't have a clean snapshot; reconstruct from the seed values used
    # in instantiate_ticker_universe.py.
    SEED_PRIOR = {
        # bullish
        "earnings_beat": 0.55, "regulatory_clearance_received": 0.50,
        "major_product_launch_positive_reception": 0.50,
        "activist_investor_takes_stake": 0.45,
        "target_in_announced_ma": 0.75,
        "dividend_increase_announced": 0.35,
        "buyback_program_announced": 0.40,
        # bearish
        "earnings_miss": 0.75, "guidance_cut": 0.75,
        "regulatory_action_announced": 0.70,
        "material_lawsuit_filed": 0.65,
        "ceo_departure_unplanned": 0.65,
        "acting_as_acquirer_in_announced_ma": 0.40,
        "technical_support_broken": 0.55,
        "short_interest_high_and_rising": 0.45,
    }
    DIRECTIONS = {
        "earnings_beat": +1, "regulatory_clearance_received": +1,
        "major_product_launch_positive_reception": +1,
        "activist_investor_takes_stake": +1,
        "target_in_announced_ma": +1,
        "dividend_increase_announced": +1,
        "buyback_program_announced": +1,
        "earnings_miss": -1, "guidance_cut": -1,
        "regulatory_action_announced": -1,
        "material_lawsuit_filed": -1,
        "ceo_departure_unplanned": -1,
        "acting_as_acquirer_in_announced_ma": -1,
        "technical_support_broken": -1,
        "short_interest_high_and_rising": -1,
    }

    # link_strength: keyed by (template, ticker) → current strength
    link_strength: dict[tuple[str, str], float] = {}

    def get_link(template: str, ticker: str) -> float:
        key = (template, ticker)
        if key not in link_strength:
            link_strength[key] = SEED_PRIOR.get(template, 0.5)
        return link_strength[key]

    def update_link(template: str, ticker: str, n_correct: int,
                    n_total: int) -> None:
        key = (template, ticker)
        cur = get_link(template, ticker)
        link_strength[key] = _bayes_update(cur, n_correct, n_total,
                                           prior_weight=prior_weight)

    # Re-project: for each pass, replay the carnivore using current
    # link_strength values, then update strengths from this pass's outcome.
    ecology_pairs = []
    bare_pairs = []
    ecology_pairs_strict = []  # excluding flat
    bare_pairs_strict = []

    # Per-day commit-rate matched scoring
    per_day_ecology = []  # list of (sorted observations by salience desc, gt)
    per_day_bare = []

    n_links_moved = 0

    for path in paths:
        d = json.load(gzip.open(path, "rt"))
        gt = {g["ticker"]: g for g in d.get("ground_truth", [])}

        # Re-project ecology p_up using updated link strengths
        ecology_obs = []
        for obs in d.get("observations", []):
            tk = obs.get("ticker")
            if not tk:
                continue
            # Recompute outcome log-odds from causal chain with new strengths
            lo = 0.0
            for c in obs.get("causal_chain", []) or []:
                parent = c.get("parent", "")
                if "belief.company." not in parent:
                    continue
                template = parent.replace("belief.company.", "").split("__")[0]
                p_now = c.get("p_now", 0.5)
                p_prior = c.get("p_prior", 0.5)
                lo_delta = _logodds(p_now) - _logodds(p_prior)
                strength = get_link(template, tk)
                sign = DIRECTIONS.get(template, 0)
                lo += strength * sign * lo_delta
            for c in obs.get("conflicting_signals", []) or []:
                parent = c.get("parent", "")
                if "belief.company." not in parent:
                    continue
                template = parent.replace("belief.company.", "").split("__")[0]
                p_now = c.get("p_now", 0.5)
                p_prior = c.get("p_prior", 0.5)
                lo_delta = _logodds(p_now) - _logodds(p_prior)
                strength = get_link(template, tk)
                sign = DIRECTIONS.get(template, 0)
                lo += strength * sign * lo_delta

            new_p_up = _sigmoid(lo)
            salience = abs(new_p_up - 0.5)
            ecology_obs.append({
                "ticker": tk,
                "p_up": new_p_up,
                "salience": salience,
            })

        per_day_ecology.append((ecology_obs, gt))

        # Bare obs: just take the watchlist as-is
        bare_obs = [{"ticker": w["ticker"], "p_up": w["p_up"],
                     "salience": abs(w["p_up"] - 0.5),
                     "action": w["action"]}
                    for w in d.get("watchlist_bare", [])]
        per_day_bare.append((bare_obs, gt))

        # Update link strengths from this pass's actual outcomes (online).
        seen_link_keys: set[str] = set()
        for obs in d.get("observations", []):
            tk = obs.get("ticker")
            if not tk or tk not in gt:
                continue
            actual_dir = gt[tk]["actual_direction"]
            if actual_dir == "flat":
                continue
            actual_up = (actual_dir == "up")
            for c in obs.get("causal_chain", []) or []:
                parent = c.get("parent", "")
                if "belief.company." not in parent:
                    continue
                template = parent.replace("belief.company.", "").split("__")[0]
                key = (template, tk)
                if key in seen_link_keys:
                    continue
                seen_link_keys.add(key)
                contrib = c.get("contribution_log_odds", 0.0)
                n_correct = 1 if (contrib > 0) == actual_up else 0
                old = get_link(template, tk)
                update_link(template, tk, n_correct, 1)
                if abs(get_link(template, tk) - old) > 0.001:
                    n_links_moved += 1

    # ── Score full-commit (any non-flat, non-WATCH ecology obs) ──
    for obs_list, gt in per_day_ecology:
        for o in obs_list:
            tk = o["ticker"]
            if tk not in gt:
                continue
            ad = gt[tk]["actual_direction"]
            pred_up = o["p_up"] >= 0.5
            actual_up = ad == "up"
            ecology_pairs.append((pred_up, actual_up))
            if ad != "flat":
                ecology_pairs_strict.append((pred_up, actual_up))

    for obs_list, gt in per_day_bare:
        for o in obs_list:
            if o.get("action") == "WATCH":
                continue
            tk = o["ticker"]
            if tk not in gt:
                continue
            ad = gt[tk]["actual_direction"]
            pred_up = o["p_up"] >= 0.5
            actual_up = ad == "up"
            bare_pairs.append((pred_up, actual_up))
            if ad != "flat":
                bare_pairs_strict.append((pred_up, actual_up))

    # ── Score matched commit-rate: top-K per day where K = bare's avg commit-count ──
    bare_commits_per_day = [
        sum(1 for o in obs if o.get("action") != "WATCH")
        for obs, _ in per_day_bare
    ]
    K = max(1, round(sum(bare_commits_per_day) / max(1, len(bare_commits_per_day))))
    print(f"\nMatched commit-rate K = {K} per day "
          f"(bare avg committed picks)")

    eco_topK = []
    bare_topK = []
    for obs_list, gt in per_day_ecology:
        ranked = sorted(obs_list, key=lambda o: -o["salience"])[:K]
        for o in ranked:
            tk = o["ticker"]
            if tk not in gt or gt[tk]["actual_direction"] == "flat":
                continue
            pred_up = o["p_up"] >= 0.5
            actual_up = gt[tk]["actual_direction"] == "up"
            eco_topK.append((pred_up, actual_up))
    for obs_list, gt in per_day_bare:
        ranked = sorted([o for o in obs_list if o.get("action") != "WATCH"],
                        key=lambda o: -o["salience"])[:K]
        for o in ranked:
            tk = o["ticker"]
            if tk not in gt or gt[tk]["actual_direction"] == "flat":
                continue
            pred_up = o["p_up"] >= 0.5
            actual_up = gt[tk]["actual_direction"] == "up"
            bare_topK.append((pred_up, actual_up))

    # ── Print results ──
    print("\n" + "═" * 70)
    print(f"REPLAY RESULTS (decomposer prior_weight={prior_weight})")
    print("═" * 70)
    print(f"\nTotal link-strength updates that moved a link: {n_links_moved}")
    print(f"Distinct (template, ticker) links touched: {len(link_strength)}")

    print(f"\n── ECOLOGY (re-projected, all picks) ──")
    print(f"  full:    {score_pairs(ecology_pairs)}")
    print(f"  no-flat: {score_pairs(ecology_pairs_strict)}")

    print(f"\n── BARE (unchanged) ──")
    print(f"  full:    {score_pairs(bare_pairs)}")
    print(f"  no-flat: {score_pairs(bare_pairs_strict)}")

    print(f"\n── MATCHED COMMIT-RATE (top-{K} per day, no-flat) ──")
    print(f"  ecology: {score_pairs(eco_topK)}")
    print(f"  bare:    {score_pairs(bare_topK)}")

    # Show the strongest movers
    print(f"\n── Top 10 link strength shifts (template, ticker, prior → post) ──")
    moved = []
    for (tmpl, tk), strength in link_strength.items():
        prior = SEED_PRIOR.get(tmpl, 0.5)
        if abs(strength - prior) > 0.001:
            moved.append((tmpl, tk, prior, strength, strength - prior))
    moved.sort(key=lambda x: -abs(x[4]))
    for tmpl, tk, prior, post, delta in moved[:10]:
        print(f"  {tmpl:45s} {tk:6s}  {prior:.3f} → {post:.3f}  ({delta:+.3f})")


if __name__ == "__main__":
    pw = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    main(prior_weight=pw)
