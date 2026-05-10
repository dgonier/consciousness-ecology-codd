"""Temporal decomposer — jobs 1+2.

Reads per-day Pass JSONs (data/firehose_eval/passes/), grades them against
ground truth, and emits:

  Job 1: teacher hints (data/training_corpus/teacher_hints/pending.jsonl)
         + apex training pairs (data/training_corpus/apex_pairs/pending.jsonl)
  Job 2: link strength_posterior Bayesian updates (back to Neo4j)
         + per-belief d* accumulator (data/decomposer/d_star.json)
         + per-species running fitness (in :Ecology:SpeciesFitness Neo4j node)

Daemon mode: poll passes/ directory, process new resolved Passes in date
order, persist state in data/decomposer/state.json so reruns are idempotent.

NOT in this version (jobs 3 + 4):
  - LoRA / DSPy training-job triggers
  - Death / reproduction
  These are stubbed entry points (TaskCreate'd as #134, #135).
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .pass_record import PASSES_DIR_NAME, iter_passes, load_pass


# ── Paths ──────────────────────────────────────────────────────────────

def state_paths(root: Path) -> dict:
    base = root / "data" / "decomposer"
    base.mkdir(parents=True, exist_ok=True)
    corpus = root / "data" / "training_corpus"
    (corpus / "teacher_hints").mkdir(parents=True, exist_ok=True)
    (corpus / "apex_pairs").mkdir(parents=True, exist_ok=True)
    return {
        "state":   base / "state.json",
        "d_star":  base / "d_star.json",
        "fitness": base / "fitness.json",
        "teacher_hints": corpus / "teacher_hints" / "pending.jsonl",
        "apex_pairs":    corpus / "apex_pairs"    / "pending.jsonl",
    }


# ── Math helpers ──────────────────────────────────────────────────────

def _log_odds(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1 - p))


def _bayes_update_strength(
    current: float, n_correct: int, n_total: int,
    prior_weight: float = 3.0,
) -> float:
    """Beta-Bayesian update of a strength_posterior in [0,1].

    Treats `current` as the mean of a Beta prior with effective sample size
    `prior_weight`. Adds the observed (n_correct, n_total) and returns the
    new posterior mean.

    prior_weight=3 means ~1 day of evidence (3 hints per template) shifts a
    link 50% toward the empirical posterior. Was 15 (over-smoothed; 65-day
    sweep moved links by avg <2% — online learning effectively disabled).
    """
    a_prior = max(0.5, current * prior_weight)
    b_prior = max(0.5, (1 - current) * prior_weight)
    a_post = a_prior + n_correct
    b_post = b_prior + max(0, n_total - n_correct)
    return a_post / (a_post + b_post)


# ── Per-Pass grading ──────────────────────────────────────────────────

@dataclass
class PassGrading:
    """Output of grading one Pass."""
    date: str
    teacher_hints: list[dict] = field(default_factory=list)
    apex_pair: Optional[dict] = None
    link_updates: dict[str, dict] = field(default_factory=dict)
    # link_id → {n_correct: int, n_total: int, mean_contrib_log_odds: float}
    species_observations: dict[str, dict] = field(default_factory=dict)
    # species_id → {n_acts, brier_sum, brier_n, hits, misses}
    belief_d_star: dict[str, float] = field(default_factory=dict)
    # belief_id → d* (signed, in log-odds space)


def _ticker_from_belief(belief_id: str) -> Optional[str]:
    """Pull a ticker out of a ticker-suffixed belief id like
    `belief.company.earnings_beat__ticker_AAPL` → 'AAPL'."""
    if "__ticker_" not in belief_id:
        return None
    return belief_id.rsplit("__ticker_", 1)[1].split("__")[0].upper()


def grade_pass(rec: dict) -> PassGrading:
    """Walk one Pass record and produce per-link, per-species, and per-belief
    grading signals."""
    date = rec["date"]
    g = PassGrading(date=date)

    # Build ticker → ground-truth lookup
    gt_by_ticker = {t["ticker"]: t for t in rec.get("ground_truth", [])}

    # Job 2a: per-belief d*. For ticker-suffixed leaves we know the ideal p_up
    # for that ticker, so the leaf's ideal "current_p" is a direct readout.
    # For market-scope leaves (cluster_correlation, dispersion), we don't
    # have a single ideal p; we use the magnitude of "should the cluster
    # belief move at all" derived from realized cluster behavior — placeholder
    # zero for now, refine in v2.
    for act in rec.get("activations", []):
        bid = act["target_belief_id"]
        leaf_after = act.get("leaf_p_after")
        if leaf_after is None:
            continue
        tk = _ticker_from_belief(bid)
        if tk is None:
            # Market/sector scope — no per-leaf hindsight target yet
            continue
        gt = gt_by_ticker.get(tk)
        if gt is None:
            continue
        ideal = gt["ideal_p_up"]
        # d* = ideal log-odds minus observed log-odds
        d_star = _log_odds(ideal) - _log_odds(leaf_after)
        # Accumulate across activations on the same belief (they shouldn't
        # double-fire, but average if they do)
        prev = g.belief_d_star.get(bid)
        if prev is None:
            g.belief_d_star[bid] = d_star
        else:
            g.belief_d_star[bid] = (prev + d_star) / 2

    # Job 1: teacher hints — one per (article, ticker, belief_template) tuple
    # showing the observed activation and the ideal direction/magnitude given
    # ground truth. For now we emit a simpler form: the (input, output)
    # pair that the event_classifier should have produced.
    for act in rec.get("activations", []):
        if act.get("species_id") != "event_classifier.v0.qwen-local":
            continue
        bid = act["target_belief_id"]
        tk = _ticker_from_belief(bid)
        if tk is None:
            continue
        gt = gt_by_ticker.get(tk)
        if gt is None:
            continue
        actual_dir = gt["actual_direction"]
        if actual_dir == "flat":
            continue  # don't train on flat days
        # The "ideal" for an activation on a bullish-template belief
        # (earnings_beat) firing 'increases' is: keep firing if the ticker
        # actually went up, suppress if it went down.
        # Build a signal: was the activation directionally correct?
        # earnings_beat increases + actual up = aligned
        # earnings_beat increases + actual down = misaligned (overconfident)
        template = bid.split("belief.company.")[-1].split("__ticker_")[0] if "belief.company." in bid else None
        if template is None:
            continue
        # Bullish templates: list from the ticker-instantiation seed.
        BULLISH = {"earnings_beat", "regulatory_clearance_received",
                   "major_product_launch_positive_reception",
                   "activist_investor_takes_stake",
                   "target_in_announced_ma",
                   "dividend_increase_announced",
                   "buyback_program_announced"}
        BEARISH = {"earnings_miss", "guidance_cut",
                   "regulatory_action_announced", "ceo_departure_unplanned",
                   "acting_as_acquirer_in_announced_ma",
                   "material_lawsuit_filed", "technical_support_broken",
                   "short_interest_high_and_rising"}
        is_bullish_template = template in BULLISH
        is_bearish_template = template in BEARISH
        if not (is_bullish_template or is_bearish_template):
            continue
        firing_increases = act.get("direction") == "increases"
        # Outcome direction determines if firing-increases was the right call
        if firing_increases:
            template_says_up = is_bullish_template  # bullish template firing increases = saying ticker is bullish
            outcome_up = (actual_dir == "up")
            label_correct = (template_says_up == outcome_up)
        else:
            template_says_up = is_bearish_template  # bearish firing decreases (rare) = also bullish
            outcome_up = (actual_dir == "up")
            label_correct = (template_says_up == outcome_up)
        g.teacher_hints.append({
            "date": date,
            "article_id": act.get("source_article_id"),
            "ticker": tk,
            "belief_template": template,
            "observed_direction": act.get("direction"),
            "observed_magnitude": act.get("magnitude"),
            "observed_confidence": act.get("confidence"),
            "actual_direction": actual_dir,
            "actual_magnitude_bucket": gt["magnitude_bucket"],
            "actual_return": gt["actual_return"],
            "ideal_p_up_for_ticker": gt["ideal_p_up"],
            "label_correct": label_correct,
        })

    # Job 2b: link grading. For each (link, ticker, day) tuple, we count
    # ONE observation. If a link fires multiple times in causal_chains for
    # the same ticker on the same day (e.g. multiple articles activated
    # the parent belief), it's still ONE update — the underlying ticker
    # movement is one event, not N independent samples.
    seen_link_keys: set[str] = set()
    for obs in rec.get("observations", []):
        tk = obs.get("ticker")
        if not tk or tk not in gt_by_ticker:
            continue
        gt = gt_by_ticker[tk]
        actual_up = (gt["actual_direction"] == "up")
        if gt["actual_direction"] == "flat":
            continue  # skip flat for link grading
        # Aggregate contributions per parent (across multiple obs for same
        # ticker — though typically just one obs per ticker)
        per_parent_contrib: dict[str, float] = {}
        for c in obs.get("causal_chain", []) or []:
            parent = c.get("parent")
            if not parent:
                continue
            per_parent_contrib[parent] = per_parent_contrib.get(parent, 0.0) + \
                c.get("contribution_log_odds", 0.0)

        for parent, contrib in per_parent_contrib.items():
            link_key = f"{parent}__{tk}"
            if link_key in seen_link_keys:
                continue  # already counted this (link, day) pair
            seen_link_keys.add(link_key)
            entry = g.link_updates.setdefault(link_key, {
                "n_total": 0, "n_correct": 0, "sum_contrib": 0.0, "n_contrib": 0,
            })
            entry["n_total"] = 1  # one observation per (link, day)
            entry["sum_contrib"] = contrib
            entry["n_contrib"] = 1
            correct = (contrib > 0) == actual_up
            entry["n_correct"] = 1 if correct else 0

    # Job 2c: per-species fitness. For each activation, compute Brier
    # contribution = (predicted_direction_p - actual)^2 where predicted is
    # 1.0 (bullish template increasing) or 0.0 (bearish template increasing),
    # weighted by confidence.
    for act in rec.get("activations", []):
        species = act.get("species_id", "(unknown)")
        bid = act["target_belief_id"]
        tk = _ticker_from_belief(bid)
        if tk is None or tk not in gt_by_ticker:
            continue
        gt = gt_by_ticker[tk]
        if gt["actual_direction"] == "flat":
            continue
        actual_up_bin = 1.0 if gt["actual_direction"] == "up" else 0.0
        # Map activation → predicted bullishness probability based on template
        template = bid.split("belief.company.")[-1].split("__ticker_")[0] if "belief.company." in bid else None
        BULLISH = {"earnings_beat", "regulatory_clearance_received",
                   "major_product_launch_positive_reception",
                   "activist_investor_takes_stake", "target_in_announced_ma",
                   "dividend_increase_announced", "buyback_program_announced"}
        BEARISH = {"earnings_miss", "guidance_cut",
                   "regulatory_action_announced", "ceo_departure_unplanned",
                   "acting_as_acquirer_in_announced_ma",
                   "material_lawsuit_filed", "technical_support_broken",
                   "short_interest_high_and_rising"}
        if template in BULLISH:
            pred_up = 1.0 if act.get("direction") == "increases" else 0.0
        elif template in BEARISH:
            pred_up = 0.0 if act.get("direction") == "increases" else 1.0
        else:
            continue  # macro/sector — handled separately
        brier = (pred_up - actual_up_bin) ** 2
        rec_e = g.species_observations.setdefault(species, {
            "n_acts": 0, "brier_sum": 0.0, "hits": 0, "misses": 0,
        })
        rec_e["n_acts"] += 1
        rec_e["brier_sum"] += brier
        if brier < 0.25:
            rec_e["hits"] += 1
        else:
            rec_e["misses"] += 1

    # Job 1b: apex training pair. The full Observations input + watchlist
    # output, alongside hindsight ideal watchlist (BUY for actual_up,
    # SELL for actual_down, WATCH for flat).
    if rec.get("observations"):
        ideal_watchlist = []
        for tk, gt in gt_by_ticker.items():
            if gt["actual_direction"] == "flat":
                continue
            ideal_watchlist.append({
                "ticker": tk,
                "action": "BUY" if gt["actual_direction"] == "up" else "SELL",
                "p_up": gt["ideal_p_up"],
                "actual_return": gt["actual_return"],
                "magnitude_bucket": gt["magnitude_bucket"],
            })
        ideal_watchlist.sort(key=lambda x: -abs(x["actual_return"]))
        g.apex_pair = {
            "date": date,
            "input_observations": rec["observations"],
            "input_focus_tickers": rec.get("focus_tickers", []),
            "observed_watchlist": rec.get("watchlist_ecology", []),
            "ideal_watchlist": ideal_watchlist,
        }

    return g


# ── Daemon / batch driver ──────────────────────────────────────────────

@dataclass
class Decomposer:
    root: Path
    write_neo4j: bool = True

    def __post_init__(self) -> None:
        self.paths = state_paths(self.root)
        self.passes_dir = self.root / "data" / "firehose_eval" / PASSES_DIR_NAME
        self._load_state()

    def _load_state(self) -> None:
        if self.paths["state"].exists():
            self.state = json.loads(self.paths["state"].read_text())
        else:
            self.state = {"processed_dates": [], "n_resolved_passes": 0}
        if self.paths["d_star"].exists():
            self.d_star = json.loads(self.paths["d_star"].read_text())
        else:
            self.d_star = {}  # belief_id -> {sum, n}
        if self.paths["fitness"].exists():
            self.fitness = json.loads(self.paths["fitness"].read_text())
        else:
            self.fitness = {}  # species_id -> {n_acts, brier_sum, hits, misses}

    def _save_state(self) -> None:
        self.paths["state"].write_text(json.dumps(self.state, indent=2))
        self.paths["d_star"].write_text(json.dumps(self.d_star, indent=2))
        self.paths["fitness"].write_text(json.dumps(self.fitness, indent=2))

    def _append_jsonl(self, path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with path.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    def process_one(self, rec: dict) -> dict:
        date = rec["date"]
        if date in self.state["processed_dates"]:
            return {"skipped": True, "reason": "already processed"}

        g = grade_pass(rec)

        # Append teacher hints + apex pairs to corpus
        self._append_jsonl(self.paths["teacher_hints"], g.teacher_hints)
        if g.apex_pair is not None:
            self._append_jsonl(self.paths["apex_pairs"], [g.apex_pair])

        # Update d* accumulator
        for bid, ds in g.belief_d_star.items():
            entry = self.d_star.setdefault(bid, {"sum": 0.0, "n": 0, "abs_sum": 0.0})
            entry["sum"] += ds
            entry["abs_sum"] += abs(ds)
            entry["n"] += 1

        # Update species fitness
        for sp, obs in g.species_observations.items():
            entry = self.fitness.setdefault(sp, {
                "n_acts": 0, "brier_sum": 0.0, "hits": 0, "misses": 0,
            })
            entry["n_acts"] += obs["n_acts"]
            entry["brier_sum"] += obs["brier_sum"]
            entry["hits"] += obs["hits"]
            entry["misses"] += obs["misses"]

        # Push link strength updates to Neo4j (this is the eval-pipeline-readable
        # thing that closes the learning loop)
        n_links_updated = 0
        if self.write_neo4j and g.link_updates:
            n_links_updated = self._update_links(g.link_updates)

        # Mark as processed
        self.state["processed_dates"].append(date)
        self.state["n_resolved_passes"] += 1
        self._save_state()

        return {
            "date": date,
            "teacher_hints": len(g.teacher_hints),
            "apex_pair": g.apex_pair is not None,
            "link_updates_inferred": len(g.link_updates),
            "neo4j_links_actually_updated": n_links_updated,
            "species_observed": len(g.species_observations),
            "beliefs_with_d_star": len(g.belief_d_star),
        }

    def _update_links(self, link_updates: dict[str, dict]) -> int:
        """For each parent_belief→ticker_outcome key, find the matching link
        in Neo4j and Bayesian-update its strength_posterior.

        link_updates key format: '{parent_belief_id}__{ticker}'
        """
        from .neo4j_store import Neo4jBeliefStore
        store = Neo4jBeliefStore()
        all_links = store.all_links()
        # Index by (premise, conclusion-ticker-suffix)
        by_pair: dict[tuple, list] = {}
        for l in all_links:
            conc = l.conclusion_belief_id
            # conclusion may be outcome.{TICKER}.next_day_direction or similar
            if conc.startswith("outcome.") and ".next_day_direction" in conc:
                tk = conc.split("outcome.")[1].split(".")[0]
                by_pair.setdefault((l.premise_belief_id, tk), []).append(l)
        n_updated = 0
        for key, upd in link_updates.items():
            parent, tk = key.rsplit("__", 1)
            cands = by_pair.get((parent, tk), [])
            if not cands:
                continue
            # Update each (usually one)
            for l in cands:
                old = l.strength_posterior
                new = _bayes_update_strength(
                    current=old,
                    n_correct=upd["n_correct"],
                    n_total=upd["n_total"],
                )
                # Persist via direct write — neo4j_store doesn't have a
                # specific update-strength helper, so use upsert_link.
                from dataclasses import replace
                new_l = replace(
                    l,
                    strength_posterior=new,
                    n_validations=l.n_validations + upd["n_total"],
                    n_correct=l.n_correct + upd["n_correct"],
                    updated_at=time.time(),
                )
                store.upsert_link(new_l)
                n_updated += 1
        return n_updated

    def run_batch(self, limit: Optional[int] = None) -> list[dict]:
        """Process all unprocessed passes in date order."""
        out = []
        n = 0
        for rec in iter_passes(self.passes_dir):
            if rec["date"] in self.state["processed_dates"]:
                continue
            res = self.process_one(rec)
            out.append(res)
            n += 1
            if limit is not None and n >= limit:
                break
        return out

    def report(self) -> str:
        lines = [f"Decomposer state: {self.state['n_resolved_passes']} passes processed"]
        lines.append(f"  d* tracked beliefs: {len(self.d_star)}")
        if self.d_star:
            lines.append(f"  top-5 |d*| beliefs:")
            ranked = sorted(self.d_star.items(),
                            key=lambda kv: -kv[1]["abs_sum"] / max(1, kv[1]["n"]))
            for bid, e in ranked[:5]:
                avg_abs = e["abs_sum"] / max(1, e["n"])
                avg_signed = e["sum"] / max(1, e["n"])
                lines.append(f"    {bid:60s} |d*|_mean={avg_abs:.3f}  "
                             f"signed={avg_signed:+.3f}  n={e['n']}")
        lines.append(f"  species fitness:")
        for sp, e in self.fitness.items():
            n = e["n_acts"] or 1
            brier = e["brier_sum"] / n
            hit_rate = e["hits"] / max(1, e["hits"] + e["misses"])
            lines.append(f"    {sp:35s}  n={e['n_acts']:>4d}  "
                         f"brier={brier:.3f}  hit_rate={hit_rate:.3f}  "
                         f"hits={e['hits']} misses={e['misses']}")
        return "\n".join(lines)
