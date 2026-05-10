"""Pass persistence: JSONL trail (for the decomposer) + Neo4j aggregates.

Each trading-day eval produces one Pass record containing the full causal
chain: news → activations → propagation contributions → observations →
watchlist → ground truth.

Two storage destinations:
  1. JSONL: data/firehose_eval/passes/{date}.json  — full chain. The decomposer
     reads these sequentially. Fast, gzipped, decomposer-friendly.
  2. Neo4j: only the running aggregates the *eval pipeline* needs to read
     back next day — link strength_posterior updates, species fitness,
     accumulated d* per belief. Written by the decomposer, NOT by the
     pass-writer.

This module handles writing the per-day JSONL and the lightweight Neo4j
markers that say "this Pass exists, here's its summary." The decomposer
reads JSONL for the heavy lifting.

Schema reference (JSONL):
{
  "date":              "2026-02-03",
  "prev_trading_day":  "2026-01-30",
  "runner_version":    "v0",
  "model_id":          "Qwen/Qwen3.5-4B",
  "focus_tickers":     [...],
  "n_news":            26,

  "activations": [
    {
      "id":                 stable hash of (date, target_belief_id, species_id, magnitude, ...),
      "target_belief_id":   "belief.company.earnings_beat__ticker_AAPL",
      "direction":          "increases",
      "magnitude":          "strong",
      "decay_class":        "slow",
      "confidence":         "very_high",
      "reasoning":          "...",
      "species_id":         "event_classifier.v0.qwen-local",
      "source_article_id":  "...",       // null if not an article-driven act
      "magnitude_logit":    null,
      "applied_log_odds_shift": +1.064,  // signed shift this activation injected
      "leaf_p_before":      0.50,
      "leaf_p_after":       0.821,
    },
    ...
  ],

  "observations": [
    // The carnivore Observation.to_dict() output for every observation
    // emitted to apex (top-K). Includes causal_chain + conflicting_signals.
    {...},
    ...
  ],

  "watchlist_ecology": [
    {"ticker", "action", "p_up", "reason", "rank"},
    ...
  ],
  "watchlist_bare": [
    {"ticker", "action", "p_up", "reason", "rank"},
    ...
  ],

  "ground_truth": [
    {
      "ticker":           "AAPL",
      "actual_direction": "up" | "down" | "flat",
      "actual_return":    0.0123,
      "magnitude_bucket": "large" | "medium" | "flat",
      "ideal_p_up":       0.85,   // soft label per magnitude bucket
    },
    ...
  ],

  "elapsed": {"ecology_s": 30.78, "bare_s": 29.76},
}
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


PASSES_DIR_NAME = "passes"


# ── Soft-label per magnitude bucket (decision: B) ─────────────────────

def ideal_p_up(actual_direction: str, magnitude_bucket: str) -> float:
    """Hindsight target p_up. Soft labels graded by magnitude.

    flat / no-move days are always 0.5 — the network shouldn't have committed
    to a direction.
    """
    if actual_direction == "flat" or magnitude_bucket == "flat":
        return 0.5
    if actual_direction == "up":
        if magnitude_bucket == "large":
            return 0.85
        return 0.65   # medium
    if actual_direction == "down":
        if magnitude_bucket == "large":
            return 0.15
        return 0.35   # medium
    return 0.5


def stable_activation_id(
    date: str,
    target_belief_id: str,
    species_id: str,
    magnitude: str,
    direction: str,
) -> str:
    """Deterministic id for an activation so the decomposer can dedup
    across reruns of the same date."""
    raw = f"{date}|{target_belief_id}|{species_id}|{magnitude}|{direction}"
    return "act." + hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


# ── Pass writer ───────────────────────────────────────────────────────

@dataclass
class PassPersister:
    """Builds + writes per-day Pass records.

    Usage:
      pp = PassPersister(out_dir, model_id="Qwen/Qwen3.5-4B")
      pp.add_activation(activation_dict)
      pp.add_observations(carnivore_obs_dicts)
      pp.add_watchlist("ecology", entries)
      pp.set_ground_truth(targets_list)
      pp.write(date="2026-02-03", prev="2026-01-30", focus=[], n_news=26,
               elapsed_ecology_s=30.78, elapsed_bare_s=29.76)
    """
    out_dir: Path
    model_id: str
    runner_version: str = "v0"

    def __post_init__(self) -> None:
        (self.out_dir / PASSES_DIR_NAME).mkdir(parents=True, exist_ok=True)
        self._activations: list[dict] = []
        self._observations: list[dict] = []
        self._watchlists: dict[str, list[dict]] = {}
        self._ground_truth: list[dict] = []

    def add_activation(
        self,
        target_belief_id: str,
        direction: str,
        magnitude: str,
        decay_class: str,
        confidence: str,
        species_id: str,
        reasoning: str = "",
        source_article_id: Optional[str] = None,
        magnitude_logit: Optional[float] = None,
        applied_log_odds_shift: Optional[float] = None,
        leaf_p_before: Optional[float] = None,
        leaf_p_after: Optional[float] = None,
    ) -> None:
        # Date stamp will be filled in at write() time
        self._activations.append({
            "_target_belief_id": target_belief_id,
            "direction": direction,
            "magnitude": magnitude,
            "decay_class": decay_class,
            "confidence": confidence,
            "species_id": species_id,
            "reasoning": reasoning[:300],
            "source_article_id": source_article_id,
            "magnitude_logit": magnitude_logit,
            "applied_log_odds_shift": applied_log_odds_shift,
            "leaf_p_before": leaf_p_before,
            "leaf_p_after": leaf_p_after,
        })

    def add_observations(self, observations: list[dict]) -> None:
        self._observations.extend(observations)

    def add_watchlist(self, path: str, entries: list[dict]) -> None:
        # Annotate with rank
        ranked = []
        for i, e in enumerate(entries):
            ranked.append({**e, "rank": i + 1})
        self._watchlists[path] = ranked

    def set_ground_truth(self, targets: list[dict]) -> None:
        """Convert dataset targets[] (with actual_direction + magnitude_bucket)
        into ground-truth records carrying the ideal_p_up per soft-label
        scheme."""
        gt = []
        for t in targets:
            gt.append({
                "ticker": t["ticker"],
                "actual_direction": t["actual_direction"],
                "actual_return": t["actual_return"],
                "magnitude_bucket": t["magnitude_bucket"],
                "ideal_p_up": ideal_p_up(t["actual_direction"], t["magnitude_bucket"]),
                "was_mentioned_in_news": t.get("was_mentioned_in_news", False),
            })
        self._ground_truth = gt

    def write(
        self,
        date: str,
        prev_trading_day: str,
        focus_tickers: list[str],
        n_news: int,
        elapsed_ecology_s: Optional[float] = None,
        elapsed_bare_s: Optional[float] = None,
    ) -> Path:
        # Stamp activations with the date and stable id
        for a in self._activations:
            tbid = a.pop("_target_belief_id")
            a["target_belief_id"] = tbid
            a["id"] = stable_activation_id(
                date, tbid, a["species_id"], a["magnitude"], a["direction"],
            )

        record = {
            "date": date,
            "prev_trading_day": prev_trading_day,
            "runner_version": self.runner_version,
            "model_id": self.model_id,
            "focus_tickers": list(focus_tickers),
            "n_news": n_news,
            "activations": self._activations,
            "observations": self._observations,
            "watchlist_ecology": self._watchlists.get("ecology", []),
            "watchlist_bare":    self._watchlists.get("bare", []),
            "ground_truth":      self._ground_truth,
            "elapsed": {
                "ecology_s": elapsed_ecology_s,
                "bare_s":    elapsed_bare_s,
            },
        }

        path = self.out_dir / PASSES_DIR_NAME / f"{date}.json.gz"
        with gzip.open(path, "wt") as fh:
            json.dump(record, fh)

        # Reset state so the persister can be reused for the next day
        self._activations.clear()
        self._observations.clear()
        self._watchlists.clear()
        self._ground_truth.clear()

        return path


def load_pass(path: Path) -> dict:
    """Read a Pass JSON.gz back."""
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    return json.loads(path.read_text())


def iter_passes(passes_dir: Path):
    """Yield Pass dicts in date order from a passes/ directory."""
    files = sorted(passes_dir.glob("*.json.gz")) + sorted(passes_dir.glob("*.json"))
    seen = set()
    for f in files:
        date_key = f.stem.replace(".json", "")
        if date_key in seen:
            continue
        seen.add(date_key)
        yield load_pass(f)
