"""Training corpus management — Stage 0 of #134 (decomposer job 3).

Reads teacher_hints + apex_pairs from data/training_corpus/, partitions by
(species, template, label_correct), and provides a clean handoff API for
downstream consumers:

  1. DSPy refinement of event_classifier (Stage 1)
  2. Per-species d* extraction (Stage 2)
  3. Hexis SFT on apex_pairs (Stage 3)

After a consumer processes a batch, archive the hints to
data/training_corpus/teacher_hints/consumed/{timestamp}.jsonl so the next
sweep doesn't re-train on stale data.

Idempotent — safe to run multiple times.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

ROOT = Path(__file__).resolve().parents[2]
HINTS_DIR = ROOT / "data" / "training_corpus" / "teacher_hints"
APEX_DIR = ROOT / "data" / "training_corpus" / "apex_pairs"


@dataclass
class HintBucket:
    """One (species, template) bucket with positive and negative examples."""
    species: str
    template: str
    correct: list[dict] = field(default_factory=list)
    wrong: list[dict] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.correct) + len(self.wrong)

    @property
    def precision(self) -> float:
        return len(self.correct) / self.n_total if self.n_total else 0.0

    @property
    def is_refinable(self, min_n: int = 10) -> bool:
        return len(self.correct) >= min_n and len(self.wrong) >= min_n


def load_pending_hints(path: Optional[Path] = None) -> list[dict]:
    """Read all hints from teacher_hints/pending.jsonl (one per line)."""
    p = path or HINTS_DIR / "pending.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().strip().split("\n") if line]


def load_pending_apex_pairs(path: Optional[Path] = None) -> list[dict]:
    """Read all apex pairs from apex_pairs/pending.jsonl."""
    p = path or APEX_DIR / "pending.jsonl"
    if not p.exists():
        return []
    txt = p.read_text().strip()
    if not txt:
        return []
    return [json.loads(line) for line in txt.split("\n") if line]


def bucket_hints(
    hints: list[dict],
    min_total: int = 10,
) -> dict[tuple[str, str], HintBucket]:
    """Partition hints by (species_id, belief_template) and split by
    label_correct.

    species_id is inferred from the consumer side (event_classifier vs
    cross_correlation). For now, hints currently in pending.jsonl come
    only from event_classifier (cross_correlation hints don't have ticker
    ground-truth alignment yet — the decomposer skips them).
    """
    buckets: dict[tuple[str, str], HintBucket] = {}
    for h in hints:
        # event_classifier is the only species producing hints right now
        species = h.get("species", "event_classifier.v0.qwen-local")
        template = h.get("belief_template")
        if template is None:
            continue
        key = (species, template)
        b = buckets.setdefault(key, HintBucket(species=species, template=template))
        if h.get("label_correct"):
            b.correct.append(h)
        else:
            b.wrong.append(h)
    return {k: b for k, b in buckets.items() if b.n_total >= min_total}


def refinement_targets(
    buckets: dict[tuple[str, str], HintBucket],
    precision_threshold: float = 0.55,
    min_total: int = 30,
    min_per_class: int = 10,
) -> list[HintBucket]:
    """Return buckets that warrant DSPy/d* refinement.

    Criteria:
      precision < threshold (the herbivore is wrong often enough to learn from)
      n_total >= min_total (enough data to learn from)
      each class has >= min_per_class examples (avoids degenerate splits)
    """
    out = []
    for b in buckets.values():
        if b.n_total < min_total:
            continue
        if b.precision >= precision_threshold:
            continue
        if len(b.correct) < min_per_class or len(b.wrong) < min_per_class:
            continue
        out.append(b)
    out.sort(key=lambda b: -b.n_total)
    return out


def archive_hints(
    consumed: list[dict],
    consumer_tag: str,
    hints_dir: Optional[Path] = None,
) -> Path:
    """Move a batch of consumed hints into consumed/{timestamp}__{consumer}.jsonl.

    Note: this does NOT remove them from pending.jsonl yet — the consumer
    decides whether the refinement was successful before pruning. Use
    `prune_consumed_from_pending` to actually remove them after MCC verification.
    """
    base = hints_dir or HINTS_DIR
    consumed_dir = base / "consumed"
    consumed_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = consumed_dir / f"{ts}__{consumer_tag}.jsonl"
    with out_path.open("w") as fh:
        for h in consumed:
            fh.write(json.dumps(h) + "\n")
    return out_path


def prune_consumed_from_pending(
    consumed_article_ids: set[str],
    hints_dir: Optional[Path] = None,
) -> int:
    """Remove consumed hints from pending.jsonl. Idempotent.

    Match by article_id (the most stable identifier).
    Returns count of pruned lines.
    """
    base = hints_dir or HINTS_DIR
    pending = base / "pending.jsonl"
    if not pending.exists():
        return 0
    keep = []
    pruned = 0
    for ln in pending.read_text().strip().split("\n"):
        if not ln:
            continue
        h = json.loads(ln)
        if h.get("article_id") in consumed_article_ids:
            pruned += 1
        else:
            keep.append(ln)
    pending.write_text("\n".join(keep) + ("\n" if keep else ""))
    return pruned


def summarize() -> dict:
    """One-shot summary of the corpus state — for logging / decision-making."""
    hints = load_pending_hints()
    pairs = load_pending_apex_pairs()
    buckets = bucket_hints(hints)
    targets = refinement_targets(buckets)
    return {
        "n_pending_hints": len(hints),
        "n_pending_apex_pairs": len(pairs),
        "n_buckets": len(buckets),
        "n_refinable_targets": len(targets),
        "targets": [
            {
                "species": t.species,
                "template": t.template,
                "n_correct": len(t.correct),
                "n_wrong": len(t.wrong),
                "precision": round(t.precision, 3),
            }
            for t in targets
        ],
    }


if __name__ == "__main__":
    s = summarize()
    print(json.dumps(s, indent=2))
