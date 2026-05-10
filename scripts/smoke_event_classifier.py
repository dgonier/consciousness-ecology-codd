"""Smoke-test the event classifier on 5 real Polygon news articles.

Picks articles with strong-looking signals (earnings, M&A, regulatory) by
keyword search so we exercise the meaningful path. Prints the LLM's
returned activations and validates schema mapping.
"""
from __future__ import annotations

import json
import os
import re
import sys
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

import dspy

from trophic.beliefs.dspy_qwen_lm import QwenLocalLM
from trophic.beliefs.herbivores.event_classifier import EventClassifierHerbivore
from trophic.config import ModelConfig
from trophic.model_host import ModelHost


UNIVERSE = [
    "AAPL", "ABBV", "AMZN", "AVGO", "CSCO", "CVX", "GOOG", "HD", "JNJ", "JPM",
    "KO", "MA", "MCD", "MRK", "MSFT", "PEP", "PG", "UNH", "V", "WMT",
]

DATASET = ROOT / "data" / "firehose_dataset.jsonl"
CACHE = ROOT / "external" / "event_classifier_cache"


def hr() -> None:
    print("─" * 78)


def main() -> None:
    print("Event classifier smoke test (local Qwen3-4B via DSPy)")
    hr()
    print("loading Qwen3-4B...")
    cfg = ModelConfig()
    host = ModelHost(cfg)
    lm = QwenLocalLM(host, max_tokens=1024, temperature=0.0)
    dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
    print(f"  configured DSPy with local Qwen ({cfg.chat_model_id}, mock={cfg.mock})")
    hr()

    # Load all articles, dedupe by id
    articles: dict[str, dict] = {}
    with DATASET.open() as fh:
        for line in fh:
            row = json.loads(line)
            for art in row["firehose_news"]:
                articles[art["id"]] = art
    print(f"unique articles in corpus: {len(articles)}")

    # Pick 5 articles likely to have strong signals
    keyword_priorities = [
        "earnings beat", "earnings miss", "guidance",
        "acquir", "merge",  # M&A
        "lawsuit", "FDA",
        "buyback",
    ]
    picked: list[dict] = []
    seen_ids: set[str] = set()
    for kw in keyword_priorities:
        if len(picked) >= 5:
            break
        for art in articles.values():
            if art["id"] in seen_ids:
                continue
            haystack = (art.get("title", "") + " " + (art.get("description") or "")).lower()
            if kw.lower() in haystack:
                picked.append(art)
                seen_ids.add(art["id"])
                break
    if len(picked) < 5:
        # Top up with random
        for art in articles.values():
            if len(picked) >= 5:
                break
            if art["id"] in seen_ids:
                continue
            picked.append(art)
            seen_ids.add(art["id"])

    print(f"selected {len(picked)} articles for classification")
    hr()

    herb = EventClassifierHerbivore(
        universe=UNIVERSE, cache_dir=CACHE,
    )
    for i, art in enumerate(picked):
        print(f"\n[{i+1}] id={art['id'][:16]}...")
        print(f"    title: {art.get('title')}")
        tagged_in_uni = [t for t in (art.get('all_tickers') or art.get('tickers') or []) if t in UNIVERSE]
        print(f"    in-universe tickers: {tagged_in_uni}")
        result = herb.classify(art)
        print(f"    cache_hit: {result.get('_cache_hit')}  no_signal: {result.get('no_signal')}  "
              f"subject={result.get('_subject_tickers')}")
        if result.get("_error"):
            print(f"    ERROR: {result['_error']}")
        for a in result.get("activations", []):
            print(f"    → {a['ticker']:5s}  {a['belief_template']:40s}  {a['direction']:9s}  "
                  f"{a['magnitude']:8s}  conf={a['confidence']:9s}")
            if a.get("reasoning"):
                print(f"      reasoning: {a['reasoning'][:120]}")

    hr()
    print("dispatch round-trip — converting result #1 to BeliefActivation list:")
    acts = herb.classify_to_activations(picked[0])
    for a in acts:
        print(f"  BeliefActivation: target={a.target_belief_id}  "
              f"dir={a.direction_of_effect}  mag={a.magnitude}  species={a.species_id}")


if __name__ == "__main__":
    main()
