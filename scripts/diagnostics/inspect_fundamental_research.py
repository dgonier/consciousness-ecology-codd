"""Run the fundamental analyst on a handful of StockNet scenarios and
   print the EXACT Serper query, the EXACT snippets returned, and the
   resulting synthesis. Lets us see whether the research call surfaces
   anything that would change the analyst's read."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Load .env
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

os.environ.setdefault("TROPHIC_COMPUTE_FORECAST", "1")

from trophic.training.stocknet_loader import build_stocknet_scenarios
from trophic.apex_voters.evidence import build_evidence_packet
from trophic.decomposers import SpeciesRegistry
from trophic.herbivore_voters.factory import herbivore_from_species

# Load fundamental species
reg = SpeciesRegistry(path=Path("external/decomposer_kg/species_4b_only.jsonl"))
fund_sp = next(sp for sp in reg.alive() if sp.species_id == "qwen.analyst.fundamental")

# Build the herb. Will lazily resolve research_tool=Serper.
herb = herbivore_from_species(fund_sp)
print(f"herb: {herb.herb_id}")
print(f"research_tool attached: {type(herb.research_tool).__name__ if herb.research_tool else None}")
print(f"research_tool available: {herb.research_tool.is_available() if herb.research_tool else False}")
print()

# Pick 3 contrasting scenarios: 1 with target=down, 1 with target=up,
# and 1 from the AAPL_2015-10-01 case where Serper flipped the apex
# to wrong.
scens = build_stocknet_scenarios(
    split="test", tickers=["AAPL"], max_per_ticker=20, compute_forecast=True,
)
picks = [s for s in scens if s.name in (
    "stocknet_test_AAPL_2015-10-01",  # the harmful flip
    "stocknet_test_AAPL_2015-10-08",  # apex was correct in both runs (down)
    "stocknet_test_AAPL_2015-10-29",  # apex was correct in both runs (up)
)]

# Patch the herb's _qwen_forward to also print so we see what Qwen produces
# at each step.
original_forward = herb._qwen_forward
trace = []

def traced_forward(system, user, max_new_tokens):
    out = original_forward(system, user, max_new_tokens)
    trace.append({"system": system[:200], "user_tail": user[-1500:], "out": out[:800]})
    return out

herb._qwen_forward = traced_forward

# Patch the research tool's query to record raw response too
orig_query = herb.research_tool.query
research_log = []

def traced_query(ticker, as_of_date, focus="", n_results=5):
    rs = orig_query(ticker, as_of_date, focus, n_results)
    research_log.append({"ticker": ticker, "as_of": as_of_date, "focus": focus, "n_returned": len(rs), "snippets": rs})
    return rs

herb.research_tool.query = traced_query

# Build packet, then synthesize for each scenario
import json as _json
for scen in picks:
    print("=" * 100)
    print(f"SCENARIO: {scen.name}  target={scen.predator_target}")
    print("=" * 100)
    packet = build_evidence_packet(scen, herb_broadcasts=None)
    text = packet.text
    trace.clear(); research_log.clear()
    synth = herb.synthesize(text)

    # 1. The Qwen-formulated focus query
    print("\n--- STEP 1: Qwen-formulated research query ---")
    if trace:
        print(f"raw model output for query step:\n{trace[0]['out'][:300]}")
    print(f"final query string sent to Serper: {synth.provider_meta.get('research_query', '(none)')!r}")

    # 2. Serper response
    print("\n--- STEP 2: Serper results ---")
    if research_log:
        rl = research_log[0]
        print(f"called with: ticker={rl['ticker']!r}, as_of={rl['as_of']!r}, focus={rl['focus']!r}")
        print(f"got {rl['n_returned']} snippets:")
        for i, s in enumerate(rl["snippets"], 1):
            print(f"  [{i}] date={s.date} src={s.source}")
            print(f"      title: {s.title[:120]}")
            print(f"      snip:  {(s.snippet or '')[:200]}")
    else:
        print("(no research call recorded)")

    # 3. Synthesis output
    print("\n--- STEP 3: fundamental analyst synthesis ---")
    print(f"diet_tag:        {synth.diet_tag}")
    print(f"direction_hint:  {synth.direction_hint}")
    print(f"synthesis:       {synth.synthesis}")
    print(f"raw text:")
    print(synth.raw_text[:600])
    print()
