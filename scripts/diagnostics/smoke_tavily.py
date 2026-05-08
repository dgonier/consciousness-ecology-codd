"""Smoke test: TavilyResearchTool against 5 StockNet test-split scenarios.

Picks a small set of (ticker, as_of_date) inside the StockNet test window
(2015-10-01 .. 2016-01-01) and dumps the snippets each one returns so we
can eyeball quality + date adherence.

Run:
  .venv/bin/python -u scripts/diagnostics/smoke_tavily.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env (TAVILY_API_KEY) without adding a dep.
ROOT = Path(__file__).resolve().parents[2]
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"'))

sys.path.insert(0, str(ROOT))
from trophic.research_tools.tavily_search import TavilyResearchTool  # noqa: E402


SCENARIOS = [
    # (ticker, as_of_date) — all inside StockNet test window 2015-10-01 .. 2016-01-01
    ("AAPL", "2015-10-01"),
    ("AAPL", "2015-10-02"),
    ("GOOG", "2015-10-22"),  # Alphabet Q3 earnings era
    ("FB",   "2015-11-04"),  # Facebook Q3 earnings era
    ("AMZN", "2015-12-15"),
]


def main() -> int:
    tool = TavilyResearchTool(window_days=5)
    if not tool.is_available():
        print("ERROR: TAVILY_API_KEY not in env")
        return 1

    print(f"Tavily smoke test — {len(SCENARIOS)} scenarios, 5-day window each\n")
    total = 0
    in_window = 0
    for ticker, as_of in SCENARIOS:
        print(f"=== {ticker} as_of={as_of} (window {tool.window_days}d back) ===")
        snippets = tool.query(ticker, as_of, focus="stock news earnings", n_results=5)
        print(f"  returned: {len(snippets)}")
        total += len(snippets)
        for i, s in enumerate(snippets, 1):
            in_w = ""
            if s.date:
                try:
                    from datetime import date
                    d = date.fromisoformat(s.date)
                    end = date.fromisoformat(as_of)
                    start = end.replace(day=1) if False else None
                    from datetime import timedelta
                    start = end - timedelta(days=tool.window_days)
                    if start <= d <= end:
                        in_window += 1
                        in_w = " [IN-WINDOW]"
                    else:
                        in_w = f" [OUT-OF-WINDOW: {s.date}]"
                except ValueError:
                    pass
            print(f"  [{i}] {s.date or '?date'}  {s.source}{in_w}")
            print(f"      {s.title[:120]}")
            if s.snippet:
                print(f"      {s.snippet[:200]}")
        print()

    print(f"summary: {total} snippets total, {in_window} dated within their window")
    return 0


if __name__ == "__main__":
    sys.exit(main())
