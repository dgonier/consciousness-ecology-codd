"""Probe GoogleCSE: load .env, print availability, run a date-bound query,
   and surface the raw error body if Google rejects."""
from __future__ import annotations

import json
import os
import re
import sys
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def load_env(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
                continue
            os.environ.setdefault(k, v)


def main() -> int:
    load_env()
    key = os.environ.get("GOOGLE_API_KEY")
    cx = os.environ.get("GOOGLE_CSE_ID") or os.environ.get("GOOGLE_CSE")
    if not key or not cx:
        print(f"[FAIL] missing creds: api_key={bool(key)} cse={bool(cx)}")
        return 1
    print(f"key prefix: {key[:6]}…  cx: {cx}")

    # 1) raw API probe (no date filter)
    url = "https://www.googleapis.com/customsearch/v1?" + urlencode({
        "key": key, "cx": cx, "q": "apple stock", "num": 1,
    })
    try:
        with urlopen(Request(url), timeout=15) as r:
            data = json.loads(r.read().decode())
            n = data.get("searchInformation", {}).get("totalResults")
            print(f"[OK] plain query reachable, totalResults={n}")
    except HTTPError as e:
        body = e.read().decode()
        print(f"[FAIL] HTTP {e.code} on plain query")
        print(body[:500])
        return 1

    # 2) date-bounded benchmark probe
    from trophic.research_tools import GoogleCSEResearchTool
    t = GoogleCSEResearchTool()
    print(f"\n--- AAPL earnings news, as_of 2015-10-01 (window 2015-09-24 → 2015-10-01) ---")
    rs = t.query("AAPL", "2015-10-01", focus="earnings news", n_results=5)
    print(f"got {len(rs)} snippets")
    for s in rs:
        print(f"  date={s.date} | src={s.source}")
        print(f"    title: {(s.title or '')[:100]}")
        print(f"    snip:  {(s.snippet or '')[:140]}")
    return 0 if rs else 2


if __name__ == "__main__":
    sys.exit(main())
