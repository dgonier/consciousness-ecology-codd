"""Probe SerperResearchTool: load .env, run a benchmark-date query,
   verify date-filter is being respected (results from 2015-09-2X)."""
from __future__ import annotations

import os
import re
import sys


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
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        print("[FAIL] SERPER_API_KEY not in env")
        return 1
    print(f"key prefix: {key[:6]}…  len={len(key)}")

    from trophic.research_tools import SerperResearchTool
    t = SerperResearchTool()
    print(f"available: {t.is_available()}")

    print(f"\n--- AAPL earnings, as_of 2015-10-01 (window 2015-09-24 → 2015-10-01) ---")
    rs = t.query("AAPL", "2015-10-01", focus="earnings news", n_results=5)
    print(f"got {len(rs)} snippets")
    for s in rs:
        print(f"  date={s.date} | src={s.source}")
        print(f"    title: {(s.title or '')[:100]}")
        print(f"    url:   {s.url}")
        print(f"    snip:  {(s.snippet or '')[:160]}")
        print()

    if not rs:
        return 2

    # Sanity check: do any snippets mention 2015 or 2016? If all dates
    # are recent (2025/2026), the date filter isn't being respected.
    all_text = " ".join((s.title or "") + " " + (s.snippet or "") + " " + (s.date or "") for s in rs)
    has_old = any(y in all_text for y in ("2015", "2016"))
    has_new = any(y in all_text for y in ("2024", "2025", "2026"))
    print(f"[check] mentions 2015/2016: {has_old}   mentions 2024+: {has_new}")
    if has_new and not has_old:
        print("[WARN] all snippets appear recent — date filter may not be working")
        return 3
    if has_old:
        print("[OK] date filter respected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
