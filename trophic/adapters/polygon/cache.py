"""On-disk cache for Polygon API responses.

Pattern: cache by (ticker, year-month) shards. Bars are appended to a
JSONL shard per ticker per month; news the same. Reading is O(months
in window). Writing is idempotent — re-fetching a window merges by
date / news id.

Structure under TROPHIC_CACHE_DIR/polygon/:
  bars/{TICKER}/{YYYY-MM}.jsonl       — one OHLCV record per line
  news/{TICKER}/{YYYY-MM}.jsonl       — one news item per line

The cache is the canonical source — `get_bars()` and `get_news()`
return cached data first and only call the API for missing months.
This is what makes free-tier Polygon viable for benchmark runs.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .client import PolygonClient


_CACHE_ROOT = Path(
    os.environ.get(
        "TROPHIC_POLYGON_CACHE",
        os.path.join(
            os.environ.get("TROPHIC_CACHE_DIR",
                           os.path.expanduser("~/ecology_experiment/trophic/external/polygon_cache")),
            "polygon",
        ),
    )
)


class PolygonCache:
    """Append-only JSONL shards by ticker + year-month."""

    def __init__(self, root: Path = _CACHE_ROOT):
        self.root = Path(root)

    def _shard(self, kind: str, ticker: str, ym: str) -> Path:
        d = self.root / kind / ticker
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{ym}.jsonl"

    @staticmethod
    def _months_between(start: str, end: str) -> List[str]:
        s = date.fromisoformat(start)
        e = date.fromisoformat(end)
        out = []
        y, m = s.year, s.month
        while (y, m) <= (e.year, e.month):
            out.append(f"{y:04d}-{m:02d}")
            if m == 12:
                y += 1; m = 1
            else:
                m += 1
        return out

    def has_month(self, kind: str, ticker: str, ym: str) -> bool:
        p = self.root / kind / ticker / f"{ym}.jsonl"
        return p.exists() and p.stat().st_size > 0

    def read_month(self, kind: str, ticker: str, ym: str) -> List[Dict]:
        p = self.root / kind / ticker / f"{ym}.jsonl"
        if not p.exists():
            return []
        out = []
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def write_month(self, kind: str, ticker: str, ym: str, items: Iterable[Dict]) -> int:
        path = self._shard(kind, ticker, ym)
        n = 0
        with path.open("w") as f:
            for it in items:
                f.write(json.dumps(it, separators=(",", ":")) + "\n")
                n += 1
        return n


def _bars_to_records(aggs: List[Dict]) -> List[Dict]:
    """Convert Polygon agg shape (t,o,h,l,c,v,vw) → our schema."""
    out = []
    for x in aggs:
        ts_ms = x.get("t")
        if ts_ms is None:
            continue
        d = datetime.utcfromtimestamp(ts_ms / 1000.0).date().isoformat()
        out.append({
            "date": d,
            "open": x.get("o"),
            "high": x.get("h"),
            "low": x.get("l"),
            "close": x.get("c"),
            "volume": x.get("v"),
            "vwap": x.get("vw"),
        })
    return out


def get_bars(
    ticker: str, start: str, end: str,
    *, client: Optional[PolygonClient] = None, cache: Optional[PolygonCache] = None,
) -> List[Dict]:
    """Daily OHLCV bars in [start, end] inclusive. Cache-first; calls
    Polygon for any month not on disk. Returns list of dicts with keys
    date, open, high, low, close, volume, vwap.
    """
    cache = cache or PolygonCache()
    months = cache._months_between(start, end)

    # Find missing months
    missing = [m for m in months if not cache.has_month("bars", ticker, m)]
    if missing:
        client = client or PolygonClient()
        if not client.is_available():
            # Cache-only mode: return what we have
            pass
        else:
            # Fetch the full missing window in one API call (Polygon
            # supports arbitrary ranges; we then split into month-shards
            # for cache reuse).
            m_start = f"{missing[0]}-01"
            # Last day of last missing month
            ly, lm = map(int, missing[-1].split("-"))
            if lm == 12:
                next_first = date(ly + 1, 1, 1)
            else:
                next_first = date(ly, lm + 1, 1)
            m_end = (next_first - timedelta(days=1)).isoformat()
            aggs = client.list_aggs(ticker, m_start, m_end)
            recs = _bars_to_records(aggs)
            # Shard into months
            by_month: Dict[str, List[Dict]] = {}
            for r in recs:
                ym = r["date"][:7]
                by_month.setdefault(ym, []).append(r)
            for m in missing:
                cache.write_month("bars", ticker, m, by_month.get(m, []))

    # Read all months and filter to window
    out: List[Dict] = []
    for m in months:
        out.extend(cache.read_month("bars", ticker, m))
    out = [r for r in out if start <= r["date"] <= end]
    out.sort(key=lambda r: r["date"])
    return out


def get_news(
    ticker: str, start: str, end: str,
    *, client: Optional[PolygonClient] = None, cache: Optional[PolygonCache] = None,
    max_per_month: int = 1000,
) -> List[Dict]:
    """News items published in [start, end] (ISO yyyy-mm-dd). Cache-first,
    sharded by month. Each item has at least keys: id, published_utc,
    title, description, article_url, publisher.
    """
    cache = cache or PolygonCache()
    months = cache._months_between(start, end)
    missing = [m for m in months if not cache.has_month("news", ticker, m)]

    if missing:
        client = client or PolygonClient()
        if client.is_available():
            for m in missing:
                y, mo = map(int, m.split("-"))
                m_start = f"{y:04d}-{mo:02d}-01T00:00:00Z"
                if mo == 12:
                    next_first = date(y + 1, 1, 1)
                else:
                    next_first = date(y, mo + 1, 1)
                m_end = (next_first - timedelta(days=1)).isoformat() + "T23:59:59Z"

                items: List[Dict] = []
                cursor: Optional[str] = None
                pages = 0
                seen_ids: set = set()
                while pages < 20:  # hard cap to bound free-tier cost
                    page, cursor = client.list_ticker_news(
                        ticker, gte=m_start, lte=m_end, limit=100,
                        cursor=cursor,
                    )
                    pages += 1
                    if not page:
                        break
                    new = [it for it in page if it.get("id") not in seen_ids]
                    items.extend(new)
                    for it in new:
                        if it.get("id"):
                            seen_ids.add(it.get("id"))
                    if not cursor or not new or len(items) >= max_per_month:
                        break
                items = items[:max_per_month]
                cache.write_month("news", ticker, m, items)

    # Read window from cache
    out: List[Dict] = []
    for m in months:
        out.extend(cache.read_month("news", ticker, m))
    # Filter precisely
    def _within(it: Dict) -> bool:
        d = (it.get("published_utc") or "")[:10]
        return start <= d <= end
    out = [it for it in out if _within(it)]
    out.sort(key=lambda it: it.get("published_utc", ""))
    return out
