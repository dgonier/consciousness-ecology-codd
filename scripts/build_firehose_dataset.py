"""Build a per-trading-day firehose dataset from the Polygon cache.

For each trading day t in the corpus, emit one row containing:
  - firehose_news: every article in the (yesterday-after-close, today-before-close)
    window, deduplicated by article id, with all the tickers it tags
  - firehose_bars: yesterday's daily bars for the watchlist universe
  - targets: {ticker -> actual close-to-close direction, return, magnitude_bucket,
              was_mentioned_in_news}

Output: data/firehose_dataset.jsonl + a tiny stats summary.

This dataset is the input to the head-to-head eval:
  * bare-Qwen baseline reads the firehose directly
  * ecology pipeline routes the firehose through herbivores → belief network
    → carnivore → apex, then we compare predictions against the same targets.

Cutoff convention: a trading day t's firehose includes news published from
(t-1 16:00 ET) through (t 16:00 ET). Bars used as input are the t-1 daily bar.
Target is computed from t's actual close vs t-1 close.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "external" / "polygon_cache" / "polygon"
OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "firehose_dataset.jsonl"
STATS_PATH = OUT_DIR / "firehose_dataset_stats.json"

# Market close in ET; we use 21:00 UTC as a reasonable close-time approximation
# (post-DST it's 20:00 UTC; we'll use 21:00 for consistency since most news
# explicitly tagged "after-hours" lands well after this).
MARKET_CLOSE_UTC_HOUR = 21
MAGNITUDE_FLAT_THRESHOLD = 0.005  # |return| < 0.5% = flat
MAGNITUDE_LARGE_THRESHOLD = 0.02  # |return| >= 2.0% = large


def load_news(ticker_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(ticker_dir.glob("*.jsonl")):
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def load_bars(ticker_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(ticker_dir.glob("*.jsonl")):
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def parse_news_ts(s: str) -> datetime:
    # Polygon: "2026-02-28T01:15:00Z"
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def magnitude_bucket(ret: float) -> str:
    a = abs(ret)
    if a < MAGNITUDE_FLAT_THRESHOLD:
        return "flat"
    if a >= MAGNITUDE_LARGE_THRESHOLD:
        return "large"
    return "medium"


def direction(ret: float) -> str:
    if abs(ret) < MAGNITUDE_FLAT_THRESHOLD:
        return "flat"
    return "up" if ret > 0 else "down"


def main() -> None:
    news_root = CACHE / "news"
    bars_root = CACHE / "bars"

    universe = sorted(d.name for d in news_root.iterdir() if d.is_dir())
    print(f"universe: {len(universe)} tickers — {universe}")

    # Load all news (de-dup by id; an article can appear under multiple
    # ticker directories since Polygon mirrors per-ticker)
    all_news_by_id: dict[str, dict] = {}
    for tk in universe:
        for art in load_news(news_root / tk):
            aid = art.get("id")
            if not aid:
                continue
            if aid in all_news_by_id:
                # Already have it from another ticker's mirror; merge tickers
                existing = all_news_by_id[aid].setdefault("tickers", [])
                for t in art.get("tickers", []):
                    if t not in existing:
                        existing.append(t)
            else:
                all_news_by_id[aid] = art
    all_news = list(all_news_by_id.values())
    print(f"unique news articles: {len(all_news)}")

    # Load all bars per ticker, build a date → ticker → close lookup
    bars_by_ticker: dict[str, dict[str, dict]] = {}
    for tk in universe:
        rows = load_bars(bars_root / tk)
        bars_by_ticker[tk] = {r["date"]: r for r in rows}
    all_dates: set[str] = set()
    for tk_bars in bars_by_ticker.values():
        all_dates.update(tk_bars.keys())
    trading_days = sorted(all_dates)
    print(f"trading days in corpus: {len(trading_days)} from {trading_days[0]} to {trading_days[-1]}")

    # Index news by date-window assignment. For trading day t, an article is
    # in-window if published in [t-1 21:00 UTC, t 21:00 UTC).
    # Pre-bucket articles: round their UTC timestamp to (date, in-window-for-day)
    news_by_day: dict[str, list[dict]] = defaultdict(list)
    skipped_no_ts = 0
    for art in all_news:
        ts_str = art.get("published_utc")
        if not ts_str:
            skipped_no_ts += 1
            continue
        try:
            ts = parse_news_ts(ts_str)
        except Exception:
            skipped_no_ts += 1
            continue
        # Find the trading day t whose window (t-1 21:00 UTC, t 21:00 UTC]
        # contains ts. That's the trading day on which this article is
        # actionable. Close convention: published at 21:00:00Z exactly is
        # assigned to the *next* trading day.
        # Approach: take ts.date() — if ts hour < 21, assignable to ts.date();
        # else assignable to ts.date() + 1
        d = ts.date()
        if ts.hour >= MARKET_CLOSE_UTC_HOUR:
            d = d + timedelta(days=1)
        news_by_day[d.isoformat()].append(art)
    print(f"news bucketed into {len(news_by_day)} day-windows; "
          f"skipped {skipped_no_ts} without timestamp")

    # Build dataset rows: for each trading day t with bars, compute targets,
    # gather firehose news from news_by_day[t], gather firehose bars from t-1.
    dataset: list[dict] = []
    skipped_no_prev: int = 0
    for i, t in enumerate(trading_days):
        if i == 0:
            skipped_no_prev += 1
            continue
        t_prev = trading_days[i - 1]
        # Targets per ticker
        targets = []
        firehose_bars = []
        # Build news ticker-mention set across the day's firehose so we can
        # set was_mentioned_in_news per ticker.
        day_news = news_by_day.get(t, [])
        mentioned: set[str] = set()
        for art in day_news:
            for tk in art.get("tickers", []):
                if tk in universe:
                    mentioned.add(tk)
        for tk in universe:
            tk_bars = bars_by_ticker[tk]
            bar_t = tk_bars.get(t)
            bar_prev = tk_bars.get(t_prev)
            if bar_t is None or bar_prev is None:
                continue
            ret = (bar_t["close"] - bar_prev["close"]) / bar_prev["close"]
            targets.append({
                "ticker": tk,
                "prev_close": bar_prev["close"],
                "actual_close": bar_t["close"],
                "actual_return": round(ret, 5),
                "actual_direction": direction(ret),
                "magnitude_bucket": magnitude_bucket(ret),
                "was_mentioned_in_news": tk in mentioned,
            })
            firehose_bars.append({
                "ticker": tk,
                "date": t_prev,
                "open":   bar_prev["open"],
                "high":   bar_prev["high"],
                "low":    bar_prev["low"],
                "close":  bar_prev["close"],
                "volume": bar_prev["volume"],
                "vwap":   bar_prev.get("vwap"),
            })
        # Trim news to fields we actually need
        firehose_news = [{
            "id": a.get("id"),
            "published_utc": a.get("published_utc"),
            "title": a.get("title"),
            "description": a.get("description"),
            "tickers": [t for t in a.get("tickers", []) if t in universe],
            "all_tickers": a.get("tickers", []),  # keep extra-universe context
            "keywords": a.get("keywords", []),
            "publisher": a.get("publisher", {}).get("name"),
        } for a in day_news]
        dataset.append({
            "date": t,
            "prev_trading_day": t_prev,
            "firehose_news": firehose_news,
            "firehose_bars": firehose_bars,
            "targets": targets,
            "n_news": len(firehose_news),
            "n_targets": len(targets),
            "n_mentioned": len(mentioned),
        })

    print(f"dataset rows: {len(dataset)}; skipped {skipped_no_prev} for no prev-day")

    with OUT_PATH.open("w") as f:
        for row in dataset:
            f.write(json.dumps(row) + "\n")

    # Stats
    n_news_per_day = [r["n_news"] for r in dataset]
    n_mentioned_per_day = [r["n_mentioned"] for r in dataset]
    n_targets_per_day = [r["n_targets"] for r in dataset]
    direction_counts = {"up": 0, "down": 0, "flat": 0}
    magnitude_counts = {"flat": 0, "medium": 0, "large": 0}
    for r in dataset:
        for t in r["targets"]:
            direction_counts[t["actual_direction"]] += 1
            magnitude_counts[t["magnitude_bucket"]] += 1
    total_ticker_days = sum(n_targets_per_day)
    stats = {
        "rows": len(dataset),
        "first_day": dataset[0]["date"] if dataset else None,
        "last_day": dataset[-1]["date"] if dataset else None,
        "tickers": universe,
        "ticker_count": len(universe),
        "total_ticker_day_predictions": total_ticker_days,
        "news_per_day": {
            "min": min(n_news_per_day) if n_news_per_day else 0,
            "max": max(n_news_per_day) if n_news_per_day else 0,
            "mean": round(sum(n_news_per_day) / len(n_news_per_day), 1) if n_news_per_day else 0,
            "median": sorted(n_news_per_day)[len(n_news_per_day)//2] if n_news_per_day else 0,
        },
        "mentioned_tickers_per_day": {
            "min": min(n_mentioned_per_day) if n_mentioned_per_day else 0,
            "max": max(n_mentioned_per_day) if n_mentioned_per_day else 0,
            "mean": round(sum(n_mentioned_per_day) / len(n_mentioned_per_day), 1) if n_mentioned_per_day else 0,
        },
        "direction_distribution": direction_counts,
        "magnitude_distribution": magnitude_counts,
        "class_balance_up_pct": round(100 * direction_counts["up"] / total_ticker_days, 1) if total_ticker_days else None,
    }
    with STATS_PATH.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nwrote {OUT_PATH}")
    print(f"wrote {STATS_PATH}")
    print(f"\n=== STATS ===")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
