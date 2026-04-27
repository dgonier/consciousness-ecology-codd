"""StockNet (ACL-18) loader. Adapts the public dataset into our Scenario shape.

Reference:
  Yumo Xu and Shay B. Cohen. 2018. Stock Movement Prediction from Tweets and
  Historical Prices. ACL.
  Dataset: https://github.com/yumoxu/stocknet-dataset

Conventions:
  - Following the paper's supervision: label=1 ("up") if next-day movement_pct
    >= 0.0055, label=0 ("down") if <= -0.005. Days with movement in the
    "no-action" band (-0.005, 0.0055) are filtered.
  - Each scenario covers ONE (ticker, date) and contains the day's tweets +
    a history window of OHLCV bars.
  - The smoke-test mode pulls data directly from raw GitHub URLs to avoid
    needing a vendored copy on disk. For full-set runs, vendor at
    `external/stocknet/` and set use_local=True.
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator, Literal

from ..types import RawInput
from .scenarios import Scenario
from .xml_schema import emit_prediction, emit_synthesis


REPO_RAW = "https://raw.githubusercontent.com/yumoxu/stocknet-dataset/master"

# StockNet paper bands.
UP_THRESHOLD = 0.0055
DOWN_THRESHOLD = -0.005

# Standard splits per the paper.
TRAIN_DATES = ("2014-01-01", "2015-08-01")
DEV_DATES = ("2015-08-01", "2015-10-01")
TEST_DATES = ("2015-10-01", "2016-01-01")

# Top-5 by-cap tickers that overlap with our training tickers — best chance
# for coherent first-pass output. Expand to all 88 once smoke is green.
SMOKE_TICKERS_TOP5 = ["AAPL", "GOOG", "MSFT", "AMZN", "JPM"]


def _http_get(url: str, cache_dir: Path | None = None) -> str | None:
    """Fetch a URL with optional file cache. Returns None on 404."""
    if cache_dir:
        # Cache key: last two path segments under cache_dir.
        key = url.split(REPO_RAW + "/")[-1].replace("/", "__")
        path = cache_dir / key
        if path.exists():
            return path.read_text()
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            text = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return text


def _read_price_file(ticker: str, cache_dir: Path | None = None) -> dict[str, dict]:
    """Returns {date_str: {"movement_pct": float, "open": ..., "high": ..., ...}}.

    Uses preprocessed (normalized) price file. Per the paper's convention,
    the columns after movement_pct are normalized OHLC, then volume raw.
    """
    url = f"{REPO_RAW}/price/preprocessed/{ticker}.txt"
    text = _http_get(url, cache_dir)
    if text is None:
        raise FileNotFoundError(f"price file missing for {ticker}: {url}")

    rows: dict[str, dict] = {}
    for line in text.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        d = parts[0]
        rows[d] = {
            "movement_pct": float(parts[1]),
            "open_norm": float(parts[2]),
            "high_norm": float(parts[3]),
            "low_norm": float(parts[4]),
            "close_norm": float(parts[5]),
            "volume": float(parts[6]),
        }
    return rows


def _read_tweets(ticker: str, day: str, cache_dir: Path | None = None) -> list[str]:
    """Returns a list of tokenized tweet texts for (ticker, day). Tokens are
    space-joined; URLs/at-mentions are preserved as 'URL' / 'AT_USER'.
    """
    url = f"{REPO_RAW}/tweet/preprocessed/{ticker}/{day}"
    text = _http_get(url, cache_dir)
    if text is None:
        return []
    out: list[str] = []
    for line in text.strip().split("\n"):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        tokens = obj.get("text", [])
        if isinstance(tokens, list):
            out.append(" ".join(tokens))
    return out


def _label_from_movement(mvt: float) -> Literal["up", "down"] | None:
    if mvt >= UP_THRESHOLD:
        return "up"
    if mvt <= DOWN_THRESHOLD:
        return "down"
    return None


def _ohlcv_payload_from_history(history: list[dict]) -> dict:
    """Pack the N-day history into the same payload shape our OHLCV producer
    expects. Keys mirror trophic.training.scenarios._ohlcv_payload output.
    """
    return {
        "bars": [
            {
                "open": h["open_norm"],
                "high": h["high_norm"],
                "low": h["low_norm"],
                "close": h["close_norm"],
                "volume": h["volume"],
            }
            for h in history
        ],
        "source": "stocknet_normalized",
    }


def _press_payload_from_tweets(ticker: str, tweets: list[str]) -> dict:
    """Pack a day's tweets as a press-style payload. Concatenated as text
    for our `press` producer, with ticker echoed for grounding.
    """
    text = "\n".join(tweets)
    return {
        "ticker": ticker,
        "headline": f"daily tweet feed ({len(tweets)} tweets)",
        "body": text[:4000],   # cap to avoid blowing token budget
        "source": "stocknet_tweets",
    }


def _date_iter(start: str, end: str) -> Iterator[str]:
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    while s < e:
        yield s.isoformat()
        s += timedelta(days=1)


def build_stocknet_scenarios(
    *,
    split: Literal["train", "dev", "test"] = "test",
    tickers: list[str] | None = None,
    history_days: int = 5,
    max_per_ticker: int | None = None,
    cache_dir: Path | None = Path("/home/dgonier/ecology_experiment/trophic/external/stocknet_cache"),
    require_tweets: bool = True,
) -> list[Scenario]:
    """Walk StockNet and emit a list of Scenario objects.

    Args:
      split: which date window from the paper.
      tickers: list of ticker symbols. Default: SMOKE_TICKERS_TOP5.
      history_days: how many trading days back to include in OHLCV history.
      max_per_ticker: cap to keep smoke runs cheap. None = no cap.
      cache_dir: where to cache HTTP responses. None = no cache.
      require_tweets: skip days with zero tweets (matches the paper's filter).
    """
    if split == "train":
        date_range = TRAIN_DATES
    elif split == "dev":
        date_range = DEV_DATES
    else:
        date_range = TEST_DATES

    tickers = tickers or SMOKE_TICKERS_TOP5

    scenarios: list[Scenario] = []
    for ticker in tickers:
        prices = _read_price_file(ticker, cache_dir=cache_dir)
        # Sort dates ascending so we can slice history easily.
        sorted_dates = sorted(prices.keys())
        date_to_idx = {d: i for i, d in enumerate(sorted_dates)}

        per_ticker_count = 0
        for d in _date_iter(*date_range):
            if d not in prices:
                continue
            row = prices[d]
            label = _label_from_movement(row["movement_pct"])
            if label is None:
                continue
            i = date_to_idx[d]
            if i < history_days:
                continue
            history = [prices[sorted_dates[j]] for j in range(i - history_days, i)]

            tweets = _read_tweets(ticker, d, cache_dir=cache_dir)
            if require_tweets and not tweets:
                continue

            inputs = [
                RawInput(
                    id=f"stocknet.{ticker}.{d}.ohlcv",
                    source="ohlcv",
                    payload=_ohlcv_payload_from_history(history),
                ),
            ]
            if tweets:
                inputs.append(
                    RawInput(
                        id=f"stocknet.{ticker}.{d}.press",
                        source="press",
                        payload=_press_payload_from_tweets(ticker, tweets),
                    )
                )

            # SFT-shaped targets — only the predator target is used at eval
            # time (we score direction). Herbivore targets are placeholders so
            # the existing pipeline doesn't choke.
            scenarios.append(
                Scenario(
                    name=f"stocknet_{split}_{ticker}_{d}",
                    inputs=inputs,
                    technical_target=emit_synthesis(
                        kind="technical", ticker=ticker, bias=label, signal="moderate",
                    ),
                    fundamental_target=emit_synthesis(
                        kind="fundamental", ticker=ticker, bias=label, signal="moderate",
                    ) if tweets else emit_synthesis(kind="fundamental", abstain=True),
                    predator_target=emit_prediction(
                        ticker=ticker,
                        direction=label,
                        pct_move=None,    # StockNet is direction-only
                        horizon_min=1440, # next-day
                        sigma_pct=None,
                        confidence=0.65,
                    ),
                )
            )
            per_ticker_count += 1
            if max_per_ticker is not None and per_ticker_count >= max_per_ticker:
                break

    return scenarios


if __name__ == "__main__":
    # Smoke-print: top 5 tickers, 2-week test window, no per-ticker cap.
    scenarios = build_stocknet_scenarios(
        split="test",
        tickers=SMOKE_TICKERS_TOP5,
        history_days=5,
        max_per_ticker=None,
    )
    print(f"loaded {len(scenarios)} scenarios from StockNet test split (top 5 tickers)")
    if scenarios:
        s = scenarios[0]
        print(f"first: {s.name}")
        print(f"  inputs: {len(s.inputs)}")
        for inp in s.inputs:
            print(f"    {inp.source}: {len(str(inp.payload))} bytes")
        print(f"  predator_target: {(s.predator_target or '')[:200]}")
    # Class balance check
    from collections import Counter
    labels: Counter = Counter()
    for s in scenarios:
        if "<direction>up</direction>" in (s.predator_target or ""):
            labels["up"] += 1
        elif "<direction>down</direction>" in (s.predator_target or ""):
            labels["down"] += 1
    print(f"label balance: {dict(labels)}")
