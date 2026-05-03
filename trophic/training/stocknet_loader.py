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

import os
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator, Literal

from ..adapters.stocknet import OhlcvNormalizedAdapter, TokenizedTweetAdapter
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
    """Returns {date_str: payload_dict} for a single ticker's preprocessed
    OHLCV file. Routes the raw text through `OhlcvNormalizedAdapter` so
    the format-conversion logic lives in exactly one place
    (`trophic/adapters/stocknet/ohlcv_normalized.py`).

    HTTP fetch + caching stay here (they're not adapter responsibilities).
    """
    url = f"{REPO_RAW}/price/preprocessed/{ticker}.txt"
    text = _http_get(url, cache_dir)
    if text is None:
        raise FileNotFoundError(f"price file missing for {ticker}: {url}")

    rows: dict[str, dict] = {}
    for ri in OhlcvNormalizedAdapter(ticker).adapt(text):
        rows[ri.payload["date"]] = ri.payload
    return rows


def _read_tweets(ticker: str, day: str, cache_dir: Path | None = None) -> list[str]:
    """Returns the tokenized-tweet text bodies for (ticker, day) by routing
    the raw JSONL through `TokenizedTweetAdapter`. The adapter aggregates
    a day's tweets into a single RawInput payload[`body`] (newline-joined).

    For backward compatibility we return a `list[str]` (one per tweet line);
    callers downstream just check truthiness and pass it to
    `_press_payload_from_tweets`.
    """
    url = f"{REPO_RAW}/tweet/preprocessed/{ticker}/{day}"
    text = _http_get(url, cache_dir)
    if text is None:
        return []
    out: list[str] = []
    for ri in TokenizedTweetAdapter(ticker, day).adapt(text):
        body = ri.payload.get("body", "") or ""
        if body:
            out.extend(body.split("\n"))
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


_FORECAST_FEATURE_CACHE: dict[str, list[float]] | None = None
_FORECAST_FEATURE_CACHE_PATH: Path | None = None


def _load_forecast_feature_cache(cache_dir: Path) -> dict[str, list[float]]:
    """Lazy-load the per-(ticker,date) Chronos feature cache from disk."""
    global _FORECAST_FEATURE_CACHE, _FORECAST_FEATURE_CACHE_PATH
    if _FORECAST_FEATURE_CACHE is not None:
        return _FORECAST_FEATURE_CACHE
    p = cache_dir / "forecast_features.json"
    _FORECAST_FEATURE_CACHE_PATH = p
    if p.exists():
        import json
        try:
            _FORECAST_FEATURE_CACHE = json.loads(p.read_text())
        except Exception:
            _FORECAST_FEATURE_CACHE = {}
    else:
        _FORECAST_FEATURE_CACHE = {}
    return _FORECAST_FEATURE_CACHE


def _save_forecast_feature_cache() -> None:
    if _FORECAST_FEATURE_CACHE is None or _FORECAST_FEATURE_CACHE_PATH is None:
        return
    import json
    _FORECAST_FEATURE_CACHE_PATH.write_text(json.dumps(_FORECAST_FEATURE_CACHE))


def _compute_forecast_features(
    ticker: str, date: str, history: list[dict], cache_dir: Path | None,
) -> list[float] | None:
    """Compute (or load) Chronos numeric features for one scenario.

    Returns None if Chronos isn't available or the history is unusable.
    """
    if cache_dir is None:
        return None
    cache = _load_forecast_feature_cache(cache_dir)
    key = f"{ticker}|{date}"
    if key in cache:
        return cache[key]
    try:
        from ..forecast_features import chronos_features_from_bars
        feats = chronos_features_from_bars(history)
    except Exception as e:
        # Don't fail scenario building if Chronos is unavailable; just
        # leave features=None so the trough oracle path stays in use.
        print(f"[stocknet_loader] forecast feature compute failed for {key}: {e}")
        return None
    cache[key] = feats
    return feats


def build_stocknet_scenarios(
    *,
    split: Literal["train", "dev", "test"] = "test",
    tickers: list[str] | None = None,
    history_days: int = 5,
    max_per_ticker: int | None = None,
    cache_dir: Path | None = Path("/home/dgonier/ecology_experiment/trophic/external/stocknet_cache"),
    require_tweets: bool = True,
    compute_forecast: bool | None = None,
) -> list[Scenario]:
    """Walk StockNet and emit a list of Scenario objects.

    Args:
      split: which date window from the paper.
      tickers: list of ticker symbols. Default: SMOKE_TICKERS_TOP5.
      history_days: how many trading days back to include in OHLCV history.
      max_per_ticker: cap to keep smoke runs cheap. None = no cap.
      cache_dir: where to cache HTTP responses. None = no cache.
      require_tweets: skip days with zero tweets (matches the paper's filter).
      compute_forecast: if True, run Chronos on the OHLCV history of each
        scenario at build time and cache numeric forecast features under
        cache_dir/forecast_features.json. Defaults to env var
        TROPHIC_COMPUTE_FORECAST (default 0). Set to 1 to opt-in for
        seed38+ training that uses the numeric forecaster channel.
    """
    if compute_forecast is None:
        import os
        compute_forecast = os.environ.get("TROPHIC_COMPUTE_FORECAST", "0") == "1"
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
            forecast_feats: list[float] | None = None
            if compute_forecast:
                forecast_feats = _compute_forecast_features(
                    ticker, d, history, cache_dir=cache_dir,
                )
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
                    forecaster_features=forecast_feats,
                )
            )
            per_ticker_count += 1
            if max_per_ticker is not None and per_ticker_count >= max_per_ticker:
                break

    if compute_forecast:
        _save_forecast_feature_cache()
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
