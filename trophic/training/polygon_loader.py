"""Polygon-backed scenario loader (StockBench-style).

Replaces StockNet (2014-2016) for tool-augmented analyst evaluation.
Same Scenario shape — drop-in for `apex_vote_eval.py` etc.

Window choice for "synthetic-rolling" benchmark:
  * Pick a recent K-week window (default: last 4 weeks). Last 5 trading
    days of the window are TARGETS; the prior trading days are
    OBSERVATIONS for each target's history.
  * Each scenario = (ticker, target_date) where target_date in the last
    week of the window. The OHLCV history is the prior `history_days`
    trading days; the news block is everything published in the same
    history window.
  * Direction label: next-day close vs current close. UP if pct > +0.55%,
    DOWN if pct < −0.50%. Same bands as StockNet.

This sidesteps the StockNet date-bound problem: every scenario is recent
enough that all frontier MCPs (Aiera, CapIQ, Moody's, Polygon news) cover
it natively, AND it's outside Qwen3-4B's pretrain cutoff (post-2024) so
no LLM has memorized the labels.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator, Literal

from ..adapters.polygon import get_bars, get_news
from ..types import RawInput
from .scenarios import Scenario
from .xml_schema import emit_prediction, emit_synthesis


# StockNet's bands — keeping them constant lets us compare numbers.
UP_THRESHOLD = 0.0055
DOWN_THRESHOLD = -0.005

# StockBench DJIA-top-20 universe (good liquidity, sector spread).
DJIA_TOP20 = [
    "AAPL", "MSFT", "GOOG", "AMZN", "JPM",
    "UNH", "V", "PG", "HD", "MA",
    "JNJ", "MRK", "CVX", "ABBV", "KO",
    "PEP", "WMT", "CSCO", "AVGO", "MCD",
]


def _label_from_movement(pct: float) -> Literal["up", "down"] | None:
    if pct >= UP_THRESHOLD:
        return "up"
    if pct <= DOWN_THRESHOLD:
        return "down"
    return None


def _ohlcv_payload_from_bars(bars: list[dict]) -> dict:
    """Same shape the StockNet adapter emits."""
    out_bars = []
    for b in bars:
        out_bars.append({
            "open": float(b.get("open") or 0),
            "high": float(b.get("high") or 0),
            "low": float(b.get("low") or 0),
            "close": float(b.get("close") or 0),
            "volume": int(b.get("volume") or 0),
        })
    return {"bars": out_bars}


def _press_payload_from_news(ticker: str, items: list[dict]) -> dict:
    """Render Polygon news items as a single text block — same shape
    StockNet's tweet adapter produces."""
    lines = []
    for it in items:
        title = (it.get("title") or "").strip()
        desc = (it.get("description") or "").strip()
        publisher = ((it.get("publisher") or {}).get("name") or "").strip()
        date_str = (it.get("published_utc") or "")[:10]
        if title:
            head = f"[{date_str}] {publisher}: {title}" if publisher else f"[{date_str}] {title}"
            lines.append(head)
            if desc:
                lines.append(f"  {desc}")
    body = "\n".join(lines).strip()
    return {
        "ticker": ticker,
        "headline": f"polygon news ({len(items)} items)",
        "body": body,
    }


def _build_one_scenario(
    ticker: str, target_date: str,
    *, history_days: int, all_bars: list[dict], all_news: list[dict],
    compute_forecast: bool,
) -> Scenario | None:
    """Build a Scenario for (ticker, target_date) using pre-fetched data.

    Returns None if any of: no target row, no next-day bar, label in
    no-action band, history too short.
    """
    # Index bars by date
    by_date = {b["date"]: b for b in all_bars}
    if target_date not in by_date:
        return None
    sorted_dates = sorted(by_date.keys())
    if target_date not in sorted_dates:
        return None
    idx = sorted_dates.index(target_date)
    if idx + 1 >= len(sorted_dates):
        return None  # need next-day bar for the label
    if idx < history_days:
        return None
    # Direction label: next-day close vs target-day close
    today = by_date[target_date]
    nxt = by_date[sorted_dates[idx + 1]]
    if today.get("close") in (None, 0):
        return None
    pct = (float(nxt["close"]) - float(today["close"])) / float(today["close"])
    label = _label_from_movement(pct)
    if label is None:
        return None  # no-action band, drop
    # History bars: the `history_days` ending on target_date inclusive
    history = [by_date[sorted_dates[j]] for j in range(idx - history_days + 1, idx + 1)]

    # News in the history window
    history_start = sorted_dates[idx - history_days + 1]
    news_for_window = [
        it for it in all_news
        if history_start <= (it.get("published_utc") or "")[:10] <= target_date
    ]
    inputs = [
        RawInput(
            id=f"polygon.{ticker}.{target_date}.ohlcv",
            source="ohlcv",
            payload=_ohlcv_payload_from_bars(history),
        ),
    ]
    if news_for_window:
        inputs.append(RawInput(
            id=f"polygon.{ticker}.{target_date}.press",
            source="press",
            payload=_press_payload_from_news(ticker, news_for_window),
        ))

    forecast_feats = None
    if compute_forecast:
        # Lazy import: only pay the cost if explicitly requested
        from .stocknet_loader import _compute_forecast_features
        forecast_feats = _compute_forecast_features(
            ticker, target_date, history, cache_dir=None,
        )

    return Scenario(
        name=f"polygon_{ticker}_{target_date}",
        inputs=inputs,
        technical_target=emit_synthesis(
            kind="technical", ticker=ticker, bias=label, signal="moderate",
        ),
        fundamental_target=(
            emit_synthesis(kind="fundamental", ticker=ticker, bias=label, signal="moderate")
            if news_for_window
            else emit_synthesis(kind="fundamental", abstain=True)
        ),
        predator_target=emit_prediction(
            ticker=ticker, direction=label, pct_move=pct,
            horizon_min=1440, confidence=0.65, abstain=False,
        ),
        forecaster_features=forecast_feats,
    )


def build_polygon_scenarios(
    *,
    target_window_days: int = 7,           # last N calendar days are targets
    observation_lookback_days: int = 90,   # data fetched back this far
    history_days: int = 5,                 # OHLCV history per scenario
    tickers: list[str] | None = None,
    end_date: str | None = None,           # default: today
    max_per_ticker: int | None = None,
    compute_forecast: bool | None = None,
) -> list[Scenario]:
    """Build a fresh contamination-free benchmark from Polygon.

    Default: last 7 calendar days as TARGETS, with 90 calendar days
    of OHLCV+news lookback for each. ~5 trading days × N tickers
    scenarios out the door.
    """
    if compute_forecast is None:
        compute_forecast = os.environ.get("TROPHIC_COMPUTE_FORECAST", "0") == "1"
    tickers = tickers or DJIA_TOP20

    end = date.fromisoformat(end_date) if end_date else date.today()
    obs_start = end - timedelta(days=observation_lookback_days)
    target_start = end - timedelta(days=target_window_days)

    obs_start_s = obs_start.isoformat()
    end_s = end.isoformat()

    scenarios: list[Scenario] = []
    for ticker in tickers:
        bars = get_bars(ticker, obs_start_s, end_s)
        if not bars:
            continue
        news = get_news(ticker, obs_start_s, end_s)
        # Find target dates: trading days within [target_start, end]
        target_dates = [
            b["date"] for b in bars
            if target_start.isoformat() <= b["date"] <= end_s
        ]
        per_ticker = 0
        for d in target_dates:
            sc = _build_one_scenario(
                ticker, d,
                history_days=history_days, all_bars=bars, all_news=news,
                compute_forecast=compute_forecast,
            )
            if sc is None:
                continue
            scenarios.append(sc)
            per_ticker += 1
            if max_per_ticker and per_ticker >= max_per_ticker:
                break
    return scenarios
