"""Canonical Dow 30 ticker universe for the v4 firehose sweep.

The v3.3 universe was a mixed-quality 20-ticker list defined inline at
`scripts/run_firehose_loop.py:86`. v4 replaces it with the 30 components of
the Dow Jones Industrial Average (as of mid-2026) so the tax-friction story
gets a cleaner test on blue-chip names where the slippage and h60 holding
assumptions hold up.

Membership is frozen here, not pulled from a network source — historical
purity is not the goal; "30 large-cap blue-chips" is. If the index
re-balances mid-sweep, update this file and re-run.

Both shapes are exported:
  - `DOW30`: ordered tuple (stable iteration order; useful for matrix
    builders that index by position).
  - `DOW30_SET`: frozenset (O(1) membership; used by hot-path filters).

`is_dow30()` is case-insensitive — apex models occasionally emit
lowercased tickers in their rationale; downstream filters should not
trip on that.
"""

from __future__ import annotations

from typing import Iterable

# Dow Jones Industrial Average components as of mid-2026.
# Order matches the conventional alphabetical listing.
DOW30: tuple[str, ...] = (
    "AAPL", "AMGN", "AMZN", "AXP", "BA",
    "CAT", "CRM", "CSCO", "CVX", "DIS",
    "DOW", "GS", "HD", "HON", "IBM",
    "INTC", "JNJ", "JPM", "KO", "MCD",
    "MMM", "MRK", "MSFT", "NKE", "PG",
    "TRV", "UNH", "V", "VZ", "WMT",
)

DOW30_SET: frozenset[str] = frozenset(DOW30)


def is_dow30(ticker: str) -> bool:
    """Case-insensitive Dow 30 membership check.

    Returns False for None / empty / non-string inputs rather than
    raising — most call sites are inside per-article filter loops where
    a malformed ticker should just be dropped, not abort the day.
    """
    if not ticker or not isinstance(ticker, str):
        return False
    return ticker.upper() in DOW30_SET


def filter_to_dow30(tickers: Iterable[str]) -> list[str]:
    """Filter an iterable of tickers to only Dow 30 members.

    Preserves input order (modulo case-normalization to uppercase) and
    de-duplicates. Used by watchlist builders and broadcast filters.
    """
    seen: set[str] = set()
    out: list[str] = []
    for t in tickers:
        if not isinstance(t, str):
            continue
        u = t.upper()
        if u in DOW30_SET and u not in seen:
            seen.add(u)
            out.append(u)
    return out
