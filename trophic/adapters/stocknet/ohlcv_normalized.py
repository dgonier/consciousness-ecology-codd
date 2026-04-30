"""StockNet preprocessed OHLCV adapter.

File format (per the StockNet GitHub repo):
  <date>\t<movement_pct>\t<open_norm>\t<high_norm>\t<low_norm>\t<close_norm>\t<volume>

One row per trading day. Price columns are normalized; volume is raw.

This adapter takes a single ticker's preprocessed file (as text) and emits
one RawInput per day with source="ohlcv" and a payload that mirrors what
the existing scenarios.py _ohlcv_payload(...) emits, so downstream
producers don't need to special-case StockNet.
"""
from __future__ import annotations

from typing import Iterable

from ...types import RawInput
from ..base import Adapter


class OhlcvNormalizedAdapter(Adapter):
    SOURCE_TAGS = {"ohlcv"}

    def __init__(self, ticker: str):
        self.ticker = ticker

    def adapt(self, raw: str) -> Iterable[RawInput]:
        """`raw` is the entire text of a StockNet preprocessed price file."""
        for line in raw.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 7:
                continue
            date_s = parts[0]
            try:
                payload = {
                    "ticker": self.ticker,
                    "date": date_s,
                    "movement_pct": float(parts[1]),
                    "open_norm": float(parts[2]),
                    "high_norm": float(parts[3]),
                    "low_norm": float(parts[4]),
                    "close_norm": float(parts[5]),
                    "volume": float(parts[6]),
                    "source": "stocknet_normalized",
                }
            except ValueError:
                continue
            yield RawInput(
                id=f"stocknet.{self.ticker}.{date_s}.ohlcv",
                source="ohlcv",
                payload=payload,
            )
