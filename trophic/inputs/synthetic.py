"""Synthetic fintech feed generator.

Emits RawInput batches across several source types so the three producers
have something distinct to attract to. Tunable mix lets demos stress
specific dynamics (e.g. crank up filings to feed the FundamentalHerbivore).

Sources:
  - "ohlcv":  OHLCV bar (TickDelta + Anomaly attract)
  - "trades": individual prints (TickDelta attracts)
  - "book":   order-book snapshot (TickDelta attracts)
  - "filing": SEC filing summary (Disclosure attracts)
  - "press":  company press release (Disclosure attracts)
  - "options":options-flow event (Anomaly attracts)
  - "halt":   trading halt notice (Anomaly attracts)
"""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field

from ..types import RawInput


TICKERS = ["AAPL", "NVDA", "TSLA", "AMD", "GOOG", "MSFT", "META", "JPM"]
FILING_TYPES = ["8-K", "10-Q", "13D", "S-1"]
ANOMALY_TYPES = ["volume_spike", "price_gap", "spread_blowout", "halt", "options_skew"]


@dataclass
class FeedMix:
    """Probability weights per source type. Need not sum to 1 (renormalized)."""
    ohlcv: float = 3.0
    trades: float = 2.0
    book: float = 1.0
    filing: float = 1.0
    press: float = 0.7
    options: float = 1.0
    halt: float = 0.2
    quote_series: float = 1.5  # rolling history windows for the forecaster path

    def items(self) -> list[tuple[str, float]]:
        return [
            ("ohlcv", self.ohlcv),
            ("trades", self.trades),
            ("book", self.book),
            ("filing", self.filing),
            ("press", self.press),
            ("options", self.options),
            ("halt", self.halt),
            ("quote_series", self.quote_series),
        ]


@dataclass
class SyntheticFeed:
    seed: int = 7
    mix: FeedMix = field(default_factory=FeedMix)
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def emit(self, n: int) -> list[RawInput]:
        out: list[RawInput] = []
        sources, weights = zip(*self.mix.items())
        for _ in range(n):
            src = self._rng.choices(sources, weights=weights, k=1)[0]
            ticker = self._rng.choice(TICKERS)
            out.append(self._build(src, ticker))
        return out

    def _build(self, src: str, ticker: str) -> RawInput:
        if src == "ohlcv":
            o = round(self._rng.uniform(50, 500), 2)
            ch = self._rng.uniform(-0.05, 0.05)
            payload = {
                "ticker": ticker,
                "open": o, "high": round(o * (1 + abs(ch) + 0.001), 2),
                "low": round(o * (1 - abs(ch) - 0.001), 2),
                "close": round(o * (1 + ch), 2),
                "volume": int(self._rng.uniform(1e5, 1e7)),
                "window": "1m",
            }
        elif src == "trades":
            payload = {
                "ticker": ticker,
                "price": round(self._rng.uniform(50, 500), 2),
                "size": int(self._rng.uniform(100, 100000)),
                "side": self._rng.choice(["B", "S"]),
            }
        elif src == "book":
            mid = round(self._rng.uniform(50, 500), 2)
            payload = {
                "ticker": ticker,
                "bid": round(mid - 0.05, 2), "bid_size": self._rng.randint(100, 5000),
                "ask": round(mid + 0.05, 2), "ask_size": self._rng.randint(100, 5000),
            }
        elif src == "filing":
            payload = {
                "ticker": ticker,
                "filing_type": self._rng.choice(FILING_TYPES),
                "title": f"{ticker} files {self._rng.choice(FILING_TYPES)} — material event",
                "summary": self._rng.choice([
                    "Reports unexpected guidance reduction citing supply constraints.",
                    "Announces strategic acquisition pending regulatory review.",
                    "Discloses cybersecurity incident with limited impact.",
                    "Provides updated full-year revenue outlook above consensus.",
                    "Reports material weakness in internal controls.",
                ]),
            }
        elif src == "press":
            payload = {
                "ticker": ticker,
                "headline": self._rng.choice([
                    f"{ticker} announces new flagship product line",
                    f"{ticker} CFO to depart at end of quarter",
                    f"{ticker} secures multi-year supply agreement",
                    f"{ticker} expands data center capacity",
                ]),
            }
        elif src == "options":
            payload = {
                "ticker": ticker,
                "strike": round(self._rng.uniform(50, 600), 0),
                "expiry_days": self._rng.choice([1, 7, 30, 90]),
                "side": self._rng.choice(["C", "P"]),
                "premium": round(self._rng.uniform(0.5, 25), 2),
                "size": self._rng.randint(10, 5000),
                "iv": round(self._rng.uniform(0.15, 1.2), 2),
            }
        elif src == "halt":
            payload = {
                "ticker": ticker,
                "reason": self._rng.choice(["LULD", "T1", "T12", "regulatory"]),
            }
        elif src == "quote_series":
            # Rolling history of N closes; what a forecaster wants.
            n = 64
            base = round(self._rng.uniform(50, 500), 2)
            slope = self._rng.uniform(-0.05, 0.05)
            noise_amp = base * 0.005
            history = []
            for i in range(n):
                history.append(round(
                    base + slope * i + self._rng.uniform(-noise_amp, noise_amp), 3
                ))
            payload = {
                "ticker": ticker,
                "window": "1m",
                "history": history,
                "horizon": 12,
            }
        else:
            payload = {"ticker": ticker}
        return RawInput(id=str(uuid.uuid4()), source=src, payload=payload)
