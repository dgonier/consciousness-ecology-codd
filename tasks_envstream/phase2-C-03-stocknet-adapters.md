# Mission 03: stocknet-adapters

**Handle**: `phase2-C`
**Phase**: 2 (parallel after `phase1-A:01`)
**Mission file**: `phase2-C-03-stocknet-adapters.md`
**Dependencies**: `phase1-A:01:DONE`
**Blocks**: `phase3-A:05`

---

## Before You Start

```bash
cd /home/dgonier/ecology_experiment/trophic && \
  grep -nE '@(all|phase2-C|phase2)' tasks_envstream/scratchpad.md
```

Confirm `phase1-A:01:DONE`. Then atomically flip:
```
phase2-C:03:PENDING → phase2-C:03:RUNNING
```

---

## Goal

Implement two concrete `Adapter` subclasses for the StockNet (ACL-18) dataset:

1. **`OhlcvNormalizedAdapter`** — converts StockNet's preprocessed price file (`./price/preprocessed/<TICKER>.txt`, format: `date\tmovement_pct\topen\thigh\tlow\tclose\tvolume`) into a stream of `RawInput(source="ohlcv", payload={...})`.
2. **`TokenizedTweetAdapter`** — converts StockNet's preprocessed per-day tweet JSONL files (`./tweet/preprocessed/<TICKER>/<DATE>`, one tokenized tweet per line) into a stream of `RawInput(source="tweets", payload={...})`.

These are the first concrete adapters. Future adapters (Bloomberg, Polygon, scraped 8-K) follow the same pattern.

---

## Files to Create / Modify

### Create
- `trophic/adapters/stocknet/__init__.py` — re-exports both adapters
- `trophic/adapters/stocknet/ohlcv_normalized.py` — `OhlcvNormalizedAdapter`
- `trophic/adapters/stocknet/tokenized_tweet.py` — `TokenizedTweetAdapter`
- `tests/test_stocknet_adapters.py` — at least 4 tests (2 per adapter)

### Don't touch
- `trophic/training/stocknet_loader.py` — has TODO markers from phase1-A; **leave it alone**. Phase3-A integrates.
- `trophic/agents/producer.py` — that's phase2-D
- `trophic/environment_stream.py` — phase2-B

---

## Implementation Steps

### Step 1: `OhlcvNormalizedAdapter`

```python
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
        for line_no, line in enumerate(raw.strip().split("\n")):
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
```

### Step 2: `TokenizedTweetAdapter`

```python
"""StockNet preprocessed tweet adapter.

File format: one JSON object per line, fields:
  - text: list[str]  (tokens)
  - user_id_str: str
  - created_at: str  (Twitter timestamp)

Each per-(ticker, date) file is one day's tweets. The adapter takes the
full text of a single day's file and emits one RawInput aggregating the
day's tweets (since downstream producers want day-shaped input).
"""
from __future__ import annotations

import json
from typing import Iterable

from ...types import RawInput
from ..base import Adapter


class TokenizedTweetAdapter(Adapter):
    SOURCE_TAGS = {"tweets"}

    def __init__(self, ticker: str, date: str):
        self.ticker = ticker
        self.date = date

    def adapt(self, raw: str) -> Iterable[RawInput]:
        """`raw` is the entire text of a single (ticker, date) tweet JSONL file."""
        tweets: list[str] = []
        for line in raw.strip().split("\n"):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tokens = obj.get("text", [])
            if isinstance(tokens, list):
                tweets.append(" ".join(tokens))

        if not tweets:
            return  # zero-emit on empty days

        yield RawInput(
            id=f"stocknet.{self.ticker}.{self.date}.tweets",
            source="tweets",
            payload={
                "ticker": self.ticker,
                "date": self.date,
                "tweet_count": len(tweets),
                "body": "\n".join(tweets)[:8000],  # cap to keep token budget reasonable
                "source": "stocknet_tweets",
            },
        )
```

### Step 3: Package `__init__.py`

```python
# trophic/adapters/stocknet/__init__.py
from .ohlcv_normalized import OhlcvNormalizedAdapter
from .tokenized_tweet import TokenizedTweetAdapter

__all__ = ["OhlcvNormalizedAdapter", "TokenizedTweetAdapter"]
```

### Step 4: Tests

Use small inline fixtures rather than hitting the real StockNet repo from tests (the existing `stocknet_loader.py` already has cache-on-demand for live use). Tests should be hermetic.

```python
# tests/test_stocknet_adapters.py
import pytest

from trophic.adapters.stocknet import OhlcvNormalizedAdapter, TokenizedTweetAdapter

OHLCV_SAMPLE = """\
2015-10-01\t0.0124\t0.5\t0.6\t0.45\t0.55\t1000000
2015-10-02\t-0.0085\t0.55\t0.58\t0.50\t0.51\t950000
2015-10-03\t0.0001\t0.51\t0.53\t0.49\t0.50\t800000
"""

TWEETS_SAMPLE = """\
{"text": ["$", "aapl", "looking", "strong"], "created_at": "Thu Oct 01 15:00:00 +0000 2015", "user_id_str": "1"}
{"text": ["URL", "earnings", "beat"], "created_at": "Thu Oct 01 16:30:00 +0000 2015", "user_id_str": "2"}
"""


def test_ohlcv_adapter_emits_one_raw_input_per_day():
    a = OhlcvNormalizedAdapter("AAPL")
    out = list(a.adapt(OHLCV_SAMPLE))
    assert len(out) == 3
    assert all(r.source == "ohlcv" for r in out)
    assert out[0].payload["ticker"] == "AAPL"
    assert out[0].payload["date"] == "2015-10-01"
    assert abs(out[0].payload["movement_pct"] - 0.0124) < 1e-9


def test_ohlcv_adapter_skips_malformed_rows():
    raw = "2015-10-01\tnot_a_float\t1\t2\t3\t4\t5\nbroken\n2015-10-02\t0.01\t0.5\t0.6\t0.45\t0.55\t1000\n"
    a = OhlcvNormalizedAdapter("AAPL")
    out = list(a.adapt(raw))
    assert len(out) == 1
    assert out[0].payload["date"] == "2015-10-02"


def test_tweets_adapter_aggregates_one_day():
    a = TokenizedTweetAdapter("AAPL", "2015-10-01")
    out = list(a.adapt(TWEETS_SAMPLE))
    assert len(out) == 1
    assert out[0].source == "tweets"
    assert out[0].payload["tweet_count"] == 2
    assert "$ aapl" in out[0].payload["body"]


def test_tweets_adapter_empty_day_emits_nothing():
    a = TokenizedTweetAdapter("AAPL", "2015-10-02")
    out = list(a.adapt(""))
    assert out == []


def test_adapter_source_tags_in_vocab():
    from trophic.adapters import SOURCE_TAGS_VOCAB
    assert OhlcvNormalizedAdapter.SOURCE_TAGS <= SOURCE_TAGS_VOCAB
    assert TokenizedTweetAdapter.SOURCE_TAGS <= SOURCE_TAGS_VOCAB
```

---

## Acceptance Criteria

- [ ] `trophic/adapters/stocknet/{ohlcv_normalized,tokenized_tweet}.py` exist
- [ ] Both adapters subclass `Adapter` and declare `SOURCE_TAGS`
- [ ] At least 4 new tests in `tests/test_stocknet_adapters.py`
- [ ] All existing tests still pass

---

## Testing Conditions (exit verification)

1. **New adapter tests pass**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     .venv/bin/python -m pytest tests/test_stocknet_adapters.py -v 2>&1 | tail -15
   ```
   **Expected**: ≥4 passed, 0 failed.

2. **Whole suite still green**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     .venv/bin/python -m pytest tests/ --tb=line -q 2>&1 | tail -5
   ```
   **Expected**: 65+ passed (51 baseline + phase1-A's 4 + phase2-B's 6 + your 4), 0 failed.

3. **Adapter wavelength validation**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && .venv/bin/python -c "
   from trophic.adapters.stocknet import OhlcvNormalizedAdapter, TokenizedTweetAdapter
   from trophic.adapters import SOURCE_TAGS_VOCAB
   for cls in (OhlcvNormalizedAdapter, TokenizedTweetAdapter):
       assert cls.SOURCE_TAGS <= SOURCE_TAGS_VOCAB, f'{cls.__name__} has unknown tags'
       print(f'{cls.__name__}: {cls.SOURCE_TAGS} ✓')
   "
   ```
   **Expected**: both classes print without assertion error.

4. **Round-trip from real StockNet cache (only if cache exists)**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && .venv/bin/python -c "
   from pathlib import Path
   from trophic.adapters.stocknet import OhlcvNormalizedAdapter
   p = Path('external/stocknet_cache/price__preprocessed__AAPL.txt')
   if p.exists():
       text = p.read_text()
       out = list(OhlcvNormalizedAdapter('AAPL').adapt(text))
       print(f'AAPL OHLCV rows adapted: {len(out)}')
       print(f'first: date={out[0].payload[\"date\"]}, movement={out[0].payload[\"movement_pct\"]}')
   else:
       print('cache not present; skipping live round-trip')
   "
   ```
   **Expected**: prints a row count > 0 if cache present; otherwise prints the skip message.

---

## Coordination

- You do NOT touch `trophic/training/stocknet_loader.py` or `trophic/training/scenarios.py`. They have phase1-A TODO markers; **phase3-A is the integration owner**.
- You do NOT touch `trophic/agents/producer.py` — phase2-D's territory.
- Your work is hermetic: import `from trophic.adapters import Adapter, SOURCE_TAGS_VOCAB` and emit `RawInput` objects. Don't import from anywhere else in the project.
- If the test in step (2) above counts fewer than 65 tests, phase2-B may not have landed yet — wait or coordinate.

---

## Out of Scope

- Live HTTP fetching from GitHub raw URLs — `stocknet_loader.py` has that already; adapters consume already-loaded text
- Other adapters (Bloomberg, Yahoo, Polygon) — future work
- Caching — same: not the adapter's responsibility

---

## When Done

1. Re-run inbox grep.
2. Update STATUS: `phase2-C:03:RUNNING` → `phase2-C:03:DONE`.
3. Append MESSAGES:
   ```
   - [<YYYY-MM-DD HH:MM>] phase2-C > @phase3-A: StockNet adapters landed at trophic/adapters/stocknet/. Public API: from trophic.adapters.stocknet import OhlcvNormalizedAdapter, TokenizedTweetAdapter. test_stocknet_adapters.py: <PASS>/<TOTAL>. Full suite: <PASS>/<TOTAL>. Open questions for phase3-A: how should empty-tweet days flow through EnvironmentStream — deposit nothing, or deposit a "silence" marker? Suggest deposit-nothing initially.
   ```
