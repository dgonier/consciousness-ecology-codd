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
                tweets.append(" ".join(str(t) for t in tokens))

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
