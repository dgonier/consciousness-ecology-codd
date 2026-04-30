"""Hermetic tests for the StockNet adapters.

All fixtures are inline; no HTTP, no filesystem dependence on the real
StockNet cache (the `stocknet_loader.py` module owns live fetching).
"""
from __future__ import annotations

from trophic.adapters.stocknet import OhlcvNormalizedAdapter, TokenizedTweetAdapter


OHLCV_SAMPLE = (
    "2015-10-01\t0.0124\t0.5\t0.6\t0.45\t0.55\t1000000\n"
    "2015-10-02\t-0.0085\t0.55\t0.58\t0.50\t0.51\t950000\n"
    "2015-10-03\t0.0001\t0.51\t0.53\t0.49\t0.50\t800000\n"
)

TWEETS_SAMPLE = (
    '{"text": ["$", "aapl", "looking", "strong"], '
    '"created_at": "Thu Oct 01 15:00:00 +0000 2015", "user_id_str": "1"}\n'
    '{"text": ["URL", "earnings", "beat"], '
    '"created_at": "Thu Oct 01 16:30:00 +0000 2015", "user_id_str": "2"}\n'
)


def test_ohlcv_adapter_emits_one_raw_input_per_day():
    a = OhlcvNormalizedAdapter("AAPL")
    out = list(a.adapt(OHLCV_SAMPLE))
    assert len(out) == 3
    assert all(r.source == "ohlcv" for r in out)
    assert out[0].payload["ticker"] == "AAPL"
    assert out[0].payload["date"] == "2015-10-01"
    assert abs(out[0].payload["movement_pct"] - 0.0124) < 1e-9
    assert out[0].id == "stocknet.AAPL.2015-10-01.ohlcv"


def test_ohlcv_adapter_skips_malformed_rows():
    raw = (
        "2015-10-01\tnot_a_float\t1\t2\t3\t4\t5\n"
        "broken\n"
        "\n"
        "2015-10-02\t0.01\t0.5\t0.6\t0.45\t0.55\t1000\n"
    )
    a = OhlcvNormalizedAdapter("AAPL")
    out = list(a.adapt(raw))
    assert len(out) == 1
    assert out[0].payload["date"] == "2015-10-02"
    assert out[0].payload["volume"] == 1000.0


def test_tweets_adapter_aggregates_one_day():
    a = TokenizedTweetAdapter("AAPL", "2015-10-01")
    out = list(a.adapt(TWEETS_SAMPLE))
    assert len(out) == 1
    assert out[0].source == "tweets"
    assert out[0].payload["tweet_count"] == 2
    assert "$ aapl" in out[0].payload["body"]
    assert out[0].payload["ticker"] == "AAPL"
    assert out[0].payload["date"] == "2015-10-01"
    assert out[0].id == "stocknet.AAPL.2015-10-01.tweets"


def test_tweets_adapter_empty_day_emits_nothing():
    a = TokenizedTweetAdapter("AAPL", "2015-10-02")
    assert list(a.adapt("")) == []
    assert list(a.adapt("   \n  \n")) == []


def test_tweets_adapter_skips_invalid_json():
    raw = (
        '{"text": ["good", "tweet"], "user_id_str": "1"}\n'
        "this is not json\n"
        '{"text": ["another", "tweet"], "user_id_str": "2"}\n'
    )
    a = TokenizedTweetAdapter("AAPL", "2015-10-01")
    out = list(a.adapt(raw))
    assert len(out) == 1
    assert out[0].payload["tweet_count"] == 2


def test_adapter_source_tags_in_vocab():
    from trophic.adapters import SOURCE_TAGS_VOCAB

    assert OhlcvNormalizedAdapter.SOURCE_TAGS <= SOURCE_TAGS_VOCAB
    assert TokenizedTweetAdapter.SOURCE_TAGS <= SOURCE_TAGS_VOCAB
    assert "ohlcv" in OhlcvNormalizedAdapter.SOURCE_TAGS
    assert "tweets" in TokenizedTweetAdapter.SOURCE_TAGS
