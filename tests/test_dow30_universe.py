"""Phase 3-A-05a (v4): Dow 30 universe filter contract.

Tests that the v4 universe swap correctly:

  - Defines a 30-ticker DOW30 tuple with case-insensitive membership helpers.
  - Filters watchlist construction to Dow 30 names when --universe dow30.
  - Filters producer broadcasts (news articles) so non-Dow30 articles
    do not reach the apex.
  - Restricts `build_future_returns_matrix` to the active universe so
    oracle never sees out-of-universe tickers.
  - Preserves v3.3 behavior under --universe legacy.

The implementation is data-layer only — no apex prompts, validators, or
signature changes (those belong to other 05 sub-missions).
"""
from __future__ import annotations

import importlib
import sys

import pytest


# ── DOW30 constant + helpers ────────────────────────────────────────────


def test_dow30_constant_is_30_unique():
    from trophic.data.dow30_universe import DOW30, DOW30_SET
    assert len(DOW30) == 30, f"expected 30 tickers, got {len(DOW30)}"
    assert len(set(DOW30)) == 30, "duplicates found in DOW30 tuple"
    assert len(DOW30_SET) == 30, "DOW30_SET membership mismatch"
    # All entries are uppercase ASCII tickers.
    for t in DOW30:
        assert isinstance(t, str) and t.isupper() and t.isascii() and 1 <= len(t) <= 5, \
            f"malformed ticker in DOW30: {t!r}"


def test_is_dow30_case_insensitive_and_safe():
    from trophic.data.dow30_universe import is_dow30
    assert is_dow30("AAPL")
    assert is_dow30("aapl")
    assert is_dow30("Aapl")
    assert not is_dow30("TSLA")
    assert not is_dow30("")
    assert not is_dow30(None)  # type: ignore[arg-type]
    assert not is_dow30(123)   # type: ignore[arg-type]


def test_filter_to_dow30_preserves_order_dedupes_uppercases():
    from trophic.data.dow30_universe import filter_to_dow30
    out = filter_to_dow30(["AAPL", "TSLA", "msft", "AAPL", "INTC"])
    assert out == ["AAPL", "MSFT", "INTC"], out
    assert filter_to_dow30([]) == []
    # Non-string inputs are dropped silently.
    assert filter_to_dow30(["AAPL", None, 5, "GS"]) == ["AAPL", "GS"]  # type: ignore[list-item]


# ── Watchlist construction filters to Dow 30 ────────────────────────────


def _import_runner_fresh():
    """Re-import the runner so each test sees a clean module-level
    UNIVERSE without other tests poisoning the binding."""
    mod_name = "scripts.run_firehose_loop"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


def test_watchlist_filters_to_dow30():
    """A mixed-universe watchlist input — when fed through the Dow 30
    filter — keeps only Dow 30 names. This mirrors how the broadcast
    filter at the news-classification site decides which articles
    reach the apex."""
    from trophic.data.dow30_universe import filter_to_dow30
    raw_watchlist = [
        "AAPL", "MSFT", "TSLA", "NVDA", "AMZN",  # 3 dow + 2 non-dow
        "META", "GOOG", "GS", "JPM", "PLTR",      # 3 non-dow + 2 dow
    ]
    filtered = filter_to_dow30(raw_watchlist)
    assert set(filtered).issubset(set(_dow30_set())), \
        f"non-dow30 tickers slipped through: {set(filtered) - set(_dow30_set())}"
    assert "AAPL" in filtered and "MSFT" in filtered and "AMZN" in filtered
    assert "JPM" in filtered and "GS" in filtered
    assert "TSLA" not in filtered and "NVDA" not in filtered
    assert "META" not in filtered and "GOOG" not in filtered and "PLTR" not in filtered


def _dow30_set():
    from trophic.data.dow30_universe import DOW30
    return set(DOW30)


# ── Producer broadcast filter ───────────────────────────────────────────


def test_producer_broadcasts_filtered():
    """5 Dow 30-tagged + 5 non-Dow 30-tagged articles → only the 5
    in-universe broadcasts pass the universe filter."""
    runner = _import_runner_fresh()
    from trophic.data.dow30_universe import DOW30
    in_universe = list(DOW30[:5])
    out_of_universe = ["TSLA", "NVDA", "META", "GOOG", "PLTR"]
    news = []
    for tk in in_universe:
        news.append({
            "id": f"a-{tk}", "title": f"{tk} earnings beat",
            "tickers": [tk], "all_tickers": [tk],
        })
    for tk in out_of_universe:
        news.append({
            "id": f"a-{tk}", "title": f"{tk} something",
            "tickers": [tk], "all_tickers": [tk],
        })

    kept = runner.filter_news_to_universe(news, DOW30)
    assert len(kept) == 5, f"expected 5 in-universe articles, got {len(kept)}"
    kept_tickers = {art["tickers"][0] for art in kept}
    assert kept_tickers == set(in_universe), \
        f"unexpected kept tickers: {kept_tickers}"


def test_producer_broadcasts_pass_when_any_ticker_in_universe():
    """Mixed-ticker article (1 in-universe + 1 out-of-universe) passes
    untouched — we don't tamper with article bodies."""
    runner = _import_runner_fresh()
    from trophic.data.dow30_universe import DOW30
    mixed = {
        "id": "mix", "title": "AAPL and TSLA both moved",
        "tickers": ["AAPL", "TSLA"], "all_tickers": ["AAPL", "TSLA"],
    }
    kept = runner.filter_news_to_universe([mixed], DOW30)
    assert len(kept) == 1
    # body unchanged, including the out-of-universe TSLA tag
    assert kept[0]["tickers"] == ["AAPL", "TSLA"]


def test_producer_broadcasts_untagged_pass_through():
    """Untagged articles (no `tickers` field) are broad-market
    commentary; the universe filter lets them through."""
    runner = _import_runner_fresh()
    from trophic.data.dow30_universe import DOW30
    untagged = {"id": "u", "title": "Fed minutes released", "tickers": []}
    kept = runner.filter_news_to_universe([untagged], DOW30)
    assert len(kept) == 1


# ── Future-returns matrix filter ────────────────────────────────────────


def test_matrix_universe_filter():
    """`build_future_returns_matrix` restricts to the universe passed in.
    Dataset rows may carry per-row targets for tickers outside the
    universe; the matrix builder must ignore them."""
    runner = _import_runner_fresh()
    from trophic.data.dow30_universe import DOW30
    # Build a tiny 3-day dataset where every row's `targets` includes
    # both AAPL (in DOW30) and TSLA (out).
    rows = [
        {"date": "2026-01-02", "targets": [
            {"ticker": "AAPL", "actual_return": 0.01},
            {"ticker": "TSLA", "actual_return": 0.05},
            {"ticker": "MSFT", "actual_return": 0.02},
        ]},
        {"date": "2026-01-03", "targets": [
            {"ticker": "AAPL", "actual_return": -0.02},
            {"ticker": "TSLA", "actual_return": 0.10},
            {"ticker": "MSFT", "actual_return": -0.005},
        ]},
        {"date": "2026-01-04", "targets": [
            {"ticker": "AAPL", "actual_return": 0.003},
            {"ticker": "TSLA", "actual_return": -0.04},
            {"ticker": "MSFT", "actual_return": 0.001},
        ]},
    ]
    # Request the matrix as-of day-0 (today_idx=0) with 3 days remaining.
    # The matrix should only contain AAPL + MSFT (DOW30 members), never TSLA.
    universe_dow30_only = ["AAPL", "MSFT"]  # subset of DOW30 we expect downstream
    matrix, pruned = runner.build_future_returns_matrix(
        today_idx=0,
        rows=rows,
        universe=universe_dow30_only,
        days_remaining=3,
    )
    tickers_in_matrix = {m.ticker for m in matrix}
    assert "TSLA" not in tickers_in_matrix, \
        f"TSLA leaked into matrix: {tickers_in_matrix}"
    assert tickers_in_matrix.issubset(set(universe_dow30_only)), \
        f"non-universe tickers in matrix: {tickers_in_matrix - set(universe_dow30_only)}"
    # Both AAPL and MSFT should have forward returns from the 2 future days.
    for m in matrix:
        assert len(m.forward_returns) == 2, \
            f"{m.ticker} matrix should have 2 entries (today+1, today+2)"


# ── Universe selection flag ─────────────────────────────────────────────


def test_universe_flag_choices():
    """argparse exposes --universe with dow30 + legacy choices."""
    runner = _import_runner_fresh()
    # We can't easily invoke main() (it needs Neo4j etc.), but we can
    # confirm the argparse choices by reading the source.
    src = open(runner.__file__).read()
    assert "--universe" in src, "missing --universe flag"
    assert "dow30" in src and "legacy" in src, "missing universe choices"
    # Default is dow30 (v4 default).
    assert "default=\"dow30\"" in src or "default='dow30'" in src, \
        "expected dow30 as default --universe value"


def test_legacy_universe_constant_unchanged():
    """The v3.3 legacy universe must remain available verbatim under
    --universe legacy for back-compat with v3.3-era sweeps."""
    runner = _import_runner_fresh()
    expected_v33 = [
        "AAPL", "ABBV", "AMZN", "AVGO", "CSCO", "CVX", "GOOG", "HD", "JNJ", "JPM",
        "KO", "MA", "MCD", "MRK", "MSFT", "PEP", "PG", "UNH", "V", "WMT",
    ]
    assert hasattr(runner, "LEGACY_UNIVERSE"), \
        "runner module should expose LEGACY_UNIVERSE"
    assert list(runner.LEGACY_UNIVERSE) == expected_v33, \
        "LEGACY_UNIVERSE drifted from the v3.3 ticker list"


def test_default_universe_at_module_load_is_dow30():
    """Module-level UNIVERSE defaults to DOW30 before main() runs."""
    runner = _import_runner_fresh()
    from trophic.data.dow30_universe import DOW30
    assert set(runner.UNIVERSE) == set(DOW30), \
        "module-level UNIVERSE should default to DOW30"


# ── Sanity: imports compose ─────────────────────────────────────────────


def test_dow30_importable_from_canonical_path():
    """`from trophic.data.dow30_universe import DOW30` is the contract
    advertised in the phase 3 MESSAGES handoff. Downstream 05e wiring
    depends on this import path."""
    from trophic.data.dow30_universe import DOW30  # noqa: F401
