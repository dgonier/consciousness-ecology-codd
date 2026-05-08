"""Polygon.io adapter — date-bounded OHLCV bars + ticker news.

Replaces the StockNet (2014-2016) corpus for tool-augmented analyst
species. Polygon's API serves OHLCV from 2003+ and news from ~2020+,
all server-side date-bounded so there's no leakage risk regardless
of how the analyst formulates a query.

Free tier limits (as of 2026): 5 calls/minute on /v2/aggs and
/v2/reference/news. The cache layer below makes those limits a
non-issue past first fetch.
"""
from .client import PolygonClient
from .cache import PolygonCache, get_bars, get_news

__all__ = ["PolygonClient", "PolygonCache", "get_bars", "get_news"]
