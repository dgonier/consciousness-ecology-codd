"""Thin Polygon.io REST client. Lifted and trimmed from StockBench
(github.com/ChenYXxxx/stockbench/blob/main/stockbench/adapters/polygon_client.py)
under Apache-2.0.

Only the two endpoints we need: list_aggs (OHLCV) and list_ticker_news.
"""
from __future__ import annotations

import os
import random
import time
from typing import Dict, List, Optional, Tuple

import httpx


class PolygonError(Exception):
    def __init__(self, status_code: int, message: str = ""):
        super().__init__(f"PolygonError {status_code}: {message}")
        self.status_code = status_code


class PolygonClient:
    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY", "")
        self._client: Optional[httpx.Client] = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.BASE_URL, timeout=30.0)
        return self._client

    def _request(self, method: str, path: str, params: Optional[Dict] = None) -> Dict:
        params = dict(params or {})
        params.setdefault("apiKey", self.api_key)
        url = path if path.startswith("http") else f"{self.BASE_URL}{path}"
        client = self._get_client()
        backoff = 0.5
        for attempt in range(6):
            try:
                resp = client.request(method, url, params=params)
                if resp.status_code in (429,) or resp.status_code >= 500:
                    sleep_s = min(backoff * (2 ** attempt), 30.0) * (0.8 + 0.4 * random.random())
                    time.sleep(sleep_s)
                    continue
                if 400 <= resp.status_code < 500:
                    raise PolygonError(resp.status_code, resp.text[:300])
                resp.raise_for_status()
                return resp.json()
            except httpx.RequestError:
                sleep_s = min(backoff * (2 ** attempt), 30.0) * (0.8 + 0.4 * random.random())
                time.sleep(sleep_s)
                continue
        return {}

    def list_aggs(
        self, ticker: str, start: str, end: str,
        multiplier: int = 1, timespan: str = "day", adjusted: bool = True,
    ) -> List[Dict]:
        """Daily (or other) OHLCV bars between start and end (inclusive,
        ISO yyyy-mm-dd). Polygon caps at 50000 results, paginates via
        next_url."""
        path = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
        params = {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 50000}
        results: List[Dict] = []
        url = path
        cursor: Optional[str] = None
        while True:
            if cursor:
                params["cursor"] = cursor
            data = self._request("GET", url, params)
            if not data:
                break
            results.extend(data.get("results") or [])
            next_url = data.get("next_url")
            if not next_url:
                break
            url = next_url
            params = {"apiKey": self.api_key}
        return results

    def list_ticker_news(
        self, ticker: str, gte: str, lte: str, limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Tuple[List[Dict], Optional[str]]:
        """Polygon news for ticker in [gte, lte] (ISO timestamps).
        Returns (items, next_cursor). One page per call; caller paginates
        by passing the returned cursor back in."""
        path = "/v2/reference/news"
        params: Dict = {
            "ticker": ticker,
            "limit": min(limit, 1000),
            "order": "desc",
            "published_utc.gte": gte,
            "published_utc.lte": lte,
        }
        if cursor:
            params["cursor"] = cursor
        data = self._request("GET", path, params)
        items = data.get("results") or []
        next_url = data.get("next_url")
        next_cursor = None
        if next_url:
            from urllib.parse import urlparse, parse_qs
            qp = parse_qs(urlparse(next_url).query)
            next_cursor = qp.get("cursor", [None])[0]
        return items, next_cursor
