"""Serper.dev research tool — Google SERP scraper with date filtering.

Replaces GoogleCSEResearchTool for this project because Custom Search JSON
API is closed to new GCP customers (EOL 2027). Serper bypasses the
Custom Search API entirely by scraping the public Google SERP via its
own infrastructure, exposing the same `tbs=cdr:1,cd_min,cd_max` date
parameter that Google's UI supports.

Auth: SERPER_API_KEY env var. https://serper.dev — $1 per 1000 searches,
2500 free credits on signup.

Date-bound by passing `tbs=cdr:1,cd_min:MM/DD/YYYY,cd_max:MM/DD/YYYY`.
This is the same syntax Google's "Tools → Any time → Custom range" UI
uses, and Serper passes it through to Google verbatim. Verified to
return historical results for benchmark dates (2015-2016).
"""
from __future__ import annotations

import datetime as _dt
import json
import os

from .base import ResearchSnippet, ResearchTool


SERPER_ENDPOINT = "https://google.serper.dev/search"


class SerperResearchTool(ResearchTool):
    """Serper.dev SERP scraper with absolute-date windowing.

    `as_of_date` (ISO yyyy-mm-dd) → search restricted to a window
    ending on that date. Default window is `window_days` days back.
    For benchmark eval this prevents forward-looking leakage.
    """

    name = "serper"

    def __init__(
        self,
        api_key: str | None = None,
        window_days: int = 7,
        gl: str = "us",
        hl: str = "en",
    ):
        self.api_key = api_key or os.environ.get("SERPER_API_KEY")
        self.window_days = window_days
        self.gl = gl  # geo
        self.hl = hl  # lang

    def is_available(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _tbs_window(as_of_date: str, window_days: int) -> str | None:
        """Return the `tbs=cdr:1,cd_min:MM/DD/YYYY,cd_max:MM/DD/YYYY`
        param value spanning `window_days` ending on as_of_date. None if
        the date isn't parseable."""
        try:
            d = _dt.date.fromisoformat(as_of_date)
        except (ValueError, TypeError):
            return None
        cd_max = d.strftime("%m/%d/%Y")
        cd_min = (d - _dt.timedelta(days=window_days)).strftime("%m/%d/%Y")
        return f"cdr:1,cd_min:{cd_min},cd_max:{cd_max}"

    def query(
        self,
        ticker: str,
        as_of_date: str,
        focus: str = "",
        n_results: int = 5,
    ) -> list[ResearchSnippet]:
        if not self.is_available():
            return []
        # Lazy-import requests; if unavailable fall back to urllib.
        try:
            import requests  # type: ignore
        except ImportError:
            requests = None

        q_parts = [f'"{ticker}"']
        if focus:
            q_parts.append(focus)
        else:
            q_parts.append("stock news")
        q = " ".join(q_parts)

        payload: dict = {
            "q": q,
            "num": min(max(n_results, 1), 10),
            "gl": self.gl,
            "hl": self.hl,
        }
        tbs = self._tbs_window(as_of_date, self.window_days)
        if tbs:
            payload["tbs"] = tbs

        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        try:
            if requests is not None:
                r = requests.post(
                    SERPER_ENDPOINT, headers=headers, json=payload, timeout=15,
                )
                r.raise_for_status()
                data = r.json()
            else:
                from urllib.request import Request, urlopen
                req = Request(
                    SERPER_ENDPOINT,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[serper] query error: {e}")
            return []

        organic = data.get("organic", []) or []
        snippets: list[ResearchSnippet] = []
        for item in organic[:n_results]:
            # Serper returns 'date' as a relative string ("2 days ago") or
            # absolute. Pass through whatever's there; the analyst can read
            # context from the surrounding tbs window.
            snippets.append(ResearchSnippet(
                source=item.get("source") or self._domain(item.get("link", "")),
                url=item.get("link"),
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                date=item.get("date"),
            ))
        return snippets

    @staticmethod
    def _domain(url: str) -> str:
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc or "serper"
        except Exception:
            return "serper"
