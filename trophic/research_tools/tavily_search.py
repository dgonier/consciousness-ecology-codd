"""Tavily research tool.

Date-bounded via Tavily's `topic="news"` + `start_date`/`end_date` params
(YYYY-MM-DD). The general (non-news) endpoint ignores date filters for
older content, so historical StockNet windows must use topic=news.

Auth: TAVILY_API_KEY env var.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from urllib.request import Request, urlopen

from .base import ResearchTool, ResearchSnippet


TAVILY_ENDPOINT = "https://api.tavily.com/search"


class TavilyResearchTool(ResearchTool):
    name = "tavily"

    def __init__(
        self,
        api_key: str | None = None,
        window_days: int = 5,
        include_domains: list[str] | None = None,
    ):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.window_days = window_days
        self.include_domains = include_domains

    def is_available(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _date_window(as_of_date: str, window_days: int) -> tuple[str, str] | None:
        try:
            d = _dt.date.fromisoformat(as_of_date)
        except (ValueError, TypeError):
            return None
        end = d.isoformat()
        start = (d - _dt.timedelta(days=window_days)).isoformat()
        return start, end

    def query(
        self,
        ticker: str,
        as_of_date: str,
        focus: str = "",
        n_results: int = 5,
    ) -> list[ResearchSnippet]:
        if not self.is_available():
            return []

        q_parts = [ticker]
        q_parts.append(focus if focus else "stock news")
        q = " ".join(q_parts)

        body: dict = {
            "api_key": self.api_key,
            "query": q,
            "topic": "news",
            "max_results": min(max(n_results, 1), 20),
        }
        window = self._date_window(as_of_date, self.window_days)
        if window:
            body["start_date"], body["end_date"] = window
        if self.include_domains:
            body["include_domains"] = self.include_domains

        req = Request(
            TAVILY_ENDPOINT,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "trophic/0.1"},
        )
        try:
            with urlopen(req, timeout=20) as r:
                data = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"[tavily] query error: {e}")
            return []

        results = data.get("results", []) or []
        snippets: list[ResearchSnippet] = []
        for item in results[:n_results]:
            date = None
            pub = item.get("published_date")
            if pub:
                for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                    try:
                        date = _dt.datetime.strptime(pub[:len(fmt)+5].strip(), fmt).date().isoformat()
                        break
                    except ValueError:
                        continue
                if date is None:
                    date = pub[:10] if len(pub) >= 10 else pub
            snippets.append(ResearchSnippet(
                source=item.get("url", "tavily").split("/")[2] if item.get("url") else "tavily",
                url=item.get("url"),
                title=item.get("title", ""),
                snippet=item.get("content", ""),
                date=date,
                confidence=item.get("score"),
            ))
        return snippets
