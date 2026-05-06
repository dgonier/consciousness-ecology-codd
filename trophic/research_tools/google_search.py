"""Google Custom Search JSON API research tool.

Date-bounded via the `dateRestrict` parameter (e.g. d7 = past 7 days
relative to the search) OR the more precise `tbs=cdr:1,cd_min:...,
cd_max:...` literal Google search syntax which we pass via `siterestrict`
custom search element. For benchmark scenarios (historical dates),
we need the absolute window form.

Auth: GOOGLE_API_KEY + GOOGLE_CSE_ID env vars. Get a key at
https://developers.google.com/custom-search/v1/overview ; create a CSE
and grab the cx id at https://cse.google.com/cse/all .

Cost: $5 per 1000 queries beyond the 100/day free tier.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .base import ResearchTool, ResearchSnippet


GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"


class GoogleCSEResearchTool(ResearchTool):
    """Google Custom Search JSON API with absolute-date windowing.

    `as_of_date` (ISO yyyy-mm-dd) → search restricted to a window
    ending on that date. Default window is `window_days` days back.
    For benchmark eval this prevents forward-looking leakage; for
    live predictions it just narrows to recent news.
    """

    name = "google_cse"

    def __init__(
        self,
        api_key: str | None = None,
        cse_id: str | None = None,
        window_days: int = 7,
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.cse_id = cse_id or os.environ.get("GOOGLE_CSE_ID")
        self.window_days = window_days

    def is_available(self) -> bool:
        return bool(self.api_key and self.cse_id)

    @staticmethod
    def _date_window(as_of_date: str, window_days: int) -> tuple[str, str] | None:
        """Return (cd_min, cd_max) in MM/DD/YYYY (Google's `tbs` format)
        spanning `window_days` ending on as_of_date. None if the date
        isn't parseable."""
        try:
            d = _dt.date.fromisoformat(as_of_date)
        except (ValueError, TypeError):
            return None
        cd_max = d.strftime("%m/%d/%Y")
        cd_min = (d - _dt.timedelta(days=window_days)).strftime("%m/%d/%Y")
        return cd_min, cd_max

    def query(
        self,
        ticker: str,
        as_of_date: str,
        focus: str = "",
        n_results: int = 5,
    ) -> list[ResearchSnippet]:
        if not self.is_available():
            return []
        # Build query string. Combine ticker + focus, prefer "stock" hint.
        q_parts = [f"\"{ticker}\""]
        if focus:
            q_parts.append(focus)
        else:
            q_parts.append("stock news")
        q = " ".join(q_parts)

        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": q,
            "num": min(max(n_results, 1), 10),
        }
        # Date-window: Google CSE supports `sort=date:r:YYYYMMDD:YYYYMMDD`
        # (the `r:` form takes a min:max range). This is the cleanest
        # way to bound results by publication date.
        window = self._date_window(as_of_date, self.window_days)
        if window:
            cd_min, cd_max = window
            # Convert MM/DD/YYYY → YYYYMMDD for the sort param
            try:
                d_max = _dt.datetime.strptime(cd_max, "%m/%d/%Y").strftime("%Y%m%d")
                d_min = _dt.datetime.strptime(cd_min, "%m/%d/%Y").strftime("%Y%m%d")
                params["sort"] = f"date:r:{d_min}:{d_max}"
            except ValueError:
                pass

        url = f"{GOOGLE_CSE_ENDPOINT}?{urlencode(params)}"
        try:
            with urlopen(Request(url, headers={"User-Agent": "trophic/0.1"}), timeout=15) as r:
                data = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"[google_cse] query error: {e}")
            return []

        items = data.get("items", []) or []
        snippets: list[ResearchSnippet] = []
        for item in items[:n_results]:
            # Google returns 'snippet' for excerpt, 'displayLink' for source,
            # 'pagemap.metatags[0].article:published_time' for date sometimes.
            date = None
            try:
                metatags = item.get("pagemap", {}).get("metatags", [])
                if metatags:
                    for k in ("article:published_time", "pubdate", "date"):
                        if k in metatags[0]:
                            date = metatags[0][k][:10]  # ISO date prefix
                            break
            except (AttributeError, IndexError, TypeError):
                pass
            snippets.append(ResearchSnippet(
                source=item.get("displayLink", "google"),
                url=item.get("link"),
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                date=date,
            ))
        return snippets
