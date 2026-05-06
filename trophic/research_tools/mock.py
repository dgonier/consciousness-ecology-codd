"""Mock research tool. Returns canned snippets for tests / dev."""
from __future__ import annotations

from .base import ResearchTool, ResearchSnippet


class MockResearchTool(ResearchTool):
    name = "mock_research"

    def __init__(self, snippets_per_query: list[ResearchSnippet] | None = None):
        self._snippets = snippets_per_query or []

    def is_available(self) -> bool:
        return True

    def query(
        self,
        ticker: str,
        as_of_date: str,
        focus: str = "",
        n_results: int = 5,
    ) -> list[ResearchSnippet]:
        if self._snippets:
            return self._snippets[:n_results]
        return [
            ResearchSnippet(
                source="mock", title=f"{ticker} mock result",
                snippet=f"Mock snippet for {ticker} on {as_of_date} (focus: {focus}).",
                date=as_of_date,
            )
        ]
