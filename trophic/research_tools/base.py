"""ResearchTool interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ResearchSnippet:
    """One snippet returned by a research tool."""
    source: str  # 'web' | 'tweets' | 'filings' | etc.
    url: str | None = None
    title: str = ""
    snippet: str = ""
    date: str | None = None  # ISO yyyy-mm-dd if known
    confidence: float | None = None


class ResearchTool(ABC):
    """Synchronous research lookup. Date-bounded: callers pass an
    as_of date; the tool guarantees not to return information from
    AFTER that date (modulo provider compliance).
    """
    name: str = "research_tool"

    @abstractmethod
    def query(
        self,
        ticker: str,
        as_of_date: str,
        focus: str = "",
        n_results: int = 5,
    ) -> list[ResearchSnippet]:
        """Issue a research lookup. `focus` is free text the analyst
        species can use to direct the search (e.g. 'earnings catalyst',
        'sector rotation', 'macro headwind'). Returns up to n_results
        snippets ordered by relevance."""
        ...

    def is_available(self) -> bool:
        return True

    def render_snippets(self, snippets: list[ResearchSnippet]) -> str:
        """Convert snippets to a text block the analyst can read."""
        if not snippets:
            return "(no research results)"
        lines = []
        for i, s in enumerate(snippets, 1):
            head = f"[{i}] {s.source}"
            if s.date:
                head += f" {s.date}"
            if s.title:
                head += f" — {s.title}"
            lines.append(head)
            if s.snippet:
                lines.append(f"    {s.snippet[:400]}")
            if s.url:
                lines.append(f"    {s.url}")
        return "\n".join(lines)
