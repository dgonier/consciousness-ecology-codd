"""OpenAI web-search-tool research lookup.

Uses gpt-5's native web_search tool. Date-bounded by including the
as_of_date in the query so the model preferentially returns
contemporaneous sources. NOTE: this is a best-effort temporal bound,
not a hard guarantee — for benchmark eval on historical scenarios
there's some risk of data-leakage from later articles. For live
trading runs there's no leakage problem.
"""
from __future__ import annotations

import os

from .base import ResearchTool, ResearchSnippet


class OpenAIWebSearchTool(ResearchTool):
    name = "openai_web_search"

    def __init__(self, model: str = "gpt-5"):
        self.model = model

    def is_available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def query(
        self,
        ticker: str,
        as_of_date: str,
        focus: str = "",
        n_results: int = 5,
    ) -> list[ResearchSnippet]:
        try:
            from openai import OpenAI
        except ImportError:
            return []
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        focus_text = f" focused on {focus}" if focus else ""
        prompt = (
            f"Search for news and analysis about {ticker} stock as of"
            f" {as_of_date}{focus_text}. Return up to {n_results} results"
            " ordered by relevance. For each result include source domain,"
            " title, a 2-sentence snippet, publication date if known, and"
            " URL. Format strictly as a numbered list."
        )
        try:
            resp = client.responses.create(
                model=self.model,
                tools=[{"type": "web_search"}],
                input=prompt,
            )
        except Exception as e:
            print(f"[openai_search] error: {e}")
            return []
        text = ""
        try:
            for item in resp.output:
                if hasattr(item, "content"):
                    for c in item.content:
                        if hasattr(c, "text"):
                            text += c.text + "\n"
        except (AttributeError, TypeError):
            text = str(resp)
        # Parse the numbered-list output into ResearchSnippets.
        # Tolerant parser; the model's exact format varies per call.
        snippets: list[ResearchSnippet] = []
        current: dict = {}
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                if current:
                    snippets.append(ResearchSnippet(
                        source=current.get("source", "web"),
                        title=current.get("title", ""),
                        snippet=current.get("snippet", ""),
                        date=current.get("date"),
                        url=current.get("url"),
                    ))
                    current = {}
                continue
            low = stripped.lower()
            if low.startswith(("title:", "headline:")):
                current["title"] = stripped.split(":", 1)[1].strip()
            elif low.startswith(("source:", "domain:")):
                current["source"] = stripped.split(":", 1)[1].strip()
            elif low.startswith(("date:", "published:")):
                current["date"] = stripped.split(":", 1)[1].strip()
            elif low.startswith("url:") or low.startswith("link:"):
                current["url"] = stripped.split(":", 1)[1].strip()
            elif low.startswith(("snippet:", "summary:", "excerpt:")):
                current["snippet"] = stripped.split(":", 1)[1].strip()
            elif "title" not in current and stripped[:2].rstrip(".").isdigit():
                # Numbered-list item header — treat as title fallback
                current["title"] = stripped.split(" ", 1)[-1]
            else:
                # Append to snippet body
                current["snippet"] = (current.get("snippet", "") + " " + stripped).strip()
        if current:
            snippets.append(ResearchSnippet(
                source=current.get("source", "web"),
                title=current.get("title", ""),
                snippet=current.get("snippet", ""),
                date=current.get("date"),
                url=current.get("url"),
            ))
        return snippets[:n_results]
