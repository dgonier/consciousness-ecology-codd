"""Research tools: external lookups an analyst species can invoke.

The fundamental analyst (and future analyst species) need access to
information beyond the static evidence packet. Live runs use web
search; benchmark runs need date-bounded historical lookups to
avoid information leakage about scenarios from 2015-2016.

Two-step pattern:
  1. The analyst formulates a search query from its scratch read of
     the evidence (e.g. "AAPL earnings 2015-10-01").
  2. The tool executes the query and returns snippets.
  3. The analyst's synthesis includes the snippets.

Today only `OpenAIWebSearchTool` is wired (uses gpt-5's native
search). `MockResearchTool` returns the existing StockNet tweet body
for tests.
"""
from .base import ResearchSnippet, ResearchTool
from .openai_search import OpenAIWebSearchTool
from .mock import MockResearchTool

__all__ = [
    "ResearchSnippet", "ResearchTool",
    "OpenAIWebSearchTool", "MockResearchTool",
]
