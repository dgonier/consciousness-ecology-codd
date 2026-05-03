"""Shared parsing helpers for voter responses.

Voters return free text; we want to split out the REASONING block from
the XML <prediction> block so each is reviewable in isolation.
"""
from __future__ import annotations

import re

REASONING_RE = re.compile(
    r"REASONING\s*:\s*(.*?)(?=<prediction\b|$)",
    re.IGNORECASE | re.DOTALL,
)


def extract_reasoning(raw_text: str) -> str:
    """Pull the REASONING: block out of a voter response. Falls back to
    'everything before <prediction>' if no explicit REASONING marker."""
    if not raw_text:
        return ""
    m = REASONING_RE.search(raw_text)
    if m:
        return m.group(1).strip()
    # Fallback: the substring before the XML tag, if any.
    idx = raw_text.find("<prediction")
    if idx > 0:
        head = raw_text[:idx].strip()
        # Drop conversational filler that prefixes the text.
        head = re.sub(r"^(?:sure|okay|here.s|i.ll|let me)[^\n]*\n", "", head, flags=re.IGNORECASE)
        return head
    return ""


CITATION_RE = re.compile(
    r"\b(bar\s*-?\d+|tweet\s*-?\d+|forecast\.[a-z_]+|drift|spread|snr|monotonicity|"
    r"open|high|low|close|volume|q10|q90)\b",
    re.IGNORECASE,
)


def extract_citations(reasoning: str) -> list[str]:
    """Heuristic: pull domain-specific tokens out of reasoning text as
    structured citations. Lossy but enough for KG queries like 'voters
    that cite drift tend to be right when ...'."""
    if not reasoning:
        return []
    return list({m.group(0).lower() for m in CITATION_RE.finditer(reasoning)})
