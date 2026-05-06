"""Local Qwen3-4B herbivore voter.

Mirrors the apex LocalQwenVoter but emits a synthesis paragraph instead
of a directional vote. Shares the singleton ModelHost so multiple
herbivore species running on Qwen3-4B don't reload weights.

Optional research_tool: when set, the voter does a two-step flow —
(1) ask Qwen to formulate a focus query, (2) execute the research
tool, (3) ask Qwen to synthesize given evidence + research snippets.
Used for analyst species that need to fetch data beyond the static
evidence packet.
"""
from __future__ import annotations

import os

import torch

from ..config import DEFAULT_CONFIG
from ..model_host import ModelHost
from .base import HerbivoreVoter, HerbivoreSynthesis
from .api_voters import _parse_synthesis, DEFAULT_HERB_SYSTEM


class LocalQwenHerbivore(HerbivoreVoter):
    def __init__(
        self,
        host: ModelHost | None = None,
        max_new_tokens: int = 256,
        species_template: str | None = None,
        research_tool: object | None = None,
    ):
        self.host = host or ModelHost.get(DEFAULT_CONFIG.model)
        self.max_new_tokens = max_new_tokens
        self.species_template = species_template
        self.research_tool = research_tool
        self.herb_id = "local_qwen3_4b_herb"

    def is_available(self) -> bool:
        return self.host._model is not None

    def _qwen_forward(self, system: str, user: str, max_new_tokens: int) -> str:
        """One Qwen forward; returns decoded new tokens."""
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        tok = self.host._tok
        try:
            ids = tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", enable_thinking=False,
            ).to(self.host.device)
        except TypeError:
            ids = tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.host.device)
        am = torch.ones_like(ids)
        with torch.no_grad():
            out = self.host._model.generate(
                input_ids=ids,
                attention_mask=am,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        new = out[0, ids.shape[1]:]
        return tok.decode(new, skip_special_tokens=True)

    def synthesize(self, scenario_text: str) -> HerbivoreSynthesis:
        system = self.species_template or DEFAULT_HERB_SYSTEM
        research_block = ""
        research_query = ""
        # Optional research-call two-step flow
        if self.research_tool is not None:
            try:
                # Step 1: ask Qwen to formulate a focus query
                ticker, as_of = self._extract_ticker_and_date(scenario_text)
                query_prompt = (
                    "Based on the evidence below, write ONE concise search query"
                    " (max 12 words) describing the SINGLE most important"
                    f" topic to research about {ticker} as of {as_of}. Reply"
                    " with ONLY the query string, no quotes, no preamble.\n\n"
                    + scenario_text[:3000]
                )
                research_query = self._qwen_forward(
                    "You write focused research queries.", query_prompt, 64,
                ).strip().splitlines()[0][:200]
                # Step 2: execute the research tool
                snippets = self.research_tool.query(
                    ticker=ticker, as_of_date=as_of,
                    focus=research_query, n_results=4,
                )
                rendered = self.research_tool.render_snippets(snippets)
                research_block = (
                    f"\n\nRESEARCH QUERY: {research_query}\n"
                    f"RESEARCH RESULTS:\n{rendered}\n"
                )
            except Exception as e:
                research_block = f"\n\nRESEARCH ERROR: {e}\n"
        # Main synthesis call
        user = scenario_text + research_block
        text = self._qwen_forward(system, user, self.max_new_tokens)
        diet, synth, hint = _parse_synthesis(text)
        return HerbivoreSynthesis(
            herb_id=self.herb_id,
            species_id="?",
            diet_tag=diet,
            synthesis=synth,
            confidence=None,
            raw_text=text,
            provider_meta={
                "model": "Qwen3-4B (local)",
                "research_query": research_query,
                "research_used": bool(research_block and "RESEARCH ERROR" not in research_block),
            },
            direction_hint=hint,
        )

    @staticmethod
    def _extract_ticker_and_date(scenario_text: str) -> tuple[str, str]:
        """Best-effort extraction from the EvidencePacket text. Looks for
        'OHLCV history for SYMBOL' and 'as of YYYY-MM-DD'. Falls back to
        unknown."""
        import re
        ticker = "UNKNOWN"
        m = re.search(r"OHLCV history for (\S+)", scenario_text)
        if m:
            ticker = m.group(1).strip("(),:;")
        # Date hint: scenario name like "stocknet_test_AAPL_2015-10-01" appears
        # in the text via signature; use heuristics
        m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", scenario_text)
        as_of = m.group(1) if m else "current date"
        return ticker, as_of
