"""API-based apex voters: OpenAI, Anthropic, Gemini, OpenRouter.

These are stubs that follow the ApexVoter interface. Each is gated by
its provider's env var (OPENAI_API_KEY, ANTHROPIC_API_KEY,
GEMINI_API_KEY, OPENROUTER_API_KEY); when the key is missing,
is_available() returns False and the orchestrator skips the voter.

When wired with keys, each voter:
  1. POSTs the EvidencePacket text to the provider's chat completion API
  2. Asks for logprobs on the response (where supported) so we can
     compute self-perplexity
  3. Parses the XML response for direction + confidence
"""
from __future__ import annotations

import json
import os
import math
from typing import Any

from ..training.xml_schema import parse_prediction
from .base import ApexVoter, EvidencePacket, VoterResponse
from .evidence import SYSTEM


def _split_system_user(text: str) -> tuple[str, str]:
    if text.startswith(SYSTEM):
        return SYSTEM, text[len(SYSTEM):].lstrip()
    return SYSTEM, text


class OpenAIVoter(ApexVoter):
    """Generic OpenAI chat-completion voter. Set OPENAI_MODEL=gpt-5 etc."""
    voter_id = "openai"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5")
        self.voter_id = f"openai_{self.model}"

    def is_available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def vote(self, evidence: EvidencePacket) -> VoterResponse:
        try:
            from openai import OpenAI
        except ImportError as e:
            return VoterResponse(self.voter_id, None, None, float("inf"),
                                 f"openai client not installed: {e}")
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        sys_text, user_text = _split_system_user(evidence.text)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_text},
                {"role": "user", "content": user_text},
            ],
            logprobs=True,
            top_logprobs=5,
            max_completion_tokens=128,
        )
        msg = resp.choices[0].message.content or ""
        parsed = parse_prediction(msg)
        # Derive perplexity from token logprobs of the response.
        ppl = float("inf")
        try:
            tokens = resp.choices[0].logprobs.content
            if tokens:
                lp = sum(t.logprob for t in tokens) / len(tokens)
                ppl = math.exp(-lp)
        except (AttributeError, TypeError):
            pass
        return VoterResponse(
            voter_id=self.voter_id,
            direction=parsed.direction,
            confidence=parsed.confidence,
            perplexity=ppl,
            raw_text=msg,
        )


class AnthropicVoter(ApexVoter):
    """Anthropic Claude voter (via the official SDK)."""
    voter_id = "anthropic"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get(
            "ANTHROPIC_MODEL", "claude-sonnet-4-6",
        )
        self.voter_id = f"anthropic_{self.model}"

    def is_available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def vote(self, evidence: EvidencePacket) -> VoterResponse:
        try:
            import anthropic
        except ImportError as e:
            return VoterResponse(self.voter_id, None, None, float("inf"),
                                 f"anthropic client not installed: {e}")
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        sys_text, user_text = _split_system_user(evidence.text)
        resp = client.messages.create(
            model=self.model,
            system=sys_text,
            max_tokens=128,
            messages=[{"role": "user", "content": user_text}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        parsed = parse_prediction(text)
        # Anthropic's API doesn't currently expose token logprobs; use
        # confidence parsed from the XML as an inverse-proxy.
        ppl = float("inf")
        if parsed.confidence is not None:
            # Map confidence∈[0,1] to perplexity proxy: more confident →
            # lower perplexity. Saturate at conf=0.99.
            c = max(min(parsed.confidence, 0.99), 0.01)
            ppl = 1.0 / c
        return VoterResponse(
            voter_id=self.voter_id,
            direction=parsed.direction,
            confidence=parsed.confidence,
            perplexity=ppl,
            raw_text=text,
        )


class GeminiVoter(ApexVoter):
    """Google Gemini voter."""
    voter_id = "gemini"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get(
            "GEMINI_MODEL", "gemini-2.5-pro",
        )
        self.voter_id = f"gemini_{self.model}"

    def is_available(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    def vote(self, evidence: EvidencePacket) -> VoterResponse:
        try:
            from google import genai
        except ImportError as e:
            return VoterResponse(self.voter_id, None, None, float("inf"),
                                 f"google.genai not installed: {e}")
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
        client = genai.Client(api_key=api_key)
        sys_text, user_text = _split_system_user(evidence.text)
        resp = client.models.generate_content(
            model=self.model,
            contents=user_text,
            config={"system_instruction": sys_text, "max_output_tokens": 128},
        )
        text = (resp.text or "").strip()
        parsed = parse_prediction(text)
        ppl = float("inf")
        if parsed.confidence is not None:
            c = max(min(parsed.confidence, 0.99), 0.01)
            ppl = 1.0 / c
        return VoterResponse(
            voter_id=self.voter_id,
            direction=parsed.direction,
            confidence=parsed.confidence,
            perplexity=ppl,
            raw_text=text,
        )


class OpenRouterVoter(ApexVoter):
    """OpenRouter — defaults to Qwen but can pin any model OpenRouter
    serves. Useful for diversity (e.g. open-source families)."""
    voter_id = "openrouter"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get(
            "OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct",
        )
        self.voter_id = f"openrouter_{self.model.replace('/', '_')}"

    def is_available(self) -> bool:
        return bool(os.environ.get("OPENROUTER_API_KEY"))

    def vote(self, evidence: EvidencePacket) -> VoterResponse:
        try:
            from openai import OpenAI  # OpenRouter is OpenAI-compatible
        except ImportError as e:
            return VoterResponse(self.voter_id, None, None, float("inf"),
                                 f"openai client not installed: {e}")
        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        sys_text, user_text = _split_system_user(evidence.text)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_text},
                {"role": "user", "content": user_text},
            ],
            logprobs=True,
            max_tokens=128,
        )
        msg = resp.choices[0].message.content or ""
        parsed = parse_prediction(msg)
        ppl = float("inf")
        try:
            tokens = resp.choices[0].logprobs.content
            if tokens:
                lp = sum(t.logprob for t in tokens) / len(tokens)
                ppl = math.exp(-lp)
        except (AttributeError, TypeError):
            pass
        return VoterResponse(
            voter_id=self.voter_id,
            direction=parsed.direction,
            confidence=parsed.confidence,
            perplexity=ppl,
            raw_text=msg,
        )
