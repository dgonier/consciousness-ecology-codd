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
from .parsing import extract_reasoning, extract_citations


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
        reasoning = extract_reasoning(msg)
        response = VoterResponse(
            voter_id=self.voter_id,
            direction=parsed.direction,
            confidence=parsed.confidence,
            perplexity=ppl,
            raw_text=msg,
            reasoning=reasoning,
            evidence_citations=extract_citations(reasoning),
            provider_meta={
                "model": self.model,
                "input_tokens": getattr(resp.usage, "prompt_tokens", None) if resp.usage else None,
                "output_tokens": getattr(resp.usage, "completion_tokens", None) if resp.usage else None,
            },
        )
        return self._apply_calibration(response, evidence.agent_feedback)


class AnthropicVoter(ApexVoter):
    """Anthropic Claude voter via AWS Bedrock.

    Per project decision: all Anthropic calls go through Bedrock (not
    the public Anthropic API). Auth uses standard AWS env vars
    (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION).

    Set BEDROCK_MODEL_ID to override the default sonnet-4-6 inference
    profile (e.g. "us.anthropic.claude-opus-4-7-v1:0").
    """
    voter_id = "anthropic"

    def __init__(self, model: str | None = None, region: str | None = None):
        self.model = model or os.environ.get(
            "BEDROCK_MODEL_ID",
            os.environ.get("ANTHROPIC_MODEL", "us.anthropic.claude-sonnet-4-6"),
        )
        self.region = region or os.environ.get(
            "AWS_REGION",
            os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )
        # Sanitize the model id for use as a path-safe voter id
        safe_id = self.model.replace("/", "_").replace(":", "_").replace(".", "_")
        self.voter_id = f"anthropic_bedrock_{safe_id}"

    def is_available(self) -> bool:
        # Bedrock auth: standard AWS creds (env vars or shared profile)
        # plus region. We treat the presence of access key OR a profile
        # name as sufficient; boto3 will surface auth errors at call
        # time if credentials are wrong.
        if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
            return True
        if os.environ.get("AWS_PROFILE"):
            return True
        return False

    def vote(self, evidence: EvidencePacket) -> VoterResponse:
        try:
            import boto3
        except ImportError as e:
            return VoterResponse(self.voter_id, None, None, float("inf"),
                                 f"boto3 not installed: {e}")
        client = boto3.client("bedrock-runtime", region_name=self.region)
        sys_text, user_text = _split_system_user(evidence.text)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 192,
            "temperature": 0.0,
            "system": sys_text,
            "messages": [{"role": "user", "content": user_text}],
        }
        try:
            resp = client.invoke_model(
                modelId=self.model,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(resp["body"].read())
            text = ""
            for p in payload.get("content", []):
                if p.get("type") == "text":
                    text = p.get("text", "")
                    break
            usage = payload.get("usage", {}) or {}
        except Exception as e:
            return VoterResponse(self.voter_id, None, None, float("inf"),
                                 f"bedrock invoke error: {e}")
        parsed = parse_prediction(text)
        # Bedrock's invoke_model doesn't expose token logprobs; use
        # parsed confidence as an inverse-proxy.
        ppl = float("inf")
        if parsed.confidence is not None:
            c = max(min(parsed.confidence, 0.99), 0.01)
            ppl = 1.0 / c
        reasoning = extract_reasoning(text)
        response = VoterResponse(
            voter_id=self.voter_id,
            direction=parsed.direction,
            confidence=parsed.confidence,
            perplexity=ppl,
            raw_text=text,
            reasoning=reasoning,
            evidence_citations=extract_citations(reasoning),
            provider_meta={
                "model": self.model,
                "via": "bedrock",
                "region": self.region,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
            },
        )
        return self._apply_calibration(response, evidence.agent_feedback)


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
        reasoning = extract_reasoning(text)
        response = VoterResponse(
            voter_id=self.voter_id,
            direction=parsed.direction,
            confidence=parsed.confidence,
            perplexity=ppl,
            raw_text=text,
            reasoning=reasoning,
            evidence_citations=extract_citations(reasoning),
            provider_meta={"model": self.model},
        )
        return self._apply_calibration(response, evidence.agent_feedback)


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
        reasoning = extract_reasoning(msg)
        response = VoterResponse(
            voter_id=self.voter_id,
            direction=parsed.direction,
            confidence=parsed.confidence,
            perplexity=ppl,
            raw_text=msg,
            reasoning=reasoning,
            evidence_citations=extract_citations(reasoning),
            provider_meta={"model": self.model, "via": "openrouter"},
        )
        return self._apply_calibration(response, evidence.agent_feedback)
