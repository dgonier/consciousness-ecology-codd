"""API-backed herbivore voters: OpenAI, Anthropic Bedrock, Gemini.

Each species is parameterized by its prompt_template (the herb's
analytical lens — technical, fundamental, contrarian, etc.) and runs
on whatever model_id the species record specifies.

No local-Qwen herbivore variant today; herbs run on cloud models since
the local 4B is busy serving the apex tier.
"""
from __future__ import annotations

import json
import os

from .base import HerbivoreVoter, HerbivoreSynthesis


# Default herb-tier system prompt; any species's prompt_template
# overrides this. Designed to elicit a 1-3 sentence synthesis with an
# optional direction hint.
DEFAULT_HERB_SYSTEM = (
    "You are an analytical herbivore in a multi-species market panel."
    " Read the evidence below and emit a SHORT (2-3 sentence)"
    " synthesis from your specialized analytical lens. Do NOT make a"
    " final UP/DOWN trade decision — the apex panel does that. Your"
    " job is to FRAME the evidence so the apex has better input."
    "\n\n"
    "Reply on three lines exactly:\n"
    "DIET: <one of: technical | fundamental | macro | sentiment | contrarian>\n"
    "SYNTHESIS: <2-3 sentences>\n"
    "DIRECTION_HINT: <up | down | none>"
)


def _parse_synthesis(text: str) -> tuple[str, str, str | None]:
    """Pull DIET, SYNTHESIS, DIRECTION_HINT lines out of an herb response.
    Tolerant of missing lines — synthesis defaults to the whole text."""
    diet = "unknown"
    synth = text.strip()
    hint: str | None = None
    for line in text.splitlines():
        s = line.strip()
        u = s.upper()
        if u.startswith("DIET:"):
            diet = s.split(":", 1)[1].strip().lower() or "unknown"
        elif u.startswith("SYNTHESIS:"):
            synth = s.split(":", 1)[1].strip()
        elif u.startswith("DIRECTION_HINT:") or u.startswith("DIRECTION:"):
            v = s.split(":", 1)[1].strip().lower()
            if v in ("up", "down"):
                hint = v
    return diet, synth, hint


class _HerbBase(HerbivoreVoter):
    """Common helpers for all api-backed herbivores."""

    def _system_prompt(self) -> str:
        return self.species_template or DEFAULT_HERB_SYSTEM


class OpenAIHerbivore(_HerbBase):
    """OpenAI-backed herbivore."""

    def __init__(self, model: str = "gpt-5-mini", species_template: str | None = None):
        self.model = model
        self.species_template = species_template
        self.herb_id = f"openai_{model}_herb"

    def is_available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def synthesize(self, scenario_text: str) -> HerbivoreSynthesis:
        try:
            from openai import OpenAI
        except ImportError as e:
            return HerbivoreSynthesis(
                herb_id=self.herb_id, species_id="?", diet_tag="error",
                synthesis=f"openai client not installed: {e}",
                confidence=None, raw_text="",
            )
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        is_reasoning = any(p in self.model.lower() for p in ("gpt-5", "o1", "o3", "o4"))
        max_tok = 1024 if is_reasoning else 256
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": scenario_text},
            ],
            max_completion_tokens=max_tok,
        )
        text = resp.choices[0].message.content or ""
        diet, synth, hint = _parse_synthesis(text)
        return HerbivoreSynthesis(
            herb_id=self.herb_id, species_id="?",
            diet_tag=diet, synthesis=synth,
            confidence=None, raw_text=text,
            provider_meta={"model": self.model},
            direction_hint=hint,
        )


class AnthropicBedrockHerbivore(_HerbBase):
    """Anthropic Claude herbivore via Bedrock."""

    def __init__(self, model: str = "us.anthropic.claude-sonnet-4-6",
                 species_template: str | None = None):
        self.model = model
        self.species_template = species_template
        safe = model.replace("/", "_").replace(":", "_").replace(".", "_")
        self.herb_id = f"anthropic_bedrock_{safe}_herb"
        self.region = os.environ.get(
            "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )

    def is_available(self) -> bool:
        try:
            import boto3
            return boto3.Session().get_credentials() is not None
        except Exception:
            return False

    def synthesize(self, scenario_text: str) -> HerbivoreSynthesis:
        try:
            import boto3
        except ImportError as e:
            return HerbivoreSynthesis(
                herb_id=self.herb_id, species_id="?", diet_tag="error",
                synthesis=f"boto3 not installed: {e}",
                confidence=None, raw_text="",
            )
        client = boto3.client("bedrock-runtime", region_name=self.region)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 256,
            "temperature": 0.2,
            "system": self._system_prompt(),
            "messages": [{"role": "user", "content": scenario_text}],
        }
        try:
            resp = client.invoke_model(
                modelId=self.model,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(resp["body"].read())
        except Exception as e:
            return HerbivoreSynthesis(
                herb_id=self.herb_id, species_id="?", diet_tag="error",
                synthesis=f"bedrock invoke error: {e}",
                confidence=None, raw_text="",
            )
        text = ""
        for p in payload.get("content", []):
            if p.get("type") == "text":
                text = p.get("text", "")
                break
        diet, synth, hint = _parse_synthesis(text)
        return HerbivoreSynthesis(
            herb_id=self.herb_id, species_id="?",
            diet_tag=diet, synthesis=synth,
            confidence=None, raw_text=text,
            provider_meta={"model": self.model, "via": "bedrock"},
            direction_hint=hint,
        )


class GeminiHerbivore(_HerbBase):
    """Google Gemini herbivore (default flash, thinking disabled)."""

    def __init__(self, model: str = "gemini-2.5-flash",
                 species_template: str | None = None):
        self.model = model
        self.species_template = species_template
        self.herb_id = f"gemini_{model}_herb"

    def is_available(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    def synthesize(self, scenario_text: str) -> HerbivoreSynthesis:
        try:
            from google import genai
            from google.genai import types as gtypes
        except ImportError as e:
            return HerbivoreSynthesis(
                herb_id=self.herb_id, species_id="?", diet_tag="error",
                synthesis=f"google.genai not installed: {e}",
                confidence=None, raw_text="",
            )
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
        client = genai.Client(api_key=api_key)
        cfg_kwargs = dict(
            system_instruction=self._system_prompt(),
            max_output_tokens=512,
        )
        if "flash" in self.model.lower():
            cfg_kwargs["thinking_config"] = gtypes.ThinkingConfig(thinking_budget=0)
        resp = client.models.generate_content(
            model=self.model,
            contents=scenario_text,
            config=gtypes.GenerateContentConfig(**cfg_kwargs),
        )
        text = (resp.text or "").strip()
        diet, synth, hint = _parse_synthesis(text)
        return HerbivoreSynthesis(
            herb_id=self.herb_id, species_id="?",
            diet_tag=diet, synthesis=synth,
            confidence=None, raw_text=text,
            provider_meta={"model": self.model},
            direction_hint=hint,
        )
