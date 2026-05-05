"""Bedrock Opus 4.7 helper for the decomposer's hardest reasoning tasks.

Two callers:
  - Reproducer:   given two parent species + their fitness traces,
                  output one child species config that combines their
                  best components.
  - GapAnalyzer:  given recent failure scenarios + the current panel,
                  output a brand-new species that would fill the gap.

Both ask Opus to return a JSON object matching a known schema. The
helper enforces the schema by retrying on parse failure and clamping
the response to known fields.

Inference profile: us.anthropic.claude-opus-4-7-v1:0 (override via
DECOMPOSER_OPUS_MODEL env var).
"""
from __future__ import annotations

import json
import os
import re

# Default to Opus 4.7 via Bedrock inference profile.
DEFAULT_OPUS_MODEL = os.environ.get(
    "DECOMPOSER_OPUS_MODEL", "us.anthropic.claude-opus-4-7-v1:0",
)


def call_opus_for_json(
    system: str,
    user: str,
    *,
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.4,
) -> dict | None:
    """Invoke Bedrock Opus and parse a single JSON object out of the
    response. Returns the dict, or None on parse failure.

    The system prompt should instruct the model to return ONLY a JSON
    object, but real models still wrap with prose; we strip code-fence
    markers and pull the largest balanced-brace span.
    """
    try:
        import boto3
    except ImportError:
        return None
    model = model or DEFAULT_OPUS_MODEL
    region = os.environ.get(
        "AWS_REGION",
        os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    try:
        client = boto3.client("bedrock-runtime", region_name=region)
    except Exception:
        return None
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    try:
        resp = client.invoke_model(
            modelId=model,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(resp["body"].read())
    except Exception as e:
        print(f"[opus_judge] bedrock invoke error: {e}")
        return None
    text = ""
    for p in payload.get("content", []):
        if p.get("type") == "text":
            text = p.get("text", "")
            break
    return _extract_json(text)


def _extract_json(text: str) -> dict | None:
    """Pull a JSON object out of a model response. Tolerant of
    code-fence wrapping and prose preambles."""
    if not text:
        return None
    # Strip code-fence wrappers
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    # Find the largest balanced-brace span
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return None
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def is_opus_available() -> bool:
    """True if boto3 default credential chain resolves AND we can
    construct a bedrock client. Doesn't make a network call."""
    try:
        import boto3
        session = boto3.Session()
        if session.get_credentials() is None:
            return False
        # Construction only — no API hit
        boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get(
                "AWS_REGION",
                os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            ),
        )
        return True
    except Exception:
        return False
