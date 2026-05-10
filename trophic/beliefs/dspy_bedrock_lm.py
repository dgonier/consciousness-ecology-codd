"""DSPy LM adapter for AWS Bedrock (Sonnet 4.6 by default).

Used for the ORACLE-Sonnet baseline in scripts/run_firehose_loop.py
(--path oracle-sonnet) and anywhere else we want a frontier-model
ceiling on top of the same DSPy signature plumbing.

Auth: assumes the local AWS CLI is configured (the user confirmed
`aws cli already should work for bedrock`). DSPy/LiteLLM picks up the
default credentials chain.
"""
from __future__ import annotations

import os

import dspy


def make_bedrock_lm(
    model_id: str,
    region: str = "us-east-1",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    cache: bool = True,
) -> dspy.LM:
    """Generic Bedrock LM via DSPy/LiteLLM. Pass model_id e.g.
    'us.anthropic.claude-sonnet-4-6' or 'us.anthropic.claude-opus-4-7'.
    """
    return dspy.LM(
        model=f"bedrock/{model_id}",
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        num_retries=2,
        aws_region_name=region,
    )


def make_bedrock_sonnet_lm(
    model_id: str | None = None,
    region: str = "us-east-1",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    cache: bool = True,
) -> dspy.LM:
    """Sonnet 4.6 via Bedrock (back-compat shim)."""
    if model_id is None:
        model_id = os.environ.get(
            "ORACLE_SONNET_MODEL_ID",
            "us.anthropic.claude-sonnet-4-6",
        )
    return make_bedrock_lm(
        model_id=model_id, region=region,
        temperature=temperature, max_tokens=max_tokens, cache=cache,
    )


def make_bedrock_opus_lm(
    model_id: str | None = None,
    region: str = "us-east-1",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    cache: bool = True,
) -> dspy.LM:
    """Opus via Bedrock. Default model_id is Opus 4.6 (most recent Opus
    typically entitled on AWS accounts; 4.7 is gated and may 403). Override
    with ORACLE_OPUS_MODEL_ID env var.
    """
    if model_id is None:
        model_id = os.environ.get(
            "ORACLE_OPUS_MODEL_ID",
            "us.anthropic.claude-opus-4-6-v1",
        )
    return make_bedrock_lm(
        model_id=model_id, region=region,
        temperature=temperature, max_tokens=max_tokens, cache=cache,
    )
