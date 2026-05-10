"""DSPy LM adapter that targets the local vLLM OpenAI-compatible server.

Used by the firehose-loop's herbivore tier so each article-classification
call hits vLLM with continuous batching, instead of ModelHost's batch=1
in-process generation.

The apex tier still uses QwenLocalLM (in-process) because there's exactly
one apex call per day per path — vLLM round-trip latency adds no value
for that.
"""
from __future__ import annotations

import os
from typing import Optional

import dspy


def make_vllm_lm(
    model_name: str = "Qwen/Qwen3.5-4B",
    api_base: str = "http://localhost:8001/v1",
    api_key: str = "EMPTY",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    cache: bool = True,
    enable_thinking: bool = False,
) -> dspy.LM:
    """Build a dspy.LM that hits the local vLLM server.

    DSPy resolves provider via the model name's prefix; "openai/<model>"
    routes through litellm to an OpenAI-compatible HTTP API.

    enable_thinking=False is critical for Qwen3+ models — they default to
    emitting `<think>...</think>` reasoning blocks that break JSON parsing
    and waste tokens. We pass chat_template_kwargs through extra_body so
    vLLM applies it at chat-template time.
    """
    return dspy.LM(
        model=f"openai/{model_name}",
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        num_retries=2,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
    )


def vllm_health_check(api_base: str = "http://localhost:8001/v1",
                      timeout: float = 5.0) -> bool:
    """Returns True if the vLLM server is up and responsive."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"{api_base}/models", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False
