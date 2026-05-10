"""DSPy LM adapter that targets the Hexis vLLM deploy with a per-species
session.

Wraps the OpenAI-compat /v1/chat/completions endpoint and threads a
session_id (and optionally a chat_template_kwargs) through DSPy's
extra_body so the deploy applies its M+E+slot context to that session.

Design (per call 2026-05-10):
  - Each species gets its own dspy.LM instance bound to one session_id
  - SpeciesSessionManager handles cache + refresh-on-404 lifecycle (Neo4j
    is the persistence layer, sessions are scratch workspaces)
  - At firehose-loop init, build one LM per active species and hand them
    to the herbivore via dspy.context() per-call

Usage:
    from trophic.beliefs.hexis_client import HexisClient, SpeciesSessionManager
    from trophic.beliefs.dspy_hexis_lm import make_hexis_lm

    client = HexisClient()
    mgr = SpeciesSessionManager(client)

    sid = mgr.get_or_create("event_classifier.v0")
    lm = make_hexis_lm(api_base=client.base_url, session_id=sid)

    with dspy.context(lm=lm):
        # event_classifier inference runs through the Hexis session
        ...
"""
from __future__ import annotations

import os
from typing import Optional

import dspy


def make_hexis_lm(
    session_id: str,
    api_base: Optional[str] = None,
    model_name: str = "Qwen/Qwen3.5-4B",
    api_key: str = "EMPTY",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    cache: bool = True,
    enable_thinking: bool = False,
) -> dspy.LM:
    """Build a dspy.LM that hits the Hexis vLLM deploy for a specific session.

    Args:
        session_id: Result of SpeciesSessionManager.get_or_create(species_id).
            All chat completions through this LM will reference that session
            so the deploy applies the species' Mind Tree state.
        api_base: Hexis deploy URL. Defaults to HEXIS_API_URL env var, then
            the canonical A100 deploy URL.

    The Hexis deploy adds a `session_id` top-level body param on top of the
    OpenAI chat completion schema. DSPy's litellm path forwards extra_body
    verbatim, so we slip it in there.

    Note: DSPy's prompt cache is keyed by request body. Different
    session_ids → different cache entries → no cross-species contamination.
    """
    if api_base is None:
        api_base = os.environ.get(
            "HEXIS_API_URL",
            "https://debaterhub--hexis-agentic-a100-hexisagentic-serve.modal.run",
        )
    # The deploy expects /v1/chat/completions at the root, so api_base must
    # end with /v1.
    if not api_base.rstrip("/").endswith("/v1"):
        api_base = api_base.rstrip("/") + "/v1"

    return dspy.LM(
        model=f"openai/{model_name}",
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        num_retries=2,
        # Critical: session_id is a top-level body param specific to the
        # Hexis deploy. enable_thinking=False suppresses CoT tokens.
        extra_body={
            "session_id": session_id,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
    )


def hexis_health_check(
    api_base: Optional[str] = None,
    timeout: float = 10.0,
) -> bool:
    """Returns True if the Hexis deploy is reachable + reports vllm_ready."""
    if api_base is None:
        api_base = os.environ.get(
            "HEXIS_API_URL",
            "https://debaterhub--hexis-agentic-a100-hexisagentic-serve.modal.run",
        )
    import urllib.request
    import json
    base = api_base.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=timeout) as r:
            if r.status != 200:
                return False
            body = json.loads(r.read())
            return bool(body.get("vllm_ready"))
    except Exception:
        return False
