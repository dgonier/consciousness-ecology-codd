"""Modal-hosted Qwen3 embeddings client for belief consolidation.

Mirrors the pattern in
debaterhub/.../services/embeddings/modal_embedding_provider.py
to ensure compatibility with the same backend.

Endpoint:
  https://debaterhub--embeddings-service-embeddingservice-serve.modal.run/embed
  - GET ?text=<single>     → 1024-dim embedding
  - POST {"texts": [...]}  → batch embeddings

Auth: `Authorization: Bearer <MODAL_API_KEY>`
"""
from __future__ import annotations

import logging
import math
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


DEFAULT_ENDPOINT = (
    "https://debaterhub--embeddings-service-embeddingservice-serve.modal.run/embed"
)
DEFAULT_TIMEOUT = 60


class ModalEmbedder:
    """Stateless embedding client for the Modal Qwen3 service."""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        api_token: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.endpoint_url = endpoint_url or os.environ.get(
            "MODAL_EMBEDDINGS_ENDPOINT", DEFAULT_ENDPOINT,
        )
        # Token name precedence matches debaterhub provider
        self.api_token = api_token or (
            os.environ.get("MODAL_API_TOKEN")
            or os.environ.get("debaterhub_API_TOKEN")
            or os.environ.get("MODAL_API_KEY")
        )
        self.timeout = timeout
        if not self.api_token:
            logger.warning("ModalEmbedder: no auth token in env (MODAL_API_KEY etc.)")

    def is_available(self) -> bool:
        return bool(self.api_token)

    def embed_text(self, text: str) -> list[float]:
        headers = {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}
        try:
            r = httpx.get(
                self.endpoint_url, headers=headers, params={"text": text},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            return data.get("embedding", [])
        except Exception as e:
            logger.error(f"embed_text failed: {e}")
            raise

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        try:
            r = httpx.post(
                self.endpoint_url, headers=headers,
                json={"texts": texts},
                timeout=self.timeout * 3,
            )
            r.raise_for_status()
            data = r.json()
            embs = data.get("embeddings", [])
            if len(embs) != len(texts):
                raise RuntimeError(
                    f"Embedding count mismatch: sent {len(texts)}, got {len(embs)}"
                )
            return embs
        except Exception as e:
            logger.error(f"embed_texts failed: {e}")
            raise


# ── Cosine similarity helper ──

def cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)
