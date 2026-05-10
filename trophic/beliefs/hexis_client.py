"""Hexis vLLM client — trophic firehose's adapter to the deployed Modal app.

Talks to the A100 sibling deploy at
  https://debaterhub--hexis-agentic-a100-hexisagentic-serve.modal.run

Two layers:
  - HexisClient: thin HTTP wrapper around /health, /v1/chat/completions,
    /v1/session, /v1/mind_tree/node.
  - SpeciesSessionManager: (species_id → session_id) cache with refresh-on-404
    semantics. Persists the cache to data/hexis_sessions/{species_id}.json
    so that across runs we keep the same session_id alive.

Architecture (per design call 2026-05-10):
  - Per-species Neo4j Mind Tree nodes are persistent (`root:finance:{species_id}`).
    Live forever via Mind Tree write_lesson / add_setting.
  - Sessions are short-lived in-RAM workspaces that point at a species via
    preload_node_ids. Rebuild any time from the persisted Neo4j state.
  - Mind Tree is the source of truth; sessions are scratch.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HEXIS_URL = os.environ.get(
    "HEXIS_API_URL",
    "https://debaterhub--hexis-agentic-a100-hexisagentic-serve.modal.run",
)
SESSIONS_DIR = ROOT / "data" / "hexis_sessions"


# ── HTTP client ────────────────────────────────────────────────────────────

@dataclass
class HexisClient:
    """Thin HTTP client for the Hexis vLLM Modal deploy.

    All calls go through self._client (httpx.Client) so we can configure
    timeouts and reuse connections.
    """
    base_url: str = DEFAULT_HEXIS_URL
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_s,
            headers={"Content-Type": "application/json"},
        )

    def health(self) -> dict:
        r = self._client.get("/health", timeout=30.0)
        r.raise_for_status()
        return r.json()

    def create_session(
        self,
        domain: str = "finance",
        user_request: str = "",
        task_instruction: str = "",
        ttl_seconds: int = 86400,
        preload_node_ids: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        body = {
            "initial_state": {
                "domain": domain,
                "user_request": user_request,
                "task_instruction": task_instruction,
                **(metadata or {}),
            },
            "ttl_seconds": ttl_seconds,
            "preload_node_ids": preload_node_ids or [],
        }
        r = self._client.post("/v1/session", json=body)
        r.raise_for_status()
        return r.json()

    def get_session(self, session_id: str) -> Optional[dict]:
        r = self._client.get(f"/v1/session/{session_id}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def chat(
        self,
        messages: list[dict],
        session_id: Optional[str] = None,
        model: str = "Qwen/Qwen3.5-4B",
        max_tokens: int = 256,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> dict:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if session_id:
            body["session_id"] = session_id
        body.update(kwargs)
        r = self._client.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        return r.json()

    def write_mind_tree_node(
        self,
        path: str,
        description: str = "",
        settings: Optional[list[dict]] = None,
        lesson_text: Optional[str] = None,
        lesson_author: str = "trophic_seed",
        lesson_task_types: Optional[list[str]] = None,
        lesson_failure_modes: Optional[list[str]] = None,
    ) -> dict:
        body: dict[str, Any] = {
            "path": path,
            "description": description,
        }
        if settings:
            body["settings"] = settings
        if lesson_text:
            body["lesson_text"] = lesson_text
            body["lesson_author"] = lesson_author
            body["lesson_task_types"] = lesson_task_types or []
            body["lesson_failure_modes"] = lesson_failure_modes or []
        r = self._client.post("/v1/mind_tree/node", json=body)
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self._client.close()


# ── Per-species session lifecycle ─────────────────────────────────────────

@dataclass
class CachedSession:
    species_id: str
    session_id: str
    created_at: float
    expires_at_iso: str
    base_url: str

    def to_dict(self) -> dict:
        return {
            "species_id": self.species_id,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "expires_at_iso": self.expires_at_iso,
            "base_url": self.base_url,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CachedSession":
        return cls(
            species_id=d["species_id"],
            session_id=d["session_id"],
            created_at=float(d["created_at"]),
            expires_at_iso=d["expires_at_iso"],
            base_url=d["base_url"],
        )


class SpeciesSessionManager:
    """Manages (species_id → session_id) lifecycle.

    Persistence layer is Neo4j (via the deploy's Mind Tree). The cache here
    just remembers which session_id is the *current* live one for each
    species so we don't re-create per request.
    """

    def __init__(
        self,
        client: HexisClient,
        cache_dir: Path = SESSIONS_DIR,
        ttl_seconds: int = 86400,  # 24hr default; refresh on use
    ) -> None:
        self.client = client
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, species_id: str) -> Path:
        # File-system safe: no slashes, periods OK
        safe = species_id.replace("/", "_")
        return self.cache_dir / f"{safe}.json"

    def _load_cached(self, species_id: str) -> Optional[CachedSession]:
        p = self._cache_path(species_id)
        if not p.exists():
            return None
        try:
            return CachedSession.from_dict(json.loads(p.read_text()))
        except (json.JSONDecodeError, KeyError):
            return None

    def _save_cached(self, cs: CachedSession) -> None:
        self._cache_path(cs.species_id).write_text(
            json.dumps(cs.to_dict(), indent=2)
        )

    def get_or_create(
        self,
        species_id: str,
        species_description: str = "",
        preload_node_ids: Optional[list[str]] = None,
    ) -> str:
        """Return a live session_id for the species. Creates a new one if
        the cache is empty or the cached session is no longer alive on the
        server.
        """
        cached = self._load_cached(species_id)
        if cached and cached.base_url == self.client.base_url:
            # Verify it's still alive on the server
            existing = self.client.get_session(cached.session_id)
            if existing is not None:
                return cached.session_id

        # Cache miss or stale — create new
        result = self.client.create_session(
            domain="finance",
            user_request=f"trophic herbivore species: {species_id}",
            task_instruction=species_description or
                f"Classify financial news under species {species_id}.",
            ttl_seconds=self.ttl_seconds,
            preload_node_ids=preload_node_ids or [],
            metadata={"species_id": species_id, "role": "herbivore"},
        )
        cs = CachedSession(
            species_id=species_id,
            session_id=result["session_id"],
            created_at=time.time(),
            expires_at_iso=result["expires_at"],
            base_url=self.client.base_url,
        )
        self._save_cached(cs)
        return cs.session_id

    def invalidate(self, species_id: str) -> None:
        p = self._cache_path(species_id)
        if p.exists():
            p.unlink()
