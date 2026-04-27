"""Retrieval embedder (Qwen3-Embedding-0.6B).

Separate from ModelHost because (a) different model, (b) different purpose:
these vectors are for cheap cosine ranking inside the appetite function, not
for inter-agent transmission.
"""
from __future__ import annotations

import hashlib
import threading
from typing import Optional

import numpy as np

from .config import ModelConfig, DEFAULT_CONFIG


class RetrievalEmbedder:
    _instance: Optional["RetrievalEmbedder"] = None
    _lock = threading.Lock()

    def __init__(self, cfg: ModelConfig | None = None):
        self.cfg = cfg or DEFAULT_CONFIG.model
        self._tok = None
        self._model = None
        self._dim: int = 1024  # corrected on real load
        if not self.cfg.mock:
            self._load()

    @classmethod
    def get(cls, cfg: ModelConfig | None = None) -> "RetrievalEmbedder":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(cfg)
            return cls._instance

    def _load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = dtype_map.get(self.cfg.dtype, torch.float16)
        self._tok = AutoTokenizer.from_pretrained(self.cfg.embed_model_id)
        self._model = AutoModel.from_pretrained(
            self.cfg.embed_model_id,
            torch_dtype=dtype,
            device_map=self.cfg.device,
        )
        self._model.eval()
        # The embedder reports its hidden size via config too.
        self._dim = self._model.config.hidden_size

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.cfg.mock:
            return [self._mock(t, self._dim) for t in texts]
        import torch

        out_vecs: list[list[float]] = []
        for t in texts:
            inputs = self._tok(t, return_tensors="pt", truncation=True, max_length=512).to(
                self._model.device
            )
            with torch.no_grad():
                out = self._model(**inputs)
            # Mean-pool last hidden state.
            last = out.last_hidden_state[0]
            mask = inputs["attention_mask"][0].unsqueeze(-1).float()
            pooled = (last * mask).sum(dim=0) / mask.sum().clamp(min=1.0)
            v = pooled.float().cpu().numpy()
            n = float(np.linalg.norm(v)) or 1.0
            out_vecs.append((v / n).tolist())
        return out_vecs

    @staticmethod
    def _mock(text: str, dim: int) -> list[float]:
        h = int(hashlib.sha256(("retr::" + text).encode()).hexdigest()[:12], 16)
        rng = np.random.default_rng(h)
        v = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).tolist()


def cosine(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))
