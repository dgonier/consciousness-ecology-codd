"""Hidden-state regression loss + target cache.

For training step:
  1. Get the target text from oracle.py for the agent + meal.
  2. Look up (cached) the target's pooled hidden state, computed via Qwen.
  3. MSE between the agent's actual pooled hidden state and the target.

Cache is keyed by exact target text — same string → same target hidden, no
recomputation. The cache lives in-process for the training run; for very
long runs we could persist to disk.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..model_host import ModelHost


@dataclass
class TargetCache:
    host: ModelHost
    pool: str = "mean"
    _cache: dict[str, torch.Tensor] = field(default_factory=dict)

    def get(self, text: str) -> torch.Tensor:
        if text not in self._cache:
            self._cache[text] = self.host.text_to_hidden_pooled_target(text, pool=self.pool)
        return self._cache[text]

    def __len__(self) -> int:
        return len(self._cache)


def regression_loss(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE in the model's hidden space.

    `actual` is a live tensor on the model's device with grads.
    `target` is detached and may be on cpu — we move/cast to match.
    """
    target = target.to(device=actual.device, dtype=actual.dtype)
    return F.mse_loss(actual, target)
