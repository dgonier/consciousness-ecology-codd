"""Shared agent scaffolding."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import torch

from ..types import AgentState


def new_agent_id(role: str, kind: str) -> str:
    return f"{role}.{kind}.{uuid.uuid4().hex[:8]}"


# Issue #12: prompt-length fragility fix. Trained Channel weights were learning
# against `role_prefix.mean(dim=0)` which has length-dependent magnitude (sum of
# token vectors / num_tokens — a longer prefix samples more directions but with
# the same mean-of-residuals behavior, so its L2 norm scales weakly but
# meaningfully with length). Empirically: doubling the prefix length shifts the
# Q's location enough to break trained Channels (#11 surfaced this).
#
# The fix is to normalize `role_q` to a fixed reference norm BEFORE adding the
# input-conditioning term `prey_t.mean()`. This decouples the trained Q-geometry
# from prefix length while preserving the directional information that
# different prompts carry.
#
# Reference norm chosen as a constant (1.0) rather than the Channel's
# target_norm because (a) target_norm is for output rescaling, not Q
# construction, and (b) downstream Channel.W_Q can scale freely if needed.
ROLE_Q_REF_NORM: float = 1.0


def normalize_role_q(role_prefix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Build a length-invariant Q-component from a role-prefix hidden tensor.

    Args:
        role_prefix: [seq_len, hidden] — the agent's role prefix hidden states
                     (typically host.encode_role_prefix output).
        eps: numerical floor for the divisor.

    Returns:
        [hidden] vector with L2 norm == ROLE_Q_REF_NORM, regardless of seq_len.

    Why mean-then-normalize (not normalize-then-mean):
      - mean preserves directional information across the prefix's tokens
      - normalize after mean ensures fixed magnitude regardless of how many
        tokens the prefix has, which is the load-bearing property for #12
    """
    pooled = role_prefix.mean(dim=0)
    n = pooled.norm()
    return pooled * (ROLE_Q_REF_NORM / (n + eps))


@dataclass
class BaseAgent:
    id: str
    kind: str
    role: str
    state: AgentState = field(init=False)

    def __post_init__(self):
        self.state = AgentState(id=self.id, role=self.role, kind=self.kind)
