"""Mission 06 (phase3-A) — softmax temperature τ schedules.

Two simple anneal schedules for the cross-attention softmax temperature
used inside `TroughAttention.attend()`:

  - `cosine_tau`: smooth half-cosine from `tau_start` down to `tau_end`
  - `linear_tau`: linear interpolation from `tau_start` to `tau_end`

Ecologically: τ is "exploration temperature" on the trough's attention
softmax. Warm at start (egalitarian — every alive slot gets some
attention; new niches can take hold) and cool over training (sharper
selection — winners take more share). Both schedules clamp `step` into
`[0, total_steps]` so callers can over-step at the end without blowing
up.
"""
from __future__ import annotations

import math


def cosine_tau(
    step: int,
    total_steps: int,
    tau_start: float = 2.0,
    tau_end: float = 0.5,
) -> float:
    """Cosine anneal τ from `tau_start` (warm) down to `tau_end` (cool).

    At step=0 returns `tau_start`. At step>=total_steps returns `tau_end`.
    Smoothly monotone-decreasing in between (assuming tau_start > tau_end).
    """
    if total_steps <= 0:
        return float(tau_end)
    progress = min(max(step, 0), total_steps) / float(total_steps)
    return float(tau_end) + 0.5 * (float(tau_start) - float(tau_end)) * (
        1.0 + math.cos(math.pi * progress)
    )


def linear_tau(
    step: int,
    total_steps: int,
    tau_start: float = 2.0,
    tau_end: float = 0.5,
) -> float:
    """Linear interpolation from `tau_start` to `tau_end`.

    At step=0 returns `tau_start`. At step>=total_steps returns `tau_end`.
    """
    if total_steps <= 0:
        return float(tau_end)
    progress = min(max(step, 0), total_steps) / float(total_steps)
    return float(tau_start) + (float(tau_end) - float(tau_start)) * progress
