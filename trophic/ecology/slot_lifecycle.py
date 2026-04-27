"""Pure functions for slot ecological lifecycle (Mission 02).

These are deliberately stateless / tensor-in tensor-out so they can be
unit tested without instantiating a `TroughAttention`. The bookkeeping
state (`cumulative_attention`, `under_threshold_ticks`, `alive`) lives
on the trough; this module just describes how those tensors evolve.

Usage from `TroughAttention.step_lifecycle()`:

    new_under = mark_underperforming(
        cumulative_attention, alive, epsilon, under_threshold_ticks
    )
    new_alive, killed = kill_dead_slots(new_under, alive, n_patience)

The trough then commits `new_alive` and zeroes the killed slots' state.
"""
from __future__ import annotations

import torch


def mark_underperforming(
    cumulative_attention: torch.Tensor,
    alive: torch.Tensor,
    epsilon: float,
    under_threshold_ticks: torch.Tensor,
) -> torch.Tensor:
    """Increment per-slot under-threshold counters; reset above-threshold.

    A slot is "under" if it is alive AND its (EMA-smoothed) cumulative
    attention is strictly below `epsilon`. Such slots have their counter
    bumped by 1; any other slot's counter is reset to 0.

    Dead slots' counters are also forced to 0 — there's no point
    counting starvation on an empty slot.

    Returns a fresh tensor (same shape/dtype as `under_threshold_ticks`)
    so callers can choose how to commit it back to the trough buffer.
    """
    if cumulative_attention.shape != alive.shape:
        raise ValueError(
            f"cumulative_attention shape {tuple(cumulative_attention.shape)} "
            f"!= alive shape {tuple(alive.shape)}"
        )
    if under_threshold_ticks.shape != alive.shape:
        raise ValueError(
            f"under_threshold_ticks shape {tuple(under_threshold_ticks.shape)} "
            f"!= alive shape {tuple(alive.shape)}"
        )

    under = (cumulative_attention < epsilon) & alive
    new_counts = torch.where(
        under,
        under_threshold_ticks + 1,
        torch.zeros_like(under_threshold_ticks),
    )
    return new_counts


def kill_dead_slots(
    under_threshold_ticks: torch.Tensor,
    alive: torch.Tensor,
    n_patience: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decide which slots starved out this tick.

    A slot is killed if it is alive AND its under-threshold counter has
    reached `n_patience` consecutive ticks. Returns `(new_alive,
    killed_mask)` where:

      - `new_alive` is the updated alive mask after killing
      - `killed_mask[i]` is True iff slot `i` just died this tick
    """
    if under_threshold_ticks.shape != alive.shape:
        raise ValueError(
            f"under_threshold_ticks shape {tuple(under_threshold_ticks.shape)} "
            f"!= alive shape {tuple(alive.shape)}"
        )
    if n_patience < 0:
        raise ValueError(f"n_patience must be >= 0, got {n_patience}")

    killed = (under_threshold_ticks >= n_patience) & alive
    new_alive = alive & ~killed
    return new_alive, killed


def niche_aware_spawn(
    underserved_q: torch.Tensor,
    W_K: torch.Tensor,
    target_norm: float,
    noise_scale: float = 0.02,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate a new V vector whose K = W_K @ V lands near `underserved_q`.

    Mission 04: when a slot dies, we want the replacement to occupy a
    high-similarity position with the consumer query that's been most
    starved (lowest max attention across slots in recent history). We
    can't truly invert `W_K` — it's typically square but we don't trust
    it to be full-rank — so we use the Moore-Penrose pseudo-inverse:

        new_v ≈ W_K_pinv @ underserved_q

    Then add small isotropic noise (so successive spawns don't collapse
    to the same V) and rescale to `target_norm` so the new slot looks
    like a real broadcast deposit.

    Args:
        underserved_q: shape `[hidden]`, the projected consumer query
            we want the new slot to satisfy.
        W_K: shape `[hidden, hidden]`, the K projection matrix
            (`TroughAttention.W_K.weight`).
        target_norm: norm to rescale the result to (matches the trough's
            `target_norm` so deposits/spawns share the same magnitude).
        noise_scale: stddev of the additive Gaussian noise. Default 0.02
            keeps niche-locking strong while preventing exact duplicates.
        generator: optional torch Generator for deterministic noise.

    Returns:
        new_v: shape `[hidden]`, suitable for writing into `V_store[i]`.
    """
    if underserved_q.dim() != 1:
        raise ValueError(
            f"underserved_q must be 1D, got shape {tuple(underserved_q.shape)}"
        )
    if W_K.dim() != 2:
        raise ValueError(f"W_K must be 2D, got shape {tuple(W_K.shape)}")
    if W_K.shape[0] != underserved_q.shape[0]:
        raise ValueError(
            f"W_K rows ({W_K.shape[0]}) must match underserved_q dim "
            f"({underserved_q.shape[0]})"
        )

    device = W_K.device
    dtype = W_K.dtype
    q = underserved_q.to(device=device, dtype=dtype)
    W_K_pinv = torch.linalg.pinv(W_K)
    new_v = W_K_pinv @ q
    if noise_scale > 0.0:
        if generator is not None:
            noise = torch.empty_like(new_v).normal_(0.0, 1.0, generator=generator)
        else:
            noise = torch.randn_like(new_v)
        new_v = new_v + noise_scale * noise
    new_v = new_v * (target_norm / (new_v.norm() + 1e-8))
    return new_v
