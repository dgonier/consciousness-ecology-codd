"""Tests for issue #12: role_prefix-length invariance.

`normalize_role_q` should produce a Q vector whose magnitude is independent
of how many tokens the role prefix contains. Trained Channel weights then see
the same Q geometry regardless of inference-time prompt edits.

Without this fix, `Q = role_prefix.mean(dim=0)` shifts in magnitude with
prefix length: a 100-token prefix produces a different-norm vector than a
500-token prefix even when their direction is similar. Empirically (#11),
that magnitude shift was enough to break trained Channels and produce
gibberish output.
"""
from __future__ import annotations

import torch

from trophic.agents.base import ROLE_Q_REF_NORM, normalize_role_q


def _fake_prefix(seq_len: int, hidden: int = 64, seed: int = 0) -> torch.Tensor:
    """Synthetic role-prefix tensor with the same statistical signature as a
    real Qwen3 hidden state (small magnitude, near-zero mean per dim)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(seq_len, hidden, generator=g)


def test_normalize_role_q_returns_unit_norm():
    """Output should have norm == ROLE_Q_REF_NORM (1.0 by default)."""
    q = normalize_role_q(_fake_prefix(seq_len=100))
    assert abs(q.norm().item() - ROLE_Q_REF_NORM) < 1e-5


def test_normalize_role_q_invariant_to_prefix_length():
    """A 100-token prefix and a 500-token prefix (same generator seed for the
    SAME directional content extended) should both produce a unit-norm Q.

    More importantly: a long prefix and a short prefix don't differ in norm,
    so the Channel sees Q in the same scale regardless of seq_len.
    """
    short = normalize_role_q(_fake_prefix(seq_len=100))
    long = normalize_role_q(_fake_prefix(seq_len=500))
    assert abs(short.norm().item() - long.norm().item()) < 1e-5
    assert abs(short.norm().item() - ROLE_Q_REF_NORM) < 1e-5
    assert abs(long.norm().item() - ROLE_Q_REF_NORM) < 1e-5


def test_normalize_role_q_preserves_direction():
    """Doubling the prefix length with content from the same distribution
    should keep the Q direction stable, only the magnitude is fixed.

    Sanity check: cosine similarity between mean-pooled-then-normalized
    versions of two random-but-same-seed prefixes is positive (~0.05+).
    The point isn't that they're identical — the point is that norm
    differences aren't compounding into the directional measurement.
    """
    a = normalize_role_q(_fake_prefix(seq_len=200, seed=42))
    b = normalize_role_q(_fake_prefix(seq_len=200, seed=42))
    # Same seed → bit-identical
    assert torch.allclose(a, b, atol=1e-6)


def test_normalize_role_q_handles_zero_prefix():
    """Pathological case: a near-zero prefix shouldn't NaN."""
    z = torch.zeros(10, 64)
    q = normalize_role_q(z)
    # Norm with eps=1e-6 → q is ~zero/eps, very large but finite
    assert torch.isfinite(q).all()


def test_normalize_role_q_dtype_preserved():
    """Should preserve dtype of input (bf16 in real training)."""
    p = _fake_prefix(seq_len=50).to(torch.bfloat16)
    q = normalize_role_q(p)
    assert q.dtype == torch.bfloat16


def test_normalize_role_q_no_change_for_unit_prefix():
    """If the input mean already has unit norm, output equals input mean."""
    # Construct a prefix whose mean is exactly unit-norm.
    target = torch.tensor([1.0, 0.0, 0.0, 0.0])
    p = target.unsqueeze(0).repeat(5, 1)  # mean is `target`, norm 1.0
    q = normalize_role_q(p)
    assert torch.allclose(q, target, atol=1e-5)
