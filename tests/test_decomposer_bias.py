"""Mission 05 (phase2-D) — Decomposer bias path.

Covers:
  - Decomposer is initialised so its output is exactly zero (so the
    decomposer is a no-op until gradients flow).
  - A negative bias on one slot redirects attention to other slots.
  - The output is bounded by ±max_bias for any input.
  - `set_pending_bias` is consumed exactly once: the bias modifies the
    next `attend()` call's logits but not the one after.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from trophic.decomposer import Decomposer
from trophic.trough_attention import TroughAttention


@dataclass
class _FakeBroadcast:
    id: str
    channel_embedding: list[float]


def _fake_broadcast(idx: int, hidden: int) -> _FakeBroadcast:
    torch.manual_seed(1234 + idx)
    return _FakeBroadcast(id=f"b{idx}", channel_embedding=torch.randn(hidden).tolist())


# ---------------------------------------------------------------------------
# Decomposer module — output behavior
# ---------------------------------------------------------------------------

def test_decomposer_zero_init_zero_bias():
    """A freshly-initialised Decomposer with random inputs should produce
    bias of magnitude essentially zero — its output head is zero-init."""
    torch.manual_seed(0)
    d = Decomposer(hidden_size=64, n_slots=8)
    j = torch.randn(64)
    lineage = torch.softmax(torch.randn(8), dim=0)
    bias = d(j, lineage)
    assert bias.shape == (8,)
    assert bias.abs().max().item() < 1e-4, bias.abs().max().item()


def test_decomposer_bias_clipped_to_max():
    """Even with pathologically large inputs, bias stays in [-max_bias,
    +max_bias] thanks to the tanh squashing."""
    torch.manual_seed(0)
    max_bias = 2.0
    d = Decomposer(hidden_size=32, n_slots=4, max_bias=max_bias)
    # Force the head to non-zero so we can probe the bound (otherwise
    # zero-init would hide the clipping).
    with torch.no_grad():
        d.head.weight.normal_(0.0, 5.0)
        d.head.bias.normal_(0.0, 5.0)
    # Pathological inputs.
    j = torch.randn(32) * 1000.0
    lineage = torch.randn(4) * 1000.0
    bias = d(j, lineage)
    assert bias.shape == (4,)
    assert bias.abs().max().item() <= max_bias + 1e-5, bias.abs().max().item()


def test_decomposer_batched_input():
    """Decomposer should accept batched inputs and return batched output."""
    torch.manual_seed(0)
    d = Decomposer(hidden_size=32, n_slots=4, max_bias=2.0)
    with torch.no_grad():
        d.head.weight.normal_(0.0, 1.0)
        d.head.bias.normal_(0.0, 1.0)
    j = torch.randn(3, 32)
    lineage = torch.randn(3, 4)
    bias = d(j, lineage)
    assert bias.shape == (3, 4)
    assert bias.abs().max().item() <= 2.0 + 1e-5


# ---------------------------------------------------------------------------
# Bias path through TroughAttention.attend()
# ---------------------------------------------------------------------------

def test_bias_redirects_attention_away_from_penalised_slot():
    """Strong negative bias on slot 0 should push attention toward slot 1."""
    torch.manual_seed(0)
    H, N = 32, 4
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=4, seed=10)
    trough.deposit([_fake_broadcast(i, H) for i in range(2)])

    q = torch.randn(H)

    # Baseline: no bias.
    out_baseline = trough.attend(q)
    a0_base = float(out_baseline.per_slot_attention[0].item())
    a1_base = float(out_baseline.per_slot_attention[1].item())

    # Now apply a strongly-negative bias on slot 0.
    bias = torch.zeros(N)
    bias[0] = -10.0
    out_biased = trough.attend(q, external_bias=bias)
    a0_biased = float(out_biased.per_slot_attention[0].item())
    a1_biased = float(out_biased.per_slot_attention[1].item())

    # Slot 0 should lose attention; slot 1 should gain.
    assert a0_biased < a0_base, (a0_biased, a0_base)
    assert a1_biased > a1_base, (a1_biased, a1_base)


def test_pending_bias_consumed_once():
    """`set_pending_bias` should only affect the very next `attend()`.
    A second consecutive `attend()` (with the same query) must produce
    the no-bias logits."""
    torch.manual_seed(0)
    H, N = 32, 4
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=4, seed=11)
    trough.deposit([_fake_broadcast(i, H) for i in range(2)])

    q = torch.randn(H)

    # Baseline (no bias).
    baseline_per_slot = trough.attend(q).per_slot_attention.clone()

    # Stage a strong bias and attend twice.
    bias = torch.zeros(N)
    bias[0] = -10.0
    trough.set_pending_bias(bias)
    first_per_slot = trough.attend(q).per_slot_attention.clone()
    second_per_slot = trough.attend(q).per_slot_attention.clone()

    # First attend should be different from baseline.
    assert not torch.allclose(first_per_slot, baseline_per_slot, atol=1e-3), (
        first_per_slot.tolist(), baseline_per_slot.tolist()
    )
    # Second attend, with bias already consumed, should match baseline.
    assert torch.allclose(second_per_slot, baseline_per_slot, atol=1e-5), (
        second_per_slot.tolist(), baseline_per_slot.tolist()
    )
    # And `_pending_bias` should be cleared.
    assert trough._pending_bias is None


def test_pending_bias_overridden_by_explicit_kwarg():
    """If the caller passes `external_bias` directly, the staged
    `_pending_bias` is ignored — but is still cleared (consume-once)."""
    torch.manual_seed(0)
    H, N = 32, 4
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=4, seed=12)
    trough.deposit([_fake_broadcast(i, H) for i in range(2)])

    q = torch.randn(H)

    # Stage one bias, but pass a different one explicitly.
    trough.set_pending_bias(torch.full((N,), -10.0))
    explicit_bias = torch.zeros(N)
    out_explicit = trough.attend(q, external_bias=explicit_bias)

    # Bias all-zeros is effectively no bias — match the no-bias baseline.
    out_no_bias = trough.attend(q)
    assert torch.allclose(
        out_explicit.per_slot_attention,
        out_no_bias.per_slot_attention,
        atol=1e-5,
    )
    # Pending bias was cleared by the first call.
    assert trough._pending_bias is None


def test_set_pending_bias_validates_shape():
    H, N = 16, 4
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=2, seed=13)
    try:
        trough.set_pending_bias(torch.zeros(N + 1))
    except ValueError as e:
        assert "shape" in str(e).lower()
    else:
        raise AssertionError("expected ValueError on bad bias shape")
    # Passing None is a no-op clear.
    trough.set_pending_bias(None)
    assert trough._pending_bias is None
