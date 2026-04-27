"""Mission 01 — TroughAttention behavior.

Covers the K/V container surface (deposit / evict / slot_state), the
attention readout (multi-head, null gate, target-norm rescale), and a
few smoke checks the downstream missions rely on (cumulative_attention
gets recorded; state_dict round-trips cleanly).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from trophic.trough_attention import TroughAttendOutput, TroughAttention


@dataclass
class _FakeBroadcast:
    id: str
    channel_embedding: list[float]


def _fake_broadcast(idx: int, hidden: int, vec: torch.Tensor | None = None) -> _FakeBroadcast:
    if vec is None:
        torch.manual_seed(1234 + idx)
        vec = torch.randn(hidden)
    return _FakeBroadcast(id=f"b{idx}", channel_embedding=vec.tolist())


# ---------------------------------------------------------------------------
# K/V container (deposit / evict / slot_state)
# ---------------------------------------------------------------------------

def test_deposit_into_free_slot():
    H, N = 64, 8
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=8, seed=0)
    bs = [_fake_broadcast(i, H) for i in range(4)]
    used = trough.deposit(bs)
    assert used == [0, 1, 2, 3]
    assert trough.alive.tolist() == [True] * 4 + [False] * 4
    assert trough.broadcast_ids[:4] == [b.id for b in bs]
    assert trough.broadcast_ids[4:] == [None, None, None, None]
    state = trough.slot_state()
    assert state["n_alive"] == 4
    assert len(state["cumulative_attention"]) == N


def test_deposit_overflows_when_full():
    H, N = 32, 4
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=4, seed=0)
    trough.deposit([_fake_broadcast(i, H) for i in range(N)])
    try:
        trough.deposit([_fake_broadcast(99, H)])
    except RuntimeError as e:
        assert "free slot" in str(e) or "phase2" in str(e)
    else:
        raise AssertionError("expected deposit overflow to raise")


def test_evict_frees_slot_and_lets_new_deposit_reuse():
    H, N = 32, 4
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=4, seed=0)
    bs = [_fake_broadcast(i, H) for i in range(2)]
    trough.deposit(bs)
    trough.evict([0])
    assert trough.alive.tolist() == [False, True, False, False]
    assert trough.broadcast_ids[0] is None
    used = trough.deposit([_fake_broadcast(99, H)])
    # First free slot is index 0 again.
    assert used == [0]
    assert trough.broadcast_ids[0] == "b99"


# ---------------------------------------------------------------------------
# attention readout shape / smoke
# ---------------------------------------------------------------------------

def test_attend_smoke_shapes():
    H, N = 64, 8
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=8, out_seq_len=4, seed=1)
    trough.deposit([_fake_broadcast(i, H) for i in range(3)])
    q = torch.randn(H)
    out = trough.attend(q)
    assert isinstance(out, TroughAttendOutput)
    assert out.output.shape == (4, H)
    assert out.per_slot_attention.shape == (N,)
    assert out.per_head_attention.shape == (8, N)
    assert 0.0 <= out.null_prob <= 1.0
    # Sum of per-slot attention plus null_prob should be ~1 (averaged
    # across heads it's exactly 1).
    s = float(out.per_slot_attention.sum().item()) + out.null_prob
    assert abs(s - 1.0) < 1e-4, s
    # Selected slots are exactly the alive ids, ordered by weight.
    assert sorted(out.selected_slot_ids) == [0, 1, 2]


def test_attend_picks_high_similarity_slot():
    """Deposit two broadcasts; one matches the query direction, the other is
    nearly orthogonal. Verify the matching slot wins per_slot_attention.

    We also nudge the query to be aligned in the *projected* space (post W_K)
    by using the same vector as the broadcast embedding — at scaled-small
    init W_K is near-random small but Q·K still leans toward parallel pairs.
    """
    torch.manual_seed(0)
    H, N = 64, 8
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=8, seed=2)

    # Run an "alignment" hack: copy W_K → W_Q so similar embeddings score
    # high pre-softmax. This isolates the cross-attention math from the
    # randomness of init.
    with torch.no_grad():
        trough.W_Q.weight.copy_(trough.W_K.weight)

    target_vec = torch.randn(H)
    other_vec = torch.randn(H)
    # Force orthogonality.
    other_vec = other_vec - target_vec * (other_vec @ target_vec) / (target_vec @ target_vec)

    trough.deposit([
        _FakeBroadcast(id="match", channel_embedding=target_vec.tolist()),
        _FakeBroadcast(id="orth", channel_embedding=other_vec.tolist()),
    ])
    out = trough.attend(target_vec, tau=0.5)
    # Slot 0 (the match) should pull more attention than slot 1.
    assert out.per_slot_attention[0] > out.per_slot_attention[1], (
        out.per_slot_attention.tolist()
    )


def test_attend_null_gate_fires_when_empty():
    H, N = 32, 4
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=4, seed=3)
    out = trough.attend(torch.randn(H))
    # Empty trough → null_prob should be exactly 1.0 (only the null row
    # exists in the softmax).
    assert out.null_prob == 1.0
    assert out.selected_slot_ids == []
    assert out.per_slot_attention.sum().item() == 0.0


# ---------------------------------------------------------------------------
# norm / cumulative attention
# ---------------------------------------------------------------------------

def test_attend_norm_matches_target_norm():
    H, N = 32, 4
    target = 7.5
    trough = TroughAttention(
        hidden_size=H, n_slots=N, n_heads=4, target_norm=target, seed=4
    )
    trough.deposit([_fake_broadcast(i, H) for i in range(2)])
    out = trough.attend(torch.randn(H))
    norms = out.output.norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, target), atol=1e-3)


def test_cumulative_attention_recorded_after_attend():
    H, N = 64, 8
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=8, seed=5)
    trough.deposit([_fake_broadcast(i, H) for i in range(4)])
    assert trough.cumulative_attention.sum().item() == 0.0
    trough.attend(torch.randn(H))
    # Some attention must have landed on the alive slots.
    assert trough.cumulative_attention.sum().item() > 0
    # And only on alive slots.
    assert trough.cumulative_attention[4:].sum().item() == 0


def test_state_dict_round_trip():
    H, N = 32, 4
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=4, seed=6)
    trough.deposit([_fake_broadcast(i, H) for i in range(2)])
    trough.attend(torch.randn(H))

    sd = trough.state_dict()
    other = TroughAttention(hidden_size=H, n_slots=N, n_heads=4, seed=42)
    other.load_state_dict(sd)
    # Param-level equality.
    for k in sd:
        assert torch.equal(sd[k], other.state_dict()[k]), k


def test_attend_lifecycle_stub_ok():
    H, N = 32, 4
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=4, seed=7)
    info = trough.step_lifecycle()
    assert "deaths" in info
    assert info["deaths"] == 0
