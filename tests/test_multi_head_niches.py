"""Mission 03 — multi-head specialization tracking & diet-tag-free routing.

Validates that:
  - per-(head, slot) attention EMA gets updated inside attend()
  - `slot_specialization()` returns the right dict shape
  - heads attend to different slots when given varied queries (no
    constant argmax)
  - the trough doesn't read the diet_tags field on broadcasts — slot
    routing is purely Q·K (mixed diet_tags get routed by attention)

These are the acceptance tests for the trough-as-transformer migration:
once they pass, hardcoded diet_tags-driven routing can be deleted from
the runtime code-path.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from trophic.trough_attention import TroughAttention


@dataclass
class _FakeBroadcast:
    """Minimal duck-type for `TroughAttention.deposit()`. Carries a
    diet_tags field to confirm the trough does NOT read it."""
    id: str
    channel_embedding: list[float]
    diet_tags: list[str]


def _make_trough(hidden: int = 64, n_slots: int = 8, n_heads: int = 4, seed: int = 0) -> TroughAttention:
    return TroughAttention(
        hidden_size=hidden, n_slots=n_slots, n_heads=n_heads, out_seq_len=4, seed=seed,
    )


def _make_orthogonal_broadcasts(n: int, hidden: int) -> list[_FakeBroadcast]:
    """Build n broadcasts whose embeddings are mutually (near-)orthogonal,
    so each can serve as a clear target for a directed query.
    """
    torch.manual_seed(42)
    raw = torch.randn(n, hidden)
    # Gram-Schmidt for orthogonality.
    Q, _ = torch.linalg.qr(raw.T)
    vecs = Q.T[:n]
    return [
        _FakeBroadcast(
            id=f"b{i}",
            channel_embedding=vecs[i].tolist(),
            # Mixed diet_tags — the trough must ignore these.
            diet_tags=[f"from_kind_{i % 3}"],
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1. heads attend to different slots when given varied queries
# ---------------------------------------------------------------------------

def test_each_head_attends_to_different_slots():
    """With 4 mutually orthogonal slot embeddings and varied queries,
    the per-(head, slot) EMA should not collapse to a single
    head:slot mapping. Argmax across heads should produce more than one
    distinct head id over alive slots after enough attends.
    """
    H, N, n_heads = 64, 8, 4
    trough = _make_trough(hidden=H, n_slots=N, n_heads=n_heads, seed=11)

    # Hack: tie W_Q to W_K so identical query/key directions score high.
    with torch.no_grad():
        trough.W_Q.weight.copy_(trough.W_K.weight)

    bcs = _make_orthogonal_broadcasts(4, H)
    trough.deposit(bcs)

    # Run attends with each broadcast direction as the query in turn —
    # plus a few random queries — to give heads opportunity to
    # differentiate.
    for _ in range(30):
        for bc in bcs:
            q = torch.tensor(bc.channel_embedding)
            trough.attend(q, tau=0.5)
        for _ in range(2):
            trough.attend(torch.randn(H), tau=0.5)

    spec = trough.slot_specialization()
    head_per_slot = spec["head_per_slot"]
    # Inspect the alive slots only (first 4).
    alive_assignments = head_per_slot[:4]
    distinct = set(alive_assignments)
    assert len(distinct) >= 2, (
        f"expected at least 2 distinct heads across 4 alive slots, "
        f"got {alive_assignments}"
    )


# ---------------------------------------------------------------------------
# 2. trough ignores diet_tags — routing is pure Q·K
# ---------------------------------------------------------------------------

def test_diet_tag_unused():
    """Two slots with identical embeddings but DIFFERENT diet_tags must
    receive equal attention from a query — proving the trough doesn't
    read diet_tags.
    """
    H, N, n_heads = 32, 4, 4
    trough = _make_trough(hidden=H, n_slots=N, n_heads=n_heads, seed=22)
    with torch.no_grad():
        trough.W_Q.weight.copy_(trough.W_K.weight)

    target_vec = torch.randn(H)

    a = _FakeBroadcast(
        id="a",
        channel_embedding=target_vec.tolist(),
        diet_tags=["from_technical_herbivore"],
    )
    b = _FakeBroadcast(
        id="b",
        channel_embedding=target_vec.tolist(),
        diet_tags=["from_fundamental_herbivore"],  # different tag, same vec
    )
    trough.deposit([a, b])

    out = trough.attend(target_vec, tau=0.5)
    a_w = float(out.per_slot_attention[0].item())
    b_w = float(out.per_slot_attention[1].item())
    assert abs(a_w - b_w) < 1e-4, (
        f"diet_tag must not influence attention; got slot_a={a_w}, slot_b={b_w}"
    )


# ---------------------------------------------------------------------------
# 3. head_specialization buffer updates after attend()
# ---------------------------------------------------------------------------

def test_head_specialization_buffer_updates():
    """After several attends the buffer must differ from its zero init."""
    H, N, n_heads = 32, 4, 4
    trough = _make_trough(hidden=H, n_slots=N, n_heads=n_heads, seed=33)

    # Buffer starts at zero.
    assert trough.head_specialization.abs().sum().item() == 0.0
    assert trough.head_specialization.shape == (n_heads, N)

    bcs = _make_orthogonal_broadcasts(3, H)
    trough.deposit(bcs)

    for _ in range(5):
        trough.attend(torch.randn(H), tau=1.0)

    # Now non-zero, and only on alive slots (first 3).
    spec_t = trough.head_specialization
    assert spec_t.abs().sum().item() > 0.0
    # Slot 3 (dead) must remain at zero.
    assert spec_t[:, 3].abs().sum().item() == 0.0
    # Each head's row over alive slots should sum to ~one weight unit per
    # tick scaled by (1-alpha) summed over ticks (roughly bounded by 1).
    # Just confirm "small but nonzero".
    for h in range(n_heads):
        s = spec_t[h, :3].sum().item()
        assert s > 0.0, f"head {h} got zero attention on alive slots"


# ---------------------------------------------------------------------------
# 4. slot_specialization returns the expected dict shape
# ---------------------------------------------------------------------------

def test_slot_specialization_returns_expected_shape():
    H, N, n_heads = 32, 4, 4
    trough = _make_trough(hidden=H, n_slots=N, n_heads=n_heads, seed=44)
    bcs = _make_orthogonal_broadcasts(2, H)
    trough.deposit(bcs)
    trough.attend(torch.randn(H))

    spec = trough.slot_specialization()
    assert isinstance(spec, dict)
    assert set(spec.keys()) == {"head_per_slot", "specialization_scores"}

    head_per_slot = spec["head_per_slot"]
    scores = spec["specialization_scores"]

    # head_per_slot: list of length n_slots, each in [0, n_heads)
    assert isinstance(head_per_slot, list)
    assert len(head_per_slot) == N
    for h in head_per_slot:
        assert isinstance(h, int)
        assert 0 <= h < n_heads

    # specialization_scores: nested list [n_heads][n_slots]
    assert isinstance(scores, list)
    assert len(scores) == n_heads
    for row in scores:
        assert len(row) == N


# ---------------------------------------------------------------------------
# 5. multi-head attention shape + gradient sanity (mission file Step 1)
# ---------------------------------------------------------------------------

def test_multi_head_shape_and_gradient():
    """Mission file Testing Condition #1: gradient flows through the
    multi-head readout."""
    H, N, n_heads = 64, 4, 4
    trough = _make_trough(hidden=H, n_slots=N, n_heads=n_heads, seed=55)
    bcs = _make_orthogonal_broadcasts(4, H)
    trough.deposit(bcs)

    q = torch.randn(H, requires_grad=True)
    out = trough.attend(q)
    loss = out.output.sum()
    loss.backward()

    assert q.grad is not None
    assert q.grad.norm().item() > 0.0
    assert out.per_head_attention.shape == (n_heads, N)
