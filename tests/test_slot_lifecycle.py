"""Mission 02 — slot lifecycle (attention-decay + death).

Two layers:
  1. Pure-function tests on `mark_underperforming` / `kill_dead_slots`
     (no `TroughAttention` instance needed).
  2. Integration tests that drive `TroughAttention.attend()` repeatedly
     and check that starved slots actually die after `n_patience` ticks.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from trophic.ecology.slot_lifecycle import (
    kill_dead_slots,
    mark_underperforming,
    niche_aware_spawn,
)
from trophic.trough_attention import TroughAttention


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeBroadcast:
    id: str
    channel_embedding: list[float]


def _fake_broadcast(idx: int, hidden: int, vec: torch.Tensor | None = None) -> _FakeBroadcast:
    if vec is None:
        torch.manual_seed(7777 + idx)
        vec = torch.randn(hidden)
    return _FakeBroadcast(id=f"b{idx}", channel_embedding=vec.tolist())


# ---------------------------------------------------------------------------
# pure-function tests
# ---------------------------------------------------------------------------

def test_mark_underperforming_increments():
    """Slots below ε get +1; slots above reset to 0; dead slots stay 0."""
    cum = torch.tensor([0.001, 0.5, 0.0001, 0.05])  # below, above, below, above
    alive = torch.tensor([True, True, True, True])
    under = torch.tensor([3, 4, 0, 7], dtype=torch.long)
    new = mark_underperforming(cum, alive, epsilon=0.01, under_threshold_ticks=under)
    assert new.tolist() == [4, 0, 1, 0]


def test_mark_underperforming_dead_slots_reset():
    """Dead slots ALWAYS get reset to 0 even if their stale EMA was below ε."""
    cum = torch.tensor([0.001, 0.001, 0.001, 0.001])
    alive = torch.tensor([True, False, True, False])
    under = torch.tensor([2, 5, 4, 9], dtype=torch.long)
    new = mark_underperforming(cum, alive, epsilon=0.01, under_threshold_ticks=under)
    # Dead slots reset; alive sub-ε slots increment.
    assert new.tolist() == [3, 0, 5, 0]


def test_kill_dead_slots_after_patience():
    """A slot reaches the patience counter → killed mask fires."""
    under = torch.tensor([0, 5, 4, 6], dtype=torch.long)
    alive = torch.tensor([True, True, True, True])
    new_alive, killed = kill_dead_slots(under, alive, n_patience=5)
    assert killed.tolist() == [False, True, False, True]
    assert new_alive.tolist() == [True, False, True, False]


def test_kill_dead_slots_ignores_dead_slots():
    """A slot that's already dead can't 're-die'."""
    under = torch.tensor([0, 9, 9, 9], dtype=torch.long)
    alive = torch.tensor([True, False, True, False])
    new_alive, killed = kill_dead_slots(under, alive, n_patience=5)
    assert killed.tolist() == [False, False, True, False]
    assert new_alive.tolist() == [True, False, False, False]


# ---------------------------------------------------------------------------
# integration tests (drive a real TroughAttention)
# ---------------------------------------------------------------------------

def test_step_lifecycle_kills_unattended_slot():
    """Deposit 4 broadcasts; attend with a Q that aligns with slot 0;
    after enough ticks the other 3 slots should starve and die."""
    H, N = 64, 8
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=8, seed=11)
    # Make Q·K ~ direction-aligned by copying W_K → W_Q (same trick as
    # test_attend_picks_high_similarity_slot in test_trough_attention.py).
    with torch.no_grad():
        trough.W_Q.weight.copy_(trough.W_K.weight)

    target_vec = torch.randn(H)
    others = [torch.randn(H) for _ in range(3)]
    # Make the others orthogonal to target so they really get no juice.
    for i in range(3):
        coeff = (others[i] @ target_vec) / (target_vec @ target_vec)
        others[i] = others[i] - coeff * target_vec

    bs = [_FakeBroadcast(id="match", channel_embedding=target_vec.tolist())]
    for i, v in enumerate(others):
        bs.append(_FakeBroadcast(id=f"orth{i}", channel_embedding=v.tolist()))
    trough.deposit(bs)
    assert int(trough.alive.sum().item()) == 4

    # Use a low temperature to *really* concentrate attention on the match.
    trough.epsilon_death = 0.05
    trough.n_patience = 5
    # Mission 04: opt out of fixed-population spawn so we can observe
    # actual kills (default would respawn into killed slots immediately,
    # keeping alive.sum() at 4).
    trough.population_strategy = "none"

    # Drive enough ticks that the EMA on the orthogonal slots collapses
    # below 0.05 and stays there for n_patience consecutive ticks.
    for _ in range(40):
        trough.attend(target_vec, tau=0.1)
        trough.step_lifecycle()

    n_alive_after = int(trough.alive.sum().item())
    assert n_alive_after < 4, (
        f"expected at least one slot to die under starvation, got "
        f"n_alive={n_alive_after}, cum={trough.cumulative_attention.tolist()}, "
        f"under={trough.under_threshold_ticks.tolist()}"
    )
    # The matching slot should still be alive.
    assert bool(trough.alive[0].item())
    # killed slots should have V_store zeroed.
    for sid in range(N):
        if not bool(trough.alive[sid].item()) and bool(trough.dead[sid].item()):
            assert torch.allclose(
                trough.V_store[sid], torch.zeros(H), atol=1e-6
            )
            assert trough.broadcast_ids[sid] is None


def test_revival_resets_counter():
    """A slot dropped below ε for a few ticks but then re-attended has its
    counter reset → it survives."""
    H, N = 32, 4
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=4, seed=12)
    trough.deposit([_fake_broadcast(i, H) for i in range(2)])
    # Set a high epsilon so the natural EMA values are below it; then
    # give a manual attention boost on the second slot to trigger reset.
    trough.epsilon_death = 0.5
    trough.n_patience = 5

    # 3 ticks of no attention (EMA values are small post-init, so they'll
    # fall below ε immediately for both slots).
    for _ in range(3):
        trough.attend(torch.randn(H))
        trough.step_lifecycle()
    assert (trough.under_threshold_ticks[:2] >= 1).all()

    # Now manually push slot 1's cumulative_attention above ε to simulate
    # a strong attention spike, then call lifecycle. Counter should reset.
    trough.cumulative_attention[1] = 0.99
    trough.step_lifecycle()
    assert int(trough.under_threshold_ticks[1].item()) == 0
    assert bool(trough.alive[1].item())  # slot 1 is alive


def test_age_increments_only_for_alive():
    """Dead slots don't age."""
    H, N = 32, 4
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=4, seed=13)
    trough.deposit([_fake_broadcast(i, H) for i in range(2)])
    # Two slots alive (0,1), two dead (2,3).
    age_before = trough.age.detach().clone()
    trough.attend(torch.randn(H))
    age_after = trough.age.detach()
    # Alive slots' age incremented by 1, dead slots' stay the same.
    assert int(age_after[0].item()) == int(age_before[0].item()) + 1
    assert int(age_after[1].item()) == int(age_before[1].item()) + 1
    assert int(age_after[2].item()) == int(age_before[2].item())
    assert int(age_after[3].item()) == int(age_before[3].item())


def test_step_lifecycle_returns_stats():
    """`step_lifecycle()` returns the documented stats keys."""
    H, N = 32, 4
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=4, seed=14)
    trough.deposit([_fake_broadcast(i, H) for i in range(2)])
    info = trough.step_lifecycle()
    for k in ("n_killed", "n_alive", "killed_ids", "deaths", "ages"):
        assert k in info, f"missing key {k!r} in {info}"
    assert info["n_killed"] == 0
    assert info["n_alive"] == 2
    assert info["killed_ids"] == []
    # Backwards-compat aliases kept by Mission 02.
    assert info["deaths"] == 0


def test_step_lifecycle_empty_trough_safe():
    """Running step_lifecycle on a trough that has never been attended
    must not raise — many start-up ticks fall in this regime."""
    H, N = 32, 4
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=4, seed=15)
    info = trough.step_lifecycle()
    assert info["n_alive"] == 0
    assert info["n_killed"] == 0


def test_ema_actually_smooths():
    """Sanity: cumulative_attention shrinks toward zero when a slot stops
    being attended (this is what makes death possible in finite time)."""
    H, N = 32, 4
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=4, seed=16)
    trough.deposit([_fake_broadcast(i, H) for i in range(2)])
    # One attend to seed the EMA.
    trough.attend(torch.randn(H))
    seed_cum = trough.cumulative_attention[:2].detach().clone()
    assert (seed_cum > 0).all(), seed_cum

    # Now manually zero out alive — simulating "no more attention"
    # delivered to those slots without going through attend(). We just
    # multiply cumulative_attention by alpha n times to mimic the EMA
    # decay path.
    cum = seed_cum.clone()
    alpha = trough.alpha_decay
    for _ in range(20):
        cum = cum * alpha
    # After 20 decays from ~0.025 with α=0.9 we get ~0.025*0.12 = ~0.003,
    # which is below default epsilon_death=0.01.
    assert (cum < trough.epsilon_death).all(), cum


# ---------------------------------------------------------------------------
# Mission 04 — spawn / niche-aware reallocation tests
# ---------------------------------------------------------------------------

def _force_kill(trough: TroughAttention, slot_id: int) -> None:
    """Manually mark a slot dead without going through the EMA path —
    forces the kill that the natural death pipeline would produce so
    spawn-side tests can run without depending on EMA collapse timing."""
    trough.alive[slot_id] = False
    trough.dead[slot_id] = True
    trough.V_store[slot_id] = 0.0
    trough.cumulative_attention[slot_id] = 0.0
    trough.under_threshold_ticks[slot_id] = 0
    trough.age[slot_id] = 0
    trough.head_specialization[:, slot_id] = 0.0
    trough.broadcast_ids[slot_id] = None


def test_spawn_into_dead_slot_resets_alive_to_true():
    """After force-killing a slot and calling spawn_into_dead_slot,
    `alive[slot]` flips back to True and per-slot bookkeeping resets."""
    H, N = 32, 4
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=4, seed=21)
    trough.deposit([_fake_broadcast(i, H) for i in range(N)])
    # Drive a few attends so the recent_queries ring is populated.
    for _ in range(5):
        trough.attend(torch.randn(H))

    target_slot = 2
    _force_kill(trough, target_slot)
    assert not bool(trough.alive[target_slot].item())

    info = trough.spawn_into_dead_slot(target_slot)
    assert info["spawned"] is True
    assert info["slot_id"] == target_slot
    assert bool(trough.alive[target_slot].item())
    assert not bool(trough.dead[target_slot].item())
    assert int(trough.age[target_slot].item()) == 0
    assert int(trough.under_threshold_ticks[target_slot].item()) == 0
    assert float(trough.cumulative_attention[target_slot].item()) == 0.0
    assert trough.broadcast_ids[target_slot] is not None
    assert trough.broadcast_ids[target_slot].startswith("spawn:")


def test_niche_aware_spawn_targets_underserved_q():
    """Feed many attends with a well-served Q + one underserved Q, kill
    a slot, spawn, assert the new K is closer to the underserved Q than
    to the well-served one."""
    torch.manual_seed(901)
    H, N = 64, 4
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=4, seed=22)
    # Make Q·K direction-aligned so attention concentrates predictably.
    with torch.no_grad():
        trough.W_Q.weight.copy_(trough.W_K.weight)

    well_served_vec = torch.randn(H)
    # Make the underserved query orthogonal so deposited slots can't
    # accidentally cover it.
    underserved_vec = torch.randn(H)
    coeff = (underserved_vec @ well_served_vec) / (well_served_vec @ well_served_vec)
    underserved_vec = underserved_vec - coeff * well_served_vec

    # Deposit slots ALIGNED with the well-served query (W_K ≈ W_Q means
    # K_i ≈ W_K @ V_i and Q ≈ W_K @ well_served_vec — so deposits along
    # `well_served_vec` give a K close to the well-served Q).
    deposits = []
    for i in range(N):
        # Small jitter so slot V's aren't identical.
        v = well_served_vec + 0.05 * torch.randn(H)
        deposits.append(_FakeBroadcast(id=f"ws{i}", channel_embedding=v.tolist()))
    trough.deposit(deposits)

    # Many attends with the well-served query (high max attention) +
    # one attend with the underserved query (low max attention). Use
    # a low temperature so the difference is dramatic in recent_max_attention.
    for _ in range(20):
        trough.attend(well_served_vec, tau=0.5)
    trough.attend(underserved_vec, tau=0.5)

    # Sanity: the underserved-vec attend should have produced the lowest
    # max attention in the ring buffer.
    found_q = trough.find_underserved_query()
    cos_to_underserved = torch.nn.functional.cosine_similarity(
        found_q.unsqueeze(0), underserved_vec.unsqueeze(0), dim=1
    ).item()
    cos_to_well_served = torch.nn.functional.cosine_similarity(
        found_q.unsqueeze(0), well_served_vec.unsqueeze(0), dim=1
    ).item()
    assert cos_to_underserved > cos_to_well_served, (
        f"find_underserved_query should return the underserved vec; "
        f"cos(found,underserved)={cos_to_underserved:.3f}, "
        f"cos(found,well_served)={cos_to_well_served:.3f}"
    )

    # Now kill a slot and spawn — the new K should be closer to the
    # underserved Q than to the well-served Q.
    target_slot = 1
    _force_kill(trough, target_slot)
    # Drop noise to zero to make the test deterministic.
    trough.spawn_noise_scale = 0.0
    info = trough.spawn_into_dead_slot(target_slot)
    assert info["spawned"] is True

    new_v = trough.V_store[target_slot]
    new_k = trough.W_K(new_v.unsqueeze(0)).squeeze(0).detach()

    cos_new_k_underserved = torch.nn.functional.cosine_similarity(
        new_k.unsqueeze(0), underserved_vec.unsqueeze(0), dim=1
    ).item()
    cos_new_k_well_served = torch.nn.functional.cosine_similarity(
        new_k.unsqueeze(0), well_served_vec.unsqueeze(0), dim=1
    ).item()
    assert cos_new_k_underserved > cos_new_k_well_served, (
        f"spawn should target the underserved Q; "
        f"cos(new_K, underserved)={cos_new_k_underserved:.3f}, "
        f"cos(new_K, well_served)={cos_new_k_well_served:.3f}"
    )


def test_population_stays_fixed_under_attend_then_lifecycle():
    """100 ticks of attend + step_lifecycle with population_strategy=fixed:
    `alive.sum()` stays at n_slots throughout."""
    torch.manual_seed(902)
    H, N = 32, 6
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=4, seed=23)
    trough.deposit([_fake_broadcast(i, H) for i in range(N)])
    assert int(trough.alive.sum().item()) == N
    assert trough.population_strategy == "fixed"

    # Aggressive death pressure: easy-to-trigger ε and short patience.
    trough.epsilon_death = 0.5
    trough.n_patience = 2

    for tick in range(100):
        trough.attend(torch.randn(H))
        # Force pressure on slots 3..N-1 to trigger frequent kills.
        if tick >= 2:
            trough.under_threshold_ticks[3:] = trough.n_patience + 1
        info = trough.step_lifecycle()
        assert int(trough.alive.sum().item()) == N, (
            f"tick {tick}: population dropped to {int(trough.alive.sum().item())}, "
            f"expected {N}; killed_ids={info.get('killed_ids')}, "
            f"spawned={len(info.get('spawned', []))}"
        )


def test_spawn_does_not_overwrite_alive_slot():
    """Calling spawn_into_dead_slot on an alive slot must raise."""
    H, N = 32, 4
    trough = TroughAttention(hidden_size=H, n_slots=N, n_heads=4, seed=24)
    trough.deposit([_fake_broadcast(i, H) for i in range(N)])
    assert bool(trough.alive[0].item())
    with pytest.raises(AssertionError):
        trough.spawn_into_dead_slot(0)


def test_niche_aware_spawn_pure_function():
    """Unit-level: niche_aware_spawn returns a vector with target_norm
    that, when projected through W_K, lands close to the input Q."""
    torch.manual_seed(903)
    H = 32
    W_K = torch.randn(H, H) * (1.0 / H ** 0.5)
    q = torch.randn(H)
    new_v = niche_aware_spawn(q, W_K, target_norm=1.5, noise_scale=0.0)
    assert new_v.shape == (H,)
    assert abs(new_v.norm().item() - 1.5) < 1e-4
    # With zero noise and pseudo-inverse: W_K @ new_v should align with q.
    new_k = W_K @ new_v
    cos = torch.nn.functional.cosine_similarity(
        new_k.unsqueeze(0), q.unsqueeze(0), dim=1
    ).item()
    assert cos > 0.9, f"expected near-perfect alignment, got cos={cos:.3f}"
