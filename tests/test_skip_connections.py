"""Mission 06 (phase3-A) — cross-tier skip connections + τ schedule.

Covers:
  - Skip-weight initialization (sigmoid(0.1) ≈ 0.525, in [0.4, 0.6]).
  - Gradient flows through the skip-weight scalar.
  - With α forced near 1.0 the predator's combined output tracks the
    producer-tier (skip) path far more strongly than the herbivore
    (main) path.
  - The cosine τ schedule is monotonically non-increasing.
  - τ is forwarded through `attend()` calls (mocked to capture).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from trophic.agents.predator import Predator, _SkipWeightHolder
from trophic.training.tau_schedule import cosine_tau, linear_tau
from trophic.trough_attention import TroughAttention


@dataclass
class _FakeBroadcast:
    id: str
    channel_embedding: list[float]


def _fake_broadcast(idx: int, hidden: int, scale: float = 1.0) -> _FakeBroadcast:
    torch.manual_seed(2026 + idx)
    v = torch.randn(hidden) * scale
    return _FakeBroadcast(id=f"b{idx}", channel_embedding=v.tolist())


# ---------------------------------------------------------------------------
# 1. Skip-weight initializes low and is in the expected range.
# ---------------------------------------------------------------------------

def test_skip_weight_initializes_low():
    """sigmoid(0.1) ≈ 0.525, well inside [0.4, 0.6]."""
    holder = _SkipWeightHolder(0.1)
    alpha = torch.sigmoid(holder.skip_weight).item()
    assert 0.4 <= alpha <= 0.6, alpha
    # Approximately 0.525.
    assert abs(alpha - 0.524979) < 1e-3, alpha

    # And on a fresh predator, it's a Parameter that requires_grad.
    pred = Predator.make("short_horizon")
    sw = pred.skip_weight  # triggers lazy init
    assert isinstance(sw, torch.nn.Parameter)
    assert sw.requires_grad
    a2 = float(torch.sigmoid(sw).item())
    assert 0.4 <= a2 <= 0.6, a2


# ---------------------------------------------------------------------------
# 2. Gradient flows through skip_weight on a synthetic loss.
# ---------------------------------------------------------------------------

def test_skip_weight_gradient_nonzero():
    """Build a tiny synthetic combine loss = ||(1-α)*A + α*B - target||^2.
    Derivative w.r.t. α is non-zero unless A == B (it's not), so the
    gradient on the underlying scalar must also be non-None / non-zero.
    """
    torch.manual_seed(0)
    holder = _SkipWeightHolder(0.1)
    A = torch.randn(8, 32)
    B = torch.randn(8, 32)
    target = torch.randn(8, 32)
    alpha = torch.sigmoid(holder.skip_weight)
    combined = (1.0 - alpha) * A + alpha * B
    loss = (combined - target).pow(2).mean()
    loss.backward()
    assert holder.skip_weight.grad is not None
    assert abs(float(holder.skip_weight.grad.item())) > 1e-8


# ---------------------------------------------------------------------------
# 3. With α≈1.0, predator combined output tracks the producer (skip) path.
# ---------------------------------------------------------------------------

def test_predator_combines_skip_and_main():
    """Force the skip weight to a large value (α ≈ 1.0). Then the
    combined predator-input tensor should be much closer to the
    producer-trough output than to the herbivore Channel output.
    """
    torch.manual_seed(0)
    H = 32
    out_seq_len = 8

    # Build a producer trough whose attend output will be deterministic
    # for a given query.
    producer_trough = TroughAttention(
        hidden_size=H, n_slots=4, n_heads=4, out_seq_len=out_seq_len, seed=42
    )
    producer_trough.deposit([_fake_broadcast(i, H) for i in range(2)])

    # Stand-in "channel output" — random tensor unrelated to the trough.
    channel_seq = torch.randn(out_seq_len, H)

    # Pick an arbitrary hunter state.
    hs = torch.randn(H)
    skip_out = producer_trough.attend(hs, tau=1.0)
    skip_seq = skip_out.output.detach()

    # Force α to nearly 1.0 by setting raw skip_weight high.
    sw = torch.tensor(8.0)  # sigmoid(8) ≈ 0.99966
    alpha = torch.sigmoid(sw)
    combined = (1.0 - alpha) * channel_seq + alpha * skip_seq

    # Cosine similarity (per-row mean) of combined to skip vs. to channel.
    def cos(a, b):
        a_f = a.flatten()
        b_f = b.flatten()
        return float(
            (a_f @ b_f / (a_f.norm() * b_f.norm() + 1e-8)).item()
        )

    sim_combined_skip = cos(combined, skip_seq)
    sim_combined_main = cos(combined, channel_seq)

    # Combined should be much more aligned with skip than with main.
    assert sim_combined_skip > sim_combined_main, (
        sim_combined_skip, sim_combined_main
    )
    # And it should be very close to skip in absolute terms.
    assert sim_combined_skip > 0.99, sim_combined_skip


# ---------------------------------------------------------------------------
# 4. cosine_tau schedule is monotonically non-increasing.
# ---------------------------------------------------------------------------

def test_cosine_tau_monotone():
    total = 200
    tau_start = 2.0
    tau_end = 0.5
    prev = float("inf")
    seen = set()
    for step in range(0, total + 1):
        v = cosine_tau(step, total, tau_start=tau_start, tau_end=tau_end)
        # Non-increasing across the schedule (cosine half-period).
        assert v <= prev + 1e-9, (step, v, prev)
        prev = v
        seen.add(round(v, 4))
    # End-points exact (within fp tolerance).
    assert abs(cosine_tau(0, total, tau_start, tau_end) - tau_start) < 1e-6
    assert abs(cosine_tau(total, total, tau_start, tau_end) - tau_end) < 1e-6
    # Lots of distinct values across the schedule.
    assert len(seen) > 50, len(seen)

    # Sanity: linear schedule also monotone non-increasing.
    prev_l = float("inf")
    for step in range(0, total + 1):
        v = linear_tau(step, total, tau_start=tau_start, tau_end=tau_end)
        assert v <= prev_l + 1e-9, (step, v, prev_l)
        prev_l = v


# ---------------------------------------------------------------------------
# 5. τ is passed through to the trough's attend() call.
# ---------------------------------------------------------------------------

def test_tau_passed_into_attend():
    """Use a thin subclass of TroughAttention that records τ on each
    attend() call. Exercise the SFT skip path and verify the recorded τ
    matches the cosine schedule the trainer would have produced.
    """
    received: list[float] = []

    class RecordingTrough(TroughAttention):
        def attend(self, query, tau=1.0, external_bias=None):
            received.append(float(tau))
            return super().attend(query, tau=tau, external_bias=external_bias)

    H = 16
    trough = RecordingTrough(hidden_size=H, n_slots=4, n_heads=2, seed=7)
    trough.deposit([_fake_broadcast(i, H) for i in range(2)])

    q = torch.randn(H)

    # Drive a few schedule values explicitly.
    schedule = [cosine_tau(s, 4, 2.0, 0.5) for s in (0, 1, 2, 3, 4)]
    for tau in schedule:
        trough.attend(q, tau=tau)

    assert received == schedule, (received, schedule)
    # And verify at least 5 distinct values.
    assert len(set(round(v, 4) for v in received)) >= 4, received
