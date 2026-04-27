"""Channel cross-attention behavior — null gate, capacity, scaled init."""
import torch

from trophic.channel import Channel


def test_single_prey_is_selected():
    """With one prey item and a small-init null slot, the prey should still
    appear in selected_indices regardless of which side of the soft null gate
    its weight lands on. (Post-NaN-fix init uses scaled-normal W_Q/W_K, so the
    null logit and the real-key logit are roughly comparable in magnitude;
    null_prob can land either side of 0.5. What matters here is that the
    capacity-bounded selection includes the only real prey.)
    """
    torch.manual_seed(0)
    H = 64
    ch = Channel(hidden_size=H, n_heads=8, capacity=4, seed=42)
    hunter = torch.randn(H)
    prey = torch.randn(1, H)
    out = ch(hunter, prey)
    assert out.selected_indices == [0]
    # null_prob is bounded — must not pin to 1.0 when a real key exists
    assert out.null_prob < 1.0


def test_null_gate_fires_when_keys_far_from_query():
    """Construct an extreme case: hunter Q is one direction, prey K's are
    nearly orthogonal, and we crank the null bias to ensure null wins.
    """
    H = 64
    ch = Channel(hidden_size=H, n_heads=8, capacity=4)
    # Make null very attractive by setting null_bias high.
    with torch.no_grad():
        ch.null_bias.fill_(20.0)
    hunter = torch.randn(H)
    prey = torch.randn(3, H) * 0.01  # tiny prey vectors → small real logits
    out = ch(hunter, prey)
    assert out.null_prob > 0.9


def test_capacity_caps_selection():
    H = 64
    ch = Channel(hidden_size=H, n_heads=8, capacity=2)
    hunter = torch.randn(H)
    prey = torch.randn(5, H)
    out = ch(hunter, prey)
    assert len(out.selected_indices) == 2
    assert len(out.rejected_indices) == 3


def test_empty_prey():
    H = 64
    ch = Channel(hidden_size=H, n_heads=8, out_seq_len=4, capacity=4)
    hunter = torch.randn(H)
    prey = torch.zeros(0, H)
    out = ch(hunter, prey)
    assert out.selected_indices == []
    assert out.null_prob == 1.0  # only null is available
    assert out.output.shape == (4, H)
