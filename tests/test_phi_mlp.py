"""Unit tests for trophic.phi_mlp.PhiMLP."""
import torch

from trophic.phi_mlp import PhiMLP, DEFAULT_RANK, DEFAULT_PATCHED_LAYERS


def test_shape_contract():
    H = 256
    phi = PhiMLP(hidden_size=H, rank=DEFAULT_RANK)
    x = torch.randn(H)
    out = phi(x)
    assert set(out.keys()) == set(DEFAULT_PATCHED_LAYERS)
    for d in out.values():
        assert d["M_A"].shape == (H, DEFAULT_RANK)
        assert d["M_B"].shape == (H, DEFAULT_RANK)
        assert d["E_A"].shape == (H, DEFAULT_RANK)
        assert d["E_B"].shape == (H, DEFAULT_RANK)
        assert d["s_M"].shape == ()
        assert d["s_E"].shape == ()
        # Required exact key set
        assert set(d.keys()) == {"M_A", "M_B", "E_A", "E_B", "s_M", "s_E"}


def test_init_produces_small_perturbation():
    """Per seed24 v3 diagnosis: zero-init scale × zero-init head produces a
    saddle point with zero gradient in both directions (s_M=0 kills grad
    on M_A; M_A=0 kills grad on s_M). Init must break that saddle by
    making BOTH the scale and the channel-head non-zero so the
    multiplicative path s * (x A) B^T is small but non-zero at step 0,
    and gradient flows in both directions on the first backward."""
    phi = PhiMLP(hidden_size=64, rank=4)
    x = torch.randn(64)
    out = phi(x)
    for d in out.values():
        # Channel heads small-Gaussian -> tensors are small but non-zero.
        assert d["M_A"].abs().max() > 0.0, "M_A should be non-zero at init"
        assert d["M_A"].abs().max() < 1.0, "M_A should still be small at init"
        # s_M / s_E small but non-zero (≈ 0.01).
        assert 0.0 < d["s_M"].item() < 0.1
        assert 0.0 < d["s_E"].item() < 0.1


def test_init_perturbation_is_small():
    """Total perturbation magnitude s * (x A) B^T should be << x norm at init.
    Smoke that init breaks the saddle without overwhelming the host's
    natural hidden-state distribution."""
    phi = PhiMLP(hidden_size=64, rank=4)
    x = torch.randn(64) * 10  # mock large-magnitude hidden state
    out = phi(x)
    d = out[0]  # any layer
    # Compute the actual perturbation that would be applied: s_M (x M_A) M_B^T
    pert = d["s_M"] * (x @ d["M_A"]) @ d["M_B"].T
    assert pert.abs().max() < x.abs().max(), \
        f"init perturbation ({pert.abs().max():.3f}) should be smaller than x ({x.abs().max():.3f})"


def test_input_dependent_after_perturb():
    phi = PhiMLP(hidden_size=64, rank=4)
    # Manually break zero-init on the channel heads so they produce something.
    with torch.no_grad():
        for ch in phi.channel_heads.values():
            ch.weight.normal_(0.0, 0.01)
    x1 = torch.randn(64)
    x2 = torch.randn(64) * 5
    o1 = phi(x1)[0]
    o2 = phi(x2)[0]
    cos = torch.nn.functional.cosine_similarity(
        o1["M_A"].flatten().unsqueeze(0),
        o2["M_A"].flatten().unsqueeze(0),
    ).item()
    assert cos < 0.99, f"Phi output not input-dependent: cos={cos}"


def test_rejects_wrong_input_shape():
    phi = PhiMLP(hidden_size=64, rank=4)
    bad = torch.randn(2, 64)
    try:
        phi(bad)
    except ValueError:
        return
    raise AssertionError("PhiMLP should reject 2-D input")


def test_patched_layers_param():
    phi = PhiMLP(hidden_size=64, rank=4, patched_layers=[0, 5, 10])
    out = phi(torch.randn(64))
    assert set(out.keys()) == {0, 5, 10}


def test_default_patched_layers_is_eleven_for_qwen3_4b():
    """Stride-3 over 32 layers -> 11 patched layers, matching Hexis paper."""
    assert len(DEFAULT_PATCHED_LAYERS) == 11
    assert DEFAULT_PATCHED_LAYERS[0] == 0
    assert DEFAULT_PATCHED_LAYERS[-1] == 30
