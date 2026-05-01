"""Unit tests for trophic.m_hooks.install_M_hooks."""
import torch
import torch.nn as nn

from trophic.m_hooks import install_M_hooks


class _Block(nn.Module):
    """Simulates a Qwen-style decoder block: takes hidden, returns (hidden, None)."""

    def __init__(self, H):
        super().__init__()
        self.lin = nn.Linear(H, H, bias=False)

    def forward(self, hidden_states, *args, **kwargs):
        return self.lin(hidden_states), None


class _ToyLM(nn.Module):
    """Minimal model exposing `.model.layers` for hooks to grab onto."""

    def __init__(self, H, n_layers):
        super().__init__()

        class _Inner(nn.Module):
            def __init__(self, H, n):
                super().__init__()
                self.layers = nn.ModuleList([_Block(H) for _ in range(n)])

        self.model = _Inner(H, n_layers)

    def forward(self, hidden):
        x = hidden
        for layer in self.model.layers:
            out = layer(x)
            x = out[0] if isinstance(out, tuple) else out
        return x


def _zero_mods(H, r, layers):
    return {
        l: {
            "M_A": torch.zeros(H, r),
            "M_B": torch.zeros(H, r),
            "E_A": torch.zeros(H, r),
            "E_B": torch.zeros(H, r),
            "s_M": torch.tensor(0.0),
            "s_E": torch.tensor(0.0),
        }
        for l in layers
    }


def _nonzero_mods(H, r, layers, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        l: {
            "M_A": torch.randn(H, r, generator=g) * 0.1,
            "M_B": torch.randn(H, r, generator=g) * 0.1,
            "E_A": torch.randn(H, r, generator=g) * 0.1,
            "E_B": torch.randn(H, r, generator=g) * 0.1,
            "s_M": torch.tensor(1.0),
            "s_E": torch.tensor(1.0),
        }
        for l in layers
    }


def test_zero_mod_is_identity():
    H, n, r = 32, 4, 4
    model = _ToyLM(H, n)
    x = torch.randn(2, 5, H)
    baseline = model(x.clone())
    handles = install_M_hooks(model, _zero_mods(H, r, [0, 2]), prefill_active=False)
    perturbed = model(x.clone())
    for h in handles:
        h.remove()
    assert torch.allclose(baseline, perturbed, atol=1e-6)


def test_nonzero_mod_changes_output():
    H, n, r = 32, 4, 4
    model = _ToyLM(H, n)
    x = torch.randn(2, 5, H)
    baseline = model(x.clone())
    handles = install_M_hooks(model, _nonzero_mods(H, r, [0, 2]), prefill_active=False)
    perturbed = model(x.clone())
    for h in handles:
        h.remove()
    diff = (baseline - perturbed).norm().item()
    assert diff > 1e-3, f"non-zero M tensors should change output, got diff={diff}"


def test_prefill_gating_disables_hooks():
    H, n, r = 32, 4, 4
    model = _ToyLM(H, n)
    x = torch.randn(2, 5, H)
    baseline = model(x.clone())
    handles = install_M_hooks(model, _nonzero_mods(H, r, [0, 2]), prefill_active=True)
    prefill_out = model(x.clone())
    for h in handles:
        h.remove()
    assert torch.allclose(baseline, prefill_out, atol=1e-6)


def test_prefill_toggle_reaches_hooks():
    H, n, r = 32, 4, 4
    model = _ToyLM(H, n)
    x = torch.randn(2, 5, H)
    baseline = model(x.clone())
    handles = install_M_hooks(model, _nonzero_mods(H, r, [0, 2]), prefill_active=True)
    for h in handles:
        h.set_prefill_active(False)
    out = model(x.clone())
    for h in handles:
        h.remove()
    diff = (baseline - out).norm().item()
    assert diff > 1e-3, f"after flipping prefill off, hooks should fire; diff={diff}"


def test_handles_remove_restores_baseline():
    """After removing handles, model must behave exactly like baseline."""
    H, n, r = 32, 4, 4
    model = _ToyLM(H, n)
    x = torch.randn(2, 5, H)
    baseline = model(x.clone())
    handles = install_M_hooks(model, _nonzero_mods(H, r, [0, 2]), prefill_active=False)
    _ = model(x.clone())  # hooks fire here
    for h in handles:
        h.remove()
    after = model(x.clone())
    assert torch.allclose(baseline, after, atol=1e-6)


def test_invalid_layer_idx_raises():
    H, n, r = 32, 4, 4
    model = _ToyLM(H, n)
    bad_mods = _zero_mods(H, r, [0, 99])
    try:
        install_M_hooks(model, bad_mods, prefill_active=False)
    except ValueError:
        return
    raise AssertionError("install_M_hooks should reject out-of-range layer_idx")
