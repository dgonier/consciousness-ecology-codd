"""Unit tests for trophic.dstar — DStar dataclass + install_dstar_hooks + extract_dstar."""
import torch
import torch.nn as nn

from trophic.dstar import DStar, extract_dstar, install_dstar_hooks


class _Block(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.lin = nn.Linear(H, H, bias=False)

    def forward(self, x, *args, **kwargs):
        # Returns a tuple to match HF decoder-layer shape.
        return (self.lin(x),)


class _ToyLM(nn.Module):
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
            x = layer(x)[0]
        return x


def test_dstar_shape_and_norm():
    H = 32
    layers = [0, 2, 4]
    dirs = {l: torch.randn(H) for l in layers}
    d = DStar(directions=dirs, scale=5.0, hidden_size=H)
    assert d.scale == 5.0
    assert set(d.directions.keys()) == {0, 2, 4}
    for v in d.directions.values():
        assert v.shape == (H,)


def test_dstar_save_load_roundtrip(tmp_path):
    H = 16
    d = DStar(
        directions={0: torch.randn(H), 3: torch.randn(H)},
        scale=8.0,
        hidden_size=H,
    )
    p = tmp_path / "d.pt"
    d.save(p)
    d2 = DStar.load(p)
    assert d2.scale == 8.0
    assert d2.hidden_size == H
    for l in d.directions:
        assert torch.allclose(d.directions[l], d2.directions[l])


def test_install_dstar_hooks_changes_output():
    torch.manual_seed(0)
    H, n = 32, 4
    model = _ToyLM(H, n)
    x = torch.randn(2, 5, H)
    baseline = model(x.clone()).clone()

    dstar = DStar(
        directions={0: torch.randn(H), 2: torch.randn(H)},
        scale=10.0,
        hidden_size=H,
    )
    handles = install_dstar_hooks(model, dstar, active=True)
    perturbed = model(x.clone())
    for h in handles:
        h.remove()

    diff = (baseline - perturbed).norm().item()
    assert diff > 1e-3, f"d* hooks should change output, got diff={diff}"


def test_dstar_active_gate():
    torch.manual_seed(1)
    H, n = 32, 4
    model = _ToyLM(H, n)
    x = torch.randn(2, 5, H)
    baseline = model(x.clone()).clone()

    dstar = DStar(
        directions={0: torch.randn(H), 2: torch.randn(H)},
        scale=10.0,
        hidden_size=H,
    )
    handles = install_dstar_hooks(model, dstar, active=False)
    out = model(x.clone())
    for h in handles:
        h.remove()

    assert torch.allclose(baseline, out, atol=1e-5)


def test_extract_dstar_unit_directions():
    torch.manual_seed(2)
    H = 8
    patched = [0, 1]
    n_layers = 2
    inner = _ToyLM(H, n_layers)

    class _Shim:
        """Wraps the toy LM to expose hidden_states[l+1] like an HF causal LM."""

        def __init__(self, m):
            self.m = m

        def eval(self):
            self.m.eval()

        def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
            # Embed input_ids deterministically by id mod -> a fixed vector.
            B, T = input_ids.shape
            # Use a per-id embedding so pro/con texts differ in expected hidden.
            embed = torch.randn(200, H, generator=torch.Generator().manual_seed(7))
            x = embed[input_ids.clamp(0, 199).long()]
            hs = [x]
            cur = x
            for l in self.m.model.layers:
                cur = l(cur)[0]
                hs.append(cur)

            class _Out:
                pass

            o = _Out()
            o.hidden_states = hs
            return o

    class _Tok:
        def __call__(self, txt, return_tensors=None, truncation=True, max_length=1024):
            # Hash-ish: derive ids from the string so pro/con differ.
            ids = torch.tensor(
                [[(ord(c) % 199) + 1 for c in (txt[:4] or "abcd")]],
                dtype=torch.long,
            )
            return {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
            }

    pro_texts = [f"pro_{i}" for i in range(8)]
    con_texts = [f"con_{i}" for i in range(8)]
    d = extract_dstar(
        base_model=_Shim(inner),
        tokenizer=_Tok(),
        pro_texts=pro_texts,
        con_texts=con_texts,
        patched_layers=patched,
        device=torch.device("cpu"),
    )
    assert d.hidden_size == H
    assert d.scale == 10.0
    for l in patched:
        assert d.directions[l].shape == (H,)
        assert torch.isclose(
            d.directions[l].norm(), torch.tensor(1.0), atol=1e-5
        ), f"layer {l} not unit norm: {d.directions[l].norm().item()}"
