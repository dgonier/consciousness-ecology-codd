# Mission 01: phi-mlp-and-hooks

**Handle**: phase1-A
**Phase**: 1 (sequential, foundation)
**Mission file**: `phase1-A-01-phi-mlp-and-hooks.md`
**Dependencies**: none
**Blocks**: phase2-B:02, phase2-C:03, phase2-D:04 (everything in phase 2 needs this primitive)

---

## Before You Start

```bash
# 1. Read the architectural spec.
cat /home/dgonier/debaterhub/hexis/paper/sections/hexis_architecture.tex
# Focus on eq (1): x'_ℓ = x_ℓ + s_M · (x_ℓ M_A) M_B^T  ;  V'_ℓ = V_ℓ + s_E · (x_ℓ E_A) E_B^T

# 2. Read the working implementation.
sed -n '40,75p;125,145p;190,235p' /home/dgonier/experiments/scripts/train_action_m.py

# 3. Read the memory file for trophic context.
cat ~/.claude/projects/-home-dgonier-ecology-experiment/memory/project_trophic_hexis_alignment.md

# 4. Inbox check.
grep -nE "@phase1-A|@all|@phase1" /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md

# 5. Flip your status.
sed -i 's/^phase1-A:01:PENDING$/phase1-A:01:RUNNING/' \
  /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md
```

## Goal

Build the two primitives that the rest of the swarm depends on:

1. **`PhiMLP`** — an `nn.Module` that takes a single trough-attended hidden vector `[H]` and produces per-layer rank-r `(M_A, M_B, E_A, E_B, s_M, s_E)` tensors for stride-3 patched layers of Qwen3-4B.
2. **`install_M_hooks`** — a function that installs `forward_pre_hook`s on `model.model.layers[ℓ]` for each patched layer, applying the Hexis Q+V perturbation per eq (1). Includes a `prefill_active` flag that gates whether hooks fire (to avoid the M-attractor feedback loop documented in Hexis discussion §).

No new training. Pure infrastructure. The deliverable is unit-tested primitives.

## Files to Create / Modify

Create:
- `trophic/phi_mlp.py` — `PhiMLP` class
- `trophic/m_hooks.py` — `install_M_hooks`, `apply_M_perturbation` helper
- `tests/test_phi_mlp.py` — at least 4 tests
- `tests/test_m_hooks.py` — at least 4 tests

Do NOT modify:
- `trophic/channel.py` (legacy prefix-injection; phase2-D handles its replacement)
- `trophic/trough_attention.py` (already has E_in / diet mask / etc. from earlier work)
- `trophic/training/sft.py` (phase2-D rewires this)

## Implementation Steps

### 1. PhiMLP

```python
# trophic/phi_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_RANK = 16
DEFAULT_PATCHED_LAYERS = list(range(0, 32, 3))   # stride-3 over Qwen3-4B's 32 layers
                                                  # (Hexis paper: 11 of 32; this gives 11)

class PhiMLP(nn.Module):
    """Compile trough-attended hidden state -> per-layer M+E modulation tensors.

    Architecture: shared bottleneck across layers, per-layer head producing
    (M_A, M_B, E_A, E_B, s_M, s_E). Bottleneck width: hidden_size // 4 (per Hexis).

    Forward returns a dict keyed by layer index. Each layer's value is a dict with
    M_A, M_B, E_A, E_B (each [H, r]) and scalar s_M, s_E.

    Initialization: bottleneck weights small-Gaussian; per-layer heads init so the
    output tensors have small norm at step 0 (s_M and s_E start at 0). This means
    Phi(x) ≈ zero-modulation at init — must be a no-op via the s_M/s_E gates.
    """
    def __init__(self,
                 hidden_size: int,
                 rank: int = DEFAULT_RANK,
                 patched_layers: list[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.patched_layers = patched_layers or DEFAULT_PATCHED_LAYERS

        bottleneck = hidden_size // 4
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_size, bottleneck, bias=False),
            nn.SiLU(),
        )
        # 4 (H * r) tensors per layer + 2 scalars
        per_layer_dim = 4 * hidden_size * rank + 2
        self.head = nn.Linear(bottleneck, per_layer_dim * len(self.patched_layers), bias=False)

        # Init: bottleneck small-Gaussian, head ZERO so Phi(x) outputs all-zero
        # tensors at step 0 -> s_M=0, s_E=0 -> hooks are no-ops -> baseline behavior.
        with torch.no_grad():
            for m in self.bottleneck.modules():
                if isinstance(m, nn.Linear):
                    m.weight.normal_(0.0, 1.0 / hidden_size ** 0.5)
            self.head.weight.zero_()

    def forward(self, trough_attended_hidden: torch.Tensor) -> dict:
        """trough_attended_hidden: [H] (single vector). Returns dict[layer_idx, dict]."""
        if trough_attended_hidden.dim() != 1:
            raise ValueError(f"expected 1-D [H], got {tuple(trough_attended_hidden.shape)}")
        z = self.bottleneck(trough_attended_hidden)              # [bottleneck]
        flat = self.head(z)                                       # [n_layers * per_layer_dim]
        per_layer_dim = 4 * self.hidden_size * self.rank + 2
        flat = flat.view(len(self.patched_layers), per_layer_dim)

        out = {}
        H, r = self.hidden_size, self.rank
        Hr = H * r
        for i, layer_idx in enumerate(self.patched_layers):
            row = flat[i]
            M_A = row[0*Hr : 1*Hr].view(H, r)
            M_B = row[1*Hr : 2*Hr].view(H, r)
            E_A = row[2*Hr : 3*Hr].view(H, r)
            E_B = row[3*Hr : 4*Hr].view(H, r)
            s_M = row[4*Hr]
            s_E = row[4*Hr + 1]
            out[layer_idx] = {
                "M_A": M_A, "M_B": M_B,
                "E_A": E_A, "E_B": E_B,
                "s_M": s_M, "s_E": s_E,
            }
        return out
```

### 2. M Hooks

```python
# trophic/m_hooks.py
import torch
import torch.nn as nn

class HookHandle:
    """Wrapper around a torch.nn.utils.hooks.RemovableHandle plus a way to flip
    `prefill_active` after install (so the same hooks span prefill + generate).
    """
    def __init__(self, handle, gate):
        self._handle = handle
        self._gate = gate
    def set_prefill_active(self, active: bool):
        # Despite the name, "prefill_active=True" means "we're in prefill, hooks
        # should be NO-OP" — matching Hexis's discussion: M is only applied
        # during generation, not prefill, to avoid the M-attractor feedback loop.
        self._gate["prefill"] = active
    def remove(self):
        self._handle.remove()


def install_M_hooks(
    model: nn.Module,
    m_tensors_per_layer: dict,           # {layer_idx: {"M_A":..., "M_B":..., "E_A":..., "E_B":..., "s_M":..., "s_E":...}}
    prefill_active: bool = False,        # True = currently in prefill, hooks are no-ops
) -> list[HookHandle]:
    """Install forward-pre-hooks on `model.model.layers[ℓ]` per patched layer.

    Math per Hexis paper eq (1) — applied to the layer's input hidden state x:
        x' = x + s_M * (x M_A) M_B^T

    NOTE: V-modulation (eq 1's second term, V' = V + s_E (x E_A) E_B^T) operates on
    the projected V tensor inside attention. For phase 1 we apply BOTH perturbations
    as input-side modifications via a forward_pre_hook on the layer; the V-mod is
    approximated as an additional perturbation `+ s_E * (x E_A) E_B^T` to the
    same hidden state. This is the simpler "additive at layer input" form used in
    train_action_m.py (single hook per layer modifying x). Phase 2-D may upgrade to
    a true per-projection hook if needed.

    Returns a list[HookHandle] so the caller can:
      handles = install_M_hooks(model, M, prefill_active=True)
      _ = model(prefill_inputs)             # hooks no-op
      for h in handles: h.set_prefill_active(False)
      _ = model.generate(...)               # hooks fire
      for h in handles: h.remove()
    """
    handles: list[HookHandle] = []
    gate = {"prefill": prefill_active}

    layers = _resolve_layers(model)

    for layer_idx, mod_dict in m_tensors_per_layer.items():
        if layer_idx >= len(layers):
            raise ValueError(f"layer_idx {layer_idx} out of range (have {len(layers)})")
        target = layers[layer_idx]

        # Bind closure
        def make_hook(mod_dict_local):
            def hook(module, args, kwargs=None):
                if gate["prefill"]:
                    # No-op during prefill (Hexis attractor fix).
                    return None
                if args:
                    x = args[0]
                else:
                    x = kwargs.get("hidden_states") if kwargs else None
                if x is None:
                    return None
                # Apply both perturbations as additive modifications to x.
                M_A = mod_dict_local["M_A"].to(x.dtype).to(x.device)
                M_B = mod_dict_local["M_B"].to(x.dtype).to(x.device)
                E_A = mod_dict_local["E_A"].to(x.dtype).to(x.device)
                E_B = mod_dict_local["E_B"].to(x.dtype).to(x.device)
                s_M = mod_dict_local["s_M"].to(x.dtype).to(x.device)
                s_E = mod_dict_local["s_E"].to(x.dtype).to(x.device)
                # x: [..., H]
                m_pert = s_M * torch.matmul(torch.matmul(x, M_A), M_B.T)
                e_pert = s_E * torch.matmul(torch.matmul(x, E_A), E_B.T)
                x_new = x + m_pert + e_pert
                if args:
                    return (x_new,) + args[1:], kwargs
                kw = dict(kwargs) if kwargs else {}
                kw["hidden_states"] = x_new
                return args, kw
            return hook

        h = target.register_forward_pre_hook(make_hook(mod_dict), with_kwargs=True)
        handles.append(HookHandle(h, gate))
    return handles


def _resolve_layers(model: nn.Module) -> list[nn.Module]:
    """Return the per-layer module list from a HF causal LM.

    Tries `.model.layers` (Qwen, Llama, Mistral) then `.transformer.h` (GPT-2 style).
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise AttributeError("Could not find transformer layer list on model")
```

### 3. Tests

```python
# tests/test_phi_mlp.py
import torch
from trophic.phi_mlp import PhiMLP, DEFAULT_RANK, DEFAULT_PATCHED_LAYERS

def test_shape_contract():
    H = 256
    phi = PhiMLP(hidden_size=H, rank=DEFAULT_RANK)
    x = torch.randn(H)
    out = phi(x)
    assert set(out.keys()) == set(DEFAULT_PATCHED_LAYERS)
    for layer_idx, d in out.items():
        assert d["M_A"].shape == (H, DEFAULT_RANK)
        assert d["M_B"].shape == (H, DEFAULT_RANK)
        assert d["E_A"].shape == (H, DEFAULT_RANK)
        assert d["E_B"].shape == (H, DEFAULT_RANK)
        assert d["s_M"].shape == ()
        assert d["s_E"].shape == ()

def test_zero_init_returns_zero_tensors():
    phi = PhiMLP(hidden_size=64, rank=4)
    x = torch.randn(64)
    out = phi(x)
    for d in out.values():
        # Head is zero-init, so all outputs should be zero at step 0.
        assert torch.allclose(d["M_A"], torch.zeros_like(d["M_A"]))
        assert torch.allclose(d["M_B"], torch.zeros_like(d["M_B"]))
        assert torch.allclose(d["E_A"], torch.zeros_like(d["E_A"]))
        assert torch.allclose(d["E_B"], torch.zeros_like(d["E_B"]))
        assert d["s_M"].item() == 0.0
        assert d["s_E"].item() == 0.0

def test_input_dependent_after_perturb():
    phi = PhiMLP(hidden_size=64, rank=4)
    # Manually break zero-init so the head produces something
    with torch.no_grad():
        phi.head.weight.normal_(0.0, 0.01)
    x1 = torch.randn(64)
    x2 = torch.randn(64) * 5
    o1 = phi(x1)[0]   # layer 0
    o2 = phi(x2)[0]
    # Different inputs -> different outputs (cosine < 0.99)
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
```

```python
# tests/test_m_hooks.py
import torch
import torch.nn as nn
from trophic.m_hooks import install_M_hooks


class _Block(nn.Module):
    """Simulates a Qwen-style decoder block: takes hidden, returns hidden."""
    def __init__(self, H):
        super().__init__()
        self.lin = nn.Linear(H, H, bias=False)
    def forward(self, hidden_states, *args, **kwargs):
        return self.lin(hidden_states), None


class _ToyLM(nn.Module):
    """Smallest possible model_with_layers shape for hooks to grab onto."""
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
            "M_A": torch.zeros(H, r), "M_B": torch.zeros(H, r),
            "E_A": torch.zeros(H, r), "E_B": torch.zeros(H, r),
            "s_M": torch.tensor(0.0), "s_E": torch.tensor(0.0),
        } for l in layers
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
        } for l in layers
    }


def test_zero_mod_is_identity():
    H, n, r = 32, 4, 4
    model = _ToyLM(H, n)
    x = torch.randn(2, 5, H)
    baseline = model(x.clone())
    handles = install_M_hooks(model, _zero_mods(H, r, [0, 2]), prefill_active=False)
    perturbed = model(x.clone())
    for h in handles: h.remove()
    assert torch.allclose(baseline, perturbed, atol=1e-6)


def test_nonzero_mod_changes_output():
    H, n, r = 32, 4, 4
    model = _ToyLM(H, n)
    x = torch.randn(2, 5, H)
    baseline = model(x.clone())
    handles = install_M_hooks(model, _nonzero_mods(H, r, [0, 2]), prefill_active=False)
    perturbed = model(x.clone())
    for h in handles: h.remove()
    diff = (baseline - perturbed).norm().item()
    assert diff > 1e-3, f"non-zero M tensors should change output, got diff={diff}"


def test_prefill_gating_disables_hooks():
    H, n, r = 32, 4, 4
    model = _ToyLM(H, n)
    x = torch.randn(2, 5, H)
    baseline = model(x.clone())
    handles = install_M_hooks(model, _nonzero_mods(H, r, [0, 2]), prefill_active=True)
    prefill_out = model(x.clone())
    for h in handles: h.remove()
    # During prefill, hooks must be no-op
    assert torch.allclose(baseline, prefill_out, atol=1e-6)


def test_prefill_toggle_reaches_hooks():
    H, n, r = 32, 4, 4
    model = _ToyLM(H, n)
    x = torch.randn(2, 5, H)
    baseline = model(x.clone())
    handles = install_M_hooks(model, _nonzero_mods(H, r, [0, 2]), prefill_active=True)
    # Switch off prefill -> hooks should now fire
    for h in handles: h.set_prefill_active(False)
    out = model(x.clone())
    for h in handles: h.remove()
    diff = (baseline - out).norm().item()
    assert diff > 1e-3, f"after flipping prefill off, hooks should fire; diff={diff}"
```

### 4. Iterate until tests pass

Run the tests after each meaningful change. They should all pass before you flip DONE.

## Acceptance Criteria

- [ ] `trophic/phi_mlp.py` exists with `PhiMLP` class matching the contract.
- [ ] `trophic/m_hooks.py` exists with `install_M_hooks` and `HookHandle`.
- [ ] At init, `phi(x)` returns all-zero tensors and zero scalars (s_M=s_E=0).
- [ ] `install_M_hooks(model, zero_mods)` is identity (output unchanged).
- [ ] `install_M_hooks(model, non-zero mods)` measurably changes output (diff > 1e-3).
- [ ] `prefill_active=True` makes hooks no-op even with non-zero mods.
- [ ] `set_prefill_active(False)` after install lets hooks fire.
- [ ] All 8 new tests pass.
- [ ] Pre-existing 89 tests still pass.

## Testing Conditions (exit verification)

Run each in order. Each must produce the expected output.

```bash
cd /home/dgonier/ecology_experiment/trophic

# 1. New tests pass
.venv/bin/python -m pytest tests/test_phi_mlp.py tests/test_m_hooks.py -v
# Expected: 8 passed (or more if you added extras)

# 2. Pre-existing tests still pass
.venv/bin/python -m pytest tests/ -x --ignore=tests/test_phi_mlp.py --ignore=tests/test_m_hooks.py
# Expected: 89 passed

# 3. Smoke check on actual Qwen — Phi outputs zero tensors at init, hooks are identity
.venv/bin/python -c "
import torch
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.phi_mlp import PhiMLP, DEFAULT_PATCHED_LAYERS
from trophic.m_hooks import install_M_hooks

host = ModelHost.get(DEFAULT_CONFIG.model)
phi = PhiMLP(hidden_size=host.hidden_size).to(host.device).to(host.dtype)
trough_state = torch.randn(host.hidden_size, dtype=host.dtype, device=host.device)
m_tensors = phi(trough_state)
# All zero at init -> hook should be identity
ids = host._tok('hello world', return_tensors='pt').input_ids.to(host.device)
baseline_logits = host._model(input_ids=ids).logits.float().detach().clone()
handles = install_M_hooks(host._model, m_tensors, prefill_active=False)
hooked_logits = host._model(input_ids=ids).logits.float().detach().clone()
for h in handles: h.remove()
diff = (baseline_logits - hooked_logits).abs().max().item()
print(f'init diff: {diff:.2e}  (should be < 1e-4 — zero-init Phi is identity-mod)')
assert diff < 1e-4, diff
print('OK')
"
# Expected: diff < 1e-4 ; "OK"
```

## Coordination

You're the foundation. Phase 2 has 3 parallel agents (B, C, D) all blocking on you. **Do not change the interface contracts in scratchpad.md without an `@all` MESSAGES note.** If you discover a contract issue mid-build, write it down BEFORE the change so phase 2 doesn't waste cycles on the old contract.

Specifically:
- `PhiMLP.forward` returns a `dict[layer_idx, dict]` with the exact 6 keys (`M_A`, `M_B`, `E_A`, `E_B`, `s_M`, `s_E`). Don't add or remove keys.
- `install_M_hooks` returns a `list[HookHandle]` where each has `.set_prefill_active(bool)` and `.remove()`. Don't change those method names.

## When Done

```bash
# Re-run inbox grep
grep -nE "@phase1-A|@all|@phase1" /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md

# Flip status
sed -i 's/^phase1-A:01:RUNNING$/phase1-A:01:DONE/' \
  /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md

# Unblock phase 2
sed -i 's/^phase2-B:02:BLOCKED.*$/phase2-B:02:PENDING/' \
  /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md
sed -i 's/^phase2-C:03:BLOCKED.*$/phase2-C:03:PENDING/' \
  /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md
sed -i 's/^phase2-D:04:BLOCKED.*$/phase2-D:04:PENDING/' \
  /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md

# Append a MESSAGES line broadcasting to phase 2
cat >> /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md <<EOF
[$(date +%Y-%m-%d\ %H:%M)] phase1-A > @phase2: PhiMLP at trophic/phi_mlp.py, install_M_hooks at trophic/m_hooks.py. Contracts unchanged from scratchpad. Tests in tests/test_phi_mlp.py and tests/test_m_hooks.py. Phase 2 unblocked.
EOF

# Move this mission file to completed/
mv /home/dgonier/ecology_experiment/trophic/tasks_perlayer/phase1-A-01-phi-mlp-and-hooks.md \
   /home/dgonier/ecology_experiment/trophic/tasks_perlayer/completed/
```
