# Mission 02: dstar-extraction

**Handle**: phase2-B
**Phase**: 2 (parallel after phase1-A)
**Mission file**: `phase2-B-02-dstar-extraction.md`
**Dependencies**: phase1-A:01
**Blocks**: phase3-A:05

---

## Before You Start

```bash
# 1. Confirm phase 1 is done.
grep "phase1-A:01:DONE" /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md

# 2. Read the d* template.
cat /home/dgonier/experiments/scripts/extract_d_star.py

# 3. Read what phase1-A built — d* uses similar hook plumbing.
cat /home/dgonier/ecology_experiment/trophic/trophic/m_hooks.py

# 4. Inbox check.
grep -nE "@phase2-B|@all|@phase2" /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md

# 5. Flip status.
sed -i 's/^phase2-B:02:PENDING$/phase2-B:02:RUNNING/' \
  /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md
```

## Goal

Extract a per-layer **frozen** direction vector `d*` that captures the activation difference between StockNet up days and down days. Install it via post-attention residual hooks. Verify it shifts Qwen's prefix-only baseline direction prediction by a measurable amount.

This is the second of Hexis's three orthogonal channels (M = trainable per-layer modulation; **d* = frozen direction**; curated slot = explicit text). It composes with phase1-A's M hooks but is architecturally separate.

## Files to Create / Modify

Create:
- `trophic/dstar.py` — `DStar` dataclass + `install_dstar_hooks` + `extract_dstar` (the offline extraction routine)
- `scripts/extract_dstar_stocknet.py` — entry-point script that runs the extraction over StockNet train days and saves to `checkpoints/dstar_stocknet.pt`
- `tests/test_dstar.py` — at least 4 tests

Do NOT modify:
- `trophic/phi_mlp.py` or `trophic/m_hooks.py` (phase 1's deliverables; treat as immutable)
- Any training scripts (phase 3 wires d* in)

## Implementation Steps

### 1. DStar dataclass + hooks

```python
# trophic/dstar.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from .m_hooks import _resolve_layers, HookHandle


@dataclass
class DStar:
    """Frozen per-layer pro/con direction vectors + a learnable scale.

    `directions[layer_idx]` is a unit vector in the host's hidden space.
    `scale` is a single scalar applied uniformly across patched layers
    (matches Hexis's design — you can also make it per-layer if you want).
    """
    directions: dict[int, torch.Tensor]      # layer_idx -> [H], unit-norm
    scale: float = 10.0
    hidden_size: int = 0

    def to(self, device, dtype) -> "DStar":
        return DStar(
            directions={l: v.to(device=device, dtype=dtype) for l, v in self.directions.items()},
            scale=self.scale,
            hidden_size=self.hidden_size,
        )

    def save(self, path: str | Path) -> None:
        torch.save({
            "directions": {l: v.detach().cpu().float() for l, v in self.directions.items()},
            "scale": self.scale,
            "hidden_size": self.hidden_size,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "DStar":
        st = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            directions=st["directions"],
            scale=st.get("scale", 10.0),
            hidden_size=st.get("hidden_size", next(iter(st["directions"].values())).shape[0]),
        )


def install_dstar_hooks(
    model: nn.Module,
    dstar: DStar,
    active: bool = True,
) -> list[HookHandle]:
    """Install POST-attention residual hooks: at each patched layer, add
    `scale * d_star[layer]` to the layer's output hidden state.

    NOTE: this is implemented as a forward_hook on the layer module (post),
    NOT a forward_pre_hook (which is what phase1-A's M hooks use). They run at
    different points and are explicitly orthogonal — Hexis verified the
    directional delta along d* between M-conditioned and unconditioned
    generation is -0.001 nats.
    """
    handles: list[HookHandle] = []
    gate = {"active": active}

    layers = _resolve_layers(model)

    for layer_idx, direction_vec in dstar.directions.items():
        if layer_idx >= len(layers):
            raise ValueError(f"layer_idx {layer_idx} out of range")
        target = layers[layer_idx]

        def make_hook(direction):
            def hook(module, inputs, output):
                if not gate["active"]:
                    return None
                # Decoder layer outputs are typically tuples (hidden_states, ...)
                if isinstance(output, tuple):
                    h = output[0]
                    h_new = h + dstar.scale * direction.to(h.dtype).to(h.device)
                    return (h_new,) + output[1:]
                else:
                    return output + dstar.scale * direction.to(output.dtype).to(output.device)
            return hook

        h = target.register_forward_hook(make_hook(direction_vec))
        handles.append(HookHandle(h, gate))
    return handles


# ---- extraction ----

def extract_dstar(
    base_model: nn.Module,
    tokenizer,
    pro_texts: list[str],     # texts representative of "pro" / "up" side
    con_texts: list[str],     # texts representative of "con" / "down" side
    patched_layers: list[int],
    device: torch.device,
    max_length: int = 1024,
) -> DStar:
    """Run pro and con texts through the model, accumulate per-layer mean
    activations, return d* = normalize(mean(h_pro - h_con)) per layer.

    The output is hidden_states[layer + 1] mean-pooled over the sequence
    dimension (matches extract_d_star.py template).
    """
    pro_acts: dict[int, list[torch.Tensor]] = {l: [] for l in patched_layers}
    con_acts: dict[int, list[torch.Tensor]] = {l: [] for l in patched_layers}

    base_model.eval()
    with torch.no_grad():
        for side, texts, acc in [("pro", pro_texts, pro_acts), ("con", con_texts, con_acts)]:
            for txt in texts:
                enc = tokenizer(txt, return_tensors="pt",
                                truncation=True, max_length=max_length)
                ids = enc["input_ids"].to(device)
                am = enc.get("attention_mask", torch.ones_like(ids)).to(device)
                out = base_model(input_ids=ids, attention_mask=am, output_hidden_states=True)
                for l in patched_layers:
                    # hidden_states[l + 1] is the output of layer l
                    h = out.hidden_states[l + 1].mean(dim=1).squeeze(0).float().cpu()
                    acc[l].append(h)

    directions = {}
    H = pro_acts[patched_layers[0]][0].shape[0]
    for l in patched_layers:
        pro_mean = torch.stack(pro_acts[l]).mean(dim=0)
        con_mean = torch.stack(con_acts[l]).mean(dim=0)
        diff = pro_mean - con_mean
        n = diff.norm().clamp_min(1e-9)
        directions[l] = diff / n

    return DStar(directions=directions, scale=10.0, hidden_size=H)
```

### 2. Extraction script

```python
# scripts/extract_dstar_stocknet.py
"""Extract d* per-layer direction from StockNet train-set up vs down days.

Renders each scenario as a structured prompt (same format used by
scripts/diagnostics/baseline_promptonly_stocknet.py) and feeds through Qwen.
Up-labeled days -> pro_texts; down-labeled -> con_texts.

Usage:
    .venv/bin/python -u scripts/extract_dstar_stocknet.py \
        --out checkpoints/dstar_stocknet.pt \
        --max-per-side 64

Outputs a saved DStar with directions on stride-3 layers (matching PhiMLP's
DEFAULT_PATCHED_LAYERS) and scale=10.0.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.dstar import extract_dstar
from trophic.phi_mlp import DEFAULT_PATCHED_LAYERS
from trophic.training.stocknet_loader import build_stocknet_scenarios
from trophic.training.xml_schema import parse_prediction


def render_scenario(sc) -> str:
    """Same prompt shape as the prompt-only baseline (MCC=0.292)."""
    parts = []
    for inp in sc.inputs:
        if inp.source == "ohlcv":
            bars = inp.payload.get("bars", [])
            ticker = sc.name.split("_")[2]
            parts.append(f"OHLCV history for {ticker} ({len(bars)} bars):")
            for b in bars[-5:]:
                parts.append(
                    f"  open={b.get('open',0):.6f} high={b.get('high',0):.6f} "
                    f"low={b.get('low',0):.6f} close={b.get('close',0):.6f} "
                    f"vol={b.get('volume',0)}"
                )
        elif inp.source == "press":
            t = inp.payload.get("ticker") or sc.name.split("_")[2]
            body = (inp.payload.get("body") or "").strip()
            parts.append(f"\nTweets for {t}:")
            parts.append(body[:1500])
    return "\n".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="checkpoints/dstar_stocknet.pt")
    p.add_argument("--max-per-side", type=int, default=64,
                   help="Cap on number of up days and down days each")
    p.add_argument("--split", type=str, default="train")
    args = p.parse_args()

    host = ModelHost.get(DEFAULT_CONFIG.model)
    print(f"[dstar] hidden={host.hidden_size} device={host.device}")

    scenarios = build_stocknet_scenarios(
        split=args.split,
        tickers=["AAPL", "GOOG", "MSFT", "AMZN", "JPM"],
        max_per_ticker=None,
    )
    print(f"[dstar] {len(scenarios)} {args.split} scenarios")

    pro_texts, con_texts = [], []
    for sc in scenarios:
        target = parse_prediction(sc.predator_target or "")
        txt = render_scenario(sc)
        if target.direction == "up" and len(pro_texts) < args.max_per_side:
            pro_texts.append(txt)
        elif target.direction == "down" and len(con_texts) < args.max_per_side:
            con_texts.append(txt)

    print(f"[dstar] pro={len(pro_texts)}, con={len(con_texts)}")
    if len(pro_texts) < 8 or len(con_texts) < 8:
        raise RuntimeError(f"Not enough samples: pro={len(pro_texts)} con={len(con_texts)}")

    dstar = extract_dstar(
        base_model=host._model,
        tokenizer=host._tok,
        pro_texts=pro_texts,
        con_texts=con_texts,
        patched_layers=DEFAULT_PATCHED_LAYERS,
        device=host.device,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dstar.save(args.out)
    print(f"[dstar] saved -> {args.out}")
    for l, v in dstar.directions.items():
        print(f"  layer {l:2d}: norm={v.norm().item():.4f}  (should be 1.0)")


if __name__ == "__main__":
    main()
```

### 3. Tests

```python
# tests/test_dstar.py
import torch
import torch.nn as nn

from trophic.dstar import DStar, extract_dstar, install_dstar_hooks


class _Block(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.lin = nn.Linear(H, H, bias=False)
    def forward(self, x, *args, **kwargs):
        # Returns a tuple to match HF decoder shape
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
        scale=8.0, hidden_size=H,
    )
    p = tmp_path / "d.pt"
    d.save(p)
    d2 = DStar.load(p)
    assert d2.scale == 8.0
    assert d2.hidden_size == H
    for l in d.directions:
        assert torch.allclose(d.directions[l], d2.directions[l])


def test_install_dstar_hooks_changes_output():
    H, n = 32, 4
    model = _ToyLM(H, n)
    x = torch.randn(2, 5, H)
    baseline = model(x.clone()).clone()

    dstar = DStar(
        directions={0: torch.randn(H), 2: torch.randn(H)},
        scale=10.0, hidden_size=H,
    )
    handles = install_dstar_hooks(model, dstar, active=True)
    perturbed = model(x.clone())
    for h in handles: h.remove()

    diff = (baseline - perturbed).norm().item()
    assert diff > 1e-3, f"d* hooks should change output, got diff={diff}"


def test_dstar_active_gate():
    H, n = 32, 4
    model = _ToyLM(H, n)
    x = torch.randn(2, 5, H)
    baseline = model(x.clone()).clone()

    dstar = DStar(
        directions={0: torch.randn(H), 2: torch.randn(H)},
        scale=10.0, hidden_size=H,
    )
    handles = install_dstar_hooks(model, dstar, active=False)
    out = model(x.clone())
    for h in handles: h.remove()

    assert torch.allclose(baseline, out, atol=1e-5)


def test_extract_dstar_unit_directions():
    H = 8
    patched = [0, 1]
    n_layers = 2
    model = _ToyLM(H, n_layers)
    # Provide a sentinel forward that exposes hidden_states for extract_dstar.
    # extract_dstar uses base_model(input_ids=..., output_hidden_states=True).
    # Our toy doesn't do that — write a thin shim instead.
    class _Shim:
        def __init__(self, m, n):
            self.m = m
            self.n = n
        def eval(self): self.m.eval()
        def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
            x = torch.randn(input_ids.shape[0], input_ids.shape[1], H)
            hs = [x]
            cur = x
            for l in self.m.model.layers:
                cur = l(cur)[0]
                hs.append(cur)
            class _Out: pass
            o = _Out()
            o.hidden_states = hs
            return o

    class _Tok:
        def __call__(self, txt, return_tensors=None, truncation=True, max_length=1024):
            return {
                "input_ids": torch.randint(0, 100, (1, 4)),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            }

    pro_texts = [f"pro {i}" for i in range(8)]
    con_texts = [f"con {i}" for i in range(8)]
    d = extract_dstar(
        base_model=_Shim(model, n_layers),
        tokenizer=_Tok(),
        pro_texts=pro_texts, con_texts=con_texts,
        patched_layers=patched,
        device=torch.device("cpu"),
    )
    for l in patched:
        assert torch.isclose(d.directions[l].norm(), torch.tensor(1.0), atol=1e-5)
```

### 4. Smoke (optional but recommended; not part of unit tests)

After extraction, verify d* shifts Qwen's prefix-only baseline. Use `scripts/diagnostics/baseline_promptonly_stocknet.py` as a starting point and add a `--dstar checkpoints/dstar_stocknet.pt` flag that installs d* hooks before generation. Aim for ≥0.05 absolute shift in (P(up) - P(down)) on a held-out day.

## Acceptance Criteria

- [ ] `trophic/dstar.py` exists with `DStar`, `install_dstar_hooks`, `extract_dstar`.
- [ ] `scripts/extract_dstar_stocknet.py` exists and runs to completion.
- [ ] All 5 new tests pass.
- [ ] Pre-existing 89 + phase 1's 8 tests still pass.
- [ ] After running the extraction script, `checkpoints/dstar_stocknet.pt` exists and contains unit-norm directions for every patched layer.
- [ ] Smoke check: a basic `host._model(...)` forward with d* hooks installed at `scale=10.0` produces measurably different logits than without hooks.

## Testing Conditions (exit verification)

```bash
cd /home/dgonier/ecology_experiment/trophic

# 1. New tests pass
.venv/bin/python -m pytest tests/test_dstar.py -v
# Expected: 5 passed

# 2. Pre-existing + phase 1 tests still pass
.venv/bin/python -m pytest tests/ -x --ignore=tests/test_dstar.py
# Expected: 97 passed (89 baseline + 8 from phase 1)

# 3. Extraction script runs end-to-end
.venv/bin/python -u scripts/extract_dstar_stocknet.py --out checkpoints/dstar_stocknet.pt --max-per-side 32
# Expected: prints "saved -> checkpoints/dstar_stocknet.pt", followed by per-layer norms each ≈ 1.0

# 4. Saved file loads + per-layer norms are unit
.venv/bin/python -c "
from trophic.dstar import DStar
d = DStar.load('checkpoints/dstar_stocknet.pt')
for l, v in d.directions.items():
    n = v.norm().item()
    assert 0.99 < n < 1.01, f'layer {l} not unit norm: {n}'
print(f'OK: {len(d.directions)} layers, all unit-norm, scale={d.scale}')
"
# Expected: "OK: 11 layers, all unit-norm, scale=10.0"

# 5. d* hooks change Qwen logits
.venv/bin/python -c "
import torch
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.dstar import DStar, install_dstar_hooks

host = ModelHost.get(DEFAULT_CONFIG.model)
d = DStar.load('checkpoints/dstar_stocknet.pt').to(host.device, host.dtype)
ids = host._tok('predict the direction:', return_tensors='pt').input_ids.to(host.device)
baseline = host._model(input_ids=ids).logits.float().detach().clone()
handles = install_dstar_hooks(host._model, d, active=True)
hooked = host._model(input_ids=ids).logits.float().detach().clone()
for h in handles: h.remove()
diff = (baseline - hooked).abs().max().item()
print(f'logit diff: {diff:.4f} (should be > 0.1 with scale=10.0)')
assert diff > 0.1, diff
print('OK')
"
# Expected: "logit diff: <number> > 0.1" + "OK"
```

## Coordination

You're parallel with **phase2-C** (ORPO loss) and **phase2-D** (consumer rewire). None of you touch the same files:

- B: `trophic/dstar.py`, `scripts/extract_dstar_stocknet.py`, `tests/test_dstar.py`
- C: `trophic/training/orpo.py`, `tests/test_orpo.py`
- D: `trophic/training/sft.py`, `trophic/agents/predator.py`, `trophic/agents/herbivore.py`

If you need to cross-reference phase1-A's interfaces (`_resolve_layers`, `HookHandle`), import them from `trophic.m_hooks` rather than copy-pasting.

If you discover a contract issue with `DStar` (say, you want per-layer scale instead of global), post `@all` in MESSAGES BEFORE making the change so D doesn't waste cycles wiring an old contract.

## When Done

```bash
grep -nE "@phase2-B|@all|@phase2" /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md

sed -i 's/^phase2-B:02:RUNNING$/phase2-B:02:DONE/' \
  /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md

cat >> /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md <<EOF
[$(date +%Y-%m-%d\ %H:%M)] phase2-B > @phase3-A: d* extraction landed at trophic/dstar.py. Saved checkpoint at checkpoints/dstar_stocknet.pt (11 layers, unit-norm, scale=10.0). Use install_dstar_hooks(host._model, dstar, active=True) before generation; remove handles after.
EOF

mv /home/dgonier/ecology_experiment/trophic/tasks_perlayer/phase2-B-02-dstar-extraction.md \
   /home/dgonier/ecology_experiment/trophic/tasks_perlayer/completed/
```
