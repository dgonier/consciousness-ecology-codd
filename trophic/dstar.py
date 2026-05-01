"""d* — frozen per-layer pro/con direction vectors with a learnable scale.

Hexis's second orthogonal channel:
    M       = trainable per-layer rank-r modulation     (phase1-A: phi_mlp.py + m_hooks.py)
    d*      = frozen per-layer direction vector          (this file)
    curated = explicit text in the prompt                (phase 3 wires)

Application math (per patched layer ℓ, post-attention residual):
    h'_ℓ = h_ℓ + scale * d_star.directions[ℓ]

`directions[ℓ]` is unit-norm in the host's hidden space. `scale` is a single
scalar (default 10.0); we keep it as a plain float on the dataclass so it can
be wired into a learnable parameter outside this module if desired.

The hooks are **forward (post)** hooks on each patched layer — distinct from
phase1-A's M hooks which are **forward-pre** hooks. This keeps the two
channels architecturally orthogonal (Hexis verified directional delta along
d* between M-conditioned and unconditioned generation is -0.001 nats).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .m_hooks import HookHandle, _resolve_layers


@dataclass
class DStar:
    """Frozen per-layer pro/con direction vectors + a (potentially learnable) scale.

    Attributes:
        directions: dict[layer_idx -> [H] unit-norm tensor]. Frozen after
            extraction.
        scale: single scalar, applied uniformly across patched layers.
            Default 10.0 (matches Hexis verification at scale 10–20).
        hidden_size: H, recorded for sanity checking on load.
    """

    directions: dict[int, torch.Tensor]
    scale: float = 10.0
    hidden_size: int = 0

    def to(self, device, dtype) -> "DStar":
        return DStar(
            directions={l: v.to(device=device, dtype=dtype) for l, v in self.directions.items()},
            scale=self.scale,
            hidden_size=self.hidden_size,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "directions": {l: v.detach().cpu().float() for l, v in self.directions.items()},
                "scale": float(self.scale),
                "hidden_size": int(self.hidden_size),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DStar":
        st = torch.load(path, map_location="cpu", weights_only=False)
        directions = st["directions"]
        first = next(iter(directions.values()))
        return cls(
            directions=dict(directions),
            scale=float(st.get("scale", 10.0)),
            hidden_size=int(st.get("hidden_size", first.shape[0])),
        )


def install_dstar_hooks(
    model: nn.Module,
    dstar: DStar,
    active: bool = True,
) -> list[HookHandle]:
    """Install POST forward hooks on each patched layer that add scale * d* to
    the layer's output hidden state.

    Args:
        model: HF causal LM (Qwen / Llama / Mistral) with `.model.layers` (or
            `.transformer.h`) per `_resolve_layers`.
        dstar: a DStar with `directions[layer_idx]` populated for every layer
            index to be modulated.
        active: initial gate state. Mutate via the returned HookHandles'
            `_gate["active"]` if needed (tests do this through install/remove
            cycles rather than flipping in-place).

    Returns:
        list[HookHandle], one per patched layer. Caller is responsible for
        eventually calling `.remove()` on each.
    """
    handles: list[HookHandle] = []
    gate = {"active": bool(active)}

    layers = _resolve_layers(model)

    for layer_idx, direction_vec in dstar.directions.items():
        if layer_idx >= len(layers):
            raise ValueError(
                f"layer_idx {layer_idx} out of range (model has {len(layers)} layers)"
            )
        target = layers[layer_idx]
        h = target.register_forward_hook(_make_hook(direction_vec, dstar.scale, gate))
        handles.append(HookHandle(h, gate))
    return handles


def _make_hook(direction: torch.Tensor, scale: float, gate: dict):
    """Build a post-forward hook closure binding `direction` and `scale`."""

    def hook(module, inputs, output):
        if not gate["active"]:
            return None
        if isinstance(output, tuple):
            h = output[0]
            d = direction.to(dtype=h.dtype, device=h.device)
            h_new = h + scale * d
            return (h_new,) + output[1:]
        d = direction.to(dtype=output.dtype, device=output.device)
        return output + scale * d

    return hook


# ---------------------------------------------------------------------------
# Offline extraction
# ---------------------------------------------------------------------------

def extract_dstar(
    base_model: nn.Module,
    tokenizer,
    pro_texts: list[str],
    con_texts: list[str],
    patched_layers: list[int],
    device: torch.device,
    max_length: int = 1024,
) -> DStar:
    """Run pro and con texts through the model, accumulate per-layer mean
    activations, return d* = normalize(mean(h_pro) - mean(h_con)) per layer.

    The activation captured is `out.hidden_states[layer + 1]` (the output of
    layer `layer`), mean-pooled over the sequence dimension. This matches the
    Hexis `extract_d_star.py` template.

    Args:
        base_model: HF causal LM with `output_hidden_states=True` support.
        tokenizer: HF tokenizer.
        pro_texts, con_texts: side-labeled text examples.
        patched_layers: list of layer indices (e.g. `DEFAULT_PATCHED_LAYERS`).
        device: torch device for inputs.
        max_length: tokenizer truncation max length.

    Returns:
        DStar with unit-norm `directions[ℓ]` for every ℓ in patched_layers,
        and `scale=10.0` by default.
    """
    if not pro_texts or not con_texts:
        raise ValueError(
            f"extract_dstar needs both pro_texts and con_texts, got "
            f"{len(pro_texts)} pro / {len(con_texts)} con"
        )

    pro_acts: dict[int, list[torch.Tensor]] = {l: [] for l in patched_layers}
    con_acts: dict[int, list[torch.Tensor]] = {l: [] for l in patched_layers}

    base_model.eval()
    with torch.no_grad():
        for side_name, texts, acc in [
            ("pro", pro_texts, pro_acts),
            ("con", con_texts, con_acts),
        ]:
            for txt in texts:
                enc = tokenizer(
                    txt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                ids = enc["input_ids"].to(device)
                am = enc.get("attention_mask")
                if am is None:
                    am = torch.ones_like(ids)
                am = am.to(device)
                out = base_model(
                    input_ids=ids,
                    attention_mask=am,
                    output_hidden_states=True,
                )
                for l in patched_layers:
                    # hidden_states[l+1] is the output of layer l; mean-pool seq.
                    h = out.hidden_states[l + 1].mean(dim=1).squeeze(0).float().cpu()
                    acc[l].append(h)

    H = pro_acts[patched_layers[0]][0].shape[0]
    directions: dict[int, torch.Tensor] = {}
    for l in patched_layers:
        pro_mean = torch.stack(pro_acts[l]).mean(dim=0)
        con_mean = torch.stack(con_acts[l]).mean(dim=0)
        diff = pro_mean - con_mean
        n = diff.norm().clamp_min(1e-9)
        directions[l] = diff / n

    return DStar(directions=directions, scale=10.0, hidden_size=H)
