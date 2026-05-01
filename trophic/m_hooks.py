"""Forward-pre-hook installer for Hexis M+E rank-r perturbations.

For each patched layer ℓ, applies (per Hexis paper eq 1, simplified to a single
input-side hook — mirrors `train_action_m.py:install_hooks`):

    x' = x + s_M · (x M_A) M_B^T  +  s_E · (x E_A) E_B^T

The V-modulation second term is folded into the same input-hook as an additive
perturbation. Phase 2-D may upgrade to a true per-projection (Q vs V) hook if
needed; this matches Hexis training-script behavior today.

Prefill gating: the same handles span prefill + generate. Set
`prefill_active=True` while the model processes the prompt (no-op, avoids the
M-attractor feedback loop documented in Hexis discussion §), then call
`handle.set_prefill_active(False)` before the generation forward pass.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HookHandle:
    """Wraps a torch RemovableHandle plus a flippable prefill gate.

    Multiple HookHandles installed by a single `install_M_hooks` call share
    one `gate` dict, so flipping any of their `set_prefill_active` flips them
    all simultaneously (the typical use: turn off prefill once across all
    patched layers before generation).
    """

    def __init__(self, handle, gate: dict):
        self._handle = handle
        self._gate = gate

    def set_prefill_active(self, active: bool) -> None:
        """Flip the shared prefill flag.

        `active=True` means the model is currently in prefill, so the hooks
        should be NO-OPs. `active=False` means we're generating, hooks fire.
        """
        self._gate["prefill"] = bool(active)

    def remove(self) -> None:
        self._handle.remove()


def install_M_hooks(
    model: nn.Module,
    m_tensors_per_layer: dict[int, dict],
    prefill_active: bool = False,
) -> list[HookHandle]:
    """Install forward-pre-hooks on `model.model.layers[ℓ]` per patched layer.

    Args:
        model: HF causal LM (Qwen / Llama / Mistral); must expose a per-layer
            list at `.model.layers` (or `.transformer.h` for GPT-2-style).
        m_tensors_per_layer: dict[layer_idx -> {"M_A","M_B","E_A","E_B","s_M","s_E"}]
            as produced by `PhiMLP.forward`.
        prefill_active: initial gate state. True means hooks are no-ops
            (use during prompt prefill), False means hooks fire (use during
            generation forward passes).

    Returns:
        list[HookHandle], one per patched layer. Caller is responsible for
        eventually calling `.remove()` on each.

    Typical use:
        handles = install_M_hooks(model, M, prefill_active=True)
        _ = model(prefill_inputs)              # hooks no-op
        for h in handles: h.set_prefill_active(False)
        out = model.generate(...)              # hooks fire
        for h in handles: h.remove()
    """
    handles: list[HookHandle] = []
    gate = {"prefill": bool(prefill_active)}

    layers = _resolve_layers(model)

    for layer_idx, mod_dict in m_tensors_per_layer.items():
        if layer_idx >= len(layers):
            raise ValueError(
                f"layer_idx {layer_idx} out of range (model has {len(layers)} layers)"
            )
        target = layers[layer_idx]

        hook_fn = _make_hook(mod_dict, gate)
        h = target.register_forward_pre_hook(hook_fn, with_kwargs=True)
        handles.append(HookHandle(h, gate))

    return handles


def _make_hook(mod_dict: dict, gate: dict):
    """Build a forward-pre-hook closure binding `mod_dict` and `gate`."""
    def hook(module, args, kwargs):
        if gate["prefill"]:
            # No-op during prefill (Hexis attractor fix).
            return None

        if args:
            x = args[0]
            x_from_args = True
        else:
            x = kwargs.get("hidden_states") if kwargs else None
            x_from_args = False
        if x is None:
            return None

        x_new = apply_M_perturbation(x, mod_dict)

        if x_from_args:
            new_args = (x_new,) + tuple(args[1:])
            return new_args, kwargs
        new_kwargs = dict(kwargs) if kwargs else {}
        new_kwargs["hidden_states"] = x_new
        return args, new_kwargs

    return hook


def apply_M_perturbation(x: torch.Tensor, mod_dict: dict) -> torch.Tensor:
    """Apply the rank-r M+E perturbation to a hidden tensor x of shape [..., H].

        x' = x + s_M · (x M_A) M_B^T  +  s_E · (x E_A) E_B^T

    Tensors in mod_dict are cast to x's dtype and device on the fly.
    """
    M_A = mod_dict["M_A"].to(dtype=x.dtype, device=x.device)
    M_B = mod_dict["M_B"].to(dtype=x.dtype, device=x.device)
    E_A = mod_dict["E_A"].to(dtype=x.dtype, device=x.device)
    E_B = mod_dict["E_B"].to(dtype=x.dtype, device=x.device)
    s_M = mod_dict["s_M"].to(dtype=x.dtype, device=x.device)
    s_E = mod_dict["s_E"].to(dtype=x.dtype, device=x.device)

    m_pert = s_M * torch.matmul(torch.matmul(x, M_A), M_B.transpose(-1, -2))
    e_pert = s_E * torch.matmul(torch.matmul(x, E_A), E_B.transpose(-1, -2))
    return x + m_pert + e_pert


def _resolve_layers(model: nn.Module) -> list[nn.Module]:
    """Return the per-layer module list from a HF causal LM.

    Tries `.model.layers` (Qwen, Llama, Mistral) then `.transformer.h`
    (GPT-2 style) then a top-level `.layers`.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise AttributeError("Could not find transformer layer list on model")
