"""Save/load Channel checkpoints (predator/herbivore Channels) and
CrossModelChannel checkpoints (analyst↔math).

A checkpoint bundles the state_dicts of every Channel in every herbivore +
predator into a single file. Used to:
  - Persist post-SFT weights so a separate process can pick up training.
  - Provide a frozen reference snapshot for GRPO's KL penalty.

Format: torch.save dict
  {
    "herb": {
      "<herb_kind>": {
        "<source_kind>": <Channel.state_dict()>,
        ...
      },
      ...
    },
    "pred": {
      "<pred_kind>": {
        "<source_kind>": <Channel.state_dict()>,
        ...
      },
      ...
    },
    "meta": {"sft_seed": int, "step": int, "eval_loss": float | None},
  }
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..agents.herbivore import Herbivore
from ..agents.predator import Predator


def save_channels(
    path: str,
    herbivores: list[Herbivore],
    predators: list[Predator],
    meta: dict | None = None,
    runner: object | None = None,
) -> None:
    """Save trainable Channel state_dicts for every herbivore and predator.

    If `runner` is supplied (an SFTRunner), additionally persists per-agent
    phi_mlp state_dicts (hooks-mode, phase 2-D) and any troughs the runner
    holds. This is the seed24+ path; older checkpoints (seed1..seed23) saved
    only Channels and back-load fine because phi_mlp/troughs default to None.
    """
    state = {
        "herb": {},
        "pred": {},
        "meta": dict(meta or {}),
    }
    for h in herbivores:
        state["herb"][h.kind] = {
            src: ch.state_dict() for src, ch in h.channels.items()
        }
    for p in predators:
        state["pred"][p.kind] = {
            src: ch.state_dict() for src, ch in p.channels.items()
        }

    # phase 2-D / phase 3-A bug fix: include phi_mlp + trough state so
    # hooks-mode checkpoints actually preserve the trained M+E compiler.
    # Pre-fix, phi_mlp was reinitialized on every fresh load → eval used
    # untrained phi_mlp → 100% gibberish on StockNet despite real
    # in-training dev numbers.
    phi_state = {"herb": {}, "pred": {}}
    for h in herbivores:
        if getattr(h, "phi_mlp", None) is not None:
            phi_state["herb"][h.kind] = h.phi_mlp.state_dict()
    for p in predators:
        if getattr(p, "phi_mlp", None) is not None:
            phi_state["pred"][p.kind] = p.phi_mlp.state_dict()
    if phi_state["herb"] or phi_state["pred"]:
        state["phi_mlp"] = phi_state

    if runner is not None:
        trough_state: dict = {}
        if getattr(runner, "_producer_trough", None) is not None:
            trough_state["producer"] = runner._producer_trough.state_dict()
        if getattr(runner, "_herb_trough", None) is not None:
            trough_state["herb"] = runner._herb_trough.state_dict()
        if trough_state:
            state["troughs"] = trough_state

    torch.save(state, path)


def load_channels(
    path: str,
    herbivores: list[Herbivore],
    predators: list[Predator],
    strict: bool = True,
    runner: object | None = None,
) -> dict:
    """Load checkpoint into agents' existing Channels (must already be built).

    Returns the meta dict from the checkpoint.

    If `runner` is supplied AND the checkpoint contains "troughs", load those
    state_dicts into the runner's _producer_trough / _herb_trough (constructing
    them if necessary). Otherwise troughs are left untouched.

    phi_mlp state_dicts are loaded onto each agent's `phi_mlp` if both the
    checkpoint and the agent have them. Agents without phi_mlp (prefix mode)
    silently skip — backward compat with seed1..seed23 checkpoints.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    for h in herbivores:
        kind_state = state.get("herb", {}).get(h.kind, {})
        for src, ch in h.channels.items():
            if src not in kind_state:
                if strict:
                    raise KeyError(f"missing herbivore[{h.kind}].channel[{src}] in checkpoint")
                continue
            ch.load_state_dict(kind_state[src], strict=strict)
    for p in predators:
        kind_state = state.get("pred", {}).get(p.kind, {})
        for src, ch in p.channels.items():
            if src not in kind_state:
                if strict:
                    raise KeyError(f"missing predator[{p.kind}].channel[{src}] in checkpoint")
                continue
            ch.load_state_dict(kind_state[src], strict=strict)

    # phi_mlp state (phase 2-D, hooks mode). Backward-compat: missing on
    # seed1..seed23 checkpoints, silently skipped.
    phi = state.get("phi_mlp", {})
    if phi:
        for h in herbivores:
            if getattr(h, "phi_mlp", None) is not None and h.kind in phi.get("herb", {}):
                h.phi_mlp.load_state_dict(phi["herb"][h.kind])
        for p in predators:
            if getattr(p, "phi_mlp", None) is not None and p.kind in phi.get("pred", {}):
                p.phi_mlp.load_state_dict(phi["pred"][p.kind])

    # Trough state. Optional; only loaded if runner is supplied.
    if runner is not None:
        troughs = state.get("troughs", {})
        if troughs:
            from ..trough_attention import TroughAttention
            host = getattr(runner, "host", None)
            for key, runner_attr in (("producer", "_producer_trough"),
                                      ("herb", "_herb_trough")):
                if key not in troughs:
                    continue
                cur = getattr(runner, runner_attr, None)
                # Construct a fresh trough that matches the saved shape if
                # the runner doesn't already have one. We can't infer
                # use_E_in / gated_residual from the state_dict alone, so
                # we conservatively assume they were True (hooks-mode default).
                if cur is None and host is not None:
                    n_slots = troughs[key].get("V_store", torch.zeros(32, 1)).shape[0]
                    cur = TroughAttention(
                        hidden_size=host.hidden_size,
                        n_slots=n_slots,
                        n_heads=8 if host.hidden_size % 8 == 0 else 4,
                        out_seq_len=8,
                        seed=runner.cfg.seed + (9001 if key == "producer" else 9002),
                        use_E_in=True,
                        gated_residual=True,
                    ).to(device=host.device, dtype=host.dtype)
                    setattr(runner, runner_attr, cur)
                if cur is not None:
                    cur.load_state_dict(troughs[key], strict=False)

    return state.get("meta", {})


@dataclass
class FrozenChannelSnapshot:
    """Holds a CPU-fp32 deep copy of every Channel's state_dict for KL ref.

    Reconstructed Channels (with the same architecture) are kept as nn.Modules
    in eval mode so we can run forward through them for KL probability under
    the reference policy.
    """
    herb_states: dict[str, dict[str, dict[str, torch.Tensor]]]
    pred_states: dict[str, dict[str, dict[str, torch.Tensor]]]

    @classmethod
    def from_agents(
        cls, herbivores: list[Herbivore], predators: list[Predator]
    ) -> "FrozenChannelSnapshot":
        herb_states: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
        for h in herbivores:
            herb_states[h.kind] = {
                src: {k: v.detach().clone().cpu() for k, v in ch.state_dict().items()}
                for src, ch in h.channels.items()
            }
        pred_states: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
        for p in predators:
            pred_states[p.kind] = {
                src: {k: v.detach().clone().cpu() for k, v in ch.state_dict().items()}
                for src, ch in p.channels.items()
            }
        return cls(herb_states=herb_states, pred_states=pred_states)


# ---------- cross-model channel ----------

def save_cross_model_channel(path: str, channel, meta: dict | None = None) -> None:
    """Save a CrossModelChannel's state dict."""
    state = {"channel": channel.state_dict(), "meta": dict(meta or {})}
    torch.save(state, path)


def load_cross_model_channel(path: str, channel, strict: bool = True) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    channel.load_state_dict(state["channel"], strict=strict)
    return state.get("meta", {})
