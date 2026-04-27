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
) -> None:
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
    torch.save(state, path)


def load_channels(
    path: str,
    herbivores: list[Herbivore],
    predators: list[Predator],
    strict: bool = True,
) -> dict:
    """Load checkpoint into agents' existing Channels (must already be built).

    Returns the meta dict from the checkpoint.
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
