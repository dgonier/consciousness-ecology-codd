"""PhiMLP: compile a trough-attended hidden vector into per-layer M+E modulation tensors.

Mirrors the Hexis-paper M-architecture (eq 1 in §3.1):
    x'_ℓ = x_ℓ + s_M · (x_ℓ M_A_ℓ) M_B_ℓ^T
    V'_ℓ = V_ℓ + s_E · (x_ℓ E_A_ℓ) E_B_ℓ^T

Phase 1-A foundation primitive — used by phase 2-D (consumer rewire) and phase 3-A (training).

Param-budget refactor (phase 3-A:05 follow-up, 2026-05-01):
    The naive head ``Linear(bottleneck, n_layers · 4·H·r + …)`` blows past 1 B
    params on Qwen3-4B (H=2560, r=16, n=11). Three predator agents at that size
    OOM a 24 GB GPU during training-eval.

    Refactor: keep phi as an *input-dependent compile function* (this is what
    differentiates trophic-as-federated-M from single-agent Hexis), but factor
    the head as
        shared global head:  Linear(bottleneck, H · r) per channel (M_A, M_B,
                              E_A, E_B) — produces a single [H, r] template.
        per-layer modulator:  small [r, r] matrix per (channel, layer).
        per-layer scalars:    s_M, s_E learnable scalars per layer.
    so M_A_ℓ = template_M_A @ G_M_A_ℓ, and similarly for M_B/E_A/E_B.

    Param count at H=2560, r=16, bottleneck=256, 11 layers:
        bottleneck linear:        H · bottleneck = 655,360
        4 channel heads:          4 · bottleneck · H · r ≈ 41,943,040
        per-layer [r,r] mods:     4 · n_layers · r · r = 11,264
        per-layer scalars:        2 · n_layers = 22
        total                     ≈ 42.6 M  (well under 50 M target).

    Public ``forward`` contract is unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_RANK = 16
# Stride-3 over Qwen3-4B's 32 layers gives 11 patched layers — matches the Hexis paper.
DEFAULT_PATCHED_LAYERS = list(range(0, 32, 3))
# Bottleneck dim; smaller than the legacy H/4 in order to fit param budget.
DEFAULT_BOTTLENECK = 256


class PhiMLP(nn.Module):
    """Compile a trough-attended hidden state [H] -> per-layer M+E modulation tensors.

    Architecture:
        Shared bottleneck (H -> bottleneck -> SiLU). Four shared "channel heads"
        (one each for M_A, M_B, E_A, E_B) emit a single global ``[H, r]``
        template from the bottleneck output. Per-layer ``[r, r]`` modulator
        matrices project the templates into per-layer tensors. Per-layer
        scalars (``s_M``, ``s_E``) are direct learnable parameters (not
        produced from the bottleneck — keeps the head linear-in-bottleneck).

    Initialization:
        - Bottleneck weights small-Gaussian (std = 1/sqrt(H)).
        - Channel heads zero-initialized so phi(x) returns all-zero
          M_A/M_B/E_A/E_B at step 0.
        - Per-layer modulators initialized to identity (so when the channel
          head learns, the per-layer tensor is initially equal across layers
          and per-layer divergence is learned).
        - Per-layer scalars zero-initialized so the M-hooks are an identity
          transform until the head learns. This protects the baseline Qwen
          behavior on the first forward pass.

    Args:
        hidden_size: H, model hidden dim (e.g. 2560 for Qwen3-4B).
        rank: r, low-rank dimension for each (A, B) pair (default 16).
        patched_layers: list[int] layer indices to modulate. Default
            stride-3 over 32 layers (11 patched layers).
        bottleneck: int, intermediate width (default 256).

    Forward:
        trough_attended_hidden: [H] (single 1-D vector).
        Returns dict[layer_idx, dict] with keys
            "M_A": [H, r], "M_B": [H, r],
            "E_A": [H, r], "E_B": [H, r],
            "s_M": [] scalar, "s_E": [] scalar.
    """

    _CHANNELS = ("M_A", "M_B", "E_A", "E_B")

    def __init__(
        self,
        hidden_size: int,
        rank: int = DEFAULT_RANK,
        patched_layers: list[int] | None = None,
        bottleneck: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.patched_layers = (
            list(patched_layers)
            if patched_layers is not None
            else list(DEFAULT_PATCHED_LAYERS)
        )
        n_layers = len(self.patched_layers)

        # Bottleneck — keep small; default 256 stays well under the param
        # budget on Qwen3-4B even with hidden_size=2560.
        if bottleneck is None:
            bottleneck = min(DEFAULT_BOTTLENECK, max(hidden_size // 4, 1))
        self.bottleneck_dim = bottleneck

        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_size, bottleneck, bias=False),
            nn.SiLU(),
        )

        # Shared "channel heads": one Linear(bottleneck, H * r) per channel.
        # The output is the global template, reshaped to [H, r].
        self.channel_heads = nn.ModuleDict(
            {
                ch: nn.Linear(bottleneck, hidden_size * rank, bias=False)
                for ch in self._CHANNELS
            }
        )

        # Per-layer [r, r] modulators, one per channel. Stored as a tensor
        # of shape [n_layers, r, r] per channel for fast indexing.
        self.layer_mods = nn.ParameterDict(
            {
                ch: nn.Parameter(torch.zeros(n_layers, rank, rank))
                for ch in self._CHANNELS
            }
        )

        # Per-layer scalars s_M, s_E. Initialized SMALL but non-zero so the
        # gradient saddle (zero-init scale × zero-init head -> zero gradient
        # in BOTH directions, observed in seed24 v3 where pred dev loss was
        # bit-identical across 5 evals) is broken at step 0. Once channel
        # heads start learning, s_M/s_E gradient is non-trivial.
        self.s_M = nn.Parameter(torch.full((n_layers,), 0.01))
        self.s_E = nn.Parameter(torch.full((n_layers,), 0.01))

        with torch.no_grad():
            for m in self.bottleneck.modules():
                if isinstance(m, nn.Linear):
                    m.weight.normal_(0.0, 1.0 / hidden_size**0.5)
            for ch in self._CHANNELS:
                # Small-Gaussian init the channel heads (NOT zero) so the
                # multiplicative path s_scale * (x A) B^T is non-zero at step 0.
                # See seed24 v3 diagnosis: zero-init head + zero-init scale =
                # gradient-zero saddle that can't escape.
                self.channel_heads[ch].weight.normal_(
                    0.0, 1.0 / (bottleneck * hidden_size) ** 0.5,
                )
                # Identity-init the per-layer modulator so that once the
                # channel head learns, every layer starts off equal and
                # divergence is learned.
                eye = torch.eye(rank).unsqueeze(0).expand(n_layers, rank, rank).clone()
                self.layer_mods[ch].copy_(eye)

    def forward(self, trough_attended_hidden: torch.Tensor) -> dict[int, dict]:
        if trough_attended_hidden.dim() != 1:
            raise ValueError(
                f"PhiMLP expects 1-D [H] input, got shape "
                f"{tuple(trough_attended_hidden.shape)}"
            )
        if trough_attended_hidden.shape[0] != self.hidden_size:
            raise ValueError(
                f"PhiMLP hidden_size mismatch: expected {self.hidden_size}, "
                f"got {trough_attended_hidden.shape[0]}"
            )

        H, r = self.hidden_size, self.rank
        z = self.bottleneck(trough_attended_hidden)  # [bottleneck]

        # Compute the four global templates [H, r], each input-dependent.
        templates = {}
        for ch in self._CHANNELS:
            t = self.channel_heads[ch](z)  # [H * r]
            templates[ch] = t.view(H, r)

        out: dict[int, dict] = {}
        for i, layer_idx in enumerate(self.patched_layers):
            layer_dict = {}
            for ch in self._CHANNELS:
                # template [H, r] @ G_layer [r, r] -> [H, r]
                layer_dict[ch] = templates[ch] @ self.layer_mods[ch][i]
            # Issue #15 P1 (seed26/seed28 NaN fix): bound s_M / s_E so
            # the multiplicative path s * (x A) B^T can't push Qwen
            # hidden states to NaN at deploy time on out-of-distribution
            # inputs. tanh(s) ∈ (-1, 1) keeps the perturbation magnitude
            # finite; the unconstrained scalar is used by gradient flow,
            # the bounded scalar is what hooks see.
            layer_dict["s_M"] = torch.tanh(self.s_M[i]) * 0.5
            layer_dict["s_E"] = torch.tanh(self.s_E[i]) * 0.5
            out[layer_idx] = layer_dict
        return out
