"""Decomposer — apex-judgment → per-slot attention bias.

Mission 05 (phase2-D) of the TROUGH_AS_TRANSFORMER project.

The trough-as-transformer spec collapses the original "decomposer" credit-
assignment engine (in `trophic/agents/decomposer.py`) into an attention
*bias* vector added to the next tick's Q·K logits. This module is the
small MLP that produces that bias.

Inputs:
  - `apex_judgment_embedding` — pooled hidden-state vector representing
    the apex's verdict on a predator broadcast. Mission 05 uses the
    placeholder of `host.encode_role_prefix(decoded_text)` mean-pooled.
  - `slot_lineage` — `[n_slots]` per-slot attention weights from the
    `attend()` call that produced the broadcast being judged. Tells the
    decomposer "these are the slots whose contribution is being scored".

Output:
  - `[n_slots]` bias vector, bounded to `[-max_bias, +max_bias]` via tanh.
    Added straight to the alive-slot logits in
    `TroughAttention.attend()` next tick.

The head layer is zero-initialised so this module's contribution is
exactly zero at training start; it learns to produce non-zero bias only
once gradients flow.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Decomposer(nn.Module):
    """MLP: (judgment_emb [hidden], slot_lineage [n_slots]) -> bias [n_slots].

    The output passes through `tanh * max_bias`, so bias ∈ [-max_bias,
    +max_bias]. With a zero-initialised head, the initial output is
    `tanh(0) * max_bias = 0` regardless of input.
    """

    def __init__(self, hidden_size: int, n_slots: int, max_bias: float = 2.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_slots = n_slots
        self.max_bias = float(max_bias)

        self.judgment_proj = nn.Linear(hidden_size, hidden_size)
        self.lineage_proj = nn.Linear(n_slots, hidden_size)
        self.head = nn.Linear(2 * hidden_size, n_slots)

        # Zero-init the output head so initial bias is exactly 0 for any
        # input (tanh(0) = 0). Gradients can still flow because the
        # upstream layers are not zeroed.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        apex_judgment_embedding: torch.Tensor,  # [hidden] or [B, hidden]
        slot_lineage: torch.Tensor,             # [n_slots] or [B, n_slots]
    ) -> torch.Tensor:
        """Returns a bias of shape `[n_slots]` (or `[B, n_slots]` if batched).

        Both inputs may be 1D (single example) or 2D batched. We don't
        broadcast across the leading dim — they must match.
        """
        if apex_judgment_embedding.dim() == 1:
            j_in = apex_judgment_embedding.unsqueeze(0)
        else:
            j_in = apex_judgment_embedding
        if slot_lineage.dim() == 1:
            l_in = slot_lineage.unsqueeze(0)
        else:
            l_in = slot_lineage

        if j_in.shape[-1] != self.hidden_size:
            raise ValueError(
                f"apex_judgment_embedding last dim {j_in.shape[-1]} != hidden_size {self.hidden_size}"
            )
        if l_in.shape[-1] != self.n_slots:
            raise ValueError(
                f"slot_lineage last dim {l_in.shape[-1]} != n_slots {self.n_slots}"
            )

        # Match dtype/device of the module parameters.
        target_dtype = self.judgment_proj.weight.dtype
        target_device = self.judgment_proj.weight.device
        j_in = j_in.to(dtype=target_dtype, device=target_device)
        l_in = l_in.to(dtype=target_dtype, device=target_device)

        j = self.judgment_proj(j_in)        # [B, hidden]
        l = self.lineage_proj(l_in)         # [B, hidden]
        h = torch.cat([j, l], dim=-1)       # [B, 2*hidden]
        raw = self.head(h)                  # [B, n_slots]
        bias = torch.tanh(raw) * self.max_bias

        # If the inputs were 1D, drop the leading batch dim to keep the
        # caller's life simple (TroughAttention expects [n_slots]).
        if apex_judgment_embedding.dim() == 1 and slot_lineage.dim() == 1:
            bias = bias.squeeze(0)
        return bias
