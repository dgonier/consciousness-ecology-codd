"""CrossModelChannel — the learned bridge between two model families.

A pair of trainable matrices E_a→b and E_b→a that project hidden states
between Qwen3-4B (analyst, hidden=2560) and Qwen2.5-Math-1.5B (math,
hidden=1536). Used by the InterrogatorHerbivore to bypass tokens on its
internal call to the Math node.

Architecturally identical to a producer→herbivore Channel but without
the attention layer: it's just a pair of projections plus a small MLP
for non-linearity.

The forward pass at each direction:
  E: x → linear → GELU → linear → output
where x is in src model's hidden dim and output is in dst model's hidden
dim. Identity-init via near-orthogonal-init so untrained behavior is at
least bounded (random-but-small).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModelProjection(nn.Module):
    """Single-direction projection between two model embedding spaces.

    Two linear layers with GELU between, plus optional layer norm output
    so the projected vector lives in the target model's typical norm
    range.
    """

    def __init__(
        self,
        src_dim: int,
        dst_dim: int,
        inner_dim: int | None = None,
        target_norm: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.src_dim = src_dim
        self.dst_dim = dst_dim
        self.target_norm = target_norm
        inner = inner_dim or max(src_dim, dst_dim) * 2

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(int(seed))
        else:
            gen.seed()

        self.proj_in = nn.Linear(src_dim, inner, bias=False)
        self.proj_out = nn.Linear(inner, dst_dim, bias=False)
        self.layer_norm = nn.LayerNorm(dst_dim)

        with torch.no_grad():
            self.proj_in.weight.copy_(
                torch.empty_like(self.proj_in.weight).normal_(
                    0.0, 1.0 / math.sqrt(src_dim), generator=gen,
                )
            )
            self.proj_out.weight.copy_(
                torch.empty_like(self.proj_out.weight).normal_(
                    0.0, 1.0 / math.sqrt(inner), generator=gen,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project x from src space to dst space.

        x: [..., src_dim]
        returns: [..., dst_dim]
        """
        z = F.gelu(self.proj_in(x))
        out = self.proj_out(z)
        out = self.layer_norm(out)
        # Re-norm to target_norm so it lives in dst-model's input range.
        cur_norm = out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return out / cur_norm * self.target_norm


@dataclass
class CrossModelChannel:
    """Holds the two projection directions between a pair of models.

    Convention used here:
      analyst ↔ math:
        E_a_to_m: 2560 → 1536  (Qwen3-4B → Qwen2.5-Math)
        E_m_to_a: 1536 → 2560  (Qwen2.5-Math → Qwen3-4B)

    Both are nn.Modules with their own parameters. Treat the dataclass as
    a container; iterate its `parameters()` for the trainer.
    """
    a_to_m: CrossModelProjection
    m_to_a: CrossModelProjection

    def parameters(self):
        yield from self.a_to_m.parameters()
        yield from self.m_to_a.parameters()

    def to(self, device, dtype):
        self.a_to_m.to(device=device, dtype=dtype)
        self.m_to_a.to(device=device, dtype=dtype)
        return self

    def state_dict(self) -> dict:
        return {
            "a_to_m": self.a_to_m.state_dict(),
            "m_to_a": self.m_to_a.state_dict(),
        }

    def load_state_dict(self, sd: dict, strict: bool = True) -> None:
        self.a_to_m.load_state_dict(sd["a_to_m"], strict=strict)
        self.m_to_a.load_state_dict(sd["m_to_a"], strict=strict)

    @classmethod
    def make(
        cls,
        analyst_dim: int,
        math_dim: int,
        analyst_target_norm: float,
        math_target_norm: float,
        seed: int | None = None,
    ) -> "CrossModelChannel":
        return cls(
            a_to_m=CrossModelProjection(
                src_dim=analyst_dim, dst_dim=math_dim,
                target_norm=math_target_norm,
                seed=(seed * 7 if seed is not None else None),
            ),
            m_to_a=CrossModelProjection(
                src_dim=math_dim, dst_dim=analyst_dim,
                target_norm=analyst_target_norm,
                seed=(seed * 11 if seed is not None else None),
            ),
        )
