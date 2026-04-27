"""Adam optimizer over all Channel parameters across all hunters.

The trainer:
  - Collects parameters from every herbivore.channels[*] and predator.channels[*]
  - Drives a single Adam optimizer
  - Accumulates per-tick loss; steps every K ticks (default 1)
  - Tracks loss curves + Channel weight norms for inspection
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..agents.herbivore import Herbivore
from ..agents.predator import Predator


@dataclass
class ChannelTrainer:
    lr: float = 5e-4
    step_every_k_ticks: int = 1
    grad_clip: float = 1.0
    # Adaptive LR knobs (ReduceLROnPlateau on eval loss)
    lr_factor: float = 0.5      # multiply LR by this when plateau detected
    lr_patience: int = 2         # eval blocks without improvement → reduce
    lr_min: float = 1e-6
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    losses: list[float] = field(default_factory=list)
    eval_losses: list[float] = field(default_factory=list)
    accum_loss: torch.Tensor | None = None
    accum_count: int = 0
    _params: list[torch.nn.Parameter] = field(default_factory=list)

    def attach(self, herbivores: list[Herbivore], predators: list[Predator]) -> int:
        params: list[torch.nn.Parameter] = []
        for h in herbivores:
            for ch in h.channels.values():
                params.extend(p for p in ch.parameters() if p.requires_grad)
        for p in predators:
            for ch in p.channels.values():
                params.extend(pp for pp in ch.parameters() if pp.requires_grad)
            # Mission 06 (phase3-A): include the learnable skip-connection
            # scalar in the trainable parameter set. The holder is created
            # lazily by `Predator.ensure_initialized`; if a caller attached
            # before init we touch the property here to materialise it.
            holder = getattr(p, "_skip_holder", None)
            if holder is None:
                # Trigger lazy init via property access.
                _ = p.skip_weight
                holder = p._skip_holder
            if holder is not None:
                params.extend(pp for pp in holder.parameters() if pp.requires_grad)
        self._params = params
        if params:
            self.optimizer = torch.optim.Adam(params, lr=self.lr)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.lr_factor,
                patience=self.lr_patience,
                min_lr=self.lr_min,
                threshold=0.05,        # 5% relative improvement to count as progress
                threshold_mode="rel",
            )
        return len(params)

    def add_loss(self, loss: torch.Tensor) -> None:
        if self.accum_loss is None:
            self.accum_loss = loss
        else:
            self.accum_loss = self.accum_loss + loss
        self.accum_count += 1

    def maybe_step(self, tick: int) -> float | None:
        if self.optimizer is None or self.accum_loss is None:
            return None
        if tick % self.step_every_k_ticks != 0:
            return None
        self.optimizer.zero_grad(set_to_none=True)
        loss_val = float(self.accum_loss.detach().item())
        self.accum_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._params, self.grad_clip)
        self.optimizer.step()
        self.losses.append(loss_val)
        self.accum_loss = None
        self.accum_count = 0
        return loss_val

    def on_eval(self, eval_loss: float) -> tuple[float, float]:
        """Call once per eval block with the held-out loss. Steps the scheduler.

        Returns (current_lr, eval_loss).
        """
        self.eval_losses.append(eval_loss)
        if self.scheduler is not None:
            self.scheduler.step(eval_loss)
        cur_lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0
        return cur_lr, eval_loss

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0

    def channel_weight_norms(
        self, herbivores: list[Herbivore], predators: list[Predator]
    ) -> dict[str, float]:
        """For inspection: how far have W_Q/W_K/W_V drifted from identity."""
        out: dict[str, float] = {}
        for h in herbivores:
            for pk, ch in h.channels.items():
                I = torch.eye(ch.hidden_size, device=ch.W_Q.weight.device,
                              dtype=ch.W_Q.weight.dtype)
                out[f"{h.kind}<-{pk}.WQ"] = float((ch.W_Q.weight - I).norm().item())
                out[f"{h.kind}<-{pk}.WK"] = float((ch.W_K.weight - I).norm().item())
                out[f"{h.kind}<-{pk}.WV"] = float((ch.W_V.weight - I).norm().item())
                out[f"{h.kind}<-{pk}.null"] = float(ch.null_bias.detach().item())
        for p in predators:
            for hk, ch in p.channels.items():
                I = torch.eye(ch.hidden_size, device=ch.W_Q.weight.device,
                              dtype=ch.W_Q.weight.dtype)
                out[f"{p.kind}<-{hk}.WQ"] = float((ch.W_Q.weight - I).norm().item())
                out[f"{p.kind}<-{hk}.WK"] = float((ch.W_K.weight - I).norm().item())
                out[f"{p.kind}<-{hk}.WV"] = float((ch.W_V.weight - I).norm().item())
                out[f"{p.kind}<-{hk}.null"] = float(ch.null_bias.detach().item())
        return out
