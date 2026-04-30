"""EnvironmentStream — input-layer K/V substrate.

Architectural role
------------------
The environment stream sits **at the input layer**, one trophic tier below
producers. Adapters (`trophic/adapters/`) deposit `RawInput` observations
(after embedding via the host model) into per-slot K/V cells; producers
then attend into the stream using their own queries, but each producer
only "sees" the slots whose source tag matches the producer's WAVELENGTHS
subset. The result is a cortical-column fan-out: each producer pulls a
different overlapping subset of the same shared substrate.

This module **composes** `TroughAttention` for the heavy K/V machinery
(deposit, multi-head attend, slot lifecycle, niche-aware spawn) and adds
the two stream-specific bits:

  1. Per-slot source-tag bookkeeping (`_slot_source_tags`).
  2. Wavelength-filtered attend — slots whose source_tag is not in the
     caller's `wavelength_filter` are pushed to ``-inf`` logits before
     softmax, via the existing ``external_bias`` hook on
     ``TroughAttention.attend``.

Design choices
--------------
* **Compose, don't reimplement.** TroughAttention already owns the K/V
  matrix, lifecycle pipeline, and ``external_bias`` plumbing. We only
  wrap it.
* **Deposit via minimal wrapper, not a new TroughAttention method.** We
  build a lightweight ``_RawInputBroadcastShim`` inside ``deposit`` that
  exposes the ``.id`` / ``.channel_embedding`` pair TroughAttention.deposit
  expects. This keeps the diff to ``trough_attention.py`` at zero (Mission
  02 acceptance criterion: don't break phase2-A/C/D invariants on the
  underlying trough).
* **Wavelength masking via ``external_bias``.** Build a ``[n_slots]`` bias
  vector with ``0.0`` for "in-filter & alive" slots and ``-inf`` for the
  rest, and pass it straight through. Cleanest implementation — no
  duplicated softmax code, fully reuses TroughAttention's existing path.

See ``tasks_envstream/phase2-B-02-environment-stream.md`` for the full
mission spec.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .trough_attention import TroughAttention
from .types import RawInput


@dataclass
class StreamAttendOutput:
    """Diagnostic-rich return type for ``EnvironmentStream.attend``.

    Mirrors ``TroughAttendOutput`` plus a single new field,
    ``masked_out_count``, that reports how many alive slots the
    wavelength filter excluded on this call. Producers can use that to
    detect "I'm subscribing to a wavelength nothing's depositing into."
    """
    output: torch.Tensor              # [out_seq_len, hidden]
    per_slot_attention: torch.Tensor  # [n_slots]
    per_head_attention: torch.Tensor  # [n_heads, n_slots]
    null_prob: float
    selected_slot_ids: list[int]
    rejected_slot_ids: list[int]
    masked_out_count: int             # alive slots excluded by wavelength_filter


@dataclass
class _RawInputBroadcastShim:
    """Minimal duck-type matching what ``TroughAttention.deposit`` reads.

    The trough's ``deposit`` only needs ``.id: str`` and
    ``.channel_embedding: Sequence[float]``. We construct this shim in
    ``EnvironmentStream.deposit`` so we don't have to touch
    ``trough_attention.py`` at all (zero-diff requirement on the trough
    side; phase2-A and others are reading those invariants).
    """
    id: str
    channel_embedding: list[float]


class EnvironmentStream(nn.Module):
    """Input-layer K/V substrate.

    Receives ``(RawInput, embedding)`` deposits from adapters; producers
    subscribe to wavelength subsets and read via wavelength-filtered
    cross-attention.

    Public surface (matches the INTERFACE CONTRACT in
    ``tasks_envstream/scratchpad.md``):

      - ``deposit(raw_input, embedding) -> slot_id``
      - ``attend(query, wavelength_filter, tau) -> StreamAttendOutput``
      - ``step_lifecycle() -> dict`` (delegates to underlying trough)
      - ``slot_state() -> dict`` (augments trough state with source tags)
    """

    def __init__(
        self,
        hidden_size: int,
        n_slots: int,
        n_heads: int = 8,
        out_seq_len: int = 8,
        target_norm: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__()
        self._trough = TroughAttention(
            hidden_size=hidden_size,
            n_slots=n_slots,
            n_heads=n_heads,
            out_seq_len=out_seq_len,
            target_norm=target_norm,
            seed=seed,
        )
        # Per-slot source tag (Python list, not buffer — these are
        # short strings, not tensor-friendly. Indexed parallel to
        # TroughAttention.alive / V_store / broadcast_ids.
        self._slot_source_tags: list[str | None] = [None] * n_slots
        # Per-slot raw-input id (parallel to source_tags). The trough's
        # broadcast_ids list also stores this, but we keep an explicit
        # copy here for clarity in slot_state diagnostics.
        self._slot_raw_input_ids: list[str | None] = [None] * n_slots

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    @property
    def n_slots(self) -> int:
        return self._trough.n_slots

    @property
    def hidden_size(self) -> int:
        return self._trough.hidden_size

    @property
    def n_heads(self) -> int:
        return self._trough.n_heads

    @property
    def alive(self) -> torch.Tensor:
        return self._trough.alive

    # ------------------------------------------------------------------
    # K/V membership
    # ------------------------------------------------------------------

    def deposit(self, raw_input: RawInput, embedding: torch.Tensor) -> int:
        """Place a single (raw_input, embedding) pair into a free slot.

        Returns the slot id used. Caller is responsible for having
        already embedded the RawInput's payload via the host model.

        Raises ``RuntimeError`` if no free slots are available — that's
        the underlying trough's overflow contract; ``step_lifecycle``
        is expected to free slots over time as their attention EMA
        decays below ``epsilon_death``.
        """
        if embedding.dim() != 1 or embedding.shape[0] != self.hidden_size:
            raise ValueError(
                f"embedding must be 1D of length hidden_size={self.hidden_size}, "
                f"got shape {tuple(embedding.shape)}"
            )

        shim = _RawInputBroadcastShim(
            id=raw_input.id,
            channel_embedding=embedding.detach().cpu().tolist(),
        )
        slot_ids = self._trough.deposit([shim])
        slot_id = slot_ids[0]
        self._slot_source_tags[slot_id] = raw_input.source
        self._slot_raw_input_ids[slot_id] = raw_input.id
        return slot_id

    # ------------------------------------------------------------------
    # attention readout
    # ------------------------------------------------------------------

    def attend(
        self,
        query: torch.Tensor,
        wavelength_filter: set[str],
        tau: float = 1.0,
    ) -> StreamAttendOutput:
        """Wavelength-filtered cross-attention readout.

        Slots whose ``source_tag`` is not in ``wavelength_filter`` are
        excluded by pushing their attention logits to ``-inf`` before
        softmax. Implementation: build a ``[n_slots]`` ``external_bias``
        with ``0.0`` for "alive AND in-filter" slots and ``-inf`` for
        the rest, and pass it through the underlying trough's
        ``attend`` method. Dead slots are already gated out by the
        trough's ``alive_mask``; the bias only affects slots that would
        otherwise have contributed.

        ``masked_out_count`` reports how many *alive* slots the filter
        excluded (so a producer subscribing to a quiet wavelength can
        notice it).
        """
        device = self._trough.V_store.device
        dtype = self._trough.V_store.dtype

        alive_t = self._trough.alive.detach().cpu().tolist()
        # Build the per-slot in-filter mask. None tags are never
        # in-filter (those slots are dead anyway, but belt-and-brace).
        in_filter = [
            (tag is not None) and (tag in wavelength_filter)
            for tag in self._slot_source_tags
        ]

        # Slots that are alive but masked out by the wavelength filter.
        masked_out_count = sum(
            1 for i in range(self.n_slots)
            if alive_t[i] and not in_filter[i]
        )

        # external_bias: 0 for keep-slots, -inf for mask-slots. Apply
        # to ALL slots (alive + dead); the trough only adds it to the
        # alive subset, but we set it consistently for clarity.
        bias = torch.zeros(self.n_slots, device=device, dtype=dtype)
        for i in range(self.n_slots):
            if not in_filter[i]:
                bias[i] = float("-inf")

        out = self._trough.attend(query, tau=tau, external_bias=bias)

        return StreamAttendOutput(
            output=out.output,
            per_slot_attention=out.per_slot_attention,
            per_head_attention=out.per_head_attention,
            null_prob=out.null_prob,
            selected_slot_ids=out.selected_slot_ids,
            rejected_slot_ids=out.rejected_slot_ids,
            masked_out_count=masked_out_count,
        )

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def step_lifecycle(self) -> dict:
        """Delegate the death / respawn pipeline to the underlying trough.

        Also clears our per-slot source-tag bookkeeping for any slot
        the trough killed this tick (so a future deposit into the
        recycled slot doesn't inherit a stale tag).

        Temporal decay (kill slots older than ``max_age`` regardless
        of attention) is intentionally out of scope for Mission 02 —
        the EMA-based attention decay is the only death rule for now.
        """
        stats = self._trough.step_lifecycle()
        for sid in stats.get("killed_ids", []):
            self._slot_source_tags[sid] = None
            self._slot_raw_input_ids[sid] = None
        # If the trough respawned (population_strategy='fixed' default),
        # the spawned slot has no source tag — it's a synthetic
        # niche-locked V vector, not a deposit. Tag it with a synthetic
        # marker so wavelength filters never accidentally pick it up.
        for spawn in stats.get("spawned", []) or []:
            sid = spawn.get("slot_id")
            if sid is not None:
                # `__spawn__` is intentionally NOT a real source tag in
                # SOURCE_TAGS_VOCAB, so no producer can subscribe to it.
                self._slot_source_tags[sid] = "__spawn__"
                self._slot_raw_input_ids[sid] = spawn.get("spawn_tag")
        return stats

    def slot_state(self) -> dict:
        """Per-slot diagnostic snapshot, augmented with source tags."""
        base = self._trough.slot_state()
        base["source_tags"] = list(self._slot_source_tags)
        base["raw_input_ids"] = list(self._slot_raw_input_ids)
        return base
