"""TroughAttention — stateful K/V container per tier boundary.

Combines what `SubstratePool` did (hold broadcasts; hand them out) and what
`Channel` did (cross-attention readout with null gate) into one module.

Per the architectural spec (`docs/consumption_transformers.md`):

  - V_store is the K/V matrix of broadcasts at this tier boundary.
  - `deposit(broadcasts)` writes the broadcasts' channel embeddings into
    free slots of V_store (no projection at write — projection happens at
    `attend()` time, matching the existing Channel semantics).
  - `attend(query)` runs standard cross-attention with a null gate. It
    accumulates per-slot attention into the `cumulative_attention` buffer
    so Mission 02 can apply a death threshold.

Out of scope for Mission 01 (handled by phase 2):
  - Slot death and respawn (`step_lifecycle`, `spawn_into_dead_slot` are
    stubs).
  - Per-head niche specialization tracking (`per_head_attention` is
    populated, but there's no "head specialization" learning logic yet).
  - Decomposer attention bias (`external_bias` is accepted in the
    signature but added straight to the scores; no learned routing).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ecology.slot_lifecycle import (
    kill_dead_slots,
    mark_underperforming,
    niche_aware_spawn,
)


@dataclass
class TroughAttendOutput:
    output: torch.Tensor              # [out_seq_len, hidden]
    per_slot_attention: torch.Tensor  # [n_slots]
    per_head_attention: torch.Tensor  # [n_heads, n_slots]
    null_prob: float
    selected_slot_ids: list[int]
    rejected_slot_ids: list[int]


def _diet_tag_to_id(tag_str: str, n_diet_tags: int) -> int:
    """Hash a diet-tag string to a small int id in [0, n_diet_tags).

    Deterministic: same string → same id across runs/processes. Empty string
    or "" → -1 (universal slot, never filtered out by diet mask). Used by
    deposit() to label slots and by attend() to match consumer's allowed set.
    """
    if not tag_str:
        return -1
    import hashlib as _hashlib
    h = int(_hashlib.sha256(tag_str.encode()).hexdigest(), 16)
    return h % n_diet_tags


class TroughAttention(nn.Module):
    """Stateful K/V container at one tier boundary.

    The K/V "matrix" is `V_store: [n_slots, hidden]`, with an `alive`
    boolean mask telling which slots are populated. Deposited broadcasts
    write into free (alive==False) slots; `attend()` only reads from
    alive slots plus a learned null row.

    The internal projections (`W_Q`, `W_K`, `W_V`) match the existing
    `Channel` module so weight transfer between Channel and trough is
    straightforward (Mission 02+ will start consuming this directly).
    """

    def __init__(
        self,
        hidden_size: int,
        n_slots: int,
        n_heads: int = 8,
        out_seq_len: int = 8,
        target_norm: float = 1.0,
        seed: int | None = None,
        n_diet_tags: int = 16,
        use_E_in: bool = False,
        gated_residual: bool = False,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})"
            )
        self.hidden_size = hidden_size
        self.n_slots = n_slots
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.out_seq_len = out_seq_len
        self.target_norm = target_norm
        # Phase 1 of #8 architectural fix: per-tier-pair E_in matrix that
        # projects deposited broadcasts into the destination tier's expected
        # latent space. Identity-initialized, single linear matrix (per
        # second-opinion convergent advice — not an MLP, since both endpoints
        # share the Qwen3-4B hidden distribution).
        self.use_E_in = use_E_in
        # Gated residual (per GPT-5.2 landmine): trough's contribution to
        # consumer prefix enters with α=0 + zero-init readout, so day-0 output
        # is identical to no-trough baseline; SFT opens the gate.
        self.gated_residual = gated_residual
        self.n_diet_tags = n_diet_tags

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(int(seed))
        else:
            gen.seed()

        # Q/K/V projections — same scaled-small init philosophy as `Channel`
        # so attention scores at init are near-zero (uniform softmax) and
        # the null slot lightly wins. W_V is scaled identity so values pass
        # through without distortion at init.
        self.W_Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            std_qk = 1.0 / hidden_size ** 0.5
            self.W_Q.weight.copy_(
                torch.empty_like(self.W_Q.weight).normal_(0.0, std_qk, generator=gen)
            )
            self.W_K.weight.copy_(
                torch.empty_like(self.W_K.weight).normal_(0.0, std_qk, generator=gen)
            )
            self.W_V.weight.copy_(torch.eye(hidden_size) * 0.5)

        # Null slot — learned K/V row representing "I attend to nothing".
        # Kept as a separate parameter (NOT inside V_store) so slot
        # bookkeeping stays clean.
        self.K_null = nn.Parameter(
            torch.empty(hidden_size).normal_(0.0, 0.02, generator=gen)
        )
        self.V_null = nn.Parameter(
            torch.empty(hidden_size).normal_(0.0, 0.02, generator=gen)
        )
        self.null_bias = nn.Parameter(torch.zeros(1))

        # Output projection: small MLP from attended hidden → out_seq_len
        # distinct vectors (matches Channel's proj_in/proj_out).
        inner = hidden_size * 2
        self.proj_in = nn.Linear(hidden_size, inner, bias=False)
        self.proj_out = nn.Linear(inner, hidden_size * out_seq_len, bias=False)
        with torch.no_grad():
            if gated_residual:
                # Zero-init readout per GPT-5.2 landmine: keeps trough output
                # at zero on day 0 so consumer baseline behavior is unchanged.
                self.proj_in.weight.zero_()
                self.proj_out.weight.zero_()
            else:
                self.proj_in.weight.copy_(
                    torch.empty_like(self.proj_in.weight).normal_(0.0, 1.0 / hidden_size ** 0.5, generator=gen)
                )
                self.proj_out.weight.copy_(
                    torch.empty_like(self.proj_out.weight).normal_(0.0, 1.0 / inner ** 0.5, generator=gen)
                )

        # Gated residual scale α — initialized to 0 so trough contribution
        # starts disabled. Open via SFT gradient.
        if gated_residual:
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("alpha", None)

        # E_in: per-tier-pair latent translator. Single matrix, identity init.
        # Applied at deposit time: V_store[slot] = E_in(emb).
        if use_E_in:
            self.E_in = nn.Linear(hidden_size, hidden_size, bias=False)
            with torch.no_grad():
                self.E_in.weight.copy_(torch.eye(hidden_size))
        else:
            self.register_module("E_in", None)

        # K/V slot store. Buffers (not parameters) — broadcasts come from
        # outside the module, are not gradient-trained.
        self.register_buffer("V_store", torch.zeros(n_slots, hidden_size))
        self.register_buffer("alive", torch.zeros(n_slots, dtype=torch.bool))
        # Per-slot diet tag id. -1 = no tag (universal). Used by attend()
        # to mask out slots whose tag isn't in the consumer's allowed set.
        # Phase 1 of #8 fix.
        self.register_buffer(
            "slot_tag_id", torch.full((n_slots,), -1, dtype=torch.long)
        )

        # Per-slot bookkeeping for ecological lifecycle (Mission 02).
        # `cumulative_attention` is an EMA (not a raw sum, despite the
        # legacy name kept for state_dict compat) — see attend() for the
        # update rule, and step_lifecycle() for the death threshold.
        self.register_buffer("cumulative_attention", torch.zeros(n_slots))
        self.register_buffer("age", torch.zeros(n_slots, dtype=torch.long))
        self.register_buffer(
            "under_threshold_ticks", torch.zeros(n_slots, dtype=torch.long)
        )
        self.register_buffer("dead", torch.zeros(n_slots, dtype=torch.bool))

        # Mission 03 (phase2-C): per-(head, slot) attention EMA. Tracks
        # which heads "specialize" in which slots over time, so we can
        # replace hardcoded diet_tags routing with learned multi-head
        # niche assignment. Updated inside attend() with the same
        # alpha_decay coefficient as cumulative_attention (head and slot
        # niches age on the same time-scale).
        self.register_buffer(
            "head_specialization", torch.zeros(n_heads, n_slots)
        )

        # Mission 04 (phase2-A): ring buffer of recent (Q, max_attention)
        # pairs used by `find_underserved_query`. When a slot dies and we
        # spawn a replacement, we look back over this history for the
        # consumer query that received the lowest max attention across
        # slots — that's the niche the trough is currently failing to
        # cover, so we initialize the new slot to satisfy it.
        self.q_history_size: int = 64
        self.register_buffer(
            "recent_queries", torch.zeros(self.q_history_size, hidden_size)
        )
        # Initialize to +inf so unwritten ring slots never win the
        # argmin in `find_underserved_query` — only actually-logged
        # queries are eligible to be selected as the "most underserved".
        self.register_buffer(
            "recent_max_attention",
            torch.full((self.q_history_size,), float("inf")),
        )
        # Plain int — not a buffer (not part of saved state, just a
        # wrap-around index into the ring above). Re-derivable from any
        # restart since the history itself is approximate.
        self._q_idx: int = 0

        # Ecological hyperparameters (Mission 02). Plain attributes — not
        # gradient-trained. α=0.9 means a slot's attention EMA loses ~10%
        # of its weight per tick if unattended; with epsilon_death=0.01
        # and n_patience=5, a slot getting zero attention from a uniform
        # init (1/N_alive) of, say, 1/4 = 0.25 will fall below 0.01 in
        # ~ceil(log(0.01/0.25) / log(0.9)) = ~31 ticks, then needs 5 more
        # consecutive sub-ε ticks to die. For broadcasts that arrive with
        # cumulative_attention=0 this is faster (death after n_patience).
        self.alpha_decay: float = 0.9
        self.epsilon_death: float = 0.01
        self.n_patience: int = 5

        # Mission 04 (phase2-A): population strategy controls whether
        # `step_lifecycle` immediately respawns into killed slots. The
        # default `"fixed"` matches existing pool semantics (population
        # stays at `n_slots`). Future strategies (carrying-capacity,
        # dynamic) can opt out of automatic respawn by setting this to
        # any value other than `"fixed"`.
        self.population_strategy: str = "fixed"
        # Noise stddev added to the pseudo-inverse-derived V vector when
        # spawning. Small enough to keep niche-locking strong, nonzero
        # so successive spawns don't collapse onto the same vector.
        self.spawn_noise_scale: float = 0.02

        # Provenance — the broadcast id stored in each slot. None == empty.
        # Python list because the SubstratePool shim returns Broadcast objs.
        self.broadcast_ids: list[str | None] = [None] * n_slots

        # Mission 05 (phase2-D): decomposer-feedback bias staged for the
        # next attend() call. Set via `set_pending_bias`; consumed (and
        # cleared) on the next attend() that does not receive an explicit
        # `external_bias` kwarg. Python attribute (not buffer) — the bias
        # is transient between ticks and is not part of the saved state.
        self._pending_bias: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # state inspection
    # ------------------------------------------------------------------

    def slot_state(self) -> dict:
        """Snapshot of slot-level state, for diagnostics.

        Returns a dict with python-native (or detached) values so callers
        can JSON-dump it without thinking about gradients.
        """
        return {
            "n_slots": self.n_slots,
            "n_alive": int(self.alive.sum().item()),
            "alive": self.alive.detach().cpu().tolist(),
            "broadcast_ids": list(self.broadcast_ids),
            "cumulative_attention": self.cumulative_attention.detach().cpu().tolist(),
            "age": self.age.detach().cpu().tolist(),
            "under_threshold_ticks": self.under_threshold_ticks.detach().cpu().tolist(),
            "dead": self.dead.detach().cpu().tolist(),
        }

    def slot_specialization(self) -> dict:
        """Per slot: which head specializes in it.

        Reads the `head_specialization` buffer (an EMA of per-(head, slot)
        attention) and returns:
          - `head_per_slot`: argmax over heads, length `n_slots`. Slots
            that have never received attention all return 0 (argmax of
            all-zero rows defaults to head 0). Callers that care should
            cross-reference `alive`.
          - `specialization_scores`: the raw EMA tensor as a nested
            list `[n_heads][n_slots]`.

        Mission 03 acceptance: after enough ticks of mock SFT, this
        argmax should not be constant across alive slots — heads have
        partitioned the niche space.
        """
        with torch.no_grad():
            dominant = self.head_specialization.argmax(dim=0)  # [n_slots]
        return {
            "head_per_slot": dominant.detach().cpu().tolist(),
            "specialization_scores": self.head_specialization.detach().cpu().tolist(),
        }

    # ------------------------------------------------------------------
    # K/V membership
    # ------------------------------------------------------------------

    def deposit(self, broadcasts: Sequence) -> list[int]:
        """Place broadcasts into free slots. Returns slot ids used.

        For Mission 01 we use simple fixed-size semantics: if there are not
        enough free slots, raises `RuntimeError`. Mission 02 / 04 add
        lifecycle (death, respawn, eviction) on top.

        The broadcasts are expected to expose `.id: str` and
        `.channel_embedding: list[float]` (both are present on the
        existing `Broadcast` pydantic model).
        """
        if len(broadcasts) == 0:
            return []

        free = (~self.alive).nonzero(as_tuple=False).flatten().tolist()
        if len(free) < len(broadcasts):
            raise RuntimeError(
                f"TroughAttention: tried to deposit {len(broadcasts)} broadcasts but only "
                f"{len(free)} free slots out of {self.n_slots}. "
                f"Slot eviction/death is added in phase2-B/A."
            )

        device = self.V_store.device
        dtype = self.V_store.dtype

        slot_ids = free[: len(broadcasts)]
        for slot_id, br in zip(slot_ids, broadcasts):
            emb = torch.as_tensor(br.channel_embedding, dtype=dtype, device=device)
            if emb.shape != (self.hidden_size,):
                raise ValueError(
                    f"broadcast {getattr(br, 'id', '?')} has channel_embedding of shape "
                    f"{tuple(emb.shape)}, expected ({self.hidden_size},)"
                )
            # Phase 1 of #8 fix: store raw embedding in V_store (a non-grad
            # buffer). E_in is applied at attend() time so gradient flows
            # cleanly through E_in's parameters once per attend, not once
            # per deposit. (Earlier draft applied E_in here; that broke
            # autograd because multiple consumers in the same step would
            # try to backward through the shared deposit-time computation.)
            self.V_store[slot_id] = emb.detach()
            self.alive[slot_id] = True
            self.cumulative_attention[slot_id] = 0.0
            self.age[slot_id] = 0
            self.under_threshold_ticks[slot_id] = 0
            self.dead[slot_id] = False
            self.broadcast_ids[slot_id] = br.id
            # Set diet tag from the broadcast's diet_tags list (use first
            # tag; multi-tag slots are a v2 concern). Tag is stored as a
            # hashed int id; runner-level mapping diet_tag_str → id lives in
            # caller code that prepares allowed_tag_ids for attend().
            tag_str = ""
            tags = getattr(br, "diet_tags", None)
            if tags:
                tag_str = tags[0]
            tag_id = _diet_tag_to_id(tag_str, self.n_diet_tags)
            self.slot_tag_id[slot_id] = tag_id

        return slot_ids

    def set_pending_bias(self, bias: torch.Tensor | None) -> None:
        """Stage a per-slot bias to be consumed by the NEXT `attend()` call.

        Mission 05 (phase2-D): the decomposer produces a `[n_slots]` bias
        from (apex_judgment, slot_lineage) and stages it on the predator's
        trough. The next time `attend()` runs without an explicit
        `external_bias` kwarg, this staged bias is used and then cleared
        (consume-once semantics).

        Pass `None` to clear without consuming.
        """
        if bias is None:
            self._pending_bias = None
            return
        if bias.shape != (self.n_slots,):
            raise ValueError(
                f"set_pending_bias: expected shape ({self.n_slots},), got {tuple(bias.shape)}"
            )
        self._pending_bias = bias.detach()

    def evict(self, slot_ids: Sequence[int]) -> None:
        """Mark the given slots as free (used by SubstratePool shim's
        `claim` semantics — once a hunter consumes an item, that slot is
        recycled). NOT the same as ecological death (phase2-B); this is
        bookkeeping for non-decaying claim semantics.
        """
        for sid in slot_ids:
            if sid < 0 or sid >= self.n_slots:
                raise IndexError(sid)
            self.V_store[sid].zero_()
            self.alive[sid] = False
            self.cumulative_attention[sid] = 0.0
            self.age[sid] = 0
            self.under_threshold_ticks[sid] = 0
            self.dead[sid] = False
            self.broadcast_ids[sid] = None
            # Mission 03: clear the per-(head, slot) niche EMA so a
            # respawned slot starts from a clean specialization signal.
            self.head_specialization[:, sid] = 0.0

    # ------------------------------------------------------------------
    # attention readout
    # ------------------------------------------------------------------

    def attend(
        self,
        query: torch.Tensor,
        tau: float = 1.0,
        external_bias: torch.Tensor | None = None,
        allowed_tag_strs: Sequence[str] | None = None,
    ) -> TroughAttendOutput:
        """Standard cross-attention with a null gate and optional diet mask.

        - `query`: shape `[hidden]` or `[Q, hidden]` (collapsed to one
          query vector by mean-pool — multi-query consumers are a future
          concern).
        - K/V come from `V_store` filtered to `alive==True` slots, plus
          a learned null row at the end.
        - Softmax over `Q·K / sqrt(head_dim) / tau`, with optional
          `external_bias` of shape `[n_slots]` added to scores for alive
          slots (Mission 05 hooks in here for decomposer feedback).
        - `allowed_tag_strs` (#8 phase-1 fix): if provided, slots whose
          stored `slot_tag_id` does NOT match any of the allowed tags
          (after hashing through `_diet_tag_to_id`) get a -inf bias added
          to their attention score, so they're zero-weighted in softmax.
          Slots with tag_id=-1 (universal) match any allowed set. Critical
          landmine fix per both reviewers: `cumulative_attention` is only
          updated for slots that were in the consumer's allowed set —
          incompatible consumers don't penalize a slot's fitness.
        - Output is the MLP-projected attended context, expanded to
          `[out_seq_len, hidden]`, norm-matched to `target_norm`.
        - `cumulative_attention` is incremented by the per-slot attention
          weight (averaged across heads). Mission 02 reads this buffer.

        Returns `TroughAttendOutput` with diagnostics.
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        elif query.dim() == 2:
            if query.shape[0] != 1:
                query = query.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"query must be 1D or 2D, got shape {tuple(query.shape)}")

        device = query.device
        dtype = query.dtype
        H = self.hidden_size

        # Project query.
        Q = self.W_Q(query)  # [1, H]

        # Gather alive slots.
        alive_mask = self.alive
        alive_idx = alive_mask.nonzero(as_tuple=False).flatten()  # [N_alive] (long)
        N_alive = int(alive_idx.shape[0])

        if N_alive > 0:
            V_alive_raw = self.V_store.index_select(0, alive_idx).to(dtype=dtype, device=device)
            # Phase 1 of #8 fix: project through E_in into the destination
            # tier's latent space at attend time (was deposit-time, but that
            # broke autograd when multiple consumers share a trough). E_in
            # is identity at init so this is a no-op; SFT learns it.
            if self.E_in is not None:
                V_alive_raw = self.E_in(V_alive_raw)
            K_alive = self.W_K(V_alive_raw)
            V_alive = self.W_V(V_alive_raw)
        else:
            K_alive = torch.zeros(0, H, device=device, dtype=dtype)
            V_alive = torch.zeros(0, H, device=device, dtype=dtype)

        K_null = self.K_null.to(device=device, dtype=dtype).unsqueeze(0)  # [1, H]
        V_null = self.V_null.to(device=device, dtype=dtype).unsqueeze(0)

        K_all = torch.cat([K_alive, K_null], dim=0)  # [N_alive+1, H]
        V_all = torch.cat([V_alive, V_null], dim=0)

        # Multi-head split.
        Q_h = self._split_heads(Q)            # [n_heads, 1, head_dim]
        K_h = self._split_heads(K_all)        # [n_heads, N_alive+1, head_dim]
        V_h = self._split_heads(V_all)        # [n_heads, N_alive+1, head_dim]

        scores = torch.einsum("hqd,hkd->hqk", Q_h, K_h) / math.sqrt(self.head_dim)
        # Apply temperature on the softmax (mirrors the spec's τ).
        scores = scores / max(float(tau), 1e-6)

        # Null bias on the last index.
        scores[..., -1] = scores[..., -1] + self.null_bias

        # External bias (e.g. decomposer feedback in Mission 05). Shape
        # `[n_slots]`; only the alive entries are read. If the caller
        # didn't pass one explicitly, fall back to the decomposer bias
        # staged via `set_pending_bias`. Either way the staged bias is
        # cleared at the end of this call (consume-once).
        if external_bias is None and self._pending_bias is not None:
            external_bias = self._pending_bias
        self._pending_bias = None
        if external_bias is not None and N_alive > 0:
            if external_bias.shape != (self.n_slots,):
                raise ValueError(
                    f"external_bias must be shape ({self.n_slots},), got {tuple(external_bias.shape)}"
                )
            bias_alive = external_bias.to(device=device, dtype=dtype).index_select(0, alive_idx)
            # broadcast over heads and the singleton query position.
            scores[..., :N_alive] = scores[..., :N_alive] + bias_alive

        # Phase 1 of #8 fix: diet mask. If the consumer specified an allowed
        # set of diet tags, slots whose tag is NOT in that set get -inf
        # added to their score (zero softmax weight). Slots tagged -1
        # (universal) match any allowed set. Tracked separately so the
        # cumulative_attention update below can skip masked-out slots.
        diet_mask_alive: torch.Tensor | None = None  # bool [N_alive]
        if allowed_tag_strs is not None and N_alive > 0:
            allowed_ids = {
                _diet_tag_to_id(t, self.n_diet_tags)
                for t in allowed_tag_strs if t
            }
            slot_tags_alive = self.slot_tag_id.index_select(0, alive_idx)  # [N_alive]
            # universal slots (-1) always match; otherwise tag must be in allowed set.
            allowed_tensor = torch.tensor(
                list(allowed_ids), device=device, dtype=slot_tags_alive.dtype,
            ) if allowed_ids else torch.zeros(0, device=device, dtype=slot_tags_alive.dtype)
            if allowed_tensor.numel() == 0:
                # Empty allowed set means no slots match → all real slots blocked.
                diet_mask_alive = torch.zeros(N_alive, dtype=torch.bool, device=device)
            else:
                in_allowed = (slot_tags_alive.unsqueeze(-1) == allowed_tensor.unsqueeze(0)).any(dim=-1)
                universal = slot_tags_alive == -1
                diet_mask_alive = in_allowed | universal  # True = compatible (keep)
            # Inject -inf on the score for incompatible slots (still alive but wrong-diet).
            block_alive = ~diet_mask_alive
            if block_alive.any():
                neg_inf = torch.full(
                    (N_alive,), float("-inf"), device=device, dtype=dtype,
                )
                # Add 0 where compatible, -inf where blocked. Broadcast across heads/query.
                bias = torch.where(block_alive, neg_inf, torch.zeros_like(neg_inf))
                scores[..., :N_alive] = scores[..., :N_alive] + bias

        weights = F.softmax(scores, dim=-1)  # [n_heads, 1, N_alive+1]
        attended = torch.einsum("hqk,hkd->hqd", weights, V_h)  # [n_heads, 1, head_dim]
        merged = self._merge_heads(attended)  # [1, hidden]

        # Output MLP.
        z = F.gelu(self.proj_in(merged))                 # [1, inner]
        out = self.proj_out(z)                           # [1, hidden*out_seq_len]
        output = out.view(self.out_seq_len, self.hidden_size)

        if self.gated_residual:
            # Phase 1 of #8 fix: identity-baseline + zero-init MLP residual.
            # Day-0 output is just the attended V repeated across out_seq_len
            # positions, unit-renormed. proj_in/proj_out start at zero so
            # the MLP contributes 0 at init — SFT grows it from gradient as
            # a residual on top of the identity baseline. This avoids the
            # "zeros in the prefix = OOD chimera" failure mode while still
            # preserving GPT-5.2's safety property (no untrained MLP can
            # produce OOD directions on day 0).
            baseline = merged.expand(self.out_seq_len, self.hidden_size)
            cur_norm = baseline.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            baseline = baseline / cur_norm * self.target_norm
            output = baseline + output  # residual; output is 0 at init
        else:
            # Norm match each output position to target_norm so the spliced
            # vectors live in the same range Qwen's input embeddings do.
            cur_norm = output.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            output = output / cur_norm * self.target_norm

        # ------------------------------------------------------------------
        # Bookkeeping + diagnostics.
        # ------------------------------------------------------------------
        # avg_w: per-head-averaged attention over (alive_slots + null), shape [N_alive+1]
        avg_w = weights.mean(dim=0).squeeze(0)
        null_prob = float(avg_w[-1].item())

        # Per-slot attention vector indexed by *original* slot id.
        per_slot = torch.zeros(self.n_slots, device=device, dtype=dtype)
        per_head = torch.zeros(self.n_heads, self.n_slots, device=device, dtype=dtype)
        if N_alive > 0:
            per_slot.index_copy_(0, alive_idx, avg_w[:-1].detach())
            # per_head: weights is [n_heads, 1, N_alive+1] → drop singleton query, drop null, scatter.
            per_head_alive = weights.detach().squeeze(1)[:, :-1]  # [n_heads, N_alive]
            for h in range(self.n_heads):
                per_head[h].index_copy_(0, alive_idx, per_head_alive[h])

        # Update cumulative_attention buffer (detached — buffers don't
        # carry gradient). Mission 02: this is now an EMA, not a raw
        # running sum. For each alive slot we mix in this tick's
        # attention with weight (1 - α); the smoothed value is what
        # `step_lifecycle` later compares against epsilon_death. Dead
        # slots' EMA is forced to zero (they shouldn't be receiving any
        # attention; the alive_mask gating in attend() already prevents
        # it, but we belt-and-brace here so a stale buffer can't keep a
        # dead slot from being killed cleanly).
        with torch.no_grad():
            per_slot_buf = per_slot.detach().to(self.cumulative_attention.dtype)
            alive_f = self.alive.to(self.cumulative_attention.dtype)
            alpha = float(self.alpha_decay)
            # #8 phase-1 landmine fix (per both reviewers): if a diet mask
            # was applied, slots NOT in the allowed set should not have
            # their cumulative_attention EMA updated this tick — incompatible
            # consumers shouldn't penalize a slot's fitness for being
            # ignored when they weren't allowed to look at it. Build a
            # full-size eligibility mask aligned to slot ids.
            update_mask = alive_f
            if allowed_tag_strs is not None and N_alive > 0 and diet_mask_alive is not None:
                eligible = torch.zeros_like(alive_f)
                eligible.index_copy_(
                    0, alive_idx, diet_mask_alive.to(alive_f.dtype)
                )
                update_mask = update_mask * eligible
            self.cumulative_attention.mul_(alpha)
            self.cumulative_attention.add_((1.0 - alpha) * per_slot_buf * update_mask)
            self.cumulative_attention.mul_(alive_f)  # zero out dead slots
            self.age.add_(self.alive.long())

            # Mission 03: per-(head, slot) EMA. Mirror the cumulative_attention
            # update rule, broadcast over heads. `per_head` is [n_heads, n_slots]
            # already zero-padded for dead slots.
            per_head_buf = per_head.detach().to(self.head_specialization.dtype)
            self.head_specialization.mul_(alpha)
            # Multiply by alive mask (broadcast across head axis) so dead
            # slots' niche signal decays to 0 just like cumulative_attention.
            alive_row = alive_f.unsqueeze(0)  # [1, n_slots]
            self.head_specialization.add_(
                (1.0 - alpha) * per_head_buf * alive_row
            )
            self.head_specialization.mul_(alive_row)

            # Mission 04 (phase2-A): log the unprojected query and the
            # max per-slot attention so `find_underserved_query` can
            # spot consumer Qs the trough is failing to cover.
            # `query` here is `[1, H]` (post-mean-pool); flatten to `[H]`.
            # We store the *raw* query (not Q = W_Q @ query) because the
            # spawn step pseudo-inverts W_K and writes into V_store —
            # that's the unprojected V-space, so we want the unprojected
            # consumer-side vector for similarity.
            if N_alive > 0:
                q_flat = query.detach().squeeze(0).to(self.recent_queries.dtype)
                max_att = float(per_slot.detach().max().item())
            else:
                q_flat = query.detach().squeeze(0).to(self.recent_queries.dtype)
                max_att = 0.0
            self.recent_queries[self._q_idx] = q_flat
            self.recent_max_attention[self._q_idx] = max_att
            self._q_idx = (self._q_idx + 1) % self.q_history_size

        # Selected = alive slots ranked by attention. Rejected = none in
        # Mission 01 (no capacity bound on the trough side; capacity is
        # enforced upstream by the SubstratePool shim's `claim` API).
        if N_alive > 0:
            sorted_w, sorted_local = torch.sort(avg_w[:-1].detach(), descending=True)
            selected_slot_ids = alive_idx[sorted_local].tolist()
        else:
            selected_slot_ids = []
        rejected_slot_ids: list[int] = []

        return TroughAttendOutput(
            output=output,
            per_slot_attention=per_slot.detach(),
            per_head_attention=per_head.detach(),
            null_prob=null_prob,
            selected_slot_ids=selected_slot_ids,
            rejected_slot_ids=rejected_slot_ids,
        )

    # ------------------------------------------------------------------
    # lifecycle stubs (filled by phase2-B / phase2-A)
    # ------------------------------------------------------------------

    def step_lifecycle(self) -> dict:
        """Apply ecological selection: kill slots that have starved.

        Pipeline (Mission 02):
          1. `mark_underperforming` — increment per-slot counters for
             alive slots whose EMA-smoothed cumulative_attention has
             fallen below `epsilon_death`. Reset to 0 for any slot above.
          2. `kill_dead_slots` — slots whose counter reached
             `n_patience` consecutive sub-ε ticks are killed.
          3. Commit the new `alive` mask, mark `dead`, and zero out the
             killed slots so `spawn_into_dead_slot` (Mission 04) starts
             from a clean state.

        Returns a stats dict for runner-side logging:
          {"n_killed": int, "n_alive": int, "killed_ids": [int],
           "deaths": int, "ages": [int]}

        `"deaths"` and `"ages"` are kept as legacy aliases so existing
        callers that introspected the stub keep working.
        """
        with torch.no_grad():
            new_under = mark_underperforming(
                self.cumulative_attention,
                self.alive,
                self.epsilon_death,
                self.under_threshold_ticks,
            )
            self.under_threshold_ticks.copy_(new_under)

            new_alive, killed = kill_dead_slots(
                self.under_threshold_ticks, self.alive, self.n_patience,
            )
            n_killed = int(killed.sum().item())
            killed_ids = killed.nonzero(as_tuple=False).flatten().tolist()

            if n_killed > 0:
                self.alive.copy_(new_alive)
                self.dead.copy_(self.dead | killed)
                # Clear killed slots so reallocation (Mission 04) starts
                # from a clean state. V_store, EMA, counters, age all
                # reset; broadcast_ids cleared on the python-side list.
                self.V_store[killed] = 0
                self.cumulative_attention[killed] = 0.0
                self.under_threshold_ticks[killed] = 0
                self.age[killed] = 0
                # Mission 03: also clear per-(head, slot) niche EMA on death.
                self.head_specialization[:, killed] = 0.0
                for idx in killed_ids:
                    self.broadcast_ids[idx] = None

        # Mission 04 (phase2-A): immediately spawn into killed slots so
        # the population stays at fixed size. Opt-out by setting
        # `population_strategy` to anything other than "fixed".
        spawn_stats: list[dict] = []
        if self.population_strategy == "fixed":
            for sid in killed_ids:
                spawn_stats.append(self.spawn_into_dead_slot(sid))

        n_alive = int(self.alive.sum().item())
        return {
            "n_killed": n_killed,
            "n_alive": n_alive,
            "killed_ids": killed_ids,
            "spawned": spawn_stats,
            "deaths": n_killed,                         # legacy alias
            "ages": self.age.detach().cpu().tolist(),  # legacy field
        }

    def find_underserved_query(self) -> torch.Tensor:
        """Return the recent consumer query that had the lowest max attention.

        Mission 04 (phase2-A): the ring buffer logged inside `attend()`
        records per-call (Q, max_per_slot_attention). The Q with the
        smallest max attention is the consumer query the trough is
        currently failing to cover — that's the niche the next spawn
        should occupy.

        If no queries have been logged yet (all-zeros buffer), returns a
        zero vector of shape `[hidden]`. Callers that hit that branch
        should still get a sensible (norm-rescaled to target_norm) V.
        """
        with torch.no_grad():
            idx = int(self.recent_max_attention.argmin().item())
            return self.recent_queries[idx].detach().clone()

    def spawn_into_dead_slot(self, slot_id: int) -> dict:
        """Reallocate a dead slot via niche-aware spawning (Mission 04).

        Picks the most underserved recent query, pseudo-inverts W_K to
        land near it in K-space, adds small noise, rescales to
        `target_norm`, and writes the result into `V_store[slot_id]`.
        Resets per-slot bookkeeping so the new slot starts fresh.

        Pre-condition: `alive[slot_id]` must be False (raises otherwise).

        Returns a stats dict with the slot id, the cosine similarity
        between the new K vector and the underserved Q (used as a
        diagnostic for niche-locking quality), and the underserved Q's
        index in the ring buffer.
        """
        if bool(self.alive[slot_id].item()):
            raise AssertionError(
                f"spawn_into_dead_slot: slot {slot_id} is alive — refusing to overwrite"
            )

        underserved_q = self.find_underserved_query()
        device = self.V_store.device
        dtype = self.V_store.dtype
        underserved_q = underserved_q.to(device=device, dtype=dtype)

        with torch.no_grad():
            new_v = niche_aware_spawn(
                underserved_q=underserved_q,
                W_K=self.W_K.weight.detach(),
                target_norm=self.target_norm,
                noise_scale=self.spawn_noise_scale,
            ).to(device=device, dtype=dtype)

            # Diagnostic: cosine(new_K, underserved_q). Useful for both
            # tests and the spawn-quality smoke (mission acceptance #4).
            new_k = self.W_K(new_v.unsqueeze(0)).squeeze(0)
            denom = (new_k.norm() * underserved_q.norm()).clamp_min(1e-8)
            cos_kq = float((new_k @ underserved_q / denom).item())

            self.V_store[slot_id] = new_v
            self.alive[slot_id] = True
            self.dead[slot_id] = False
            self.cumulative_attention[slot_id] = 0.0
            self.age[slot_id] = 0
            self.under_threshold_ticks[slot_id] = 0
            # Clear per-(head, slot) niche EMA so the new slot's
            # specialization is learned from scratch.
            self.head_specialization[:, slot_id] = 0.0

        # Provenance label: lets downstream logging tell spawned slots
        # apart from deposited broadcasts. The age-sum suffix makes
        # successive spawns into the same id distinguishable.
        spawn_tag = f"spawn:{slot_id}:{int(self.age.sum().item())}"
        self.broadcast_ids[slot_id] = spawn_tag

        return {
            "slot_id": slot_id,
            "spawned": True,
            "cosine_new_k_underserved_q": cos_kq,
            "spawn_tag": spawn_tag,
            "underserved_q_norm": float(underserved_q.norm().item()),
        }

    # ------------------------------------------------------------------
    # head helpers
    # ------------------------------------------------------------------

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [N, hidden] -> [n_heads, N, head_dim]
        N, H = x.shape
        return x.view(N, self.n_heads, self.head_dim).transpose(0, 1)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [n_heads, N, head_dim] -> [N, hidden]
        return x.transpose(0, 1).contiguous().view(x.shape[1], self.hidden_size)
