"""SubstratePool — thin shim over `TroughAttention` (Mission 01 refactor).

History: this module used to be a SQLite-backed broadcast pool with
claim semantics + a `rotted` bookkeeping column. The architectural
spec at `docs/consumption_transformers.md` collapses that abstraction
into the cross-attention K/V matrix at each tier boundary
(`TroughAttention`).

Mission 01 keeps `SubstratePool`'s external API intact — herbivore and
predator agents talk to it exactly as before — but the implementation is
now an in-memory cache of broadcasts plus a lazily-created
`TroughAttention` per (tier, agent_kind) tuple. SQL is gone. The
`rotted` flag is in-memory state, not a DB column.

The judgments / decomposer_records / agents / ticks tables also become
in-memory lists — they were used as a structured logbook for diagnostic
queries and the runner only `add_*`s into them. Anything that wants to
read them later can grab the lists directly off the pool.

When phase 2 starts, the shim is what gets thinner — `claim` semantics
will move into the trough's own lifecycle (`step_lifecycle` /
`spawn_into_dead_slot`) and this module will eventually disappear.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .trough_attention import TroughAttention
from .types import (
    AgentState,
    Broadcast,
    DecomposerRecord,
    PredatorJudgment,
)


@dataclass
class _BroadcastEntry:
    """In-memory bookkeeping for a single broadcast.

    Tracks what the old SQL columns tracked: who claimed it, whether it
    rotted, the trough slot id (so claim/rot can call `evict`).
    """
    broadcast: Broadcast
    claimed_by: str | None = None
    claimed_at: str | None = None
    rotted: bool = False
    # (tier, kind_key) trough this broadcast lives in, plus the slot id.
    # None if no trough was created (defensive — shouldn't happen).
    trough_key: tuple[str, str] | None = None
    slot_id: int | None = None


@dataclass
class _TickRecord:
    tick_id: int
    n_broadcasts: int
    n_eaten: int
    n_rotted: int
    n_judgments: int
    n_abstentions: int = 0
    notes: dict = field(default_factory=dict)


def _kind_key(b: Broadcast) -> str:
    """Sub-bucket within a tier — herbivore / predator agents query by
    diet-tag (which maps 1:1 to source agent kind), so we shard the
    troughs along the same axis. Falls back to `tier` if `agent_kind` is
    missing.
    """
    return b.agent_kind or b.tier


# Default trough sizing for the shim. Hidden_size is set lazily from the
# first deposit; n_slots is large enough to hold a few ticks of broadcasts
# (existing config has retrieval_k=40, runner produces a handful per tick).
_DEFAULT_N_SLOTS = 256


class SubstratePool:
    """Compatibility shim — in-memory broadcast cache + per-(tier,kind) troughs.

    Exposes the same methods the old SQLite pool did. Internally:

      - `_entries` maps broadcast id → `_BroadcastEntry` (status + provenance)
      - `_troughs` maps (tier, kind_key) → TroughAttention
      - judgments / decomposer / agents / ticks are in-memory lists
        (used as logbooks; nothing in the runtime reads them back).
    """

    def __init__(
        self,
        db_path: str | None = ":memory:",
        n_slots_per_kind: int = _DEFAULT_N_SLOTS,
        trough_seed: int | None = None,
    ):
        # `db_path` is accepted for backward-compat with config; ignored.
        self._db_path = db_path
        self._n_slots_per_kind = n_slots_per_kind
        self._trough_seed = trough_seed
        self._lock = threading.RLock()

        self._entries: dict[str, _BroadcastEntry] = {}
        self._troughs: dict[tuple[str, str], TroughAttention] = {}

        # Logbook lists — preserved for tests / diagnostics, never read
        # back by the runtime.
        self._judgments: list[PredatorJudgment] = []
        self._decomposer_records: list[DecomposerRecord] = []
        self._agents: dict[str, AgentState] = {}
        self._ticks: list[_TickRecord] = []

    # ------------------------------------------------------------------
    # trough provisioning
    # ------------------------------------------------------------------

    def get_trough(self, tier: str, kind_key: str) -> TroughAttention | None:
        """Return the trough for a given (tier, kind_key), or None if it
        hasn't been created yet (no broadcasts of that kind deposited)."""
        return self._troughs.get((tier, kind_key))

    def _ensure_trough(self, tier: str, kind_key: str, hidden_size: int) -> TroughAttention:
        key = (tier, kind_key)
        with self._lock:
            t = self._troughs.get(key)
            if t is None:
                t = TroughAttention(
                    hidden_size=hidden_size,
                    n_slots=self._n_slots_per_kind,
                    n_heads=_choose_n_heads(hidden_size),
                    out_seq_len=8,
                    seed=self._trough_seed,
                )
                self._troughs[key] = t
            return t

    # ------------------------------------------------------------------
    # broadcasts
    # ------------------------------------------------------------------

    def add_broadcast(self, b: Broadcast) -> None:
        with self._lock:
            if b.id in self._entries:
                # Idempotent — same id replaces the entry.
                self._free_slot(self._entries[b.id])
            entry = _BroadcastEntry(broadcast=b)

            # Abstention broadcasts get cached but don't go into the trough
            # (matches the old SQL filter that excluded them from query_pool).
            if not b.abstained and b.channel_embedding:
                hidden = len(b.channel_embedding)
                trough = self._ensure_trough(b.tier, _kind_key(b), hidden)
                # The old pool had no upper bound on broadcast count;
                # the trough does. If the trough is full, evict oldest
                # rotted/claimed slots first; if still full, drop the
                # oldest alive entry.
                self._make_room(trough)
                slot_ids = trough.deposit([b])
                entry.trough_key = (b.tier, _kind_key(b))
                entry.slot_id = slot_ids[0]
            self._entries[b.id] = entry

    def add_broadcast_many(self, bs: Iterable[Broadcast]) -> None:
        for b in bs:
            self.add_broadcast(b)

    def query_pool(
        self,
        tier: str,
        diet_tags: list[str] | None = None,
        limit: int = 64,
    ) -> list[Broadcast]:
        """Unclaimed, unrotted, non-abstention broadcasts at the given tier.

        diet_tags=None → any tag matches. Otherwise OR-match on tags.
        Order: most recently created first (matches the old SQL ORDER BY
        created_tick DESC).
        """
        diet_set = set(diet_tags) if diet_tags else None
        with self._lock:
            cands: list[_BroadcastEntry] = []
            for entry in self._entries.values():
                b = entry.broadcast
                if b.tier != tier:
                    continue
                if entry.claimed_by is not None or entry.rotted or b.abstained:
                    continue
                if diet_set is not None:
                    if not any(t in diet_set for t in b.diet_tags):
                        continue
                cands.append(entry)
            cands.sort(key=lambda e: e.broadcast.created_tick, reverse=True)
            return [e.broadcast for e in cands[:limit]]

    def claim(self, item_ids: list[str], hunter_id: str, tick: int) -> list[Broadcast]:
        """Atomically mark the given broadcasts as claimed by `hunter_id`.

        Returns the subset that was actually claimed (already-claimed or
        rotted items are silently skipped — matches the old SQL behavior).
        Claimed broadcasts also free their trough slot.
        """
        if not item_ids:
            return []
        claimed: list[Broadcast] = []
        with self._lock:
            for iid in item_ids:
                entry = self._entries.get(iid)
                if entry is None:
                    continue
                if entry.claimed_by is not None or entry.rotted:
                    continue
                entry.claimed_by = hunter_id
                entry.claimed_at = str(tick)
                self._free_slot(entry)
                claimed.append(entry.broadcast)
        return claimed

    def mark_rotted(self, item_ids: list[str]) -> int:
        if not item_ids:
            return 0
        n = 0
        with self._lock:
            for iid in item_ids:
                entry = self._entries.get(iid)
                if entry is None:
                    continue
                if entry.claimed_by is not None or entry.rotted:
                    continue
                entry.rotted = True
                self._free_slot(entry)
                n += 1
        return n

    # ------------------------------------------------------------------
    # logbook sinks (in-memory; preserved for tests / diagnostics)
    # ------------------------------------------------------------------

    def add_judgment(self, j: PredatorJudgment) -> None:
        with self._lock:
            self._judgments.append(j)

    def add_decomposer_record(self, r: DecomposerRecord) -> None:
        with self._lock:
            self._decomposer_records.append(r)

    def upsert_agent(self, a: AgentState) -> None:
        with self._lock:
            self._agents[a.id] = a

    def record_tick(
        self,
        tick_id: int,
        n_broadcasts: int,
        n_eaten: int,
        n_rotted: int,
        n_judgments: int,
        n_abstentions: int = 0,
        notes: dict | None = None,
    ) -> None:
        with self._lock:
            self._ticks.append(
                _TickRecord(
                    tick_id=tick_id,
                    n_broadcasts=n_broadcasts,
                    n_eaten=n_eaten,
                    n_rotted=n_rotted,
                    n_judgments=n_judgments,
                    n_abstentions=n_abstentions,
                    notes=dict(notes or {}),
                )
            )

    def close(self) -> None:
        # No persistent resources to release.
        return None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _free_slot(self, entry: _BroadcastEntry) -> None:
        """Release the trough slot held by this entry, if any."""
        if entry.trough_key is None or entry.slot_id is None:
            return
        trough = self._troughs.get(entry.trough_key)
        if trough is None:
            return
        try:
            trough.evict([entry.slot_id])
        except IndexError:
            pass
        entry.slot_id = None
        entry.trough_key = None

    def _make_room(self, trough: TroughAttention) -> None:
        """If a trough is full, evict the oldest claimed/rotted entries
        (which should already have been freed, but defensively look for
        any stragglers).

        If the trough is still full of *alive* entries (i.e. unclaimed,
        unrotted broadcasts), evict the oldest one — preserving the
        existing pool's "newest first" semantics (older broadcasts get
        squeezed out under pressure).
        """
        if int(trough.alive.sum().item()) < trough.n_slots:
            return

        # Find broadcasts living in this trough, oldest first.
        target_key = None
        for k, t in self._troughs.items():
            if t is trough:
                target_key = k
                break
        if target_key is None:
            return

        candidates = [
            e for e in self._entries.values()
            if e.trough_key == target_key and e.slot_id is not None
        ]
        candidates.sort(key=lambda e: e.broadcast.created_tick)
        if not candidates:
            return
        oldest = candidates[0]
        # Treat eviction-under-pressure as "rotted" so callers don't try
        # to claim it later.
        oldest.rotted = True
        self._free_slot(oldest)


def _choose_n_heads(hidden_size: int) -> int:
    """Pick a power-of-2 head count that divides hidden_size, capped at 8."""
    for h in (8, 4, 2, 1):
        if hidden_size % h == 0:
            return h
    return 1
