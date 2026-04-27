# Mission 01: trough-attention

**Handle**: `phase1-A`
**Phase**: 1 (foundation, sequential)
**Mission file**: `phase1-A-01-trough-attention.md`
**Dependencies**: none
**Blocks**: `phase2-B:02`, `phase2-C:03`, `phase2-D:05` (all need TroughAttention to exist); `phase2-A:04` (needs 01+02); `phase3-A:06` (needs all)

---

## Before You Start

```bash
cd /home/dgonier/ecology_experiment && \
  grep -nE '@(all|phase1-A|phase1)' tasks/scratchpad.md
```

Atomically flip your STATUS line:
```
phase1-A:01:PENDING → phase1-A:01:RUNNING
```

---

## Goal

Build `TroughAttention` — a stateful K/V container per tier boundary that combines what `SubstratePool` does (hold broadcasts, hand them out) and what `Channel` does (cross-attention readout with null gate). After this mission, the legacy `SubstratePool` is a thin shim delegating to `TroughAttention`. No behavior change yet — just unification of the two abstractions.

This is the foundation. Everything in Phase 2 attaches to it.

---

## Files to Create

1. `trophic/trophic/trough_attention.py` — the new class
2. `trophic/tests/test_trough_attention.py` — unit tests

## Files to Modify

3. `trophic/trophic/substrate.py` — make `SubstratePool` a thin shim
4. `trophic/trophic/runner.py` — instantiate `TroughAttention` at each tier boundary, pass to substrate construction

**DO NOT** modify herbivore/predator agents in this mission. They keep talking to `SubstratePool` exactly as before. The shim ensures their code works unchanged.

---

## Implementation Steps

### Step 1: Define `TroughAttention` in `trough_attention.py`

```python
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TroughAttendOutput:
    output: torch.Tensor              # [out_seq_len, hidden]
    per_slot_attention: torch.Tensor  # [n_slots]
    per_head_attention: torch.Tensor  # [n_heads, n_slots]
    null_prob: float
    selected_slot_ids: list[int]
    rejected_slot_ids: list[int]


class TroughAttention(nn.Module):
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
        self.hidden_size = hidden_size
        self.n_slots = n_slots
        self.n_heads = n_heads
        self.out_seq_len = out_seq_len
        self.target_norm = target_norm
        self.head_dim = hidden_size // n_heads
        assert hidden_size % n_heads == 0

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)

        # Q/K/V projections — same scaled init as Channel for compatibility.
        self.W_Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.W_Q.weight.copy_(torch.randn_like(self.W_Q.weight) * 0.02)
            self.W_K.weight.copy_(torch.randn_like(self.W_K.weight) * 0.02)
            self.W_V.weight.copy_(0.5 * torch.eye(hidden_size))

        # Null slot: learned K/V row for "I attend to nothing".
        self.K_null = nn.Parameter(torch.zeros(hidden_size))
        self.V_null = nn.Parameter(torch.zeros(hidden_size))
        self.null_bias = nn.Parameter(torch.zeros(1))

        # Output MLP: out_seq_len projections of pooled context.
        self.proj_in = nn.Linear(hidden_size, hidden_size)
        self.proj_out = nn.Linear(hidden_size, hidden_size * out_seq_len)
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.zeros_(self.proj_out.bias)

        # K/V slot store — non-parameter buffers (broadcasts come from outside).
        self.register_buffer("V_store", torch.zeros(n_slots, hidden_size))
        self.register_buffer("alive", torch.zeros(n_slots, dtype=torch.bool))

        # Per-slot bookkeeping (used by Mission 02 for decay).
        self.register_buffer("cumulative_attention", torch.zeros(n_slots))
        self.register_buffer("age", torch.zeros(n_slots, dtype=torch.long))

        # Provenance — broadcast id of the entry in each slot, for the SubstratePool shim.
        self.broadcast_ids: list[str | None] = [None] * n_slots

    def deposit(self, broadcasts) -> list[int]:
        """Place broadcasts into free slots. Returns slot ids used.

        If no free slots, raise — Mission 04 adds eviction. For Mission 01, fixed-size
        with overflow-error semantics is fine (matches SubstratePool's TTL bookkeeping).
        """
        # Concrete implementation: find alive==False slots, write embeddings + ids.
        ...

    def attend(self, query: torch.Tensor, tau: float = 1.0) -> TroughAttendOutput:
        """Standard cross-attention with null gate.

        - query: [hidden]
        - V_store filtered to alive slots; concat null row.
        - softmax(Q·K / sqrt(d) / tau) → weights
        - context = weights · V (filtered), excluding null
        - null_prob = weight on null row
        - output: MLP-projected context expanded to [out_seq_len, hidden]
        """
        ...
```

Key points:

- Same Q/K/V init philosophy as `Channel` (scaled-small W_Q/W_K, identity-ish W_V), so weight loading from existing checkpoints stays sane.
- Null gate matches Channel's. **Null is not stored in V_store** — it's a separate parameter, so slot accounting stays clean.
- `cumulative_attention` and `age` are buffers, not parameters. They are written by `attend()` (Mission 02 will read and decay them).
- `broadcast_ids` is a Python list, not a tensor — so `claim()` semantics in the shim can return the actual `Broadcast` objects.

### Step 2: SubstratePool shim

Refactor `SubstratePool` to hold a `TroughAttention` per (tier, kind) tuple, plus a Python-side dict `id → Broadcast`. The legacy methods become:

- `add(broadcast)`: deposit into the matching trough; cache the broadcast object.
- `claim(filters, k)`: build a query from `filters` (use the existing diet-tag lookup, but return `top-k` by attention weight from the trough, not SQL).
- `query(...)`, etc: read from the cache + the trough's `slot_state()`.

The point: **callers don't change.** Mission 01 ends with all 9 tests passing.

### Step 3: Wire troughs at tier boundaries in `runner.py`

Find where `SubstratePool` is instantiated. Initialize it with explicit trough sizes (e.g., `n_slots=32` per kind). The `runner` now owns the trough indirectly through `SubstratePool`.

### Step 4: Tests in `test_trough_attention.py`

- `test_deposit_into_free_slot`: deposits N broadcasts, verifies `alive` mask + `broadcast_ids`.
- `test_attend_smoke`: random Q, smoke check on shapes.
- `test_attend_picks_high_similarity_slot`: deposit two broadcasts, one matching Q, one orthogonal; verify high attention on the matching slot.
- `test_attend_null_gate_fires_when_empty`: empty trough → `null_prob` near 1.0.
- `test_attend_norm_matches_target_norm`: output norm within tolerance of `target_norm`.
- `test_state_dict_round_trip`: save/load via `state_dict()`/`load_state_dict()`.

### Step 5: Verify no regressions

```bash
cd /home/dgonier/ecology_experiment/trophic && python -m pytest tests/ -x
```

All 9 existing tests + the new test file should pass.

---

## Acceptance Criteria

- [ ] `TroughAttention` exists in `trophic/trophic/trough_attention.py` with the signature in scratchpad INTERFACE CONTRACTS
- [ ] `SubstratePool` is a thin shim — no SQL, no rotted-flag logic. Just `id → Broadcast` cache + a `TroughAttention` per (tier, kind).
- [ ] All 9 existing tests pass
- [ ] At least 6 new tests in `test_trough_attention.py` pass
- [ ] Mock-mode SFT smoke completes without crashes
- [ ] Per-slot attention is being recorded in `cumulative_attention` (Mission 02 will use it)

---

## Testing Conditions (exit verification)

Run these and capture the output. Do not flip to DONE until each shows the expected result.

1. **Existing tests still green**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && python -m pytest tests/ -x -v 2>&1 | tail -30
   ```
   **Expected**: `9 passed` (or however many existed at start) plus the new TroughAttention tests, no failures.

2. **New TroughAttention tests pass**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && python -m pytest tests/test_trough_attention.py -v 2>&1 | tail -20
   ```
   **Expected**: ≥6 passed, 0 failed.

3. **Mock SFT smoke**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     TROPHIC_MOCK=1 timeout 300 python scripts/run_sft.py --steps 50 2>&1 | tail -30
   ```
   **Expected**: Run completes (exit code 0), no NaN losses, no Python exceptions.

4. **cumulative_attention observable**
   Quick interactive check or a smoke script:
   ```python
   trough = TroughAttention(hidden_size=2560, n_slots=8, seed=1)
   trough.deposit([fake_broadcast() for _ in range(4)])
   trough.attend(torch.randn(2560))
   assert trough.cumulative_attention.sum() > 0
   ```
   **Expected**: assertion holds.

If any of (1)–(4) fails, do not mark DONE; post a `@all` MESSAGES entry describing the failure.

---

## Out of Scope

- Slot eviction / death (handled by `phase2-B:02`, `phase2-A:04`)
- Multi-head niche specialization (`phase2-C:03`)
- Decomposer bias (`phase2-D:05`)
- Skip connections (`phase3-A:06`)
- Modifying the `Channel` class itself — deliberately deferred. Mission 01 puts trough machinery alongside Channel; later missions migrate consumers.

---

## When Done

1. Re-run inbox grep: `grep -nE '@(all|phase1-A|phase1)' tasks/scratchpad.md`. Address anything addressed to you.
2. Update `tasks/scratchpad.md` STATUS:
   - `phase1-A:01:RUNNING` → `phase1-A:01:DONE`
   - `phase2-B:02:BLOCKED`, `phase2-C:03:BLOCKED`, `phase2-D:05:BLOCKED` → all PENDING
3. Append a MESSAGES entry, addressed to the next-phase agents:
   ```
   [<timestamp>] phase1-A > @phase2: TroughAttention landed at trophic/trough_attention.py.
     Acceptance test summary: <pasted last lines from pytest>
     Surprises: <e.g. dtype quirks>
     Open questions for downstream: <if any>
   ```
