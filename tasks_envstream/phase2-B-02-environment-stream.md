# Mission 02: environment-stream

**Handle**: `phase2-B`
**Phase**: 2 (parallel after `phase1-A:01`)
**Mission file**: `phase2-B-02-environment-stream.md`
**Dependencies**: `phase1-A:01:DONE`
**Blocks**: `phase3-A:05`

---

## Before You Start

```bash
cd /home/dgonier/ecology_experiment/trophic && \
  grep -nE '@(all|phase2-B|phase2)' tasks_envstream/scratchpad.md
```

Confirm `phase1-A:01:DONE` in STATUS. Then atomically flip:
```
phase2-B:02:PENDING → phase2-B:02:RUNNING
```

---

## Goal

Build `EnvironmentStream` — the input-layer K/V substrate. Same architectural pattern as `TroughAttention` (deposit / attend / lifecycle), with one new mechanic: **wavelength filtering on attend**, so each producer naturally pulls a different overlapping subset of the stream (cortical-column fan-out).

After this mission, the architecture has K/V at every tier: external → producer (via EnvironmentStream), producer → herbivore (via TroughAttention), herbivore → predator (via TroughAttention).

---

## Files to Create / Modify

### Create
- `trophic/environment_stream.py` — the class
- `tests/test_environment_stream.py` — at least 6 tests

### Don't touch
- `trophic/trough_attention.py` — extend or compose, don't modify
- `trophic/agents/producer.py` — that's phase2-D's job
- `trophic/runner.py` — phase3-A wires you in

---

## Implementation Steps

### Step 1: Compose, don't reimplement

`TroughAttention` already has K/V deposit, multi-head attend, lifecycle (decay → death → niche-aware spawn), decomposer-bias hook. Reuse it. Compose:

```python
# trophic/environment_stream.py
from dataclasses import dataclass
import torch
import torch.nn as nn

from .trough_attention import TroughAttention, TroughAttendOutput
from .types import RawInput


@dataclass
class StreamAttendOutput:
    output: torch.Tensor              # [out_seq_len, hidden]
    per_slot_attention: torch.Tensor  # [n_slots]
    per_head_attention: torch.Tensor  # [n_heads, n_slots]
    null_prob: float
    selected_slot_ids: list[int]
    rejected_slot_ids: list[int]
    masked_out_count: int             # how many slots were filtered by wavelength_filter


class EnvironmentStream(nn.Module):
    """Input-layer K/V substrate. Adapters deposit RawInputs (embedded);
    producers subscribe to wavelength subsets and attend.

    Composes TroughAttention for the K/V mechanics; adds:
      - per-slot source-tag bookkeeping
      - wavelength-filtered attend (mask non-matching slots before softmax)
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
            hidden_size=hidden_size, n_slots=n_slots, n_heads=n_heads,
            out_seq_len=out_seq_len, target_norm=target_norm, seed=seed,
        )
        # Per-slot source tag (Python list, not buffer — strings).
        # Indexed parallel to TroughAttention.alive / V_store.
        self._slot_source_tags: list[str | None] = [None] * n_slots

    @property
    def n_slots(self) -> int: return self._trough.n_slots
    @property
    def hidden_size(self) -> int: return self._trough.hidden_size
    @property
    def alive(self) -> torch.Tensor: return self._trough.alive

    def deposit(self, raw_input: RawInput, embedding: torch.Tensor) -> int:
        """Deposit a single RawInput. Returns the slot id used.

        Caller is responsible for embedding (run via Qwen + pool).
        """
        # Reuse TroughAttention.deposit but with single-element list-shape API.
        # If TroughAttention.deposit expects a Broadcast-shaped object, you may
        # need to either:
        #  (a) extend TroughAttention with a `deposit_raw(emb, source_tag)` method
        #  (b) construct a minimal "broadcast-shaped" wrapper here
        # Choose whichever keeps the diff smaller. Document the choice.
        ...

    def attend(
        self,
        query: torch.Tensor,
        wavelength_filter: set[str],
        tau: float = 1.0,
    ) -> StreamAttendOutput:
        """Cross-attention readout, but mask out slots whose source_tag is
        not in wavelength_filter before softmax.
        """
        # Compute alive_mask AND wavelength_mask.
        # Pass an effective_alive_mask = alive_mask & wavelength_mask into
        # the underlying attend computation. Either:
        #  (a) call self._trough.attend(query, tau, external_bias=...)
        #      and use external_bias to push masked-out slot logits to -inf
        #  (b) write a copy of attend() inline with the extra mask
        # (a) is preferred (no code duplication); (b) only if (a) doesn't
        # cleanly express -inf masking.
        ...

    def step_lifecycle(self) -> dict:
        """Delegate to underlying trough's lifecycle. Plus optional
        temporal decay: slots older than max_age get killed regardless of
        attention. (Optional — start without temporal decay; can add later.)
        """
        return self._trough.step_lifecycle()

    def slot_state(self) -> dict:
        """Per-slot state. Augmented with source_tag from this module."""
        base = self._trough.slot_state()
        base["source_tags"] = list(self._slot_source_tags)
        return base
```

### Step 2: Implement deposit

The simplest path: `TroughAttention.deposit` likely accepts `Broadcast` objects (per the trough-as-transformer migration). For EnvironmentStream we need to deposit `(embedding, source_tag)` pairs *without* requiring a full Broadcast wrapper.

Two clean ways:

1. **Add a `deposit_raw(embedding, source_tag, ext_id)` method to `TroughAttention`** — the lower-level form. Keep the existing `deposit(broadcasts)` but have it call `deposit_raw` internally. **Recommended.**
2. **Construct minimal Broadcast wrappers in EnvironmentStream's deposit** — works but creates objects that aren't really broadcasts. Wasteful and confusing.

If you go with (1), the change to `trough_attention.py` is additive and small. Make sure all existing trough tests still pass.

### Step 3: Implement attend with wavelength filter

The simplest mechanism: build a `[n_slots]` boolean mask where `True` means "this slot's source_tag is in wavelength_filter AND alive". Then push the masked-out slots' logits to `-inf` before softmax. This is the standard attention-masking trick.

If `TroughAttention.attend` already accepts an `external_bias` parameter (per #6 fix), you can pass `mask_bias = where(mask, 0.0, -inf)` as the external bias. That's option (a) above and it's the cleanest implementation.

Verify by reading `trophic/trough_attention.py` — find the `attend()` signature. If `external_bias` exists and is added pre-softmax, this works directly.

### Step 4: Tests in `tests/test_environment_stream.py`

```python
"""EnvironmentStream tests."""
from __future__ import annotations

import torch

from trophic.environment_stream import EnvironmentStream
from trophic.types import RawInput


def _ri(source: str, idx: int) -> RawInput:
    return RawInput(id=f"r{idx}", source=source, payload={"i": idx})


def test_construct_smoke():
    s = EnvironmentStream(hidden_size=64, n_slots=8, n_heads=4, seed=1)
    assert s.n_slots == 8
    assert int(s.alive.sum()) == 0


def test_deposit_marks_alive_and_records_source_tag():
    s = EnvironmentStream(hidden_size=64, n_slots=4, n_heads=4, seed=1)
    emb = torch.randn(64)
    slot = s.deposit(_ri("ohlcv", 0), emb)
    state = s.slot_state()
    assert state["source_tags"][slot] == "ohlcv"
    assert int(s.alive.sum()) == 1


def test_wavelength_filter_excludes_non_matching_slots():
    """Deposit two slots, one ohlcv one tweets. Attend with filter={'ohlcv'}.
    The tweets slot must contribute zero attention."""
    s = EnvironmentStream(hidden_size=64, n_slots=8, n_heads=4, seed=1)
    s.deposit(_ri("ohlcv", 0), torch.randn(64))
    s.deposit(_ri("tweets", 1), torch.randn(64))
    q = torch.randn(64)
    out = s.attend(q, wavelength_filter={"ohlcv"}, tau=1.0)
    # The tweets slot should have ~0 attention weight.
    # Find its slot id from slot_state and assert the per-slot weight is ~0.
    state = s.slot_state()
    tweets_slot = state["source_tags"].index("tweets")
    assert out.per_slot_attention[tweets_slot].abs() < 1e-4


def test_wavelength_filter_with_multiple_tags():
    """Filter {ohlcv, tweets} should pass both, exclude filing."""
    s = EnvironmentStream(hidden_size=64, n_slots=8, n_heads=4, seed=1)
    s.deposit(_ri("ohlcv", 0), torch.randn(64))
    s.deposit(_ri("tweets", 1), torch.randn(64))
    s.deposit(_ri("filing", 2), torch.randn(64))
    q = torch.randn(64)
    out = s.attend(q, wavelength_filter={"ohlcv", "tweets"}, tau=1.0)
    assert out.masked_out_count == 1   # only filing was masked


def test_attend_full_filter_equivalent_to_no_mask():
    """If wavelength_filter contains every source_tag present, behavior
    matches TroughAttention.attend with no bias."""
    # Compare per-slot attention against an unfiltered baseline.
    ...


def test_step_lifecycle_delegates():
    s = EnvironmentStream(hidden_size=64, n_slots=4, n_heads=4, seed=1)
    for i in range(4):
        s.deposit(_ri("ohlcv", i), torch.randn(64))
    s.step_lifecycle()  # should not raise; returns dict
```

---

## Acceptance Criteria

- [ ] `trophic/environment_stream.py` exists with `EnvironmentStream` class matching the INTERFACE CONTRACT in scratchpad
- [ ] `deposit(raw_input, embedding) -> slot_id` works
- [ ] `attend(query, wavelength_filter, tau) -> StreamAttendOutput` masks non-matching slots
- [ ] `step_lifecycle()` delegates to underlying trough
- [ ] `slot_state()` includes per-slot source tags
- [ ] At least 6 new tests in `tests/test_environment_stream.py` pass
- [ ] All existing 55+ tests still pass (don't break TroughAttention)

---

## Testing Conditions (exit verification)

1. **New EnvironmentStream tests pass**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     .venv/bin/python -m pytest tests/test_environment_stream.py -v 2>&1 | tail -15
   ```
   **Expected**: ≥6 passed, 0 failed.

2. **TroughAttention tests still green** (you may have touched it for `deposit_raw`)
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     .venv/bin/python -m pytest tests/test_trough_attention.py tests/test_slot_lifecycle.py -v 2>&1 | tail -15
   ```
   **Expected**: all pass.

3. **Whole suite green**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     .venv/bin/python -m pytest tests/ --tb=line -q 2>&1 | tail -5
   ```
   **Expected**: 60+ passed, 0 failed.

4. **Wavelength filter actually masks**
   Smoke script: deposit 5 slots with different source tags, attend with a single-tag filter, verify the per-slot attention for non-matching slots is < 1e-4.
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && .venv/bin/python -c "
   import torch
   from trophic.environment_stream import EnvironmentStream
   from trophic.types import RawInput
   s = EnvironmentStream(hidden_size=64, n_slots=5, n_heads=4, seed=1)
   for i, tag in enumerate(['ohlcv','tweets','filing','options','press']):
       s.deposit(RawInput(id=f'r{i}', source=tag, payload={}), torch.randn(64))
   out = s.attend(torch.randn(64), wavelength_filter={'tweets'}, tau=1.0)
   tags = s.slot_state()['source_tags']
   for slot, tag in enumerate(tags):
       w = float(out.per_slot_attention[slot])
       expected_low = tag != 'tweets'
       print(f'  slot{slot} tag={tag} weight={w:.4f} {\"<masked>\" if expected_low and w < 1e-4 else \"\"}')
   "
   ```
   **Expected**: only the tweets slot has non-trivial attention.

---

## Coordination

- You're modifying `trophic/trough_attention.py` (potentially adding `deposit_raw`). phase2-C and phase2-D do NOT touch this file. Stay additive.
- The interface contract (`StreamAttendOutput`, `attend(query, wavelength_filter, tau)`) is shared with phase3-A and the producer refactor in phase2-D. **Do not change without `@all` MESSAGES note.**
- Live tail listener (recommended; siblings phase2-C/D running concurrently):
  ```bash
  tail -F /home/dgonier/ecology_experiment/trophic/tasks_envstream/scratchpad.md \
    | grep --line-buffered -E '@(all|phase2-B|phase2)'
  ```

---

## Out of Scope

- Wiring EnvironmentStream into `runner.py` — that's phase3-A
- Concrete adapter implementations — phase2-C
- Producer refactor — phase2-D
- Temporal decay (slots aging out by wall clock) — defer to a follow-up issue
- Multi-modal embeddings (images, audio) — text/numerical only

---

## When Done

1. Re-run inbox grep, address pending.
2. Update STATUS in scratchpad: `phase2-B:02:RUNNING` → `phase2-B:02:DONE`.
3. Append MESSAGES:
   ```
   - [<YYYY-MM-DD HH:MM>] phase2-B > @phase3-A: EnvironmentStream landed at trophic/environment_stream.py. Composes TroughAttention via {choice: deposit_raw added / Broadcast wrapper / other}. attend(q, wavelength_filter, tau) masks non-matching slots via {choice: external_bias=-inf / inline mask / other}. test_environment_stream.py: <PASS>/<TOTAL>. Full suite: <PASS>/<TOTAL>. Open questions for phase3-A: <none / list>.
   ```
