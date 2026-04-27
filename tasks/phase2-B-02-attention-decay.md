# Mission 02: attention-decay

**Handle**: `phase2-B`
**Phase**: 2 (parallel after `phase1-A:01`)
**Mission file**: `phase2-B-02-attention-decay.md`
**Dependencies**: `phase1-A:01:DONE`
**Blocks**: `phase2-A:04` (needs decay tracking before it can spawn replacements)

---

## Before You Start

```bash
cd /home/dgonier/ecology_experiment && \
  grep -nE '@(all|phase2-B|phase2)' tasks/scratchpad.md
```

Confirm `phase1-A:01:DONE` in STATUS. Then atomically flip:
```
phase2-B:02:PENDING → phase2-B:02:RUNNING
```

---

## Goal

Wire ecological selection pressure onto the trough's K/V slots. Each slot accumulates an EMA of attention received. Slots whose cumulative attention falls below ε for `n_patience` consecutive ticks are killed (slot freed for reallocation). This is the "death" half of slot lifecycle — Mission 04 handles "birth".

---

## Files to Modify

1. `trophic/trophic/trough_attention.py` — extend with decay logic
2. `trophic/trophic/ecology/slot_lifecycle.py` — NEW: pure functions for decay/death

## Files to Create

3. `trophic/tests/test_slot_lifecycle.py`

---

## Implementation Steps

### Step 1: Decay state in `TroughAttention`

Mission 01 already added `cumulative_attention` and `age` buffers. Extend with:

```python
# In __init__:
self.register_buffer("under_threshold_ticks", torch.zeros(n_slots, dtype=torch.long))
self.register_buffer("dead", torch.zeros(n_slots, dtype=torch.bool))
self.alpha_decay = 0.9      # EMA factor on cumulative attention
self.epsilon_death = 0.01   # death threshold
self.n_patience = 5         # consecutive sub-ε ticks before death
```

### Step 2: Update bookkeeping in `attend()`

After computing softmax weights:

```python
# Update per-slot attention rolling average (alive slots only).
alive_idx = torch.nonzero(self.alive, as_tuple=True)[0]
tick_attention = weights[: len(alive_idx)]  # alive-only weights, sans null
self.cumulative_attention[alive_idx] = (
    self.alpha_decay * self.cumulative_attention[alive_idx]
    + (1 - self.alpha_decay) * tick_attention
)
self.age[alive_idx] += 1
```

### Step 3: `slot_lifecycle.py` — pure decay logic

Pure functions, easy to unit test:

```python
import torch

def mark_underperforming(
    cumulative_attention: torch.Tensor,
    alive: torch.Tensor,
    epsilon: float,
    under_threshold_ticks: torch.Tensor,
) -> torch.Tensor:
    """Returns updated under_threshold_ticks. Resets to 0 for slots above ε."""
    under = (cumulative_attention < epsilon) & alive
    new_counts = torch.where(under, under_threshold_ticks + 1, torch.zeros_like(under_threshold_ticks))
    return new_counts


def kill_dead_slots(
    under_threshold_ticks: torch.Tensor,
    alive: torch.Tensor,
    n_patience: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (new_alive, killed_mask) where killed_mask is True for slots that just died."""
    killed = (under_threshold_ticks >= n_patience) & alive
    new_alive = alive & ~killed
    return new_alive, killed
```

### Step 4: `step_lifecycle()` method on `TroughAttention`

```python
def step_lifecycle(self) -> dict:
    """Called once per tick after all attend() calls. Returns lifecycle stats."""
    self.under_threshold_ticks = mark_underperforming(
        self.cumulative_attention, self.alive, self.epsilon_death, self.under_threshold_ticks,
    )
    new_alive, killed = kill_dead_slots(
        self.under_threshold_ticks, self.alive, self.n_patience,
    )
    n_killed = int(killed.sum().item())
    self.alive.copy_(new_alive)
    self.dead = self.dead | killed
    # Clear the dead slots so reallocation (Mission 04) starts from a clean state.
    self.V_store[killed] = 0
    self.cumulative_attention[killed] = 0
    self.under_threshold_ticks[killed] = 0
    self.age[killed] = 0
    for idx in torch.nonzero(killed, as_tuple=True)[0].tolist():
        self.broadcast_ids[idx] = None
    return {
        "n_killed": n_killed,
        "n_alive": int(self.alive.sum().item()),
        "killed_ids": torch.nonzero(killed, as_tuple=True)[0].tolist(),
    }
```

### Step 5: Wire `step_lifecycle()` into the runner

In `runner.py`, after each tick's attend cycle, call `trough.step_lifecycle()` for each trough. Log the returned stats.

### Step 6: Tests in `test_slot_lifecycle.py`

- `test_mark_underperforming_increments`: slots below ε get +1; slots above reset to 0.
- `test_kill_dead_slots_after_patience`: feed N=5 ticks below ε → `killed` is True.
- `test_attend_then_lifecycle_kills_unattended_slot`: deposit 2 broadcasts, repeatedly attend with a Q that only matches one; the unattended slot dies after `n_patience` ticks.
- `test_revival_resets_counter`: slot dropped below ε for 3 ticks then attended → counter resets to 0, slot survives.
- `test_age_increments_only_for_alive`: dead slots' age frozen.

---

## Acceptance Criteria

- [ ] `step_lifecycle()` runs without error after every attend cycle
- [ ] Tests in `test_slot_lifecycle.py` (≥5) pass
- [ ] Log line per tick showing `n_alive`/`n_killed` per trough
- [ ] All previously passing tests still pass
- [ ] Mock SFT smoke completes — unattended slots die quietly without breaking the run

---

## Testing Conditions (exit verification)

1. **Lifecycle tests pass**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && python -m pytest tests/test_slot_lifecycle.py -v 2>&1 | tail -20
   ```
   **Expected**: ≥5 passed, 0 failed.

2. **No regression in TroughAttention or whole suite**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && python -m pytest tests/ -x -v 2>&1 | tail -30
   ```
   **Expected**: All tests pass.

3. **Death actually fires under starvation**
   Smoke script: deposit 4 slots, attend with a Q only matching slot 0 for `n_patience+1` ticks, call `step_lifecycle()` each tick, assert at least one slot died.
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && python -c "
   from trophic.trough_attention import TroughAttention
   import torch
   t = TroughAttention(hidden_size=64, n_slots=4, n_heads=4, seed=1)
   # ... deposit 4 broadcasts, repeatedly attend favoring slot 0 ...
   print('alive_after:', int(t.alive.sum()))
   "
   ```
   **Expected**: alive count drops from 4.

4. **Mock SFT smoke runs to completion**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     TROPHIC_MOCK=1 timeout 300 python scripts/run_sft.py --steps 50 2>&1 | tail -30
   ```
   **Expected**: exit 0, no exceptions. Per-tick `n_alive`/`n_killed` log lines present.

If any test fails, do not mark DONE; post `@all` in MESSAGES.

---

## Coordination

- Modifying `trough_attention.py` alongside `phase2-C:03` and `phase2-D:05`. Keep changes **additive** — new methods/attrs only, do not reflow existing methods.
- If you must rename or restructure something `phase1-A:01` introduced, post in MESSAGES first with `@all`.
- Live tail listener (optional, since C and D are running concurrently):
  ```bash
  tail -F /home/dgonier/ecology_experiment/tasks/scratchpad.md \
    | grep --line-buffered -E '@(all|phase2-B|phase2)'
  ```

---

## When Done

1. Re-run inbox grep, address anything pending.
2. Update `tasks/scratchpad.md` STATUS:
   - `phase2-B:02:RUNNING` → `phase2-B:02:DONE`
   - `phase2-A:04:BLOCKED` → `phase2-A:04:PENDING` (you unblock Mission 04)
3. Append a MESSAGES entry:
   ```
   [<ts>] phase2-B > @phase2-A: step_lifecycle() landed. n_killed dist over 50-step smoke: <stats>.
     epsilon_death=<v>, n_patience=<v>. Spawn handoff is yours.
   ```
