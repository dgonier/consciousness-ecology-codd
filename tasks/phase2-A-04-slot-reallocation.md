# Mission 04: slot-reallocation

**Handle**: `phase2-A`
**Phase**: 2 (after `phase2-B:02:DONE`)
**Mission file**: `phase2-A-04-slot-reallocation.md`
**Dependencies**: `phase1-A:01:DONE`, `phase2-B:02:DONE`
**Blocks**: `phase3-A:06`

---

## Before You Start

```bash
cd /home/dgonier/ecology_experiment && \
  grep -nE '@(all|phase2-A|phase2)' tasks/scratchpad.md
```

Confirm BOTH `phase1-A:01:DONE` AND `phase2-B:02:DONE`. Then atomically flip:
```
phase2-A:04:PENDING → phase2-A:04:RUNNING
```

---

## Goal

When a slot dies (Mission 02 marks it dead), reallocate it via **niche-aware spawning**: find the consumer Q that's been most underserved (lowest max attention across slots), initialize a new K vector that's high-similarity with that Q. This is the "birth" half of the lifecycle. Together with Mission 02, the trough has full population dynamics.

---

## Files to Modify

1. `trophic/trophic/ecology/slot_lifecycle.py` — add spawn function (created in Mission 02)
2. `trophic/trophic/trough_attention.py` — add `spawn_into_dead_slot()`
3. `trophic/trophic/runner.py` — track recent Q vectors per consumer, pass to spawning

## Files to Create / Extend

4. Extend `trophic/tests/test_slot_lifecycle.py` with spawn tests

---

## Implementation Steps

### Step 1: Q satisfaction tracking

Each consumer's queries against the trough get logged. We keep a small ring buffer of recent (Q, max_attention) pairs per trough:

```python
# In TroughAttention.__init__:
self.q_history_size = 64
self.register_buffer("recent_queries", torch.zeros(self.q_history_size, hidden_size))
self.register_buffer("recent_max_attention", torch.zeros(self.q_history_size))
self._q_idx = 0

# In attend(), after computing per-slot weights:
if self.training:  # or always — design choice
    self.recent_queries[self._q_idx] = query.detach()
    self.recent_max_attention[self._q_idx] = float(per_slot_alive.max().item())
    self._q_idx = (self._q_idx + 1) % self.q_history_size
```

### Step 2: Underserved query identification

```python
def find_underserved_query(self) -> torch.Tensor:
    """Return the Q from recent history that had lowest max attention."""
    idx = int(self.recent_max_attention.argmin().item())
    return self.recent_queries[idx]
```

### Step 3: Spawn function

In `slot_lifecycle.py`:

```python
def niche_aware_spawn(
    underserved_q: torch.Tensor,
    W_K: torch.Tensor,            # K projection — invert it to land near Q in K-space
    target_norm: float,
    noise_scale: float = 0.02,
) -> torch.Tensor:
    """Generate a new V vector such that K = W_K @ V is similar to underserved_q.

    We can't truly invert W_K (rank issues), so we use the pseudo-inverse:
        new_v ≈ W_K_pinv @ underserved_q
    Then add noise and rescale to target_norm so the deposit looks like a real broadcast.
    """
    W_K_pinv = torch.linalg.pinv(W_K)
    new_v = W_K_pinv @ underserved_q
    new_v = new_v + noise_scale * torch.randn_like(new_v)
    new_v = new_v * (target_norm / (new_v.norm() + 1e-8))
    return new_v
```

### Step 4: `spawn_into_dead_slot()` on TroughAttention

```python
def spawn_into_dead_slot(self, slot_id: int) -> dict:
    assert not self.alive[slot_id], f"Slot {slot_id} is alive — refusing to overwrite"
    underserved_q = self.find_underserved_query()
    new_v = niche_aware_spawn(
        underserved_q, self.W_K.weight, self.target_norm,
    )
    self.V_store[slot_id] = new_v
    self.alive[slot_id] = True
    self.dead[slot_id] = False
    self.cumulative_attention[slot_id] = 0
    self.age[slot_id] = 0
    self.under_threshold_ticks[slot_id] = 0
    self.broadcast_ids[slot_id] = f"spawn:{slot_id}:{int(self.age.sum())}"
    return {"slot_id": slot_id, "spawned_for_q_norm": float(underserved_q.norm())}
```

### Step 5: Wire into `step_lifecycle()`

After killing dead slots, immediately spawn into them (population stays at fixed size):

```python
def step_lifecycle(self) -> dict:
    # ... (existing kill logic from Mission 02)
    spawn_stats = []
    if self.population_strategy == "fixed":
        for slot_id in stats["killed_ids"]:
            spawn_stats.append(self.spawn_into_dead_slot(slot_id))
    stats["spawned"] = spawn_stats
    return stats
```

`population_strategy` is a new attribute on `TroughAttention`, default `"fixed"`. Other strategies (carrying-capacity, dynamic) can come later — for now, fixed-size matches existing pool behavior.

### Step 6: Tests

Add to `test_slot_lifecycle.py`:

- `test_spawn_into_dead_slot_resets_alive_to_true`: kill a slot, spawn, verify alive[slot_id]=True.
- `test_niche_aware_spawn_targets_underserved_q`: feed N attends with one query repeated (well-served) and one unique low-similarity query (underserved); kill a slot; spawn; verify the new slot's K is closer to the underserved Q than to the well-served one.
- `test_population_stays_fixed`: run 100 ticks, verify `alive.sum()` equals initial population at all times.
- `test_spawn_does_not_overwrite_alive_slot`: assertion fires if you call `spawn_into_dead_slot(i)` with alive[i]=True.

---

## Acceptance Criteria

- [ ] Populations stay full automatically across long mock-SFT runs
- [ ] Spawn picks targets that are under-attended (verifiable via test)
- [ ] No assertion errors during 100+ ticks of mock SFT
- [ ] Tests pass

---

## Testing Conditions (exit verification)

1. **Spawn tests pass**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     python -m pytest tests/test_slot_lifecycle.py -k "spawn" -v 2>&1 | tail -15
   ```
   **Expected**: ≥4 spawn-named tests passing.

2. **Whole suite green**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && python -m pytest tests/ -x -v 2>&1 | tail -30
   ```
   **Expected**: all tests pass.

3. **Population stability across 100 ticks**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && python -c "
   import torch
   from trophic.trough_attention import TroughAttention
   t = TroughAttention(hidden_size=64, n_slots=8, n_heads=4, seed=1)
   # populate 8 slots, then run 100 attend+lifecycle cycles
   # ... assert t.alive.sum() == 8 at every step
   print('OK')
   "
   ```
   **Expected**: no failures; `alive.sum()` constant.

4. **Niche-aware spawn quality**
   Run mock SFT 200 steps, log per-spawn `cosine(new_K, underserved_q)`. Mean should exceed 0.3 (better than random baseline of ~0).
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     TROPHIC_MOCK=1 timeout 600 python scripts/run_sft.py --steps 200 \
     --debug-spawn 2>&1 | grep "spawn-cos" | tail -20
   ```
   **Expected**: mean spawn-cos > 0.3.

If any condition fails, do not mark DONE; post `@all` in MESSAGES.

---

## Coordination

- Second mission for agent A (after `phase1-A:01`). Different handle, different phase.
- Depends on `phase2-B:02:DONE`. **Verify in STATUS before starting.**
- `step_lifecycle()` already exists from `phase2-B:02`. You're augmenting it (adding spawn calls after kill), not rewriting.

---

## When Done

1. Re-run inbox grep, address pending.
2. Update STATUS:
   - `phase2-A:04:RUNNING` → `phase2-A:04:DONE`
3. If `phase2-C:03:DONE` and `phase2-D:05:DONE` are also done, set `phase3-A:06:BLOCKED` → `phase3-A:06:PENDING`.
4. Append MESSAGES:
   ```
   [<ts>] phase2-A > @phase3-A: slot reallocation live. Spawn stats over 200 mock-SFT steps:
     mean cosine(new_K, underserved_q) = <v>; killed/spawned counts = <v>/<v>.
   ```
