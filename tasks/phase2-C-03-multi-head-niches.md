# Mission 03: multi-head-niches

**Handle**: `phase2-C`
**Phase**: 2 (parallel after `phase1-A:01`)
**Mission file**: `phase2-C-03-multi-head-niches.md`
**Dependencies**: `phase1-A:01:DONE`
**Blocks**: nothing (independent path)

---

## Before You Start

```bash
cd /home/dgonier/ecology_experiment && \
  grep -nE '@(all|phase2-C|phase2)' tasks/scratchpad.md
```

Confirm `phase1-A:01:DONE`. Then atomically flip:
```
phase2-C:03:PENDING → phase2-C:03:RUNNING
```

---

## Goal

Replace hardcoded `diet_tags` ("from_technical_herbivore", "from_fundamental_herbivore", etc.) with **learned multi-head specialization** in the trough. Each head develops its own niche through training; producers' K vectors get routed to the head whose subspace they fit best. After this mission, you can delete `diet_tags` from the data flow (broadcasts may still carry them as a debug annotation, but they no longer drive routing).

---

## Files to Modify

1. `trophic/trophic/trough_attention.py` — split Q/K projections per head, track per-head specialization
2. `trophic/trophic/agents/herbivore.py` — drop `DIETS` SQL filter; query the trough directly
3. `trophic/trophic/agents/predator.py` — same
4. `trophic/trophic/substrate.py` — `claim()` no longer filters by diet tag string, just hands back top-k by attention

## Files to Create

5. `trophic/tests/test_multi_head_niches.py`

---

## Implementation Steps

### Step 1: Real multi-head attention in `attend()`

Mission 01 stubbed `n_heads`. Implement it properly:

```python
def attend(self, query: torch.Tensor, tau: float = 1.0) -> TroughAttendOutput:
    Q = self.W_Q(query).view(self.n_heads, self.head_dim)              # [H, d_h]
    alive_mask = self.alive
    if not alive_mask.any():
        return self._null_output()

    K_all = self.W_K(self.V_store)                                      # [n_slots, hidden]
    V_all = self.W_V(self.V_store)
    K_alive = K_all[alive_mask].view(-1, self.n_heads, self.head_dim)   # [n_alive, H, d_h]
    V_alive = V_all[alive_mask].view(-1, self.n_heads, self.head_dim)

    # Per-head attention — augment with null row.
    K_null = self.K_null.view(self.n_heads, self.head_dim)              # [H, d_h]
    V_null = self.V_null.view(self.n_heads, self.head_dim)

    # Build [n_alive+1, H, d_h] augmented K/V.
    K_aug = torch.cat([K_alive, K_null.unsqueeze(0)], dim=0)
    V_aug = torch.cat([V_alive, V_null.unsqueeze(0)], dim=0)

    # Per-head logits: [H, n_alive+1]
    logits = torch.einsum("hd,nhd->hn", Q, K_aug) / (self.head_dim ** 0.5) / tau
    logits[:, -1] = logits[:, -1] + self.null_bias  # learnable null prior
    attn = F.softmax(logits, dim=-1)                                    # [H, n_alive+1]

    null_prob = float(attn[:, -1].mean().item())
    attn_no_null = attn[:, :-1]                                         # [H, n_alive]

    # Per-head context: [H, d_h]
    context_per_head = torch.einsum("hn,nhd->hd", attn_no_null, V_alive)
    context = context_per_head.reshape(self.hidden_size)                # [hidden]

    # Aggregate per-slot attention across heads (for decay tracking from Mission 02).
    per_slot_alive = attn_no_null.mean(dim=0)                           # [n_alive]
    per_slot_attention = torch.zeros(self.n_slots, device=context.device)
    per_slot_attention[alive_mask] = per_slot_alive
    per_head_attention = torch.zeros(self.n_heads, self.n_slots, device=context.device)
    per_head_attention[:, alive_mask] = attn_no_null
    ...
```

### Step 2: Track per-head specialization

Add a buffer that records, for each (head, slot) pair, an EMA of attention received:

```python
self.register_buffer("head_specialization", torch.zeros(n_heads, n_slots))
# After each attend():
self.head_specialization = (
    self.alpha_decay * self.head_specialization
    + (1 - self.alpha_decay) * per_head_attention
)
```

Expose `head_specialization()` returning the buffer + the dominant head per slot:

```python
def slot_specialization(self) -> dict:
    """Per slot: which head specializes in it (argmax)."""
    dominant = self.head_specialization.argmax(dim=0)  # [n_slots]
    return {
        "head_per_slot": dominant.tolist(),
        "specialization_scores": self.head_specialization.cpu().tolist(),
    }
```

### Step 3: Drop diet_tags from routing path

In `herbivore.py` and `predator.py`, the current code groups candidates by `agent_kind` then runs a Channel per group. Change it so:

- All candidates from the upstream tier go into ONE trough (the producer→herbivore trough or the herbivore→predator trough).
- Herbivore queries the trough with its hunter_state. Multi-head attention does the routing.
- The kind-specific Channels go away — replaced by a single trough per tier boundary.

Concretely, in `herbivore.py`:

```python
# OLD:
for producer_kind, prey in by_kind.items():
    ch = self.channels[producer_kind]
    out = ch(hs, prey_t)
    ...

# NEW:
out = self.trough.attend(hunter_state, tau=self.tau)
channel_seq = out.output  # already [out_seq_len, hidden]
```

Where does `self.trough` come from? It's passed in by the runner — agents no longer own Channel instances; the trough is shared per tier boundary.

### Step 4: Update tests

- Existing herbivore/predator tests need their fixtures updated: instead of building Channels per-kind, they build one TroughAttention per tier.
- `test_multi_head_niches.py`:
  - `test_each_head_attends_to_different_slots`: deposit 4 broadcasts, run many attends with varied queries, assert that `head_specialization` argmax is not constant.
  - `test_diet_tag_unused`: feed in broadcasts with mixed diet_tags, assert the trough doesn't read the tags (they survive on the broadcast for debug, but slot routing is purely Q·K).

---

## Acceptance Criteria

- [ ] Multi-head attention produces correct shapes and gradients
- [ ] `slot_specialization()` returns sensible per-slot head assignments after 100 ticks of mock SFT
- [ ] Herbivore + predator no longer branch on `agent_kind` for routing
- [ ] All 9 existing tests pass (with updated fixtures)
- [ ] At least 4 new tests in `test_multi_head_niches.py` pass

---

## Testing Conditions (exit verification)

1. **Shape + gradient sanity**
   ```python
   t = TroughAttention(hidden_size=64, n_slots=4, n_heads=4, seed=1)
   t.deposit([fake_broadcast() for _ in range(4)])
   q = torch.randn(64, requires_grad=True)
   out = t.attend(q)
   loss = out.output.sum()
   loss.backward()
   assert q.grad is not None and q.grad.norm() > 0
   assert out.per_head_attention.shape == (4, 4)
   ```
   **Expected**: assertions hold.

2. **New tests pass**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && python -m pytest tests/test_multi_head_niches.py -v 2>&1 | tail -15
   ```
   **Expected**: ≥4 passed, 0 failed.

3. **Whole suite green**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && python -m pytest tests/ -x -v 2>&1 | tail -30
   ```
   **Expected**: all pass (existing fixtures may need updating).

4. **Specialization emerges from SFT**
   Run mock SFT for 100+ steps and inspect `slot_specialization()`:
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     TROPHIC_MOCK=1 timeout 600 python scripts/run_sft.py --steps 100 \
     --debug-trough-spec 2>&1 | tail -50
   ```
   **Expected**: `head_per_slot` is not constant across slots — at least 2 distinct head ids appear, indicating heads have specialized.

5. **`agent_kind` branching deleted**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     grep -n "agent_kind" trophic/agents/herbivore.py trophic/agents/predator.py
   ```
   **Expected**: no routing-related matches; `agent_kind` survives only for logging/debug.

If any condition fails, post `@all` MESSAGES; do not mark DONE.

---

## Coordination

- Modifying `trough_attention.py` alongside `phase2-B:02` and `phase2-D:05`. **Read MESSAGES before starting.** If `phase2-B:02:DONE`, build on `step_lifecycle()` rather than competing with it.
- Keep `attend()` signature compatible with INTERFACE CONTRACTS in scratchpad. Augment the dataclass fields rather than renaming.
- This mission deletes a meaningful chunk of agent code (per-kind Channel branching). Update ALL callers — `herbivore.py`, `predator.py`, eval scripts.

---

## When Done

1. Re-run inbox grep, address pending.
2. Update STATUS:
   - `phase2-C:03:RUNNING` → `phase2-C:03:DONE`
3. Append MESSAGES:
   ```
   [<ts>] phase2-C > @all: multi-head trough live. Specialization signal: <e.g. "head 3→disclosure, head 7→anomaly">.
     Diet-tag routing removed from herbivore.py + predator.py.
   ```
