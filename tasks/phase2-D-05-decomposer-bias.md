# Mission 05: decomposer-bias

**Handle**: `phase2-D`
**Phase**: 2 (parallel after `phase1-A:01`)
**Mission file**: `phase2-D-05-decomposer-bias.md`
**Dependencies**: `phase1-A:01:DONE`
**Blocks**: nothing (independent path)

---

## Before You Start

```bash
cd /home/dgonier/ecology_experiment && \
  grep -nE '@(all|phase2-D|phase2)' tasks/scratchpad.md
```

Confirm `phase1-A:01:DONE`. Then atomically flip:
```
phase2-D:05:PENDING → phase2-D:05:RUNNING
```

---

## Goal

The current architecture has a vague "decomposer" concept — a thing that traces apex-judge feedback back to which producers contributed to good vs. bad predictions. The trough-as-transformer spec collapses it into an **attention bias vector** added to Q·K scores. This mission builds that bias path.

After this mission, when the apex judges a prediction, a small decomposer model produces a per-slot bias that nudges the next tick's attention away from slots that contributed to bad predictions and toward slots that contributed to good ones.

---

## Files to Create

1. `trophic/trophic/decomposer.py` — small MLP that maps (apex_judgment, slot_lineage) → per-slot bias
2. `trophic/tests/test_decomposer_bias.py`

## Files to Modify

3. `trophic/trophic/trough_attention.py` — accept optional bias in `attend()`
4. `trophic/trophic/runner.py` — invoke decomposer after apex judgment, pass bias to next tick's attend()

---

## Implementation Steps

### Step 1: `Decomposer` module

```python
import torch
import torch.nn as nn

class Decomposer(nn.Module):
    """Maps (apex_judgment_embedding, slot_lineage_attention) → per-slot bias.

    The bias is added to Q·K logits in the next tick:
        biased_scores = scores + decomposer_bias

    Magnitude is small (clipped), so it nudges rather than dominates.
    """
    def __init__(self, hidden_size: int, n_slots: int, max_bias: float = 2.0):
        super().__init__()
        self.n_slots = n_slots
        self.max_bias = max_bias
        self.judgment_proj = nn.Linear(hidden_size, hidden_size)
        self.lineage_proj = nn.Linear(n_slots, hidden_size)
        self.head = nn.Linear(2 * hidden_size, n_slots)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        apex_judgment_embedding: torch.Tensor,   # [hidden]
        slot_lineage: torch.Tensor,              # [n_slots] — attention weights from when this prediction was made
    ) -> torch.Tensor:
        j = self.judgment_proj(apex_judgment_embedding)
        l = self.lineage_proj(slot_lineage)
        h = torch.cat([j, l], dim=-1)
        bias = self.head(h)
        return torch.tanh(bias) * self.max_bias  # bounded
```

The decomposer is initialized to output zero (so initial training is unaffected); it learns to produce non-zero bias only as gradients flow.

### Step 2: Bias path in `attend()`

```python
def attend(
    self,
    query: torch.Tensor,
    tau: float = 1.0,
    external_bias: torch.Tensor | None = None,  # [n_slots] — from decomposer
) -> TroughAttendOutput:
    ...
    logits = torch.einsum("hd,nhd->hn", Q, K_aug) / (self.head_dim ** 0.5) / tau
    if external_bias is not None:
        # Broadcast bias across heads, exclude null position.
        bias_expanded = torch.zeros(self.n_heads, alive_count + 1, device=logits.device)
        bias_expanded[:, :alive_count] = external_bias[alive_mask].unsqueeze(0)
        logits = logits + bias_expanded
    ...
```

### Step 3: Lineage tracking in runner

Each tick, when the predator emits a broadcast, record:

```python
predator_lineage = {
    "trough": predator_trough,
    "tick": tick,
    "per_slot_attention": last_attend_output.per_slot_attention.detach().clone(),
}
```

When apex judges that prediction, take its judgment embedding (e.g., the parsed reward_total + the decoded text encoded via host) and feed it + the lineage to the decomposer:

```python
bias = decomposer(apex_judgment_emb, predator_lineage["per_slot_attention"])
predator_trough.set_pending_bias(bias)  # used in next attend
```

### Step 4: `set_pending_bias` / consumed in next attend

```python
# In TroughAttention:
def set_pending_bias(self, bias: torch.Tensor) -> None:
    self._pending_bias = bias

def attend(self, query, tau=1.0, external_bias=None):
    if external_bias is None and self._pending_bias is not None:
        external_bias = self._pending_bias
    self._pending_bias = None  # consume once
    ...
```

### Step 5: Tests

- `test_decomposer_zero_init_zero_bias`: fresh Decomposer + random inputs → bias.abs().max() near 0.
- `test_bias_modifies_attention`: deposit 2 slots, attend without bias, then with strong negative bias on slot 0 — verify slot 1 gets more weight.
- `test_bias_clipped_to_max`: feed pathological inputs, verify bias stays in [-max_bias, +max_bias].
- `test_pending_bias_consumed_once`: set bias, attend, attend again — second attend should not use the bias.

---

## Acceptance Criteria

- [ ] Decomposer compiles, has near-zero output at init, runs without affecting baseline behavior
- [ ] Tests pass
- [ ] Mock SFT smoke runs end-to-end with decomposer wired in
- [ ] Logging shows when bias is being applied (debug level fine)

---

## Testing Conditions (exit verification)

1. **Decomposer tests pass**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && python -m pytest tests/test_decomposer_bias.py -v 2>&1 | tail -15
   ```
   **Expected**: ≥4 passed, 0 failed.

2. **Whole suite green, no regression from optional-bias path**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && python -m pytest tests/ -x -v 2>&1 | tail -30
   ```
   **Expected**: all tests pass.

3. **Zero-init means zero bias at start**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && python -c "
   import torch
   from trophic.decomposer import Decomposer
   d = Decomposer(hidden_size=64, n_slots=8)
   bias = d(torch.randn(64), torch.randn(8))
   print('max bias:', bias.abs().max().item())
   assert bias.abs().max() < 1e-4
   print('OK')
   "
   ```
   **Expected**: max bias < 1e-4 at init.

4. **Bias path actually fires under SFT**
   Mock SFT with debug logging, count non-zero bias applications:
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     TROPHIC_MOCK=1 timeout 600 python scripts/run_sft.py --steps 100 \
     --debug-decomposer 2>&1 | grep "bias-applied" | wc -l
   ```
   **Expected**: > 0 after some training steps (bias becomes non-zero as decomposer learns).

5. **`pending_bias` consumed-once semantics**
   Quick check that calling `attend()` twice in a row only uses the bias the first time (test from `test_pending_bias_consumed_once`).

If any condition fails, post `@all` in MESSAGES; do not mark DONE.

---

## Coordination

- Modifying `trough_attention.py` alongside `phase2-B:02` and `phase2-C:03`. Keep changes **additive** — `external_bias` is a new optional kwarg, do not change positional signature.
- Judgment-embedding extraction is fuzzy for now. Use `host.encode_role_prefix(decoded_text)` pooled — that's a placeholder. Future mission can swap in real apex hidden-state.
- Live tail listener (optional):
  ```bash
  tail -F /home/dgonier/ecology_experiment/tasks/scratchpad.md \
    | grep --line-buffered -E '@(all|phase2-D|phase2)'
  ```

---

## When Done

1. Re-run inbox grep, address pending.
2. Update STATUS:
   - `phase2-D:05:RUNNING` → `phase2-D:05:DONE`
3. Append MESSAGES:
   ```
   [<ts>] phase2-D > @phase3-A: decomposer bias path live. Bias-applied count over 100-step smoke: <v>.
     Max bias magnitude observed: <v>. Apex judgment uses placeholder embedding (text→encode_role_prefix→pool).
   ```
