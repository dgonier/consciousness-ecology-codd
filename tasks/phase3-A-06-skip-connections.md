# Mission 06: skip-connections

**Handle**: `phase3-A`
**Phase**: 3 (sequential, runs alone after all of phase 2)
**Mission file**: `phase3-A-06-skip-connections.md`
**Dependencies**: `phase1-A:01:DONE`, `phase2-A:04:DONE`, `phase2-B:02:DONE`, `phase2-C:03:DONE`, `phase2-D:05:DONE`
**Blocks**: nothing (final integration)

---

## Before You Start

```bash
cd /home/dgonier/ecology_experiment && \
  grep -nE '@(all|phase3-A|phase3)' tasks/scratchpad.md
```

Confirm ALL of phase 1 + phase 2 are DONE. Then atomically flip:
```
phase3-A:06:PENDING → phase3-A:06:RUNNING
```

---

## Goal

Polish the integration with two finishing touches from the architectural spec:

1. **Cross-tier skip connections**: the predator (and apex) attend not only to the immediately upstream tier but also directly to the producer trough, with a learnable skip weight α. This is residual connection across the trophic chain — fixes deep-attention gradient flow and is diagnostically useful (high α = intermediate tier is bottlenecking; low α = intermediate is adding value).

2. **Temperature annealing schedule** on the softmax τ: warm at start (egalitarian), cool over training (sharp selection). Ecologically: explore early, exploit late.

After this, the migration is done. The trough-as-transformer architecture is fully implemented.

---

## Files to Modify

1. `trophic/trophic/agents/predator.py` — pull skip connection from producer trough
2. `trophic/trophic/runner.py` — temperature schedule wiring
3. `trophic/trophic/training/sft.py` (and grpo.py, ipo.py) — pass τ schedule to runner
4. `trophic/trophic/trough_attention.py` — accept τ as a runtime arg if not already

## Files to Create

5. `trophic/trophic/training/tau_schedule.py` — small util for cosine/linear schedule
6. `trophic/tests/test_skip_connections.py`

---

## Implementation Steps

### Step 1: Skip path in Predator

Predator currently attends to herbivore_trough only. Add a parallel attend on producer_trough:

```python
class Predator(BaseAgent):
    def __init__(self, ..., skip_weight_init: float = 0.1):
        ...
        self.skip_weight = nn.Parameter(torch.tensor(skip_weight_init))

    async def hunt_and_predict(
        self,
        herbivore_trough: TroughAttention,
        producer_trough: TroughAttention,
        tick: int,
        host: ModelHost,
        tau: float = 1.0,
    ):
        hunter_state = self.role_prefix.mean(dim=0)

        herb_out = herbivore_trough.attend(hunter_state, tau=tau)
        prod_out = producer_trough.attend(hunter_state, tau=tau)

        # Residual: combine with learnable α (clipped to [0, 1] via sigmoid).
        alpha = torch.sigmoid(self.skip_weight)
        combined_seq = (1 - alpha) * herb_out.output + alpha * prod_out.output
        ...
```

Sigmoid keeps α ∈ (0, 1). At init it's ~0.525 — small enough that the herbivore tier still dominates, but the gradient through the skip is non-zero from the start.

### Step 2: τ schedule

`tau_schedule.py`:

```python
import math

def cosine_tau(step: int, total_steps: int, tau_start: float = 2.0, tau_end: float = 0.5) -> float:
    """Cosine anneal from tau_start to tau_end."""
    if total_steps <= 0:
        return tau_end
    progress = min(step, total_steps) / total_steps
    return tau_end + 0.5 * (tau_start - tau_end) * (1 + math.cos(math.pi * progress))


def linear_tau(step: int, total_steps: int, tau_start: float = 2.0, tau_end: float = 0.5) -> float:
    if total_steps <= 0:
        return tau_end
    progress = min(step, total_steps) / total_steps
    return tau_start + (tau_end - tau_start) * progress
```

Wire into runner: each training script computes `tau = cosine_tau(step, total_steps)` and passes it to the runner; runner passes it to every `attend()` call this tick.

### Step 3: Tests

- `test_skip_weight_initializes_low`: sigmoid(0.1) ≈ 0.525 — assert it's in [0.4, 0.6].
- `test_skip_weight_gradient_nonzero`: backprop on a synthetic loss; assert `skip_weight.grad` is non-None and non-zero.
- `test_predator_combines_skip_and_main`: deposit producer-tier broadcasts that match Q strongly, herbivore-tier broadcasts that don't; with α=0.9, predator output should resemble the producer-tier output.
- `test_cosine_tau_monotone`: cosine schedule monotonically decreasing.
- `test_tau_passed_into_attend`: mock the trough; verify it received the right τ.

### Step 4: Diagnostic logging

Per epoch, log:
- mean skip weight (sigmoid(skip_weight))
- current τ
- per-trough alive/killed/spawned counts (from Mission 02/04)
- per-head specialization entropy (from Mission 03 — high entropy means heads are interchangeable; low entropy means real specialization)

These together give a full ecology-state snapshot per epoch. Useful for diagnosing whether the migration is actually helping.

---

## Acceptance Criteria

- [ ] One end-to-end mock-mode SFT run completes
- [ ] Eval CE on the v4 scenario library does not regress vs current 0.135 baseline
- [ ] Skip connection α gradients flow during training
- [ ] τ schedule wired and visible in logs
- [ ] Per-tick ecology snapshot logs are present
- [ ] All tests pass

---

## Testing Conditions (exit verification)

1. **Skip + tau tests pass**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && python -m pytest tests/test_skip_connections.py -v 2>&1 | tail -15
   ```
   **Expected**: ≥5 passed, 0 failed.

2. **Whole suite green**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && python -m pytest tests/ -x -v 2>&1 | tail -30
   ```
   **Expected**: all tests pass.

3. **End-to-end mock SFT**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     TROPHIC_MOCK=1 timeout 1200 python scripts/run_sft.py --steps 200 2>&1 | tail -50
   ```
   **Expected**: exit 0, no exceptions, loss decreasing, per-tick ecology snapshot lines visible.

4. **Eval CE not regressed**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     timeout 600 python scripts/run_eval_ce.py 2>&1 | tail -10
   ```
   **Expected**: reported CE ≤ 0.135 (the existing 4-channel baseline), or post a MESSAGES note explaining why a small regression is acceptable.

5. **Skip α gradient and value sanity**
   Smoke script after a brief training run: assert `predator.skip_weight.grad` is non-None and that `sigmoid(skip_weight)` lies in (0, 1).

6. **τ schedule observable**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     TROPHIC_MOCK=1 timeout 600 python scripts/run_sft.py --steps 100 2>&1 \
     | grep "tau=" | awk '{print $NF}' | sort -u | head
   ```
   **Expected**: at least 5 distinct τ values printed (proves the schedule actually anneals).

If any condition fails, post `@all` in MESSAGES; do not mark DONE.

---

## Coordination

- Final mission. Verify all of `phase1-A:01`, `phase2-A:04`, `phase2-B:02`, `phase2-C:03`, `phase2-D:05` are DONE before starting.
- Integration sanity check: re-run all mock-mode training scripts and confirm nothing crashes. Real-data training is out of scope for this project — that's a follow-up.

---

## When Done

1. Re-run inbox grep, address pending.
2. Update STATUS:
   - `phase3-A:06:RUNNING` → `phase3-A:06:DONE`
3. Update `00-README.md` Mission Status table to mark all six as COMPLETE.
4. Append MESSAGES:
   ```
   [<ts>] phase3-A > @all: TROUGH_AS_TRANSFORMER complete.
     Final eval CE: <v> (baseline 0.135).
     Final mean skip weight (sigmoid): <v>.
     Multi-head specialization entropy: <v> (low = strong specialization).
     Population dynamics: mean killed/spawned per tick = <v>/<v>.
     Ready for git review.
   ```
