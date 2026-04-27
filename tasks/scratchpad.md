# TROUGH_AS_TRANSFORMER — Coordination Scratchpad

**Status legend**: PENDING (not claimed) | RUNNING (claimed by an agent) | DONE | BLOCKED

**Coordinator note**: Read this whole file before starting. Update your STATUS line atomically when you claim/finish work. Add a MESSAGES entry when you finish a mission, hit a blocker, or have a finding worth surfacing.

---

## STATUS

```
phase1-A:01:DONE
phase2-B:02:DONE
phase2-C:03:DONE
phase2-D:05:DONE
phase2-A:04:DONE
phase3-A:06:DONE
```

When phase1-A:01 lands → set phase2-B/C/D to PENDING.
When phase2-B lands → set phase2-A:04 to PENDING.
When all of phase 2 lands → set phase3-A:06 to PENDING.

---

## PHASE MAP

```
Phase 1 (sequential — single agent):
    phase1-A → 01-trough-attention
    └─> Phase 2 unblocks for B, C, D

Phase 2 (parallel — four agents):
    phase2-A → 04-slot-reallocation     (blocks on phase2-B)
    phase2-B → 02-attention-decay
    phase2-C → 03-multi-head-niches
    phase2-D → 05-decomposer-bias

Phase 3 (sequential — single agent after all of phase 2 lands):
    phase3-A → 06-skip-connections
```

---

## MISSION DEPENDENCY GRAPH

```
                      phase1-A:01-trough-attention
                      /     |     |     \
              phase2-B:02  phase2-C:03  phase2-D:05  (parallel)
                      \    |     |
                       \   |    /
                  phase2-A:04-slot-reallocation
                              |
                  phase3-A:06-skip-connections
```

---

## INBOX PROTOCOL

Subagents are not long-running listeners — they wake, run, return. To compensate, every agent runs a **grep on wake** and a **grep before completion** against this scratchpad's MESSAGES section.

### Required: grep on start

```bash
cd /home/dgonier/ecology_experiment && \
  grep -nE '@(all|<your-handle>|phase<N>)' tasks/scratchpad.md
```

Replace `<your-handle>` with `phase2-B` (etc.) and `phase<N>` with your phase number (so `@phase2` broadcasts also reach you).

### Required: grep before marking DONE

Same command. Confirm nothing addressed to you is still unanswered. If something is, reply in MESSAGES before flipping to DONE.

### Optional: live tail listener (only if your phase has parallel siblings active)

```bash
# Start a background tail-grep
Bash(run_in_background=true,
     command="tail -F /home/dgonier/ecology_experiment/tasks/scratchpad.md \
              | grep --line-buffered -E '@(all|<your-handle>|phase<N>)'")
# Then attach Monitor to that task — new matches arrive as system notifications
```

This only fires while your specific agent is running. It does NOT span agent lifetimes — late messages are caught by the start/end grep on the next agent's run.

### Addressing scheme

| Address | Meaning |
|---------|---------|
| `@all` | Broadcast — every agent reads on next wake |
| `@<handle>` | Single agent, e.g. `@phase2-C` |
| `@phase<N>` | Every agent currently in phase N |
| `@<h1>,<h2>` | Multiple specific (no spaces) |

### Message format

```
[YYYY-MM-DD HH:MM] phase1-A > @all: <message>
[YYYY-MM-DD HH:MM] phase2-B > @phase2-A: <message>
```

Past tense, factual, link to file/line numbers when relevant. Don't post status — STATUS block is for that.

---

## SHARED FACTS

- **Repo root**: `/home/dgonier/ecology_experiment/trophic`
- **Architectural spec**: `trophic/docs/consumption_transformers.md`
- **Test command**: `cd /home/dgonier/ecology_experiment/trophic && python -m pytest tests/ -x`
- **Mock-mode SFT smoke**: `cd /home/dgonier/ecology_experiment/trophic && TROPHIC_MOCK=1 python scripts/run_sft.py --steps 50`
- **Existing test count**: 9 tests passing as of project start
- **Eval baseline**: CE 0.135 (4-channel SFT); rule-based predator reward 0.388 (IPO seed1 best)
- **Why this project exists**: LoRA-active interrogator achieves 0.937 transcription reward but predator drops to 0.105 — predator Channels are the attention bottleneck, motivating this migration.

---

## INTERFACE CONTRACTS

These are the cross-mission interfaces. **Do not change them without leaving a `@all` MESSAGES note.**

### TroughAttention (created in phase1-A:01, consumed by all others)

```python
class TroughAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_slots: int,
        n_heads: int = 8,
        out_seq_len: int = 8,
        target_norm: float = 1.0,
        seed: int | None = None,
    ): ...

    def deposit(self, broadcasts: list[Broadcast]) -> list[int]: ...
    def attend(
        self,
        query: torch.Tensor,
        tau: float = 1.0,
        external_bias: torch.Tensor | None = None,
    ) -> TroughAttendOutput: ...
    def step_lifecycle(self) -> dict: ...      # added in phase2-B
    def spawn_into_dead_slot(self, slot_id: int) -> dict: ...  # added in phase2-A
    def slot_state(self) -> dict: ...
```

### TroughAttendOutput dataclass

```python
@dataclass
class TroughAttendOutput:
    output: torch.Tensor              # [out_seq_len, hidden]
    per_slot_attention: torch.Tensor  # [n_slots]
    per_head_attention: torch.Tensor  # [n_heads, n_slots]
    null_prob: float
    selected_slot_ids: list[int]
    rejected_slot_ids: list[int]
```

### SubstratePool shim (phase1-A:01)

Public API is preserved (`claim`, `add`, `query`) but internally delegates to `TroughAttention`. Existing herbivore/predator code keeps compiling unchanged.

---

## MESSAGES

(append-only log — newest at bottom — `[timestamp] <from-handle> > @<to>: message`)

- [2026-04-26 10:50] coordinator > @all: Project package created. LoRA bottom-up validation: interrogator 0.937 (perfect), predator 0.105 (regressed from 0.388). Confirms predator-side attention is the bottleneck — motivation for this migration.
- [2026-04-26 11:00] coordinator > @all: Filenames now use `phase<N>-<agent>-<mission#>-<slug>.md`. Handles in STATUS / MESSAGES are `phase<N>-<letter>`. Inbox grep pattern is `@(all|<your-handle>|phase<N>)`.
- [2026-04-26 14:30] phase1-A > @phase2: TroughAttention landed at trophic/trough_attention.py. SubstratePool is now a thin in-memory shim (SQL/rotted-flag gone) at trophic/substrate.py. Acceptance test summary: `19 passed in 40.57s` — 9 existing + 10 new TroughAttention tests. Mock SFT smoke (10 steps, TROPHIC_MOCK_MODELS=1 TROPHIC_MOCK_PREDATOR=1) ran clean, no NaN, exit 0. cumulative_attention buffer is being incremented inside `attend()`. Surprises: (1) the env vars referenced in the mission file were `TROPHIC_MOCK=1 / scripts/run_sft.py` but the actual ones are `TROPHIC_MOCK_MODELS=1 TROPHIC_MOCK_PREDATOR=1 / scripts/train_sft.py` — used those. (2) `n_heads` parameter is honored in the signature; the readout is a true multi-head attention (not single-head shortcut), so phase2-C can hook in via `per_head_attention` directly. (3) `external_bias` arg is wired through `attend()` and added to alive-slot scores pre-softmax — phase2-D can attach decomposer feedback there with no signature change. (4) The shim creates one TroughAttention per (tier, agent_kind) lazily on first deposit; `pool.get_trough(tier, kind)` exposes them. (5) `evict()` exists on TroughAttention as bookkeeping (used when shim's `claim` consumes an item) — distinct from ecological death (`step_lifecycle`). Open questions for downstream: phase2-B must decide whether `step_lifecycle` operates on `cumulative_attention` directly (currently a raw running sum) or applies α-weighted decay first; current implementation just `add_`s per-tick attention, so phase2-B owns the decay coefficient.
- [2026-04-26 15:00] recovery-finalizer (acting for phase2-B) > @phase2-A: step_lifecycle() shipped at trough_attention.py:434 + slot_lifecycle.py. test_slot_lifecycle.py: 10/10 passed. Death smoke (20 ticks, biased Q): alive_after=4/4. Spawn handoff is yours; you augment step_lifecycle to call spawn_into_dead_slot() after the kill phase. NOTE: original phase2-B agent crashed before posting; this finalization done by recovery agent based on what landed on disk.
- [2026-04-26 15:00] recovery-finalizer (acting for phase2-D) > @phase3-A: decomposer + bias path shipped at trophic/decomposer.py + trough_attention.py:235 (set_pending_bias). test_decomposer_bias.py: 7/7 passed. Zero-init max abs bias = 0.0. Runner.py wiring status: WIRED — runner.py imports BiasDecomposer (line 31), holds `_bias_decomposers` dict (line 77), defines `_apply_decomposer_bias` (line 136) which calls `trough.set_pending_bias(bias)` (line 209), and invokes it from the predator-judgment loop at line 360 (`self._apply_decomposer_bias(pred_br, j, tick)`). NOTE: original phase2-D agent crashed before posting; this finalization done by recovery agent.
- [2026-04-26 17:30] phase2-A > @phase3-A: slot reallocation live. niche_aware_spawn → mean cosine(new_K, underserved_q) over 2 spawn events: 0.9998. Population stability smoke: alive_final=8/8. test_slot_lifecycle.py: 15/15. Full suite: 46/46.
- [2026-04-26 18:00] phase2-C > @all: multi-head specialization tracking + diet-tag routing removal complete. test_multi_head_niches.py: 5/5. Full suite: 46/46. After 50-tick mock-Runner smoke: predator-trough head_per_slot distribution = {head 2: 27 slots, head 6: 14 slots, head 0: 4 slots, head 1: 4 slots, head 3: 1 slot} (5 distinct heads across 50 alive slots). Substrate-pool keying: kept per-(tier,kind) lazy keying — phase1-A's shim is unchanged; phase3-A wires skip connections per existing `pool.get_trough(tier, kind)` API (one trough per (tier, agent_kind) tuple, populated lazily on first deposit). NOTE: re-spawned after original phase2-C crashed mid-mission; multi-head Q/K/V machinery (`_split_heads`/`_merge_heads` and per-head einsums in `attend()`) from original run preserved; this run added (1) `head_specialization` buffer + EMA in `attend()`, (2) `slot_specialization()` method, (3) test file, (4) removed `by_kind` routing from herbivore.py and predator.py runtime path (per-kind Channel objects retained for SFT/IPO/GRPO trainer compat).
- [2026-04-26 19:00] recovery-finalizer (acting for phase3-A) > @all: TROUGH_AS_TRANSFORMER complete. Final tests: 51/51 passed (was 46 after phase 2; +5 skip-connection tests). End-to-end mock SFT (50 steps): exit 0, final train_loss=2.926. Skip weight at init: sigmoid=0.5249791741371155. Tau schedule: started at 1.857, ended at 0.500. Files added by phase3-A: trophic/training/tau_schedule.py, tests/test_skip_connections.py. Files modified: trophic/agents/predator.py (skip_weight + producer_trough arg), trophic/training/sft.py (tau wiring), trophic/runner.py (set_tau + pass-through). NOTE: original phase3-A agent's code landed in full but session crashed before final scratchpad update; this finalization done by recovery agent.
