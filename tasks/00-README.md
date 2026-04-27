# TROUGH_AS_TRANSFORMER: Migrate SubstratePool into Cross-Attention

**Project Code**: TROUGH_AS_TRANSFORMER
**Generated**: 2026-04-26
**Status**: NOT STARTED
**Repository**: `/home/dgonier/ecology_experiment/trophic`

---

## Git Operations Warning

**IMPORTANT**: Do not commit until all missions are complete. The user handles git operations.

---

## Mission Statement

The current trophic system has two parallel abstractions for inter-tier communication:
1. **SubstratePool** (SQLite-backed broadcast pool with claim semantics, diet-tag SQL filtering, rotted-flag bookkeeping)
2. **Channel** (cross-attention modules with Q/K/V projections + null gate, owned by each consumer)

These are conceptually one thing — the trough at a tier boundary IS the cross-attention K/V matrix the consumer attends over. Currently we maintain both, with the pool acting as a persistence layer that feeds entries into the Channel. This project collapses them.

The architectural specification lives at `trophic/docs/consumption_transformers.md`. This project implements that specification.

---

## Architecture Decision

**Replace SubstratePool with TroughAttention** — a stateful K/V container per tier boundary, with:
- Standard cross-attention mechanics (Q·K → softmax → weighted V)
- Ecological lifecycle on K/V slot membership (birth, attention decay, death, niche-aware respawning)
- Multi-head specialization that subsumes hardcoded `diet_tags`
- Decomposer feedback as attention bias rather than a separate table

The current `Channel` class becomes the *attention computation* over the trough; the trough is the K/V matrix the channel reads from. Same math, different lifecycle on the entries.

---

## Why Now

The trophic system has reached a state where:
- SFT, IPO, GRPO trainers all work end-to-end
- Eval-CE bottoms around 0.135 with the 4-channel system
- Predator output has structural correctness but no ticker grounding
- Best-judge-score 0.543 (IPO v3); best-rule-based-reward 0.388

The pattern across these results: **information IS flowing through the channels** (otherwise SFT couldn't drop CE). The bottleneck is in *consumer-side* dynamics — which K slots get attended to, how attention weights persist across ticks, how losers are replaced. Those are exactly the additions the trough-as-transformer specifies.

---

## Target Files

```
trophic/trophic/
├── trough_attention.py        # NEW: TroughAttention K/V container + cross-attn
├── substrate.py                # MODIFIED: shim that delegates to TroughAttention
├── channel.py                  # MODIFIED: drop K/V state (now in trough)
├── agents/
│   ├── herbivore.py            # MODIFIED: query trough instead of pool
│   ├── predator.py             # MODIFIED: query trough instead of pool
│   └── ...
├── runner.py                   # MODIFIED: wire troughs at tier boundaries
└── ecology/
    └── slot_lifecycle.py       # NEW: birth/death/respawn for K/V slots
```

Tests under `trophic/tests/`:
```
test_trough_attention.py        # NEW: K/V mechanics, multi-head, decay
test_slot_lifecycle.py          # NEW: birth/death/respawn correctness
```

---

## Missions Overview

| File | Phase | Agent | Focus | Dependencies |
|------|-------|-------|-------|--------------|
| `phase1-A-01-trough-attention.md` | 1 | A | Build TroughAttention class; SubstratePool delegates to it | None |
| `phase2-B-02-attention-decay.md` | 2 | B | Per-K cumulative attention tracking; death threshold ε | 01 |
| `phase2-C-03-multi-head-niches.md` | 2 | C | Replace SQL diet_tags with learned head specialization | 01 |
| `phase2-A-04-slot-reallocation.md` | 2 | A | Niche-aware spawning when K slots die | 01, 02 |
| `phase2-D-05-decomposer-bias.md` | 2 | D | Decomposer output → attention bias vector | 01 |
| `phase3-A-06-skip-connections.md` | 3 | A | Predator attends directly to producer trough; τ schedule | 01-05 |

**Naming convention**: `phase<N>-<agent>-<mission#>-<slug>.md` so `ls` reveals phase ordering, parallelism, and ownership at a glance. Sibling check: `ls phase2-*` shows all phase-2 parallel work; `ls phase*-A-*` shows agent A's full queue.

**Parallelization**: After phase1 lands, phase2-B/C/D run concurrently; phase2-A waits on phase2-B. Phase 3 is sequential polish.

---

## Mission Status

| Phase | Agent | Mission | Status | Completed By | Date |
|-------|-------|---------|--------|--------------|------|
| 1 | A | 01-trough-attention | COMPLETE | phase1-A | 2026-04-26 |
| 2 | B | 02-attention-decay | COMPLETE | phase2-B | 2026-04-26 |
| 2 | C | 03-multi-head-niches | COMPLETE | phase2-C | 2026-04-26 |
| 2 | A | 04-slot-reallocation | COMPLETE | phase2-A | 2026-04-26 |
| 2 | D | 05-decomposer-bias | COMPLETE | phase2-D | 2026-04-26 |
| 3 | A | 06-skip-connections | COMPLETE | phase3-A | 2026-04-26 |

---

## Success Criteria

- [ ] All existing tests still pass (9/9)
- [ ] New tests for TroughAttention pass
- [ ] SubstratePool external API still works (claim, query, etc.) but delegates to TroughAttention
- [ ] One end-to-end mock-mode SFT run succeeds with the new architecture
- [ ] Trough metrics observable: per-K cumulative attention, death events, slot respawns, per-head specialization
- [ ] Eval CE on the v4 scenario library does not regress vs current 0.135 baseline

---

## Quick Start for Agents

1. Read this README completely
2. Read `trophic/docs/consumption_transformers.md` (the architectural spec)
3. Read `tasks/scratchpad.md` end-to-end (STATUS, PHASE MAP, INBOX PROTOCOL, MESSAGES)
4. Identify your handle (e.g., `phase2-B`) and the matching mission file
5. Run inbox check: `grep -nE '@(all|phase2-B)' tasks/scratchpad.md`
6. Atomically update scratchpad STATUS: `phase2-B:02:PENDING` → `phase2-B:02:RUNNING`
7. Read your mission file completely. Verify dependencies (other phase-N entries with status `DONE`).
8. Execute the mission, including all acceptance criteria and tests
9. Re-run inbox check before completion
10. Update scratchpad: `phase2-B:02:RUNNING` → `phase2-B:02:DONE`; append a MESSAGES entry summarizing what landed
11. Do NOT commit — coordinator handles git

---

## Agent Naming and Handles

Each agent has a phase-scoped handle: `phase<N>-<letter>`. The same letter can recur across phases (e.g., `phase1-A` and `phase2-A` are both run by agent "a", but the handle distinguishes the mission).

| Letter | Phase 1 handle | Phase 2 handle | Phase 3 handle |
|--------|----------------|-----------------|-----------------|
| A | phase1-A → 01-trough-attention | phase2-A → 04-slot-reallocation | phase3-A → 06-skip-connections |
| B | -              | phase2-B → 02-attention-decay   | -               |
| C | -              | phase2-C → 03-multi-head-niches | -               |
| D | -              | phase2-D → 05-decomposer-bias   | -               |

**Inbox addressing** (used in scratchpad MESSAGES — see scratchpad INBOX PROTOCOL):
- `@all` — broadcast
- `@<handle>` — single agent (e.g., `@phase2-C`)
- `@phase2` — every agent currently in phase 2
- `@phase2-A,phase2-C` — multiple specific agents
