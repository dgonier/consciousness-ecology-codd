# ENVSTREAM: Build EnvironmentStream — input-layer K/V substrate

**Project Code**: ENVSTREAM
**GitHub Issue**: dgonier/consciousness-ecology-codd#9
**Generated**: 2026-04-27
**Status**: NOT STARTED
**Repository**: `/home/dgonier/ecology_experiment/trophic`
**Tasks dir**: `tasks_envstream/` (this directory; the older `tasks/` holds the trough-as-transformer migration history)

---

## Git Operations Warning

**IMPORTANT**: Do not commit. The user handles git.

---

## Mission Statement

The architecture has TroughAttention at every tier boundary except the input layer. Below the producer tier, it's a heterogeneous mess: ad-hoc loader scripts, format-coupled producer code, scenario-builder logic conflating adapter and producer roles. This project introduces three layered abstractions that bring **input-layer uniformity** to the architecture:

1. **Adapters** — single-responsibility format converters (one wavelength of light each)
2. **EnvironmentStream** — input-layer K/V substrate, parallel to TroughAttention at higher tiers
3. **Wavelength-based producer subscription** — many-to-many adapter↔producer routing

After this project: the architecture is uniform from external world → producer → herbivore → predator → apex, with K/V + cross-attention + lifecycle at every tier boundary.

The architectural specification is GitHub issue #9 (`gh issue view 9 --repo dgonier/consciousness-ecology-codd`). The bug that motivated this is #8 (direction collapse on StockNet).

---

## Why Now

We're trying to break the direction-collapse described in #8. Diagnostics ruled out training-distribution prior, IPO sampling bias, and (likely) decode greediness. The remaining hypothesis: **synthetic scenarios deliver one input per scenario through a narrow pipe, never forcing the producer/herbivore tiers to actually discriminate signal**. EnvironmentStream is the architectural fix that lets us train against multi-input streams (StockNet adapters provide that) with the right uniformity guarantees.

---

## Target Files

```
trophic/
├── adapters/                       # NEW: format converters
│   ├── __init__.py
│   ├── base.py                     # NEW: Adapter ABC + wavelength registry
│   └── stocknet/
│       ├── __init__.py
│       ├── ohlcv_normalized.py     # NEW: StockNet preprocessed price → RawInput
│       └── tokenized_tweet.py      # NEW: StockNet preprocessed tweets → RawInput
├── environment_stream.py           # NEW: EnvironmentStream class
├── types.py                        # MODIFIED: SOURCE_TAGS controlled vocabulary
├── agents/
│   ├── producer.py                 # MODIFIED: WAVELENGTHS class attr replaces ATTRACTION dict
│   ├── quant_producer.py           # MODIFIED: same
│   └── social_signal.py            # NEW: SocialSignal producer for tweets
├── runner.py                       # MODIFIED (in phase 3 only): wire EnvStream into tick loop
└── training/
    └── stocknet_loader.py          # MODIFIED: refactor format-conversion into adapters

tests/
├── test_adapters.py                # NEW: adapter unit tests
├── test_environment_stream.py      # NEW: deposit / attend / lifecycle / wavelength filter
├── test_stocknet_adapters.py       # NEW: per-adapter integration tests
└── test_producer_wavelengths.py    # NEW: producer wavelength-attraction tests
```

---

## Missions Overview

| File | Phase | Agent | Focus | Dependencies |
|------|-------|-------|-------|--------------|
| `phase1-A-01-adapter-layer.md` | 1 | A | Adapter ABC + wavelength registry + refactor existing format code | None |
| `phase2-B-02-environment-stream.md` | 2 | B | EnvironmentStream class with deposit/attend/lifecycle | 01 |
| `phase2-C-03-stocknet-adapters.md` | 2 | C | StockNet OHLCV + tweet adapters concrete impls | 01 |
| `phase2-D-04-social-signal-producer.md` | 2 | D | SocialSignal producer + refactor existing producers to WAVELENGTHS class attr | 01 |
| `phase3-A-05-integrate-and-train.md` | 3 | A | Wire EnvStream into runner; train SFT+IPO seed 9; eval cascade | 02, 03, 04 |

**Naming convention**: `phase<N>-<agent>-<mission#>-<slug>.md`. `ls phase2-*` shows all phase-2 parallel work; `ls phase*-A-*` shows agent A's full queue.

**Parallelization**: Phase 2 fans out to B/C/D after phase 1 lands. Phase 3 runs alone after all of phase 2 is DONE.

---

## Mission Status

| Phase | Agent | Mission | Status | Completed By | Date |
|-------|-------|---------|--------|--------------|------|
| 1 | A | 01-adapter-layer | COMPLETE | phase1-A | 2026-04-26 |
| 2 | B | 02-environment-stream | COMPLETE | phase2-B | 2026-04-26 |
| 2 | C | 03-stocknet-adapters | COMPLETE | phase2-C | 2026-04-27 |
| 2 | D | 04-social-signal-producer | COMPLETE | phase2-D | 2026-04-26 |
| 3 | A | 05-integrate-and-train | COMPLETE | phase3-A | 2026-04-27 |

---

## Success Criteria

- [ ] Existing 51 tests still pass post-refactor
- [ ] New tests for Adapter ABC pass (≥4)
- [ ] New tests for EnvironmentStream pass (≥6)
- [ ] New tests for StockNet adapters pass (≥4)
- [ ] New tests for producer wavelengths pass (≥4)
- [ ] `runner.py` wires EnvStream successfully; mock-mode SFT smoke completes
- [ ] Real training (SFT seed 9 + IPO seed 9) completes; eval cascade reports a number
- [ ] **Decision criterion**: StockNet test MCC for ipo_seed9_best.pt moves off zero

---

## Quick Start for Agents

1. Read this README completely
2. Read `gh issue view 9 --repo dgonier/consciousness-ecology-codd` for the architectural spec
3. Read `docs/consumption_transformers.md` for prior architectural context
4. Read `tasks_envstream/scratchpad.md` end-to-end (STATUS, INBOX PROTOCOL, INTERFACE CONTRACTS, MESSAGES)
5. Identify your handle (e.g., `phase2-B`) and the matching mission file
6. Run inbox grep: `grep -nE '@(all|<your-handle>|phase<N>)' tasks_envstream/scratchpad.md`
7. Atomically flip STATUS: `phase2-B:02:PENDING` → `phase2-B:02:RUNNING`
8. Read your mission file completely. Verify dependencies are DONE.
9. Execute the mission, including all acceptance criteria and Testing Conditions
10. Re-run inbox grep before completion
11. Update scratchpad: `phase2-B:02:RUNNING` → `phase2-B:02:DONE`; append a MESSAGES entry
12. Do NOT commit — coordinator handles git

---

## Agent Naming and Handles

| Letter | Phase 1 handle | Phase 2 handle | Phase 3 handle |
|--------|----------------|-----------------|-----------------|
| A | phase1-A → 01-adapter-layer | -              | phase3-A → 05-integrate-and-train |
| B | -              | phase2-B → 02-environment-stream | -                |
| C | -              | phase2-C → 03-stocknet-adapters  | -                |
| D | -              | phase2-D → 04-social-signal-producer | -            |

**Inbox addressing** (used in scratchpad MESSAGES — see scratchpad INBOX PROTOCOL):
- `@all` — broadcast
- `@<handle>` — single agent (e.g., `@phase2-C`)
- `@phase2` — every agent currently in phase 2
- `@phase2-A,phase2-C` — multiple specific agents
