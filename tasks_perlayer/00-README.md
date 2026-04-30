# PER_LAYER_MODULATION — refactor trophic to Hexis-aligned per-layer Q+V hooks

## Project overview

After exhausting #10 paths A (FinCoT LoRA) and B (SocialSignal LoRA) — both StockNet MCC=0 with constant-direction collapse — the Hexis paper (`/home/dgonier/debaterhub/hexis/paper/sections/`) gave us the architectural diagnosis: trophic has been doing prefix-injection (synthesizing channel_output hidden vectors and pasting them into Qwen's input sequence). That is exactly the failure mode Hexis explicitly contrasts against in §2 ("context-level methods... compete for attention with everything else and dilutes with length"). Hexis's discussion section names it: **"constant additive perturbation creates positive feedback through autoregressive generation"** — the M attractor problem. Our `P_ I _ T E R I C H PRO S...` Cyrillic-transliteration gibberish in Channel v2 / trough mode is exactly this.

This swarm executes the corrected design: trophic = federated meta-M of Hexis. The trough is the federation/aggregation primitive (composing M-states across the population); each consumer's interface to its trough should be **per-layer rank-r Q+V modulation hooks**, not prefix injection. d* (frozen contrast direction, post-attention residual) is added as the second orthogonal channel. ORPO loss replaces teacher-forcing-only.

Decision criterion: StockNet MCC > 0.10 = paper-grade (matches Hexis banking_knowledge +20pp); > 0.05 = signal worth keeping; ~0 = root cause is somewhere else and we pivot.

## Phase plan

```
Phase 1 (sequential, agent A): foundation
  └─ phase1-A:01 — phi-MLP + per-layer M hooks + prefill gating

Phase 2 (parallel, B + C + D): three orthogonal extensions
  ├─ phase2-B:02 — d* extraction + post-attention hook
  ├─ phase2-C:03 — ORPO loss replacing teacher_forcing_loss
  └─ phase2-D:04 — consumer rewire (drop forward_with_prefix in hooks mode)

Phase 3 (sequential, agent A): integration
  └─ phase3-A:05 — train SFT seed24 + eval cascade + writeup
```

## Missions

| Handle      | # | Slug                          | Phase | Depends on          | Blocks    |
|-------------|---|-------------------------------|-------|---------------------|-----------|
| phase1-A    | 01 | phi-mlp-and-hooks            | 1     | none                | 02,03,04  |
| phase2-B    | 02 | dstar-extraction             | 2     | 01                  | 05        |
| phase2-C    | 03 | orpo-loss-shape              | 2     | 01                  | 05        |
| phase2-D    | 04 | consumer-rewire              | 2     | 01                  | 05        |
| phase3-A    | 05 | train-and-eval               | 3     | 02, 03, 04          | nothing   |

## Status table

```
phase1-A:01:PENDING
phase2-B:02:BLOCKED  (needs phase1-A:01)
phase2-C:03:BLOCKED  (needs phase1-A:01)
phase2-D:04:BLOCKED  (needs phase1-A:01)
phase3-A:05:BLOCKED  (needs phase2-B,C,D)
```

## Agent assignments

- **A** = foundational primitive (phase 1) and integration (phase 3). Same letter recurs because phase 3 closes the loop on phase 1's contract.
- **B, C, D** = three orthogonal extensions in phase 2. They all depend on the phi-MLP+hooks landing first.

## Success criteria

- `phase1-A:01` done: phi-MLP forward returns per-layer modulation tensors of correct shape; install_M_hooks returns handles; with M_tensors=zeros the model output matches no-hook baseline exactly; with M_tensors=non-zero, the output is measurably different. Tests pass.
- `phase2-B:02` done: d* extracted as `dict[layer_idx, unit_vector]`, frozen, smoke-tested by showing prompt-only Qwen + d* at scale=10 shifts direction probability ≥0.05 absolute on a held-out StockNet day.
- `phase2-C:03` done: ORPO loss returns `L = NTP(preferred) + lambda * -logsigmoid(log_odds_pref - log_odds_rej)`. Both pred and rej are scored via teacher-forcing through `compute_log_probs`. Existing teacher_forcing_loss kept for backward-compat behind env var.
- `phase2-D:04` done: under `TROPHIC_CONSUMER_INTERFACE=hooks`, predator+herbivores forward via M hooks not prefix; under `prefix` (default), behavior unchanged.
- `phase3-A:05` done: ipo_seed24 trained, full eval cascade run, results in CHANGELOG and a new GitHub issue. MCC reported alongside prompt-only baseline (0.292) for context.

## Quick start (for agents)

1. Read `tasks_perlayer/scratchpad.md` first.
2. Run the inbox grep for your handle: `grep -nE "@<your-handle>|@all|@phase<N>" tasks_perlayer/scratchpad.md`
3. Flip your STATUS line from PENDING → RUNNING.
4. Read your mission file (`phase<N>-<letter>-<##>-<slug>.md`).
5. Do the work.
6. Run the Testing Conditions block. Every command must produce expected output before flipping DONE.
7. Re-run the inbox grep before flipping DONE.
8. Flip your STATUS line from RUNNING → DONE.
9. Append a MESSAGES entry addressed to the next-phase agent(s).

## References

- Hexis paper sections: `/home/dgonier/debaterhub/hexis/paper/sections/`
  - `hexis_architecture.tex` — the per-layer M+E formula (eq 1)
  - `three_layer.tex` — the three orthogonal channels (M, d*, curated slot)
  - `discussion.tex` — the M attractor problem and the prefill-disable fix
- Hexis training reference: `/home/dgonier/experiments/scripts/train_action_m.py`
  - `DirectM` rank-r class (line ~50)
  - `install_hooks` per-layer modulation (line ~126)
  - ORPO loss (line ~220)
- d* extraction template: `/home/dgonier/experiments/scripts/extract_d_star.py`
- Memory: `project_issue_8_diagnosis.md`, `project_trophic_hexis_alignment.md`

## Baseline metrics (set before swarm starts)

- Tests: 89 passing
- Prompt-only Qwen3-4B + structured prompt on StockNet 50: ACC 0.460, **MCC +0.292**
- All 5 prior trophic checkpoints (seed8/12/14/16/18): ACC 0.720, MCC 0.000 (constant-direction collapse, confirmed via `diag_input_responsiveness.py`)
- Channel v2 (seed20) and trough mode (seed22) trained but produce OOD-prefix gibberish at decode time → 100% abstain on StockNet
- Prompt-only baseline ALONE beats every trained checkpoint by MCC. The architecture has been destroying signal; this swarm fixes that.
