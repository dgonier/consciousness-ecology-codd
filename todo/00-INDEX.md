# TODO INDEX

Centralized planning across all open threads. Update STATUS column as items move.

| File | Thread | Status | Owner | ETA |
|---|---|---|---|---|
| 01-stocknet-smoke.md | Benchmark goalpost (Tier-1) | NOT STARTED | self | 1-2 days |
| 02-stockbench-smoke.md | Benchmark goalpost (Tier-2, paper-grade) | NOT STARTED | self | 3-4 days |
| 03-bayesian-nodes.md | New species at tier 1 + tier 2 | DESIGNED, NOT BUILT | self | 5-7 days |
| 04-paper-decision.md | Should this be a NeurIPS submission? | DECISION-PENDING-ON-IPO-RESULT | self | depends |

## Status legend
- NOT STARTED — design captured, no code yet
- DESIGNED, NOT BUILT — plan + architecture exist, implementation not begun
- IN PROGRESS — actively coding/training
- BLOCKED — waiting on another todo or external thing
- DONE — landed and verified
- DECISION-PENDING — waiting for an experimental result before committing

## Shared dependencies

These are real cross-thread dependencies, not metaphor:
- StockNet smoke (#01) is **prerequisite** for paper decision (#04). Without an external benchmark number, we have no goalpost.
- StockBench smoke (#02) is the **strongest paper claim**. Run after #01 if #01 lands a non-disastrous number.
- Bayesian nodes (#03) is **independent** of #01/#02 from a code standpoint, but competes for GPU and the user's attention.

## Current in-flight work (not in this folder)

- IPO seed 7 post-migration training, ~30 min remaining. PID at logs/ipo_seed7_post_migration.pid. Will produce ipo_seed7_best.pt for use in #01/#02 evals.
- Pending after IPO finishes:
  - Real predator-reward eval (dev) on ipo_seed7_best.pt
  - Held-out eval (68 fresh tickers) on ipo_seed7_best.pt
  - Held-out eval on ipo_seed1_best.pt (pre-migration baseline; we have its dev=0.388 but never ran held-out)

## Long-running-process logging policy

Every long-running process (training, eval, subagent task, anything that takes >2 min) **MUST** write its own log to `/home/dgonier/ecology_experiment/trophic/logs/<descriptive_name>.log` (or a similar path inside the project).

Why: the harness-managed task transcripts at `/tmp/claude-1000/<session>/tasks/<id>.output` are:
- Cleaned up between sessions / on crash
- Truncated for tool-result display (we lost the first held-out eval result this way)
- Not addressable after the parent agent forgets the task ID

Concrete patterns:

**Background process started via Bash:**
```bash
nohup .venv/bin/python -u scripts/train_X.py \
  > /home/dgonier/ecology_experiment/trophic/logs/X.log 2>&1 &
echo $! > /home/dgonier/ecology_experiment/trophic/logs/X.pid
```
Note: `python -u` for unbuffered output; `nohup` so it survives parent disconnect.

**Subagent started via Agent tool:**
Brief the agent in its prompt: "All durable output goes to `logs/<name>.log` via explicit `> logs/<name>.log 2>&1` redirect on commands you run. Don't rely on tool-result for any output you'll need recoverable."

**Resumability after a crash:**
- Agent IDs are session-scoped. If we crash, we lose `SendMessage` access.
- But on-disk artifacts (logs/*, checkpoints/*) survive. Resume by reading those.
- Pattern from prior recovery: a new "finalization" agent reads logs + checkpoint mtimes + prior MESSAGES entries to reconstruct state.

This policy applies to all four todo threads in this folder. Update mission files in `tasks/` to reflect it (already mostly done — phase1/2/3 tests reference logs/, but verify on next agent spawn).

## Decision criteria

- **Paper-go decision** (#04): if post-migration IPO ≥ 0.50 on dev AND held-out gap < 30%, start drafting. If <0.45 dev OR held-out collapses, stop and ablate.
- **StockNet criterion** (#01): subsample accuracy ≥ 60% → proceed to full set. <55% → debug ticker conditioning before further benchmark work. 55-60% → ablation matrix.
- **Bayesian build trigger** (#03): start when either (a) #01/#02 reveal a clear failure mode that calibrated probabilities would fix, or (b) the textual-only stack hits a soft ceiling on benchmark numbers.
