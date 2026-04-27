# Contributing

This is a research codebase, primarily a single-author project. The notes below are for collaborators or for re-orienting the author after a context break.

## Working agreements

- **Don't commit checkpoints.** They're 2-4 GB each. `.gitignore` excludes them by default.
- **Don't commit `external/` or `data/`.** Datasets are pulled on demand via loader scripts.
- **Don't commit `logs/`.** Training and eval output goes there; we keep it local for diagnostics, not in the repo.
- **Tests pass before commits.** `pytest tests/` should report all green.
- **No emoji in code or commit messages** unless the user explicitly asks for them.
- **Commit messages end with a `Co-Authored-By: Claude` line** if the change was Claude-assisted. Keep it factual; describe what changed and why.

## Repo layout

See [README.md](README.md). The short version:

- `trophic/` — package code
- `scripts/` — runnable entry points
- `tests/` — pytest test suite
- `tasks/` — the historical six-mission build plan (kept for reference)
- `todo/` — current ongoing experimental threads
- `docs/` — architecture spec, integration plans, results
- `checkpoints/` — gitignored; trained weights
- `external/` — gitignored; vendored datasets
- `logs/` — gitignored; training/eval logs

## Workflow

### Code changes
1. Branch from main: `git checkout -b feature/<short-name>`
2. Edit. Keep changes small and focused.
3. `pytest tests/` — must be green.
4. If you added a new architectural mechanism, add a test. The bar is "the test would have caught this if it broke."
5. Commit with a clear message explaining *why*.

### Training runs
1. Bump the `TROPHIC_SEED` env var to a fresh number so checkpoints don't collide.
2. Redirect output to `logs/<descriptive_name>.log`. Save the PID to `logs/<name>.pid`.
3. Use `python -u` for unbuffered output; `nohup` for survival across parent disconnect.
4. After the run, follow checkpoint hygiene (see below) before starting the next.

### Checkpoint hygiene
- **Keep**: the run's `_best.pt` and the previous `_best.pt` (rewind buffer for overfitting).
- **Delete**: the run's `_final.pt` and any older intermediates.
- **Never auto-delete** pre-existing baselines (`ipo_seed1_best.pt`, `interrogator_lora/`, etc.) without explicit confirmation. They're the comparison points for every future experiment.

### Documenting results
After every training run, append a row to [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). Include:
- Date / seed / config
- Dev reward
- Held-out reward (if eval ran)
- Notable observations (mode collapse, instability, schedule changes, etc.)

## Architectural conventions

- **Hidden-state-native between same-model agents**: Qwen3-4B agents pass `channel_embedding` (pooled last hidden state) directly without re-tokenizing. Cross-model bridges (Qwen3 ↔ Qwen2.5-Math) use a learned projection (`CrossModelChannel`).
- **Trough is the K/V matrix**: one `TroughAttention` per `(tier, agent_kind)` tuple. Not per agent.
- **Lifecycle is opt-out via `population_strategy="none"`**: useful for tests where you need stable slot membership.
- **Decomposer bias is consumed once**: `set_pending_bias()` is reset on every `attend()` call to avoid stacking.
- **Skip α is a single learnable scalar on the predator**, sigmoid-bounded.

## Testing

### Test taxonomy
- `tests/test_trough_attention.py` — K/V mechanics, multi-head, decay buffer
- `tests/test_slot_lifecycle.py` — kill, spawn, niche targeting, population stability
- `tests/test_multi_head_niches.py` — head specialization emergence, diet-tag removal
- `tests/test_decomposer_bias.py` — zero-init, bias path, consume-once
- `tests/test_skip_connections.py` — α gradient, τ schedule, mixing
- `tests/test_pipeline.py` — end-to-end smoke on mock models
- `tests/test_substrate.py`, `tests/test_channel.py` — legacy interfaces still working

### Running tests
```bash
.venv/bin/python -m pytest tests/                # all
.venv/bin/python -m pytest tests/test_trough_attention.py -v   # one file
.venv/bin/python -m pytest tests/ -k "lifecycle"  # by name
```

51 tests at last count. The suite is fast enough (~40s) that there's no excuse for skipping it.

## Asking the model for help

When asking Claude to make changes, prefer:
- **Specific over vague**: "add a test for the skip-α gradient flow" not "make the predator better"
- **One file at a time** unless the change is genuinely multi-file
- **Provide context**: paste error messages, link to file:line, say what you tried

The model has access to a `swarm` skill (`/swarm <objective>`) for scaffolding multi-agent coordination on large tasks. Not needed for ordinary changes.
