# Mission 05: train-and-eval

**Handle**: phase3-A
**Phase**: 3 (sequential, integration)
**Mission file**: `phase3-A-05-train-and-eval.md`
**Dependencies**: phase2-B:02, phase2-C:03, phase2-D:04
**Blocks**: nothing (terminal)

---

## Before You Start

```bash
# 1. Confirm all three phase 2 missions are done.
grep -E "phase2-(B|C|D):..:DONE" /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md
# All three lines should be present.

# 2. Inbox check.
grep -nE "@phase3-A|@all|@phase3" /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md

# 3. Verify d* checkpoint exists (phase 2-B should have made it).
ls -la /home/dgonier/ecology_experiment/trophic/checkpoints/dstar_stocknet.pt

# 4. Confirm baseline metrics still hold by re-running prompt-only.
TROPHIC_EVAL_MAX_TOKENS=256 .venv/bin/python -u scripts/diagnostics/baseline_promptonly_stocknet.py 2>&1 | tail -8
# Expected: ACC ~0.46, MCC ~0.29 (the prompt-only target to beat)

# 5. Flip status.
sed -i 's/^phase3-A:05:BLOCKED.*$/phase3-A:05:RUNNING/' \
  /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md
```

## Goal

Tie all three phase 2 deliverables into a single SFT seed24 training run, then eval the resulting checkpoint on the full StockNet smoke. This is the architectural validation moment — the rewrite is justified IFF MCC ≥ 0.05 (worth keeping) or ≥ 0.10 (paper-grade, matches Hexis's banking_knowledge +20pp result).

## Files to Create / Modify

Create:
- `scripts/train_sft_perlayer.py` — entry script that:
  1. Sets `TROPHIC_CONSUMER_INTERFACE=hooks`
  2. Loads d* via `DStar.load(checkpoints/dstar_stocknet.pt)`
  3. Plumbs d* into the SFT runner (via the hook installed before each step's `compute_orpo_loss` call)
  4. Calls the existing SFTRunner with seed=24
- `tests/test_perlayer_integration.py` — at least 2 e2e tests that wire all three together on a tiny synthetic scenario

Modify:
- `CHANGELOG.md` — add `[Unreleased]` entry under `### Added` and `### Results` summarising the per-layer rewrite + outcome
- `docs/EXPERIMENTS.md` — append a new section "Per-layer M+E refactor (2026-04-29)" with results table and architectural notes

After eval, also:
- Open a GitHub issue in `dgonier/consciousness-ecology-codd` summarising the rewrite and result. Use the `remember` skill via `gh issue create` directly.

## Implementation Steps

### 1. Add d* installation alongside M hooks in the SFT runner

The cleanest plumbing: SFTRunner gets a `_dstar` field that's loaded once at runtime. In `_hooks_orpo_loss_for_predator` (built by phase 2-D), if `_dstar is not None`, also install d* hooks alongside M hooks. **Both** must be removed in the `finally` block.

```python
# Pseudo, inside SFTRunner:
def __post_init__(self):
    # ... existing init ...
    dstar_path = os.environ.get("TROPHIC_DSTAR_PATH", "")
    self._dstar = None
    if dstar_path and Path(dstar_path).exists():
        from ..dstar import DStar
        self._dstar = DStar.load(dstar_path).to(self.host.device, self.host.dtype)
        print(f"[sft] loaded d* from {dstar_path}: "
              f"{len(self._dstar.directions)} layers, scale={self._dstar.scale}")

# In _hooks_orpo_loss_for_predator:
m_handles = install_M_hooks(self.host._model, m_tensors, prefill_active=False)
d_handles = []
if self._dstar is not None:
    from ..dstar import install_dstar_hooks
    d_handles = install_dstar_hooks(self.host._model, self._dstar, active=True)
try:
    out = compute_orpo_loss(...)
finally:
    for h in m_handles: h.remove()
    for h in d_handles: h.remove()
```

### 2. Train script

```python
# scripts/train_sft_perlayer.py
"""SFT seed24 with the per-layer M+E rewrite + ORPO + d*.

Sets TROPHIC_CONSUMER_INTERFACE=hooks and TROPHIC_DSTAR_PATH=checkpoints/dstar_stocknet.pt
before invoking the existing SFTRunner; everything else flows through the
existing infrastructure. Saves checkpoints to:
    checkpoints/sft_seed24_perlayer_best.pt
    checkpoints/sft_seed24_perlayer_final.pt
"""
import os
import sys
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Activate the new architecture BEFORE any trophic imports load.
os.environ.setdefault("TROPHIC_CONSUMER_INTERFACE", "hooks")
os.environ.setdefault("TROPHIC_DSTAR_PATH", str(ROOT / "checkpoints" / "dstar_stocknet.pt"))

from scripts.train_sft import main as sft_main  # reuses existing entry

if __name__ == "__main__":
    os.environ.setdefault("TROPHIC_SEED", "24")
    os.environ.setdefault("TROPHIC_STEPS", "200")
    asyncio.run(sft_main())
```

(If the existing train_sft.py main doesn't honor the seed24 checkpoint name, adjust either the env var pattern or write a thin wrapper that names checkpoints with a `_perlayer` suffix.)

### 3. Eval script (reuse existing)

The existing `scripts/eval_stocknet.py` works with the new checkpoint as long as `TROPHIC_CONSUMER_INTERFACE=hooks` is set when it runs (so the predator's eval_decode dispatches to the hooks path). Add the same env-var preamble as the train script if needed.

### 4. Tests

```python
# tests/test_perlayer_integration.py
import os
import asyncio
import torch


def test_dstar_loads_into_runner(monkeypatch, tmp_path):
    """Verify SFTRunner.__post_init__ picks up d* if env var is set."""
    from trophic.dstar import DStar
    H = 32
    layers = [0, 3, 6]
    d = DStar(directions={l: torch.randn(H) for l in layers}, scale=10.0, hidden_size=H)
    p = tmp_path / "dstar.pt"
    d.save(p)
    monkeypatch.setenv("TROPHIC_DSTAR_PATH", str(p))
    monkeypatch.setenv("TROPHIC_CONSUMER_INTERFACE", "hooks")

    from trophic.config import DEFAULT_CONFIG
    cfg = DEFAULT_CONFIG
    if not cfg.model.mock:
        import pytest
        pytest.skip("Real-host integration test runs only in smoke section")
    # ... build a runner with mock host, assert runner._dstar is not None

def test_three_channels_compose():
    """Predator forward with M hooks + d* hooks installed yields a loss tensor
    distinct from either alone."""
    # Skip if mock env doesn't support real generate; this test is a placeholder
    # that can be filled in once mock host supports `_model.generate`.
    pass
```

(Tests are minimal — the real validation is the StockNet eval, not unit tests.)

### 5. Train + eval cascade

```bash
# 5.1 Extract d* if not already done (phase 2-B should have done this)
ls checkpoints/dstar_stocknet.pt || \
  .venv/bin/python -u scripts/extract_dstar_stocknet.py --out checkpoints/dstar_stocknet.pt

# 5.2 Train SFT seed24 with hooks + d*
.venv/bin/python -u scripts/train_sft_perlayer.py > logs/sft_seed24_perlayer.log 2>&1
# Expected: dev loss descends; final checkpoints at sft_seed24_*_best.pt

# 5.3 Run StockNet smoke
TROPHIC_CONSUMER_INTERFACE=hooks TROPHIC_DSTAR_PATH=checkpoints/dstar_stocknet.pt \
TROPHIC_CKPT=checkpoints/sft_seed24_best.pt TROPHIC_SEED=24 \
TROPHIC_LABEL=sft_seed24_perlayer STOCKNET_MAX_PER_TICKER=10 \
TROPHIC_EVAL_MAX_TOKENS=96 \
.venv/bin/python -u scripts/eval_stocknet.py > logs/eval_stocknet_sft_seed24_perlayer.log 2>&1

# Read the result
tail -25 logs/eval_stocknet_sft_seed24_perlayer.log
```

### 6. Document

Update CHANGELOG.md and EXPERIMENTS.md per the format used by prior entries:
- Per-layer architecture description
- Loss shape change (ORPO)
- d* extraction details + scale
- Result table: prefix-mode baseline (seed18 prior numbers) vs hooks-mode seed24
- StockNet ACC + MCC vs prompt-only baseline (0.46/0.29)

Open a GitHub issue:
```bash
gh issue create \
  --title "Per-layer M+E refactor: hooks-based consumer interface (Hexis-aligned)" \
  --body "$(cat <<'EOF'
## Status
DONE

## Motivation
After Issue #10 paths A and B closed (FinCoT and SocialSignal LoRAs both StockNet MCC=0), the diagnostic
chain pinned the bottleneck to trophic's prefix-injection design. Reading the Hexis paper
(`/home/dgonier/debaterhub/hexis/paper/sections/`) clarified: trophic was doing context-level memory
injection, exactly the failure mode Hexis was designed to avoid via per-layer Q+V hook modulation.

## Proposal
Five-mission swarm refactor:
1. PhiMLP: trough-attended hidden -> per-layer (M_A, M_B, E_A, E_B) rank-16 tensors
2. install_M_hooks: forward-pre-hooks on stride-3 Qwen layers, prefill-disabled
3. d* extraction: per-layer pro/con direction (frozen), post-attention residual
4. ORPO loss: NTP(preferred) + lambda * sigmoid(log_odds_pref - log_odds_rej)
5. Consumer rewire: TROPHIC_CONSUMER_INTERFACE=hooks dispatches new path

## Result
StockNet seed24 MCC = <fill in>
Prompt-only baseline: MCC 0.292 (the bar to beat)
Prior trophic best: MCC 0.000 (constant collapse)

## Decision
- MCC > 0.10: paper-grade, ramp.
- MCC > 0.05: signal, keep iterating on the architecture.
- MCC ~ 0: bottleneck is elsewhere; back to architectural questions.

## Code
- trophic/phi_mlp.py
- trophic/m_hooks.py
- trophic/dstar.py
- trophic/training/orpo.py
- scripts/train_sft_perlayer.py
- scripts/extract_dstar_stocknet.py
EOF
)" \
  --label "topic:architecture,topic:training,origin:swarm,status:done"
```

## Acceptance Criteria

- [ ] `scripts/train_sft_perlayer.py` exists and runs end-to-end without crashing.
- [ ] `checkpoints/sft_seed24_best.pt` (or equivalent named checkpoint) exists post-training.
- [ ] StockNet smoke ran on the new checkpoint and produced a numeric MCC.
- [ ] Result documented in CHANGELOG.md, EXPERIMENTS.md, and a new GitHub issue.
- [ ] All tests still pass (89 baseline + ~20 from phases 1-2).

## Testing Conditions (exit verification)

```bash
cd /home/dgonier/ecology_experiment/trophic

# 1. All tests pass
.venv/bin/python -m pytest tests/ -x
# Expected: all pass (>=110)

# 2. Train script completes
.venv/bin/python -u scripts/train_sft_perlayer.py > logs/sft_seed24_perlayer.log 2>&1
# Expected: log ends with "[ckpt] final → sft_seed24_*"; no Python tracebacks

# 3. StockNet smoke completed
test -f logs/eval_stocknet_sft_seed24_perlayer.log
grep -E "MCC:|ACCURACY:" logs/eval_stocknet_sft_seed24_perlayer.log
# Expected: prints MCC and ACCURACY values

# 4. CHANGELOG + EXPERIMENTS updated
grep -i "per-layer" CHANGELOG.md docs/EXPERIMENTS.md
# Expected: matches in both files

# 5. GitHub issue exists
gh issue list --search "per-layer M+E refactor" --json number,title
# Expected: returns at least one issue
```

## Coordination

You're the integration. All three phase 2 deliverables must land cleanly before you start:
- phase2-B's `DStar` + `install_dstar_hooks` + saved `checkpoints/dstar_stocknet.pt`
- phase2-C's `compute_orpo_loss` + `compute_log_probs` + `sample_rejected_response`
- phase2-D's `TROPHIC_CONSUMER_INTERFACE=hooks` dispatch + per-agent `phi_mlp`

If any phase 2 mission landed a different contract than scratchpad's INTERFACE CONTRACTS section, post `@all` in MESSAGES BEFORE wiring. Don't silently work around contract drift.

## When Done

```bash
grep -nE "@phase3-A|@all|@phase3" /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md

sed -i 's/^phase3-A:05:RUNNING$/phase3-A:05:DONE/' \
  /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md

cat >> /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md <<EOF
[$(date +%Y-%m-%d\ %H:%M)] phase3-A > @all: Per-layer refactor landed end-to-end. seed24 MCC=<value>. Documented in CHANGELOG/EXPERIMENTS + GitHub issue. Swarm complete.
EOF

mv /home/dgonier/ecology_experiment/trophic/tasks_perlayer/phase3-A-05-train-and-eval.md \
   /home/dgonier/ecology_experiment/trophic/tasks_perlayer/completed/
```
