# Mission 04: consumer-rewire

**Handle**: phase2-D
**Phase**: 2 (parallel after phase1-A)
**Mission file**: `phase2-D-04-consumer-rewire.md`
**Dependencies**: phase1-A:01
**Blocks**: phase3-A:05

---

## Before You Start

```bash
# 1. Confirm phase 1 is done.
grep "phase1-A:01:DONE" /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md

# 2. Read what gets replaced.
grep -n "forward_with_prefix\|teacher_forcing_loss" /home/dgonier/ecology_experiment/trophic/trophic/training/sft.py | head
grep -n "_herb_output\|_pred_output\|_channel_output_for\|_trough_output_for" /home/dgonier/ecology_experiment/trophic/trophic/training/sft.py | head

# 3. Read what you're wiring in (phase 1's output).
cat /home/dgonier/ecology_experiment/trophic/trophic/phi_mlp.py
cat /home/dgonier/ecology_experiment/trophic/trophic/m_hooks.py

# 4. Inbox check.
grep -nE "@phase2-D|@all|@phase2" /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md

# 5. Flip status.
sed -i 's/^phase2-D:04:PENDING$/phase2-D:04:RUNNING/' \
  /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md
```

## Goal

Wire phase 1's `PhiMLP` + `install_M_hooks` into the SFT runner so each consumer (herbivore + predator) can use **per-layer M hooks instead of prefix-injection**, gated by `TROPHIC_CONSUMER_INTERFACE={prefix,hooks}`. In `prefix` mode (default), nothing changes — backward compat with all seed1..seed22 checkpoints. In `hooks` mode:

- The trough is attended for a single pooled hidden vector (not a `[out_seq_len, H]` block).
- That vector is fed through a per-consumer `PhiMLP` to produce per-layer modulation tensors.
- Hooks are installed on `host._model.model.layers` for the duration of the consumer's forward, prefill-disabled.
- The model receives `[role_prefix_text + curated_slot + query_text]` as primary context — **no synthesized channel_output prefix.**

This is the thing that fixes the OOD-decoding gibberish from seed20 / seed22.

## Files to Create / Modify

Modify:
- `trophic/training/sft.py` — `SFTRunner` gets a per-consumer `PhiMLP` + a new `_hooks_loss` helper that uses ORPO (from phase 2-C); existing `_herb_loss`/`_pred_loss`/`_herb_output`/`_pred_output` learn to dispatch on `TROPHIC_CONSUMER_INTERFACE`.
- `trophic/agents/herbivore.py` — Herbivore gets a `phi_mlp` field (ensure_initialized creates it).
- `trophic/agents/predator.py` — Predator gets a `phi_mlp` field too.

Create:
- `tests/test_consumer_rewire.py` — at least 3 tests validating dispatch behavior + that hooks-mode does not crash on a real Qwen forward.

Do NOT modify:
- `trophic/phi_mlp.py` or `trophic/m_hooks.py` (phase 1's deliverables, treat as immutable)
- `trophic/dstar.py` (phase 2-B's deliverable)
- `trophic/training/orpo.py` (phase 2-C's deliverable)
- `trophic/model_host.py` (host stays as-is; you call `host._model` directly inside the new helpers)

## Implementation Steps

### 1. Add PhiMLP to agents

Both Herbivore and Predator need to lazy-instantiate a PhiMLP whose hidden-size matches the host. The simplest place: inside `ensure_initialized`, after the existing Channel construction, build a phi_mlp if `TROPHIC_CONSUMER_INTERFACE=hooks`.

```python
# Inside Herbivore.ensure_initialized and Predator.ensure_initialized:
import os
if os.environ.get("TROPHIC_CONSUMER_INTERFACE", "prefix") == "hooks":
    from ..phi_mlp import PhiMLP
    if not hasattr(self, "phi_mlp") or self.phi_mlp is None:
        self.phi_mlp = PhiMLP(hidden_size=host.hidden_size).to(device=host.device, dtype=host.dtype)
```

(You may need to add `phi_mlp = None` as a default field on the dataclass, depending on how each agent class is defined. Look at the existing field structure first.)

### 2. Add a hook-mode dispatch helper to SFTRunner

In `trophic/training/sft.py`, add a sibling to `_channel_output_for` and `_trough_output_for` that runs the hooks-based path:

```python
def _consumer_interface(self) -> str:
    return os.environ.get("TROPHIC_CONSUMER_INTERFACE", "prefix").lower()

def _build_curated_slot(self, sc, candidates) -> str:
    """Render the role's curated slot. For now, a placeholder that summarises
    producer broadcasts as text — phase 3 may upgrade this."""
    parts = []
    for b in candidates[:8]:
        if b.decoded_text:
            parts.append(b.decoded_text[:200])
    return "\n".join(parts)

def _trough_pooled_hidden(self, trough, role_q) -> torch.Tensor:
    """One trough.attend() call returning a single pooled hidden vector
    (mean over the out_seq_len positions)."""
    out = trough.attend(role_q, tau=self._last_tau)
    # out.output is [out_seq_len, H]; pool to [H].
    return out.output.mean(dim=0)

def _hooks_orpo_loss_for_predator(
    self, sc, herb_outputs_for_predator, lambda_or: float = 0.5,
) -> tuple[torch.Tensor | None, dict]:
    """Hooks-mode predator loss. Returns (loss_tensor, diagnostics) or (None, {})."""
    if not sc.predator_target:
        return None, {}
    from ..training.orpo import compute_orpo_loss, sample_rejected_response

    # 1. Build pooled hidden from the herb-trough.
    self._role_prefix_mean = self.predator.role_prefix.mean(dim=0)
    trough = self._ensure_herb_trough(herb_outputs_for_predator)
    if trough is None:
        return None, {}
    from ..agents.base import ROLE_Q_REF_NORM
    rq_raw = self._role_prefix_mean.to(self.host.device, self.host.dtype)
    role_q = rq_raw * (ROLE_Q_REF_NORM / (rq_raw.norm() + 1e-6))
    pooled = self._trough_pooled_hidden(trough, role_q)

    # 2. Compile per-layer M tensors via predator's phi_mlp.
    m_tensors = self.predator.phi_mlp(pooled)

    # 3. Build prefix text. Decode role prefix tokens back to text via host.
    #    Cheaper alternative: store the original role-prefix text on the agent
    #    and reuse it. For phase 2 we use a hard-coded prefix scheme that
    #    matches scripts/diagnostics/baseline_promptonly_stocknet.py.
    role_text = "You are a short-horizon market direction predictor."
    curated = self._build_curated_slot(sc, herb_outputs_for_predator)
    query = "Now produce the PREDICTION and CONFIDENCE."
    prefix_text = f"{role_text}\n\n{curated}\n\n{query}"

    # 4. Sample rejected from the BARE host (no hooks active).
    rejected = sample_rejected_response(self.host, prefix_text, max_new_tokens=64)

    # 5. Install M hooks (prefill-disabled until generation, but compute_log_probs
    #    runs a single teacher-forced forward — treat that as "generate" too).
    from ..m_hooks import install_M_hooks
    handles = install_M_hooks(self.host._model, m_tensors, prefill_active=False)
    try:
        out = compute_orpo_loss(
            self.host, prefix_text,
            preferred_target=sc.predator_target,
            rejected_target=rejected,
            lambda_or=lambda_or,
        )
    finally:
        for h in handles: h.remove()
    return out["loss"], {k: v for k, v in out.items() if k != "loss"}
```

Then in `_pred_output` (the existing dispatch), add a branch:

```python
def _pred_output(self, herb_for_pred):
    if self._consumer_interface() == "hooks":
        # Hooks mode produces no prefix — return None to signal upstream
        # to use the alternative loss path (_hooks_orpo_loss_for_predator).
        return None, []
    # ... existing dispatch (prefix mode unchanged) ...
```

And in the SFT step / eval, add a parallel branch when `_consumer_interface() == "hooks"`:

```python
# In step() and eval_loss() and eval_decode():
if self._consumer_interface() == "hooks":
    if self.predator and sc.predator_target:
        loss, diag = self._hooks_orpo_loss_for_predator(sc, herb_for_pred)
        if loss is not None:
            self.trainer.add_loss(loss)
            # Record diag in result["per_loss"]["pred.short_horizon"] if you want
else:
    # existing prefix-mode path
```

The herbivores in hooks mode are similar but use the producer-trough; for phase 2 keep the herbivore loss path unchanged (still teacher_forcing or _herb_output prefix mode) — phase 3 may extend hooks mode to herbivores if needed.

**The minimum viable rewire is predator-only ORPO+hooks mode.** Herbivores can stay on the prefix path for now.

Make sure:
- `phi_mlp` parameters end up in the trainer's optimizer. Use `extra_modules` arg of `ChannelTrainer.attach` (already supported from earlier work — confirm). Re-attach trainer after `ensure_initialized` if needed.
- `eval_decode` in hooks mode uses `host._model.generate` directly with the prefix_text (no `forward_with_prefix`).

### 3. Tests

```python
# tests/test_consumer_rewire.py
"""Tests for the hooks-mode consumer rewire. Uses MOCK ModelHost so we don't
load Qwen3-4B in CI."""
import os
import torch
import pytest


def test_predator_has_phi_mlp_in_hooks_mode(monkeypatch):
    monkeypatch.setenv("TROPHIC_CONSUMER_INTERFACE", "hooks")
    from trophic.config import DEFAULT_CONFIG
    cfg = DEFAULT_CONFIG
    # Skip if mock not available — this test only runs locally where the host loads
    if not cfg.model.mock:
        pytest.skip("Test only meaningful with mock host")
    from trophic.model_host import ModelHost
    host = ModelHost.get(cfg.model)

    from trophic.agents.predator import Predator
    pred = Predator.make("short_horizon")
    pred.ensure_initialized(host, seed_base=42)
    assert hasattr(pred, "phi_mlp"), "predator should have phi_mlp in hooks mode"
    assert pred.phi_mlp is not None


def test_predator_no_phi_mlp_in_prefix_mode(monkeypatch):
    monkeypatch.setenv("TROPHIC_CONSUMER_INTERFACE", "prefix")
    from trophic.config import DEFAULT_CONFIG
    cfg = DEFAULT_CONFIG
    if not cfg.model.mock:
        pytest.skip("Test only meaningful with mock host")
    from trophic.model_host import ModelHost
    host = ModelHost.get(cfg.model)

    from trophic.agents.predator import Predator
    pred = Predator.make("short_horizon")
    pred.ensure_initialized(host, seed_base=42)
    # In prefix mode, phi_mlp should be absent or None.
    assert getattr(pred, "phi_mlp", None) is None


def test_consumer_interface_helper_default():
    os.environ.pop("TROPHIC_CONSUMER_INTERFACE", None)
    from trophic.training.sft import SFTConfig, SFTRunner
    # We don't construct a runner — just check the helper directly via a stub.
    import trophic.training.sft as sft_mod
    # Whatever the helper is named, check via env var directly.
    assert os.environ.get("TROPHIC_CONSUMER_INTERFACE", "prefix") == "prefix"
```

(Tests are minimal because the heavy stuff requires a real Qwen forward — that's covered in the smoke check below.)

## Acceptance Criteria

- [ ] `Herbivore` and `Predator` both gain a `phi_mlp` attribute (None in prefix mode, PhiMLP instance in hooks mode).
- [ ] `SFTRunner._consumer_interface()` returns the env-var value, default `"prefix"`.
- [ ] In prefix mode, behavior is bit-identical to before (prove via `pytest tests/`).
- [ ] In hooks mode, `_hooks_orpo_loss_for_predator` returns a finite loss tensor with grad.
- [ ] PhiMLP parameters are picked up by the trainer (verify by counting trainable params with hooks mode active vs prefix mode).
- [ ] All 3 new tests + 89 baseline + phase 1's 8 + phase 2-B/C's tests still pass.

## Testing Conditions (exit verification)

```bash
cd /home/dgonier/ecology_experiment/trophic

# 1. Tests pass in BOTH modes (prefix unchanged + hooks new tests)
TROPHIC_CONSUMER_INTERFACE=prefix .venv/bin/python -m pytest tests/ -x
# Expected: all pass (>=104)

TROPHIC_CONSUMER_INTERFACE=hooks .venv/bin/python -m pytest tests/test_consumer_rewire.py -v
# Expected: 3 passed (or skipped if non-mock environment)

# 2. Smoke: in hooks mode, a single SFT step on a real scenario produces a finite loss.
TROPHIC_CONSUMER_INTERFACE=hooks .venv/bin/python -c "
import asyncio, os
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.agents.herbivore import Herbivore
from trophic.agents.predator import Predator
from trophic.agents.producer import Producer
from trophic.agents.quant_producer import QuantitativeProducer
from trophic.training.sft import SFTConfig, SFTRunner
from trophic.training.scenarios import build_scenarios, split

async def go():
    cfg = DEFAULT_CONFIG
    host = ModelHost.get(cfg.model)
    sft_cfg = SFTConfig(seed=24, steps=1)
    herbs = [Herbivore.make(k, capacity=cfg.population.intake_budget)
             for k in ('technical','fundamental')]
    pred = Predator.make('short_horizon')
    producers = [Producer.make(k) for k in ('tickdelta','disclosure','anomaly')]
    producers.append(QuantitativeProducer.make('quote_series'))
    scenarios = build_scenarios()
    train, eval_ = split(scenarios)
    runner = SFTRunner(cfg=sft_cfg, host=host, producers=producers,
                       herbivores=herbs, predator=pred, train=train[:1], eval_=eval_[:1])
    # Cache producers
    for sc in train[:1] + eval_[:1]:
        items = []
        for inp in sc.inputs:
            for prod in producers:
                if prod.attracts(inp):
                    b = await prod.produce(inp, tick=0, host=host)
                    if b is not None: items.append(b)
        runner._producer_cache[sc.name] = items
    # Run one step
    res = runner.step(train[0], train_step=1)
    print(f'one-step result: {res}')
    print(f'pred has phi_mlp: {pred.phi_mlp is not None}')
    print('OK')

asyncio.run(go())
"
# Expected: prints loss diagnostics, "pred has phi_mlp: True", "OK"

# 3. Backward-compat sanity: prefix mode produces same result as before this change.
TROPHIC_CONSUMER_INTERFACE=prefix .venv/bin/python -c "
import asyncio
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.agents.herbivore import Herbivore
from trophic.agents.predator import Predator
host = ModelHost.get(DEFAULT_CONFIG.model)
pred = Predator.make('short_horizon')
pred.ensure_initialized(host, seed_base=42)
print(f'prefix mode phi_mlp: {getattr(pred, \"phi_mlp\", None)}')
assert getattr(pred, 'phi_mlp', None) is None, 'prefix mode should not build phi_mlp'
print('OK')
"
# Expected: "prefix mode phi_mlp: None" + "OK"
```

## Coordination

You're parallel with **phase2-B** (d*) and **phase2-C** (ORPO). Both will land their files independently. You depend on **phase2-C's** `compute_orpo_loss` signature — confirm at start that scratchpad's INTERFACE CONTRACTS match what's in `trophic/training/orpo.py` (phase 2-C is supposed to publish there). If 2-C lands a different signature, post `@phase2-C @all` in MESSAGES, sync, then proceed.

Phase 3 will need:
- `TROPHIC_CONSUMER_INTERFACE=hooks` to dispatch the new path
- A way to also install d* hooks alongside M hooks. Don't conflict — `install_M_hooks` uses `forward_pre_hook`, `install_dstar_hooks` uses `forward_hook`. They can coexist.

You should also leave a hook for phase 3 to install d* hooks before `compute_orpo_loss`. Either:
- Add a `dstar: DStar | None = None` arg to `_hooks_orpo_loss_for_predator` and install d* hooks if provided.
- Or expose a runner-level d* state that's installed at the start of step() and removed at the end.

Pick whichever is cleaner; document in the function docstring.

## When Done

```bash
grep -nE "@phase2-D|@all|@phase2" /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md

sed -i 's/^phase2-D:04:RUNNING$/phase2-D:04:DONE/' \
  /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md

cat >> /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md <<EOF
[$(date +%Y-%m-%d\ %H:%M)] phase2-D > @phase3-A: Consumer rewire landed. TROPHIC_CONSUMER_INTERFACE=hooks activates per-layer M hook path with ORPO loss for predator. Herbivores still on prefix mode (phase 3 can extend if needed). Each agent has self.phi_mlp in hooks mode; trainer optimizes phi_mlp params via extra_modules.
EOF

mv /home/dgonier/ecology_experiment/trophic/tasks_perlayer/phase2-D-04-consumer-rewire.md \
   /home/dgonier/ecology_experiment/trophic/tasks_perlayer/completed/
```
