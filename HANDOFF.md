# Handoff — 2026-05-02 evening

Session goal: get the trophic ecology to beat bare prompt-only Qwen3-4B on the
StockNet ACL-18 next-day binary direction benchmark.

## TL;DR

**Bare frozen Qwen3-4B with the benchmark's expected XML prompt scores MCC
+0.292 on 50 held-out test scenarios.** Every variant of the trophic stack
attempted in this session (seed28/30/32/33/34/35/36) scored MCC ≤ +0.13
under strict scoring. **The ecology has not yet justified its existence
relative to the trivial baseline.**

The session also discovered that all prior MCC numbers in the project
(+0.331 prompt-only baseline, +0.358 trained seed34, etc.) were **parser
hallucinations** — the regex parser was matching schema-echo enumerations
like "DIRECTION is up or down" inside meta-explanation outputs. Strict
parser is now in place and all numbers below are real.

## Numbers (strict parser, single source of truth)

| Setup | scenarios | decided | acc | MCC | class output |
|-------|-----------|---------|-----|-----|--------------|
| Bare Qwen + benchmark XML prompt | 50 | 50/50 | 46% | **+0.292** | 9 up / 41 down |
| Bare Qwen + force-prefix `DIRECTION:` + constrained decode | 100 | 100/100 | 36% | 0 | 0 up / 100 down |
| seed35 trained + free decode | 100 | 0/100 | — | 0 | abstain (schema-echo) |
| seed35 trained + constrained decode | 100 | 100/100 | 64% | 0 | 100 up / 0 down |
| seed36 trained (binary head) | 62 (killed) | 62/62 | 52% | +0.07 | 27 up / 35 down |

The benchmark XML prompt is at `scripts/diagnostics/baseline_promptonly_stocknet.py`.
The seed36 binary head is at `Predator.binary_head` (gated by
`TROPHIC_BINARY_HEAD=1` env var).

## What landed during this session

- `scripts/diagnostics/inspect_signals_jsonl.py` — writes one JSON node
  per signal with parents/child IDs + logit-lens decode + raw text.
  Captures pre-@E signals at every tier.
- `scripts/diagnostics/compare_signals.py` — side-by-side JSONL diff.
- `scripts/diagnostics/inspect_signals_batch.sh` — multi-scenario wrapper.
- `viz/` — Vite + react-flow app (reads `logs/signals/*.jsonl`,
  http://localhost:5173).
- `scripts/diagnostics/loop1_constrained.py` — bare-Qwen first-token
  baseline (constrained to {up,down}).
- **Tier 3 herbivore fix**: `_real_herb_broadcasts_for_predator` now uses
  role_q-conditioned attention pool (not mean) AND injects OHLCV numeric
  features. Inspector confirms Tier 3 broadcasts now genuinely differ
  across opposite-direction scenarios. Toggle: `TROPHIC_HERB_POOL=attn`
  (default), `TROPHIC_HERB_NUMERIC=1` (default), `TROPHIC_BARE_HERB=1`
  (skip M hooks for diagnostic).
- **Strict parser**: `xml_schema.parse_prediction` no longer matches loose
  natural-language patterns that fired on schema-echo enumerations.
  Now accepts only strict XML or leading-token responses.
- **Constrained decode**: `_hooks_eval_decode_predator` does a single
  forward, masks logits to {up,down} ids, returns argmax + softmax conf.
  Toggle: `TROPHIC_CONSTRAINED_DECODE=1` (default).
- **Binary classification head**: `Predator.binary_head` is a
  `nn.Linear(hidden_size, 2)` over the herb-trough-pooled hidden. Trained
  via CE on {up, down}. Save/load wired into checkpoint.py. Bypasses
  phi_mlp + M-hooks + LM-head entirely. Toggle: `TROPHIC_BINARY_HEAD=1`.
- **sft.py prompt wiring**: previously hardcoded role/query strings,
  bypassing predator.py's ROLE_PROMPTS/QUERIES. Now imports them.
- **Over-fit guard**: training aborts when train_loss < 0.3
  (`TROPHIC_LOSS_FLOOR`) before the NaN cascade fires, preserving the
  best ckpt cleanly.
- **Eval cache hygiene**: `eval_loss` calls `torch.cuda.empty_cache()`
  between scenarios. Did NOT fix the silent crash at training step 150
  (still happens — but consistent best ckpt at step 100/150 means we
  always have a usable artifact).

## Why each trained ckpt failed

- **seed28/30/32**: M weights collapsed hidden states toward `<` token
  (XML-tag direction) because predator target was XML-formatted; gradient
  backed up through herb hooks too. Constant-class output in free decode.
- **seed33/34**: After herb attn-pool + numeric inject landed, Tier 3
  broadcasts became genuinely differentiated per inspector. But predator
  M weights didn't learn to map this to up/down at the LM-head's first-
  token logit. Free decode produced meta-explanations ("The PREDIION is
  the direction of the market in the next 1-3 days...") that loose
  parser hallucinated as predictions.
- **seed35**: Same issue plus a force-prefix prompt that didn't actually
  reach the trained model (sft.py was hardcoding prompts, bypassing
  predator.py edits — fixed in `e0acb02`). With strict parser: 0/100.
  With constrained decode: 100/100 'up' (matches StockNet up-class
  prior; pure prior memorization).
- **seed36**: Binary head over pooled hidden. First trained ckpt that
  emits both classes balanced under strict scoring. But MCC trended
  down through the eval (+0.13 at 36/100, +0.07 at 62/100), and even
  the early lift falls short of the bare-Qwen +0.292.

## Where the signal IS and ISN'T

The inspector (run via `scripts/diagnostics/inspect_signals_jsonl.py +
viz/`) shows:

- **Tier 1 producers**: input-conditioned ✓ (DOWN scenarios decode '$'
  in disclosure, UP scenarios decode 'a'; tickdelta digit distributions
  differ).
- **Tier 2 trough slots**: pass-through ✓.
- **Tier 3 herbivore broadcasts (after the fix)**: input-conditioned ✓
  (DOWN: 'up'/'future' tokens at 4-6%; UP: 'as'/'mol'/'gate'/'synth').
  Different content in opposite-direction scenarios.
- **Tier 4 attended-pooled (predator input)**: input-conditioned ✓
  (DOWN: 'unders'/'tone' / 'towing'; UP: 'conductor'/'cribed'/'sil').
- **Tier 5 apex output (decoded)**: NOT input-conditioned in any seed.
  The pooled hidden has signal; the LM head's first-token distribution
  ignores it (rank-16 M perturbation can't override Qwen's intrinsic
  prior on a single token).

The binary head bypasses tier 5 by reading from tier 4 directly.
seed36 confirmed there IS some directional signal in the pooled hidden
(+0.07 MCC > 0), but the herb tier alone doesn't carry enough to beat
the prompt-engineered bare model.

## Recommended next steps

In priority order. Each is a real architectural attempt, not a re-skin
of an exhausted approach.

### 1. Train seed37 with binary head + longer training
- 300+ steps with lr=2e-5 + the over-fit floor as a backstop
- Eval every 25 steps with the binary head — track MCC trajectory
- Need: does seed36's MCC rise to +0.29+ given more steps, or saturate
  early and over-fit?

### 2. Larger/deeper binary head
seed36's head is `Linear(hidden_size, 2)` — single layer. Try:
- `Linear(H, 256) → ReLU → Linear(256, 2)` (small MLP head)
- Or `Linear(H, H) + LayerNorm → Linear(H, 2)`

The current 1-layer head can only learn linear separations of the
herb-tier signal. If the signal is non-linearly separable, a deeper
head might extract it.

### 3. Combine multiple herbivore signals
seed36 only feeds the predator's `attended_pooled` (one fused signal
across both herbs) into the head. Try concatenating each herb's
broadcast directly:
```python
head_input = torch.cat([
    pooled,
    herb_for_pred[0].channel_embedding,  # technical
    herb_for_pred[1].channel_embedding,  # fundamental
], dim=-1)
binary_head: Linear(3*H, 2)
```

### 4. Per-ticker conditioning
The bare baseline showed strong per-ticker variance (AAPL 50%, JPM 30%,
AMZN 60%). Adding a learnable ticker embedding to the head's input
might help the model account for ticker-specific patterns in the
training data.

### 5. Train herbivore phi_mlp for direction-discriminative pool
Currently the herb's role_q is its mean role_prefix vector — fixed.
Make the herb's role_q a learnable parameter that's updated by the
binary head's loss flowing back through the pool. The herb starts
attending to the parts of its own forward that are most predictive of
direction.

### 6. Skip the herb tier entirely (sanity check)
Train a binary head directly on the **producer** trough's pooled
hidden:
```python
prod_pooled = self._trough_pooled_hidden(self._producer_trough, role_q)
logits = predator.binary_head(prod_pooled)
```
If this scores higher than seed36's herb-tier path, the herb tier is
ablating signal rather than adding it. If lower, the herb is at least
not hurting. Either way, this isolates whether the herbivore is
pulling its weight architecturally.

### 7. Don't add apex voting / decomposers (#95 / #96 / #97) yet
These multiply whatever signal the apex carries. Until the architecture
beats bare on a single forward, voting just amplifies noise. Pending
issues should stay pending.

## Files / state

- Latest commits on main:
  - `e466c41` add bare-Qwen constrained-decode baseline
  - `e0acb02` strict parser + agent-defined prompts + constrained decode
  - `5f15618` herb attn-pool + OHLCV inject + signal viz
  - `1a332a8` Loop 2 + Loop 3 iteration ladder
- Best checkpoints:
  - `checkpoints/sft_seed36_stable_best.pt` (binary head, dev=1.59)
  - `checkpoints/sft_seed35_stable_best.pt` (M-hooks ORPO, dev=2.95)
  - `checkpoints/sft_seed34_stable_best.pt` (similar, dev=3.06)
- Logs:
  - `logs/baseline_promptonly_stocknet_v2.log` — bare 50 scenarios MCC +0.292
  - `logs/loop2_seed36_full.log` — partial 62/100 MCC +0.07
  - `logs/loop2_seed35_constrained.log` — 100/100 constant 'up' MCC 0
- Diagnostic infra:
  - `scripts/diagnostics/inspect_signals_jsonl.py`
  - `scripts/diagnostics/compare_signals.py`
  - `scripts/diagnostics/loop1_constrained.py`
  - `viz/` (Vite+react-flow)

## How to verify state on next session

```bash
cd /home/dgonier/ecology_experiment/trophic
git log --oneline -5
ls -lh checkpoints/sft_seed3{4,5,6}_stable_best.pt

# Quick sanity check the binary head architecture is wired correctly:
TROPHIC_BINARY_HEAD=1 .venv/bin/python -u scripts/diagnostics/loop2_hooks_smoke.py \
  --ckpt checkpoints/sft_seed36_stable_best.pt --seed 36 --n-per-ticker 1

# Re-run the bare benchmark baseline:
.venv/bin/python -u scripts/diagnostics/baseline_promptonly_stocknet.py
# Expect: MCC +0.292 on 50 scenarios.
```

## Memory updated

`/home/dgonier/.claude/projects/-home-dgonier-ecology-experiment/memory/`:
- `MEMORY.md` — index updated with the corrected baseline finding
- `project_signal_flow_2026_05_02.md` — full audit + verdict appended
- `project_issue_8_diagnosis.md` — path C (SFT on M+E) marked exhausted
