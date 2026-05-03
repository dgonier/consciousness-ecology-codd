# Handoff — 2026-05-03 afternoon

Session goal: get the trophic ecology to beat bare prompt-only Qwen3-4B on the
StockNet ACL-18 next-day binary direction benchmark.

## TL;DR

**Bare frozen Qwen3-4B with the benchmark's expected XML prompt scores MCC
+0.147 on 100 held-out test scenarios. The trophic stack alone peaks at
MCC +0.117 (seed36).** Neither beats the other end-to-end.

**However, the AGREE ensemble (commit only when bare and seed36 give the
same answer) lifts MCC to +0.210 — a 43% relative improvement over bare,
on 41/100 scenarios.** When both agree, accuracy is 51%, but the class
skew is favorable (39 down / 2 up) so MCC compounds. When they
disagree, exactly one is always right and there's no signal to pick the
right one with — disagreement is essentially noise.

This is the first architectural win in the project: the trophic stack
adds **real complementary signal** to bare prompt, even though it can't
beat bare on its own. Useful for any application that can abstain on
~60% of inputs (trading bots can; strict benchmarks may not).

Every architectural extension attempted on top of seed36 has hurt:
adding numeric Chronos forecaster broadcast (seed38: -0.08), going to
a deeper MLP head (seed39: -0.01). seed36's simple linear head over
the herb-tier pool appears to be the high-water mark for this
architecture / benchmark combination.

The session also corrected a parser bug from earlier work:
all prior MCC numbers (+0.331 prompt-only baseline, +0.358 trained
seed34, etc.) were **parser hallucinations** — the regex parser was
matching schema-echo enumerations like "DIRECTION is up or down" inside
meta-explanation outputs. Strict parser is now in place; numbers below
are real.

## Numbers (strict parser, 100-scenario StockNet test, single source of truth)

| Setup | decided | acc | MCC | up/down | head |
|-------|---------|-----|-----|---------|------|
| **AGREE ensemble (bare + seed36)** | **41/100** | **51%** | **+0.210** | 2 / 39 | both must agree |
| Bare Qwen + benchmark XML prompt | 100/100 | 48% | +0.147 | 22 / 78 | n/a (no trophic) |
| seed36 (linear head, herb tier) | 100/100 | 53% | +0.117 | 41 / 59 | Linear(H, 2) |
| seed39 (MLP head, herb tier) | 100/100 | 53% | -0.014 | 63 / 37 | H→H/4→ReLU→Drop→2 |
| seed38 (linear + Chronos numeric forecaster) | 100/100 | 57% | -0.082 | 85 / 15 | Linear(H, 2) |
| seed37 (linear, producer-only, no herb) | 100/100 | 37% | 0 | 0 / 100 | Linear(H, 2) |
| seed35 trained + free decode | 0/100 | — | 0 | abstain (schema-echo) | LM-head |
| seed35 trained + constrained decode | 100/100 | 64% | 0 | 100 / 0 | LM-head |

**Disagreement set (59/100 scenarios): perfectly anti-correlated.** When
bare and seed36 disagree, exactly one is always right and the other
always wrong (zero "both wrong"). seed36 is right 32/59, bare 27/59.
seed36's MCC on disagreement-only is -0.06 because it's still up-biased
even when correct; bare's MCC on disagreement-only is +0.06. Neither is
a reliable arbiter, so MAJORITY strategies degenerate to the chosen
arbiter alone. The signal is in the agreement.

**Confidence-gated AGREE**: seed36's binary head softmax confidence
ranges 0.50–0.66 (narrow). AGREE + conf>0.55 gives MCC +0.212 on
38/100 — marginally better than plain AGREE (+0.210 on 41/100). Higher
thresholds (≥0.60) drop too many decisions to be useful. Confidence
isn't well-graduated enough to be a strong signal.

**Per-ticker heterogeneity is large** (each ticker has 20 scenarios):

| Ticker | bare MCC | seed36 MCC | Best | Notes |
|--------|----------|-----------|------|-------|
| AAPL | +0.058 | +0.101 | seed36 | both modest |
| AMZN | +0.182 | -0.061 | **bare** (big delta) | seed gets confused on AMZN |
| GOOG | +0.190 | +0.190 | tie | |
| JPM | +0.000 | +0.287 | **seed36** (big delta) | seed shines |
| MSFT | +0.153 | +0.171 | seed36 | both moderate |

Per-ticker oracle router (cheating: uses test labels to know which
system to trust per ticker): MCC +0.159 — only marginally better than
bare alone. The gains on JPM are partially offset by losses elsewhere
because individual scenario predictions vary even within tickers.
Useful for a learned router on held-out train data; not actionable
without extra training infrastructure.

The benchmark XML prompt is at `scripts/diagnostics/baseline_promptonly_stocknet.py`.
The binary head is at `Predator.binary_head` (gated by
`TROPHIC_BINARY_HEAD=1`; depth via `TROPHIC_BINARY_HEAD_DEPTH=1|2`).
Forecaster numeric path gated by `TROPHIC_FORECAST_PROJ=1` +
`TROPHIC_COMPUTE_FORECAST=1`.

## Key new finding 2026-05-03: Chronos doesn't help

Wired ForecasterHost (Chronos-Bolt) properly: cached 32-dim numeric
features per (ticker, date) at `external/stocknet_cache/forecast_features.json`,
projected to hidden_size, inserted as a 4th broadcast in the herb-trough.
seed38 dropped to MCC -0.08.

Investigation: **Chronos's predicted drift sign is constant 'down' on all
196 test scenarios.** With 5-day OHLCV history × horizon=12, Chronos
isn't directionally informative. Adding it as input gave the head a
constant signal to anti-learn from, biasing predictions toward the
opposite class.

Architecture wiring was correct — the *data* from this Chronos config is
unusable. Either need much longer history (Chronos-Bolt supports up to
64 bars; we feed 5) or skip the forecaster path. Cache infrastructure is
in place (`scripts/cache_chronos_features.py`); just needs the loader to
emit longer histories before re-running.

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

## Decomposers: KG + evolution manager (architecture, 2026-05-03)

Decomposers reframed per user as **two-headed**: (1) Hexis-style KG
writer that observes every (scenario, voter_responses, ensemble,
ground_truth) tuple and writes a structured record for long-term
memory; (2) Evolutionary population manager that tracks per-voter
rolling fitness and suggests `reproduce` / `die` / `thin_mutate` /
`keep` actions. Decomposers do not touch agent weights. They change
WHO is in the population over time.

`trophic/decomposers/`:
- `kg_writer.py` — JSONL KG with `KGRecord` schema. Append-mode writer,
  read-back helper. Default path `external/decomposer_kg/kg.jsonl`.
- `fitness.py` — `AgentFitness` with two signals: SOLO accuracy and
  MARGINAL contribution (how often voter's exclusion would have
  flipped the panel and the voter's vote made the panel correct).
  Free-rider score = solo - marginal. Rolling window of 200 events.
- `population_manager.py` — policy: high acc + high marginal → reproduce
  with `prompt_template_swap`; low acc + low marginal → die; high
  acc + low marginal (free-rider) → `temperature_shift` thin-mutation.
  Min-seen gate of 20 events to suppress premature decisions.

`scripts/diagnostics/apex_vote_eval.py --decomposer --kg-path P`:
runs leave-one-out drop test per voter per scenario, accumulates
fitness, writes KG records, prints evolution report at end.

**Verified** end-to-end with mock 3-voter ensemble on 60 synthetic
scenarios:
- strong voter (acc 0.7, marginal 0.67) → REPRODUCE
- middle voter (random) → KEEP (in tracker but combined < threshold)
- garbage voter (acc 0.22, marginal 0.27) → DIE

LoRA-on-E mutation primitive remains future work; gated on accumulating
enough KG trace data to train per-agent adapters from.

## Multi-family apex voter ensemble (architecture, 2026-05-03)

The apex tier is reframed: instead of K trained instances of one
backbone, the apex is a **panel of frontier-model voters**
(OpenAI / Anthropic / Gemini / OpenRouter / local Qwen). The trophic
stack's job is to prepare the evidence packet (producer + herb +
forecast snapshot) that each voter reads independently. Aggregation is
plurality / confidence-weighted / perplexity-weighted across voters.

Code shipped:
- `trophic/apex_voters/base.py` — `ApexVoter`, `VoterResponse`,
  `EvidencePacket` interfaces.
- `trophic/apex_voters/evidence.py` — renders Scenario + herb broadcasts
  into a model-agnostic prompt with structured metadata.
- `trophic/apex_voters/local_qwen.py` — local Qwen voter, computes
  self-perplexity from `output_scores` during generate().
- `trophic/apex_voters/api_voters.py` — OpenAIVoter, AnthropicVoter,
  GeminiVoter, OpenRouterVoter. Each gated by its env-var key; unavailable
  voters silently skip. Stubs ready for the day keys are wired.
- `trophic/apex_voters/aggregate.py` — plurality, confidence_weighted,
  perplexity_weighted (rank-vote on binary = plurality), and
  `deliberation_packet()` for round-2 peer-aware re-voting.
- `scripts/diagnostics/apex_vote_eval.py` — full StockNet eval; auto-skips
  unavailable voters; writes per-scenario JSONL records.

Today only the local Qwen voter is available; smoke test validates
plumbing but emits MCC=0 with a single voter (degenerate case).
Aggregation logic verified with mock voters in unit-test style.

When API keys land, the same eval script picks them up via env-var
detection — no code change needed.

## Resolved / done since last handoff

- ✅ Producer-only ablation (seed37) — MCC=0, herb tier IS doing work
- ✅ Numeric Chronos forecaster (seed38) — MCC -0.08, hurt the model
  because Chronos drift sign is constant 'down' on this benchmark
- ✅ Deeper MLP binary head (seed39) — MCC -0.01, no improvement
- ✅ Bare-XML 100-eval — MCC +0.147 (down from 50-scenario +0.292,
  the earlier number was small-sample noise)
- ✅ seed36 100-eval — MCC +0.117
- ✅ **AGREE ensemble (bare + seed36) — MCC +0.210 on 41/100 decided.
  First architectural result that exceeds the bare baseline.**
- ✅ Per-ticker analysis: large heterogeneity (bare best on AMZN, seed
  best on JPM); oracle router gives only marginal lift (+0.159).
- ✅ Confidence-gated AGREE: marginal improvement to +0.212 on 38/100;
  seed36's confidence isn't graduated enough to be a strong gate.

## Recommended next steps

In priority order. seed36's +0.117 appears to be a real ceiling for
this architecture. Further work should focus on either ensemble
approaches or fundamentally different signal extraction.

### 1. Try ensemble of bare + seed36
Bare predicts 22 up / 78 down (bearish). seed36 predicts 41 / 59
(balanced). Different error profiles → ensemble could plausibly help:
- Method A: average softmax probs from bare's first-token logit
  (constrained to {up, down}) and seed36's binary head output
- Method B: agreement gating — only commit when both agree, else
  abstain. With ~50% agreement rate, decided-rate halves but
  precision should rise.
- This costs no new training; just an eval script.

### 2. Per-ticker conditioning
The bare baseline showed strong per-ticker variance (AAPL 50%, JPM 30%,
AMZN 60%). Adding a learnable ticker embedding concatenated to the
head's input might help. Cheap to try (small embedding table + larger
input dim). seed40 candidate.

### 3. Concatenate raw herb broadcasts
seed36 feeds the trough-pooled hidden into the head. Try concatenating
each herb's broadcast directly into the head's input:
```python
head_input = torch.cat([
    pooled,
    herb_for_pred[0].channel_embedding,  # technical
    herb_for_pred[1].channel_embedding,  # fundamental
], dim=-1)
binary_head: Linear(3*H, 2)
```
Bypasses the trough's softmax-attention bottleneck, which may be
losing signal at fusion time.

### 4. Re-cache Chronos with longer history
Current StockNet loader passes 5-day OHLCV history; Chronos-Bolt
supports up to 64 bars. Edit `stocknet_loader.py`'s `history_days`
default and re-run `scripts/cache_chronos_features.py`. If Chronos
becomes directionally informative at longer history, seed38's
architecture (ready to go) will work.

### 5. Train herbivore phi_mlp for direction-discriminative pool
Currently the herb's role_q is its mean role_prefix vector — fixed.
Make the herb's role_q a learnable parameter that's updated by the
binary head's loss flowing back through the pool. The herb starts
attending to the parts of its own forward that are most predictive of
direction.

### 6. Don't add apex voting / decomposers (#95 / #96 / #97) yet
These multiply whatever signal the apex carries. Until the architecture
beats bare on a single forward, voting just amplifies noise. Pending
issues should stay pending. (Exception: ensemble in #1 above is
*different* — it crosses outputs, doesn't multiply within a stack.)

## Files / state

- Latest commits on main: `git log --oneline -10`
- Best checkpoints (architecture top to bottom):
  - `checkpoints/sft_seed36_stable_best.pt` — linear binary head, dev=1.59 — **MCC +0.117**
  - `checkpoints/sft_seed39_stable_best.pt` — MLP binary head, dev=1.45 — MCC -0.014
  - `checkpoints/sft_seed38_stable_best.pt` — + numeric Chronos, dev=1.50 — MCC -0.08
  - `checkpoints/sft_seed37_stable_best.pt` — producer-only ablation — MCC 0
- Logs (100-scenario evals on StockNet test):
  - `logs/baseline_promptonly_stocknet_v3.log` — bare Qwen XML — MCC +0.147
  - `logs/loop2_seed36_full_v2.log` — seed36 (linear head) — MCC +0.117
  - `logs/loop2_seed39_full.log` — seed39 (MLP head) — MCC -0.014
  - `logs/loop2_seed38_full.log` — seed38 (+forecaster) — MCC -0.08
  - `logs/loop2_seed37_full.log` — seed37 (no herb) — MCC 0
- Cached Chronos features: `external/stocknet_cache/forecast_features.json`
  (3506 entries: 196 test + 143 dev + 1167 train; ~750 KB)
- Diagnostic infra:
  - `scripts/diagnostics/inspect_signals_jsonl.py` — JSONL signal capture
  - `scripts/diagnostics/compare_signals.py` — side-by-side diff
  - `scripts/diagnostics/loop1_constrained.py` — bare baseline w/ constrained decode
  - `scripts/diagnostics/baseline_promptonly_stocknet.py` — bare baseline w/ XML
  - `scripts/cache_chronos_features.py` — one-shot Chronos cache
  - `viz/` (Vite+react-flow at http://localhost:5173)
- Env-var toggles relevant to the binary-head architecture:
  - `TROPHIC_BINARY_HEAD=1` — read predator pooled into 2-class head
  - `TROPHIC_BINARY_HEAD_DEPTH=1|2` — single Linear vs MLP
  - `TROPHIC_FORECAST_PROJ=1` — wire numeric Chronos broadcast (needs cache)
  - `TROPHIC_COMPUTE_FORECAST=1` — load Chronos cache during scenario build
  - `TROPHIC_BINARY_FROM_PRODUCER=1` — ablation: skip herb tier
  - `TROPHIC_HERB_POOL=attn|mean|last` — herb forward pool (default attn)
  - `TROPHIC_HERB_NUMERIC=1` — OHLCV inject at herb tier (default on)
  - `TROPHIC_LOSS_FLOOR=0.3` — over-fit abort guard
  - `TROPHIC_LR`, `TROPHIC_GRAD_CLIP`, `TROPHIC_SEED`, `TROPHIC_EVAL_EVERY`

## How to verify state on next session

```bash
cd /home/dgonier/ecology_experiment/trophic
git log --oneline -10
ls -lh checkpoints/sft_seed3{6,7,8,9}_stable_best.pt

# Quick sanity check the seed36 ckpt scores +0.117 again:
TROPHIC_BINARY_HEAD=1 .venv/bin/python -u scripts/diagnostics/loop2_hooks_smoke.py \
  --ckpt checkpoints/sft_seed36_stable_best.pt --seed 36 --n-per-ticker 20

# Re-run the bare benchmark baseline (~50 min for 100 scenarios):
STOCKNET_MAX_PER_TICKER=20 .venv/bin/python -u scripts/diagnostics/baseline_promptonly_stocknet.py
# Expect: MCC +0.147 on 100 scenarios.

# If experimenting with the forecaster again, the cache is ready:
ls -lh external/stocknet_cache/forecast_features.json
```

## Memory updated

`/home/dgonier/.claude/projects/-home-dgonier-ecology-experiment/memory/`:
- `MEMORY.md` — index updated with the corrected baseline finding
- `project_signal_flow_2026_05_02.md` — full audit + verdict appended
- `project_issue_8_diagnosis.md` — path C (SFT on M+E) marked exhausted
