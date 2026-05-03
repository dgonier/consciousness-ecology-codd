# Trophic — Claude resume notes

## On "continue where you left off"

Read **`HANDOFF.md`** first. It is the single source of truth for the
current state of the project. Do not act on prior memory or git history
without reconciling against `HANDOFF.md` — both can be stale relative to
what was actually decided last session.

Specifically check:
- The "TL;DR" section for whether the project has hit its current bar
  (bare Qwen3-4B + benchmark XML prompt = MCC +0.292 on 50 StockNet test
  scenarios; trained variants must beat this to justify themselves).
- The "Numbers" table for what's been tried and what each scored.
- The "Recommended next steps" section — execute the highest-priority
  unfinished item there unless the user redirects.

After reading HANDOFF.md, run the sanity-check commands at the bottom of
that file to verify the state on disk matches what the doc claims, and
report any discrepancies before starting new work.

## Project basics

- Working dir: `/home/dgonier/ecology_experiment/trophic`
- Python env: `.venv/` (use `.venv/bin/python` directly; do not activate)
- Frozen base model: Qwen3-4B
- Benchmark: StockNet ACL-18 next-day binary direction prediction
- Primary metric: **MCC** (not accuracy — class is ~64% up so accuracy
  can be high while MCC is 0). MCC=0 means the predictor carries no
  information beyond the class prior.

## Hard rules learned this project

- **Do not trust loose-pattern parsers on free-text model output.** The
  apex emits long meta-explanations; loose regex patterns will hallucinate
  predictions from schema enumerations. Use only strict XML or first-token
  matching (already enforced in `xml_schema.parse_prediction`).
- **Always re-verify MCC numbers against the strict parser before
  declaring a win.** Several "+0.358 MCC" announcements during this
  project were parser hallucinations.
- **The bare-Qwen baseline IS the bar.** Beat MCC +0.292 on the 50- or
  100-scenario StockNet test split with the benchmark's XML prompt
  (`scripts/diagnostics/baseline_promptonly_stocknet.py`). Anything else
  is just churn.
- **Don't add apex voting / decomposers / perplexity weighting yet.**
  Those issues (#95, #96, #97) multiply whatever signal the apex carries.
  Until a single trained forward beats bare, voting just amplifies noise.
- **Training crashes silently around step 150** (likely CUDA OOM during
  dev eval inside the ORPO grad-enabled forward). Best ckpts always
  save by step 100/150 so this hasn't blocked progress, but it's a
  ticking item if longer training becomes necessary. `eval_loss` already
  has `empty_cache` per scenario; that didn't fix it.

## How to inspect signal flow

When debugging "why is the model not differentiating between scenarios":

```bash
# 1. JSONL signal capture for two contrasting scenarios:
.venv/bin/python -u scripts/diagnostics/inspect_signals_jsonl.py \
  --ckpt checkpoints/sft_seed36_stable_best.pt \
  --scenario stocknet_test_AAPL_2015-10-01  # target=down
.venv/bin/python -u scripts/diagnostics/inspect_signals_jsonl.py \
  --ckpt checkpoints/sft_seed36_stable_best.pt \
  --scenario stocknet_test_AAPL_2015-10-02  # target=up

# 2. Diff at every tier:
.venv/bin/python -u scripts/diagnostics/compare_signals.py \
  logs/signals/stocknet_test_AAPL_2015-10-01.jsonl \
  logs/signals/stocknet_test_AAPL_2015-10-02.jsonl

# 3. Or open the visual graph (vite + react-flow):
cd viz && npm run dev   # http://localhost:5173
```

## Tight iteration loops (run before committing real training cycles)

- Loop 1 (~30s): bare-Qwen prompt smoke
  `.venv/bin/python -u scripts/diagnostics/loop1_prompt_smoke.py`
- Loop 1c (~5min): bare-Qwen first-token / constrained baseline
  `.venv/bin/python -u scripts/diagnostics/loop1_constrained.py`
- Loop 2 (~70s for 5 scenarios; ~10min for 100): trained-ckpt forward smoke
  `.venv/bin/python -u scripts/diagnostics/loop2_hooks_smoke.py --ckpt <path>`
- Loop 3 (~13min): 50-step training smoke
  `.venv/bin/python -u scripts/diagnostics/loop3_train_smoke.py --seed N`

## Auto memory

Persistent memory lives at
`/home/dgonier/.claude/projects/-home-dgonier-ecology-experiment/memory/`.
The relevant entries are indexed in `MEMORY.md`; the most load-bearing
ones for this project are `project_signal_flow_2026_05_02.md` (the audit
that found Tier 5 was the bottleneck) and `project_issue_8_diagnosis.md`
(prior dead ends — paths A, B, C all marked exhausted).

## Tone

- The user (`dgonier@gmail.com`) is the lead architect of this project
  and writes deeply thoughtful design docs. Read what's already in
  `HANDOFF.md` and the memory files before suggesting "new" approaches —
  most have been tried.
- They prefer terse responses with concrete numbers, not narrated work
  in progress.
- Don't say "you're absolutely right". Don't add summaries the user
  could read off a diff.
- They run the GPU on a shared box; ask before launching anything that
  takes more than ~10 min, and free GPU promptly when asked.
