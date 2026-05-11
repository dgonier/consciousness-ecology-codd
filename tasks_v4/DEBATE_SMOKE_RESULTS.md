# Debate Smoke Gate Results (phase3-A-05f)

**Verdict: PASS**

- runner-log: `/tmp/v4_debate_smoke.log`
- eval-jsonl: `data/firehose_eval/eval_1778463672.jsonl`
- transcript: `data/firehose_eval/debate_transcripts/v4_debate_smoke.jsonl`
- days observed: 3
- total orders (PM, summed): 10
- total rejections: 8
- rejection rate: 44.4%

## Caught debate parse failures
(These are caught by the runner's try/except — a no-trade fallback day. Counted toward zero-trades hard-fail check.)
- 2026-02-05: `ValidationError: 10 validation errors for DebatePhase1Proposal`

## Per-day PM line

| date | equity | cash | inv% | orders | rejected |
|---|---|---|---|---|---|
| 2026-02-03 | $99,993 | $86,243 | 14.0% | 4 | 7 |
| 2026-02-04 | $100,073 | $77,486 | 23.0% | 6 | 1 |
| 2026-02-05 | $100,037 | $77,486 | 23.0% | 0 | 0 |

## Per-predator equity (final day's leaderboard)

| predator | philosophy | equity | valid | rejected | active_theses |
|---|---|---|---|---|---|
| momentum | momentum | $24,978 | 0 | 0 | 1 |
| value | value | $25,076 | 0 | 0 | 2 |
| mean_revert | mean_revert | $25,014 | 0 | 0 | 0 |
| event_driven | event_driven | $24,969 | 0 | 0 | 0 |

## Phase coverage (days a predator emitted ≥1 item)

| predator | phase1_propose | phase2_respond | phase3_revise | phase4_commit |
|---|---|---|---|---|
| event_driven | 2 | 2 | 2 | 2 |
| mean_revert | 2 | 2 | 2 | 1 |
| momentum | 2 | 2 | 2 | 2 |
| value | 1 | 2 | 2 | 2 |

## Diversity check (Jaccard on thesis tickers)

Thesis tickers per predator (across all observed days):

- **event_driven**: ['CVX', 'WMT']
- **mean_revert**: ['AMZN', 'CSCO', 'JNJ']
- **momentum**: ['AMZN', 'CVX', 'MSFT', 'WMT', 'XOM']
- **value**: ['CVX', 'JNJ', 'JPM', 'KO', 'PG']

Pairwise Jaccard:

| a | b | shared | size_a | size_b | jaccard |
|---|---|---|---|---|---|
| event_driven | mean_revert | [] | 2 | 3 | 0.00 |
| event_driven | momentum | ['CVX', 'WMT'] | 2 | 5 | 0.40 |
| event_driven | value | ['CVX'] | 2 | 5 | 0.17 |
| mean_revert | momentum | ['AMZN'] | 3 | 5 | 0.14 |
| mean_revert | value | ['JNJ'] | 3 | 5 | 0.14 |
| momentum | value | ['CVX'] | 5 | 5 | 0.11 |

Average pairwise Jaccard: **0.16** (diversity goal: < 0.7)

## Order tickers per predator (realized footprint)

- **event_driven**: ['CVX', 'WMT']
- **mean_revert**: ['CSCO', 'WMT']
- **momentum**: ['AMD', 'CVX', 'WMT', 'XOM']
- **value**: ['CVX', 'JNJ', 'JPM', 'KO', 'PG']

## Token-budget summary

Max prompt tokens: **n/a** (DSPy/vLLM did not surface prompt-token usage in the runner log). Indirect check: no day raised a context-overflow error and no `DEBATE FAILED` line cited a context-length issue.

## Sample debate excerpt

> momentum → mean_revert on AMZN: Your AMZN h5 long thesis relies on relative weakness, but AMZN has not shown the volume expansion or breakout structure required for a momentum entry. The mean-revert edge here is likely too small (+180bps) to overcome the tax drag of an h1

## Parse-error status

Phase-2 `DebatePhase2Responses` parse error from the earlier 1-day smoke (05e) — Qwen-4B returned a list-shape (either bare list of `Response` dicts or per-peer wrappers) where DSPy expected the canonical `{predator_id, responses}` object — was diagnosed and fixed in this mission. Fix is in `trophic/beliefs/debate.py`:

  1. `DebatePhase2Responses` gained a `mode="before"` model validator (`_normalize_shape`) that detects either list form and rewrites it into the canonical wrapper before Pydantic validates the fields.
  2. `_coerce_responses` was extended to route raw-list output through the normalizer and inject the orchestrator's `predator_id` when the flat-list form drops it.
  3. The Phase-2 instruction block in `predator_prompts.py` / `debate.py` gained an explicit `OUTPUT SHAPE` paragraph directing the model to emit ONE wrapper object, not a list.
  4. Three new tests in `tests/test_debate_mechanism.py` cover (a) flat list of Response dicts → canonical wrapper, (b) list of per-peer wrappers → flattened, (c) canonical dict unchanged.

