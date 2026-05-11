# v4 Phase-2.5 Smoke Gate — RESULTS

**Verdict: PASS**

- runner rc: `0`
- elapsed: `187s`
- eval jsonl: `/tmp/v4_smoke_b_out/eval_1778456609.jsonl`
- runner log: `/tmp/v4_smoke_b.log`

## PART A — contract verifier

Part A was run prior to Part B. See `scripts/v4_smoke_part_a.py` and its stdout for the full assertion list. Part A reports `SMOKE_PART_A: PASS` on exit 0 — that exit code is enforced by the smoke-gate driver before this report is generated.

Assertions covered (18 total):
- happy path h20 BUY (all 3 strategies BUY, aggregator BUY, all 4 validators accept)
- happy path h20 SELL (4 validators accept; sizing skipped on SELL per phase2-D)
- adversarial: h20 BUY size=50 → horizon_sizing rejects
- adversarial: BUY committed to h5 with negative forecast.h5 → forecast_consistency rejects
- adversarial: BUY h5 with alpha=50 below dynamic floor → edge_floor rejects
- h60 edge-floor waiver: same alpha=50 with h60 → accepts (waived)
- re-buy preservation: BUY h60, BUY h1 same ticker, advance 3d, SELL → min_hold rejects (committed to h60)
- committee: 2BUY+1SELL → strict-majority BUY
- committee: 1BUY+1SELL+1BUY equal-conviction → strict-majority BUY
- committee: all votes <0.20 conviction → None (aggregator drops)
- committee: all HOLD votes → None
- kelly edge-floor end-to-end: kelly HOLDs on low alpha, 2/3 BUY → aggregator BUY
- full run_tax_aware_chain on clean h20 BUY → accepts

## PART B — 3-day ECO-QWEN mini-sweep

### Headline numbers

| metric | value |
|---|---|
| days run | 3 |
| total raw orders (post-committee) | 5 |
| total validated orders | 4 |
| total rejections | 1 |
| rejection rate (rej / (val+rej)) | 20.0% |
| order ceiling (mission threshold) | 12 |
| rejection ceiling (mission threshold) | 50% |
| starting equity | $100,000 |
| final equity (day-3) | $100,653 |
| day-3 return | +0.65% |

### Horizon distribution (validated orders)

| horizon | count |
|---|---|
| h1 | 0 |
| h5 | 0 |
| h20 | 0 |
| h60 | 4 |

### Per-day equity trajectory

| date | raw | validated | rejected | equity | cash | inv% |
|---|---|---|---|---|---|---|
| 2026-02-03 | 4 | 3 | 1 | $99,986 | $39,976 | 60 |
| 2026-02-04 | 1 | 1 | 0 | $100,906 | $19,789 | 80 |
| 2026-02-05 | 0 | 0 | 0 | $100,653 | $19,792 | 80 |

### Sample validated orders (up to 5)

| date | side | ticker | size_pct | primary_horizon | expected_alpha_bps |
|---|---|---|---|---|---|
| 2026-02-03 | BUY | CVX | 20.0 | h60 | 1200.0 |
| 2026-02-03 | BUY | WMT | 20.0 | h60 | 600.0 |
| 2026-02-03 | BUY | AAPL | 20.0 | h60 | 500.0 |
| 2026-02-04 | BUY | ABBV | 20.0 | h60 | 1200.0 |

### Notes / flags

- (no flags; clean pass)

### Soft-fails for phase 3 to watch

- (no soft-fail items)
