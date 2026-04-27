# 04 — NeurIPS Paper Decision

**Status**: DECISION-PENDING-ON-IPO-RESULT
**Triggers**: IPO seed 7 (post-migration) finishes; held-out eval lands; StockNet smoke runs.

## Question

Does the trough-as-transformer migration justify a NeurIPS submission, given the user already has 3 papers in flight (FAccT, ICCL, NeurIPS-Phase1)?

## Current evidence

| Signal | Value | Direction |
|---|---|---|
| Tests passing | 51/51 | ✓ structurally sound |
| Pre-migration SFT (dev) | 0.326 | baseline |
| Post-migration SFT (dev) | 0.331 | tied with baseline at SFT |
| Post-migration SFT (held-out 68) | 0.257 | -22% generalization gap (honest) |
| Pre-migration IPO (dev) | 0.388 | baseline (best pre-migration) |
| Post-migration IPO @ step 400 | 0.543 | **+40% over baseline, training in progress** |
| Pre-migration IPO held-out | not yet measured | TBD |
| Post-migration IPO held-out | not yet measured | TBD |
| External-benchmark number | not yet measured | TBD (StockNet) |

## Decision tree

After IPO finishes and held-out eval runs:

| post-IPO dev | post-IPO held-out | StockNet subsample acc | Decision |
|---|---|---|---|
| ≥0.50 | ≥0.40 | ≥60% | **PAPER GO** — strong claim across three signals |
| ≥0.50 | ≥0.40 | 55–60% | Paper, but with weaker external claim — write up as "matched LLM-baselines on real data + novel architecture" |
| ≥0.50 | <0.30 | any | **Generalization-gap fix first.** No paper until held-out is honest. |
| <0.45 dev | any | any | **Ablation phase, not paper.** Architecture isn't paying off; identify which mechanism is the load-bearing one. |
| 0.45–0.50 | ≥0.35 | any | Workshop paper or NeurIPS-Datasets&Benchmarks track, not main NeurIPS. |

## Required for any NeurIPS submission

Hard reviewer-blockers we'd have to clear:

1. **Multi-seed.** Single-seed numbers won't pass review. Need 3+ seeds × the matrix. ~2 days of GPU time per seed for SFT+IPO.
2. **Ablations.** Six new mechanisms need to be ablated to show which actually matter. Even a 4-row ablation table satisfies most reviewers.
3. **Held-out test set with disjoint distribution.** Already have this (68 fresh-ticker scenarios). Could expand to 200+.
4. **External benchmark with published baselines.** StockNet (or StockBench) at minimum. This is what `01-stocknet-smoke.md` is for.
5. **A second domain** (stretch). StockBench would count if we run it. Otherwise this is a single-domain paper.

## Soft requirements (would strengthen the paper)

- **Calibration analysis.** Brier scores on direction probability. (See `03-bayesian-nodes.md` — full Bayesian build is overkill, but a calibration evaluation on the existing predator's confidence field is cheap and reviewer-loved.)
- **Population dynamics study.** We log alive/killed/spawned per tick. Plotting this across training would be a unique contribution — no MoE paper has this signature because static MoE doesn't have it.
- **Head specialization emergence.** We log head-specialization entropy. Showing it drops over training is a nice "the architecture self-organizes" plot.
- **Comparison to static MoE baseline.** This is the most important comparison and we don't have it. To say "dynamic MoE > static MoE" we'd need to train a fixed-population variant. ~1 day.

## Time budget if PAPER GO

- 2 weeks for multi-seed training + ablations + StockNet full
- 1 week for paper draft
- 1 week for revision + figures + appendix
- = **4 weeks total** to a submission-quality draft

User has 3 other papers in flight. If those have hard near-term deadlines, this paper is competing with them for revision time, not just GPU time.

## Recommendation framework (decide after IPO + held-out)

The right move depends entirely on the next two numbers (post-IPO dev + held-out). Don't decide now. Don't draft now. Just:

1. Wait for IPO to finish. ~30 min.
2. Run held-out on `ipo_seed7_best.pt`. ~30 min.
3. Run StockNet smoke. ~1 day.
4. Look at the table and apply the decision tree above.

If the table says PAPER GO, then start with multi-seed + ablations *before* paper drafting. Single-seed paper drafts are how reviewers reject otherwise-good work.

## Comparison to user's existing 3 papers

I haven't read those papers. To honestly recommend "should this be the third paper or a fourth," I'd need to:
- Read FAccT draft retrospectively (what's the contribution, how strong is it)
- Read ICCL draft (same)
- Read existing NeurIPS draft (same)
- Compare each on (a) strength of result, (b) revision time remaining, (c) deadline pressure

That comparison is **outside this todo** but should happen before paper-go.
