# Experiments Log

Append-only log of training runs and eval results. Newest at bottom.

## Result tables

### Predator reward (XML rule-based, 6 fields × weights)

| Run | Date | Seed | Architecture | Dev (14) | Held-out (68) | Notes |
|---|---|---|---|---|---|---|
| sft_seed1_best | 2026-04-25 | 1 | pre-migration | 0.326 | TBD | SFT-only baseline |
| sft_seed7_best | 2026-04-26 | 7 | post-migration | 0.331 | 0.257 | matches pre-migration SFT; honest -22% gap on held-out |
| ipo_seed1_best | 2026-04-26 | 1 | pre-migration | 0.388 | TBD | IPO v3 reference |
| ipo_seed7 (in flight) | 2026-04-26 | 7 | post-migration | 0.543 (best @ step 400) | TBD | **+40% over pre-migration IPO**; eval@500/600 = 0.521 (2 non-improvements) |

### Per-component eval losses (post-migration SFT seed 7)

| step | total | herb.tech | herb.fund | pred.short |
|---|---|---|---|---|
| 50 | 2.236 | 2.07 | 3.06 | 1.58 |
| 100 | 1.434 | 0.91 | 2.13 | 1.26 |
| 150 | 1.023 | — | — | — |
| 200 | 0.630 | — | — | — |
| 350 | 0.394 | 0.28 | 0.34 | 0.56 |
| 400 | 0.401 | 0.26 | 0.35 | 0.60 |
| 450 | 0.379 | 0.29 | 0.33 | 0.52 |
| 500 | 0.327 | 0.33 | 0.19 | 0.47 |
| 550 | **0.279 (best)** | 0.26 | 0.13 | 0.45 |
| 600 | 0.296 | 0.24 | 0.14 | 0.51 |

8x reduction in eval loss over 550 steps. Best at step 550, +1 non-improvement at step 600 (natural stopping point).

### IPO eval reward (post-migration seed 7)

| step | EVAL_MEAN_SCORE | new best? |
|---|---|---|
| 100 | 0.414 | ✓ (vs 0.388 baseline) |
| 200 | 0.471 | ✓ |
| 300 | (mid-instability) | — |
| 400 | **0.543** | ✓ |
| 500 | 0.521 | ✗ |
| 600 | 0.521 | ✗ |
| 700 | TBD | TBD |
| 800 | TBD | TBD |

## Notable observations

### Generalization gap (post-migration SFT)
Dev reward 0.331 → held-out reward 0.257. **-22% drop** when tickers are disjoint from training. This is a real, expected signal for any SFT-only training; documents it for paper purposes.

### Mode collapse on novel tickers
Both SFT seed 7 and IPO seed 7 show the predator emitting one or two "hub" tickers (BBRY, MSFT) regardless of input ticker. The XML *structure* is correct, the *content conditioning on input* is weak. This is the prior bottleneck the migration was supposed to address; needs ablation to determine if any of the new mechanics moved the needle.

### IPO instability spikes
h_w spikes up to ±22.5 (loss=306) at steps 450-550. KL anchor absorbs and recovers within ~2 steps. Same dynamics as the pre-migration IPO v3 run; not new instability.

### Skip α drift
Predator's skip_weight (sigmoid of learnable scalar) drifted from 0.523 → 0.490 across SFT training. Sigmoid is hovering around 0.5 — predator weighing producer-skip and herbivore paths roughly equally. This is the "is the herbivore tier earning its keep?" diagnostic; current answer: marginally yes (α < 0.5 means herbivore path > producer skip).

### Head specialization
At end of SFT seed 7: 5 distinct heads dominate different slots. After 100 ticks: head_per_slot distribution = {0:1, 1:4, 2:27, 3:1, 6:14, 7:4}. Head 2 is dominant, head 6 is secondary. Suggests the trough is *partially* developing niches but with strong concentration on one head.

## Open evaluation tasks

- [ ] Run held-out eval on `ipo_seed1_best.pt` (pre-migration baseline). We have 0.388 dev but never the held-out number.
- [ ] Run held-out eval on `ipo_seed7_best.pt` (post-migration, in flight).
- [ ] Run StockNet smoke (top-5 tickers, 2-week test window) on `ipo_seed7_best.pt`. Decision criterion: ≥60% acc → expand, 55-60% → ablate, <55% → debug ticker conditioning.
- [ ] Multi-seed: replicate sft+ipo with seeds 8, 9, 10 to get error bars.
- [ ] Ablations: train with each of the six new mechanisms toggled off, measure delta vs full architecture.

## Methodology notes

- **Dev set**: 14 scenarios, named `eval_*` in `trophic/training/scenarios.py`. Same tickers as training (16 large-caps).
- **Held-out test**: 68 scenarios, named `holdout_*` in `trophic/training/holdout_scenarios.py`. **Disjoint tickers** (16 fresh large-caps: BRK.B, V, JNJ, WMT, PG, UNH, HD, BAC, DIS, PYPL, T, PFE, KO, PEP, XOM, CVX). Same archetype distribution as training.
- **External benchmark**: ACL-18 StockNet, 88 stocks, binary direction prediction from tweets + price history. Public dataset at github.com/yumoxu/stocknet-dataset. Loader at `trophic/training/stocknet_loader.py`.
- **Reward function**: weighted sum of XML field matches: ticker_w=0.4, direction_w=0.25, magnitude_w=0.15, horizon_w=0.10, sigma_w=0.05, confidence_w=0.05. Defined in `trophic/training/xml_schema.py`.
- **Train-eval contamination**: dev shares tickers with train (pre-migration concern, also true for the 0.388 IPO baseline). Held-out is disjoint. **Always report both.**
