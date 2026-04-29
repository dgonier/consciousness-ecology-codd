# Experiments Log

Append-only log of training runs and eval results. Newest at bottom.

## Result tables

### Predator reward (XML rule-based, 6 fields × weights)

| Run | Date | Seed | Architecture | Dev (14) | Held-out (68) | Notes |
|---|---|---|---|---|---|---|
| sft_seed1_best | 2026-04-25 | 1 | pre-migration | 0.326 | TBD | SFT-only baseline |
| sft_seed7_best | 2026-04-26 | 7 | post-migration | 0.331 | 0.257 | matches pre-migration SFT; honest -22% gap on held-out |
| ipo_seed1_best | 2026-04-26 | 1 | pre-migration | 0.388 | **0.289** | IPO v3 reference. Held-out gap = -25% from dev. |
| ipo_seed7_best | 2026-04-26 | 7 | post-migration | **0.370** (rule-based) / 0.543 (LLM-judge @ step 400) | **0.334** | early-stopped step 775 (3 non-improvements at 500/600/700 = 0.521 judge). **Generalization gap narrowed from -22% (SFT) to -10% (IPO).** Held-out improved +30% vs SFT seed 7 (0.257 → 0.334), more than dev did (+12%). Two metrics disagree on dev: judge says +40% over baseline, rule-based says -4.6%; likely from mode collapse to MSFT/AMD which keeps judge happy but kills ticker-match weight. **StockNet smoke (top-5 × 10 days, 50 scenarios): acc=0.720, MCC=0.000 — at chance, model emits constant "up". See #5/#6.** |
| sft_seed8_best | 2026-04-27 | 8 | post-migration + #6 fix | TBD (run via ipo_seed8) | TBD | **eval_loss=0.1919** at step 600 (-31% vs seed 7's 0.279 best). #6 architectural fix lands: hunter_state = role_q + prey.mean. Reached seed 7's all-time best at step ~370 (150 steps earlier). herb.technical converged to ~0.001 range (vs seed 7's ~0.28). Predator final eval components in 0.26-0.39 range. Skip α held at 0.500 (no longer drifting down — herbivore tier is now informative). |
| **ipo_seed8_best** | 2026-04-27 | 8 | post-migration + #6 fix | **0.414** | **0.372** | **EXCEEDS pre-migration IPO baseline (0.388 dev / 0.289 held-out) by +6.7% / +28.7%.** First run in project history to clear 0.40 dev. Held-out 0.372 also clears every prior held-out (sft_seed7 0.257, ipo_seed1 0.289, ipo_seed7 0.334). Generalization gap identical to pre-fix at -10% — both dev and held-out moved up by similar absolute amounts (+0.044 dev / +0.038 held-out), confirming the gain is real, not memorization. IPO judge metric peaked 0.564 step 400 (+3.9% over seed 7's 0.543), early-stopped step 625. **The #6 fix validates end-to-end.** |
| sft_seed9_best (v1, premature) | 2026-04-27 | 9 | post-migration + #6 fix + ENVSTREAM | **0.050** | n/a | First seed 9 attempt; killed at step 100 due to eval_decode wall-time bottleneck (192 max_new_tokens × greedy decode on undertrained model = 7-15 min/eval). Best eval_loss 1.142 — never converged. Predator emitted constant GOOG. Preserved as `sft_seed9_best_premature_step100.pt`. |
| **sft_seed9_best (v2, post-eval-cap-fix)** | 2026-04-27 | 9 | post-migration + #6 fix + ENVSTREAM + eval-cap fix | TBD (run via ipo_seed9) | TBD | **eval_loss=0.208** at step 550 (vs seed 8's 0.192). Eval-cap fix `TROPHIC_EVAL_MAX_TOKENS=96` removed the bottleneck; trajectory: 2.273 → 1.192 → 0.834 → 0.557 → 0.411 → 0.343 → 0.286 → 0.258 → 0.241 → **0.208** → 0.220. Eight new bests, one non-improvement at step 600 (natural stop). Within ~8% of seed 8's best — EnvStream architecture is competitive at SFT, neither lift nor regression at this stage. The decisive #8 test is IPO + StockNet. |
| **ipo_seed9_best (v2, EnvStream)** | 2026-04-27 | 9 | post-migration + #6 fix + ENVSTREAM | **0.417** | **0.346** | **Mixed result**: dev +0.7% (best-ever), held-out -7% vs ipo_seed8 (0.372). Dev/held-out gap widened from -10% (seed 8) to **-17%** (seed 9 v2). Held-out still +19% above pre-migration IPO baseline (0.289). Judge eval = 0.607 at step 100 (+7.6% over seed 8's 0.564 ceiling). Early-stopped step 325 after h_w spike at step 200 (loss=496, h_w=-17.25). **StockNet smoke: ACCURACY=0.720, MCC=0.000** — bit-identical to ipo_seed7/seed8, predator still emits constant "up". EnvStream is architecturally **orthogonal to direction collapse**: lifts training-set metrics but doesn't change what the policy emits at decode. The bottleneck is reasoning structure or pretraining distribution, not input substrate. #8 routes to #11 (FinCoT prompting) and/or #10 (per-species CPT). |

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
Both SFT seed 7 and IPO seed 7 show the predator emitting one or two "hub" tickers (BBRY, MSFT, AMD) regardless of input ticker. The XML *structure* is correct, the *content conditioning on input* is weak. This is the prior bottleneck the migration was supposed to address; needs ablation to determine if any of the new mechanics moved the needle.

### Migration helps on held-out (the apples-to-apples comparison)
Pre-migration IPO held-out: **0.289**. Post-migration IPO held-out: **0.334**. **+15.6%**.

This is the comparison that matters for paper purposes — fresh tickers, same eval harness, same reward function. Pre-migration's dev-to-held-out gap was -25% (0.388 → 0.289); post-migration's was -10% (0.370 → 0.334). The post-migration architecture both achieves *higher* held-out and *closes the generalization gap*. That's a real result.

The dev-set picture (post-migration 0.370 vs pre-migration 0.388 = -4.6%) is the misleading one. Dev shares tickers with training and rewards memorization; held-out doesn't.

### Two metrics, two stories (post-IPO seed 7)
The IPO trainer's internal `EVAL_MEAN_SCORE` (LLM-judge from Bedrock Claude) reported 0.543 at step 400 — a **+40%** lift over the 0.388 baseline, which had been the headline. But the rule-based field-wise reward on the same checkpoint scored 0.370 on dev — a **-4.6%** regression vs the same pre-migration IPO baseline.

These metrics are measuring different things:
- **LLM-judge** rewards plausible-looking, well-formed predictions. Forgiving on factual errors if presentation is good.
- **Rule-based reward** weights ticker_w=0.4 most heavily; mode collapse on tickers tanks this score regardless of how well-formed the rest is.

The honest paper-grade story: the migration's IPO appears to **trade off ticker-conditioning for output-quality** (judge-perceived plausibility goes up; rule-based ticker-match goes down). Whether this is a net win depends on which metric you trust. For the decision tree in `todo/04-paper-decision.md`, this means we're now in the "<0.45 dev" branch on rule-based — **ablation phase, not paper**, until we figure out which of the six new mechanics caused the ticker-conditioning regression.

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

## #10 phase 1 — FinCoT LoRA + co-training (negative result, 2026-04-28)

### What we did
Per #10, train a Qwen3-4B LoRA on TheFinAI/Fino1_Reasoning_Path_FinQA (5499 rows of multi-step financial reasoning), then co-train Channels (SFT + IPO) with the LoRA frozen and active throughout. The architectural claim was that the bottleneck for #8 (StockNet direction collapse) is reasoning-conditioning of evidence, and FinCoT installs that as a default behavior.

Two co-training runs as the LoRA grew:

| run | LoRA | LoRA eval_loss | dev | held-out | StockNet ACC | StockNet MCC | judge peak |
|---|---|---|---|---|---|---|---|
| ipo_seed12 | seed11_partial_step50 | 0.898 | **0.423** | 0.328 | 0.720 | 0.000 | 0.614 |
| ipo_seed14 | seed14_step200 (4× more trained) | 0.862 | 0.280 | (skipped) | 0.720 | 0.000 | 0.514 |
| ipo_seed8 (no LoRA, ref) | — | — | 0.414 | **0.372** | 0.720 | 0.000 | 0.564 |

### What we found
1. **More FinCoT training made things worse.** seed14's 4×-more-trained adapter dropped dev reward from 0.423 → 0.280. Predator collapsed to "MSFT up" on nearly every dev scenario (ticker AND direction collapse simultaneously, where seed12 had only direction collapse).
2. **MCC stayed 0.000 across all three runs.** Pre-LoRA baseline, partial-LoRA co-train, full-LoRA co-train: same constant-"up" failure on real ACL-18 data. FinCoT does not touch the bottleneck.
3. **Held-out regression with even the partial LoRA.** seed12 held-out 0.328 < seed8 held-out 0.372. The dev gain from co-training was memorization of training tickers, not real generalization.
4. **Architecture is correct, dataset is wrong.** Co-trained Channels read LoRA hidden states cleanly — no schema collapse, well-formed XML output throughout. The mechanism works; FinQA's reasoning style (calculate ratios, extract numbers from tables) is just not what installs directional priors for stock movement.

### What this rules out
- "Reasoning conditioning is the missing input distribution for #8." False — installing reasoning structure via LoRA does not break direction collapse on tweet+price data.
- "More LoRA training → better downstream." False — at least for FinCoT-on-Qwen3-4B, more training overfits the model toward verbose financial commentary that crowds out the structured XML schema.

### What this does NOT rule out
- A *different* species CPT (e.g., SocialSignal LoRA on tweet-sentiment data, or Disclosure LoRA on 10-K filings) could break #8.
- The architectural co-training story remains intact. The eval scripts now have the LoRA hook (`scripts/eval_xml_checkpoint.py`, `scripts/eval_stocknet.py`, `scripts/diagnostics/eval_holdout_general.py`), so any future LoRA can be evaluated through this pipeline directly.

### Operational notes
- **FinQA stall (4× reproduced)** — PEFT autograd hangs (process state `SNl`, GPU 100%, log frozen) on long-context FinQA rows around step 50-200. Pre-filtering rows by tokenized length (`max_input=1280`, `max_target=720`, drops 22% of dataset) survives past step 200 but stalls again at step 230. Root cause unknown; treated pragmatically as "best-of-200-steps is the LoRA we use."
- **Best LoRA artifact: `checkpoints/interrogator_fincot_lora/seed14_step200/`** (47 MB, eval_loss 0.862). Even though it didn't help on the trophic eval cascade, it's a correctly-trained adapter — preserved as the "what FinCoT-on-Qwen3-4B looks like at adequate training" reference.
- **IPO peak-then-decline** confirmed across both seed12 and seed14: judge metric peaks at step 100, then declines through step 300-400. Best checkpoint reliably at step 100. This is now a known property of our IPO run shape.

### Where this leaves us (path A)
Path (A) — "more FinCoT" — closed. Pivoting to path (B) of #10's plan: SocialSignal LoRA on tweet sentiment.

## #10 phase 1 path B — SocialSignal LoRA + co-training (negative result, 2026-04-28)

### What we did
Trained a Qwen3-4B LoRA on TimKoornstra/financial-tweets-sentiment (38,091 tweet-noisy rows, MIT license, directional bull/bear/neutral labels) — the input modality StockNet's tweet stream actually contains. Then co-trained Channels (SFT + IPO) with the SocialSignal LoRA active throughout.

LoRA training itself was clean: filter dropped only 22 of 38k rows, training survived past step 200 and 300 without the FinCoT stall, dev loss reached 0.142 at step 300 (6× lower than FinCoT seed14's 0.862 best), early-stopped at step 500 cleanly.

Co-training was healthier than the FinCoT runs in shape:
- SFT seed16: dev 2.60 → 1.20 → 0.77 → 0.73 → 0.48 → 0.30 (final dev 0.301)
- IPO seed16: judge 0.443 → 0.436 → 0.457 → 0.486, **monotone improvement through step 400** — unlike FinCoT's early-peak-decline shape

### Eval cascade
| run | LoRA | dev | StockNet ACC | StockNet MCC | predator default |
|---|---|---|---|---|---|
| ipo_seed8 (no LoRA, ref) | — | 0.414 | 0.720 | 0.000 | "up" |
| ipo_seed12 | FinCoT (50 steps) | **0.423** | 0.720 | 0.000 | "MSFT up" |
| ipo_seed14 | FinCoT (200 steps) | 0.280 | 0.720 | 0.000 | "MSFT up" |
| **ipo_seed16** | **SocialSignal (300 steps)** | **0.298** | **0.280** | **0.000** | **"MSFT/GOOGL down"** |

StockNet seed16 confusion matrix: TP=0, TN=14, FP=0, FN=36 — predator emits "down" on every single one of 50 scenarios. Mirror image of seed12/14's "up" collapse.

### What the cross-checkpoint comparison reveals
**Species CPT installs a directional default but does NOT give the predator the ability to read the input.** Three independent LoRAs on three different datasets, three different "constant" prediction patterns:
- No LoRA → bullish-default (matches majority class of pretraining + ACL-18 test)
- FinCoT LoRA → bullish-default (FinQA's reasoning style is mostly answering "what is X" with positive numerical reasoning; doesn't shift the prior much)
- SocialSignal LoRA → bearish-default (the dataset is ~32% bearish vs ~22% in pretraining baseline; LoRA shifts the prior toward bearish)

Each LoRA shifts the *prior* the model expresses. None of them fixes the *conditioning* the predator should be doing on its actual input. **The model has a default; it does not have a function.**

### What this rules out
- "Species CPT (the right pretraining-distribution shift) is the cure for #8." False across two completely different dataset distributions (reasoning vs sentiment).
- "More LoRA training → more directional signal." False — more training in either direction makes the default more rigid, not the conditioning more responsive.

### What this implicates
The bottleneck is now strongly localized: **the predator's cross-attention output is not input-responsive across StockNet inputs**. The #6 fix (input-conditioned hunter_state via `role_q + prey_t.mean()`) was diagnosed as solved at the SFT-loss level but the StockNet behavior says otherwise. Possibilities:
1. `prey_t.mean()` is itself approximately constant across StockNet days for the same ticker (herbivore output collapses upstream).
2. The Channel cross-attention output has nonzero per-input variance but the predator's decode head ignores it (e.g., layernorm + skip dominates).
3. The decode greedy/sampling temperature is washing out small attention-output differences.

### Where this leaves us
Both paths (A) and (B) of #10 closed. **Pivoting to architectural diagnostic** on the predator's input-responsiveness: measure Channel cross-attention output cosine similarity across StockNet days for the same ticker, and herbivore prey_t variance over the same axis. The decisive question is now: where exactly does input variance get squashed in the trophic stack?

## Methodology notes

- **Dev set**: 14 scenarios, named `eval_*` in `trophic/training/scenarios.py`. Same tickers as training (16 large-caps).
- **Held-out test**: 68 scenarios, named `holdout_*` in `trophic/training/holdout_scenarios.py`. **Disjoint tickers** (16 fresh large-caps: BRK.B, V, JNJ, WMT, PG, UNH, HD, BAC, DIS, PYPL, T, PFE, KO, PEP, XOM, CVX). Same archetype distribution as training.
- **External benchmark**: ACL-18 StockNet, 88 stocks, binary direction prediction from tweets + price history. Public dataset at github.com/yumoxu/stocknet-dataset. Loader at `trophic/training/stocknet_loader.py`.
- **Reward function**: weighted sum of XML field matches: ticker_w=0.4, direction_w=0.25, magnitude_w=0.15, horizon_w=0.10, sigma_w=0.05, confidence_w=0.05. Defined in `trophic/training/xml_schema.py`.
- **Train-eval contamination**: dev shares tickers with train (pre-migration concern, also true for the 0.388 IPO baseline). Held-out is disjoint. **Always report both.**
