# StockNet (ACL-18) Integration Plan

**Goal**: Establish a real-world external benchmark goalpost for the trophic system. Target: ACL-18 StockNet predicting next-day binary direction across 88 S&P stocks from tweets + OHLCV.

**Why this benchmark**:
- Direct fit with our `direction` output field (binary up/down).
- Public dataset on GitHub (yumoxu/stocknet-dataset), no gating.
- Established baselines: Adv-ALSTM ~57% acc, MAN-SF ~58%, BERT fine-tuned ~74%, LLMFactor (2024) reports +2.9% MCC over prior SOTA.
- LLM-based agentic methods have entered this leaderboard recently (Koa et al. 2024 self-reflective LLMs), so an ecological multi-agent stack is a natural extension.
- Effort estimate: 1-2 days for adapter + eval; no architecture changes.

**Out of scope**:
- StockBench / InvestorBench (Tier 2). Larger lift, save for stretch goal.
- LiveTradeBench. Requires live submission window; cannot retroactively benchmark.

## Dataset shape

- 88 tickers, 2014-01-01 to 2016-01-01.
- Per-day-per-ticker: `(tweets-today, price-history-N-days, label)` where `label ∈ {0, 1}`.
- Standard split: 2014 train, Aug-Sep 2015 dev, Oct 2015–Jan 2016 test.
- ~26K total samples; binary balance is roughly even.

## Mapping to trophic data shapes

```python
# StockNet sample → our pipeline:

stocknet_sample = {
    "ticker": "AAPL",
    "date": "2015-10-13",
    "tweets": [str, ...],           # day-of tweets, possibly 0
    "ohlcv_history": List[bar],     # N days back, OHLCV tuples
    "label": 0 | 1,                 # next-day close > today close?
}

# Maps to:
inputs = [
    RawInput(source="press",  payload={"text": "; ".join(tweets), "ticker": ticker}),
    RawInput(source="ohlcv",  payload={"ticker": ticker, "bars": ohlcv_history}),
]

predator_target = emit_prediction(
    ticker=ticker,
    direction="up" if label == 1 else "down",
    pct_move=None,           # StockNet is direction-only
    horizon_min=1440,        # next-day horizon
    sigma_pct=None,
    confidence=0.5,          # placeholder; we evaluate direction match only
)

scenario = Scenario(
    name=f"stocknet_{ticker}_{date}",
    inputs=inputs,
    technical_target=_technical_target(ticker, direction, pct_move=None),
    fundamental_target=None if not tweets else _fundamental_target(ticker, summary, direction),
    predator_target=predator_target,
)
```

## Implementation

### Phase 1: Loader (`trophic/training/stocknet_loader.py`)
- Clone yumoxu/stocknet-dataset as a git submodule or vendored at `external/stocknet/`.
- `build_stocknet_scenarios(split: Literal["train", "dev", "test"]) -> list[Scenario]` that walks the date range, joins tweets with ohlcv windows, returns a list of Scenarios.
- Cache parsed scenarios to `external/stocknet/cache_{split}.pkl` so we don't repay file-walking cost on every run.
- Subsample for early experiments: keep top-20 tickers (matches StockBench's choice) and a 6-month window. Full-set runs only when the system shows lift on the subsample.

### Phase 2: Eval script (`scripts/eval_stocknet.py`)
- Accepts `TROPHIC_CKPT` env var (default `ipo_seed7_best.pt` once it lands).
- Loops over the test split; predator emits direction; we parse, compare to label.
- Metrics: **accuracy + Matthews Correlation Coefficient (MCC)** — both reported in the StockNet leaderboard.
- Output: per-ticker breakdown + headline accuracy + MCC.
- Latency budget: ~15 min for the 6-month/top-20 subsample on the 4090.

### Phase 3: Comparison rows (post-IPO + held-out + StockNet)
After IPO finishes, the table looks like:

| Setup | Dev (14) | Held-out (68 fresh tickers) | StockNet test (subsample) |
|---|---|---|---|
| Pre-migration SFT | 0.326 | TBD | TBD |
| Post-migration SFT | 0.331 | 0.257 | TBD |
| Pre-migration IPO | 0.388 | TBD | TBD |
| Post-migration IPO | TBD (running) | TBD | **paper goalpost: TBD** |

StockNet baselines we'd want to print alongside our number:
- Random (50% binary balance) — lower bound
- Adv-ALSTM (Feng et al. 2018): ~57% acc
- MAN-SF (Sawhney et al. 2020): ~58% acc
- StockEmbed/BERT-tuned: ~74% acc
- LLMFactor (Wang et al. 2024): SOTA at time of search

Our number doesn't need to be SOTA. It needs to be **competitive with similarly-sized LLM systems while citing structural novelty** — multi-agent ecological architecture that no other StockNet entry uses.

## Risks

- **Mode collapse**: SFT seed 7 already exhibited "BBRY for everything" behavior on the dev set. StockNet has 88 tickers, all real; if the predator can't condition on ticker input, accuracy will be at chance (~50%). Mitigation: post-IPO eval first to see if mode collapse cleared; if not, ablation showing which architectural piece broke ticker conditioning.
- **Tweet quality**: StockNet tweets are noisy 2014-2016 Twitter — emojis, cashtags, URLs. Our `press` producer is currently designed for clean disclosure summaries. Need a tweet-cleaning preprocessor or an explicit "noisy_press" RawInput source so the herbivore can learn to discount.
- **Time horizon mismatch**: StockNet is next-day (1440 min); our training scenarios are 30/60/90/120 min. The horizon field of the predator may bias predictions toward shorter horizons. Mitigation: set `horizon_min=1440` in the SFT supervision target and reuse the existing horizon-grounding logic.

## Decision criterion

If post-IPO StockNet **subsample** accuracy is ≥ 60% (above Adv-ALSTM, below LLMFactor), proceed to full-test eval and start drafting the paper. If <55%, debug ticker conditioning before any further runs. Between 55-60%: ablation matrix to figure out which mechanism is helping vs hurting before committing to full StockNet.

## Order of operations after IPO finishes

1. Run real predator-reward eval on `ipo_seed7_best.pt` → dev number
2. Run held-out eval on `ipo_seed7_best.pt` → 68-scenario number
3. Run held-out eval on `ipo_seed1_best.pt` (pre-migration baseline) → apples-to-apples held-out for IPO
4. Build StockNet loader + eval; run on top-20 subsample
5. Compare against published baselines; decide paper vs ablation matrix
