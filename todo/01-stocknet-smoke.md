# 01 — StockNet (ACL-18) Smoke Test

**Status**: NOT STARTED
**Goal**: Run the trophic system on a small slice of ACL-18 StockNet to get a real-world directional-prediction number we can compare against published baselines.

Detailed integration plan already written: `trophic/docs/stocknet_integration_plan.md`. This file is the *task list* for executing it.

## Why this benchmark

- Public, no gating. Direct fit with our `direction` output field.
- Published baselines we can cite: Adv-ALSTM ~57%, MAN-SF ~58%, BERT-tuned ~74%, LLMFactor ~SOTA.
- Used by recent agentic-LLM work (Koa et al. 2024) so reviewers will accept it.

## Smoke-test scope (NOT full benchmark — first pass)

Subsample policy:
- **Top 5 tickers** (by sample count): AAPL, NVDA, GOOG, MSFT, AMZN — overlap with our training tickers, so the system has the best chance of producing sensible output. **First sanity check, not the headline number.**
- **2-week test window** (Oct 1–Oct 14, 2015) — about 50 sample-days, ~10 min on the 4090
- Skip days with no tweets initially (label-only days are a separate problem)

If smoke-test accuracy ≥ 55%, expand to full StockNet test set with all 88 tickers + the full Oct 2015–Jan 2016 test window.

## Implementation order (when GPU is free)

### Step 1: Vendor the dataset
```bash
mkdir -p /home/dgonier/ecology_experiment/trophic/external
cd /home/dgonier/ecology_experiment/trophic/external
git clone https://github.com/yumoxu/stocknet-dataset stocknet
du -sh stocknet
```

Disk check first — don't pull until we know the size. README at the repo says ~1GB total.

### Step 2: Build the loader
File: `trophic/training/stocknet_loader.py`

```python
def build_stocknet_scenarios(
    split: Literal["train", "dev", "test"] = "test",
    tickers: list[str] | None = None,    # None = all 88
    date_range: tuple[str, str] | None = None,
) -> list[Scenario]:
    """Walks stocknet-dataset, joins (tweets, ohlcv-history, label) into Scenarios."""
```

Output: list of `Scenario` objects with `inputs=[press_RawInput(tweets), ohlcv_RawInput(history)]` and `predator_target=emit_prediction(direction=label)`.

Cache parsed scenarios to `external/stocknet/cache_test_top5.pkl` so re-runs skip the file-walk.

### Step 3: Build the eval script
File: `scripts/eval_stocknet.py`

Mirror `scripts/eval_xml_checkpoint.py` but:
- Replace `build_scenarios()` + dev split with `build_stocknet_scenarios(split="test", tickers=TOP5, date_range=(...))`.
- Replace `reward_prediction(parsed, target)` (rule-based reward) with `accuracy + MCC` over the binary direction label.
- Print per-ticker breakdown so we can tell if the model is collapsing on one ticker.

### Step 4: Run smoke
```bash
TROPHIC_CKPT=checkpoints/ipo_seed7_best.pt TROPHIC_SEED=7 \
  .venv/bin/python -u scripts/eval_stocknet.py 2>&1 | tee logs/stocknet_smoke.log
```

### Step 5: Decide
- ≥60% acc → run full StockNet (88 tickers, 4-month test window). ~3 hr on 4090.
- 55-60% → ablation matrix to find what's helping/hurting before scaling up.
- <55% → debug ticker conditioning. Don't escalate the benchmark.

## Risks

- **Mode collapse**: SFT seed 7 emitted "BBRY" repeatedly. If ipo_seed7 still does this on novel tickers, accuracy is at chance.
- **Tweet noise**: 2014-2016 Twitter is messy. Our `press` producer expects clean disclosures. Either clean tweets first or accept the noise as test of robustness.
- **Horizon mismatch**: StockNet is next-day (1440 min); we trained on 30-120 min. May need to set `horizon_min=1440` in our SFT supervision and retrain a small adaptation pass (50-100 steps) before this eval is fair.

## Out of scope (defer to follow-up)

- Multi-seed StockNet runs
- Per-day temporal validity check (no "test" tweets leaking from "train" period)
- Sentiment-only baselines for ablation
