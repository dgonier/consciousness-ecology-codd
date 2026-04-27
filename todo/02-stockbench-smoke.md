# 02 — StockBench (Oct 2025) Smoke Test

**Status**: NOT STARTED
**Goal**: Run the trophic system on the StockBench DJIA-20 trading benchmark for the strongest paper claim we can make.

## Why this benchmark

- **Fresh** (Oct 2025) — only ~6 months of competing entries.
- **Agent-explicit** — was designed for LLM-agent evaluation. GPT-5 / Claude-4 / Qwen3 / Kimi-K2 are the listed entries. Our ecological multi-agent stack is the natural new entrant.
- **Real economic metrics**: Sortino ratio, cumulative return, max drawdown. These convert reviewer skepticism into "how does this compare to actual trading."
- **Contamination-free** by design. Test period: Mar 1 – Jun 30 2025, on top-20 DJIA stocks.
- **Headroom**: most LLMs lose to equal-weight buy-and-hold. Even a flat result is publishable.

## Smoke-test scope

Subsample first — running a 4-month live trading sim is expensive even at the agent-decision level. Try:
- **One ticker (NVDA)** for 1 month (April 2025) — ~20 trading-day decisions
- Decision cadence: daily close (matches StockBench's daily decision step)

If our predator can produce coherent buy/hold/sell decisions on this slice with non-trivial Sharpe, scale up.

## Major architectural question

**StockBench expects buy/hold/sell decisions, not (direction, pct_move, sigma) predictions.** Three integration shapes:

1. **Translate-only**: map our `(direction, pct_move, confidence)` → action via fixed rule (e.g., `buy if direction=up AND confidence ≥ 0.65 AND pct_move ≥ 0.5`). Cheap, but the rule is arbitrary.
2. **Add an apex action layer**: route predator broadcasts into a small decision head that produces `{buy, hold, sell, weight}`. Trainable. **More work but much more honest.**
3. **Re-frame predator output**: change the SFT target schema to include action directly. Most invasive; would touch the post-migration architecture we just stabilized. **Probably not worth it for the smoke test.**

Recommendation: start with #1 for the smoke test, plan #2 if smoke results justify it.

## Implementation order

### Step 1: Pull StockBench
```bash
mkdir -p /home/dgonier/ecology_experiment/trophic/external/stockbench
# Check stockbench.github.io or arXiv 2510.02209 for the data/eval framework — likely
# a pip package since they reference live evaluation.
```

### Step 2: Get NVDA April 2025 data path
- Daily OHLCV
- Daily news/headlines if StockBench provides them
- The exact decision-cadence interface they expect

### Step 3: Translation rule for our predator output → buy/hold/sell
```python
def predator_to_action(parsed_prediction, ticker, current_holdings):
    if parsed_prediction.ticker != ticker:
        return ("hold", 0.0)   # ticker mismatch = abstain
    if parsed_prediction.confidence < 0.6:
        return ("hold", 0.0)
    if parsed_prediction.direction == "up" and parsed_prediction.pct_move > 0.3:
        return ("buy", weight=conf)
    if parsed_prediction.direction == "down":
        return ("sell" if current_holdings > 0 else "hold", weight=conf)
    return ("hold", 0.0)
```

This rule is a placeholder. Refine based on what StockBench actually accepts.

### Step 4: Eval script + report
- Per-day decision log
- End-of-period: cumulative return, Sortino, max drawdown
- Compare against StockBench's published GPT-5 / Claude-4 numbers for that ticker/period

## Decision criteria

- Beats equal-weight buy-and-hold → strong paper signal
- Within 10% of buy-and-hold → reasonable, write up as "ecological agent matches frontier LLMs"
- Underperforms by >25% → architecture is doing the wrong thing in this regime; do not scale up

## Out of scope

- Full DJIA-20 4-month live run (only after smoke is green)
- Multi-strategy ensembles
- Live submission to StockBench leaderboard

## Order vs StockNet (#01)

**Run StockNet first.** StockNet is faster, simpler, and tells us whether our predator can produce sensible directional predictions on real data at all. StockBench requires several layers (data pipeline, action translation, portfolio sim) all working before any signal is observable. If StockNet is at chance, StockBench will be too — wasted effort.
