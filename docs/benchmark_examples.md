# Benchmark Examples & Integration Guide

This file walks through the external benchmarks the trophic system targets,
how a sample flows from upstream raw data into our pipeline, how scoring
works, and how the metrics we emit line up with numbers reported by other
LLM/ML systems on the same datasets.

Sample artifacts are in [`docs/examples/stocknet/`](examples/stocknet/).
All example numbers there are synthetic and labeled as such — they exist to
document **format**, not to claim a result.

Companion documents:
- [`docs/stocknet_integration_plan.md`](stocknet_integration_plan.md) — design rationale
- [`todo/01-stocknet-smoke.md`](../todo/01-stocknet-smoke.md) — execution checklist
- [`todo/02-stockbench-smoke.md`](../todo/02-stockbench-smoke.md) — Tier-2 plan

---

## 1. Benchmarks at a glance

| Benchmark           | Status     | Task                                        | Metric(s)                          | Source                          |
|---------------------|------------|---------------------------------------------|------------------------------------|---------------------------------|
| StockNet (ACL-18)   | Integrated | Next-day binary direction (88 S&P tickers)  | Accuracy, MCC                      | `yumoxu/stocknet-dataset`       |
| Internal dev (14)   | Integrated | Predator-target match on curated scenarios  | Reward (rule-based, 0..1)          | `trophic.training.scenarios`    |
| Internal held-out (68) | Integrated | Same as dev but on fresh tickers         | Reward (rule-based, 0..1)          | `trophic.training.holdout_scenarios` |
| StockBench (DJIA-20)| Planned    | Daily buy/hold/sell on 20 DJIA stocks       | Cumulative return, Sortino, max DD | arXiv 2510.02209                |
| LiveTradeBench      | Out-of-scope | Live trading with submission window       | —                                  | live-only, can't be retro-evaluated |

The rest of this document focuses on **StockNet** since that is what is
actually wired up. StockBench and the internal sets are described in their
own planning docs.

---

## 2. How a StockNet sample flows through the pipeline

```
upstream repo (yumoxu/stocknet-dataset)
   │
   │  /price/preprocessed/<TICKER>.txt           ← TSV: date, mvt%, OHLC*, vol
   │  /tweet/preprocessed/<TICKER>/<YYYY-MM-DD>  ← JSONL: pre-tokenized tweets
   ▼
trophic/training/stocknet_loader.py
   │  _read_price_file()  → {date_str: {movement_pct, open_norm, ...}}
   │  _read_tweets()      → list[str]   (space-joined tokens)
   │  _label_from_movement(mvt) → "up" | "down" | None    (filter band)
   ▼
build_stocknet_scenarios(split=, tickers=, history_days=5, ...)
   │  for each (ticker, date) where label is not None and history fits:
   │    inputs = [
   │      RawInput(source="ohlcv", payload={"bars": last 5 OHLCV}),
   │      RawInput(source="press", payload={"body": joined tweets}),
   │    ]
   │    predator_target = emit_prediction(direction=label, horizon_min=1440)
   ▼
list[Scenario]
   │
   ▼
scripts/eval_stocknet.py
   │  SFTRunner._cache_producer_broadcasts()   ← producers see RawInputs
   │  for each scenario:
   │    pred_text = runner.eval_decode(sc)     ← predator emits XML
   │    parsed    = parse_prediction(pred_text)
   │    update TP/TN/FP/FN against parsed.direction vs target.direction
   │
   ▼
report: accuracy + MCC + per-ticker breakdown
```

### Concrete example

The sample scenario in
[`docs/examples/stocknet/scenario_AAPL_2015-10-13.json`](examples/stocknet/scenario_AAPL_2015-10-13.json)
shows what one `Scenario` looks like once serialized. Walking through it:

- **Upstream raw**: one row from `price/preprocessed/AAPL.txt`
  ([format excerpt](examples/stocknet/raw/price_AAPL_excerpt.tsv)) plus a
  day's tweet JSONL ([format excerpt](examples/stocknet/raw/tweets_AAPL_2015-10-13.jsonl)).
- **Label**: `movement_pct = 0.0061` is `>= UP_THRESHOLD (0.0055)` → `"up"`.
- **History window**: the prior 5 trading days are packed into the `ohlcv`
  RawInput payload (StockNet's *normalized* OHLC columns are passed straight
  through under `open_norm`/`high_norm`/`low_norm`/`close_norm`).
- **Press payload**: the day's tokenized tweets are newline-joined into
  `body`, capped at 4000 characters, with `source="stocknet_tweets"` so the
  herbivore can decide how much to discount the noisy channel.
- **Predator target**: an XML `<prediction>` block with `direction=up`,
  `horizon_min=1440` (next-day), `confidence=0.65`. `pct_move` and
  `sigma_pct` are intentionally omitted because StockNet is direction-only;
  scoring them would be unfair.

A second example,
[`scenario_GOOG_2015-10-15.json`](examples/stocknet/scenario_GOOG_2015-10-15.json),
shows the `down` class. A third,
[`scenario_AAPL_2015-10-09_filtered.json`](examples/stocknet/scenario_AAPL_2015-10-09_filtered.json),
documents a day that is **filtered out** because its movement falls in the
no-action band — useful for sanity-checking class balance after a run.

---

## 3. Scoring (how the metrics are computed)

`scripts/eval_stocknet.py` produces two leaderboard-comparable numbers and a
diagnostics block.

### 3.1 Accuracy

```
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Only **decisions** count toward the denominator. Abstentions and parse
failures are reported separately. This matches StockNet leaderboard
conventions (entries that abstain on every day score 0, not 0.5).

### 3.2 Matthews Correlation Coefficient (MCC)

```
MCC = (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

MCC ∈ [−1, +1]. It is the right summary statistic for StockNet because
binary class balance is roughly even but not exactly 50/50, and because the
metric **collapses to 0 when a model emits a single class for every input**
— the failure mode we expect from mode-collapsed checkpoints. A model that
yells "up" 100 times in a row will score ~0.5 accuracy by class balance
alone but exactly 0.0 MCC.

Implementation: `scripts/eval_stocknet.py:46`.

### 3.3 Per-ticker breakdown

Reported alongside the headline numbers so a single mode-collapsed ticker
doesn't hide behind ensemble accuracy. If a checkpoint scores 0.72 on AAPL
and 0.41 on JPM, that is a much weaker result than 0.60 across the board.

### 3.4 Diagnostics also reported

| Field          | What it tells you                                                          |
|----------------|----------------------------------------------------------------------------|
| `decisions`    | Coverage — % of scenarios where the predator produced a parseable verdict |
| `abstained`    | The model explicitly chose `<abstain>true</abstain>`                       |
| `parse_failed` | The output didn't match `<direction>` schema; degraded language gen        |
| `TP/TN/FP/FN`  | The full 2×2 — needed to spot direction-specific bias                      |

---

## 4. Running it

### 4.1 Smoke run (top-5 tickers, ~10 days/ticker, ~25 min on 4090)

```bash
TROPHIC_CKPT=checkpoints/ipo_seed7_best.pt \
TROPHIC_SEED=7 \
TROPHIC_LABEL=ipo_seed7_smoke \
STOCKNET_MAX_PER_TICKER=10 \
.venv/bin/python -u scripts/eval_stocknet.py 2>&1 | tee logs/stocknet_smoke.log
```

Expected report shape: see
[`docs/examples/stocknet/eval_output_smoke_example.txt`](examples/stocknet/eval_output_smoke_example.txt).

### 4.2 Full run (all 88 tickers, full Oct 2015–Jan 2016 test window, ~3 hr)

```bash
TROPHIC_CKPT=checkpoints/ipo_seed7_best.pt \
TROPHIC_SEED=7 \
TROPHIC_LABEL=ipo_seed7_full \
STOCKNET_TICKERS=$(python -c "from trophic.training.stocknet_loader import *; print(','.join(STOCKNET_88))" ) \
.venv/bin/python -u scripts/eval_stocknet.py 2>&1 | tee logs/stocknet_full.log
```

### 4.3 Env-var reference

| Variable                  | Default                              | Purpose                                       |
|---------------------------|--------------------------------------|-----------------------------------------------|
| `TROPHIC_CKPT`            | `checkpoints/ipo_seed7_best.pt`      | checkpoint to load                            |
| `TROPHIC_SEED`            | `7`                                  | channel-init seed (must match training seed)  |
| `TROPHIC_LABEL`           | `ipo_seed7`                          | label used in the report header               |
| `STOCKNET_TICKERS`        | top-5 (`SMOKE_TICKERS_TOP5`)         | comma-separated ticker list                   |
| `STOCKNET_MAX_PER_TICKER` | unset (= no cap)                     | smoke knob; cap days per ticker               |
| `STOCKNET_DATE_FROM/TO`   | unset (= full test split)            | narrow the window for fast iteration          |

---

## 5. How our scores compare to other setups

StockNet has been around since 2018 and is one of the more crowded
financial-NLP leaderboards. Numbers below are pulled from the cited papers
and are the figures we benchmark against.

### 5.1 Published StockNet baselines

| System                                | Year | Approach                              | Acc    | MCC   |
|---------------------------------------|------|---------------------------------------|--------|-------|
| Random (50/50 prior)                  | —    | lower bound                           | 0.500  | 0.000 |
| Buy-and-hold "always up"              | —    | majority-class baseline               | ~0.51  | 0.000 |
| HAN (Hu et al.)                       | 2018 | hierarchical attention on tweets      | 0.576  | 0.052 |
| StockNet (Xu & Cohen, original paper) | 2018 | VAE-based fusion                      | 0.582  | 0.081 |
| Adv-ALSTM (Feng et al.)               | 2018 | adversarial LSTM on prices            | 0.572  | 0.149 |
| MAN-SF (Sawhney et al.)               | 2020 | multi-modal attention                 | 0.582  | 0.196 |
| StockEmbed / BERT-tuned (various)     | 2021 | fine-tuned encoder + price features   | ~0.74  | ~0.40 |
| LLMFactor (Wang et al.)               | 2024 | LLM-extracted factors → classifier    | SOTA at time of writing |
| Self-Reflective LLM (Koa et al.)      | 2024 | agentic LLM, multi-pass critique      | mid-0.6s, varies by setting |

Notes:
- The two clusters worth caring about are **classical ML (~0.57–0.58 acc, MCC ~0.05–0.20)** and **modern fine-tuned/agentic LLM (~0.6–0.74 acc, MCC 0.2–0.4)**.
- MCC is more discriminating than accuracy in this regime; a 1–2 point accuracy gap is often noise, but a 0.05 MCC gap is usually real.
- Recent agentic-LLM entries (Koa 2024) make a multi-agent ecological stack a natural new submission rather than a left-field one.

### 5.2 Where the trophic system is expected to land

From [`stocknet_integration_plan.md`](stocknet_integration_plan.md), the
target band for the post-IPO checkpoint is:

- **≥ 0.60 acc / MCC ≥ 0.15** → competitive with classical baselines, paper-worthy with the architectural novelty story
- **0.55–0.60** → ablation matrix before scaling
- **< 0.55** → debug ticker conditioning; likely mode-collapse remnants from SFT seed 7

We do not need to beat BERT-tuned or LLMFactor; we need to be **competitive
with similarly-sized LLM systems while citing structural novelty** — the
multi-agent ecological architecture that no other StockNet entry uses.

### 5.3 Cross-benchmark score translation

If/when StockBench is wired up (Tier-2, planned), the scoring shape changes
because the task changes:

| Benchmark   | Output we emit             | Score type          | Comparable to                                  |
|-------------|----------------------------|---------------------|------------------------------------------------|
| StockNet    | `direction` only           | Acc, MCC            | All StockNet papers above                      |
| StockBench  | `(direction, pct_move, conf)` translated to buy/hold/sell | Cumulative return, Sortino, max DD | GPT-5, Claude-4, Qwen3, Kimi-K2 entries (per arXiv 2510.02209) |
| Internal dev/held-out | Full XML prediction | Rule-based reward (0..1) | Our own ablation lineage only                  |

The StockBench translation rule is sketched in
[`todo/02-stockbench-smoke.md`](../todo/02-stockbench-smoke.md) §3 — short
version, the predator's `(direction, confidence, pct_move)` triple is mapped
to `{buy, hold, sell, weight}` via a fixed threshold rule for the smoke
test, with the option of training a learned action head later.

---

## 6. Reproducing the example artifacts from real data

The JSON/TSV/JSONL files in `docs/examples/stocknet/` are synthetic. To
regenerate the same shapes from the upstream dataset:

```python
from trophic.training.stocknet_loader import build_stocknet_scenarios

scenarios = build_stocknet_scenarios(
    split="test",
    tickers=["AAPL"],
    history_days=5,
    max_per_ticker=3,
)
for s in scenarios:
    print(s.name)
    for inp in s.inputs:
        print(" ", inp.source, str(inp.payload)[:120])
    print("  target:", (s.predator_target or "")[:120])
```

The first run pulls files from the upstream repo over HTTPS and caches them
under `cache_dir` (default in the loader points at the project's
`external/stocknet_cache/`). Subsequent runs are local.

---

## 7. Pitfalls observed in past runs

These have bitten earlier attempts; flagging here so the next eval doesn't
re-hit them.

1. **Mode collapse from SFT seed 7** — the predator emitted `BBRY` regardless
   of input. Symptom on StockNet: per-ticker accuracy is uniformly ~0.5 and
   MCC is near 0. Fix: re-check post-IPO conditioning before scaling.
2. **Horizon mismatch** — training scenarios use 30/60/90/120 min horizons;
   StockNet is 1440 min. The loader sets `horizon_min=1440` in the target,
   but if the channel learned a strong horizon prior during SFT this is the
   first place to check on parse_failed spikes.
3. **Tweet noise** — 2014–2016 Twitter is dirty (cashtags, AT_USER tokens,
   URLs). The press producer was tuned on clean disclosure summaries; large
   `parse_failed` counts on tweet-heavy days suggest the herbivore isn't
   discounting the channel hard enough.
4. **Class balance per ticker** — even though the overall split is roughly
   balanced, individual tickers can be 70/30. Don't read per-ticker
   accuracy without checking that ticker's own class prior in the report.
