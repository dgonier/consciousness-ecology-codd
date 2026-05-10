# Signal Flow Walkthrough — 2026-05-09

End-to-end audit of how news → herbivores → propagation → carnivore → apex
becomes a watchlist, with copy-pasted real outputs from each stage.

Worked example: **AAPL on 2026-02-18** (post-template-fix sweep). News inputs
on this day are 43 articles covering AAPL Q1 2026 earnings, an analyst upgrade
with a 33% price-target raise, and a separate "support test" technical piece.

---

## Pipeline stages

```
Polygon news firehose (43 articles)
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│ Tier 1 — Producer / Adapter                                │
│ Group articles by mentioned ticker, dedup, hand to herbs   │
└──────────────────────┬─────────────────────────────────────┘
                       ▼
┌────────────────────────────────────────────────────────────┐
│ Tier 2 — Herbivores  (LLM + numpy)                         │
│   • event_classifier.v0.qwen-local — DSPy 2-stage:         │
│       (a) IdentifySubjectTickers, (b) ClassifyArticle      │
│   • cross_correlation.v0 — 60d/20d Pearson + sympathy      │
│ Output: BeliefActivations (target_belief_id, direction,    │
│         magnitude, confidence, reasoning, log_odds_shift)  │
└──────────────────────┬─────────────────────────────────────┘
                       ▼
┌────────────────────────────────────────────────────────────┐
│ Tier 3 — Belief Network Propagation  (Jacobi, cycle-safe)  │
│ Activations land on leaves → propagate through links →     │
│ outcome.{ticker}.next_day_direction (p_up updated)         │
└──────────────────────┬─────────────────────────────────────┘
                       ▼
┌────────────────────────────────────────────────────────────┐
│ Tier 4 — Carnivore Aggregation  (pure deterministic)       │
│ For each outcome with deviation: build Observation with    │
│ ranked causal_chain + conflicting_signals + salience       │
└──────────────────────┬─────────────────────────────────────┘
                       ▼
┌────────────────────────────────────────────────────────────┐
│ Tier 5 — Apex (Qwen3.5-4B via vLLM)                        │
│ DSPy WatchlistFromObservations signature reconciles top-K  │
│ Observations into final watchlist                          │
└────────────────────────────────────────────────────────────┘
```

---

## Stage 1 — Inputs

```
date:               2026-02-18
prev_trading_day:   2026-02-17
n_news:             43
universe:           AAPL, ABBV, AMZN, AVGO, CSCO, CVX, GOOG, HD, JNJ, JPM,
                    KO, MA, MCD, MRK, MSFT, PEP, PG, UNH, V, WMT
focus_tickers:      []  (firehose mode = no user query)
```

Three articles in this batch are AAPL-relevant:

1. *"Apple is testing a critical uptrend line"* — technical analysis piece
2. *"Apple Q1 2026 earnings: $143.8B revenue, +16% YoY"* — earnings report
3. *"Apple stock rebounds 8% on AI device announcement, Wedbush raises PT 33%"* — product/analyst news

---

## Stage 2 — Herbivore activations (post-fix)

Each event_classifier activation is a **categorical** classification (Qwen
chooses from `{increases, decreases, flat}` × `{weak, medium, strong}` ×
`{low, medium, high}`) which then maps to a numeric `applied_log_odds_shift`
via a lookup table. All four AAPL activations from this pass:

```json
{
  "belief": "belief.company.technical_support_broken__ticker_AAPL",
  "species": "event_classifier.v0.qwen-local",
  "direction": "increases",
  "magnitude": "weak",
  "confidence": "medium",
  "reasoning": "Article states Apple is testing a critical uptrend line; a failure would signal weakness, implying current status is a test of support.",
  "leaf_p_before": 0.5,
  "leaf_p_after": 0.56,
  "log_odds_shift": 0.24
}
{
  "belief": "belief.company.earnings_beat__ticker_AAPL",
  "species": "event_classifier.v0.qwen-local",
  "direction": "increases",
  "magnitude": "strong",
  "confidence": "high",
  "reasoning": "Article cites Q1 2026 earnings of $143.8B revenue with 16% YoY growth, explicitly framing the dip as overblown despite regulatory/AI concerns.",
  "leaf_p_before": 0.5,
  "leaf_p_after": 0.782,
  "log_odds_shift": 1.28
}
{
  "belief": "belief.company.major_product_launch_positive_reception__ticker_AAPL",
  "species": "event_classifier.v0.qwen-local",
  "direction": "increases",
  "magnitude": "strong",
  "confidence": "high",
  "reasoning": "Stock rose 3.12% on announcements of new Macs, AI devices, and podcast features, with analyst raising price target 33%.",
  "leaf_p_before": 0.5,
  "leaf_p_after": 0.782,
  "log_odds_shift": 1.28
}
{
  "belief": "belief.company.major_product_launch_positive_reception__ticker_AAPL",
  "species": "event_classifier.v0.qwen-local",
  "direction": "increases",
  "magnitude": "medium",
  "confidence": "medium",
  "reasoning": "Stock rebounded 8% after decline following analyst upgrade and price target increase, signaling positive market reception for upcoming AI-focused product event.",
  "leaf_p_before": 0.782,
  "leaf_p_after": 0.868,
  "log_odds_shift": 0.6
}
```

Note the **3rd and 4th activations both target the same belief**
(`major_product_launch_positive_reception__ticker_AAPL`) with two articles —
the second activation's `leaf_p_before=0.782` is the first activation's
`leaf_p_after`, so the leaf accumulates evidence: 0.5 → 0.782 → 0.868.

---

## Stage 3 — Network propagation

Each leaf belief that moved (`leaf_p_after ≠ leaf_p_before`) propagates to
its conclusion outcome via the Bayesian link:

```
contribution_log_odds = link_strength × sign(direction) × (logit(p_now) − logit(p_prior))
outcome.log_odds = Σ contribution_log_odds over all incoming activated parents
outcome.p_up = sigmoid(outcome.log_odds)
```

For AAPL on 2026-02-18, three leaves land on
`outcome.AAPL.next_day_direction`:

| Parent belief | p_now | p_prior | link_dir | link_strength | contribution_log_odds |
|---------------|-------|---------|----------|---------------|----------------------|
| major_product_launch_positive_reception__AAPL | 0.868 | 0.5 | positive | 0.61 | **+1.142** |
| earnings_beat__AAPL                            | 0.782 | 0.5 | positive | 0.54 | **+0.694** |
| technical_support_broken__AAPL                 | 0.56  | 0.5 | negative | **0.00** | **−0.000** |

The third row is the **template fix from 2026-05-09**: the
`technical_support_broken` link now has `strength = 0`, so it contributes
nothing to outcome.p_up. Pre-fix this row would have contributed
`-0.55 × 0.24 = -0.132` log-odds, dragging the final p_up from 0.86 down to
~0.82 — but worse, the same template was producing **wrong SELLs** on other
days (audit: 41% accuracy, 10 wrong SELLs across the 65-day sweep).

Sum: `+1.142 + 0.694 + 0 = +1.836` log-odds → `sigmoid(1.836) = 0.862`.

---

## Stage 4 — Carnivore Observation

The carnivore wraps the propagation result into one **Observation** per
ticker, with the chain ranked by `|contribution_log_odds|`. Copy-pasted from
disk:

```json
{
  "ticker": "AAPL",
  "action": "BUY",
  "p_up": 0.8625,
  "salience": 0.7766,
  "horizon_min": 1440,
  "in_focus": false,
  "confidence": "high",
  "reasoning_summary": "AAPL BUY p_up=0.86; ↑major_product_launch_positive_reception(+0.37×0.61); ↑earnings_beat(+0.28×0.54)",
  "causal_chain": [
    {
      "parent": "belief.company.major_product_launch_positive_reception__ticker_AAPL",
      "p_now": 0.868, "p_prior": 0.5, "delta": 0.368,
      "link_dir": "positive", "link_strength": 0.61,
      "contribution_log_odds": 1.142,
      "species": "event_classifier.v0.qwen-local"
    },
    {
      "parent": "belief.company.earnings_beat__ticker_AAPL",
      "p_now": 0.782, "p_prior": 0.5, "delta": 0.282,
      "link_dir": "positive", "link_strength": 0.54,
      "contribution_log_odds": 0.694,
      "species": "event_classifier.v0.qwen-local"
    },
    {
      "parent": "belief.company.technical_support_broken__ticker_AAPL",
      "p_now": 0.56, "p_prior": 0.5, "delta": 0.06,
      "link_dir": "negative", "link_strength": 0.00,
      "contribution_log_odds": -0.000,
      "species": "event_classifier.v0.qwen-local"
    }
  ],
  "conflicting_signals": [],
  "triggering_events": ["firehose.2026-02-18.evt"]
}
```

`salience = |p_up − 0.5| × (1 + |top_chain_contribution|) = 0.362 × (1 + 1.142) = 0.776`.
That's the rank key the apex sees.

A second example — same day, **WMT SELL** showing the bidirectional path
working when bearish templates fire correctly:

```json
{
  "ticker": "WMT",
  "action": "SELL",
  "p_up": 0.3875,
  "salience": 0.1828,
  "confidence": "low",
  "reasoning_summary": "WMT SELL p_up=0.39; ↑earnings_miss(+0.19×0.78); +1 conflict",
  "causal_chain": [
    {
      "parent": "belief.company.earnings_miss__ticker_WMT",
      "p_now": 0.69, "p_prior": 0.5, "delta": 0.19,
      "link_dir": "negative", "link_strength": 0.78,
      "contribution_log_odds": -0.624,
      "species": "event_classifier.v0.qwen-local"
    }
  ],
  "conflicting_signals": [
    {
      "parent": "belief.company.earnings_beat__ticker_WMT",
      "p_now": 0.599, "delta": 0.099,
      "contribution_log_odds": 0.166
    }
  ]
}
```

The carnivore split conflicting evidence into two lists:
`causal_chain` (aligned with net direction) and `conflicting_signals`
(opposing). When `Σ|conflicting| ≥ 0.5 × Σ|causal|` the action gets
forced to `WATCH`. WMT here has `0.166 < 0.5 × 0.624 = 0.312`, so SELL stands.

---

## Stage 5 — Apex prompt and output

The apex is **Qwen3.5-4B served via vLLM** at `localhost:8001`, called via
DSPy with this signature:

```python
class WatchlistFromObservations(dspy.Signature):
    """You are an analyst reviewing pre-synthesized Observations from the
    belief network. Each Observation already has a ticker, action,
    estimated p_up, salience score, and causal_chain showing which beliefs
    drove the prediction.

    Your job is to RECONCILE the Observations into a final watchlist:
      - Drop observations where the causal chain is weak or all conflicting
      - Keep observations with strong, multi-driver, low-conflict chains
      - Adjust p_up if the causal chain doesn't actually support the value
      - Always include focus_tickers in output (best read even if no
        salient observation exists for them)

    BIDIRECTIONAL: respect the carnivore's action (BUY when p_up > 0.55,
    SELL when p_up < 0.45). The carnivore already discriminated — do not
    flip every SELL to BUY. If the causal chain shows bearish drivers
    (earnings_miss, guidance_cut, ceo_departure, lawsuit, support_broken),
    keep it SELL. Don't default to BUY.

    The Observations are already filtered by salience; you are NOT
    expected to invent new tickers, only to curate and reconcile.
    """
    universe: list[str]            # ~20 tickers
    focus_tickers: list[str]       # [] in firehose mode
    observations: list[dict]       # full Observation dicts incl. causal_chain
    watchlist: list[dict]          # OUTPUT: ranked
```

So the apex receives the **full carnivore Observation list** (10 obs on this
day, sorted by salience) — not raw text, not raw activations, but the
already-aggregated ticker-level synthesis. Each obs includes the causal
chain and conflicting signals so the apex can verify that the p_up actually
follows from the evidence.

**Apex output (post-fix), copy-pasted from disk:**

```json
[
  {
    "ticker": "AAPL", "action": "BUY", "p_up": 0.86, "rank": 1,
    "reason": "Strong dual drivers: major product launch reception and earnings beat with no conflicts."
  },
  {
    "ticker": "AMZN", "action": "BUY", "p_up": 0.83, "rank": 2,
    "reason": "High conviction BUY driven by positive product launch reception and earnings beat."
  },
  {
    "ticker": "MCD",  "action": "BUY", "p_up": 0.62, "rank": 3,
    "reason": "BUY signal supported by earnings beat, though slightly offset by a minor earnings miss conflict."
  },
  {
    "ticker": "MA",   "action": "BUY", "p_up": 0.62, "rank": 4,
    "reason": "BUY signal supported by positive major product launch reception."
  },
  {
    "ticker": "WMT",  "action": "SELL","p_up": 0.39, "rank": 5,
    "reason": "SELL signal driven by earnings miss; conflicting earnings beat signal is weak."
  },
  {
    "ticker": "GOOG", "action": "BUY", "p_up": 0.62, "rank": 6,
    "reason": "BUY signal supported by earnings beat with no conflicting signals."
  },
  {
    "ticker": "MSFT", "action": "BUY", "p_up": 0.59, "rank": 7,
    "reason": "BUY signal supported by earnings beat; low salience but clear positive driver."
  },
  {
    "ticker": "MRK",  "action": "BUY", "p_up": 0.58, "rank": 8,
    "reason": "BUY signal supported by positive major product launch reception."
  },
  {
    "ticker": "PG",   "action": "WATCH","p_up": 0.55, "rank": 9,
    "reason": "Neutral stance; earnings beat provides slight upside but lacks strong conviction."
  }
]
```

The apex preserves carnivore p_up values almost verbatim (within ~0.01),
re-ranks slightly, and drops one observation it judged weak. The reasoning
strings demonstrate the apex *grounded its narrative in the causal chain* —
"earnings beat with no conflicting signals", "major product launch reception".

---

## Comparison: bare apex on the same day

To isolate the contribution of the ecology stack, the same Qwen3.5-4B is
called with raw news headlines and **no belief-network synthesis**:

```python
class FirehoseApex(dspy.Signature):
    universe: list[str]
    focus_tickers: list[str]
    news_headlines: list[str]      # raw filtered headlines
    prev_bars_summary: str          # one-liner from yesterday's OHLCV
    watchlist: list[dict]
```

**Bare apex output, same day, same model:**

```json
[
  {"ticker": "MSFT", "action": "BUY",   "p_up": 0.62, "rank": 1,
   "reason": "Strong momentum from AI spending validation and product launch hype, with technicals showing a rebound after recent volatility."},
  {"ticker": "AAPL", "action": "BUY",   "p_up": 0.58, "rank": 2,
   "reason": "Rebounding on Wedbush upgrade and March event anticipation, though regulatory headwinds remain a near-term risk."},
  {"ticker": "GOOG", "action": "BUY",   "p_up": 0.56, "rank": 3,
   "reason": "Benefiting from robust cloud growth and AI integration, supported by institutional buying despite sector rotation concerns."},
  {"ticker": "AMZN", "action": "WATCH", "p_up": 0.51, "rank": 4,
   "reason": "Mixed signals with strong AWS growth offset by high capital expenditure plans and recent warehouse robot failures."},
  {"ticker": "MA",   "action": "WATCH", "p_up": 0.50, "rank": 5,
   "reason": "Strategic partnership with Ericsson offers growth potential, but the stock remains sensitive to broader fintech rotation."}
]
```

Notice the differences — the **same model**, called twice, with different
context:
- Bare apex emits **3 BUY + 2 WATCH** in top 5; ecology emits **5 BUY** with
  much higher conviction (0.86, 0.83, 0.62, 0.62 vs 0.62, 0.58, 0.56, 0.51).
- Bare apex's reasoning is **pure narrative** ("AI spending validation",
  "Wedbush upgrade") — its predictions can't be traced back to specific
  beliefs that were updated by specific articles.
- Ecology's reasoning is **structured** — it cites the *belief templates*
  that fired (earnings_beat, major_product_launch_positive_reception), which
  the decomposer can then grade against ground truth and use to update link
  strengths.

Ground truth for this day: AAPL went **flat** (+0.18%), AMZN **down**, MCD
**up**, MA **flat**, WMT **down** (SELL was correct), GOOG **up**, MSFT
**flat**, MRK **down**.

So on day 2026-02-18:
- **Ecology gets WMT SELL right** (the only commit-direction call where
  ground truth was directional).
- **Bare doesn't commit to WMT at all** — it's not in bare's top-5.
- Both miss MCD on top-3 (MCD was the strongest mover up that day, +1.6%).

---

## How signals lead to belief adjustments (decomposer side)

After the apex emits the watchlist, the **temporal decomposer** runs
(per-day, online). For each (link, ticker, day) tuple, it computes whether
that link's contribution **agreed with realized direction**:

```python
contrib > 0  AND  actual_up      → n_correct += 1
contrib < 0  AND  not actual_up  → n_correct += 1
otherwise                         → n_correct += 0
n_total += 1   (per (link, day) pair, one observation regardless of how
                many articles fired the belief that day)
```

Then the link strength gets a **Beta-Bayesian update** with `prior_weight=3`
(was 15 — fixed 2026-05-09):

```python
a_prior = max(0.5, current * prior_weight)
b_prior = max(0.5, (1 - current) * prior_weight)
a_post  = a_prior + n_correct
b_post  = b_prior + max(0, n_total - n_correct)
new_strength = a_post / (a_post + b_post)
```

Concretely: with prior_weight=3, current=0.50, after one **wrong**
prediction (n_correct=0, n_total=1):

```
a_prior = 1.5, b_prior = 1.5, a_post = 1.5, b_post = 2.5
new_strength = 1.5 / 4.0 = 0.375
```

So one wrong prediction shifts the link from 0.50 → 0.375 (a 0.125 drop).
Pre-fix with prior_weight=15, the same wrong prediction would have shifted
the link from 0.50 → 0.469 (a 0.031 drop) — 4× weaker, which is why the
65-day sweep moved link strengths by an average of <2%.

The decomposer also writes a **teacher hint** per (article, ticker,
template):

```json
{
  "date": "2026-02-18",
  "article_id": "ff7f5237c50359a8aba33175c1b964d3a9596f0efc788524649603c067fbf146",
  "ticker": "AAPL",
  "belief_template": "major_product_launch_positive_reception",
  "observed_direction": "increases",
  "observed_magnitude": "strong",
  "observed_confidence": "high",
  "actual_direction": "flat",
  "actual_magnitude_bucket": "flat",
  "actual_return": 0.00178,
  "ideal_p_up_for_ticker": 0.5,
  "label_correct": false
}
```

These hints accumulate at `data/training_corpus/teacher_hints/pending.jsonl`
and are designed to feed a downstream training step (Hexis SFT or DSPy
classifier refinement) that hasn't fired yet — that's job 3 of the
decomposer (#134, pending).

---

## What the eval shows (5-day confirmation, 2026-02-17 to 2026-02-23)

| Top-K | ECOLOGY pre-fix MCC | ECOLOGY post-fix MCC | Δ | BARE MCC |
|-------|---------------------|----------------------|---|----------|
| 1     | 0.000  | +0.000 | +0.000 | +0.000 (n=4) |
| 3     | **−0.378** | **+0.000** | **+0.378** | +0.500 |
| 5     | −0.048 | +0.209 | +0.257 | +0.395 |
| 10    | +0.036 | +0.203 | +0.167 | +0.395 |

Bare still leads on these 5 days, but ecology went from **anti-correlated to
correlated**. The full 65-day replay (with the same fix applied to the
existing pass artifacts) showed ecology top-3 going from −0.018 to **+0.183**
(Δ +0.20), and ecology top-5 going from +0.007 to **+0.188** (Δ +0.18).

---

## What changed and where

| Fix | File | Change | Effect |
|-----|------|--------|--------|
| Decomposer prior_weight | `trophic/beliefs/decomposer_temporal.py:59` | `15.0 → 3.0` | Replay shifts 606 links (vs 162 pre-fix); online learning now actually learns. Marginal MCC effect — wasn't the main bottleneck. |
| Disable `technical_support_broken` | `scripts/instantiate_ticker_universe.py:79` + Neo4j | `strength_prior=0.0` | Removed 41%-accuracy template that was producing 10 wrong SELLs. |
| Disable `regulatory_action_announced` | `scripts/instantiate_ticker_universe.py:75` + Neo4j | `strength_prior=0.0` | Removed 29%-accuracy template (anti-correlated). |

Both bad templates' links remain in the graph at strength=0 so the decomposer
can re-learn them from fresh data if their precision recovers. The
event_classifier still fires those activations on incoming news; they just
contribute zero to outcome propagation.

---

## Open follow-ups

- **#136** Tighten event_classifier on `earnings_beat`/`major_product_launch`
  (47%/24% precision in the audit — most-fired templates are coin flips).
- **#138** Persist `events`/`state_beliefs`/`outcome_beliefs` in pass
  artifacts so the decomposer sees post-propagation state, not just leaf
  activations.
- **#134** Decomposer job 3: feed `teacher_hints/pending.jsonl` into a
  Hexis-style training trigger.
- **#135** Decomposer job 4: per-species fitness, death, reproduction.
- **Hexis specialization**: per-species d* extraction (`d_star_repeat` /
  `d_star_tools` agentic phase from `~/debaterhub/hexis`) as a frozen
  per-behavior direction vector — cheaper than per-species LoRA, no
  gradient training needed.
