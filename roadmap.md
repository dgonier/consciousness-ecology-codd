# Trophic Roadmap

Living document. What's done, what's in flight, what's been discussed but not built.

Last updated: 2026-05-11 03:30 UTC (run9 salvaged at day 54; v6 interrogator redesign is the active workstream).

---

## Done

### v3.3 — 9-way model-ablation sweep (2026-05-10)
- 66-day firehose window, 3 apex models × 3 pipelines (BARE / ECOLOGY / ORACLE-next-day).
- Pydantic everywhere in DSPy.
- Cash yield (4% annual), 5bps slippage, 37% short-term cap gains tax, 2-retry validator loop.
- Bedrock Sonnet 4.6 + Opus 4.6 alongside local Qwen3.5-4B vLLM.
- Result: BARE-QWEN +9.67% net (winner), ECO-QWEN +8.32%, ORACLE-OPUS +0.17% (tax ate all alpha).
- Archived at `data/firehose_eval/runs/run_2026-05-10_pm_v3_3_9way/`.

### v4 phase 1 — multi-horizon signatures (2026-05-10)
- `HorizonForecast(h1, h5, h20, h60)` + per-horizon confidence + `regime_note`.
- `TickerView` with `primary_horizon` commitment.
- `Order` extended with `primary_horizon` + `expected_alpha_bps`.
- `ApexPMResponse` cross-validator (orders must reference views).
- Tax-framing block in all PM prompts (37%, 5bps, net_profit, dip-and-recover example).
- `state_for_apex.profit_definition` surfacing live tax / slippage / yield.

### v4 phase 2 — three parallel workstreams (2026-05-10)
- **Full-window Oracle**: `WatchlistOracleFullWindow` sees the full 66-day forward-return matrix, buy-low-sell-high directive, ≤0.5 orders/day suggestion, h60 preference.
- **Strategy committee**: 3 strategies (conviction-weighted, tiered-discrete, kelly-edge), weighted-average aggregator with strict-majority side resolution, `philosophy_weights.yaml`.
- **Tax-aware validators**: `validate_min_hold` (position's stored horizon, regime invalidation exceptions), `validate_forecast_consistency`, `validate_horizon_sizing`, `validate_edge_floor` (h60 waiver).
- Position-horizon tracking: BUY records `primary_horizon`; re-buys preserve original commitment (anti-gaming).

### v4 phase 2.5 — smoke gate (PASSED 2026-05-10)
- Contract verifier (Part A): 18/18 assertions passed.
- 3-day ECO-QWEN mini-sweep (Part B).
- **Round 1**: 100% rejection (composition bug + regime_note crash). Fixed by phase2-fix.
- **Round 2**: 89.5% rejection (conflicting `≤20 per name` directive vs new tier anchors). Fixed by phase2-fix-2.
- **Round 3**: 90% rejection — Qwen-4B can't hit ±5pp tier precision; empirical drift is 5-11pp short. Tolerance widened 5→10pp by phase2-fix-3.
- **Round 4**: **PASS** at 44.4% rejection rate. Diversity Jaccard 0.16 — predators picking near-disjoint tickers.

### v4 phase 3 sub-missions (all DONE 2026-05-10)
- **05a** Dow 30 universe: data layer + `--universe {dow30,legacy}` flag + producer filter at ingress.
- **05b** `InvestmentThesis` + `ThesisBook`: full Pydantic types with lifecycle (open/mature/invalidate/expire) and overdue sweep.
- **05c** `PredatorSubPortfolio` + `ApexPortfolio.is_debate_mode()`: per-predator capital + thesis book + tax isolation. `make_debate_portfolio()` factory.
- **05d** 4-phase debate mechanism: 4 philosophy headers + Pydantic types (Phase1Proposal, Response with agree/concede_to/object_to/extend, Phase2Responses, Concession, HeldFirm, Phase3RevisedProposal, Phase4Commit, CompressedProposal). `run_debate(...)` async orchestrator with phase barriers and `asyncio.gather` within each phase. Transcript JSONL.
- **05e** Runner wiring: 10th path (ECO-DEBATE-QWEN) added, ORACLE-SONNET dropped, `PMInterrogator` plumbed into ECO paths (DSPy-based, no in-process MathHost). Dow 30 default. `scripts/v4_run_sweep.sh` with `--async-paths` ON.
- **05f** Debate smoke gate: PASS at 20% rejection rate after fix iteration (one round of triage on HOLD-alpha-default, validator slice-fraction scaling, HorizonForecast nested-dict normalization).
- **phase3-fix-2**: `validate_in_universe` validator catches off-universe ticker hallucinations (XOM/AMD/TGT leaked through extend-responses in 05f smoke).

### v4 phase 3g sweep — run1 halted, run2 in flight (2026-05-10)
- **Run1**: launched at 16K context. Day 1 hit `Context window exceeded` on ORACLE-QWEN (full 66-day Dow 30 matrix overflows on early-window days). User called for window restart.
- **vLLM restart**: 16K → 32K context, GPU mem 0.7 → 0.85.
- **Run2 attempt 1**: launched at 32K. Reached day 9 in ~30 min. ECO-QWEN tracking v3.3 baseline almost exactly (-2.25% at day 8 vs v3.3's -2.23%). All 8 non-debate paths producing PM events. **Debate path hitting ~60% Pydantic parse failures** across DebatePhase1Proposal, DebatePhase3RevisedProposal, DebatePhase4Commit. Sweep halted to triage.
- **phase3-fix-3**: defensive `model_validator(mode="before")` normalizers landed on Phase1/3/4 (mirroring the phase 2 fix from 05f). Three Qwen-4B drift patterns identified and absorbed: `views[i].forecasts` missing horizons (truncation), `views[i].primary_horizon=None`, missing wrapper-level `predator_id` on thesis dicts. Test suite 267/1.
- **phase3-fix-3**: defensive `model_validator(mode="before")` normalizers on Phase1/3/4 (mirroring 05f Phase2 fix). Three Qwen-4B drift patterns absorbed.
- **phase3-fix-4**: `_drop_malformed_views` now drops any non-TickerView, non-dict entry. Catches the `views.25=['ticker']` truncation artifact.
- **phase3-fix-5**: `HorizonForecast._fill_missing_confidences` defaults missing `confidence_h*` fields to `"med"`. Catches Qwen-4B emitting forecasts dict with numeric horizons but no confidence labels.
- **phase3-fix-6**: `_coerce_close_to_id_strings` for `DebatePhase4Commit.final_theses_to_close`. Catches Qwen-4B mirroring the `final_theses_to_open` (full-dict) shape into the close slot.
- **phase3-fix-7**: `TickerView.rationale` defaults to `"unspecified"` (mirrors `regime_note` default). Saves numerical signal when Qwen-4B truncates the rationale field.
- **phase3-fix-8**: `Order._default_alpha_on_missing` generalized — defaults `expected_alpha_bps=0.0` for ANY side (was HOLD-only). Validator chain handles the semantics: edge_floor rejects BUY/ROTATE with 0 alpha; SELL is exempt by design.
- **phase3-fix-9**: `Order.reasoning` + 4 debate-model rationale fields (`Response.rationale`, `Concession.what_changed`, `Concession.rationale`, `HeldFirm.what_held`, `HeldFirm.rationale`) all default to `"unspecified"`. Closes the "missing scalar string field" failure family proactively.
- **phase3-fix-10**: new `_drop_orders_missing_required` helper drops orders missing any of {`side`, `primary_horizon`, `ticker` (except HOLD)} — fields that can't be safely defaulted. Wired into Phase1 (`orders`), Phase3 (`revised_orders`), Phase4 (`final_orders`). Final test suite: 276 passed, 1 skipped.

### v4 phase 3g sweep — run9 salvaged at day 54/66 (2026-05-11)
- Across runs 1-9, ten phase3-fixes shipped to absorb Qwen-4B drift patterns one at a time. Run9 was the first clean 54-day stretch.
- Halted at day 54 because Bedrock account-level daily token quota for Opus was exhausted from day 1, making BARE-OPUS, ECOLOGY-OPUS, and ORACLE-OPUS non-comparable (49+48+48 RateLimitErrors = 143 total). Three Opus paths were sitting idle on cash for 20-30 days at a stretch while their LLM calls failed.
- Archive: `data/firehose_eval/runs/run_2026-05-11_pm_v4_run9_salvaged/{sweep.log, debate_transcripts.jsonl, eval.jsonl}`.
- **Final non-Opus standings (day 54)**: ORACLE-QWEN $105,673 (+5.7% — winner, 100% invested for 50 of 53 days, no dip-buying — pure stock selection while always fully invested), BARE-QWEN $101,148 (+1.1%), ECO-DEBATE-QWEN $99,943 (break-even), ECOLOGY-SONNET $99,130 (-0.9%), BARE-SONNET $97,945 (-2.1%).
- **Findings from 54-day forensic on debate transcripts**:
  - Phase coverage perfect: 53 × 4 predators × 4 phases all populated.
  - **Concessions = 0 / 411 held_firm** (#30 still open) — Phase 3 prompt doesn't surface the act of conceding even when peers in Phase 2 wrote 33 `concede_to` against the predator.
  - **Universe leak**: 73 phase4 orders on non-Dow30 tickers (49 AMD + 24 others). `validate_in_universe` not catching the debate path's commit step (#32).
  - **ROTATE primitive is dead code** (0 / 798 commits used it). Either fix the prompt to use it, or remove it. Logged #38.
  - **Value predator HOLDs 78% of the time** (199 of 254 commits). Capital largely idle. Logged #39.
  - **Forecast magnitude bug**: value predator emits raw decimals as fractions — mean h60 = +0.317 with std 1.089, examples include JNJ h60=+8.50 (i.e., +850%). Will be clamped at the schema layer (#42).
  - **h60 monotonic bullish bias**: 83% non-negative across 918 phase1 views, 8 sign disagreements out of 292 cross-predator shared views. Predators can't construct a bear thesis (#33). Root cause likely upstream in the belief layer; defer architectural fix to v6 (Hexis d*) — see below.
  - **Order wash-out is real**: 16 same-day cross-predator BUY-vs-SELL conflicts; within-predator churn 96 reversal pairs across 4 predators; gross-to-net wash fractions up to 99% (value predator on JPM: gross 100.6%, net 0.6%). The architecture forces each predator to commit to its own sub-portfolio, so opposing trades execute fully against each other. Logged #40 — consensus voting layer is the proposed fix.
  - **Interrogator (PMInterrogator) abstained 331 / 332 days** — never invoked the solver. The math sub-agent is dead weight as currently designed. Logged #41 → expanded into v6 workstream below.
- Per `feedback_checkpoint_hygiene`: best + 2nd-best per run9 are preserved. Partial run logs (run3 day 4, run4 day 7, run5 day 9, run6 day 18, run7 day 33, run8 day 1) kept under `/tmp/run_2026-05-10_pm_v4_run{N}.partial_day{D}.log`.

---

## In flight

- **v6 interrogator redesign — ReAct loop + market-analysis tools** (the active workstream — see "v6 — Daily Signal Analyst" section below).
- Write `tasks_v4/V4_VS_V3_3.md` comparison artifact from the run9 salvaged data: pre-tax + net columns; per-predator leaderboard for the debate path; explicit caveat that 3 Opus paths are excluded due to Bedrock quota throttling.

---

## Discussed but not yet implemented

### v6 — Daily Signal Analyst (interrogator redesign) (NEW — 2026-05-11 03:00 UTC)
**Status**: ACTIVE. Highest-priority workstream, gating v5 and the next sweep. The existing PMInterrogator never fires (331 abstentions out of 332 invocations across run9's 54 days — see #41 diagnosis). Replacing it with an analyst agent grounded in ReAct + concrete market-analysis tools, modeled on the prompting style in `/home/dgonier/ecology_experiment/financial-services/plugins/agent-plugins/{market-researcher,earnings-reviewer,model-builder}` (reference for prompt design only, not 1:1 reimplementation).

**Reframe of the role**

From "math sub-agent that abstains" to "Daily Signal Analyst — senior quantitative analyst who owns the daily read of the firehose + belief network for the apex PM." The agent does NOT produce trade orders — it produces a structured Signal Brief with detailed observations and forecasted impacts, which the apex consumes alongside its other inputs. Per-agent fan-in to the trough.

**Output: Signal Brief**

```yaml
signal_brief:
  date: <iso timestamp>
  signals:
    - ticker: <symbol>
      headline: <one-sentence plain English>
      evidence: [{tool, args, result}, ...]   # every numeric claim traces to a tool call
      forecast:
        direction: up | down | flat
        horizon: h1 | h5 | h20
        expected_pct_range: [low, high]
        confidence: low | med | high
        invalidation: <what would prove this wrong>
      read_through: [list of correlated tickers]  # optional
  flags:                                          # data-quality issues, NOT trade ideas
    - ticker, headline, evidence, recommendation: ignore_upstream_value | etc
  context:
    regime_note, active_beliefs, thin_sample_warning
```

The apex consumes `forecast.direction + horizon + range + confidence + invalidation` — enough to size and time, but no order primitives.

**Tool surface (7 deterministic functions, no LM)**

1. `percentile_in_history(ticker, value, field, lookback_days)` — where does today's value rank?
2. `forward_return_after_signal(ticker, condition, horizon, lookback_days)` — historical: when condition held, what happened next?
3. `cross_ticker_extreme_check(condition, horizon, lookback_days)` — pooled across the universe for statistical power.
4. `bar_statistics(ticker, lookback_days)` — realized vol, mean return, max drawdown, current z-score.
5. `belief_chain(ticker)` — structured rerender of today's Bayesian decomposition for one ticker.
6. `correlated_movers(ticker, lookback_days, threshold)` — names that move with this ticker.
7. `flag_unit_error(field_name, value)` — sanity check for forecast magnitude blowups (catches the JNJ h60=+8.5 class of bugs).

All seven implement against the existing Polygon cache + belief network. No new data sources. The math agent dies — `bar_statistics` + `percentile_in_history` cover what the planner kept abstaining about.

**ReAct loop**
- Up to 7 tool calls per day (hard cap).
- Each iteration: THOUGHT (specific question about a specific ticker) → ACTION (one tool with concrete args) → OBSERVATION (quoted result).
- Stops when conviction stabilizes or budget exhausts.
- No "ABSTAIN" escape hatch — every day produces a brief (which may be short on quiet days).

**Iterative build plan**

Per user direction (2026-05-11 03:00 UTC): isolate and iterate on this layer before re-running the sweep.

1. **v6-A — Build the tool layer** (7 deterministic Python functions over Polygon cache + belief network; unit-tested in isolation; no LM calls).
2. **v6-B — Build the ReAct Daily Signal Analyst agent** standalone (Qwen3.5-4B via vLLM, system prompt finalized 2026-05-11 03:00 UTC). Run against 10 random days sampled from the run9 salvaged transcripts; hand-grade the briefs. NOT in the sweep — a standalone notebook/script.
3. **v6-C — Iterate prompts and tools** until brief quality is consistently good. Add tools or refine descriptions as gaps surface. May require multiple passes.
4. **v6-D — Wire the revised interrogator into the runner** as a drop-in replacement for `PMInterrogator`. 3-day smoke; gate on signal-brief quality vs the prior abstention rate.

Sample inputs for v6-B already dumped (2026-05-11 03:30 UTC) to `data/v6_interrogator_smoke_inputs/input_<DATE>.json` — 10 days sampled at random from run9 (2026-02-19, 02-20, 03-16, 03-17, 03-20, 03-25, 03-27, 04-13, 04-15, 04-20). Each file carries {date, universe (30 Dow tickers), focus_tickers, carnivore_observations[...], metadata}. Bar cache verified at `external/polygon_cache/polygon/bars/<TICKER>/<YYYY-MM>.jsonl` (fields: date, open, high, low, close, volume, vwap; all 30 Dow tickers covered Feb-May 2026).

**Why Qwen3.5-4B can do this without distillation**
- Per-day scope is small (one date, ~10-20 ticker observations).
- Tools deliver structured numbers; the model only has to choose questions and synthesize.
- Hard cap of 7 tool calls per day bounds context growth.
- No requirement for cross-day memory — each brief is fresh-state.

Distillation on this agent specifically (Hexis-style d* run on its plan/synthesis trajectories) is a follow-up that gets cheaper once the prompt is stable.

**Roadmap-level deferral**

v5 (TrajectoryBelief / precedent retrieval / seasonal / watchers) sits behind v6. Reason: the monotonic-bullish-bias finding (#33) probably has its root cause in the belief layer's training data, not the forecasting prompt. v5 is where that gets fixed — but trying to ship v5 before the interrogator is producing useful analysis would just stack model changes underneath an unreliable downstream consumer. Get the consumer working first.

---

### v5 — Richer belief-layer terminal nodes (DEFERRED behind v6 — 2026-05-10)
**Status**: scoped, queued behind v6 interrogator landing. All four sub-types are in scope; v5 will not ship piecemeal.

**The architectural diagnosis driving v5**

The current Bayesian network does sophisticated **single-step probabilistic classification**. Each terminal belief carries priors, link strengths, posteriors, and contribution log-odds — and produces a single output: `p_up_tomorrow ∈ [0, 1]`. The apex sees a rich Bayesian decomposition (parent beliefs, deltas, log-odds contributions, conflicting signals) but **collapsed to a 1-day binary direction question**.

Meanwhile the apex has to do four orthogonal jobs that the belief layer should be doing for it:
- Estimate **duration** of any predicted move
- Estimate **magnitude** of the move (not just direction)
- Recall **historical precedents** (how did this kind of event play out last time?)
- Track **standing conditions** (watchers — "if X crosses Y, trigger Z")

v4 partially addresses this by having the apex commit to a horizon (h1/h5/h20/h60) — but the apex is *inferring* horizon from a belief layer that only speaks "up tomorrow." The richer signal has to come from below, not be reconstructed above.

Plus an unrelated but compounding issue: belief node `statement` fields are **stubbed** as `"[TICKER] (stub for belief.company.X)"`. The apex sees structured Python identifiers like `belief.company.dividend_increase_announced__ticker_CVX` and has to infer semantics from the identifier alone. A 4B model loses signal here; Sonnet/Opus handle it better. Stub hydration is a content task (one-line English glosses for each belief type, ~50-100 distinct types), foundational for v5.

**Four sub-types — all in scope:**

#### v5-A: TrajectoryBelief (replaces scalar posterior)

Terminal node output type changes from `p_up: float` to:
```python
class TrajectoryBelief(BaseModel):
    p_up_h1: float
    p_up_h5: float
    p_up_h20: float
    expected_magnitude_h5: float    # bps
    expected_magnitude_h20: float
    typical_duration_days: int      # how long this kind of move usually lasts
    historical_precedent_count: int
    historical_precedent_summary: str
```

Every downstream consumer benefits. Apex can size by horizon-appropriate signal instead of inferring horizon from a 1-day bit.

#### v5-B: Historical-precedent retrieval

When a belief fires, attach the last N matching events from history with their outcomes (price trajectory, duration of move, regime context). Requires a precedent index over the firehose data. Self-contained; foundational for v5-A's `typical_duration_days` and `historical_precedent_summary` fields.

#### v5-C: Seasonal / calendar conditioning

Add `temporal_priors` species producing time-conditioned beliefs:
- `belief.market.santa_rally__window_dec22_jan02` (non-uniform prior across the year)
- `belief.company.earnings_proximity__ticker_X__days_to_earnings_<N>`
- `belief.market.fomc_blackout_window`

These are conditioning beliefs, not catalysts. They modulate priors of other beliefs based on time-of-year, day-of-week, days-to-event. Smallest leverage of the four; cheapest to ship.

#### v5-D: Watcher beliefs (conditional triggers)

Standing conditions the network monitors:
```python
class WatcherBelief(BaseModel):
    watcher_id: str
    condition: str                   # "if AAPL close < 180"
    triggers_belief: str             # "belief.technical.support_break__ticker_AAPL"
    triggers_value: float
    armed_at: str
    armed_by: str                    # which agent or apex armed it
    expires_at: Optional[str]
```

Apex (or other beliefs) can arm watchers. A watcher persists across days and fires when its condition becomes true. **This is the first-class connection to v4's InvestmentThesis**: a thesis can carry an `invalidation_watcher_id` that auto-fires when the breakout-failure condition triggers, without requiring the apex to re-evaluate every day.

Highest impact on debate-path quality — predators can encode "this thesis is invalid if X" as machinery, not prompt instructions that get forgotten.

#### v5 prerequisites (must land before any of A–D)

- **Belief statement hydration**: replace `"[TICKER] (stub for belief.company.X)"` with one-line semantic descriptions per belief type. Content task. Without this, even rich Bayesian traces are lossy because the apex (especially Qwen-4B) has to parse identifiers to recover meaning.

**Ordering when v5 starts:**

1. Belief statement hydration (prerequisite, no architectural risk)
2. v5-B Historical-precedent retrieval (provides data for A)
3. v5-A TrajectoryBelief (replaces terminal output type — largest blast radius)
4. v5-C Seasonal conditioning
5. v5-D Watchers (depends on InvestmentThesis from v4; lands cleanest after v4 sweep is measured)

### Debate + per-predator budgets + per-predator thesis books (UNIFIED — landed in v4 phase 3)
**Status**: **LANDED.** Implemented in v4 phase 3 sub-missions 05b/05c/05d/05e. Documented below for the architectural record; not a discussion item anymore. Measurement happens via the v4 sweep run2 (in flight).

**The unification.** A predator isn't defined by a *prompt template*; it's defined by its **persistent book of open investment theses**. The prompt is just the lens through which new theses get generated and existing ones reviewed.

- "Momentum predator" = a predator whose open theses have momentum-shaped catalysts and h5/h20 horizons.
- "Value predator" = a predator whose open theses have valuation catalysts and h60 horizons.
- "Mean-revert predator" = open theses anchored to z-score deviations with explicit revert-to-mean targets.
- "Event-driven predator" = open theses with specific dated catalysts (earnings, FOMC, etc.).

The character emerges from the book, not from a prompt template. This is also why "predators see the world differently" connects naturally to Hexis: a predator looking at NVDA after a 20% drop reads it through the lens of its open theses — value predator asks "is the new price within my valuation framework?", momentum predator asks "is the trend broken?". Same input, different question. **Hexis's per-predator perception layer becomes "does today's M extraction support the open theses, or are they drifting?"** — a real wiring path, not metaphor.

The v4 strategy committee (phase 2-C) is a deliberation among *algorithms* that average their sizing into a single order for a single apex. **Debate is deliberation among multiple apexes**, each with its own identity, capital slice, and persistent character across days. Voting averages out conviction; debate lets the strongest case win on that trade specifically while preserving diversity elsewhere.

**Design decisions made:**

- **Diversity source**: distinct **thesis books** carried on the same Qwen-4B backbone. Each predator starts the run with a philosophy-shaped prompt frame (momentum / value / mean-revert / event-driven), but its *identity over time* is the set of theses it has opened, closed, and currently holds. Cheap to test; isolates the debate mechanic from model-variance noise.
- **Thesis ownership**: **per-predator (private)**. Predator A's thesis #7 belongs to A; A's capital backs it; A's tax lot reflects its lifecycle. Predator B can object to A's thesis during round 2 but cannot co-sign or attach. This keeps the "I committed my capital, you didn't" pressure intact, which is the whole point of the diversity. Shared/marketplace thesis pool deferred to v6 once thesis lifecycle is observed under pressure.
- **Thesis visibility**: each predator sees other predators' open thesis books and round-1 proposals during round 2. They can argue against any of them; they just can't seize them.
- **Resolution mechanism**: sequential debate, final-round commit. No central judge.
  - **Round 1**: each predator proposes orders against its own budget.
  - **Round 2**: each predator sees the others' proposals + rationales; can revise, withdraw, or object.
  - **Round 3**: each predator's final order against *its own* budget executes independently. No averaging, no overruling.
- **Budget**: each predator starts with an equal slice of the portfolio's total equity ($100K / N). Money compounds based only on that predator's own trades. No rebalancing in v5 round 1 — end-of-run leaderboard reveals which predator's philosophy won the regime. Rebalancing/evolution mechanics deferred to v6.
- **v4 placement**: add as a **10th path** (`ECO-DEBATE-QWEN`) alongside the 9-way matrix in phase 3. Tests the mechanic with minimal scope expansion; the established 9 act as the control group.

**Why this is genuinely ecology-shaped (not just "vote more loudly"):**

1. **Real evolutionary pressure**: if the value predator's slice grows to 60% over the 66-day run and momentum shrinks to 5%, that's a *measurable* selection signal we can carry into v6's rebalancing.
2. **Conviction preservation**: voting at the strategy-committee level smooths the apex toward a midpoint nobody believes in. Per-predator debate lets each predator commit fully to its slice. The momentum predator's AAPL trade isn't watered down by the value predator's KO trade — they're separate decisions on separate capital.
3. **Diversity is built in, not assumed**: today's BARE/ECO/ORACLE difference is structural (different signal pipelines feeding one apex). Debate diversity is *philosophical* (same signals, different priors).

**Final landed shape** (all the "open questions" were resolved during 05c/05d implementation; recording the resolutions):
- Resolution mechanism: **4 phases** (not 3) — propose / respond / revise / commit. Each phase parallel across predators via `asyncio.gather`; phase barriers preserve ordering.
- Cross-predator visibility: `CompressedProposal` (ticker + side + horizon + 1-line rationale per item), NOT full proposals. Required to fit Qwen-4B 32K context with phase 2/4 having 3 others' visibility.
- Per-predator budget: `PredatorSubPortfolio` dataclass; each owns cash + positions + tax_owed + ThesisBook. Aggregate equity = sum across subs. Anti-gaming guarantee: re-buy on open position preserves original `bought_at_date` + `primary_horizon`.
- Validators: per-predator via `state_for_predator(pid, today, prices)`. `validate_horizon_sizing` gains `slice_fraction` parameter (tier bases scale by predator's share of total equity).
- Decomposer scope expansion: phase-3 `Concession` + `HeldFirm` fields carry `target_predator_id` + `what_changed/held` + rationale. The "who persuaded whom on what" graph is fully mineable.

### Blue-chip benchmark reorientation (new — 2026-05-10)
**Status**: **IN SCOPE for the next major sweep**. Priority alongside debate mechanism.

The current firehose universe is mixed across volatility/quality regimes. For the next major benchmark run, reorient toward blue chip names — large-cap, established, lower-volatility — so that:
- Tax friction matters more relative to alpha (blue chips don't move 5% a day; the 162bps edge floor becomes a tighter gate).
- Horizon discipline gets a fairer test (h60 holds are realistic for a blue chip; less so for a meme name).
- Slippage assumptions hold (5bps round-trip is reasonable on AAPL, less so on a $300M-cap name).

Open questions before scoping:
- Hard universe definition: S&P 100? Dow 30? A curated 20-name list?
- News firehose: do existing producers cover blue-chip news flow well, or do we need a different ingest?
- StockNet ACL-18 data has both AAPL/MSFT/GOOG and small-caps mixed — do we filter the universe inside the existing data slice or pull a new slice?
- Comparison framing: do we want blue-chip results comparable to v3.3 (same dates, filtered universe) or a clean new window?

### Initial conditions exploration (new — 2026-05-10)
**Status**: **POSTPONED.** Genuinely orthogonal to the decision-architecture work happening in v4 phase 3 + the debate mechanism. Revisit after the next major sweep produces clean debate-mechanic numbers; initial-conditions adds another dimension to the experimental matrix that's hard to interpret until the mechanics are stable.

v3.3 and v4 both start each portfolio at $100K cash, no positions. That's a strong assumption — real portfolios arrive with legacy holdings, locked-in capital, embedded gains/losses with tax lots, and constraints from the prior allocation.

Things to explore:
- **Pre-loaded portfolios**: start each path with a realistic mix (e.g., 60% SPY, 20% cash, 20% rotation candidates). Tests how the apex behaves when most of its budget is already committed.
- **Underwater positions**: start with a position bought at a higher price. Tax-loss-harvesting becomes a real decision; the apex should learn to realize losses against gains.
- **Embedded gains**: start with a position held >1 year. Long-term cap gains rate (15-20%) vs short-term (37%) is a real lever — the apex currently treats all realizations as short-term.
- **Mixed starting timestamps**: different paths start on different dates to factor out path-dependence on the regime that prevailed at $100K-on-day-1.

Connection to validators: most of these surface tax-lot tracking as a first-class concern. Today positions track `primary_horizon` + `bought_at_date`; expanding to multi-lot would require restructuring `_Position`.

### Investment thesis as a first-class artifact (ABSORBED into debate mechanism — 2026-05-10)
**Status**: **merged into the debate + per-predator design above**. No longer a standalone frontier. The thesis schema below is the implementation shape for what each predator carries in its private book.

Today each Order has a `rationale: str`. That's free-form text — the apex says "earnings tailwind h20" or "breakout failure". A thesis is structured and durable across days:
- Stated at first BUY, persisted on the position.
- Has expiry conditions (event date passed, target price reached, time horizon elapsed).
- Has invalidation conditions (regime_note transitions, specific price triggers).
- Decomposer (issue #4) consumes accuracy of past theses to learn the apex's calibration.

Possible shape:
```python
class InvestmentThesis(BaseModel):
    thesis_id: str
    ticker: str
    opened_at_date: str
    direction: Literal["long", "short", "neutral"]
    primary_horizon: Literal["h1", "h5", "h20", "h60"]
    catalysts: list[str]
    target_price: Optional[float]
    invalidation_triggers: list[str]
    expected_alpha_bps: float
    confidence: Literal["low", "med", "high"]
```

Positions carry their thesis_id; when the apex considers SELL, it has to either reference the original thesis (matured / invalidated) or open a new explicit reversal thesis.

Connection: this is what the **decomposer (#4 in v3.3 tasks)** would extract patterns over — not just rejection patterns, but thesis-outcome patterns ("apex's h20 momentum theses on tech names: 38% hit target, 12% hit invalidation, 50% expired untouched").

### Multi-hop tool-use for the interrogator (deferred, scaffolded)
**Status**: vision doc at `tasks_v4/phase4-PARKED-06-multi-hop-tool-use.md`.

`InterrogatorHerbivore` today is ask → MathHost → synthesize. Tool-use extension:
- Tool registry (price history, news bodies, options flow, correlated tickers, math).
- Planner emits tool calls mid-thought, results loop back, planner can chain.
- Budget cap per scenario (e.g., 5 calls).
- Logging for decomposer.

Held until v4 result is measured cleanly. Lift expected to be uneven — large on event-driven days, small on quiet ones.

### Decomposer pass (v3.3 task #4 — never run)
**Status**: pending, defined.

Extract validator-rejection patterns into Mind Tree lessons that flow back into the next-run apex prompt. With v4's multi-horizon, decomposer scope expands:
- Rejection patterns (current).
- Per-horizon forecast accuracy (compare predicted h5/h20/h60 to realized).
- Thesis-outcome patterns (once theses are first-class).
- Per-model calibration deltas (Qwen vs Sonnet vs Opus on the same scenarios).

Should run **after** v4 sweep completes, not before — the v4 run produces the data the decomposer needs to be informative.

### Async firehose dispatch (deferred)
**Status**: shipped as a flag (`--async-paths`), never exercised in a sweep.

`asyncio.gather` parallelizes the 8 non-ECO paths per day. Estimated ~45s/day vs current ~90s/day on A100 80GB. Not the bottleneck right now; revisit when sweep runtime becomes a constraint.

---

## Notes / principles that have hardened

- **Pydantic everywhere in DSPy** is non-negotiable (user directive).
- **Tests must stay green between every commit-sized change** (188 → 191 → 193 → currently in flight).
- **Validators are layered**: existing v3.3 chain runs first; tax-aware v4 chain layers on top. Don't replace, don't reorder without explicit reasoning.
- **Re-buys preserve the original commitment** (anti-gaming). Position tracks the horizon set at first BUY; tops-up don't reset it.
- **Smoke gates are non-negotiable before sweeps**. Phase 2.5 caught two composition bugs that would have cost a multi-hour GPU sweep.
- **Frontier model isn't a substitute for prompt engineering**. ORACLE-Sonnet finished worst-of-9 in v3.3 with perfect 1-day foresight — the bottleneck was behavior, not capability.
- **Tax friction is the dominant force in net P&L for high-turnover strategies**. v3.3 ECO was +13.26% pre-tax, +8.32% net. v4 thesis lives or dies on whether multi-horizon discipline reduces turnover enough to convert pre-tax alpha to net alpha.
