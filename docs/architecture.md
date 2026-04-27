# Trophic Multi-Agent System — v1 Architecture (revised from initial brief)

This document supersedes the initial brief on the points where they conflict.
Where it is silent, the initial brief still applies.

## What changed from the initial brief

The initial brief described a synchronous pipeline (producers → primary
consumers → consumer → apex) with embeddings-as-handoff between same-model
tiers and tokens at higher tiers. Several conversations refined this into
something materially different. This document captures the v1 we are actually
building.

Most important changes:

1. **No vLLM, no OpenAI-compatible API in v1.** Producers and herbivores run
   in-process via HuggingFace transformers against a single shared local model.
   We need direct access to hidden states for the channel-embedding handoff;
   vLLM's serving layer doesn't expose those.
2. **Asynchronous tick-driven loop, not synchronized cycles.** Producers fire
   when input arrives; herbivores consume when hungry; there is no global
   "everyone runs once per cycle" barrier.
3. **Substrate is a food pool, not a write-once log.** Items are added by
   producers, claimed-and-removed by consumers, and rotted by lack of appetite.
4. **`C` and `E` factoring.** Consumption is parameterized by a coupling
   regime `C` (market, ecology, democracy, hierarchy, …) and a learnable
   appetite function `E` over substrate features. These are the only places
   coordination behavior is encoded.
5. **Energy-based population dynamics.** Agents starve when their substrate
   isn't eaten (producers) or when they can't fill their stomach (consumers).
6. **Predator tier = teacher model (Claude on Bedrock).** Not stubbed. The
   predator provides the optimization signal that flows back through the
   ecology. Apex stays stubbed for v1.
7. **Open-ended feature space.** Substrate carries minimal metadata; `E` is a
   function over substrate that can in principle learn arbitrary weightings.
   No hand-coded "decay" mechanism — decay is just a particular `E`.
8. **Fintech toy domain.** 3 producers (TickDelta / Disclosure / Anomaly), 2
   herbivores (Technical / Fundamental), synthetic input generator.

## Training status (rolling)

Where the system is as of the latest training run. Updated as runs land.

### Training story so far

Training has gone through five phases, each surfacing the next bottleneck:

**Phase 0 — architecture validation (done).**
Three-tier hidden-state pipeline runs end-to-end on real Qwen3-4B + real
Bedrock. Channels with null-attention gate work; abstention fires
correctly when diet doesn't match; multi-tier broadcast pool routes
items via diet tags. Untrained baseline produces token-loops ("THE THE
THE") — the expected starting point.

**Phase 1 — SFT v1, 25 scenarios, direction-only targets.**
Eval CE 4.62 → 1.34 over 500 steps. Format becomes correct (parseable
SYNTHESIS:/CONFIDENCE:/PREDICTION: structure). Diet routing learned
through Channels (technical herbivore correctly abstains on disclosure
scenarios, etc). Bottleneck surfaced: outputs collapse to a few favored
tickers regardless of input.

**Phase 2 — three-seed reproducibility + scenario expansion.**
Deterministic Channel seeds added; expanded scenario library 25 → 212.
Three-seed runs at 500 steps land at eval CE 0.49 ± 0.14. Reproducibility
fixed (same seed → same trajectory). Wrong-ticker / wrong-direction
problem confirmed as fundamental SFT limitation: token-CE rewards
"output looks like a target on average" without conditioning on input
specifics.

**Phase 3 — multi-modal SFT, 285 scenarios.**
Forecaster herbivore (Chronos-bolt-base) added as third herbivore
species. Predator's Channel set extended to 3 sources (technical /
fundamental / forecaster). 800-step SFT lands at eval CE 0.35.
Multi-modal templates appear in output (`"forecast trend confirms
qualitative thesis"`). Wrong-ticker problem persists as the *defining*
failure mode.

**Phase 4 — GRPO from SFT v1 checkpoint (100 steps).**
Group sample G=4, self-judge with scenario context. Eval mean score
0.39 → 0.49 on held-out. The mechanism works (judge correctly
penalizes wrong-ticker outputs at 0.3, rewards right-ticker at 0.6+);
but most groups have all-wrong-ticker completions → zero within-group
variance → zero advantage gradient. Predator mode-collapses to "CNA up"
across all scenarios. **Diagnosis: GRPO is the wrong tool when the
policy is too narrow to explore productively.**

**Phase 5 — magnitude-aware targets + IPO (in flight).**
Targets upgraded to include exact percent moves (from OHLCV open/close
arithmetic) and forecaster σ (from quote-series slope/horizon/noise).
Re-SFT lands at eval CE 0.42 — predictions now contain real magnitudes
(`"+1.65% over next 60 minutes σ≈1.72%"`). IPO trainer built (Identity
Preference Optimization, paired oracle target vs sampled rejected,
squared loss to avoid DPO instability under deterministic preferences).
IPO from magnitude-aware checkpoint currently running.

### What we know about each failure mode

  - **Word salad / multilingual collapse**: bf16 fixes fp16 NaN;
    out_seq_len=8 MLP fixes single-vector mode-collapse;
    repetition_penalty=1.3 fixes within-output token loops.
  - **Wrong format (no SYNTHESIS: prefix)**: SFT fixes within ~200 steps.
  - **Wrong ticker / wrong direction**: SFT cannot fix (token-CE
    optimizes form, not grounding). GRPO partially fixes if the policy
    has enough diversity; otherwise mode-collapses. **IPO is the
    expected fix** — paired oracle vs sampled rejected gives direct
    preference signal on the grounding tokens.
  - **Magnitude absence**: fixed by writing magnitudes into the targets
    in the first place. The model can't predict what isn't in the
    training distribution.
  - **Calibrated confidence**: not fixable by token-CE or RL on text.
    See *Bayesian causal reasoning* section below for the design we'd
    use.

### Active checkpoints

  - `checkpoints/sft_seed1_best.pt` — magnitude-aware SFT, 800 steps,
    seed=1, eval CE 0.42. Currently the warm-start for IPO.
  - IPO checkpoints (`ipo_seed1_best.pt`) appear once the IPO run
    completes.

### Next decisions when IPO settles

  1. If IPO fixes the wrong-ticker problem: move to GRPO on top of IPO
     (now meaningful because the policy is grounded enough for
     within-group reward variance).
  2. If IPO doesn't fix it: investigate Channel-side exploration
     (Gumbel-softmax over attention, or per-completion noise on
     `channel_output`) to give group sampling real diversity.
  3. Either way: prototype the *Bayesian* species pair in parallel
     (BayesianNodeHerbivore + BayesianPredator), since calibrated
     numerical output and counterfactual reasoning aren't on the
     trophic-only roadmap at all. See dedicated section below.

## Stack

- Python 3.12, asyncio
- PyTorch + transformers (HF), local in-process model = **Qwen3-4B** (cached)
- Retrieval embeddings = **Qwen3-Embedding-0.6B** (cached)
- AWS Bedrock for predator teacher = **Claude Sonnet 4.6** by default,
  configurable to Opus 4.7. Reached via `boto3` (or `aws bedrock-runtime`
  invoke under the hood).
- SQLite for substrate persistence
- numpy for embedding math
- pydantic for typed messages

No vLLM, no httpx, no sentence-transformers in v1. Hardware: single RTX 4090
(24GB), Qwen3-4B fits comfortably with headroom for two concurrent forward
passes.

## Tiers (v1)

```
raw input  →  PRODUCERS  →  [substrate pool]  →  HERBIVORES  →  [syntheses]  →  PREDATOR (teacher)  →  [judgments]
   real /         3                                     2                            Claude on Bedrock
 synthetic    in-process                          in-process
              Qwen3-4B                            Qwen3-4B

                                              [APEX — stubbed in v1]
```

Producers and herbivores share the same in-process model with different
prompts (and eventually different `E` parameters / heads). Predator is an
external API call. Apex raises NotImplementedError.

## Producers (3)

Each producer has an objective `p_o`, an attraction filter on raw input, and
an output substrate type. Diets are designed to barely overlap on inputs and
to produce contested vs. routable substrate on outputs.

| Producer            | Objective                                      | Input diet (attracts)                                | Output substrate                                                     |
|---------------------|------------------------------------------------|------------------------------------------------------|----------------------------------------------------------------------|
| TickDeltaProducer   | "Track price/volume deltas in the last hour"   | OHLCV bars, trade prints, order-book snapshots       | `PriceDeltaEvent` — dense, numeric, high frequency, low semantics    |
| DisclosureProducer  | "Surface new company-released information"     | SEC filings (8-K/10-Q/13D), PRs, earnings transcripts | `DisclosureEvent` — sparse, textual, high semantics                  |
| AnomalyProducer     | "Find anomalous behavior in price+volume jointly" | OHLCV, options flow, short-interest, halt notices  | `AnomalySignal` — bursty, mixed numeric/textual, contested resource  |

Attraction is a hard filter in v1 (`bool` over `RawInput.source` + metadata).
Soft attraction is a future tunable.

Producers emit substrate to the food pool with:
- `producer_id`, `created_at`
- `payload` (the substrate content; numeric fields rendered to a short text
  string for embedding uniformity in v1)
- `diet_tags` (categorical, e.g. `is_price_event`, `is_disclosure_event`,
  `is_anomaly`)
- `retrieval_embedding` (Qwen3-Embedding-0.6B over the rendered text — for
  cheap cosine ranking)
- `channel_embedding` (Qwen3-4B last-layer pooled hidden state — for the
  inter-agent transmission channel)

## Herbivores (2)

| Herbivore             | Diet preference                                            | Output                                |
|-----------------------|------------------------------------------------------------|---------------------------------------|
| TechnicalHerbivore    | wants `PriceDeltaEvent` + `AnomalySignal`; ignores filings | technical-analysis-style synthesis    |
| FundamentalHerbivore  | wants `DisclosureEvent` + relevant `AnomalySignal`         | event-driven synthesis                |

`AnomalySignal` is the contested resource — both herbivores want some of it,
for different reasons. This is what gives the market dynamic something to
clear over.

A herbivore's consumption pathway:
1. **Filter** the food pool by diet tags (cheap; effectively a SQL WHERE).
2. **Score** appetite over each candidate using `E_{producer→herbivore}`.
   In v1 `E` is hand-initialized (no gradients) — see "Appetite & coupling"
   below.
3. **Select** up to capacity using `C_{producer→herbivore}` (the coupling
   regime; v1 ships ecology + market).
4. **Eat** — claim the items atomically from the pool, mark as consumed.
5. **Synthesize** — Qwen3-4B forward pass with the consumed items' channel
   embeddings injected as input. Emits a `ConsumedSynthesis` with cited
   substrate IDs and a self-assessed confidence.

Capacity is a hard cap (stomach size). If the diet-filtered pool can't fill
capacity, the herbivore eats what's available and is partially hungry.

## Appetite (`E`) and coupling (`C`)

`E_{producer→herbivore}: (substrate_item, herbivore_state) → ℝ` is the
appetite function. Conceptually a learnable function over substrate features
+ herbivore state. In v1:

- **Structured (per-feature) form.** `E` is a small object with named
  weighted contributions: `age_weight * age_ticks +
  retrieval_weight * cos(retrieval_emb, query_emb) +
  diversity_weight * (-max_sim_to_meal_so_far) +
  reputation_weight * producer_reputation + …`
- **Hand-initialized, no gradients in v1.** We set initial weights and
  watch what happens. This is the "play around with `E`" surface from the
  conversation.
- **Open-ended in principle.** The structured form is a v1 convenience for
  inspection and tunability. The interface allows replacing `E` with a
  learnable function (small MLP, kernel, etc.) in v2 without touching
  callers. The `appetite()` function gets a substrate item and a herbivore
  state — what features it derives from those is up to the `E`
  implementation.

`C_{producer→herbivore}: (scored_candidates, herbivores) → meal_assignments`
is the coupling regime — *how* appetite scores resolve into actual
consumption. v1 ships:

- **EcologyCoupling**: greedy diversity-weighted selection per herbivore, in
  randomized herbivore order per tick. (This is the brief's original
  consumption ranker, lifted into the `C` interface.)
- **MarketCoupling**: simultaneous bidding — each herbivore submits ranked
  bids with appetite scores; clearing engine resolves contention by highest
  bid, ties broken randomly. Unmatched substrate goes back to the pool.

Future `C`s (DemocracyCoupling, CommunityCoupling, HierarchyCoupling) get
stub classes raising `NotImplementedError`.

## Predator (teacher)

Claude on Bedrock, called per herbivore synthesis. Returns a
`PredatorJudgment` with:

- `score` (0..1) — the optimization signal
- `rationale` — why
- `predicted_action` — what the predator would have predicted given this
  synthesis (recorded for future apex bootstrap, unused in v1)

The teacher prompt is load-bearing and lives in `predators/prompts.py`.
Default model: `anthropic.claude-sonnet-4-6`. Configurable via env to
`anthropic.claude-opus-4-7`.

Authentication: standard AWS credentials chain (we verified
`aws sts get-caller-identity` returns a valid identity). No `ANTHROPIC_API_KEY`
needed.

## Substrate pool (`substrate.py`)

SQLite-backed pool with claim semantics. Tables:

- `substrate(id, producer_id, created_at, claimed_by, claimed_at, payload_json,
  diet_tags_json, retrieval_embedding BLOB, channel_embedding BLOB)`
- `syntheses(id, herbivore_id, created_at, content, confidence,
  consumed_substrate_ids_json, rejected_substrate_ids_json,
  retrieval_embedding BLOB, channel_embedding BLOB)`
- `judgments(id, synthesis_id, score, rationale, predicted_action, created_at)`
- `decomposer_records(id, run_id, record_type, target_id, rationale,
  retrieval_embedding BLOB, created_at)`
- `agents(id, role, alive, energy, born_at, died_at, params_json)`
- `ticks(tick_id, started_at, ended_at, n_substrate_added, n_eaten, n_rotted,
  n_judgments)`

Methods:

- `add_substrate(item)`
- `query_diet(diet_tags, limit, exclude_claimed=True) -> list[Substrate]`
- `claim(item_ids, herbivore_id) -> list[Substrate]` — atomic
- `mark_rotted(item_ids)` — for items no consumer wants any more
- `add_synthesis(synthesis)`, `add_judgment(judgment)`, …
- `update_agent_energy(agent_id, delta)`

Reaping: not on a wall-clock timer. An item is rotted only when *no live
herbivore's* `appetite > min_threshold` for it. Decay-with-age is just a
component of `E`; if the age weight is high enough negative, items
effectively rot when they get old, but it's emergent rather than imposed.

## Energy / population dynamics

Each agent carries a scalar `energy`, updated by events:

- Producer: `+r_eaten` per substrate item consumed; `+r_useful` per item that
  fed a positively-judged synthesis (decomposer attribution); `−c_produce`
  per item produced; `−c_exist` per tick.
- Herbivore: `+r_intake * fill_ratio` per meal; `+r_judged * judgment_score`
  per predator judgment on its synthesis; `−c_exist` per tick.

Below `energy_dormant_threshold` an agent skips its next firing opportunity.
Below `energy_cull_threshold` it's removed from the registry; a new agent of
the same role is spawned with mutated parameters (different prompt, slightly
perturbed `E` weights).

v1 keeps this *light*: no spatial simulation, no foraging behavior. Just
floats and registry updates.

## Decomposer (attribution engine)

The decomposer is shaped as a credit-assignment engine, not a quality critic.
Inputs: a chain of (substrate → synthesis → judgment). Outputs:

- Per-substrate `useful` / `misleading` / `unreliable_source` records
- Per-producer reputation deltas
- Per-herbivore reward deltas

In v1 it consumes predator judgments directly (no apex). Falsification record
type is defined but unused (requires apex feedback).

## Async tick loop (`runner.py`)

```
while running:
    tick_id += 1

    # Producers fire on whatever new input is available (pull or pushed)
    new_inputs = input_source.poll()
    for producer in alive_producers():
        for inp in new_inputs:
            if producer.attracts(inp):
                substrate_items = await producer.produce(inp)
                pool.add_many(substrate_items)

    # Hungry herbivores consume (in randomized order)
    hungry = shuffle([h for h in alive_herbivores() if h.is_hungry()])
    for h in hungry:
        candidates = pool.query_diet(h.diet, limit=retrieval_k)
        scored = [(item, E.appetite(item, h.state)) for item in candidates]
        meal = C.select(scored, h)  # may be partial
        pool.claim([m.id for m in meal], h.id)
        synthesis = await h.synthesize(meal)
        pool.add_synthesis(synthesis)

    # Predator judges new syntheses
    new_syntheses = pool.unjudged_syntheses()
    judgments = await predator.judge_batch(new_syntheses)
    pool.add_judgments(judgments)

    # Decomposer attributes credit
    decomposer.attribute(new_syntheses, judgments)

    # Energy update + census
    population.tick(tick_id)
    population.cull_and_spawn()

    # Metrics
    metrics.record(tick_id, …)
```

There's no fixed cycle count in production usage; for tests and the demo
we'll cap to N ticks.

## v3+ direction: producer evolution

Producers are pre-processors — modality adaptors at the world-facing
boundary. Currently we hand-design them. Eventually they should evolve.

**The mechanic**: each producer has a params dict (e.g. for
QuantitativeProducer: `history_len`, `normalization`, `ticker_filter`,
`aggregation`). Reproduction = spawn child with perturbed params from a
high-energy parent. Death = standard energy-cull threshold. The fitness
signal is downstream: producers whose substrate consistently feeds
high-judged apex outputs accumulate energy via the decomposer's parent-id
chain walk.

**Why this matters**: producer evolution is where *modality discovery*
happens automatically. If the apex judge consistently rewards predictions
that use volume-Z-scores over raw volume, mutated producers emitting
Z-scores will outcompete raw-volume producers. The system learns the
right preprocessing through selection rather than hand-engineering.

**Three independent adaptation rates**:
  - Producer evolution: slow (births/deaths over many ticks). Modality.
  - Channel learning: continuous. Routing.
  - LM-stack RL (GRPO): continuous. Use of routed features.

All three are driven by the same downstream apex fitness signal but
operate on different time scales and different parameter spaces. This is
the architectural payoff of putting agents at every tier — each kind of
adaptation gets its own loop.

**Build prerequisites** (none of this is in v2):
  - Producers carry params dicts (currently hardcoded).
  - Decomposer walks parent_input_ids back to producers (currently stops
    at the herbivore tier).
  - Reproduction operators per producer kind (mutation primitives).
  - Population layer handles births/deaths for producers (it tracks
    energy already; just needs spawn logic).

**Mental model**: this is a **sweep config integrated with the
ecosystem** — same shape as W&B Sweeps / Optuna / Ray Tune, but with two
critical differences:

  1. The fitness function is *not given* — it emerges from the trophic
     chain. Producers never see a closed-form objective; they only know
     whether their energy goes up or down based on whether their
     substrate eventually contributed to high-judged apex outputs.
  2. The sweep is **online with steady-state replacement** — a live
     population of N agents all contribute actual data concurrently, with
     the worst getting culled and replaced. Not a generational batch
     algorithm.

The mechanism generalizes to *every* agent tier (producers, herbivores,
predators), so this isn't producer-specific code — it's a shared
`(mutate + select)` infrastructure in `ecology/` that operates on any
agent with a params dict and a `param_space` declaration. Standard EA
concerns apply: minimum-diversity-per-kind and mutation-rate scheduling
both belong in the population layer.

## v3+ direction: prediction as fusion of trends and reasoning

Prediction is not one job. It's a fusion of two kinds of evidence with
opposite failure modes:

  - **Trends**: time-series forecasting (Chronos, TimesFM, Lag-Llama,
    PatchTST). Calibrated probabilistic output, exploits temporal
    patterns. Fails on anything that isn't in the time series — a CFO
    departure announcement has zero signal in OHLCV.
  - **Reasoning**: causal inference from heterogeneous evidence. LLMs
    approximate this; structural causal models (Pearl-style), GNNs over
    knowledge graphs, or symbolic+neural hybrids do it more honestly.
    Fails on calibrated quantitative output and on patterns that need
    actual numerical extrapolation.

These complement: where one degrades, the other has signal. The right
predator tier is therefore not "pick one"; it's a fusion architecture.

Three options, in order of how ecological they are:

  1. **Side-by-side specialist predators**: two (or more) predator agents
     with different diets and different base models. Apex (or a fusion
     tier) integrates their broadcasts. Trophic dynamics drive
     specialization — a trend predator that tries to read filings will
     starve, a reasoning predator that tries to forecast volatility from
     price will starve. Cleanest fit with existing architecture.
  2. **Single predator with FusionChannel**: one agent, two-stream input
     channel that projects heterogeneous embedding spaces into a common
     space before the predator body. More research-interesting, commits
     to building cross-modal fusion early.
  3. **Reasoning predator using forecaster as a tool**: LLM at top calls
     forecasting model as needed. Most powerful, most machinery.

Suggested progression when we get there:
  - **Step A**: add a second predator kind (`event_driven` alongside
    `short_horizon`) with a different diet bias. Both still on Qwen.
    Tests whether multiple predators competing for apex attention work
    in the existing architecture.
  - **Step B**: replace one predator's base model with Chronos-T5-small.
    The Channel between herbivore and that predator becomes a true
    cross-model adaptor (Qwen-2560 → Chronos-patch-tokens). Tests
    whether the trophic mechanics generalize across model families.

## v3+ direction: Bayesian causal reasoning as a parallel species axis

> **Status: design-stage. Not implemented. Captured here so we can return
> to it after the current SFT/IPO cycle settles.**

The trophic stack handles *soft* coordination (cross-attention, channels,
hidden states) well but doesn't do *symbolic* reasoning. Two SFT/GRPO
runs and one IPO run have surfaced consistent failure modes — wrong
ticker, wrong direction, uncalibrated confidence numbers — that are
fundamentally hard for token-CE-trained Channels to solve, because the
training signal can't distinguish "format right" from "content right" at
the token level.

A Bayesian causal network alongside the trophic stack fills exactly that
gap: every assertion gets a *named, typed home* with *calibrated
probability*, and the predator can run real do-calculus queries instead
of pattern-matching on text where causal reasoning was previously
expressed.

The cleanest way to fold this in *without* breaking the trophic framing
is to add **two new species** at existing tiers — not a new substrate or
a new mechanism, just new agents — and let trophic dynamics decide how
much trust the rest of the ecology puts in their output.

### Two new species, both at existing tiers

**1. BayesianNodeHerbivore** (tier 1 — new species alongside Technical /
Fundamental / Forecaster).

Eats the same producer broadcasts the consolidator herbivores eat
(filings, press, anomalies) but its output is *not* a hidden-state
broadcast. Instead it broadcasts a **structured posterior payload**:

```
{
  "AAPL_governance_concern": 0.72,
  "AAPL_supply_disruption": 0.81,
  "AAPL_disclosure_signal": -0.6,    # negative event
  ...
}
```

Internally: a small learned head maps each consumed substrate's hidden
state to `(node_name, posterior, uncertainty)` tuples against a fixed,
hand-built node taxonomy. The head is trainable; the taxonomy is not
(at least not initially).

Its broadcast carries:
  - The structured payload (machine-readable).
  - A short rendered text for legibility ("AAPL governance_concern=0.72,
    supply_disruption=0.81").
  - Optionally a hidden-state vector (so the LLM predator can also attend
    to it as soft signal — but the BayesianPredator reads the structured
    payload directly).

**2. BayesianPredator** (tier 2 — new species alongside the existing LLM
predator).

Eats `BayesianNodeHerbivore` broadcasts (and optionally the textual
herbivores' broadcasts via additional extractor heads). For each meal:

  1. Aggregate posteriors across consumed broadcasts via standard
     Bayesian update on the causal graph.
  2. Run do-calculus query: `P(AAPL_30min_direction | evidence)`,
     `P(AAPL_pct_move | evidence)`.
  3. Render the calibrated posterior as the predator's broadcast:
     `"PREDICTION: AAPL P(up_60min) = 0.78 (95% CI: [0.71, 0.84])"`.

Its broadcast also carries the structured payload so apex (when it
exists as a real tier) can read calibrated numbers directly rather than
parsing them out of text.

```
                    ┌───────────────────────┐
                    │   producer pool       │
                    └────┬─────┬────────────┘
                         │     │
              ┌──────────┘     └────────────────┐
              │                                  │
        textual herbivores              BayesianNodeHerbivore
        (Technical, Fundamental,        (new — emits structured
         Forecaster)                     posterior assertions)
              │                                  │
              ↓                                  ↓
         hidden-state                      structured-payload
         broadcasts                         broadcasts
              │                                  │
       ┌──────┴───────┐                          │
       │              │                          │
       ↓              ↓                          ↓
  LLMPredator    BayesianPredator    (also reads textual herb
  (current)      (new)               broadcasts via extractor heads)
       │              │
       └──────┬───────┘
              ↓
            apex
       (fusion tier)
```

### Why this design over the alternatives

We considered three shapes for Bayesian integration, in increasing order
of how ecological they are:

1. **Network as a parallel substrate** (sibling to the food pool, agents
   write evidence in / read posteriors out). Cleanest plumbing, but
   commits the *whole system* to using Bayesian reasoning regardless of
   whether it helps. Bypasses the trophic mechanism we built.

2. **Predator queries a network on the side**. Same architectural
   centrality problem — the predator is forced to interact with the
   graph whether it wants to or not.

3. **Two new species, predator + herbivore tiers** (this proposal).
   Specialization emerges from trophic dynamics. If the BayesianPredator
   is worse than the LLMPredator on apex-judged scores, it starves. If
   the BayesianNodeHerbivore's payloads are unused (because no
   downstream agent eats from it), it starves. The market decides
   whether the symbolic machinery earns its compute. **This is the
   honest answer to "should we use Bayesian networks?" — let the
   ecology decide.**

### What this gets right that pure trophic can't

  - **Wrong-ticker errors are caught by graph type constraints.** Each
    ticker has its own subgraph; an extractor that emits "AAPL up"
    can't also assert into the NVDA subgraph. The current Channel-based
    system has no analog of this — it learns soft correlations and fails
    when training data is thin.
  - **Calibrated uncertainty.** LLM "confidence: 0.7" is uncalibrated —
    the model produces those tokens because they look like confidence.
    A Bayesian posterior is calibrated by construction.
  - **Counterfactual reasoning.** "Would the prediction change if we
    hadn't seen the filing?" is a do-calculus query, trivially
    answerable on the graph. Currently impossible without ablation runs.
  - **Auditability at every tier.** `network.posterior("AAPL_governance")`
    is inspectable. The current predator's reasoning is `[role_prefix +
    3 channel vectors + query] → tokens`; nobody can ask "did the
    predator actually use the disclosure?" except by ablation.

### What it doesn't help with

  - **The graph has to come from somewhere.** Hand-built (expensive,
    brittle, doesn't generalize), learned from data (NOTEARS, FCI, GES;
    real but data-hungry), or LLM-extracted (mixed results). For a
    fintech v1 we'd hand-build ~128 nodes covering the 16 tickers × 5
    input-node-types + 3 target nodes. Tractable; not free.
  - **Continuous variables** require Gaussian assumptions, discretization,
    or sampling.
  - **Regime shift.** Financial relationships change faster than graphs
    typically learn. The graph needs to be either re-learnable or wide
    enough to remain valid across regimes.

### Apex becomes a real fusion tier

With both predators alive, apex stops being "judge predictor output" and
becomes "integrate two heterogeneous predictor outputs":

```
[LLMPredator]:        "AAPL likely up +1.5% over next 60 minutes,
                       confidence 0.8, supported by guidance reaffirmation
                       and tape consistency"

[BayesianPredator]:   P(AAPL_up_60min) = 0.78 (95% CI [0.71, 0.84]);
                      evidence: governance_concern=0.72,
                      tape_signal_up=0.91, forecast μ=+0.15% σ=1.7%.
```

Apex's job is the easier one of "these agree, high conviction" vs
"these disagree, low conviction, surface uncertainty." That's exactly
the right shape for an apex tier — it doesn't need to do the
calibration or the language, just the integration.

### Concrete build plan

Once the current SFT/IPO cycle settles and we want to break the next
ceiling:

**Phase 1: structured-payload broadcasts**
  - Extend `Broadcast` type to carry an optional `structured_payload`
    field alongside `channel_embedding` and `decoded_text`.
  - Add a small `extractors` head per textual-herbivore-kind that maps
    its hidden state to node assertions. Train against oracle assertions
    derived from scenario targets (every disclosure target already
    implies node values; we just have to extract them).
  - **No graph yet.** Just the typed-claim infrastructure.

**Phase 2: hand-built fintech graph**
  - 16 tickers × 5 input nodes + 3 target nodes ≈ 128 nodes.
  - CPTs from hand-coded priors over the 16 disclosure cases we have.
  - Inference engine: `pgmpy`. One-liner pip install, supports
    do-calculus and approximate inference.

**Phase 3: BayesianNodeHerbivore species**
  - Implements the tier-1 herbivore interface (eats producer broadcasts).
  - Internally: claim extractor heads (reuse Phase 1) + pgmpy update.
  - Broadcasts structured payloads.

**Phase 4: BayesianPredator species**
  - Implements the tier-2 predator interface (eats herbivore broadcasts).
  - Internally: aggregate posteriors → pgmpy query → render.
  - Broadcasts structured + text.

**Phase 5: apex as fusion**
  - Apex starts being a real tier (Bedrock Claude or in-process model)
    that takes both predator broadcasts and integrates.
  - Trophic dynamics: which predator gets cited more often by apex eats
    more.

**Phase 6: counterfactual evaluation**
  - Even before the network is fully wired, implement "drop one input,
    re-run, measure prediction delta" as a poor-man's intervention.
    Useful for explainability now.

Total scope estimate: ~600-800 lines for Phases 1-4, similar again for
5-6. Well-contained side experiment that runs alongside the trophic
work, doesn't replace it.

### Open research questions this raises

- **Trust ratio dynamics.** When apex weights LLMPredator and
  BayesianPredator differently across scenarios, can we read off the
  apex's revealed preference for symbolic vs neural reasoning per regime?
  Is that an interesting time series?
- **Graph learning from trophic feedback.** Decomposer judgments could
  update the CPTs in addition to (or instead of) updating Channel
  weights. The graph itself becomes a learned object.
- **When does the symbolic predator help?** Probably when training data
  is thin (graph type constraints prevent overfitting) and when
  calibration matters (apex needs to know "how sure"). Probably hurts
  when the regime is novel and the graph is wrong. The trophic dynamics
  surface this distinction empirically.
- **What's the right node taxonomy?** Hand-built is a guess; learned
  from data requires NOTEARS-class algorithms; LLM-extracted is the
  newest direction (Anthropic and others are exploring). Worth keeping
  the taxonomy as a swappable component.

## v3+ direction: heterogeneous models per tier

v1.5 uses Qwen3-4B at every tier as a deliberate simplification. The
architecture is in fact agnostic to which model family each tier runs on —
the Channel is the adaptor, and its job generalizes naturally to bridging
different latent spaces.

The honest tier-by-tier mapping is:

  - **Producers**: text/embedding models. Qwen or sentence-transformers
    fine — job is "raw input → hidden state representation."
  - **Herbivores**: text models. Synthesizing structured features from
    substrate is text-shaped. Qwen is reasonable.
  - **Predators**: should be a *forecasting* model, not a chat model.
    Their job is causal prediction of market outcomes from upstream
    syntheses — exactly what TimesFM / Chronos / Lag-Llama / Moirai /
    PatchTST are built for. A chat model trained on token-level CE will
    produce text *about* predictions; a forecasting model will produce
    actual probabilistic forecasts as its native output.
  - **Apex**: chat model again, for human-readable recommendations
    grounded in the predator's forecast distribution.

When models differ across tiers, `Channel_{tier_n → tier_{n+1}}` becomes
a true cross-model adaptor — projecting from one model family's embedding
space into another's input space (different dim, different distribution).
This is where representation-engineering meets cross-architecture
distillation, and it's the place where the system's design becomes most
interesting from a research perspective.

## v1 non-goals

- No `E` training (gradients off; energy dynamics only).
- No tier-2 consumer / tier-3 apex implementation (stubs only).
- No P&L simulator (predator is the optimization signal; P&L attribution is
  a v2+ concern).
- No real market data feed (synthetic generator only).
- No human-in-the-loop UI.
- No spatial / agent-based simulation.
- No vector DB.

## Build order

1. Project skeleton + `pyproject.toml` + this doc.
2. `config.py` (paths, model names, knobs).
3. `types.py` — pydantic message types.
4. `substrate.py` — pool with claim semantics + tests.
5. `model_host.py` — load Qwen3-4B once, expose generate + last-hidden-state.
6. `embeddings.py` — Qwen3-Embedding-0.6B for retrieval embeddings.
7. `appetite.py` + `couplings/{ecology,market}.py` + tests.
8. `inputs/synthetic.py` — synthetic fintech feed generator.
9. `agents/producer.py` — three concrete producer classes.
10. `agents/herbivore.py` — two concrete herbivore classes.
11. `predators/teacher.py` — Bedrock client + prompt + parsing.
12. `agents/decomposer.py` — attribution engine.
13. `ecology/{population,metrics}.py` — energy and instrumentation.
14. `runner.py` — async tick loop.
15. `agents/{consumer,apex}.py` — stubs with NotImplementedError.
16. `scripts/demo.py` + README.

Tests at minimum: substrate roundtrip + claim atomicity, appetite scoring,
ecology + market couplings, end-to-end smoke test with mocked Bedrock and
mocked Qwen.
