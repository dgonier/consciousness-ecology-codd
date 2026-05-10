# Bayesian belief network — schema (v0, redline draft)

Working name: `trophic/beliefs/`. Parallel to existing apex/herbivore tiers — not a replacement, an additional substrate. The ecology continues; the belief network is what apex now consults instead of guessing direction from text.

## Three node kinds, one edge kind

### `InternalLink` — the world model
Stable causal claim. Hand-seeded first, LLM-proposed extensions later (with human review). The validated weight on a link is a CPT entry — the conditional probability of the conclusion given the premise.

```python
@dataclass
class InternalLink:
    id: str                                    # e.g. "link.regulatory_standing→stock_price"
    premise_belief_id: str                     # references a state belief
    conclusion_belief_id: str                  # the belief affected
    scope: Literal["market", "sector", "company"]
    direction: Literal["positive", "negative"] # premise increase → conclusion increase / decrease
    strength_prior: float                      # initial belief in this link's reliability [0,1]
    strength_posterior: float                  # updated by historical validation
    n_validations: int                         # how many scenarios validated this link
    n_correct: int                             # how many of those agreed
    citation: str                              # paper / heuristic / "common-sense"
    validation_status: Literal["unverified", "validated", "rejected", "uncertain"]
    created_at: float
    updated_at: float
```

### `StateBelief` — current world state
What we currently believe is true. Has scope context. Decays toward prior without fresh evidence.

```python
@dataclass
class StateBelief:
    id: str                                    # e.g. "belief.regulatory_standing.AAPL"
    statement_template: str                    # "{ticker} has good standing with regulators"
    context: dict[str, str]                    # {"ticker": "AAPL"} or {"sector": "tech"} or {}
    scope: Literal["market", "sector", "company"]
    prior_p: float                             # what we believe absent any evidence
    current_p: float                           # current posterior
    decay_class: Literal["instant", "fast", "normal", "slow", "glacial"] = "normal"
        # global default with LLM-overridable speed; see DECAY_HALF_LIFE table
    last_updated: float                        # for decay calculation
    evidence_log: list[str]                    # ids of Events that updated this
    polymarket_market_id: Optional[str] = None
        # if set, current_p is read-through from market price; on resolution snaps
        # and propagates through standard log-odds machinery. LLM activations on
        # this belief are ignored — Polymarket is the authority.
```

**Decay half-life per class** (in hours):
```python
DECAY_HALF_LIFE_HOURS = {
    "instant":  0.001,    # ~snap (used for Polymarket-resolved beliefs)
    "fast":     1/60.0,   # 1 minute — intraday momentum, breaking news
    "normal":   1.0,      # 1 hour — standard news / sentiment
    "slow":     24.0,     # 1 day — earnings, guidance, regulatory
    "glacial":  168.0,    # 1 week — M&A pending, litigation, structural shifts
}
```
The LLM is instructed at activation time to pick the appropriate decay class for the belief being updated. Default when unspecified: `normal`.

### `OutcomeBelief` — prediction target
Computed, not predicted. Conditioned on internal links + state beliefs through belief propagation. The apex consumes this instead of guessing.

```python
@dataclass
class OutcomeBelief:
    id: str                                    # e.g. "outcome.AAPL.next_day_up.2026-05-08"
    ticker: str
    horizon_min: int                           # 1440 = next day close
    statement: str                             # "AAPL closes up tomorrow"
    p_up: float                                # posterior from belief propagation
    contributing_state_beliefs: list[str]      # what fed this
    contributing_links: list[str]              # which links carried the signal
    activated_at: float
    resolved: bool
    actual_direction: Optional[str]            # filled when ground truth known
```

### `Event` — the input
News headline, filing, price move, Polymarket resolution. The LLM's job: classify event → activate beliefs.

```python
@dataclass
class Event:
    id: str                                    # hash of source+timestamp+content
    timestamp: float
    source: str                                # "polygon_news" | "polymarket" | "ohlcv_anomaly"
    raw_content: str                           # the headline / event text
    ticker: Optional[str]                      # if company-scoped
    sector: Optional[str]                      # if sector-scoped
    activations: list["BeliefActivation"]      # what the LLM thinks this event affects
```

```python
@dataclass
class BeliefActivation:
    target_belief_id: str
    direction_of_effect: Literal["increases", "decreases"]
    magnitude: Literal["weak", "medium", "strong", "decisive"]
    decay_class: Literal["instant", "fast", "normal", "slow", "glacial"] = "normal"
        # LLM picks decay timescale appropriate to this activation
    self_rated_confidence: float               # LLM's self-rated confidence [0,1]
    magnitude_logit: Optional[float] = None    # logprob of the chosen magnitude class
        # Used as the actual confidence weight in math; self_rated kept for audit.
        # When logit is unavailable, fall back to self_rated.
    reasoning: str = ""                        # for audit
    species_id: str = ""                       # which species did the activation
```

**Two confidence fields, deliberate.** The LLM's self-rated number is unreliable
(LLMs systematically overconfident on classification). The logit on the magnitude
class is the model's actual posterior over the categorical choice and correlates
with calibration. Keep both: use logit for math; surface self-rated for audit and
for tracking which species' self-ratings are well-calibrated over time.

**Magnitude → log-odds shift table** (calibrated against historical data, refined by Hexis later):
```python
MAGNITUDE_TO_LOG_ODDS = {
    "weak":     0.4,    # ~odds × 1.5
    "medium":   1.0,    # ~odds × 2.7
    "strong":   1.6,    # ~odds × 5
    "decisive": 2.3,    # ~odds × 10
}
```

The LLM never outputs probabilities. It outputs categorical magnitudes + reasoning; the table converts.

## Categorical-only LLM interface

**The LLM's entire interface to the belief network is categorical.** Every choice
the LLM makes is one-of-N from a fixed enum. Every numeric value used in the
math is computed from the categorical via a lookup table. Recalibration =
adjust the tables; no retraining/reprompting.

Why: LLM self-rated probabilities are systematically miscalibrated. Categorical
classification, by contrast, exposes the model's actual posterior via logprobs —
a real calibration signal. And lookup tables make the network's behavior
transparent and tunable.

### Categorical vocabularies

```python
# Strength of evidence behind a proposed belief or link
EvidenceLevel = Literal["novel", "weak", "moderate", "strong", "well_established"]

# Directional lean — used together with EvidenceLevel to derive prior_p
DirectionalLean = Literal[
    "strongly_false", "leans_false", "neutral", "leans_true", "strongly_true",
]

# Magnitude of an activation's effect on a belief (existing)
Magnitude = Literal["weak", "medium", "strong", "decisive"]

# Decay timescale (existing)
DecayClass = Literal["instant", "fast", "normal", "slow", "glacial"]

# Self-rated confidence (replaces float)
ConfidenceLevel = Literal["low", "medium", "high", "very_high"]
```

### Lookup tables

```python
# (EvidenceLevel, DirectionalLean) → prior_p
PRIOR_P_TABLE = {
    # Novel: no historical evidence of any kind. Always agnostic.
    ("novel", "strongly_false"):  0.50,
    ("novel", "leans_false"):     0.50,
    ("novel", "neutral"):         0.50,
    ("novel", "leans_true"):      0.50,
    ("novel", "strongly_true"):   0.50,
    # Weak: one or two observations, anecdotal pattern.
    ("weak", "strongly_false"):   0.40,
    ("weak", "leans_false"):      0.45,
    ("weak", "neutral"):          0.50,
    ("weak", "leans_true"):       0.55,
    ("weak", "strongly_true"):    0.60,
    # Moderate: documented in some research, ~years of data.
    ("moderate", "strongly_false"): 0.30,
    ("moderate", "leans_false"):    0.40,
    ("moderate", "neutral"):        0.50,
    ("moderate", "leans_true"):     0.60,
    ("moderate", "strongly_true"):  0.70,
    # Strong: well-cited, decade-plus track record.
    ("strong", "strongly_false"):   0.20,
    ("strong", "leans_false"):      0.35,
    ("strong", "neutral"):          0.50,
    ("strong", "leans_true"):       0.65,
    ("strong", "strongly_true"):    0.80,
    # Well-established: textbook regularity, multiple decades, citable base rate.
    ("well_established", "strongly_false"): 0.15,
    ("well_established", "leans_false"):    0.30,
    ("well_established", "neutral"):        0.50,
    ("well_established", "leans_true"):     0.70,
    ("well_established", "strongly_true"):  0.85,
}

# ConfidenceLevel → numeric weight for math
CONFIDENCE_TABLE = {"low": 0.4, "medium": 0.6, "high": 0.8, "very_high": 0.95}
```

### How the LLM proposes a new belief

When the LLM identifies a need for a new state belief (rare, decomposer-cycle
work), it produces:

```
NEW_BELIEF_STATEMENT: "<the proposition>"
EVIDENCE_LEVEL: novel | weak | moderate | strong | well_established
DIRECTIONAL_LEAN: strongly_false | leans_false | neutral | leans_true | strongly_true
SCOPE: macro | sector | company
DECAY_CLASS: instant | fast | normal | slow | glacial
REASONING: <why these classifications>
```

We look up `prior_p = PRIOR_P_TABLE[(evidence_level, directional_lean)]` and
initialize `current_p = prior_p`. The LLM never says "0.7" — it says
`well_established + leans_true`.

Examples to ground the LLM's reasoning:
- "S&P 500 closes up on a given trading day" → `well_established` + `leans_true` → 0.70
  (Decades of data; documented positive equity drift.)
- "AAPL CEO Tim Cook resigns this quarter" → `well_established` + `strongly_false` → 0.15
  (Established that CEO transitions happen at predictable rates; this specific
  event is unlikely-given-no-news.)
- "AI Lab ABC files for IPO this year" → `weak` + `neutral` → 0.50
  (Not enough evidence to lean either way without specific signals.)
- "New regulatory framework adopted in EU on AI safety" → `moderate` + `leans_true`
  → 0.60 (Pattern of EU regulatory action in tech, but specific outcome uncertain.)
- "AAPL faces SEC enforcement this quarter" → `well_established` + `leans_false`
  → 0.30 (SEC enforcement happens; specific company in any quarter is rare.)

### Hand-seeded vs LLM-proposed

For hand-seeded beliefs (the v0 YAML inventory), humans can set numeric
`prior_p` directly with a citation. They're effectively pre-classifying. The
categorical interface is enforced only for LLM-proposed beliefs.

## The hierarchy and edge directions

Strict downward propagation:
```
Macro → Sector → Company → Outcome
```

An InternalLink can only go from a higher or same scope to a same or lower scope. Never upward. This prevents feedback loops in the network and keeps the propagation graph a DAG.

Examples that ARE allowed:
- `link.macro.fed_hawkish → company.AAPL.borrow_cost_up` (macro → company)
- `link.sector.ai_under_scrutiny → company.NVDA.regulatory_risk_up` (sector → company)
- `link.company.AAPL.earnings_beat → outcome.AAPL.next_day_up` (company → outcome)

Not allowed:
- `link.company.AAPL.x → sector.tech.y` (company → sector — would be feedback)

## Belief propagation

Standard Bayesian update via log-odds. For an outcome belief with parent state beliefs `B1..Bn` connected by links `L1..Ln`:

```
log_odds(outcome) = log_odds(prior)
                  + sum_i [ link_i.strength_posterior
                          × sign(link_i.direction)
                          × (current_p(B_i) − prior_p(B_i))
                          × scaling_factor ]
```

Decay between events — **toward `prior_p` (the base rate)**:
```
current_p(t) = prior_p + (current_p(t-Δt) − prior_p) × 0.5^(Δt / half_life)
```

Or equivalently in log-odds space (the deviation from prior log-odds decays to 0):
```
delta_log_odds(t) = delta_log_odds(t-Δt) × 0.5^(Δt / half_life)
log_odds(t) = log_odds(prior_p) + delta_log_odds(t)
```

Rationale: credence ranges over [0, 1] where 0 = certain-false, 1 = certain-true,
0.5 = maximum uncertainty. What decays is **the evidence's contribution to
current credence**, not credence itself. After full decay we should be left with
whatever we'd have believed absent the evidence — i.e., the base rate `prior_p`.

For most beliefs `prior_p = 0.5` (we have no meaningful base rate; absent evidence
we're agnostic). For these beliefs, decay-to-prior and decay-to-0.5 are identical.

For beliefs with a real base rate (e.g. "company faces SEC enforcement in any
given quarter" ≈ 0.05), `prior_p ≠ 0.5` and the distinction matters: after a
probe announcement decays out, credence should return to ~0.05 (the base rate),
not to 0.5 (claiming we're now agnostic about whether SEC enforcement is likely).

Worked examples:
- "AAPL has good standing" (`prior_p = 0.5`): after a probe pushes credence to
  0.2 and 24h pass at slow decay, current_p = 0.35. After 2 weeks, current_p ≈
  0.500 (back to agnostic — correct, no base-rate signal either way).
- "Company faces SEC enforcement" (`prior_p = 0.05`): after probe pushes credence
  to 0.7 and 24h pass, current_p = 0.375. After 1 week, current_p ≈ 0.055 (back
  near base rate — correct).
- "Earnings beat lands" (`prior_p = 0.55`): after a whisper pushes credence to
  0.8 and 1 day passes, current_p = 0.675. After 1 week, current_p ≈ 0.552
  (back to base rate — correct).

## LLM's role — three discrete jobs

The LLM never predicts direction. It does:

1. **Event activation**: "Given this headline, which state beliefs does it affect, in which direction, with what magnitude, at what scope context?"
2. **Belief proposal** (rare, with human review): "Based on this scenario's outcome, propose a new internal link that would have predicted it correctly."
3. **Reasoning summarization** (audit only): "The outcome came out wrong; given the activated beliefs and links, what was the dominant failure path?"

Job 1 is the hot path. Jobs 2 and 3 are decomposer-cycle work.

## Storage — Neo4j-native, JSONL-backed for now

Cypher schema:
```cypher
CREATE CONSTRAINT belief_id IF NOT EXISTS
  FOR (b:StateBelief) REQUIRE b.id IS UNIQUE;
CREATE CONSTRAINT outcome_id IF NOT EXISTS
  FOR (o:OutcomeBelief) REQUIRE o.id IS UNIQUE;
CREATE CONSTRAINT event_id IF NOT EXISTS
  FOR (e:Event) REQUIRE e.id IS UNIQUE;

(:StateBelief)-[:LINK {strength, direction, scope}]->(:StateBelief)
(:StateBelief)-[:LINK {strength, direction, scope}]->(:OutcomeBelief)
(:Event)-[:ACTIVATED {magnitude, confidence, reasoning, species_id, ts}]->(:StateBelief)
(:Event)-[:OBSERVED_AT {ts}]->(:Scenario)
(:OutcomeBelief)-[:RESOLVED_AS {actual, ts}]->(:Outcome)
```

Adapter pattern:
- `BeliefStore` ABC with `store_state_belief`, `store_link`, `query_links_for`, `query_state_beliefs_in_scope`, etc.
- `JSONLBeliefStore` impl writes one file per kind under `external/beliefs/`. Used in dev.
- `Neo4jBeliefStore` impl when DB is available. Same interface.

## Polymarket as evidence source

A `StateBelief` can be **bound** to a Polymarket market by setting
`polymarket_market_id`. While the market is open:
- `current_p` is read-through from the market's current YES price (no LLM, no
  log-odds machinery).
- LLM-emitted activations targeting this belief are ignored. Polymarket is the
  authority on its own state belief.
- Downstream propagation runs normally: when the market price moves, child
  beliefs see the delta as a regular log-odds update event.

When the market resolves:
- Bound state belief snaps: `current_p = 1.0` if YES, `0.0` if NO.
- The snap propagates through the standard log-odds machinery (the parent's
  delta gets converted to log-odds and pushed to children).
- After resolution, the bound belief has `decay_class = "instant"` (locked in
  truth) — but in v0 we leave resolved beliefs frozen to keep the experiment
  clean.

**Why hybrid (snap-the-bound + log-odds-propagation) over either pure approach:**
- Pure instant snap on everything: discontinuities propagate as discontinuities;
  downstream math gets distorted by sudden jumps.
- Pure log-odds: a resolved YES market becomes 0.97 instead of 1.0; numerically
  sloppy at the edges where we have ground truth.
- Hybrid: bound belief respects truth; downstream math stays uniform.

State beliefs that map cleanly to Polymarket markets (concrete examples for
v0 wiring):
- `belief.macro.fed_cuts_rates_by_july_2026` ← Polymarket "Fed cuts by July 2026"
- `belief.macro.recession_called_2026` ← Polymarket "US recession in 2026"
- `belief.sector.ai_regulation_passes_2026` ← Polymarket "AI Act variants 2026"
- `belief.company.{ticker}.sec_enforcement_2026` ← Polymarket per-company markets

The Polymarket bridge is a **baseline-belief seeder**: it gives us probability
estimates from the human community, free, without needing LLMs to formulate
priors.

## Hexis integration (later, not now)

Hexis-style decomposers update **link.strength_posterior** from observed outcomes. The mechanism:
1. Each scenario produces an `OutcomeBelief` with predicted `p_up`.
2. Ground truth resolves the outcome (`actual_direction`).
3. For every link that contributed to that outcome's posterior, attribution is assigned proportional to its load on the prediction.
4. If the prediction was right, link strength nudges up; wrong, down.
5. Across many scenarios, `strength_posterior` converges to the link's actual reliability — a per-link MCC.

This is a textbook Hexis update. Defer until network is stable + producing predictions.

## OutcomeBelief consumes everything (best-signal hybrid)

Decision (locked): `outcome.p_up` propagates from ALL contributing evidence on
equal mathematical footing:

- **News events** → LLM-classified `BeliefActivation` records → state beliefs
- **Quant signals** (multi-timeframe direction, OBV slope, etc.) → deterministic
  `BeliefActivation` records emitted by a "quant species" with high
  `magnitude_logit` (these are computed, not classified, so logit ≈ 1.0)
- **Herbivore syntheses** → each herb species emits one `BeliefActivation` per
  scenario; species_id is the herb's id, magnitude is parsed from the synthesis,
  reasoning is the synthesis text
- **Polymarket prices** → bound state beliefs read-through; deltas propagate via
  log-odds
- **Chronos forecast** → numeric activation on `belief.company.{ticker}.short_drift_positive`
  with magnitude scaled to the forecast's signal-to-noise

Apex tier reads `outcome.p_up` rather than guessing direction from text. No
parallel "apex predicts → belief network predicts" comparison; the apex IS the
network's posterior.

## What's NOT in this version

- No automated link discovery (LLM proposes, human accepts in v1)
- No belief contradiction resolution (two events one bullish one bearish — current rule: both update independently, log-odds sum)
- No multi-step reasoning across multiple links per scenario (only direct parent-of-outcome state beliefs influence outcome posterior)
- No time-series of state beliefs (only "current" — history is in evidence_log)
- Polymarket-bound beliefs frozen post-resolution (no decay back toward 0.5 once
  market resolves; revisit in v1)

Each of these can layer on once the v0 round-trip works.

## Locked decisions (was: open questions)

1. **`BeliefActivation` confidence** → both fields. `magnitude_logit` for math,
   `self_rated_confidence` for audit + species calibration tracking.
2. **Decay** → global default with LLM-overridable per-activation `decay_class`
   ∈ {instant, fast, normal, slow, glacial}. Half-lives in hours: 0.001, 1/60,
   1, 24, 168.
3. **Polymarket resolution** → bound belief snaps to 0/1; downstream propagates
   via standard log-odds machinery. Hybrid approach.
4. **OutcomeBelief consumption** → everything (best-signal hybrid). News +
   quant + herbs + Polymarket + Chronos all feed the network on equal footing
   through `BeliefActivation` records.
