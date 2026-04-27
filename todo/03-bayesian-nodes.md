# 03 — Bayesian Probability Node Species

**Status**: DESIGNED, NOT BUILT
**Designed in**: `trophic/docs/architecture.md` v3+ section (~lines 473-650)

## What this is

Two new agent species that join the existing trophic stack at existing tiers. NOT a parallel substrate, NOT a top-down rewrite — just two new "kinds" alongside Technical/Fundamental/Forecaster (tier 1) and the LLM Predator (tier 2):

1. **`BayesianNodeHerbivore`** — eats producer broadcasts, emits a structured posterior payload (calibrated `P(node_name | evidence)` for each node in a hand-built taxonomy).
2. **`BayesianPredator`** — eats `BayesianNodeHerbivore` broadcasts, runs Bayesian update + do-calculus on a fixed causal graph, emits calibrated predictions with confidence intervals.

The current LLM predator and the new Bayesian predator coexist. Apex (when it becomes real) decides how much trust to put in each.

## Why it matters

- Three failure modes from prior runs are about **calibration and content correctness**: wrong ticker, wrong direction, uncalibrated confidence. These are hard for token-CE-trained Channels to fix because the loss can't distinguish "format right" from "content right."
- Bayesian nodes give every assertion a *typed home* and a *calibrated probability*. Reviewers love calibration evidence.
- Stays inside the trophic frame — no architectural inversion. Just new species.

## Build order (when triggered — see decision criterion in 00-INDEX)

### Phase 1: Hand-built node taxonomy
- File: `trophic/agents/bayesian/taxonomy.py`
- Define ~30-50 nodes: `{ticker}_{event_type}` (e.g., `AAPL_governance_concern`, `NVDA_supply_disruption`, `MSFT_earnings_surprise`).
- Each node: name, type (binary | categorical | continuous), parent nodes (for the causal graph).
- The taxonomy is fixed and shared across tickers — node templates parameterize over ticker.

### Phase 2: Causal graph + do-calculus engine
- Use `pgmpy` or roll-our-own. Keep small (~50 nodes, ≤5 parents each).
- Cache compiled inference per query shape; query latency must be ≤100ms.
- Test with synthetic evidence to verify P(direction | evidence) makes sense.

### Phase 3: BayesianNodeHerbivore
- New file: `trophic/agents/bayesian_node_herbivore.py`
- Architecture: small head on top of Qwen3-4B's last hidden state that maps the producer broadcast to (node_name, posterior, uncertainty) for the K most-likely nodes.
- Output type: `Broadcast(tier="herbivore_broadcast", agent_kind="bayesian_node", payload={node_posteriors: dict[str, float], rendered_text: str})`.
- The payload is the contract. The rendered_text is for legibility / mode debugging.

### Phase 4: BayesianPredator
- New file: `trophic/agents/bayesian_predator.py`
- Eats BayesianNodeHerbivore broadcasts. Aggregates posteriors (Bayesian update across multiple herbivore agents using same node).
- Runs `P(direction | evidence)` and `P(pct_move | evidence)` queries on the causal graph.
- Renders calibrated XML prediction.

### Phase 5: Training
- Two-stage: SFT first (the Bayesian head learns to map text → node_posteriors against synthetic supervision; calibration is rule-based).
- Then preference optimization: use the same IPO setup we have. Reward function adds calibration term (Brier score against ground truth).

### Phase 6: Eval
- Integrated into the same `scripts/eval_xml_checkpoint.py` flow.
- New metric: **Brier score on direction probability** alongside our existing rule-based reward.

## Open design decisions

- **Should BayesianNodeHerbivore replace or augment textual herbivores?** Augment is the spec. Replace would commit too hard to symbolic reasoning.
- **How does the LLM predator combine its hidden-state attention with the Bayesian predator's structured payload?** TBD. Probably: apex tier (when real) consumes both broadcasts and learns a fusion.
- **Does the causal graph stay fixed or is it learned?** Phase 1: fixed. Future: structure learning if we have data volume.

## Estimated effort

5-7 days of focused work. Splits roughly:
- 1 day: taxonomy + graph
- 1 day: BayesianNodeHerbivore architecture + training scaffold
- 1 day: BayesianPredator + do-calculus integration
- 1-2 days: eval integration + first training run
- 1-2 days: debugging the inevitable calibration issues

## Trigger conditions for starting

Don't start unless ONE of these is true:
- (a) StockNet/StockBench smoke reveals a calibration failure that this design would fix.
- (b) Textual stack hits a clear ceiling at <0.55 dev reward despite IPO + held-out validation.
- (c) Paper draft outline reaches the "limitations" section and lists "calibration is uncalibrated" as the headline limitation.

If none of those are true, deprioritize and stay on the textual stack.
