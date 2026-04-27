# trophic

Multi-agent fintech-prediction system organized along ecological lines: producers → herbivores → predator → apex. Inter-tier coordination happens through **learned cross-attention with population dynamics** — slots compete, decay, die, and respawn against a learned niche model — rather than through orchestration or a static mixture-of-experts.

```
raw input  →  PRODUCERS  →  TROUGH (cross-attn K/V)  →  HERBIVORES  →  TROUGH  →  PREDATOR
                                                                                       │
                                                                              [APEX — judging tier]
                                                                                       │
                                                                                  DECOMPOSER
                                                                          (writes back as attention bias)
```

The architecture spec lives in [`docs/consumption_transformers.md`](docs/consumption_transformers.md). The full build plan (six parallelized missions) lives in [`tasks/00-README.md`](tasks/00-README.md). Running experimental results live in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Architecture in one paragraph

Each tier boundary is a `TroughAttention` — a stateful K/V container with multi-head cross-attention readout, slot-level lifecycle (attention decay → death → niche-aware respawn), and a decomposer feedback path that injects judgment as attention bias. Producers (TickDelta, Disclosure, Anomaly, QuoteSeries) emit hidden-state broadcasts into the producer trough. Herbivores (Technical, Fundamental, Forecaster, Interrogator) attend to the producer trough through the same mechanics; their output goes into the herbivore trough. The Predator attends to the herbivore trough *plus* a learnable skip path back to the producer trough (residual across two tiers, with a learned mixing weight α). A cosine temperature schedule anneals the softmax τ from 2.0 → 0.5 over training (warm exploration → sharp exploitation). All trained end-to-end with SFT then IPO.

## What's where

| Path | What's there |
|---|---|
| `trophic/trough_attention.py` | The core `TroughAttention` class. K/V deposit, multi-head attend, lifecycle |
| `trophic/substrate.py` | Thin shim that delegates to `TroughAttention` (legacy `SubstratePool` API preserved) |
| `trophic/agents/` | `Producer`, `Herbivore`, `Predator`, `InterrogatorHerbivore` (math model bridge) |
| `trophic/ecology/slot_lifecycle.py` | Pure functions: decay, kill, niche-aware spawn |
| `trophic/decomposer.py` | Apex judgment → attention bias |
| `trophic/training/sft.py` | SFT trainer with τ schedule |
| `trophic/training/ipo.py` | IPO (Identity Preference Optimization) trainer |
| `trophic/training/scenarios.py` | 299 synthetic scenarios (16 tickers × archetypes) |
| `trophic/training/holdout_scenarios.py` | 68 held-out scenarios with disjoint tickers |
| `trophic/training/stocknet_loader.py` | ACL-18 StockNet adapter for external benchmarking |
| `scripts/eval_xml_checkpoint.py` | Real-data predator-reward eval on dev |
| `scripts/eval_stocknet.py` | Directional-prediction eval on the StockNet test split |
| `scripts/train_sft.py` / `scripts/train_ipo.py` | Training entry points |
| `tests/` | 51 tests covering trough mechanics, lifecycle, decomposer, skip connections |
| `tasks/` | The six-mission build plan that produced the current architecture |
| `todo/` | Ongoing experimental threads (benchmarks, paper decision, Bayesian nodes) |
| `docs/` | Architecture spec, integration plans, results log |

## Models in the stack

- **Qwen3-4B** — primary herbivore + predator forward
- **Qwen2.5-Math-1.5B** — interrogator's solver phase (token-bypass via `CrossModelChannel`)
- **Chronos-Bolt** — quantitative forecaster herbivore

## Run it

```bash
pip install -e .

# Tests (51, no GPU, runs in ~40s):
.venv/bin/python -m pytest tests/

# Mock-mode SFT smoke (no real models, exercises the full pipeline):
TROPHIC_MOCK_MODELS=1 TROPHIC_MOCK_PREDATOR=1 TROPHIC_STEPS=10 \
  .venv/bin/python scripts/train_sft.py

# Real SFT on a 4090 (~85 min for 600 steps):
TROPHIC_SEED=7 TROPHIC_STEPS=600 .venv/bin/python -u scripts/train_sft.py \
  > logs/sft_seed7.log 2>&1 &

# Real IPO on top of the SFT checkpoint (~75 min for 800 steps):
TROPHIC_SEED=7 TROPHIC_IPO_STEPS=800 .venv/bin/python -u scripts/train_ipo.py \
  > logs/ipo_seed7.log 2>&1 &

# Real eval on the 14-scenario dev set:
TROPHIC_CKPT=checkpoints/ipo_seed7_best.pt TROPHIC_SEED=7 \
  .venv/bin/python -u scripts/eval_xml_checkpoint.py

# StockNet smoke (top-5 tickers × 10 days, ~25 min):
TROPHIC_CKPT=checkpoints/ipo_seed7_best.pt TROPHIC_SEED=7 STOCKNET_MAX_PER_TICKER=10 \
  .venv/bin/python -u scripts/eval_stocknet.py
```

## Headline results (snapshot — see [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the full table)

| Setup | Dev (14) | Held-out (68 fresh tickers) | StockNet test |
|---|---|---|---|
| Pre-migration SFT (seed 1) | 0.326 | TBD | — |
| Post-migration SFT (seed 7) | 0.331 | **0.257** | TBD |
| Pre-migration IPO v3 (seed 1) | 0.388 | TBD | — |
| Post-migration IPO (seed 7) | **0.543 (best @ step 400)** | TBD | TBD |

The post-migration IPO at +40% over the pre-migration IPO baseline (0.543 vs 0.388) is the result that motivates a paper. **Numbers will solidify after the in-flight IPO run finishes and the held-out / StockNet evals run.**

## Status

Migration complete — `SubstratePool` collapsed into `TroughAttention` with full ecological lifecycle. 51/51 tests passing. End-to-end SFT + IPO runs healthy. Currently:

- IPO seed 7 finishing
- Held-out evals queued
- StockNet plumbing ready, smoke run pending IPO completion

## Memory snapshot for collaborators

The project has lived through several distinct architecture eras. The relevant context is:

- **v1 (deprecated)**: SQLite SubstratePool, hand-coded diet-tag SQL, separate `Channel` modules per herbivore.
- **v2 (current)**: TroughAttention unifies pool + channel; multi-head specialization replaces diet tags; population dynamics on K/V slots.
- **v3 (designed, not built)**: Bayesian probability nodes alongside textual herbivores. See [`docs/architecture.md`](docs/architecture.md) v3+ section and [`todo/03-bayesian-nodes.md`](todo/03-bayesian-nodes.md).

## Related work

The TroughAttention architecture is closest to a **dynamic-population mixture-of-experts** (Shazeer 2017, Switch Transformer 2021) with the difference that experts are *born and die during training*, with niche-aware spawning replacing static expert allocation. Decomposer feedback as attention bias is a soft analogue of Pearl-style do-calculus running on the attention matrix. Skip connections from predator to producer are residuals across the trophic chain; if α stays low (< 0.5), intermediate tiers are doing useful work. See [`docs/consumption_transformers.md`](docs/consumption_transformers.md) for the full motivation.

## License

See [LICENSE](LICENSE). Research code; rights retained pending publication. Contact dgonier@gmail.com.
