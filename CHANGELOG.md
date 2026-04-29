# Changelog

All notable changes to this project will be documented in this file. Append new entries to the top of the relevant section. Reference GitHub issues (`#N`) and commit SHAs where possible so future readers can navigate back to the source of truth.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions are `0.X.Y` until first release.

## [Unreleased]

### Added
- **EnvironmentStream — input-layer K/V substrate** (`trophic/environment_stream.py`, ENVSTREAM project). Composes `TroughAttention` with per-slot source-tag bookkeeping and wavelength-filtered attend (`-inf` external_bias). Producers can deposit/attend through the stream as a parallel substrate to the existing producer-tier troughs. Wired into `runner.py` behind `TROPHIC_USE_ENVSTREAM=1` opt-in flag. (#9)
- **Adapter ABC** (`trophic/adapters/base.py`) and **wavelength registry** (`SOURCE_TAGS_VOCAB` in `trophic/types.py`). Single source of truth for what tags concrete adapters may emit; `__init_subclass__` validates at class-definition time. (#9)
- **StockNet adapters** (`trophic/adapters/stocknet/`): `OhlcvNormalizedAdapter` (source="ohlcv"), `TokenizedTweetAdapter` (source="tweets"). 6 new tests; live round-trip verified on cached AAPL data (1257 rows). (#9)
- **`Producer.WAVELENGTHS` + `KIND` class attrs**, replacing the old global `ATTRACTION` dict. `attracts(inp) = inp.source in self.WAVELENGTHS`. `Producer.make(kind)` walks subclasses for backward compat. (#9)
- **`SocialSignal` producer** (`trophic/agents/social_signal.py`) subscribing to `WAVELENGTHS={"tweets"}`. Seeded into the runner population alongside the three legacy text producers. (#9)
- **`Producer.produce_from_stream` + `role_q`** for the EnvironmentStream-driven path (Mission 05). Producer takes a `StreamAttendOutput`, emits a substrate broadcast whose embedding is the attended hidden-state vector. (#9)
- **EnvironmentStream smoke diag** (`scripts/diagnostics/diag_envstream_smoke.py`): deposits 5 days of AAPL OHLCV+tweets through adapters and reports per-producer attention distributions. Validates cortical-column fan-out (TickDelta/Anomaly attend ohlcv slots; SocialSignal attends tweets slots; Disclosure/QuantitativeProducer correctly mask out everything when their wavelengths are silent). (#9)
- **8 new tests** in `tests/test_environment_stream.py`, **6** in `tests/test_stocknet_adapters.py`, **8** in `tests/test_producer_wavelengths.py`, **7** in `tests/test_adapters.py` — total **83 passing** (51 baseline + 32 new). (#9)
- StockNet (ACL-18) loader and eval harness — `trophic/training/stocknet_loader.py`, `scripts/eval_stocknet.py`. Smoke-test mode caches HTTP responses at `external/stocknet_cache/`. (#5)
- Held-out test set with 68 fresh-ticker scenarios disjoint from training — `trophic/training/holdout_scenarios.py`. (#4)
- `remember` and `recall` global Claude Code skills for capturing ideas as structured GitHub issues and navigating them. Standardized labels: `status:*`, `origin:remember`, `topic:*`. Both skills auto-detect repo from cwd.
- `swarm` global Claude Code skill for scaffolding multi-agent coordination directories.
- `tasks/` directory: phase-ordered swarm coordination history from the trough-as-transformer migration (six missions, all DONE).
- `todo/` directory: ongoing experimental threads (StockNet, StockBench, Bayesian nodes, paper-go decision).
- `docs/EXPERIMENTS.md`: append-only experiment log with reward tables and methodology notes.
- `docs/stocknet_integration_plan.md`: full integration spec for the StockNet benchmark.
- `CONTRIBUTING.md`: working agreements, checkpoint hygiene, repo layout.
- `LICENSE`, `.gitignore`, `.gitattributes`.

### Changed
- `trophic/training/stocknet_loader.py` now routes parsing through `OhlcvNormalizedAdapter` / `TokenizedTweetAdapter` instead of inline tab-split / JSON-decode loops. HTTP fetch + caching stay in the loader. Phase1-A TODO markers (loader lines 69, 99; scenarios line 45) removed. (#9)
- `trophic/runner.py` now seeds `SocialSignal` into the producer population unconditionally; lazily allocates an `EnvironmentStream` and routes through the deposit-then-attend pipeline when `TROPHIC_USE_ENVSTREAM=1`. (#9)
- `README.md` rewritten from v1-era (SQLite SubstratePool + C/E coupling framing) to v2 (TroughAttention with ecological lifecycle).
- `SubstratePool` is now a thin shim delegating to `TroughAttention` per (tier, agent_kind) tuple. Public API preserved.
- `trough_attention.py`: replaced static cross-attention with multi-head + EMA-decay + niche-aware spawn + decomposer-bias hook + skip-α connection. (Six-mission migration; see `tasks/`.)
- `predator.py`: added `producer_trough` arg to `hunt_and_predict` for skip-connection residual; `skip_weight` learnable scalar.
- `sft.py`: cosine τ schedule wired in via `tau_schedule.py`; per-tick ecology snapshot logging (alive/killed/spawned/head_entropy/skip_α).
- Diet-tag SQL routing removed from `herbivore.py` and `predator.py`. Multi-head attention does the routing; `Broadcast.diet_tags` survives as debug annotation only.

### Results
- **Post-migration IPO held-out reward = 0.334**, vs pre-migration IPO held-out = 0.289 (+15.6%). Generalization gap closed from -25% to -10%. (#4)
- **StockNet smoke: 0.720 accuracy / 0.000 MCC** — apparent baseline-beat is mechanically a constant-"up" predictor. Confirms second axis of mode collapse alongside the ticker-collapse. (#5, escalates #6)
- **Post-fix SFT seed 8: best eval_loss = 0.192** at step 600, vs seed 7's pre-fix 0.279 best (-31%). Reached seed 7's all-time best 150 steps earlier. herb.technical converged to ~0.001 (vs ~0.282 pre-fix). The architectural #6 fix (input-conditioned hunter_state) is paying off at SFT.
- **Post-fix IPO seed 8: dev rule-based reward = 0.414, held-out = 0.372.** Exceeds every prior baseline on both metrics. vs pre-migration IPO baseline (0.388 dev / 0.289 held-out): **+6.7% dev, +28.7% held-out**. vs pre-fix ipo_seed7 (0.370 / 0.334): +11.9% dev, +11.4% held-out. Generalization gap identical to pre-fix at -10% — both dev and held-out moved up by similar absolute amounts (+0.044 / +0.038), confirming the gain is real, not memorization. Judge metric peaked 0.564 (+3.9% over seed 7's 0.543 ceiling). The #6 fix validates end-to-end on rule-based metrics.
- **Post-fix StockNet smoke: ACCURACY=0.720, MCC=0.000** — *bit-identical* to pre-fix (TP=36 TN=0 FP=14 FN=0). Direction-field collapse is a separate failure mode from the ticker collapse the #6 fix addressed. Tracked as #8.
- **Post-EnvironmentStream IPO seed 9 v2: dev = 0.417 (best ever), held-out = 0.346 (regression vs ipo_seed8 0.372), StockNet MCC = 0.000** — judge eval 0.607 at step 100 (best ever, 4× faster than seed 8). EnvironmentStream lifts training-set metrics but widens dev/held-out gap (-10% → -17%) AND does NOT touch direction collapse. Verdict: architecturally orthogonal to #8. The bottleneck is reasoning structure or pretraining distribution, not input substrate. Routes to #11 (FinCoT prompting) and #10 (per-species CPT). (#9)
- **FinCoT LoRA + co-training (#10 phase 1 path A, NEGATIVE result):** Co-trained Channels with a FinCoT-tuned Qwen3-4B LoRA active throughout SFT+IPO. Two co-training runs:
  - **ipo_seed12** (with partial-50 LoRA, eval_loss 0.898): dev = 0.423 (new best), held-out = 0.328 (regression), StockNet ACC=0.720 / MCC=0.000. Judge peaked 0.614 at step 100. Predator emits "MSFT up" constant.
  - **ipo_seed14** (with step-200 LoRA, eval_loss 0.862, 4× more trained): dev = 0.280 (collapse), StockNet ACC=0.720 / MCC=0.000. Same "MSFT up" constant.
  - Verdict: more FinCoT training makes things worse, not better. FinQA reasoning is not the bottleneck for #8.
- **SocialSignal LoRA + co-training (#10 phase 1 path B, NEGATIVE result):** Trained Qwen3-4B LoRA on TimKoornstra/financial-tweets-sentiment (38k bullish/bearish/neutral tweets). LoRA training was clean — best dev loss 0.142 at step 300, no stalls (vs FinCoT's 4 stall reproductions). Co-trained Channels (sft_seed16 dev 0.301, ipo_seed16 judge peaked 0.486 at step 400 with monotone improvement, unlike FinCoT's early-peak-decline).
  - **ipo_seed16 dev = 0.298, StockNet ACC=0.280 / MCC=0.000.** Predator now collapses to **"MSFT down" or "GOOGL down"** on every scenario (TP=0, TN=14, FP=0, FN=36). Mirror image of FinCoT's "up" collapse.
  - Combined verdict: species CPT (any LoRA dataset) installs a directional default but does NOT give the predator the ability to read the input. **Both paths (A) and (B) of #10 are closed. The bottleneck is in the predator's cross-attention input-responsiveness itself, not the base-model distribution.** Routes to a new diagnostic on Channel cross-attention output variance across StockNet days. (#10)
- 51/51 tests passing post-migration and post-fix.

### Known issues
- **Direction collapse on StockNet** (#8): Post-fix architecture still emits constant "up" on every StockNet day. MCC=0.000. Three plausible root causes ranked in #8 (reward-weight bias from IPO sampling, decode greediness, or genuine architectural blind spot). Open for investigation.

### Resolved
- **Ticker mode collapse** (#6, status:done): Predator Channel cross-attention was structurally input-independent because `hunter_state = role_prefix.mean()` — a fixed string per agent. Diagnosed via `scripts/diagnostics/diag_predator_attention.py` (Channel output bit-identical across scenarios, cosine 1.0000). Fixed by `hunter_state = role_q + prey_t.mean()` across all 6 code paths (predator, herbivore, sft, ipo, grpo, reward_rl). Validated end-to-end: rule-based dev/held-out gains landed at +12% over pre-fix.
- **Two metrics disagree on dev** — LLM-judge metric (0.543) saturates regardless of architecture; rule-based reward (0.370) is the reliable signal. Held-out is the apples-to-apples comparison.
- **`save_channels` overwrites `_best.pt` in place** — no rewind buffer for overfitting. Patch deferred per user.
- **Held-out eval was lost to harness output truncation once** — durable-log policy now in place at `logs/<name>.log`; documented in `todo/00-INDEX.md`.

---

## [0.2.0] — 2026-04-26 — Trough-as-Transformer migration

The migration that collapses `SubstratePool` (SQLite + diet-tag SQL) and `Channel` (per-consumer cross-attention) into a single `TroughAttention` abstraction with population dynamics.

### Added
- `trophic/trough_attention.py` — `TroughAttention` class with multi-head Q/K/V, learned null gate, MLP output projection, per-slot attention buffer, decomposer-bias hook.
- `trophic/ecology/slot_lifecycle.py` — pure functions for `mark_underperforming`, `kill_dead_slots`, `niche_aware_spawn`.
- `trophic/decomposer.py` — `BiasDecomposer` module mapping apex judgment + slot lineage to attention bias vector. Zero-init.
- `trophic/training/tau_schedule.py` — `cosine_tau` and `linear_tau` for τ annealing.
- `tests/test_trough_attention.py`, `test_slot_lifecycle.py`, `test_multi_head_niches.py`, `test_decomposer_bias.py`, `test_skip_connections.py` — 32 new tests across the six new mechanisms.
- `Predator.skip_weight` — learnable sigmoid-bounded scalar for residual path from producer trough into predator.

### Changed
- `runner.py` — instantiates one `TroughAttention` per (tier, agent_kind) lazily; passes τ to all `attend()` calls; invokes `BiasDecomposer` after apex judgment; logs per-tick ecology snapshot.
- `substrate.py` — fully rewritten as in-memory shim (SQL/rotted-flag gone). Public API (`add_broadcast`, `claim`, `query_pool`, etc.) preserved for backward compat.

### Removed
- SQLite-backed substrate pool.
- Diet-tag SQL filter logic in `herbivore.py` / `predator.py` `by_kind` routing.
- `rotted` flag and TTL bookkeeping (subsumed by ecological death via `step_lifecycle`).

### Mission record
Migration was executed via the `swarm` skill across 6 phase-coordinated missions. Full record in `tasks/`:
- phase1-A:01 trough-attention (foundation)
- phase2-A:04 slot-reallocation
- phase2-B:02 attention-decay
- phase2-C:03 multi-head-niches
- phase2-D:05 decomposer-bias
- phase3-A:06 skip-connections + τ schedule

All six DONE; recovery-finalizers handled two mid-mission session crashes without data loss.

---

## [0.1.0] — pre-2026-04-26 — Pre-migration baseline

The pre-migration architecture as it existed before the trough-as-transformer work began. Captured here for historical context; not a release.

### Architecture
- SQLite `SubstratePool` with diet-tag SQL filtering and rotted-flag bookkeeping.
- Per-consumer `Channel` modules (each herbivore/predator owns its own Q/K/V projections).
- Token-bypass cross-model channel (`CrossModelChannel`) between Qwen3-4B and Qwen2.5-Math-1.5B.
- LoRA on the InterrogatorHerbivore's synthesizer phase.

### Best results
- SFT seed 1 dev: 0.326
- IPO v3 seed 1 dev: 0.388
- IPO v3 seed 1 held-out: 0.289 (computed retroactively post-migration)
- LoRA-active interrogator: 0.937 transcription reward; predator dropped to 0.105 (motivated the migration)

### Reference checkpoints (preserved)
- `checkpoints/sft_seed1_best.pt`
- `checkpoints/ipo_seed1_best.pt`
- `checkpoints/interrogator_lora/seed1/`
- `checkpoints/cross_model_seed1_best.pt`

---

## Conventions

- **Versions** are `[X.Y.Z]` even though we're not releasing yet. Bump `Y` for major arch changes (like the migration); `Z` for incremental improvements.
- **Dates** are the date the change landed locally, not the date it merged. Source of truth for "when" is the git log.
- **References** to issues use `#N`. References to commits use the short SHA. References to files use backticks: `path/to/file.py`.
- **Sections** in each version: Added, Changed, Removed, Fixed, Results, Known issues. Keep order consistent.
- **Don't list every commit.** Group related commits into a single bullet describing the user-visible or architecturally-meaningful change.
- **Issue cross-references are bidirectional.** When you list a result here, comment on the issue that the result was logged.
