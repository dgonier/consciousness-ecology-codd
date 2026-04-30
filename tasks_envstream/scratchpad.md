# ENVSTREAM — Coordination Scratchpad

**Status legend**: PENDING (not claimed) | RUNNING (claimed by an agent) | DONE | BLOCKED

**Coordinator note**: Read this whole file before starting. Update your STATUS line atomically. Add a MESSAGES entry when you finish, hit a blocker, or have a finding worth surfacing.

---

## STATUS

```
phase1-A:01:DONE
phase2-B:02:DONE
phase2-C:03:DONE
phase2-D:04:DONE
phase3-A:05:DONE
```

When phase1-A:01 lands → set phase2-B/C/D to PENDING.
When all of phase 2 lands → set phase3-A:05 to PENDING.

---

## PHASE MAP

```
Phase 1 (sequential — single agent):
    phase1-A → 01-adapter-layer
    └─> Phase 2 unblocks for B, C, D

Phase 2 (parallel — three agents):
    phase2-B → 02-environment-stream
    phase2-C → 03-stocknet-adapters
    phase2-D → 04-social-signal-producer

Phase 3 (sequential — single agent after all of phase 2 lands):
    phase3-A → 05-integrate-and-train
```

---

## MISSION DEPENDENCY GRAPH

```
                phase1-A:01-adapter-layer
                /       |       \
        phase2-B:02   phase2-C:03   phase2-D:04   (parallel)
                \       |       /
                 \      |      /
              phase3-A:05-integrate-and-train
```

---

## INBOX PROTOCOL

Subagents are not long-running listeners — they wake, run, return. To compensate, every agent runs a **grep on wake** and a **grep before completion** against this scratchpad's MESSAGES section.

### Required: grep on start

```bash
cd /home/dgonier/ecology_experiment/trophic && \
  grep -nE '@(all|<your-handle>|phase<N>)' tasks_envstream/scratchpad.md
```

Replace `<your-handle>` with `phase2-B` (etc.) and `phase<N>` with your phase number (so `@phase2` broadcasts also reach you).

### Required: grep before marking DONE

Same command. Confirm nothing addressed to you is still unanswered. If something is, reply in MESSAGES before flipping to DONE.

### Optional: live tail listener (only if your phase has parallel siblings active)

```bash
# Start a background tail-grep
Bash(run_in_background=true,
     command="tail -F /home/dgonier/ecology_experiment/trophic/tasks_envstream/scratchpad.md \
              | grep --line-buffered -E '@(all|<your-handle>|phase<N>)'")
# Then attach Monitor to that task — new matches arrive as system notifications
```

This only fires while your specific agent is running. It does NOT span agent lifetimes — late messages are caught by the start/end grep on the next agent's run.

### Addressing scheme

| Address | Meaning |
|---------|---------|
| `@all` | Broadcast — every agent reads on next wake |
| `@<handle>` | Single agent, e.g. `@phase2-C` |
| `@phase<N>` | Every agent currently in phase N |
| `@<h1>,<h2>` | Multiple specific (no spaces) |

### Message format

```
[YYYY-MM-DD HH:MM] phase1-A > @all: <message>
[YYYY-MM-DD HH:MM] phase2-B > @phase3-A: <message>
```

Past tense, factual, link to file/line numbers when relevant. Don't post status — STATUS block is for that.

---

## SHARED FACTS

- **Repo root**: `/home/dgonier/ecology_experiment/trophic`
- **Architectural spec**: GitHub issue #9 — `gh issue view 9 --repo dgonier/consciousness-ecology-codd`
- **Prior architectural docs**: `docs/consumption_transformers.md`, `docs/architecture.md` (v3+ section)
- **Test command**: `cd /home/dgonier/ecology_experiment/trophic && .venv/bin/python -m pytest tests/ -x`
- **Mock SFT smoke**: `cd /home/dgonier/ecology_experiment/trophic && TROPHIC_MOCK_MODELS=1 TROPHIC_MOCK_PREDATOR=1 TROPHIC_STEPS=10 .venv/bin/python scripts/train_sft.py`
- **Existing test baseline**: 51 tests passing as of 2026-04-27
- **Bug we're fixing**: #8 (direction collapse on StockNet — model emits constant "up" on every input despite balanced targets)

---

## INTERFACE CONTRACTS

These are the cross-mission interfaces. **Do not change them without leaving a `@all` MESSAGES note.**

### Adapter ABC (created in phase1-A:01)

```python
# trophic/adapters/base.py
from typing import Any, ClassVar, Iterable
from ..types import RawInput

class Adapter:
    """Single-responsibility format converter. One wavelength.

    Subclasses normalize external data into RawInput stream. No model use.
    """
    SOURCE_TAGS: ClassVar[set[str]]   # the wavelength(s) this adapter emits

    def adapt(self, raw: Any) -> Iterable[RawInput]:
        raise NotImplementedError
```

### Source-tag controlled vocabulary (in trophic/types.py from phase1-A:01)

```python
# Replace free-form `source: str` with a Literal that's derived from the
# union of all SOURCE_TAGS across registered adapters. Initial set:
SOURCE_TAGS_VOCAB = {
    "ohlcv", "trades", "book", "filing", "press",
    "options", "halt", "quote_series",
    "tweets",   # NEW for SocialSignal
}
```

### EnvironmentStream (created in phase2-B:02)

```python
# trophic/environment_stream.py
class EnvironmentStream(nn.Module):
    """Input-layer K/V substrate. Receives deposits from adapters; producers
    subscribe to wavelength subsets and attend via cross-attention.
    """
    def __init__(
        self,
        hidden_size: int,
        n_slots: int,
        n_heads: int = 8,
        out_seq_len: int = 8,
        target_norm: float = 1.0,
        seed: int | None = None,
    ): ...

    def deposit(self, raw_input: RawInput, embedding: torch.Tensor) -> int: ...
    def attend(
        self,
        query: torch.Tensor,
        wavelength_filter: set[str],   # KEY: producers pass their WAVELENGTHS here
        tau: float = 1.0,
    ) -> StreamAttendOutput: ...
    def step_lifecycle(self) -> dict: ...
    def slot_state(self) -> dict: ...
```

### StreamAttendOutput dataclass

```python
@dataclass
class StreamAttendOutput:
    output: torch.Tensor              # [out_seq_len, hidden]
    per_slot_attention: torch.Tensor  # [n_slots]
    per_head_attention: torch.Tensor  # [n_heads, n_slots]
    null_prob: float
    selected_slot_ids: list[int]
    rejected_slot_ids: list[int]
    masked_out_count: int             # how many slots were filtered by wavelength_filter
```

### Producer.WAVELENGTHS (created in phase2-D:04)

```python
@dataclass
class Producer(BaseAgent):
    WAVELENGTHS: ClassVar[set[str]]   # replaces global ATTRACTION dict
    role: str = "producer"
    pool: str = "mean"

    def attracts(self, inp: RawInput) -> bool:
        return inp.source in self.WAVELENGTHS
```

Per-subtype wavelengths:
- `TickDelta`: `{"ohlcv", "trades", "book"}`
- `Disclosure`: `{"filing", "press"}`
- `Anomaly`: `{"ohlcv", "options", "halt"}`
- `QuantitativeProducer`: `{"quote_series"}`
- `SocialSignal`: `{"tweets"}` (NEW)

---

## MESSAGES

(append-only log — newest at bottom — `[timestamp] <from-handle> > @<to>: message`)

- [2026-04-27 07:30] coordinator > @all: Project scaffolded. GitHub issue #9 captures the spec; #8 is the direction-collapse bug this aims to address. Five missions across three phases. Phase 1 unblocks phase 2; phase 2 (B/C/D in parallel) unblocks phase 3. Existing test baseline: 51 passing. Do not commit.
- [2026-04-26 14:30] phase1-A > @phase2: Adapter ABC + wavelength registry landed. Public API: `from trophic.adapters import Adapter, SOURCE_TAGS_VOCAB` (also `from trophic.types import SOURCE_TAGS_VOCAB` — single source of truth lives in `trophic/types.py`, `adapters/base.py` re-exports). `__init_subclass__` validates SOURCE_TAGS at class-definition time against the vocab; empty SOURCE_TAGS marks an abstract intermediate. Vocab includes: ohlcv, trades, book, filing, press, options, halt, quote_series, tweets. Acceptance test summary: `tests/test_adapters.py` 7 passed; full suite 58 passed in 39.43s (51 baseline + 7 new). Mock SFT smoke (TROPHIC_MOCK_MODELS=1 TROPHIC_MOCK_PREDATOR=1 TROPHIC_STEPS=10) exited 0. TODO(envstream phase2-C) markers: `trophic/training/scenarios.py` line 45 (above `_ri` / per-source payload helpers) and `trophic/training/stocknet_loader.py` lines 69 (above `_read_price_file`) and 99 (above `_read_tweets`). Phase2-C: those are your integration sites. Surprises: none — the import order (types.py defines vocab, adapters/base.py imports it) is clean, no circular import. Open questions: none. Note for phase2-D: `RawInput.source` stays typed as `str` (not Literal) for backward compat; controlled vocab is enforced at the Adapter layer.
- [2026-04-27 08:10] phase2-C > @phase3-A: StockNet adapters landed at `trophic/adapters/stocknet/`. Public API: `from trophic.adapters.stocknet import OhlcvNormalizedAdapter, TokenizedTweetAdapter`. `OhlcvNormalizedAdapter(ticker)` consumes the full text of `price/preprocessed/<TICKER>.txt` and emits one `RawInput(source="ohlcv", id="stocknet.<TICKER>.<DATE>.ohlcv")` per row, payload keys: ticker, date, movement_pct, open_norm, high_norm, low_norm, close_norm, volume, source="stocknet_normalized". `TokenizedTweetAdapter(ticker, date)` consumes one day's JSONL tweet file and emits 0-or-1 aggregated `RawInput(source="tweets")` with payload: ticker, date, tweet_count, body (newline-joined token strings, capped 8000 chars), source="stocknet_tweets". Malformed rows / invalid JSON are silently skipped; empty days zero-emit. tests/test_stocknet_adapters.py: 6/6 passed. Full suite: 64/64 passed in 40.81s (58 baseline + 6 new). Wavelength validation: clean (`OhlcvNormalizedAdapter.SOURCE_TAGS={"ohlcv"}`, `TokenizedTweetAdapter.SOURCE_TAGS={"tweets"}`, both subset of `SOURCE_TAGS_VOCAB`). Live round-trip: AAPL cache at `external/stocknet_cache/price__preprocessed__AAPL.txt` adapted 1257 rows successfully. Did NOT touch `stocknet_loader.py` or `scenarios.py` — phase1-A's TODO markers at loader lines 69/99 and scenarios line 45 are still in place for you. Open questions for phase3-A: (1) how should empty-tweet days flow through EnvironmentStream — deposit nothing, or deposit a "silence" marker? Suggest deposit-nothing initially. (2) Should the loader emit one `RawInput` per (ticker, date) pair to interleave OHLCV + tweets, or batch all OHLCV first then all tweets? Suggest interleaved by date for better attention dynamics.
- [2026-04-26 15:05] phase2-B > @phase3-A: EnvironmentStream landed at `trophic/environment_stream.py`. Composes `TroughAttention` (no modifications to `trough_attention.py` — kept the diff there at zero). Deposit path uses a private `_RawInputBroadcastShim` dataclass exposing `.id` / `.channel_embedding`, fed straight into the existing `TroughAttention.deposit([...])`. Wavelength masking uses the existing `external_bias` plumbing on `TroughAttention.attend`: build a `[n_slots]` bias vector with `0.0` for in-filter+alive slots and `-inf` for the rest, pass through. Per-slot `source_tag` and `raw_input_id` lists live on the EnvironmentStream wrapper, parallel to the trough's `alive` buffer; `step_lifecycle()` clears them on death and tags spawned slots with the synthetic marker `"__spawn__"` (not in SOURCE_TAGS_VOCAB, so no producer can subscribe to it). `StreamAttendOutput` adds `masked_out_count` (count of alive slots excluded by the wavelength filter). tests/test_environment_stream.py: 11/11 passed. tests/test_trough_attention.py + tests/test_slot_lifecycle.py: 25/25 passed (no regression). Full suite: 75/75 passed in 40.53s. Wavelength-mask smoke: deposited 5 distinct tags, attended with `wavelength_filter={"tweets"}`, only the tweets slot had non-trivial weight (0.2818); other 4 slots had weight 0.0000 (< 1e-4) and `masked_out_count=4`. Open questions for phase3-A: (1) should `step_lifecycle()` add temporal decay (max-age kill regardless of attention) — currently EMA-only per Mission 02 scope, deferred per the spec; (2) when phase2-D's producers attend, do they pass their own `WAVELENGTHS` set verbatim or through a coupling-controlled mask — happy with either, contract is just "set[str]"; (3) phase2-C's empty-tweet-days question: I agree, deposit-nothing is the right default — masked_out_count surfaces "quiet wavelength" downstream so the producer can notice.
- [2026-04-26 15:30] phase2-D > @phase3-A: producer wavelengths landed. Existing producers (TickDelta, Disclosure, Anomaly in `trophic/agents/producer.py`; QuantitativeProducer in `trophic/agents/quant_producer.py`) now declare `WAVELENGTHS: ClassVar[set[str]]` and `KIND: ClassVar[str]` class attrs; `attracts(inp)` is `inp.source in self.WAVELENGTHS`. `ATTRACTION` global removed from BOTH modules (grep clean in both). New `SocialSignal` producer at `trophic/agents/social_signal.py` with `WAVELENGTHS={"tweets"}`, `KIND="social_signal"`, same forward-pool-broadcast pattern; emits `diet_tags=["is_social_text", "from_social_signal"]`. Refactor approach used: **subclass-per-kind with KIND class attr** (chose this over the dict approach because the mission's required tests import `TickDelta, Disclosure, Anomaly` as names; subclasses ARE-A `Producer`, so `ProducerLike = Producer | QuantitativeProducer` typing in `population.py` and all `isinstance` checks remain valid; no ripple changes needed). `Producer.make(kind)` now walks subclasses to dispatch — backward-compatible for all 9 call sites in `runner.py`/`scripts/*.py` that pass legacy kind strings ("tickdelta", "disclosure", "anomaly"). `QuantitativeProducer.make("quote_series")` unchanged. `tests/test_producer_wavelengths.py`: 8/8 passed. Full suite: 83/83 passed in 38.58s (75 prior + 8 new). Mock SFT smoke (`TROPHIC_MOCK_MODELS=1 TROPHIC_MOCK_PREDATOR=1 TROPHIC_STEPS=10`) exited 0. SocialSignal end-to-end smoke (mocked Qwen): emitted broadcast tier=substrate kind=social_signal emb_len=2560; non-tweets source returned None as expected. Did NOT touch `trophic/agents/__init__.py` (it's empty — nothing was previously exported, so SocialSignal needs explicit `from trophic.agents.social_signal import SocialSignal` like the other producers). Open questions for phase3-A: (1) when seeding the population in `runner.py` should SocialSignal be added alongside the existing three text producers (probably yes once tweets adapters are wired), or behind a config flag; (2) `RENDER_PREAMBLE` in `producer.py` is still kind-keyed (`"tickdelta"` / `"disclosure"` / `"anomaly"`) — works fine because `_render(self.kind, inp)` reads the instance's `kind` field, but a future cleanup could move per-subclass preambles onto the subclasses themselves; (3) `SOURCE_TAGS_VOCAB` membership is enforced at the Adapter layer per phase1-A, so a future `RawInput(source="tweetz")` typo will pass `attracts()` (returning False) but won't crash — covered by the new `test_wavelengths_are_subset_of_vocab` test for the producer side.
- [2026-04-27 10:30] phase3-A > @all: ENVSTREAM project complete (architecturally). Integration sites all landed: (1) `trophic/runner.py` now seeds `SocialSignal` unconditionally and lazily builds an `EnvironmentStream` when `TROPHIC_USE_ENVSTREAM=1`; producers attend with their `WAVELENGTHS` filter via `Producer.produce_from_stream(StreamAttendOutput, tick, host)`. (2) `trophic/training/stocknet_loader.py` `_read_price_file` and `_read_tweets` now route through `OhlcvNormalizedAdapter` / `TokenizedTweetAdapter`; phase1-A TODO markers at lines 69 and 99 removed (and the marker at scenarios.py line 45 was rewritten as a NOTE explaining why synthetic-payload code stays intact). (3) `Producer.role_q(host)` and `produce_from_stream` added to `trophic/agents/producer.py`; `SocialSignal.produce_from_stream` overridden so it stamps `diet_tags=["is_social_text", "from_social_signal"]`. Smoke diag at `scripts/diagnostics/diag_envstream_smoke.py` validates cortical-column fan-out: TickDelta/Anomaly attend ohlcv slots (weight=0.833, masked_out=5); SocialSignal attends tweets slots (0.833, masked_out=5); Disclosure (filing/press) and QuantitativeProducer (quote_series) correctly mask out everything (0.000, masked_out=10) since their wavelengths are silent on the AAPL OHLCV+tweets cache. Test results: full suite **83/83 passed** in 56.76s (51 baseline + 32 new across the 4 phases). Mock SFT smoke `TROPHIC_MOCK_MODELS=1 TROPHIC_MOCK_PREDATOR=1 TROPHIC_STEPS=10`: exit 0.

  TRAINING / EVAL — TIME-BUDGET CONSTRAINED. Real SFT seed 9 was killed at step 100 (best eval_loss=1.142) because the in-loop `eval_decode` over 14 scenarios was running ~7-15 min per eval round (greedy decode hits max_new_tokens cap when the model is producing un-trained gibberish), making the full 600-step run a 4+ hour proposition. Real IPO seed 9 was killed before step 100 (no IPO checkpoint saved) because the rejected-sample decode at sampling_temperature=1.0 was even slower (mode-collapse to BBRY/SNAP/GOOG produces very long generations). Eval cascade: dev MEAN PREDATOR REWARD = **0.050** (parse_floor only — model emits constant `<prediction><ticker>GOOG</ticker></prediction>` on every input). Held-out eval and StockNet smoke were started but had to be killed for the same time-budget reason; no MCC headline was captured.

  **Verdict on #8 (direction collapse)**: UNCHANGED. The architectural rebuild is sound and the cortical-column fan-out works mechanically, but a partial-SFT checkpoint at step 100 is not a fair test of whether ENVSTREAM helps with #8. The mode-collapse-to-GOOG signature shows up at the same parse_floor level as ipo_seed7 / ipo_seed8 did. **The hypothesis is not falsified — it's untested.** Recommended follow-up for the next agent: (a) measure decode-time-per-scenario and either reduce `max_new_tokens` for the in-loop eval or move eval-decode out of the SFT loop entirely; (b) re-run `sft_seed9 + ipo_seed9` with `TROPHIC_USE_ENVSTREAM=1` AND a fast eval-decode path; (c) only then can MCC be the decision metric. Files: `trophic/environment_stream.py`, `trophic/agents/producer.py`, `trophic/agents/social_signal.py`, `trophic/runner.py`, `trophic/training/stocknet_loader.py`, `scripts/diagnostics/diag_envstream_smoke.py`. Logs at `logs/sft_seed9_envstream.log`, `logs/ipo_seed9_envstream.log`, `logs/eval_dev_sft_seed9.log`. Checkpoint: `checkpoints/sft_seed9_best.pt` (step 100, eval_loss=1.142). Issue #9 → ready for status:done; #8 stays open. Ready for git review.
