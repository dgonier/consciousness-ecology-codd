# Mission 04: social-signal-producer

**Handle**: `phase2-D`
**Phase**: 2 (parallel after `phase1-A:01`)
**Mission file**: `phase2-D-04-social-signal-producer.md`
**Dependencies**: `phase1-A:01:DONE`
**Blocks**: `phase3-A:05`

---

## Before You Start

```bash
cd /home/dgonier/ecology_experiment/trophic && \
  grep -nE '@(all|phase2-D|phase2)' tasks_envstream/scratchpad.md
```

Confirm `phase1-A:01:DONE`. Then atomically flip:
```
phase2-D:04:PENDING → phase2-D:04:RUNNING
```

---

## Goal

Two-part change to producers:

1. **Refactor existing producers** — `TickDelta`, `Disclosure`, `Anomaly` (in `producer.py`) and `QuantitativeProducer` (in `quant_producer.py`) — to declare wavelength absorption via a `WAVELENGTHS` class attribute instead of the existing global `ATTRACTION` function dict. The dict goes away.
2. **Add `SocialSignal` producer** — a new producer kind with `WAVELENGTHS = {"tweets"}`. Same forward-pool-broadcast pattern as the others; just reads tweets-shaped payloads.

After this mission, producers are decoupled from the adapter format — adding a new adapter is purely a wavelength registry update, and existing producers automatically pick up matching deposits.

---

## Files to Create / Modify

### Create
- `trophic/agents/social_signal.py` — new `SocialSignal` producer subtype
- `tests/test_producer_wavelengths.py` — at least 4 tests

### Modify
- `trophic/agents/producer.py` — add `WAVELENGTHS` class attr to `Producer` base + per-subtype overrides; remove `ATTRACTION` dict; rewrite `attracts()` to use `WAVELENGTHS`
- `trophic/agents/quant_producer.py` — same pattern: add `WAVELENGTHS = {"quote_series"}`
- `trophic/agents/__init__.py` — export `SocialSignal` if it's exported there

### Don't touch
- `trophic/environment_stream.py` (phase2-B)
- `trophic/adapters/` (phase2-C)
- `trophic/runner.py` (phase3-A integration)

---

## Implementation Steps

### Step 1: Refactor `producer.py`

The current pattern (look at `trophic/agents/producer.py` around lines 33-49):

```python
def attracts_tickdelta(inp): return inp.source in {"ohlcv", "trades", "book"}
def attracts_disclosure(inp): return inp.source in {"filing", "press"}
def attracts_anomaly(inp): return inp.source in {"ohlcv", "options", "halt"}
ATTRACTION = {
    "tickdelta": attracts_tickdelta,
    "disclosure": attracts_disclosure,
    "anomaly": attracts_anomaly,
}
# ...
class Producer(BaseAgent):
    def attracts(self, inp): return ATTRACTION[self.kind](inp)
```

Replace with class-level `WAVELENGTHS`:

```python
from typing import ClassVar

@dataclass
class Producer(BaseAgent):
    role: str = "producer"
    pool: str = "mean"
    WAVELENGTHS: ClassVar[set[str]] = set()  # subclasses override
    KIND: ClassVar[str] = "producer"          # subclasses override

    @classmethod
    def make(cls, kind: str, pool: str = "mean") -> "Producer":
        # backward-compat: kind argument selects a subclass
        kind_to_cls = {sub.KIND: sub for sub in cls.__subclasses__()}
        if kind in kind_to_cls:
            target = kind_to_cls[kind]
            return target(id=new_agent_id("producer", kind), kind=kind, pool=pool)
        # fallback for legacy "tickdelta"/"disclosure"/"anomaly" kind strings
        # if subclasses use those KINDs, the above branch hits first.
        raise ValueError(f"unknown producer kind: {kind}")

    def attracts(self, inp: RawInput) -> bool:
        return inp.source in self.WAVELENGTHS


@dataclass
class TickDelta(Producer):
    WAVELENGTHS: ClassVar[set[str]] = {"ohlcv", "trades", "book"}
    KIND: ClassVar[str] = "tickdelta"


@dataclass
class Disclosure(Producer):
    WAVELENGTHS: ClassVar[set[str]] = {"filing", "press"}
    KIND: ClassVar[str] = "disclosure"


@dataclass
class Anomaly(Producer):
    WAVELENGTHS: ClassVar[set[str]] = {"ohlcv", "options", "halt"}
    KIND: ClassVar[str] = "anomaly"
```

**Backward compat**: keep `Producer.make("tickdelta")` working for existing call sites (`scenarios.py`, `runner.py`, etc. all use this string-keyed factory). Subclassing with a `KIND` class attr is the cleanest way; the `make` method dispatches.

If subclassing causes too many ripple changes (e.g., if every place that uses `Producer` now needs to know about `TickDelta`), an alternative is to keep a single `Producer` class and just have a `WAVELENGTHS_BY_KIND` dict at module level. **Either is acceptable** — pick the smaller diff. Document the choice in MESSAGES.

### Step 2: Refactor `quant_producer.py`

Same pattern. The existing `QuantitativeProducer` should get `WAVELENGTHS: ClassVar[set[str]] = {"quote_series"}` and `KIND: ClassVar[str] = "quote_series"` (or whatever its kind string already is — preserve it).

### Step 3: New `trophic/agents/social_signal.py`

```python
"""SocialSignal — producer for tweet-shaped social/text streams.

Wavelength: {"tweets"}. Future-proofed for {"reddit", "news_headlines"}
once those wavelengths land in the registry.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import ClassVar

import torch

from ..model_host import ModelHost
from ..types import Broadcast, RawInput
from .base import new_agent_id
from .producer import Producer


@dataclass
class SocialSignal(Producer):
    role: str = "producer"
    pool: str = "mean"
    WAVELENGTHS: ClassVar[set[str]] = {"tweets"}
    KIND: ClassVar[str] = "social_signal"

    @classmethod
    def make(cls, pool: str = "mean") -> "SocialSignal":
        return cls(id=new_agent_id("producer", cls.KIND), kind=cls.KIND, pool=pool)

    async def produce(
        self,
        inp: RawInput,
        tick: int,
        host: ModelHost | None = None,
    ) -> Broadcast | None:
        if not self.attracts(inp):
            return None
        host = host or ModelHost.get()

        # Render tweets to a short text string for Qwen. The tweet adapter
        # has already aggregated a day's tweets into payload["body"]; we
        # pass it with a minimal preamble.
        body = inp.payload.get("body", "") or ""
        ticker = inp.payload.get("ticker", "")
        text = f"SOCIAL_SIGNAL ticker={ticker}; {body}"
        pooled = host.text_to_hidden(text, pool=self.pool)
        emb = pooled.detach().cpu().tolist()

        return Broadcast(
            id=str(uuid.uuid4()),
            tier="substrate",
            agent_id=self.id,
            agent_kind=self.kind,
            parent_input_ids=[inp.id],
            diet_tags=["is_social_text", "from_social_signal"],
            payload=inp.payload,
            decoded_text=text,
            channel_embedding=emb,
            created_tick=tick,
        )
```

### Step 4: Tests

```python
# tests/test_producer_wavelengths.py
from trophic.agents.producer import Producer, TickDelta, Disclosure, Anomaly
from trophic.agents.quant_producer import QuantitativeProducer
from trophic.agents.social_signal import SocialSignal
from trophic.types import RawInput


def _ri(source: str) -> RawInput:
    return RawInput(id="r0", source=source, payload={})


def test_tickdelta_wavelengths():
    assert TickDelta.WAVELENGTHS == {"ohlcv", "trades", "book"}
    p = Producer.make("tickdelta")
    assert p.attracts(_ri("ohlcv"))
    assert p.attracts(_ri("trades"))
    assert not p.attracts(_ri("filing"))


def test_disclosure_wavelengths():
    p = Producer.make("disclosure")
    assert p.attracts(_ri("filing"))
    assert p.attracts(_ri("press"))
    assert not p.attracts(_ri("ohlcv"))


def test_anomaly_overlaps_ohlcv_with_tickdelta():
    """Many-to-many: ohlcv feeds both tickdelta AND anomaly."""
    td = Producer.make("tickdelta")
    an = Producer.make("anomaly")
    inp = _ri("ohlcv")
    assert td.attracts(inp)
    assert an.attracts(inp)


def test_quantitative_wavelengths():
    qp = QuantitativeProducer.make()  # however the existing API works
    assert "quote_series" in qp.WAVELENGTHS
    assert qp.attracts(_ri("quote_series"))


def test_social_signal_wavelengths_only_tweets():
    ss = SocialSignal.make()
    assert ss.WAVELENGTHS == {"tweets"}
    assert ss.attracts(_ri("tweets"))
    assert not ss.attracts(_ri("ohlcv"))
    assert not ss.attracts(_ri("filing"))


def test_no_global_attraction_dict():
    """ATTRACTION dict from the old code should be gone."""
    import trophic.agents.producer as prod
    assert not hasattr(prod, "ATTRACTION"), "ATTRACTION dict should be removed"
```

### Step 5: Update `trophic/agents/__init__.py` if it exports producer kinds

Look for whether `SocialSignal` should be exported from the package root. If existing producers are exported, add `SocialSignal` alongside them.

---

## Acceptance Criteria

- [ ] All four existing producer subtypes (TickDelta, Disclosure, Anomaly, QuantitativeProducer) declare `WAVELENGTHS` class attribute
- [ ] `ATTRACTION` global dict is removed from `producer.py`
- [ ] `Producer.attracts(inp)` uses `inp.source in self.WAVELENGTHS`
- [ ] `Producer.make(kind)` factory still works for all legacy kind strings (`"tickdelta"`, `"disclosure"`, `"anomaly"`, `"quote_series"` or whatever the existing one was)
- [ ] `SocialSignal` producer exists with `WAVELENGTHS = {"tweets"}`
- [ ] `SocialSignal.produce(...)` runs end-to-end on a fake `RawInput(source="tweets", ...)`
- [ ] At least 4 new tests in `tests/test_producer_wavelengths.py`
- [ ] All existing tests still pass

---

## Testing Conditions (exit verification)

1. **New wavelength tests pass**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     .venv/bin/python -m pytest tests/test_producer_wavelengths.py -v 2>&1 | tail -15
   ```
   **Expected**: ≥4 passed, 0 failed.

2. **Existing producer/runner tests still pass** (you may have broken something)
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     .venv/bin/python -m pytest tests/ --tb=short -q 2>&1 | tail -10
   ```
   **Expected**: 65+ passed, 0 failed.

3. **ATTRACTION dict removed**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     grep -n "ATTRACTION" trophic/agents/producer.py
   ```
   **Expected**: no matches (dict gone).

4. **Mock SFT smoke (existing producers)** — confirms the refactor didn't break the live pipeline
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     TROPHIC_MOCK_MODELS=1 TROPHIC_MOCK_PREDATOR=1 TROPHIC_STEPS=10 \
     timeout 300 .venv/bin/python scripts/train_sft.py 2>&1 | tail -10
   ```
   **Expected**: exit 0.

5. **SocialSignal runs end-to-end (mocked)** — small smoke
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && .venv/bin/python -c "
   import asyncio
   from trophic.agents.social_signal import SocialSignal
   from trophic.types import RawInput
   from trophic.model_host import ModelHost
   import os
   os.environ['TROPHIC_MOCK_MODELS'] = '1'
   ss = SocialSignal.make()
   inp = RawInput(id='t0', source='tweets', payload={'ticker': 'AAPL', 'body': '\$ aapl bullish breakout'})
   br = asyncio.run(ss.produce(inp, tick=0))
   print(f'broadcast tier={br.tier} kind={br.agent_kind} emb_len={len(br.channel_embedding)}')
   "
   ```
   **Expected**: prints `broadcast tier=substrate kind=social_signal emb_len=<some int>`.

---

## Coordination

- You're modifying `producer.py` and `quant_producer.py`. phase2-B and phase2-C do NOT touch these files. Conflict-free.
- Many call sites use `Producer.make("tickdelta")` etc. **Verify each still works.** Test (4) is your safety net.
- If you choose the alternative (single `Producer` class + `WAVELENGTHS_BY_KIND` dict instead of subclasses), document the choice in MESSAGES so phase3-A knows.

---

## Out of Scope

- New producer kinds beyond `SocialSignal` (Reddit, news headlines, etc.) — defer to follow-up issues
- Adapter implementations — phase2-C
- EnvironmentStream wiring — phase2-B and phase3-A
- Reputation / lifecycle on producers themselves — orthogonal concern

---

## When Done

1. Re-run inbox grep.
2. Update STATUS: `phase2-D:04:RUNNING` → `phase2-D:04:DONE`.
3. Append MESSAGES:
   ```
   - [<YYYY-MM-DD HH:MM>] phase2-D > @phase3-A: producer wavelengths landed. Existing producers (TickDelta, Disclosure, Anomaly, QuantitativeProducer) now use WAVELENGTHS class attr; ATTRACTION dict removed. New SocialSignal producer at trophic/agents/social_signal.py with WAVELENGTHS={"tweets"}. test_producer_wavelengths.py: <PASS>/<TOTAL>. Full suite: <PASS>/<TOTAL>. Refactor approach used: <subclass-per-kind / single-class-with-dict>.
   ```
