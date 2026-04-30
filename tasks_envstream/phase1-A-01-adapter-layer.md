# Mission 01: adapter-layer

**Handle**: `phase1-A`
**Phase**: 1 (foundation, sequential)
**Mission file**: `phase1-A-01-adapter-layer.md`
**Dependencies**: none
**Blocks**: `phase2-B:02`, `phase2-C:03`, `phase2-D:04`, `phase3-A:05`

---

## Before You Start

```bash
cd /home/dgonier/ecology_experiment/trophic && \
  grep -nE '@(all|phase1-A|phase1)' tasks_envstream/scratchpad.md
```

Atomically flip your STATUS line in `tasks_envstream/scratchpad.md`:
```
phase1-A:01:PENDING → phase1-A:01:RUNNING
```

---

## Goal

Stand up the **Adapter** abstraction and the **wavelength registry** that the rest of the project depends on. After this mission lands, the project has a clean separation between *external format conversion* (adapters) and *signal interpretation* (producers). No model changes; pure plumbing.

---

## Files to Create / Modify

### Create
- `trophic/adapters/__init__.py` — package init with public exports
- `trophic/adapters/base.py` — `Adapter` ABC + `SOURCE_TAGS_VOCAB` registry
- `tests/test_adapters.py` — at least 4 tests for the ABC contract

### Modify
- `trophic/types.py` — keep `RawInput.source: str` for compat, but add `SOURCE_TAGS_VOCAB: Final[set[str]]` constant; document that all new code MUST use a tag from the vocabulary
- `trophic/training/scenarios.py` — find any inline format-conversion logic (currently mixed into the scenario builders) and refactor it to use placeholder adapter calls. Identify the boundaries; do NOT migrate it yet — that's phase2-C's job. Just leave clear TODOs (`# TODO(envstream phase2-C): replace with OhlcvNormalizedAdapter.adapt(...)`) where adapters will plug in.
- `trophic/training/stocknet_loader.py` — same pattern: identify the format-conversion sections (price file parsing, tweet JSON parsing) and mark them with TODOs pointing to phase2-C adapters. Do NOT delete or rewrite the loader; phase3-A will do the final integration.

---

## Implementation Steps

### Step 1: Create `trophic/adapters/__init__.py`

```python
"""Adapter layer: format converters that turn external data into RawInput streams.

Adapters are single-responsibility format converters. Each adapter declares
which SOURCE_TAGS it emits (its "wavelength"). Producers subscribe to
wavelength subsets via their WAVELENGTHS class attr (set in phase2-D).

Many-to-many is the contract: one adapter's output (e.g. source="ohlcv")
can feed multiple producers (TickDelta, Anomaly).
"""
from .base import Adapter, SOURCE_TAGS_VOCAB

__all__ = ["Adapter", "SOURCE_TAGS_VOCAB"]
```

### Step 2: Create `trophic/adapters/base.py`

```python
"""Adapter ABC + the wavelength registry."""
from __future__ import annotations

from typing import Any, ClassVar, Final, Iterable

from ..types import RawInput


# All source tags that any adapter is allowed to emit. Adding a new wavelength
# means adding it here AND defining an Adapter subclass that emits it.
SOURCE_TAGS_VOCAB: Final[set[str]] = {
    "ohlcv",         # OHLCV bars (any granularity)
    "trades",        # tick-level trade prints
    "book",          # depth-of-book snapshots
    "filing",        # SEC filings (8-K, 10-Q, S-1, 13D)
    "press",         # press releases / news headlines
    "options",       # options activity
    "halt",          # trading halts
    "quote_series",  # multi-bar quote time series for forecasting
    "tweets",        # tokenized social media text (NEW for ENVSTREAM)
}


class Adapter:
    """Convert raw external data into RawInput stream. No model use.

    Subclass contract:
      - declare SOURCE_TAGS as a class attribute (subset of SOURCE_TAGS_VOCAB)
      - implement adapt(raw) -> Iterable[RawInput]
      - never call model code; adapters are pure data transforms

    Many-to-many: one adapter can be consumed by multiple producers. One
    producer can subscribe to multiple wavelengths.
    """
    SOURCE_TAGS: ClassVar[set[str]] = set()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.SOURCE_TAGS:
            return  # base ABC, abstract intermediates allowed
        unknown = cls.SOURCE_TAGS - SOURCE_TAGS_VOCAB
        if unknown:
            raise ValueError(
                f"Adapter {cls.__name__} declares unknown SOURCE_TAGS: {unknown}. "
                f"Add them to SOURCE_TAGS_VOCAB in trophic/adapters/base.py first."
            )

    def adapt(self, raw: Any) -> Iterable[RawInput]:
        raise NotImplementedError(
            f"{type(self).__name__} must implement adapt(raw) -> Iterable[RawInput]"
        )
```

The `__init_subclass__` hook gives us **wavelength-registry validation at class-definition time**. Any subclass declaring an unknown source tag fails import. This is the audit hook the README mentioned.

### Step 3: Add `SOURCE_TAGS_VOCAB` to `trophic/types.py`

Re-export the constant from `trophic.types` so older code can import it from there too (some places may grep `from trophic.types import` and we keep it tidy):

```python
# At the top of trophic/types.py, after the existing imports:
from typing import Final

# Re-export the wavelength registry from adapters; the canonical home is
# trophic/adapters/base.py but exposing it here keeps `from trophic.types
# import SOURCE_TAGS_VOCAB` working for callers.
# NOTE: cannot import from trophic.adapters at module load time (circular
# import). Inline the literal here and document that they must stay in sync.
SOURCE_TAGS_VOCAB: Final[set[str]] = {
    "ohlcv", "trades", "book", "filing", "press",
    "options", "halt", "quote_series", "tweets",
}
```

If you can solve the circular import cleanly (adapters/base.py imports the constant from types.py rather than defining it), do that — single source of truth is better. Otherwise inline + document.

### Step 4: Mark TODOs in legacy format-conversion code

In `trophic/training/scenarios.py`, find functions that translate between payload dicts and RawInput-like structures. Search for `RawInput(`, `_ri(` calls. Add a comment block where a future adapter would land:

```python
# TODO(envstream phase2-C): the format conversion below should be migrated
# into trophic/adapters/{kind}.py:Adapter so that scenarios.py only consumes
# RawInput streams. For now, leave this in place — phase3-A will do the
# integration once adapters exist.
```

In `trophic/training/stocknet_loader.py`, mark the price-file parser and the tweet-JSON parser with the same comment. Phase2-C will replace these with `OhlcvNormalizedAdapter.adapt(...)` and `TokenizedTweetAdapter.adapt(...)`.

**Do not delete or rewrite the legacy code.** That's phase3-A's integration step.

### Step 5: Tests in `tests/test_adapters.py`

```python
"""Tests for the Adapter ABC contract."""
from __future__ import annotations

import pytest

from trophic.adapters import Adapter, SOURCE_TAGS_VOCAB
from trophic.types import RawInput


class _GoodAdapter(Adapter):
    SOURCE_TAGS = {"ohlcv"}
    def adapt(self, raw):
        yield RawInput(id="x", source="ohlcv", payload=raw)


def test_subclass_with_known_tag_loads():
    a = _GoodAdapter()
    out = list(a.adapt({"x": 1}))
    assert len(out) == 1
    assert out[0].source == "ohlcv"


def test_subclass_with_unknown_tag_raises_at_definition():
    with pytest.raises(ValueError, match="unknown SOURCE_TAGS"):
        class _Bad(Adapter):
            SOURCE_TAGS = {"this_does_not_exist"}


def test_subclass_with_multiple_known_tags_loads():
    class _Multi(Adapter):
        SOURCE_TAGS = {"ohlcv", "trades"}
        def adapt(self, raw):
            yield RawInput(id="x", source="ohlcv", payload=raw)
    assert _Multi.SOURCE_TAGS == {"ohlcv", "trades"}


def test_adapt_must_be_implemented():
    class _Stub(Adapter):
        SOURCE_TAGS = {"ohlcv"}
    with pytest.raises(NotImplementedError):
        list(_Stub().adapt({}))


def test_source_tags_vocab_includes_tweets():
    assert "tweets" in SOURCE_TAGS_VOCAB
```

---

## Acceptance Criteria

- [ ] `trophic/adapters/base.py` exists with `Adapter` ABC + `SOURCE_TAGS_VOCAB`
- [ ] `trophic/adapters/__init__.py` re-exports both
- [ ] `trophic/types.py` has `SOURCE_TAGS_VOCAB` constant (single source of truth or documented inline)
- [ ] `__init_subclass__` validates SOURCE_TAGS at class-definition time
- [ ] Legacy format-conversion code in `scenarios.py` and `stocknet_loader.py` marked with TODO comments pointing to phase2-C
- [ ] At least 4 new tests in `tests/test_adapters.py`
- [ ] All 51 existing tests still pass

---

## Testing Conditions (exit verification)

Run each command and capture output. Do not flip to DONE until each shows the expected result.

1. **New adapter tests pass**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     .venv/bin/python -m pytest tests/test_adapters.py -v 2>&1 | tail -15
   ```
   **Expected**: ≥4 passed, 0 failed.

2. **Existing tests still green**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     .venv/bin/python -m pytest tests/ --tb=line -q 2>&1 | tail -5
   ```
   **Expected**: 55+ passed (51 existing + 4 new), 0 failed.

3. **Validation hook fires on bad subclass**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && .venv/bin/python -c "
   from trophic.adapters import Adapter
   try:
       class Bad(Adapter):
           SOURCE_TAGS = {'totally_made_up'}
       print('FAIL: validation did not fire')
   except ValueError as e:
       print(f'OK: validation fired: {e}')
   "
   ```
   **Expected**: `OK: validation fired: ...`

4. **TODO markers present**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     grep -c "TODO(envstream phase2-C)" trophic/training/scenarios.py trophic/training/stocknet_loader.py
   ```
   **Expected**: ≥1 marker per file (phase2-C uses these as integration hooks).

5. **Mock SFT smoke still completes**
   ```bash
   cd /home/dgonier/ecology_experiment/trophic && \
     TROPHIC_MOCK_MODELS=1 TROPHIC_MOCK_PREDATOR=1 TROPHIC_STEPS=10 \
     timeout 300 .venv/bin/python scripts/train_sft.py 2>&1 | tail -10
   ```
   **Expected**: exit 0, no exceptions. Confirms the TODO markers / refactor didn't break the existing pipeline.

If any test fails, do not mark DONE; post a `@all` MESSAGES entry describing the failure.

---

## Coordination

- This is the foundation. Phase 2's three agents all import `Adapter` and `SOURCE_TAGS_VOCAB` from your work.
- Don't change `RawInput`'s field types — leave `source: str` so existing code keeps working. The `SOURCE_TAGS_VOCAB` constant is advisory; the `__init_subclass__` hook is the enforcement mechanism for new code.
- If you discover an existing source tag in the codebase that's NOT in the vocab (e.g., something like `"depth_book_snapshot"`), add it to the vocab AND post `@all` so phase 2 agents know.

---

## Out of Scope

- Implementing concrete adapters (phase2-C does StockNet adapters; future projects can add Bloomberg, Polygon, etc.)
- Building EnvironmentStream (phase2-B)
- Producer wavelength refactor (phase2-D)
- Rewriting loaders / scenario builders (phase3-A integration)

---

## When Done

1. Re-run inbox grep: `grep -nE '@(all|phase1-A|phase1)' tasks_envstream/scratchpad.md`. Address anything new.
2. Edit `tasks_envstream/scratchpad.md`:
   - Flip your STATUS line: `phase1-A:01:RUNNING` → `phase1-A:01:DONE`
   - Unblock phase 2 — change these three STATUS lines:
     `phase2-B:02:BLOCKED  (needs phase1-A:01)` → `phase2-B:02:PENDING`
     `phase2-C:03:BLOCKED  (needs phase1-A:01)` → `phase2-C:03:PENDING`
     `phase2-D:04:BLOCKED  (needs phase1-A:01)` → `phase2-D:04:PENDING`
3. Append a MESSAGES entry, exact format:
   ```
   - [<YYYY-MM-DD HH:MM>] phase1-A > @phase2: Adapter ABC + wavelength registry landed. Public API: from trophic.adapters import Adapter, SOURCE_TAGS_VOCAB. Acceptance test summary: <pasted last lines from pytest>. TODO markers in scenarios.py / stocknet_loader.py at lines <X> point to phase2-C integration sites. Surprises: <none / list>. Open questions: <none / list>.
   ```
