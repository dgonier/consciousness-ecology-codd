"""Tests for the Adapter ABC contract (phase1-A:01-adapter-layer)."""
from __future__ import annotations

import pytest

from trophic.adapters import Adapter, SOURCE_TAGS_VOCAB
from trophic.types import SOURCE_TAGS_VOCAB as TYPES_VOCAB
from trophic.types import RawInput


class _GoodAdapter(Adapter):
    SOURCE_TAGS = {"ohlcv"}

    def adapt(self, raw):
        yield RawInput(id="x", source="ohlcv", payload=raw)


def test_subclass_with_known_tag_loads_and_emits_rawinput():
    a = _GoodAdapter()
    out = list(a.adapt({"x": 1}))
    assert len(out) == 1
    assert isinstance(out[0], RawInput)
    assert out[0].source == "ohlcv"
    assert out[0].payload == {"x": 1}


def test_subclass_with_unknown_tag_raises_at_definition():
    with pytest.raises(ValueError, match="unknown SOURCE_TAGS"):
        class _Bad(Adapter):  # noqa: F841
            SOURCE_TAGS = {"this_does_not_exist"}


def test_subclass_with_multiple_known_tags_loads():
    class _Multi(Adapter):
        SOURCE_TAGS = {"ohlcv", "trades"}

        def adapt(self, raw):
            yield RawInput(id="x", source="ohlcv", payload=raw)

    assert _Multi.SOURCE_TAGS == {"ohlcv", "trades"}
    # Spot-check both tags are in the vocab.
    assert _Multi.SOURCE_TAGS.issubset(SOURCE_TAGS_VOCAB)


def test_adapt_must_be_implemented():
    class _Stub(Adapter):
        SOURCE_TAGS = {"ohlcv"}

    with pytest.raises(NotImplementedError):
        list(_Stub().adapt({}))


def test_source_tags_vocab_includes_tweets():
    # "tweets" was added by ENVSTREAM for the SocialSignal producer.
    assert "tweets" in SOURCE_TAGS_VOCAB


def test_source_tags_vocab_single_source_of_truth():
    # The vocab re-exported from trophic.adapters MUST be the very same
    # object as the one defined in trophic.types — otherwise drift between
    # the two will silently break adapter validation.
    assert SOURCE_TAGS_VOCAB is TYPES_VOCAB


def test_partial_unknown_tags_in_mixed_set_still_raises():
    # If at least one tag is unknown, the whole subclass must fail.
    with pytest.raises(ValueError, match="unknown SOURCE_TAGS"):
        class _Mixed(Adapter):  # noqa: F841
            SOURCE_TAGS = {"ohlcv", "definitely_not_real"}
