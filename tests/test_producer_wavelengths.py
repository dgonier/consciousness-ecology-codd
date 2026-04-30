"""Producer wavelength contract tests (phase2-D:04).

These tests verify the WAVELENGTHS class-attr refactor:
  - existing producer subtypes (TickDelta, Disclosure, Anomaly,
    QuantitativeProducer) declare correct wavelength sets
  - many-to-many holds (one source feeds multiple producers)
  - the legacy `ATTRACTION` global dict is gone from `producer.py`
  - the new SocialSignal producer attracts only "tweets"
  - the `Producer.make(kind)` factory still dispatches correctly
"""
from __future__ import annotations

from trophic.agents.producer import Producer, TickDelta, Disclosure, Anomaly
from trophic.agents.quant_producer import QuantitativeProducer
from trophic.agents.social_signal import SocialSignal
from trophic.types import RawInput


def _ri(source: str) -> RawInput:
    return RawInput(id=f"r-{source}", source=source, payload={})


def test_tickdelta_wavelengths():
    assert TickDelta.WAVELENGTHS == {"ohlcv", "trades", "book"}
    p = Producer.make("tickdelta")
    assert isinstance(p, TickDelta)
    assert p.attracts(_ri("ohlcv"))
    assert p.attracts(_ri("trades"))
    assert p.attracts(_ri("book"))
    assert not p.attracts(_ri("filing"))
    assert not p.attracts(_ri("tweets"))


def test_disclosure_wavelengths():
    assert Disclosure.WAVELENGTHS == {"filing", "press"}
    p = Producer.make("disclosure")
    assert isinstance(p, Disclosure)
    assert p.attracts(_ri("filing"))
    assert p.attracts(_ri("press"))
    assert not p.attracts(_ri("ohlcv"))


def test_anomaly_overlaps_ohlcv_with_tickdelta():
    """Many-to-many: an `ohlcv` source feeds BOTH tickdelta and anomaly."""
    assert Anomaly.WAVELENGTHS == {"ohlcv", "options", "halt"}
    td = Producer.make("tickdelta")
    an = Producer.make("anomaly")
    inp = _ri("ohlcv")
    assert td.attracts(inp)
    assert an.attracts(inp)
    # And anomaly-only sources don't trigger tickdelta:
    assert not td.attracts(_ri("options"))
    assert an.attracts(_ri("options"))
    assert an.attracts(_ri("halt"))


def test_quantitative_wavelengths():
    qp = QuantitativeProducer.make("quote_series")
    assert qp.WAVELENGTHS == {"quote_series"}
    assert "quote_series" in QuantitativeProducer.WAVELENGTHS
    assert qp.attracts(_ri("quote_series"))
    assert not qp.attracts(_ri("ohlcv"))
    assert not qp.attracts(_ri("tweets"))


def test_social_signal_wavelengths_only_tweets():
    ss = SocialSignal.make()
    assert ss.WAVELENGTHS == {"tweets"}
    assert ss.attracts(_ri("tweets"))
    assert not ss.attracts(_ri("ohlcv"))
    assert not ss.attracts(_ri("filing"))
    assert not ss.attracts(_ri("quote_series"))


def test_no_global_attraction_dict():
    """ATTRACTION dict from the old code should be gone from both producer
    modules. The contract is class-attr-driven now."""
    import trophic.agents.producer as prod
    import trophic.agents.quant_producer as qprod
    assert not hasattr(prod, "ATTRACTION"), (
        "ATTRACTION dict should be removed from trophic.agents.producer"
    )
    assert not hasattr(qprod, "ATTRACTION"), (
        "ATTRACTION dict should be removed from trophic.agents.quant_producer"
    )


def test_producer_make_factory_dispatches_to_subclasses():
    """Producer.make(kind) returns the correct subclass instance for every
    legacy kind string still used by call sites in scripts/ and runner.py."""
    td = Producer.make("tickdelta")
    ds = Producer.make("disclosure")
    an = Producer.make("anomaly")
    assert type(td) is TickDelta
    assert type(ds) is Disclosure
    assert type(an) is Anomaly
    # social_signal also dispatches via the same factory because it is a
    # Producer subclass and thus discovered by Producer.make's subclass walk.
    ss = Producer.make("social_signal")
    assert type(ss) is SocialSignal


def test_wavelengths_are_subset_of_vocab():
    """Every producer's WAVELENGTHS must be tags from the shared vocabulary."""
    from trophic.types import SOURCE_TAGS_VOCAB
    for cls in (TickDelta, Disclosure, Anomaly, QuantitativeProducer, SocialSignal):
        assert cls.WAVELENGTHS, f"{cls.__name__} declares empty WAVELENGTHS"
        assert cls.WAVELENGTHS <= SOURCE_TAGS_VOCAB, (
            f"{cls.__name__}.WAVELENGTHS ({cls.WAVELENGTHS}) leaks tags outside "
            f"SOURCE_TAGS_VOCAB ({SOURCE_TAGS_VOCAB})"
        )
