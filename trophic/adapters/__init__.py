"""Adapter layer: format converters that turn external data into RawInput streams.

Adapters are single-responsibility format converters. Each adapter declares
which SOURCE_TAGS it emits (its "wavelength"). Producers subscribe to
wavelength subsets via their WAVELENGTHS class attr (set in phase2-D).

Many-to-many is the contract: one adapter's output (e.g. source="ohlcv")
can feed multiple producers (TickDelta, Anomaly).
"""
from .base import Adapter, SOURCE_TAGS_VOCAB

__all__ = ["Adapter", "SOURCE_TAGS_VOCAB"]
