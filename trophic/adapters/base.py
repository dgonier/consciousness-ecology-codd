"""Adapter ABC + the wavelength registry.

The canonical home of `SOURCE_TAGS_VOCAB` is `trophic/types.py` (so that
`from trophic.types import SOURCE_TAGS_VOCAB` works for any caller without
pulling in the adapters package). We re-export it here so adapter authors
can do `from trophic.adapters import SOURCE_TAGS_VOCAB`.
"""
from __future__ import annotations

from typing import Any, ClassVar, Iterable

from ..types import SOURCE_TAGS_VOCAB, RawInput

__all__ = ["Adapter", "SOURCE_TAGS_VOCAB"]


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
        # An empty SOURCE_TAGS marks an abstract intermediate (allowed).
        if not cls.SOURCE_TAGS:
            return
        unknown = set(cls.SOURCE_TAGS) - SOURCE_TAGS_VOCAB
        if unknown:
            raise ValueError(
                f"Adapter {cls.__name__} declares unknown SOURCE_TAGS: "
                f"{sorted(unknown)}. Add them to SOURCE_TAGS_VOCAB in "
                f"trophic/types.py first."
            )

    def adapt(self, raw: Any) -> Iterable[RawInput]:
        raise NotImplementedError(
            f"{type(self).__name__} must implement adapt(raw) -> Iterable[RawInput]"
        )
