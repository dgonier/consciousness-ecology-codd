"""InvestmentThesis: persistent identity artifact for an apex predator.

A predator's `ThesisBook` is what gives it cross-day identity. The apex
agent commits to a thesis when it opens a position (catalyst story,
expected alpha, invalidation triggers) and the thesis carries forward
until it matures (horizon elapsed), is invalidated (trigger fires), or
expires (overdue without an explicit close).

This file is type plumbing + lifecycle helpers only — no predator code,
no debate code, no apex/portfolio wiring. 05c will compose `ThesisBook`
into `PredatorSubPortfolio`; 05d reads catalysts + invalidation_triggers
when predators debate one another's open theses.

Validation invariants:
  - status='active'    ⇒ closed_at_date / close_reason / realized_* are None
  - status≠'active'    ⇒ closed_at_date and close_reason are set, and
                         realized_pnl + realized_return_pct are set
"""
from __future__ import annotations

import uuid
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trophic.beliefs.validators import HORIZON_DAYS


# ── Type aliases ────────────────────────────────────────────────────

ThesisStatus = Literal["active", "matured", "invalidated", "expired"]
ThesisDirection = Literal["long", "short", "neutral"]
PredatorPhilosophy = Literal["momentum", "value", "mean_revert", "event_driven"]
ThesisHorizon = Literal["h1", "h5", "h20", "h60"]
ThesisConfidence = Literal["low", "med", "high"]


# ── Core artifact ────────────────────────────────────────────────────

class InvestmentThesis(BaseModel):
    """A single thesis owned by a predator, with a full open→close lifecycle.

    A thesis is born `active` and ends in one of three closed states:
        matured        — primary_horizon elapsed naturally, position exited.
        invalidated    — an invalidation trigger fired; close_reason carries it.
        expired        — horizon passed without an explicit close (sweep_overdue).
    """

    model_config = ConfigDict(populate_by_name=True)

    thesis_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="12-char uuid4 hex, immutable after construction.",
    )
    predator_id: str = Field(..., description="Which predator owns this thesis.")
    philosophy: PredatorPhilosophy
    ticker: str = Field(..., min_length=1, max_length=10)
    opened_at_date: str
    direction: ThesisDirection
    primary_horizon: ThesisHorizon

    catalysts: list[str] = Field(
        default_factory=list,
        description="Named events / signals that motivated the thesis.",
    )
    invalidation_triggers: list[str] = Field(
        default_factory=list,
        description="Conditions under which the thesis should close early.",
    )
    target_price: Optional[float] = None
    expected_alpha_bps: float = Field(..., ge=-10000, le=10000)
    confidence: ThesisConfidence

    status: ThesisStatus = "active"
    closed_at_date: Optional[str] = None
    close_reason: Optional[str] = None  # "matured" / "invalidated:<trigger>" / "expired" / "manual"

    realized_pnl: Optional[float] = None
    realized_return_pct: Optional[float] = None

    @model_validator(mode="after")
    def _check_lifecycle_consistency(self) -> "InvestmentThesis":
        active = self.status == "active"
        if active:
            if self.closed_at_date is not None:
                raise ValueError("active thesis must not have closed_at_date set")
            if self.close_reason is not None:
                raise ValueError("active thesis must not have close_reason set")
            if self.realized_pnl is not None:
                raise ValueError("active thesis must not have realized_pnl set")
            if self.realized_return_pct is not None:
                raise ValueError("active thesis must not have realized_return_pct set")
        else:
            if self.closed_at_date is None:
                raise ValueError(f"status={self.status!r} requires closed_at_date")
            if self.close_reason is None:
                raise ValueError(f"status={self.status!r} requires close_reason")
            if self.realized_pnl is None:
                raise ValueError(f"status={self.status!r} requires realized_pnl")
            if self.realized_return_pct is None:
                raise ValueError(f"status={self.status!r} requires realized_return_pct")
        return self


# ── Lifecycle helpers ────────────────────────────────────────────────

def open_thesis(
    predator_id: str,
    philosophy: PredatorPhilosophy,
    ticker: str,
    opened_at_date: str,
    direction: ThesisDirection,
    primary_horizon: ThesisHorizon,
    catalysts: list[str],
    invalidation_triggers: list[str],
    expected_alpha_bps: float,
    confidence: ThesisConfidence,
    target_price: Optional[float] = None,
) -> InvestmentThesis:
    """Construct a fresh active thesis. Thin wrapper to keep call sites tidy."""
    return InvestmentThesis(
        predator_id=predator_id,
        philosophy=philosophy,
        ticker=ticker,
        opened_at_date=opened_at_date,
        direction=direction,
        primary_horizon=primary_horizon,
        catalysts=list(catalysts),
        invalidation_triggers=list(invalidation_triggers),
        expected_alpha_bps=expected_alpha_bps,
        confidence=confidence,
        target_price=target_price,
    )


def mature_thesis(
    thesis: InvestmentThesis,
    closed_at_date: str,
    realized_pnl: float,
    realized_return_pct: float,
) -> InvestmentThesis:
    """Mark a thesis 'matured': its primary_horizon elapsed naturally.

    Returns a new InvestmentThesis (immutability via model_copy)."""
    return thesis.model_copy(update={
        "status": "matured",
        "closed_at_date": closed_at_date,
        "close_reason": "matured",
        "realized_pnl": realized_pnl,
        "realized_return_pct": realized_return_pct,
    })


def invalidate_thesis(
    thesis: InvestmentThesis,
    closed_at_date: str,
    trigger: str,
    realized_pnl: float,
    realized_return_pct: float,
) -> InvestmentThesis:
    """Mark a thesis 'invalidated' because a trigger fired.

    close_reason is "invalidated:<trigger>" so callers can recover the
    trigger string downstream.
    """
    return thesis.model_copy(update={
        "status": "invalidated",
        "closed_at_date": closed_at_date,
        "close_reason": f"invalidated:{trigger}",
        "realized_pnl": realized_pnl,
        "realized_return_pct": realized_return_pct,
    })


def expire_thesis(
    thesis: InvestmentThesis,
    closed_at_date: str,
    realized_pnl: float,
    realized_return_pct: float,
) -> InvestmentThesis:
    """Mark a thesis 'expired': horizon elapsed without an explicit close."""
    return thesis.model_copy(update={
        "status": "expired",
        "closed_at_date": closed_at_date,
        "close_reason": "expired",
        "realized_pnl": realized_pnl,
        "realized_return_pct": realized_return_pct,
    })


def is_thesis_overdue(
    thesis: InvestmentThesis,
    today_idx: int,
    opened_idx: int,
) -> bool:
    """True iff the thesis is still active AND its horizon has elapsed.

    `today_idx` and `opened_idx` are indices into the runner's trading
    calendar (`_dates_seen`).
    """
    if thesis.status != "active":
        return False
    horizon = HORIZON_DAYS[thesis.primary_horizon]
    return (today_idx - opened_idx) >= horizon


# ── Book wrapper ─────────────────────────────────────────────────────

class ThesisBook(BaseModel):
    """The per-predator book of theses (active + history).

    A book is owned by exactly one predator. The 4 apex predators each
    carry their own book; books never cross-pollinate. See 05c for
    `PredatorSubPortfolio` which composes this with a capital slice.
    """

    model_config = ConfigDict(populate_by_name=True)

    predator_id: str
    theses: list[InvestmentThesis] = Field(default_factory=list)

    def active_theses(self) -> list[InvestmentThesis]:
        return [t for t in self.theses if t.status == "active"]

    def active_for_ticker(self, ticker: str) -> list[InvestmentThesis]:
        tk = ticker.upper()
        return [t for t in self.theses if t.status == "active" and t.ticker.upper() == tk]

    def closed_theses(self) -> list[InvestmentThesis]:
        return [t for t in self.theses if t.status != "active"]

    def add(self, thesis: InvestmentThesis) -> None:
        """Append a thesis. Caller is responsible for predator_id consistency.

        Raises ValueError if `thesis.predator_id` does not match this book's
        owner — book ownership is a hard invariant.
        """
        if thesis.predator_id != self.predator_id:
            raise ValueError(
                f"thesis.predator_id={thesis.predator_id!r} does not match "
                f"book.predator_id={self.predator_id!r}"
            )
        self.theses.append(thesis)

    def close(
        self,
        thesis_id: str,
        status: ThesisStatus,
        closed_at_date: str,
        realized_pnl: float,
        realized_return_pct: float,
        trigger: Optional[str] = None,
    ) -> InvestmentThesis:
        """Close one thesis in-place; returns the closed thesis.

        `status` must be one of matured / invalidated / expired (not 'active').
        For status='invalidated', `trigger` must be supplied.
        Raises KeyError if no thesis with that id exists.
        Raises ValueError on illegal status / missing trigger.
        """
        if status == "active":
            raise ValueError("close() requires a terminal status, got 'active'")
        if status == "invalidated" and not trigger:
            raise ValueError("close() with status='invalidated' requires a trigger")

        for i, t in enumerate(self.theses):
            if t.thesis_id == thesis_id:
                if t.status != "active":
                    raise ValueError(
                        f"thesis {thesis_id} already closed (status={t.status!r})"
                    )
                if status == "matured":
                    new_t = mature_thesis(t, closed_at_date, realized_pnl, realized_return_pct)
                elif status == "invalidated":
                    assert trigger is not None  # narrowed above
                    new_t = invalidate_thesis(t, closed_at_date, trigger, realized_pnl, realized_return_pct)
                else:  # expired
                    new_t = expire_thesis(t, closed_at_date, realized_pnl, realized_return_pct)
                self.theses[i] = new_t
                return new_t
        raise KeyError(f"no thesis with id={thesis_id!r} in book for {self.predator_id!r}")

    def sweep_overdue(
        self,
        today_idx: int,
        opened_idx_by_thesis: dict[str, int],
    ) -> list[InvestmentThesis]:
        """Expire every overdue active thesis. Returns the newly-expired list.

        Realized P&L is set to 0.0 / 0.0% for expiry-by-sweep — the runner
        should ALSO push a real close when it has an actual realized number;
        sweep only catches theses that the runner failed to close explicitly.
        Caller must supply `opened_idx_by_thesis` mapping thesis_id → calendar idx.
        Theses missing from the map are skipped (no calendar info ⇒ no expiry).
        """
        newly_expired: list[InvestmentThesis] = []
        # Use synthetic closed_at_date sentinel — the runner that owns the
        # calendar should re-close with the real date. We carry today_idx in
        # the sentinel so it's recoverable.
        closed_at_sentinel = f"sweep-expired@day{today_idx}"
        for i, t in enumerate(self.theses):
            if t.status != "active":
                continue
            opened_idx = opened_idx_by_thesis.get(t.thesis_id)
            if opened_idx is None:
                continue
            if is_thesis_overdue(t, today_idx, opened_idx):
                new_t = expire_thesis(t, closed_at_sentinel, 0.0, 0.0)
                self.theses[i] = new_t
                newly_expired.append(new_t)
        return newly_expired
