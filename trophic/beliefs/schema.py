"""Bayesian belief network — node and edge dataclasses.

See design_docs/belief_network_schema.md for the full design.

Categorical-only LLM interface: every choice the LLM makes is one-of-N
from a fixed enum. Every numeric value used in math is computed from
the categorical via lookup table. Recalibration = adjust the tables.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional


# ───── Categorical type aliases ─────

Scope = Literal["macro", "market", "sector", "company"]
Direction = Literal["positive", "negative"]
Magnitude = Literal["weak", "medium", "strong", "decisive"]
DecayClass = Literal["instant", "fast", "normal", "slow", "glacial"]
ValidationStatus = Literal["unverified", "validated", "rejected", "uncertain"]
EvidenceLevel = Literal["novel", "weak", "moderate", "strong", "well_established"]
DirectionalLean = Literal[
    "strongly_false", "leans_false", "neutral", "leans_true", "strongly_true",
]
ConfidenceLevel = Literal["low", "medium", "high", "very_high"]


# ───── Calibration tables ─────

# Half-life in HOURS for each decay class. LLM picks one per activation;
# default "normal" if unspecified.
DECAY_HALF_LIFE_HOURS: dict[str, float] = {
    "instant":  0.001,    # ~snap (Polymarket-resolved beliefs)
    "fast":     1 / 60,   # 1 minute (intraday momentum)
    "normal":   1.0,      # 1 hour (standard news/sentiment)
    "slow":     24.0,     # 1 day (earnings, regulatory)
    "glacial":  168.0,    # 1 week (M&A, structural)
}

# Magnitude → log-odds shift. LLM emits categorical; this converts to numeric.
MAGNITUDE_TO_LOG_ODDS: dict[str, float] = {
    "weak":     0.4,    # ~odds × 1.5
    "medium":   1.0,    # ~odds × 2.7
    "strong":   1.6,    # ~odds × 5
    "decisive": 2.3,    # ~odds × 10
}

# (EvidenceLevel, DirectionalLean) → prior_p. Used when LLM proposes a new
# belief or when humans hand-classify.
PRIOR_P_TABLE: dict[tuple[str, str], float] = {
    # Novel: no historical evidence. Always agnostic regardless of lean.
    ("novel", "strongly_false"):  0.50,
    ("novel", "leans_false"):     0.50,
    ("novel", "neutral"):         0.50,
    ("novel", "leans_true"):      0.50,
    ("novel", "strongly_true"):   0.50,
    # Weak: one or two observations, anecdotal pattern.
    ("weak", "strongly_false"):   0.40,
    ("weak", "leans_false"):      0.45,
    ("weak", "neutral"):          0.50,
    ("weak", "leans_true"):       0.55,
    ("weak", "strongly_true"):    0.60,
    # Moderate: documented in research, ~years of data.
    ("moderate", "strongly_false"): 0.30,
    ("moderate", "leans_false"):    0.40,
    ("moderate", "neutral"):        0.50,
    ("moderate", "leans_true"):     0.60,
    ("moderate", "strongly_true"):  0.70,
    # Strong: well-cited, decade-plus track record.
    ("strong", "strongly_false"):   0.20,
    ("strong", "leans_false"):      0.35,
    ("strong", "neutral"):          0.50,
    ("strong", "leans_true"):       0.65,
    ("strong", "strongly_true"):    0.80,
    # Well-established: textbook regularity, multiple decades, citable base rate.
    ("well_established", "strongly_false"): 0.15,
    ("well_established", "leans_false"):    0.30,
    ("well_established", "neutral"):        0.50,
    ("well_established", "leans_true"):     0.70,
    ("well_established", "strongly_true"):  0.85,
}

# ConfidenceLevel → numeric weight for math
CONFIDENCE_TABLE: dict[str, float] = {
    "low":       0.4,
    "medium":    0.6,
    "high":      0.8,
    "very_high": 0.95,
}


def prior_p_from_categorical(
    evidence_level: EvidenceLevel,
    directional_lean: DirectionalLean,
) -> float:
    """Look up prior_p from the categorical pair the LLM emits."""
    return PRIOR_P_TABLE[(evidence_level, directional_lean)]


# ───── Node and edge dataclasses ─────

@dataclass
class InternalLink:
    """A causal claim wired into the world model.

    Hand-seeded first; LLM-proposed extensions later (with human review).
    `strength_posterior` is the CPT entry — the conditional probability of the
    conclusion belief shifting given the premise belief shifting. Updated by
    historical validation and (eventually) Hexis decomposers.

    For LLM-proposed links, strength_prior is derived from EvidenceLevel via
    PRIOR_P_TABLE neutral-lean column. Hand-seeded links can use any value
    a human cites.
    """
    id: str
    premise_belief_id: str
    conclusion_belief_id: str
    scope: Scope
    direction: Direction
    strength_prior: float
    strength_posterior: float
    n_validations: int = 0
    n_correct: int = 0
    citation: str = ""
    validation_status: ValidationStatus = "unverified"
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength_prior <= 1.0:
            raise ValueError(f"strength_prior must be in [0,1], got {self.strength_prior}")
        if not 0.0 <= self.strength_posterior <= 1.0:
            raise ValueError(f"strength_posterior must be in [0,1], got {self.strength_posterior}")


@dataclass
class StateBelief:
    """Current credence in a world-state proposition.

    Decays toward `prior_p` (the base rate) without fresh evidence — credence
    representing only the evidence's contribution fades, leaving the base rate.

    If `polymarket_market_id` is set, current_p is read-through from the market
    while open; LLM activations on this belief are ignored.
    """
    id: str
    statement_template: str
    scope: Scope
    prior_p: float
    current_p: float
    decay_class: DecayClass = "normal"
    context: dict[str, str] = field(default_factory=dict)
    last_updated: float = 0.0
    evidence_log: list[str] = field(default_factory=list)
    polymarket_market_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.prior_p <= 1.0:
            raise ValueError(f"prior_p must be in [0,1], got {self.prior_p}")
        if not 0.0 <= self.current_p <= 1.0:
            raise ValueError(f"current_p must be in [0,1], got {self.current_p}")

    @property
    def is_polymarket_bound(self) -> bool:
        return self.polymarket_market_id is not None

    @classmethod
    def from_categorical(
        cls,
        id: str,
        statement_template: str,
        scope: Scope,
        evidence_level: EvidenceLevel,
        directional_lean: DirectionalLean,
        decay_class: DecayClass = "normal",
        context: Optional[dict[str, str]] = None,
        polymarket_market_id: Optional[str] = None,
    ) -> "StateBelief":
        """Construct from the categorical LLM-friendly interface.

        prior_p is derived from (evidence_level, directional_lean) via
        PRIOR_P_TABLE. current_p is initialized to prior_p (no evidence yet).
        """
        p = prior_p_from_categorical(evidence_level, directional_lean)
        return cls(
            id=id,
            statement_template=statement_template,
            scope=scope,
            prior_p=p,
            current_p=p,
            decay_class=decay_class,
            context=context or {},
            polymarket_market_id=polymarket_market_id,
        )


@dataclass
class OutcomeBelief:
    """Prediction target. Computed by belief propagation, not predicted.

    Apex tier reads `p_up` rather than guessing direction from text.
    """
    id: str
    ticker: str
    horizon_min: int
    statement: str
    p_up: float
    contributing_state_beliefs: list[str] = field(default_factory=list)
    contributing_links: list[str] = field(default_factory=list)
    activated_at: float = 0.0
    resolved: bool = False
    actual_direction: Optional[Literal["up", "down"]] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_up <= 1.0:
            raise ValueError(f"p_up must be in [0,1], got {self.p_up}")


@dataclass
class BeliefActivation:
    """One species' (or one event's) attempt to update a state belief.

    Two confidence representations by design:
      - `magnitude_logit`: model's logprob on the chosen magnitude class.
        The actual confidence weight in math when available.
      - `self_rated_confidence`: LLM's self-rated category. Audit + species
        calibration tracking.

    Both are derived from categorical LLM output. The LLM never emits
    floats directly.
    """
    target_belief_id: str
    direction_of_effect: Literal["increases", "decreases"]
    magnitude: Magnitude
    decay_class: DecayClass = "normal"
    self_rated_confidence: ConfidenceLevel = "medium"
    magnitude_logit: Optional[float] = None
    reasoning: str = ""
    species_id: str = ""

    @property
    def effective_confidence(self) -> float:
        """Confidence used in math. Prefer magnitude_logit (better calibrated);
        fall back to self_rated_confidence's table value when logit unavailable.

        magnitude_logit is a logprob (typically [-10, 0]). Convert to a
        confidence weight in [0, 1] via exp.
        """
        if self.magnitude_logit is not None:
            return min(1.0, max(0.0, math.exp(self.magnitude_logit)))
        return CONFIDENCE_TABLE[self.self_rated_confidence]


@dataclass
class Event:
    """One observed input (news headline, filing, price move, Polymarket update).

    Carries one or more `BeliefActivation` records — the LLM's interpretation
    of which beliefs the event affects.
    """
    id: str
    timestamp: float
    source: str
    raw_content: str
    ticker: Optional[str] = None
    sector: Optional[str] = None
    activations: list[BeliefActivation] = field(default_factory=list)
