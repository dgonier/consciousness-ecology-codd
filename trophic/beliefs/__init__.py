"""Bayesian belief network for trophic.

A causal-graph substrate that the apex tier consults instead of guessing
direction from text. Three node kinds (InternalLink, StateBelief,
OutcomeBelief) plus Event records that carry BeliefActivations.

Designed to write to Neo4j eventually; JSONL-backed in dev so we can
build without the DB. Same interface either way.

See design_docs/belief_network_schema.md for the full design.
"""
from .schema import (
    InternalLink,
    StateBelief,
    OutcomeBelief,
    Event,
    BeliefActivation,
    Scope,
    DecayClass,
    Magnitude,
    Direction,
    DECAY_HALF_LIFE_HOURS,
    MAGNITUDE_TO_LOG_ODDS,
)
from .propagation import (
    log_odds, prob_from_log_odds,
    apply_decay, apply_activation, propagate_outcome,
    propagate_network, topological_order,
)
from .store import BeliefStore, JSONLBeliefStore
from .viz import snapshot_pass, render_pass_viz
from .embeddings import ModalEmbedder, cosine_sim
from .consolidate import consolidate, ConsolidationReport, MergeCandidate

# Neo4jBeliefStore lazy-imported (depends on neo4j package being installed
# AND on env vars being set). Surface it via accessor not top-level import
# so the module loads even when neo4j is unavailable.
def Neo4jBeliefStore(*args, **kwargs):
    from .neo4j_store import Neo4jBeliefStore as _Impl
    return _Impl(*args, **kwargs)


__all__ = [
    # schema
    "InternalLink", "StateBelief", "OutcomeBelief", "Event", "BeliefActivation",
    "Scope", "DecayClass", "Magnitude", "Direction",
    "DECAY_HALF_LIFE_HOURS", "MAGNITUDE_TO_LOG_ODDS",
    # math
    "log_odds", "prob_from_log_odds",
    "apply_decay", "apply_activation", "propagate_outcome",
    "propagate_network", "topological_order",
    # storage
    "BeliefStore", "JSONLBeliefStore", "Neo4jBeliefStore",
    # viz
    "snapshot_pass", "render_pass_viz",
    # embeddings & consolidation
    "ModalEmbedder", "cosine_sim",
    "consolidate", "ConsolidationReport", "MergeCandidate",
]
