"""Belief-network herbivore species.

Each herbivore reads some slice of the firehose (news, bars, Polymarket,
filings) and emits BeliefActivation records into the belief store. They
are the system's "what does this evidence mean for our beliefs?" layer.

Architecturally distinct from `trophic.agents.herbivore` — those are
trained-LLM herbivores that feed the trophic-stack predator/apex chain.
The belief-network herbivores here feed the belief store + propagation
+ carnivore aggregator pipeline.
"""
from .cross_correlation import (
    CorrelationState,
    CrossCorrelationHerbivore,
)
from .event_classifier import (
    EventClassifierHerbivore,
    ALLOWED_TEMPLATES,
)

__all__ = [
    "CorrelationState",
    "CrossCorrelationHerbivore",
    "EventClassifierHerbivore",
    "ALLOWED_TEMPLATES",
]
