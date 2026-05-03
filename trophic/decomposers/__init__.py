"""Decomposers — two jobs, both ecosystem-level (not gradient-level):

  1. KG WRITER — observe each (scenario, voter_responses, ensemble_decision,
     ground_truth) tuple and emit Hexis-style records to a knowledge graph
     (file-based JSONL today; queryable later). The KG is the ecosystem's
     long-term memory — what worked, what didn't, in retrievable form
     keyed by scenario signature.

  2. EVOLUTION MANAGER — track per-agent fitness over a rolling window;
     decide which agents should reproduce (clone with mutation; e.g.
     temperature variation, prompt-template variation) and which should
     die (be removed from the population). Reproduction and death are
     POLICY choices, not arithmetic — the manager exposes signals, the
     orchestrator chooses to act on them at session boundaries.

The decomposer does NOT touch agent weights directly. It changes WHO IS
IN THE POPULATION over time. For LLM voters the unit of selection is
configuration (model id, temperature, system prompt); LoRA-on-E
mutations are a future extension.
"""
from .kg_writer import KGWriter, KGRecord
from .fitness import FitnessTracker, AgentFitness
from .population_manager import PopulationManager, EvolutionDecision
from .observation import (
    Observation, ObservationWriter, NodeSignal, DecomposerJudgment,
)
from .capture import capture_evidence_signals

__all__ = [
    "KGWriter", "KGRecord",
    "FitnessTracker", "AgentFitness",
    "PopulationManager", "EvolutionDecision",
    "Observation", "ObservationWriter", "NodeSignal", "DecomposerJudgment",
    "capture_evidence_signals",
]
