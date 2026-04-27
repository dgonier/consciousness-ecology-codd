"""Centralized configuration knobs for the trophic system.

Everything tunable lives here so demos and tests can override via env or
explicit construction without touching call sites.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ModelConfig:
    chat_model_id: str = "Qwen/Qwen3-4B"
    embed_model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    forecaster_model_id: str = "amazon/chronos-bolt-base"
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 256
    channel_pool: str = "mean"  # mean | last
    forecaster_history_len: int = 64
    forecaster_horizon: int = 12
    # When MOCK_MODELS=1, model_host returns deterministic stand-in vectors so
    # the runner works without GPU/weights. Used by tests and quick demos.
    mock: bool = field(default_factory=lambda: os.environ.get("TROPHIC_MOCK_MODELS") == "1")


@dataclass
class BedrockConfig:
    region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1"))
    model_id: str = field(
        default_factory=lambda: os.environ.get(
            "TROPHIC_PREDATOR_MODEL", "us.anthropic.claude-sonnet-4-6"
        )
    )
    max_tokens: int = 512
    temperature: float = 0.2
    # Mirror of model mock — when set, predator returns canned judgments.
    mock: bool = field(default_factory=lambda: os.environ.get("TROPHIC_MOCK_PREDATOR") == "1")


@dataclass
class PoolConfig:
    db_path: str = field(default_factory=lambda: str(REPO_ROOT / "substrate.db"))
    retrieval_k: int = 40  # candidates pulled before appetite scoring


@dataclass
class AppetiteWeights:
    """Hand-initialized weights for the structured E in v1.

    These are inspectable knobs. Replace with a learnable function in v2.
    """
    age_weight: float = -0.05            # newer is more attractive (per tick)
    relevance_weight: float = 1.0         # cosine(retrieval_emb, query_emb)
    diversity_weight: float = 0.3         # penalty for similarity to meal-so-far
    reputation_weight: float = 0.2        # producer running reputation [-1..1]
    diet_match_weight: float = 1.0        # categorical multiplier; off-diet = -inf via filter
    min_appetite: float = 0.05            # below this, the consumer rejects the item


@dataclass
class PopulationConfig:
    intake_budget: int = 6                # stomach size per herbivore per meal
    hunger_threshold: float = 0.5         # below this fill ratio → hungry next tick
    energy_baseline: float = 1.0
    energy_dormant_threshold: float = 0.2
    energy_cull_threshold: float = -0.5
    cost_exist_per_tick: float = 0.02
    cost_produce_per_item: float = 0.01
    reward_eaten_per_item: float = 0.05
    reward_useful_bonus: float = 0.15      # per positively-judged downstream synthesis
    reward_intake_per_item: float = 0.04
    reward_judgment_scale: float = 0.5     # multiplied by predator score [0..1]


@dataclass
class RunnerConfig:
    max_ticks: int = 20
    inputs_per_tick: int = 4
    seed: int = 7
    # Initial agent counts
    n_producers_per_kind: int = 1   # one of each of the three kinds
    n_herbivores_per_kind: int = 1  # one of each of the two kinds


@dataclass
class TrophicConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    bedrock: BedrockConfig = field(default_factory=BedrockConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)
    appetite: AppetiteWeights = field(default_factory=AppetiteWeights)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)


DEFAULT_CONFIG = TrophicConfig()
