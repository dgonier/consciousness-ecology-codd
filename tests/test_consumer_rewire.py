"""Tests for the phase2-D:04 hooks-mode consumer rewire.

Tests dispatch behavior gated by TROPHIC_CONSUMER_INTERFACE.
Uses the mock ModelHost (TROPHIC_MOCK_MODELS=1, set by conftest.py) so
we don't load Qwen3-4B in CI. The heavy real-Qwen smoke is in the
mission's Testing Conditions block, run separately.
"""
import os

import pytest
import torch


def test_predator_has_phi_mlp_in_hooks_mode(monkeypatch):
    """In hooks mode, ensure_initialized must build a PhiMLP on the predator."""
    monkeypatch.setenv("TROPHIC_CONSUMER_INTERFACE", "hooks")
    from trophic.config import DEFAULT_CONFIG
    from trophic.model_host import ModelHost
    from trophic.agents.predator import Predator
    from trophic.phi_mlp import PhiMLP

    host = ModelHost(DEFAULT_CONFIG.model)
    pred = Predator.make("short_horizon")
    pred.ensure_initialized(host, seed_base=42)

    assert hasattr(pred, "phi_mlp"), "predator should have phi_mlp attribute"
    assert pred.phi_mlp is not None, "phi_mlp should be built in hooks mode"
    assert isinstance(pred.phi_mlp, PhiMLP)
    # And its params should be alive — count > 0.
    n_params = sum(p.numel() for p in pred.phi_mlp.parameters())
    assert n_params > 0


def test_predator_no_phi_mlp_in_prefix_mode(monkeypatch):
    """In default `prefix` mode, phi_mlp must remain None — backward compat."""
    monkeypatch.setenv("TROPHIC_CONSUMER_INTERFACE", "prefix")
    from trophic.config import DEFAULT_CONFIG
    from trophic.model_host import ModelHost
    from trophic.agents.predator import Predator

    host = ModelHost(DEFAULT_CONFIG.model)
    pred = Predator.make("short_horizon")
    pred.ensure_initialized(host, seed_base=42)
    assert getattr(pred, "phi_mlp", None) is None, (
        "prefix mode must NOT build phi_mlp (backward compat with seed1..seed22)"
    )


def test_herbivore_has_phi_mlp_in_hooks_mode(monkeypatch):
    """In hooks mode, herbivores also get a phi_mlp built (parallel to predator)."""
    monkeypatch.setenv("TROPHIC_CONSUMER_INTERFACE", "hooks")
    from trophic.config import DEFAULT_CONFIG
    from trophic.model_host import ModelHost
    from trophic.agents.herbivore import Herbivore
    from trophic.phi_mlp import PhiMLP

    host = ModelHost(DEFAULT_CONFIG.model)
    h = Herbivore.make("technical")
    h.ensure_initialized(host, seed_base=42)
    assert hasattr(h, "phi_mlp")
    assert h.phi_mlp is not None
    assert isinstance(h.phi_mlp, PhiMLP)


def test_consumer_interface_helper_default(monkeypatch):
    """SFTRunner._consumer_interface() returns env-var value, default `prefix`."""
    monkeypatch.delenv("TROPHIC_CONSUMER_INTERFACE", raising=False)
    # Default value must be 'prefix'.
    assert os.environ.get("TROPHIC_CONSUMER_INTERFACE", "prefix") == "prefix"

    monkeypatch.setenv("TROPHIC_CONSUMER_INTERFACE", "hooks")
    assert os.environ.get("TROPHIC_CONSUMER_INTERFACE") == "hooks"

    # And the helper on the runner respects it. Build a minimal runner via
    # the mock host so we can call the method.
    from trophic.config import DEFAULT_CONFIG
    from trophic.model_host import ModelHost
    from trophic.agents.herbivore import Herbivore
    from trophic.agents.predator import Predator
    from trophic.training.sft import SFTConfig, SFTRunner

    host = ModelHost(DEFAULT_CONFIG.model)
    herbs = [Herbivore.make("technical"), Herbivore.make("fundamental")]
    pred = Predator.make("short_horizon")
    runner = SFTRunner(
        cfg=SFTConfig(seed=7, steps=1),
        host=host,
        producers=[],
        herbivores=herbs,
        predator=pred,
        train=[],
        eval_=[],
    )
    assert runner._consumer_interface() == "hooks"

    monkeypatch.setenv("TROPHIC_CONSUMER_INTERFACE", "prefix")
    assert runner._consumer_interface() == "prefix"


def test_phi_mlp_params_in_trainer_in_hooks_mode(monkeypatch):
    """When hooks mode is active, phi_mlp parameters MUST be picked up by
    the ChannelTrainer's optimizer (verified by counting trainable params)."""
    from trophic.config import DEFAULT_CONFIG
    from trophic.model_host import ModelHost
    from trophic.agents.herbivore import Herbivore
    from trophic.agents.predator import Predator
    from trophic.training.sft import SFTConfig, SFTRunner

    host = ModelHost(DEFAULT_CONFIG.model)

    # 1. Build runner in PREFIX mode and capture param count.
    monkeypatch.setenv("TROPHIC_CONSUMER_INTERFACE", "prefix")
    herbs_p = [Herbivore.make("technical"), Herbivore.make("fundamental")]
    pred_p = Predator.make("short_horizon")
    runner_p = SFTRunner(
        cfg=SFTConfig(seed=7, steps=1),
        host=host,
        producers=[],
        herbivores=herbs_p,
        predator=pred_p,
        train=[],
        eval_=[],
    )
    n_prefix = len(runner_p.trainer._params)
    prefix_total = sum(p.numel() for p in runner_p.trainer._params)

    # 2. Build runner in HOOKS mode — must have strictly more trainable
    #    parameters because phi_mlp params join the optimizer.
    monkeypatch.setenv("TROPHIC_CONSUMER_INTERFACE", "hooks")
    herbs_h = [Herbivore.make("technical"), Herbivore.make("fundamental")]
    pred_h = Predator.make("short_horizon")
    runner_h = SFTRunner(
        cfg=SFTConfig(seed=7, steps=1),
        host=host,
        producers=[],
        herbivores=herbs_h,
        predator=pred_h,
        train=[],
        eval_=[],
    )
    n_hooks = len(runner_h.trainer._params)
    hooks_total = sum(p.numel() for p in runner_h.trainer._params)

    assert n_hooks > n_prefix, (
        f"hooks mode should add phi_mlp params: prefix={n_prefix}, hooks={n_hooks}"
    )
    assert hooks_total > prefix_total, (
        f"hooks mode total numel {hooks_total} should exceed prefix total {prefix_total}"
    )
    # And every herb + the predator should now have a phi_mlp.
    assert pred_h.phi_mlp is not None
    for h in herbs_h:
        assert h.phi_mlp is not None
