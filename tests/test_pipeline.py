"""End-to-end smoke against the mocked model + mocked apex judge."""
import asyncio

from trophic.config import TrophicConfig
from trophic.runner import Runner


def test_runner_three_tier_no_errors():
    cfg = TrophicConfig()
    cfg.runner.max_ticks = 3
    cfg.runner.inputs_per_tick = 6
    cfg.pool.db_path = ":memory:"

    runner = Runner(cfg=cfg)
    results = asyncio.get_event_loop().run_until_complete(runner.run())
    assert len(results) == 3

    total_substrate = sum(len(r.substrate) for r in results)
    total_herb = sum(len(r.herbivore_broadcasts) for r in results)
    total_pred = sum(len(r.predator_broadcasts) for r in results)
    assert total_substrate > 0
    assert total_herb > 0
    assert total_pred > 0


def test_runner_judgments_attach_to_predator_broadcasts():
    cfg = TrophicConfig()
    cfg.runner.max_ticks = 2
    cfg.runner.inputs_per_tick = 5
    cfg.pool.db_path = ":memory:"
    runner = Runner(cfg=cfg)
    results = asyncio.get_event_loop().run_until_complete(runner.run())
    for r in results:
        for br in r.predator_broadcasts:
            assert br.id in r.judgments
            j = r.judgments[br.id]
            assert 0.0 <= j.score <= 1.0
