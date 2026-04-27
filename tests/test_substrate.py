import uuid

from trophic.substrate import SubstratePool
from trophic.types import Broadcast


def _b(tier="substrate", kind="tickdelta", diet=("from_tickdelta",)):
    return Broadcast(
        id=str(uuid.uuid4()),
        tier=tier,
        agent_id="a.x",
        agent_kind=kind,
        diet_tags=list(diet),
        channel_embedding=[0.1] * 8,
    )


def test_roundtrip_and_diet_filter_by_tier():
    pool = SubstratePool(":memory:")
    a = _b(tier="substrate", diet=("from_tickdelta",))
    b = _b(tier="substrate", diet=("from_disclosure",))
    c = _b(tier="herbivore_broadcast", diet=("from_technical_herbivore",))
    pool.add_broadcast_many([a, b, c])

    sub_tick = pool.query_pool("substrate", ["from_tickdelta"])
    sub_disc = pool.query_pool("substrate", ["from_disclosure"])
    herb = pool.query_pool("herbivore_broadcast", ["from_technical_herbivore"])
    assert {x.id for x in sub_tick} == {a.id}
    assert {x.id for x in sub_disc} == {b.id}
    assert {x.id for x in herb} == {c.id}


def test_claim_atomicity():
    pool = SubstratePool(":memory:")
    a = _b()
    pool.add_broadcast(a)
    first = pool.claim([a.id], "h1", tick=1)
    second = pool.claim([a.id], "h2", tick=1)
    assert len(first) == 1 and first[0].id == a.id
    assert second == []


def test_query_pool_excludes_abstentions():
    pool = SubstratePool(":memory:")
    real = _b(tier="herbivore_broadcast", diet=("from_technical_herbivore",))
    abst = Broadcast(
        id=str(uuid.uuid4()),
        tier="herbivore_broadcast",
        agent_id="a.x",
        agent_kind="technical",
        diet_tags=["from_technical_herbivore"],
        channel_embedding=[0.0] * 8,
        abstained=True,
    )
    pool.add_broadcast_many([real, abst])
    out = pool.query_pool("herbivore_broadcast", ["from_technical_herbivore"])
    assert {x.id for x in out} == {real.id}
