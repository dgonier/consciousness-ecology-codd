"""EnvironmentStream tests (Mission 02).

Covers:
  - construction smoke
  - deposit marks alive + records source_tag
  - wavelength filter excludes non-matching slots (single tag)
  - wavelength filter with multiple tags (and masked_out_count)
  - full filter == no-mask baseline (via TroughAttention sanity)
  - step_lifecycle delegates and clears source tags on death
  - slot_state surfaces source_tags
"""
from __future__ import annotations

import torch

from trophic.environment_stream import EnvironmentStream, StreamAttendOutput
from trophic.types import RawInput


def _ri(source: str, idx: int) -> RawInput:
    return RawInput(id=f"r{idx}", source=source, payload={"i": idx})


# ---------------------------------------------------------------------------
# construction / membership
# ---------------------------------------------------------------------------

def test_construct_smoke():
    s = EnvironmentStream(hidden_size=64, n_slots=8, n_heads=4, seed=1)
    assert s.n_slots == 8
    assert s.hidden_size == 64
    assert s.n_heads == 4
    assert int(s.alive.sum()) == 0
    state = s.slot_state()
    assert state["n_alive"] == 0
    assert state["source_tags"] == [None] * 8


def test_deposit_marks_alive_and_records_source_tag():
    s = EnvironmentStream(hidden_size=64, n_slots=4, n_heads=4, seed=1)
    emb = torch.randn(64)
    slot = s.deposit(_ri("ohlcv", 0), emb)
    assert 0 <= slot < 4
    state = s.slot_state()
    assert state["source_tags"][slot] == "ohlcv"
    assert state["raw_input_ids"][slot] == "r0"
    assert int(s.alive.sum()) == 1


def test_deposit_multiple_distinct_tags():
    s = EnvironmentStream(hidden_size=32, n_slots=4, n_heads=4, seed=1)
    s.deposit(_ri("ohlcv", 0), torch.randn(32))
    s.deposit(_ri("tweets", 1), torch.randn(32))
    s.deposit(_ri("filing", 2), torch.randn(32))
    tags = s.slot_state()["source_tags"]
    assert "ohlcv" in tags
    assert "tweets" in tags
    assert "filing" in tags
    assert int(s.alive.sum()) == 3


def test_deposit_rejects_wrong_shape_embedding():
    s = EnvironmentStream(hidden_size=64, n_slots=4, n_heads=4, seed=1)
    bad = torch.randn(32)  # wrong size
    try:
        s.deposit(_ri("ohlcv", 0), bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError on wrong-size embedding")


# ---------------------------------------------------------------------------
# wavelength filter
# ---------------------------------------------------------------------------

def test_wavelength_filter_excludes_non_matching_slots():
    """Two slots, one ohlcv one tweets. Filter={'ohlcv'} → tweets weight ~ 0."""
    s = EnvironmentStream(hidden_size=64, n_slots=8, n_heads=4, seed=1)
    s.deposit(_ri("ohlcv", 0), torch.randn(64))
    s.deposit(_ri("tweets", 1), torch.randn(64))
    q = torch.randn(64)
    out = s.attend(q, wavelength_filter={"ohlcv"}, tau=1.0)
    assert isinstance(out, StreamAttendOutput)

    state = s.slot_state()
    tweets_slot = state["source_tags"].index("tweets")
    ohlcv_slot = state["source_tags"].index("ohlcv")

    assert out.per_slot_attention[tweets_slot].abs() < 1e-4, (
        f"tweets slot should be masked, got {float(out.per_slot_attention[tweets_slot])}"
    )
    # ohlcv slot should have meaningful attention (not exactly zero — could
    # be small if null wins, but its softmax slice is in the live set).
    assert out.per_slot_attention[ohlcv_slot] >= 0.0
    assert out.masked_out_count == 1  # tweets slot was excluded


def test_wavelength_filter_with_multiple_tags():
    """Filter {ohlcv, tweets} should pass both, exclude filing."""
    s = EnvironmentStream(hidden_size=64, n_slots=8, n_heads=4, seed=1)
    s.deposit(_ri("ohlcv", 0), torch.randn(64))
    s.deposit(_ri("tweets", 1), torch.randn(64))
    s.deposit(_ri("filing", 2), torch.randn(64))
    q = torch.randn(64)
    out = s.attend(q, wavelength_filter={"ohlcv", "tweets"}, tau=1.0)
    assert out.masked_out_count == 1  # only filing was masked

    state = s.slot_state()
    filing_slot = state["source_tags"].index("filing")
    assert out.per_slot_attention[filing_slot].abs() < 1e-4


def test_attend_full_filter_equivalent_to_no_mask():
    """If wavelength_filter contains every source_tag present, behavior
    matches TroughAttention.attend with no external_bias."""
    s = EnvironmentStream(hidden_size=32, n_slots=4, n_heads=4, seed=42)
    embs = [torch.randn(32, generator=torch.Generator().manual_seed(i)) for i in range(2)]
    s.deposit(_ri("ohlcv", 0), embs[0])
    s.deposit(_ri("tweets", 1), embs[1])
    q = torch.randn(32, generator=torch.Generator().manual_seed(99))

    # Full-filter attend (no slots masked).
    out_full = s.attend(q, wavelength_filter={"ohlcv", "tweets"}, tau=1.0)

    # Build a fresh trough with identical state to compare against the
    # underlying no-mask attend. Easiest way: call the trough directly
    # via a sibling EnvironmentStream that we DON'T attend on first
    # (so its attention buffers stay clean). Then compare per-slot
    # attention values.
    s2 = EnvironmentStream(hidden_size=32, n_slots=4, n_heads=4, seed=42)
    s2.deposit(_ri("ohlcv", 0), embs[0])
    s2.deposit(_ri("tweets", 1), embs[1])
    out_unfiltered = s2._trough.attend(q, tau=1.0)

    # The "full-filter" stream attend should produce the same per-slot
    # weights as the trough's no-bias attend (both run on equivalent state).
    # We compare on alive-slot indices.
    diff = (out_full.per_slot_attention - out_unfiltered.per_slot_attention).abs().max()
    assert float(diff) < 1e-5, f"full-filter attend diverged from no-mask: {float(diff)}"
    assert out_full.masked_out_count == 0


def test_wavelength_filter_empty_set_masks_everything():
    """Empty filter means no slot is in scope → only the null can win."""
    s = EnvironmentStream(hidden_size=64, n_slots=4, n_heads=4, seed=1)
    s.deposit(_ri("ohlcv", 0), torch.randn(64))
    s.deposit(_ri("tweets", 1), torch.randn(64))
    out = s.attend(torch.randn(64), wavelength_filter=set(), tau=1.0)
    # Every alive slot was masked → null prob should dominate (≈ 1.0).
    assert out.null_prob > 0.99
    assert float(out.per_slot_attention.sum()) < 1e-4
    assert out.masked_out_count == 2


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------

def test_step_lifecycle_delegates():
    s = EnvironmentStream(hidden_size=64, n_slots=4, n_heads=4, seed=1)
    for i in range(4):
        s.deposit(_ri("ohlcv", i), torch.randn(64))
    stats = s.step_lifecycle()
    assert isinstance(stats, dict)
    assert "n_killed" in stats
    assert "n_alive" in stats
    assert "killed_ids" in stats


def test_step_lifecycle_clears_source_tag_on_death():
    """When a slot dies (under-attended for long enough), its source tag
    should clear so a future deposit doesn't inherit the stale tag.

    We simulate this by directly killing a slot via the trough's
    bookkeeping (set under_threshold_ticks past patience), then running
    step_lifecycle.
    """
    s = EnvironmentStream(hidden_size=32, n_slots=4, n_heads=4, seed=1)
    s.deposit(_ri("ohlcv", 0), torch.randn(32))
    s.deposit(_ri("tweets", 1), torch.randn(32))

    # Force slot 0 over the death threshold.
    trough = s._trough
    # Mark slot 0 as under-attended past patience.
    trough.under_threshold_ticks[0] = trough.n_patience + 1
    # Disable spawn-respawn so we can observe the killed-tag-cleared state
    # without it being immediately overwritten by a synthetic spawn tag.
    trough.population_strategy = "no_respawn"

    stats = s.step_lifecycle()
    assert 0 in stats["killed_ids"]
    state = s.slot_state()
    # After death (and no respawn), the killed slot's source tag is cleared.
    assert state["source_tags"][0] is None


def test_slot_state_includes_source_tags_field():
    """slot_state must expose a per-slot source_tags list parallel to alive."""
    s = EnvironmentStream(hidden_size=32, n_slots=4, n_heads=4, seed=1)
    s.deposit(_ri("ohlcv", 0), torch.randn(32))
    state = s.slot_state()
    assert "source_tags" in state
    assert len(state["source_tags"]) == s.n_slots
    assert state["source_tags"].count("ohlcv") == 1
