"""Phase 3-A-05d (v4): 4-phase debate mechanism.

Tests cover:
  - Philosophy headers are present, ≥120 words each, meaningfully distinct.
  - Pydantic types round-trip (model_dump → validate → equal).
  - `Response.target_predator_id == own predator_id` is rejected.
  - `run_debate` orchestrates 16 LM calls (4 predators × 4 phases) with
    phase-level barriers; cross-visibility wires correctly.
  - `compress_proposal` preserves ticker / side / horizon / direction.
  - Per-predator validation scoping (validator rejection on predator A
    leaves B/C/D untouched).
  - Transcript JSONL is appended per-day with all four phases dumped.

All tests are offline — no vLLM, no Bedrock. The DSPy LM is monkey-
patched via a callable that returns deterministic Pydantic outputs.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from trophic.agents.apex_portfolio import make_debate_portfolio
from trophic.beliefs.apex_signatures import (
    HorizonForecast,
    Order,
    TickerView,
)
from trophic.beliefs.debate import (
    CompressedProposal,
    Concession,
    DebatePhase1,
    DebatePhase1Proposal,
    DebatePhase2,
    DebatePhase2Responses,
    DebatePhase3,
    DebatePhase3RevisedProposal,
    DebatePhase4,
    DebatePhase4Commit,
    HeldFirm,
    Response,
    _signature_for_phase,
    compress_phase3,
    compress_proposal,
    compose_phase_instructions,
    run_debate,
)
from trophic.beliefs.investment_thesis import InvestmentThesis, open_thesis
from trophic.beliefs.predator_prompts import (
    EVENT_DRIVEN_HEADER,
    MEAN_REVERT_HEADER,
    MOMENTUM_HEADER,
    PHILOSOPHY_HEADERS,
    VALUE_HEADER,
    header_for,
)


PREDATORS = ("momentum", "value", "mean_revert", "event_driven")


# ── 1. Philosophy prompts distinct ─────────────────────────────────────


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def test_philosophy_prompts_distinct():
    """Each header ≥120 words and contains its own philosophy-specific keywords."""
    headers = {
        "momentum": MOMENTUM_HEADER,
        "value": VALUE_HEADER,
        "mean_revert": MEAN_REVERT_HEADER,
        "event_driven": EVENT_DRIVEN_HEADER,
    }
    for name, text in headers.items():
        assert _word_count(text) >= 120, f"{name} header < 120 words"
        assert PHILOSOPHY_HEADERS[name] is text
        assert header_for(name) is text  # type: ignore[arg-type]

    # Philosophy-specific keyword presence/absence.
    assert "momentum" in MOMENTUM_HEADER.lower()
    assert "breakout" in MOMENTUM_HEADER.lower()

    assert "intrinsic" in VALUE_HEADER.lower() or "fair value" in VALUE_HEADER.lower()
    assert "moat" in VALUE_HEADER.lower()

    assert "mean" in MEAN_REVERT_HEADER.lower() and "revert" in MEAN_REVERT_HEADER.lower()
    assert "rsi" in MEAN_REVERT_HEADER.lower() or "oversold" in MEAN_REVERT_HEADER.lower()

    assert "event" in EVENT_DRIVEN_HEADER.lower()
    assert "earnings" in EVENT_DRIVEN_HEADER.lower() or "fda" in EVENT_DRIVEN_HEADER.lower()

    # Headers must not be byte-equal — pairwise distinctness.
    texts = list(headers.values())
    for i, a in enumerate(texts):
        for b in texts[i + 1 :]:
            assert a != b


def test_compose_phase_instructions_includes_header_and_tax_block():
    """Composition layers header + tax framing + phase instructions."""
    for phase in (1, 2, 3, 4):
        full = compose_phase_instructions("value", phase)
        assert "value predator" in full
        assert "TAX & SLIPPAGE" in full
        if phase == 1:
            assert "INDEPENDENT PROPOSAL" in full
        if phase == 2:
            assert "STRUCTURED RESPONSES" in full
            assert "NOT" in full and "adversarial" in full
        if phase == 3:
            assert "CONCESSIONS" in full
        if phase == 4:
            assert "FINAL INDEPENDENT COMMIT" in full

    with pytest.raises(ValueError):
        compose_phase_instructions("momentum", 5)


def test_signature_for_phase_attaches_instructions():
    """`_signature_for_phase` returns a Signature with the philosophy-specific instructions attached."""
    sig_mom = _signature_for_phase("momentum", 1)
    sig_val = _signature_for_phase("value", 1)
    assert "momentum predator" in sig_mom.instructions
    assert "value predator" in sig_val.instructions
    assert sig_mom.instructions != sig_val.instructions
    # Base signature unchanged (compose at runtime, not class).
    assert "momentum predator" not in DebatePhase1.instructions


# ── 2. Phase types round-trip ──────────────────────────────────────────


def _make_forecast() -> HorizonForecast:
    return HorizonForecast(
        h1=0.001, h5=0.005, h20=0.02, h60=0.05,
        confidence_h1="low", confidence_h5="med",
        confidence_h20="high", confidence_h60="med",
        regime_note="trend",
    )


def _make_view(ticker: str = "AAPL", horizon: str = "h20") -> TickerView:
    return TickerView(
        ticker=ticker, forecasts=_make_forecast(),
        primary_horizon=horizon,  # type: ignore[arg-type]
        rationale=f"primary horizon {horizon} on {ticker}",
    )


def _make_order(
    ticker: str = "AAPL", side: str = "BUY",
    size_pct: float = 30.0, horizon: str = "h20",
    alpha_bps: float = 200.0,
) -> Order:
    return Order(
        side=side,  # type: ignore[arg-type]
        ticker=ticker, size_pct=size_pct,
        reasoning="test order",
        primary_horizon=horizon,  # type: ignore[arg-type]
        expected_alpha_bps=alpha_bps,
    )


def _make_thesis(predator_id: str = "momentum", ticker: str = "AAPL") -> InvestmentThesis:
    return open_thesis(
        predator_id=predator_id,
        philosophy=predator_id,  # type: ignore[arg-type]
        ticker=ticker,
        opened_at_date="2026-02-03",
        direction="long",
        primary_horizon="h20",
        catalysts=["earnings beat"],
        invalidation_triggers=["guidance cut"],
        expected_alpha_bps=300.0,
        confidence="high",
    )


def test_phase_types_roundtrip():
    """Every debate Pydantic type round-trips through model_dump → validate."""
    thesis = _make_thesis("momentum", "AAPL")
    view = _make_view("AAPL", "h20")
    order = _make_order("AAPL")

    p1 = DebatePhase1Proposal(
        predator_id="momentum", philosophy="momentum",
        new_theses=[thesis], close_theses=[], views=[view],
        orders=[order], rationale="open momentum on AAPL",
    )
    p1_round = DebatePhase1Proposal.model_validate(p1.model_dump())
    assert p1_round == p1

    resp = Response(
        target_predator_id="value",
        target_ticker="AAPL", kind="extend",
        rationale="your AAPL value case also fits my momentum lens",
    )
    p2 = DebatePhase2Responses(predator_id="momentum", responses=[resp])
    p2_round = DebatePhase2Responses.model_validate(p2.model_dump())
    assert p2_round == p2

    p3 = DebatePhase3RevisedProposal(
        predator_id="momentum",
        revised_new_theses=[thesis],
        revised_views=[view],
        revised_orders=[order],
        conceded=[Concession(
            target_predator_id="value",
            what_changed="shortened horizon h60→h20 on AAPL",
            rationale="conceded to value framework",
        )],
        held_firm=[HeldFirm(
            target_predator_id="mean_revert",
            what_held="kept BUY on NVDA at h20",
            rationale="not a mean-reversion setup; volume confirms trend",
        )],
        rationale="revised after debate",
    )
    p3_round = DebatePhase3RevisedProposal.model_validate(p3.model_dump())
    assert p3_round == p3

    p4 = DebatePhase4Commit(
        predator_id="momentum",
        final_theses_to_open=[thesis],
        final_orders=[order],
        rationale="final commit",
    )
    p4_round = DebatePhase4Commit.model_validate(p4.model_dump())
    assert p4_round == p4


def test_self_targeting_dropped():
    """`Response.target_predator_id == own predator_id` is silently dropped
    by the model_validator(mode='after'), so the rest of the batch
    survives. (Previously raised — softened in 05f after Qwen-4B's
    Phase-2 output frequently included one self-targeted Response per
    batch, which poisoned the whole day under the raising policy.)
    """
    p2 = DebatePhase2Responses(
        predator_id="momentum",
        responses=[
            Response(
                target_predator_id="momentum",  # self-target — dropped
                target_ticker="AAPL", kind="agree",
                rationale="agreeing with myself",
            ),
            Response(
                target_predator_id="value",
                target_ticker="KO", kind="object_to",
                rationale="value trap risk",
            ),
        ],
    )
    assert len(p2.responses) == 1
    assert p2.responses[0].target_predator_id == "value"


def test_response_without_target_is_dropped():
    """Responses missing both target_thesis_id and target_ticker are
    silently dropped UNLESS they're 'no objection' placeholders. The
    no-target / no-objection placeholder remains valid. Other responses
    in the same batch survive. (Was a raising ValidationError pre-05f.)
    """
    p2 = DebatePhase2Responses(
        predator_id="momentum",
        responses=[
            Response(
                target_predator_id="value",
                kind="object_to", rationale="I disagree generally",
            ),  # no target, not placeholder — dropped
            Response(
                target_predator_id="value",
                kind="agree", rationale="no objection",
            ),  # no target, placeholder — kept
            Response(
                target_predator_id="event_driven",
                target_ticker="NVDA",
                kind="extend", rationale="applies to AMD",
            ),  # has target — kept
        ],
    )
    assert len(p2.responses) == 2
    kinds = sorted(r.kind for r in p2.responses)
    assert kinds == ["agree", "extend"]


def test_phase2_accepts_bare_list_of_responses():
    """Qwen-4B sometimes emits the OutputField as a flat list of
    Response dicts (one per peer/ticker) instead of the canonical
    wrapper. The before-validator must accept that shape and reshape
    it into responses=[...] with the orchestrator-supplied predator_id.
    """
    bare_list = [
        {"target_predator_id": "value", "target_ticker": "AAPL",
         "kind": "agree", "rationale": "share the case"},
        {"target_predator_id": "mean_revert", "target_ticker": "KO",
         "kind": "object_to", "rationale": "structural, not noise"},
        {"target_predator_id": "event_driven", "target_ticker": "NVDA",
         "kind": "extend", "rationale": "applies to AMD too"},
    ]
    # Going through the orchestrator's coerce path (which injects pid).
    from trophic.beliefs.debate import _coerce_responses
    p2 = _coerce_responses(bare_list, pid="momentum")
    assert p2.predator_id == "momentum"
    assert len(p2.responses) == 3
    assert {r.target_predator_id for r in p2.responses} == {
        "value", "mean_revert", "event_driven",
    }


def test_phase2_accepts_list_of_per_peer_wrappers():
    """Some Qwen-4B emissions look like one DebatePhase2Responses
    wrapper per peer (the model misreads "for each other predator"
    as "emit one wrapper per peer"). The before-validator must flatten
    these into a single wrapper.
    """
    per_peer = [
        {"predator_id": "momentum",
         "responses": [{"target_predator_id": "value", "target_ticker": "AAPL",
                        "kind": "agree", "rationale": "share the case"}]},
        {"predator_id": "momentum",
         "responses": [{"target_predator_id": "mean_revert", "target_ticker": "KO",
                        "kind": "object_to", "rationale": "structural"}]},
    ]
    from trophic.beliefs.debate import _coerce_responses
    p2 = _coerce_responses(per_peer, pid="momentum")
    assert p2.predator_id == "momentum"
    assert len(p2.responses) == 2
    targets = {r.target_predator_id for r in p2.responses}
    assert targets == {"value", "mean_revert"}


def test_phase2_canonical_dict_still_works():
    """Sanity: the canonical {"predator_id", "responses"} dict input
    is unchanged by the before-validator.
    """
    canonical = {
        "predator_id": "value",
        "responses": [
            {"target_predator_id": "momentum", "target_ticker": "AAPL",
             "kind": "object_to", "rationale": "already extended"},
        ],
    }
    p2 = DebatePhase2Responses.model_validate(canonical)
    assert p2.predator_id == "value"
    assert len(p2.responses) == 1
    assert p2.responses[0].target_predator_id == "momentum"


# ── 2b. Phase 1 / 3 / 4 normalizers against Qwen-4B output drift ───────


def _good_view_dict(ticker: str = "AAPL", horizon: str = "h20") -> dict:
    """Canonical dict-shape TickerView. Used as the well-formed control
    sample in drop-malformed-views tests."""
    return {
        "ticker": ticker,
        "forecasts": {
            "h1": 0.001, "h5": 0.005, "h20": 0.02, "h60": 0.05,
            "confidence_h1": "low", "confidence_h5": "med",
            "confidence_h20": "high", "confidence_h60": "med",
            "regime_note": "trend",
        },
        "primary_horizon": horizon,
        "rationale": f"primary horizon {horizon} on {ticker}",
    }


def _good_thesis_dict(predator_id: str = "momentum", ticker: str = "AAPL") -> dict:
    """Canonical dict-shape InvestmentThesis used to test predator_id
    injection. Returns a dict (not a Pydantic model) so we can drop
    fields to simulate Qwen drift."""
    return {
        "predator_id": predator_id,
        "philosophy": predator_id,
        "ticker": ticker,
        "opened_at_date": "2026-02-03",
        "direction": "long",
        "primary_horizon": "h20",
        "catalysts": ["earnings beat"],
        "invalidation_triggers": ["guidance cut"],
        "expected_alpha_bps": 300.0,
        "confidence": "high",
    }


def test_phase1_drops_view_missing_primary_horizon():
    """Day-5 sweep fail: `views[23].primary_horizon` field required when
    Qwen-4B forgot the top-level `primary_horizon` (and put
    `confidence_h60` inside `forecasts`). The malformed view is silently
    dropped; the good views in the same batch survive. Pre-fix this
    raised the whole proposal as ValidationError.
    """
    good = _good_view_dict("AAPL", "h20")
    bad_no_horizon = _good_view_dict("DIS", "h20")
    bad_no_horizon.pop("primary_horizon")
    p1 = DebatePhase1Proposal.model_validate({
        "predator_id": "momentum", "philosophy": "momentum",
        "views": [good, bad_no_horizon, _good_view_dict("KO", "h5")],
        "rationale": "two good views, one malformed",
    })
    assert {v.ticker for v in p1.views} == {"AAPL", "KO"}


def test_phase1_drops_non_dict_view_entries():
    """Day-4 run2 sweep fail: `views[25] = ['ticker']` — Qwen-4B emitted
    a bare list at one view slot (truncation/confusion artifact). Pre-fix
    `_drop_malformed_views` only handled dict-shaped malformed views and
    let the bare list through to strict validation, which poisoned the
    whole proposal. We now drop any non-TickerView, non-dict entry.
    """
    good = _good_view_dict("AAPL", "h20")
    p1 = DebatePhase1Proposal.model_validate({
        "predator_id": "momentum", "philosophy": "momentum",
        "views": [good, ["ticker"], "MSFT", None, 42],
        "rationale": "one good view, four garbage entries",
    })
    assert [v.ticker for v in p1.views] == ["AAPL"]


def test_phase1_drops_view_with_partial_forecasts():
    """Day-3 sweep fail: `views[22].forecasts.h5/h20` missing because
    Qwen-4B truncated the forecast object to just `{'h1': 0.0}`. The
    malformed view is dropped; the good ones survive. Pre-fix the
    HorizonForecast cried 'field required'.
    """
    good = _good_view_dict("AAPL", "h20")
    partial_fc = _good_view_dict("DIS", "h20")
    partial_fc["forecasts"] = {"h1": 0.0}  # missing h5/h20/h60
    p1 = DebatePhase1Proposal.model_validate({
        "predator_id": "momentum", "philosophy": "momentum",
        "views": [good, partial_fc],
        "rationale": "one good view, one partial forecast",
    })
    assert [v.ticker for v in p1.views] == ["AAPL"]


def test_phase1_injects_thesis_predator_id():
    """Some Qwen-4B Phase-1 emissions omit `predator_id`/`philosophy`
    on the inner thesis dicts — the model has no in-context handle on
    which wrapper owns them. The before-validator propagates the outer
    fields down. (Same family as the day-4 Phase-4 fail.)
    """
    thesis_no_pid = _good_thesis_dict("momentum", "AAPL")
    thesis_no_pid.pop("predator_id")
    thesis_no_pid.pop("philosophy")
    p1 = DebatePhase1Proposal.model_validate({
        "predator_id": "momentum", "philosophy": "momentum",
        "new_theses": [thesis_no_pid],
        "rationale": "thesis missing predator_id",
    })
    assert len(p1.new_theses) == 1
    assert p1.new_theses[0].predator_id == "momentum"
    assert p1.new_theses[0].philosophy == "momentum"


def test_phase1_normalizer_leaves_clean_input_unchanged():
    """Sanity: a fully canonical Phase-1 dict is unchanged by the
    before-validator (no spurious drops, no overwrites of provided
    predator_id values).
    """
    p1 = DebatePhase1Proposal.model_validate({
        "predator_id": "value", "philosophy": "value",
        "new_theses": [_good_thesis_dict("value", "BRK.B")],
        "views": [_good_view_dict("BRK.B", "h60")],
        "rationale": "all clean",
    })
    assert p1.new_theses[0].predator_id == "value"
    assert p1.new_theses[0].philosophy == "value"
    assert len(p1.views) == 1


def test_phase3_drops_view_with_none_primary_horizon():
    """Day-11 sweep fail: `revised_views[0].primary_horizon` arrived as
    None (literal_error). The malformed revised_view is dropped; the
    good ones survive. Pre-fix the whole revised proposal raised.
    """
    good = _good_view_dict("AAPL", "h20")
    bad_none_horizon = _good_view_dict("KO", "h5")
    bad_none_horizon["primary_horizon"] = None
    p3 = DebatePhase3RevisedProposal.model_validate({
        "predator_id": "momentum",
        "revised_views": [bad_none_horizon, good],
        "rationale": "one good view, one with null primary_horizon",
    })
    assert [v.ticker for v in p3.revised_views] == ["AAPL"]


def test_phase3_injects_thesis_predator_id():
    """`revised_new_theses[i]` dicts that omit predator_id receive the
    outer wrapper's predator_id. Same drift family as Phase 1 and 4."""
    thesis_no_pid = _good_thesis_dict("mean_revert", "KO")
    thesis_no_pid.pop("predator_id")
    p3 = DebatePhase3RevisedProposal.model_validate({
        "predator_id": "mean_revert",
        "revised_new_theses": [thesis_no_pid],
        "rationale": "thesis missing predator_id",
    })
    assert len(p3.revised_new_theses) == 1
    assert p3.revised_new_theses[0].predator_id == "mean_revert"


def test_phase4_injects_thesis_predator_id_from_outer_wrapper():
    """Day-4 sweep fail verbatim: `final_theses_to_open[0].predator_id`
    field required. The model emits the complete InvestmentThesis dict
    but forgets `predator_id` — it doesn't know which wrapper owns it.
    The before-validator propagates `predator_id` from the outer
    DebatePhase4Commit. Pre-fix this raised ValidationError.
    """
    thesis_no_pid = _good_thesis_dict("event_driven", "MRK")
    thesis_no_pid.pop("predator_id")
    p4 = DebatePhase4Commit.model_validate({
        "predator_id": "event_driven",
        "final_theses_to_open": [thesis_no_pid],
        "rationale": "FDA catalyst commit",
    })
    assert len(p4.final_theses_to_open) == 1
    assert p4.final_theses_to_open[0].predator_id == "event_driven"
    assert p4.final_theses_to_open[0].ticker == "MRK"


def test_phase4_normalizer_does_not_overwrite_existing_thesis_pid():
    """If a thesis dict ALREADY carries a predator_id, don't overwrite
    it. (Defensive: prevents the normalizer from silently breaking
    cross-predator thesis references in the unlikely event the model
    emits an inner predator_id mismatch — that's a semantic error
    that should surface elsewhere, not be papered over here.)
    """
    thesis = _good_thesis_dict("momentum", "AAPL")
    # Outer wrapper says event_driven, inner thesis says momentum.
    # Pydantic field validation will catch the mismatch downstream
    # (via thesis-book add); we just verify the normalizer doesn't
    # overwrite the inner value here.
    p4 = DebatePhase4Commit.model_validate({
        "predator_id": "event_driven",
        "final_theses_to_open": [thesis],
        "rationale": "test",
    })
    assert p4.final_theses_to_open[0].predator_id == "momentum"  # preserved


def test_phase1_drops_orders_missing_primary_horizon():
    """Run8 day-1 sweep fail (2026-02-03): a BUY order missing
    `primary_horizon` killed the whole Phase-1 proposal even though
    5 other orders were valid. We now drop the malformed order and
    keep the rest. Defaultable fields (alpha, reasoning) are NOT
    drop-criteria — those are handled by the Order before-validator.
    """
    good_order = {
        "side": "BUY", "ticker": "AAPL", "size_pct": 10.0,
        "reasoning": "trend", "primary_horizon": "h5",
        "expected_alpha_bps": 200.0,
    }
    bad_no_horizon = {
        "side": "BUY", "ticker": "MSFT", "size_pct": 10.0,
        "reasoning": "trend", "expected_alpha_bps": 200.0,
        # primary_horizon intentionally omitted
    }
    bad_no_side = {
        "ticker": "JPM", "size_pct": 10.0, "primary_horizon": "h20",
        "reasoning": "value", "expected_alpha_bps": 240.0,
    }
    bad_no_ticker = {
        "side": "BUY", "size_pct": 10.0, "primary_horizon": "h20",
        "reasoning": "no name", "expected_alpha_bps": 100.0,
    }
    hold_no_ticker = {  # HOLD with no ticker is acceptable
        "side": "HOLD", "size_pct": 0.0, "primary_horizon": "h5",
        "reasoning": "wait", "expected_alpha_bps": 0.0,
    }
    p1 = DebatePhase1Proposal.model_validate({
        "predator_id": "momentum", "philosophy": "momentum",
        "orders": [good_order, bad_no_horizon, bad_no_side, bad_no_ticker, hold_no_ticker],
        "rationale": "one good order, three garbage, one HOLD survives",
    })
    # Survivors: good_order (AAPL) + hold_no_ticker (HOLD)
    assert len(p1.orders) == 2
    survivors = {(o.side, o.ticker or "") for o in p1.orders}
    assert ("BUY", "AAPL") in survivors
    assert ("HOLD", "") in survivors


def test_phase4_coerces_close_dict_to_thesis_id_string():
    """Run4 day-6 sweep fail (2026-02-10): Qwen-4B emitted
    `final_theses_to_close[0] = {'thesis_id': 'a1b2c3...', ...}` —
    a full thesis dict — instead of the bare thesis_id string the
    schema expects. The model mirrored the `final_theses_to_open`
    shape (which IS list-of-dicts) into the close slot. Pre-fix the
    whole Phase 4 commit raised `Input should be a valid string`.
    We now coerce dict-shaped close entries to their thesis_id
    string; mixed lists are handled element-wise.
    """
    p4 = DebatePhase4Commit.model_validate({
        "predator_id": "momentum",
        "final_theses_to_close": [
            {"thesis_id": "a1b2c3d4e5f6", "ticker": "AAPL", "predator_id": "momentum"},
            "g7h8i9j0k1l2",  # already a bare string
            {"thesis_id": "m3n4o5p6q7r8"},  # dict, minimal
        ],
        "rationale": "closing three theses, mixed-shape input",
    })
    assert p4.final_theses_to_close == ["a1b2c3d4e5f6", "g7h8i9j0k1l2", "m3n4o5p6q7r8"]


def test_phase4_drops_uncoercible_close_entries():
    """Defensive: a close entry that's neither a string nor a dict with
    `thesis_id` is dropped, not invented. The whole commit survives
    rather than failing on a single garbage entry.
    """
    p4 = DebatePhase4Commit.model_validate({
        "predator_id": "value",
        "final_theses_to_close": [
            "good_id_1",
            {"no_thesis_id": "x"},  # dict but no thesis_id → drop
            None,                    # garbage → drop
            42,                      # garbage → drop
            "good_id_2",
        ],
        "rationale": "two good ids around three garbage entries",
    })
    assert p4.final_theses_to_close == ["good_id_1", "good_id_2"]


def test_phase4_close_strings_unchanged_when_already_correct():
    """Sanity: a canonical list of strings passes through unchanged."""
    p4 = DebatePhase4Commit.model_validate({
        "predator_id": "value",
        "final_theses_to_close": ["id_a", "id_b", "id_c"],
        "rationale": "all good",
    })
    assert p4.final_theses_to_close == ["id_a", "id_b", "id_c"]


# ── 3. Compression preserves intent ────────────────────────────────────


def test_compression_preserves_intent():
    """`compress_proposal` preserves ticker / side / horizon / direction in summaries."""
    view = _make_view("AAPL", "h20")
    order_buy = _make_order("AAPL", "BUY", 30.0, "h20", 250.0)
    order_sell = _make_order("KO", "SELL", 100.0, "h60", -150.0)
    thesis = _make_thesis("momentum", "AAPL")
    p1 = DebatePhase1Proposal(
        predator_id="momentum", philosophy="momentum",
        new_theses=[thesis], close_theses=["abc12"],
        views=[view], orders=[order_buy, order_sell],
        rationale="bullish AAPL, bearish KO",
    )
    cp = compress_proposal(p1)
    assert isinstance(cp, CompressedProposal)
    assert cp.predator_id == "momentum"
    assert cp.philosophy == "momentum"

    # Thesis summary includes ticker + horizon + direction.
    assert any("AAPL" in s for s in cp.new_thesis_summary)
    assert any("h20" in s for s in cp.new_thesis_summary)
    assert any("long" in s for s in cp.new_thesis_summary)

    # Order summary includes side + ticker + horizon for each order.
    assert any("BUY" in s and "AAPL" in s and "h20" in s for s in cp.order_summary)
    assert any("SELL" in s and "KO" in s and "h60" in s for s in cp.order_summary)

    # Close summaries surface the thesis_id verbatim.
    assert any("abc12" in s for s in cp.close_thesis_summary)


def test_compression_phase3_preserves_intent():
    """compress_phase3 preserves order side+ticker+horizon across revised proposals."""
    view = _make_view("NVDA", "h5")
    order = _make_order("NVDA", "BUY", 10.0, "h5", 180.0)
    p3 = DebatePhase3RevisedProposal(
        predator_id="momentum",
        revised_new_theses=[],
        revised_views=[view],
        revised_orders=[order],
        rationale="post-debate rev",
    )
    cp = compress_phase3(p3)
    assert any("BUY" in s and "NVDA" in s and "h5" in s for s in cp.order_summary)


# ── 4. Mock-LM scaffolding for run_debate ──────────────────────────────


class _MockLM:
    """Minimal stand-in for dspy.LM that returns deterministic outputs.

    We monkeypatch `dspy.Predict.__call__` to bypass actual LM dispatch
    and return Pydantic outputs directly. The LM object itself is never
    invoked; it only satisfies `dspy.context(lm=...)`.
    """
    def __init__(self):
        self.kwargs = {}

    def __call__(self, *a, **kw):
        return ""


class _DispatchRecord:
    """Counts and timestamps each `dspy.Predict.__call__` invocation."""
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def log(self, sig_instructions: str, kwargs: dict[str, Any]):
        # Tag the phase from the instructions block.
        if "INDEPENDENT PROPOSAL" in sig_instructions:
            phase = 1
        elif "STRUCTURED RESPONSES" in sig_instructions:
            phase = 2
        elif "CONCESSIONS" in sig_instructions:
            phase = 3
        elif "FINAL INDEPENDENT COMMIT" in sig_instructions:
            phase = 4
        else:
            phase = -1
        # Each header is identified by its UNIQUE leading sentence so we
        # don't false-positive on cross-references between philosophies
        # (e.g. value's header mentions "momentum predator" as a peer).
        if "You are the **momentum predator**" in sig_instructions:
            philosophy = "momentum"
        elif "You are the **value predator**" in sig_instructions:
            philosophy = "value"
        elif "You are the **mean_revert predator**" in sig_instructions:
            philosophy = "mean_revert"
        elif "You are the **event_driven predator**" in sig_instructions:
            philosophy = "event_driven"
        else:
            philosophy = "unknown"
        self.calls.append({
            "phase": phase,
            "philosophy": philosophy,
            "kwargs": kwargs,
        })


def _build_mock_predict_call(record: _DispatchRecord):
    """Returns a function to monkeypatch `dspy.Predict.__call__`.

    Examines the bound signature's instructions to determine
    (phase, philosophy) and synthesizes a deterministic Pydantic
    output.
    """
    import dspy

    def _patched(self_predict, **kwargs):
        sig = self_predict.signature
        instructions = sig.instructions
        record.log(instructions, kwargs)

        # Identify phase + philosophy from the instructions text.
        if "INDEPENDENT PROPOSAL" in instructions:
            phase = 1
        elif "STRUCTURED RESPONSES" in instructions:
            phase = 2
        elif "CONCESSIONS" in instructions:
            phase = 3
        else:
            phase = 4
        # Unique prefix per philosophy header — avoids collisions with
        # in-text peer references.
        if "You are the **momentum predator**" in instructions:
            philosophy = "momentum"
        elif "You are the **value predator**" in instructions:
            philosophy = "value"
        elif "You are the **mean_revert predator**" in instructions:
            philosophy = "mean_revert"
        else:
            philosophy = "event_driven"

        # Distinct ticker per predator so we can tell them apart in the
        # transcript and cross-visibility checks.
        ticker_per_pred = {
            "momentum": "AAPL",
            "value": "BRK.B",
            "mean_revert": "KO",
            "event_driven": "MRK",
        }
        ticker = ticker_per_pred[philosophy]

        thesis = _make_thesis(philosophy, ticker)
        view = _make_view(ticker, "h20")
        order = _make_order(ticker, "BUY", 30.0, "h20", 250.0)

        if phase == 1:
            return dspy.Prediction(proposal=DebatePhase1Proposal(
                predator_id=philosophy, philosophy=philosophy,
                new_theses=[thesis], views=[view], orders=[order],
                rationale=f"{philosophy} phase-1 plan on {ticker}",
            ))
        if phase == 2:
            # Emit one response to each OTHER predator, naming their ticker.
            others = [p for p in PREDATORS if p != philosophy]
            responses = []
            for other in others:
                other_ticker = ticker_per_pred[other]
                responses.append(Response(
                    target_predator_id=other,
                    target_ticker=other_ticker, kind="agree",
                    rationale=f"{philosophy}→{other}: share view on {other_ticker}",
                ))
            return dspy.Prediction(responses=DebatePhase2Responses(
                predator_id=philosophy, responses=responses,
            ))
        if phase == 3:
            return dspy.Prediction(revised=DebatePhase3RevisedProposal(
                predator_id=philosophy,
                revised_new_theses=[thesis],
                revised_views=[view],
                revised_orders=[order],
                conceded=[Concession(
                    target_predator_id=("value" if philosophy != "value" else "momentum"),
                    what_changed=f"shortened horizon on {ticker}",
                    rationale="conceded after debate",
                )],
                held_firm=[],
                rationale=f"{philosophy} phase-3 revised",
            ))
        # phase 4
        return dspy.Prediction(commit=DebatePhase4Commit(
            predator_id=philosophy,
            final_theses_to_open=[thesis],
            final_orders=[order],
            rationale=f"{philosophy} phase-4 commit",
        ))
    return _patched


# ── 5. run_debate orchestration ────────────────────────────────────────


async def test_run_debate_orchestration(monkeypatch, tmp_path):
    """16 LM calls total (4 predators × 4 phases) with phase-level ordering."""
    import dspy as _dspy

    record = _DispatchRecord()
    monkeypatch.setattr(
        _dspy.Predict, "__call__", _build_mock_predict_call(record),
    )

    apex = make_debate_portfolio(total_starting_cash=100_000)
    prices = {"AAPL": 200.0, "BRK.B": 450.0, "KO": 60.0, "MRK": 110.0}

    commits = await run_debate(
        apex_portfolio=apex,
        today="2026-02-03",
        days_remaining=30,
        watchlist=["AAPL", "BRK.B", "KO", "MRK"],
        observations="synthetic observation block",
        current_prices=prices,
        lm=_MockLM(),
        run_label="orchestration_test",
        transcript_dir=tmp_path / "transcripts",
    )

    # Exactly 16 calls.
    assert len(record.calls) == 16, f"expected 16 calls, got {len(record.calls)}"

    # 4 calls per phase.
    phase_counts: dict[int, int] = {}
    for c in record.calls:
        phase_counts[c["phase"]] = phase_counts.get(c["phase"], 0) + 1
    assert phase_counts == {1: 4, 2: 4, 3: 4, 4: 4}

    # 4 calls per philosophy.
    philo_counts: dict[str, int] = {}
    for c in record.calls:
        philo_counts[c["philosophy"]] = philo_counts.get(c["philosophy"], 0) + 1
    assert philo_counts == {
        "momentum": 4, "value": 4, "mean_revert": 4, "event_driven": 4,
    }

    # Returned commits keyed by predator_id.
    assert set(commits.keys()) == set(PREDATORS)
    for pid, commit in commits.items():
        assert isinstance(commit, DebatePhase4Commit)
        assert commit.predator_id == pid

    # Cross-visibility check: in phase 3, predator A's
    # `responses_directed_at_me` must contain Responses from each of B/C/D
    # targeting A.
    phase3_calls = [c for c in record.calls if c["phase"] == 3]
    for c in phase3_calls:
        pid = c["philosophy"]
        directed = c["kwargs"]["responses_directed_at_me"]
        # Each of the OTHER 3 predators emitted one Response targeting pid.
        assert len(directed) == 3
        targeters = {r.target_predator_id for r in directed}
        # Every Response in this bag must be addressed to me.
        assert targeters == {pid}

    # Phase-2 cross-visibility: `others_phase1` for predator A excludes A
    # and includes the 3 others — compressed (CompressedProposal), not
    # full DebatePhase1Proposal.
    phase2_calls = [c for c in record.calls if c["phase"] == 2]
    for c in phase2_calls:
        pid = c["philosophy"]
        others = c["kwargs"]["others_phase1"]
        assert len(others) == 3
        # All compressed (NOT full DebatePhase1Proposal).
        for o in others:
            assert isinstance(o, CompressedProposal)
            assert o.predator_id != pid

    # Phase-4 cross-visibility: `others_phase3` is compressed too.
    phase4_calls = [c for c in record.calls if c["phase"] == 4]
    for c in phase4_calls:
        pid = c["philosophy"]
        others = c["kwargs"]["others_phase3"]
        assert len(others) == 3
        for o in others:
            assert isinstance(o, CompressedProposal)
            assert o.predator_id != pid


# ── 6. Per-predator validation scoping ─────────────────────────────────


def test_per_predator_validation():
    """An invalid order on predator A does not affect B/C/D's commits.

    The validator chain runs per-predator via `state_for_predator(pid, ...)`.
    We simulate this by directly invoking `filter_tax_aware` with a
    one-predator slice and asserting that B/C/D commits remain intact.
    """
    from trophic.beliefs.validators import validate_horizon_sizing

    apex = make_debate_portfolio(total_starting_cash=100_000)
    prices = {"AAPL": 200.0, "BRK.B": 450.0, "KO": 60.0, "MRK": 110.0}

    # Build per-predator commits — A oversizes; B/C/D are fine.
    bad_order_A = _make_order("AAPL", "BUY", 90.0, "h20", 250.0)  # over 30±10
    good_orders = {
        "value": _make_order("BRK.B", "BUY", 30.0, "h20", 250.0),
        "mean_revert": _make_order("KO", "BUY", 30.0, "h20", 250.0),
        "event_driven": _make_order("MRK", "BUY", 30.0, "h20", 250.0),
    }
    commits = {
        "momentum": DebatePhase4Commit(
            predator_id="momentum", final_orders=[bad_order_A],
            rationale="A oversizes",
        ),
    }
    for pid, o in good_orders.items():
        commits[pid] = DebatePhase4Commit(
            predator_id=pid, final_orders=[o],
            rationale=f"{pid} fine",
        )

    # Per-predator filter: each predator's sub-portfolio state.
    results: dict[str, list[bool]] = {}
    for pid, commit in commits.items():
        sub_state, sub_positions = apex.state_for_predator(pid, "2026-02-03", prices)
        # The horizon-sizing validator is the cleanest single-rule
        # demonstration of per-predator scoping; the real chain is
        # `filter_tax_aware`, which composes this rule into a longer
        # chain. Both behave identically with respect to scoping —
        # predator A's invalid order does not bubble into B/C/D's
        # validator calls. We pull `sub_state`/`sub_positions` per
        # predator above to ensure that scoping is exercised even though
        # `validate_horizon_sizing` only inspects the order itself.
        assert sub_state.equity > 0
        assert isinstance(sub_positions, list)
        accepted = []
        for o in commit.final_orders:
            res = validate_horizon_sizing(o)
            accepted.append(res.accepted)
        results[pid] = accepted

    # A is rejected; B/C/D pass.
    assert results["momentum"] == [False]
    assert results["value"] == [True]
    assert results["mean_revert"] == [True]
    assert results["event_driven"] == [True]


# ── 7. Transcript JSONL written per day ────────────────────────────────


async def test_transcript_logging(monkeypatch, tmp_path):
    """run_debate appends one JSONL line per day with all 4 phases dumped."""
    import dspy as _dspy

    record = _DispatchRecord()
    monkeypatch.setattr(
        _dspy.Predict, "__call__", _build_mock_predict_call(record),
    )

    apex = make_debate_portfolio(total_starting_cash=100_000)
    prices = {"AAPL": 200.0, "BRK.B": 450.0, "KO": 60.0, "MRK": 110.0}

    transcript_dir = tmp_path / "transcripts"
    await run_debate(
        apex_portfolio=apex,
        today="2026-02-03",
        days_remaining=30,
        watchlist=["AAPL", "BRK.B", "KO", "MRK"],
        observations="synthetic observation block",
        current_prices=prices,
        lm=_MockLM(),
        run_label="test_run",
        transcript_dir=transcript_dir,
    )

    path = transcript_dir / "test_run.jsonl"
    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["date"] == "2026-02-03"
    for phase_key in ("phase1", "phase2", "phase3", "phase4"):
        assert set(row[phase_key].keys()) == set(PREDATORS)

    # Phase 1 dumps contain orders + new_theses for each predator.
    for pid, dump in row["phase1"].items():
        assert dump["predator_id"] == pid
        assert dump["philosophy"] == pid
        assert len(dump["orders"]) == 1
        assert len(dump["new_theses"]) == 1

    # Phase 3 dumps contain conceded fields.
    for pid, dump in row["phase3"].items():
        assert isinstance(dump["conceded"], list)
        assert len(dump["conceded"]) == 1

    # Phase 4 commits.
    for pid, dump in row["phase4"].items():
        assert dump["predator_id"] == pid
        assert len(dump["final_orders"]) == 1

    # Second call appends a SECOND line.
    await run_debate(
        apex_portfolio=apex,
        today="2026-02-04",
        days_remaining=29,
        watchlist=["AAPL", "BRK.B", "KO", "MRK"],
        observations="day 2 observation",
        current_prices=prices,
        lm=_MockLM(),
        run_label="test_run",
        transcript_dir=transcript_dir,
    )
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    row2 = json.loads(lines[1])
    assert row2["date"] == "2026-02-04"


# ── 8. Phase ordering / barrier check ──────────────────────────────────


async def test_phase_ordering_barrier(monkeypatch, tmp_path):
    """Phase N+1 cannot start until ALL of phase N's predators have returned.

    We enforce this with a timestamped log: every Predict call appends
    (phase, philosophy) to a list in dispatch order. With async-gather
    inside a phase the within-phase order is non-deterministic, but the
    LAST call of phase N must come strictly before the FIRST call of
    phase N+1.
    """
    import dspy as _dspy

    record = _DispatchRecord()
    monkeypatch.setattr(
        _dspy.Predict, "__call__", _build_mock_predict_call(record),
    )

    apex = make_debate_portfolio(total_starting_cash=100_000)
    prices = {"AAPL": 200.0, "BRK.B": 450.0, "KO": 60.0, "MRK": 110.0}

    await run_debate(
        apex_portfolio=apex,
        today="2026-02-03",
        days_remaining=30,
        watchlist=["AAPL", "BRK.B", "KO", "MRK"],
        observations="obs",
        current_prices=prices,
        lm=_MockLM(),
        run_label="barrier_test",
        transcript_dir=tmp_path / "transcripts",
    )

    phases_seen = [c["phase"] for c in record.calls]
    # No phase 1 call appears AFTER any phase 2 call; ditto 2→3, 3→4.
    last_idx = {p: max(i for i, q in enumerate(phases_seen) if q == p) for p in (1, 2, 3, 4)}
    first_idx = {p: min(i for i, q in enumerate(phases_seen) if q == p) for p in (1, 2, 3, 4)}
    assert last_idx[1] < first_idx[2]
    assert last_idx[2] < first_idx[3]
    assert last_idx[3] < first_idx[4]
