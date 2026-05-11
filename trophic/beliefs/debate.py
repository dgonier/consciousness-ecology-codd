"""4-phase debate mechanism for the v4 ECO-DEBATE path.

The mechanic:

    Phase 1 (parallel × 4 predators):
        Each predator independently proposes their action plan.

    Phase 2 (parallel × 4 predators):
        Each predator emits structured Responses to every OTHER
        predator's phase-1 proposal. Four response modes:
            agree | concede_to | object_to | extend
        This is NOT strictly adversarial.

    Phase 3 (parallel × 4 predators):
        Each predator sees own phase-1, own phase-2 responses, and
        the responses directed AT them. Emits a revised proposal
        with explicit `Concession` and `HeldFirm` fields.

    Phase 4 (parallel × 4 predators):
        Each predator sees every other predator's phase-3 revised
        proposal (compressed). Commits final orders independently
        against own capital.

Total per day: 16 LM calls (4 predators × 4 phases). Each phase fires
N concurrent calls via `asyncio.gather`; the inter-phase barrier is
strict (phase N+1 cannot start until all of phase N has returned).

Cross-visibility payloads are compressed to fit Qwen-4B's 16K context
window. See `CompressedProposal` and `compress_proposal`.

The full transcript is written as one JSONL line per day to
`data/firehose_eval/debate_transcripts/<run_label>.jsonl` for
downstream decomposer mining.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Literal, Optional

import dspy
from pydantic import BaseModel, Field, model_validator

from trophic.agents.apex_portfolio import ApexPortfolio
from trophic.beliefs.apex_signatures import (
    Order,
    PortfolioState,
    PositionSnapshot,
    TickerView,
    _PM_TAX_FRAMING_BLOCK,
)
from trophic.beliefs.investment_thesis import (
    InvestmentThesis,
    PredatorPhilosophy,
)
from trophic.beliefs.predator_prompts import header_for


ResponseKind = Literal["agree", "concede_to", "object_to", "extend"]


# ── Pydantic types ─────────────────────────────────────────────────────


def _drop_malformed_views(views: Any) -> list[Any]:
    """Filter a `views`/`revised_views` list to entries that have a
    discernible `primary_horizon` at the top level.

    Qwen-4B failure modes observed in the v4 sweep day-1..day-9:

      (a) `{'ticker': 'DIS', 'forecasts': {...confidence_h60: 'med'}}`
          — `primary_horizon` omitted at the top level (the model often
          mistakenly puts `confidence_h*` inside `forecasts` and forgets
          the sibling `primary_horizon`). The HorizonForecast itself
          may also be missing required `confidence_h*` fields.
      (b) `{'ticker': 'X', 'primary_horizon': None, ...}` — explicit
          null for the literal-typed `primary_horizon` field.
      (c) `{'ticker': 'Y', 'forecasts': {'h1': 0.0}, ...}` — only one
          horizon present; HorizonForecast requires h1/h5/h20/h60.

    We DROP rather than try to invent values. Inventing forecasts would
    be a real semantic error (the predator wouldn't have a defensible
    view on the dropped ticker); dropping just narrows the proposal,
    which is exactly the fallback we want. The runner still gets a
    valid (smaller) proposal it can act on, instead of a no-trade day.

    Dict-shaped views with all required HorizonForecast fields are
    preserved. Already-constructed TickerView instances pass through
    unchanged.
    """
    if not isinstance(views, list):
        return views  # let strict validation raise
    kept: list[Any] = []
    for v in views:
        if isinstance(v, TickerView):
            kept.append(v)
            continue
        if not isinstance(v, dict):
            # Drop garbage entries (e.g. bare list ['ticker'] observed
            # at views.25 on day-4 of run2 — a truncation/confusion
            # artifact). Letting strict validation raise here would
            # poison the whole proposal.
            continue
        # Drop if primary_horizon is missing or null at the top level.
        ph = v.get("primary_horizon")
        if ph not in ("h1", "h5", "h20", "h60"):
            continue
        # Drop if forecasts is missing or lacks any of the required
        # numeric horizons (h1/h5/h20/h60) at the top of forecasts.
        # HorizonForecast._flatten_nested_forecasts handles the nested-
        # dict case, but cannot fabricate horizons that aren't there.
        fc = v.get("forecasts")
        if isinstance(fc, dict):
            has_all_horizons = all(
                isinstance(fc.get(h), (int, float))
                or isinstance(fc.get(h), dict)  # nested-dict shape; flattener will handle
                for h in ("h1", "h5", "h20", "h60")
            )
            if not has_all_horizons:
                continue
        elif fc is None:
            continue
        kept.append(v)
    return kept


def _drop_orders_missing_required(orders: Any) -> list[Any]:
    """Filter an `orders` / `revised_orders` / `final_orders` list to
    entries that carry the fields we CAN'T safely default.

    Strictly required (drop if missing):
      - `side`            — must be in {BUY, SELL, HOLD, ROTATE}; can't invent.
      - `primary_horizon` — must be in {h1, h5, h20, h60}; determines tier
        sizing, edge floor waiver, min-hold. Inventing it would silently
        misroute downstream sizing logic.
      - `ticker`          — except on HOLD (which validators no-op).
        Inventing a ticker would BUY/SELL the wrong name.

    Defaultable (handled in Order before-validator, NOT dropped here):
      - `expected_alpha_bps` → defaults to 0.0
      - `reasoning`          → defaults to "unspecified"
      - `size_pct`           → has Field default

    Run8 day-1 sweep fail: `orders[6].primary_horizon: Field required`
    on a BUY that had nothing else wrong with it. Pre-fix this poisoned
    the whole Phase-1 proposal even though 5 other orders were valid.
    """
    if not isinstance(orders, list):
        return orders  # let strict validation raise
    HORIZONS = ("h1", "h5", "h20", "h60")
    SIDES = ("BUY", "SELL", "HOLD", "ROTATE")
    kept: list[Any] = []
    for o in orders:
        if not isinstance(o, dict):
            # Already-constructed Order instances pass through; garbage
            # (lists, strings, None) is dropped.
            if hasattr(o, "primary_horizon"):
                kept.append(o)
            continue
        side = (o.get("side") or "").upper().strip()
        if side not in SIDES:
            continue
        ph = o.get("primary_horizon")
        if ph not in HORIZONS:
            continue
        # Ticker is required except for HOLD (which validators no-op).
        if side != "HOLD":
            tk = (o.get("ticker") or "").strip()
            if not tk:
                continue
        kept.append(o)
    return kept


def _inject_thesis_predator_fields(
    theses: Any, predator_id: str, philosophy: Any,
) -> Any:
    """Set `predator_id` and `philosophy` on every thesis-shaped dict
    in `theses` that's missing them.

    Qwen-4B reality (sweep day-4 Phase-4 fail):

        final_theses_to_open[0].predator_id
          Field required [type=missing, input_value={'thesis_id': ...}, ...]

    The model emits an `InvestmentThesis` dict that's complete except
    for `predator_id` (and sometimes `philosophy`) — fields it has no
    visibility into from its own context. The outer wrapper carries
    the authoritative `predator_id`/`philosophy`; we propagate them
    down into each thesis dict that's missing them.

    Already-constructed `InvestmentThesis` instances pass through
    unchanged (they already carry the fields). Non-dict, non-thesis
    entries pass through so strict validation can raise the standard
    error.
    """
    if not isinstance(theses, list):
        return theses
    out: list[Any] = []
    for t in theses:
        if isinstance(t, InvestmentThesis):
            out.append(t)
            continue
        if isinstance(t, dict):
            t = dict(t)  # don't mutate input
            if not t.get("predator_id") and predator_id:
                t["predator_id"] = predator_id
            if not t.get("philosophy") and philosophy:
                t["philosophy"] = philosophy
            out.append(t)
            continue
        out.append(t)
    return out


class DebatePhase1Proposal(BaseModel):
    """Phase 1: each predator's independent action plan, drafted before
    they see any other predator's view.

    Permissive input shapes (Qwen-4B reality):

      - `new_theses[i]` dicts may omit `predator_id` / `philosophy` — the
        model has no in-context handle on which wrapper owns it.
        Injected from the outer wrapper by `_normalize_shape`.
      - `views[i]` dicts may omit `primary_horizon` (≈ half of Phase-1
        fails in the day-1..day-9 sweep) or carry a partial
        HorizonForecast (only `h1`). These views are silently DROPPED
        rather than failing the whole proposal — the runner sees a
        smaller-but-valid proposal instead of a no-trade fallback day.

    The drop-malformed-views policy mirrors the
    `_filter_invalid_responses` policy on `DebatePhase2Responses`:
    structurally broken entries are filtered; semantically reasonable
    entries pass strict validation unmodified.
    """
    predator_id: str
    philosophy: PredatorPhilosophy
    new_theses: list[InvestmentThesis] = Field(default_factory=list)
    close_theses: list[str] = Field(default_factory=list)
    views: list[TickerView] = Field(default_factory=list)
    orders: list[Order] = Field(default_factory=list)
    rationale: str = Field(default="", description="High-level plan, ≤500 chars.")

    @model_validator(mode="before")
    @classmethod
    def _normalize_shape(cls, data: Any) -> Any:
        """Inject thesis predator_id/philosophy from the outer wrapper,
        and drop malformed `views` entries.

        Only operates on dict input — already-constructed
        DebatePhase1Proposal instances pass through. The orchestrator's
        `_coerce_proposal` is responsible for filling outer
        `predator_id`/`philosophy` before this validator runs.
        """
        if not isinstance(data, dict):
            return data
        pid = data.get("predator_id", "")
        phil = data.get("philosophy")
        if "new_theses" in data:
            data["new_theses"] = _inject_thesis_predator_fields(
                data["new_theses"], pid, phil,
            )
        if "views" in data:
            data["views"] = _drop_malformed_views(data["views"])
        if "orders" in data:
            data["orders"] = _drop_orders_missing_required(data["orders"])
        return data


class Response(BaseModel):
    """One predator addressing one specific element of another predator's
    proposal. Mineable by a decomposer downstream — each Response is a
    typed edge in the argument graph.

    Modes (`kind`):
      - agree:      "I share your h20 momentum case on AAPL; my own
                     watchlist flagged the same pattern."
      - concede_to: "Conceding my AAPL SELL because your h60 value
                     framework is sound — I'm withdrawing."
      - object_to:  "Your h60 thesis on KO ignores the volume divergence
                     I see; expect a 5d pullback."
      - extend:     "Your NVDA momentum thesis applies to AMD too; same
                     chip-cycle catalyst."
    """
    target_predator_id: str = Field(
        ..., min_length=1,
        description="The OTHER predator this Response is addressed to. "
                    "Must not equal the predator_id emitting it.",
    )
    target_thesis_id: Optional[str] = Field(
        default=None,
        description="Set if responding to a specific thesis in the target's proposal.",
    )
    target_ticker: Optional[str] = Field(
        default=None,
        description="Set if responding to a specific order or view (by ticker).",
    )
    kind: ResponseKind
    rationale: str = Field(default="unspecified", min_length=1)


class DebatePhase2Responses(BaseModel):
    """Phase 2: one predator's full response set across every other predator.

    Validators:
      - No `Response.target_predator_id` may equal the outer `predator_id`
        (self-targeting is rejected).
      - Every Response must have at least one of `target_thesis_id` /
        `target_ticker` populated, UNLESS it is a `kind="agree"` "no
        objection" placeholder (rationale contains "no objection").

    Permissive input shape (Qwen-4B reality):
      The prompt asks the model to "emit one or more structured Responses"
      and the model frequently returns the OutputField as a bare list of
      Responses (or, in some calls, as a list of per-peer
      DebatePhase2Responses-shaped wrappers — one entry per other
      predator). A `mode="before"` validator unwraps these shapes back
      into the canonical `{"predator_id": ..., "responses": [...]}` form
      so DSPy's parse succeeds. Empirically this restored the debate path
      from 100% no-trade-fallback days to a live trading path; see
      `tasks_v4/DEBATE_SMOKE_RESULTS.md`.
    """
    # `predator_id` is conceptually required, but Qwen-4B routinely
    # emits the OutputField as a bare list of Responses with no wrapper
    # — the wrapper's `predator_id` field has no place in that shape.
    # We default it to "" here so DSPy's JSON-adapter materialization
    # succeeds; the orchestrator's `_coerce_responses` then rebuilds the
    # model with the correct `predator_id` injected. Tests still assert
    # the post-coerce model carries the right pid.
    predator_id: str = Field(default="")
    responses: list[Response] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_shape(cls, data: Any) -> Any:
        """Accept the canonical dict shape, OR a list of Response dicts,
        OR a list of nested DebatePhase2Responses-shaped wrappers.

        Returns a dict in the canonical shape; the field-level validators
        run after this to enforce the per-Response constraints. We
        deliberately do NOT inject `predator_id` here — the orchestrator's
        coercion layer is the source of truth for that; if it's missing
        the model_validator will raise the standard required-field error.
        """
        if isinstance(data, list):
            if not data:
                return {"responses": []}
            # Case A: list of Response dicts (e.g. one-per-peer flat list).
            # Heuristic: each element has the Response keys but NOT a
            # nested "responses" list.
            if all(
                isinstance(item, dict)
                and "target_predator_id" in item
                and "responses" not in item
                for item in data
            ):
                return {"responses": list(data)}
            # Case B: list of DebatePhase2Responses wrappers — one per peer.
            # Each element looks like {"predator_id": ..., "responses": [...]}.
            # Flatten the nested responses; keep the first non-empty
            # predator_id if present (caller will overwrite via setdefault).
            if all(
                isinstance(item, dict) and "responses" in item
                for item in data
            ):
                merged: list[Any] = []
                pid_from_inner: str | None = None
                for item in data:
                    inner = item.get("responses") or []
                    if isinstance(inner, list):
                        merged.extend(inner)
                    if pid_from_inner is None:
                        candidate = item.get("predator_id")
                        if isinstance(candidate, str) and candidate:
                            pid_from_inner = candidate
                out: dict[str, Any] = {"responses": merged}
                if pid_from_inner:
                    out["predator_id"] = pid_from_inner
                return out
            # Unknown list shape — let Pydantic raise its standard error.
            return data
        return data

    @model_validator(mode="after")
    def _filter_invalid_responses(self) -> "DebatePhase2Responses":
        """Drop structurally-invalid Responses rather than rejecting the
        whole batch.

        Original behavior raised on (a) self-targeting and (b) Responses
        missing both `target_thesis_id` and `target_ticker` without a
        "no objection" placeholder. Both are real lint signals but
        empirically Qwen-4B emits ~1 such Response per Phase-2 batch
        (often a generic "broadly aligned" toward the emitting predator).
        Raising poisoned 100% of debate days in the 1-day smoke; the
        runner caught the exception and produced a no-trade fallback day,
        but the entire debate output for that day was lost.

        Filtering instead preserves the substantive Responses (the ones
        with concrete `target_thesis_id`/`target_ticker`) and only
        silently drops the structurally-broken ones. Downstream
        decomposer mining only consumes well-formed Responses anyway, so
        the lint loss is local.

        If filtering empties the responses list entirely, we leave the
        empty list in place — phase 3 just sees no `responses_directed_at_me`
        for that emitter, which is a valid (if anodyne) phase-2 outcome.
        """
        own = self.predator_id
        kept: list[Response] = []
        for r in self.responses:
            if r.target_predator_id == own:
                # Self-targeting: silently drop.
                continue
            no_target = r.target_thesis_id is None and r.target_ticker is None
            is_no_objection = (
                r.kind == "agree" and "no objection" in r.rationale.lower()
            )
            if no_target and not is_no_objection:
                # No target + not a no-objection placeholder: silently drop.
                continue
            kept.append(r)
        # In-place mutation is safe here — Pydantic v2 model_validator(mode="after")
        # receives the constructed model and may mutate fields.
        object.__setattr__(self, "responses", kept)
        return self


class Concession(BaseModel):
    """Phase 3: predator records something it CHANGED in response to phase-2 input.

    A concession costs the conceding predator capital — it's the typed
    record of "X moved me." Decomposer-mineable.
    """
    target_predator_id: str = Field(
        ..., min_length=1,
        description="The predator whose response drove this change.",
    )
    what_changed: str = Field(
        default="unspecified", min_length=1,
        description="Concrete change: 'withdrew SELL on AAPL', 'shortened horizon h60→h20 on KO', etc.",
    )
    rationale: str = Field(default="unspecified", min_length=1)


class HeldFirm(BaseModel):
    """Phase 3: predator records something it KEPT despite phase-2 pressure.

    A held_firm bets the predator's capital on its own conviction in
    the face of a specific objection. Decomposer-mineable.
    """
    target_predator_id: Optional[str] = Field(
        default=None,
        description="The predator who challenged this. None when held without a specific challenger.",
    )
    what_held: str = Field(
        default="unspecified", min_length=1,
        description="What was kept: 'kept BUY on NVDA at h20', etc.",
    )
    rationale: str = Field(default="unspecified", min_length=1)


class DebatePhase3RevisedProposal(BaseModel):
    """Phase 3: each predator's revised proposal after absorbing phase-2.

    `conceded` + `held_firm` together explain the diff from phase 1 to
    phase 3. The decomposer can mine which predators most often
    persuade and which most often hold firm.

    Permissive input shapes (Qwen-4B reality, day-11 sweep fail):

      - `revised_views[i].primary_horizon` arrived as `None`
        (literal_error). Malformed views are silently DROPPED by
        `_normalize_shape`, same policy as Phase 1.
      - `revised_new_theses[i]` may omit `predator_id` — injected from
        the outer wrapper.

    The outer `predator_id` is NOT inferred from the revised inputs —
    the orchestrator's `_coerce_revised` is authoritative.
    """
    predator_id: str
    revised_new_theses: list[InvestmentThesis] = Field(default_factory=list)
    revised_close_theses: list[str] = Field(default_factory=list)
    revised_views: list[TickerView] = Field(default_factory=list)
    revised_orders: list[Order] = Field(default_factory=list)
    conceded: list[Concession] = Field(default_factory=list)
    held_firm: list[HeldFirm] = Field(default_factory=list)
    rationale: str = Field(default="")

    @model_validator(mode="before")
    @classmethod
    def _normalize_shape(cls, data: Any) -> Any:
        """Inject thesis predator_id/philosophy and drop malformed
        `revised_views`. Only operates on dict input.
        """
        if not isinstance(data, dict):
            return data
        pid = data.get("predator_id", "")
        # Phase 3 has no `philosophy` field on the outer wrapper, so
        # we pass None and rely on the thesis dict to carry its own
        # philosophy. Inject only the predator_id.
        if "revised_new_theses" in data:
            data["revised_new_theses"] = _inject_thesis_predator_fields(
                data["revised_new_theses"], pid, None,
            )
        if "revised_views" in data:
            data["revised_views"] = _drop_malformed_views(data["revised_views"])
        if "revised_orders" in data:
            data["revised_orders"] = _drop_orders_missing_required(data["revised_orders"])
        return data


class DebatePhase4Commit(BaseModel):
    """Phase 4: each predator's final commit, executed against its own
    sub-portfolio by the runner. Predators do NOT coordinate orders
    here — each acts on its own conviction with its own capital.

    Permissive input shape (Qwen-4B reality, day-4 sweep fail):

        final_theses_to_open[0].predator_id
          Field required [type=missing, input_value={'thesis_id': 'a1b2c3d4e5...}]

    The model emits a complete InvestmentThesis dict but forgets to
    set `predator_id` — it has no in-context handle on which wrapper
    owns the thesis. `_normalize_shape` propagates the outer
    `predator_id` down into each thesis dict that's missing it.
    """
    predator_id: str
    final_theses_to_open: list[InvestmentThesis] = Field(default_factory=list)
    final_theses_to_close: list[str] = Field(default_factory=list)
    final_orders: list[Order] = Field(default_factory=list)
    rationale: str = Field(default="")

    @model_validator(mode="before")
    @classmethod
    def _normalize_shape(cls, data: Any) -> Any:
        """Inject thesis predator_id from the outer wrapper into each
        thesis dict in `final_theses_to_open` that's missing it. Also
        coerce `final_theses_to_close` entries from full thesis dicts
        (Qwen-4B drift) down to their `thesis_id` strings.

        Phase 4 has no `philosophy` field on the outer wrapper, so we
        pass None — the thesis dict must carry its own philosophy
        (which is a Literal and cannot be defaulted without losing the
        per-predator typing).

        Day-6 run4 sweep fail: Qwen-4B emitted
            final_theses_to_close[0] = {'thesis_id': 'a1b2c3...', ...}
        instead of the expected
            final_theses_to_close[0] = 'a1b2c3...'
        The model mirrored the `final_theses_to_open` shape (which IS
        a list of full thesis dicts) into the close slot — easy mistake.
        We coerce dict-shaped close entries to their thesis_id string.
        Non-coercible entries (no `thesis_id` key) are dropped.
        """
        if not isinstance(data, dict):
            return data
        pid = data.get("predator_id", "")
        if "final_theses_to_open" in data:
            data["final_theses_to_open"] = _inject_thesis_predator_fields(
                data["final_theses_to_open"], pid, None,
            )
        if "final_theses_to_close" in data:
            data["final_theses_to_close"] = _coerce_close_to_id_strings(
                data["final_theses_to_close"]
            )
        if "final_orders" in data:
            data["final_orders"] = _drop_orders_missing_required(data["final_orders"])
        return data


def _coerce_close_to_id_strings(close: Any) -> list[str]:
    """Force a `final_theses_to_close` field to a list of thesis_id
    strings. Accepts:
      - already-correct: list of strings → unchanged
      - Qwen-4B drift: list of full thesis dicts → extract thesis_id
      - mixed list → coerce dicts, keep strings

    Non-string, non-dict entries and dicts without `thesis_id` are
    dropped (we never invent an ID).
    """
    if not isinstance(close, list):
        return close  # let strict validation raise
    out: list[str] = []
    for item in close:
        if isinstance(item, str):
            if item:
                out.append(item)
            continue
        if isinstance(item, dict):
            tid = item.get("thesis_id")
            if isinstance(tid, str) and tid:
                out.append(tid)
            continue
        # Unknown shape — drop rather than poison the whole commit.
    return out


class CompressedProposal(BaseModel):
    """Token-budget-friendly view of a Phase 1 or Phase 3 proposal.

    Used as cross-visibility payload in phase 2 (`others_phase1` →
    `list[CompressedProposal]`) and phase 4 (`others_phase3` →
    `list[CompressedProposal]`). Full proposals can run 1-3K tokens
    each; with 3 peers per predator that's 3-9K tokens of cross-talk
    on top of headers + observations + tax framing. Compressing to
    one-line summaries keeps the assembled prompt under ~14K on
    Qwen-4B.
    """
    predator_id: str
    philosophy: PredatorPhilosophy
    new_thesis_summary: list[str] = Field(default_factory=list)
    close_thesis_summary: list[str] = Field(default_factory=list)
    order_summary: list[str] = Field(default_factory=list)
    rationale_summary: str = Field(default="")


# ── Compression helpers ────────────────────────────────────────────────


def _summarize_thesis(t: InvestmentThesis) -> str:
    cats = ", ".join(t.catalysts[:2]) if t.catalysts else "no-catalyst"
    return (
        f"{t.ticker} {t.primary_horizon} {t.direction} "
        f"({t.expected_alpha_bps:+.0f}bps, conf={t.confidence}): {cats}"
    )


def _summarize_close(thesis_id: str) -> str:
    return f"close thesis_id={thesis_id}"


def _summarize_order(o: Order) -> str:
    if o.side == "ROTATE":
        return (
            f"ROTATE {o.from_ticker}→{o.to_ticker} "
            f"@{o.size_pct:.0f}% {o.primary_horizon} "
            f"({o.expected_alpha_bps:+.0f}bps)"
        )
    if o.side == "HOLD":
        return f"HOLD {o.ticker or '*'}"
    return (
        f"{o.side} {o.ticker} @{o.size_pct:.0f}% {o.primary_horizon} "
        f"({o.expected_alpha_bps:+.0f}bps)"
    )


def _truncate(text: str, limit: int = 240) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def compress_proposal(p: DebatePhase1Proposal) -> CompressedProposal:
    """Compress a Phase-1 proposal to a one-line-per-element view."""
    return CompressedProposal(
        predator_id=p.predator_id,
        philosophy=p.philosophy,
        new_thesis_summary=[_summarize_thesis(t) for t in p.new_theses],
        close_thesis_summary=[_summarize_close(tid) for tid in p.close_theses],
        order_summary=[_summarize_order(o) for o in p.orders],
        rationale_summary=_truncate(p.rationale),
    )


def compress_phase3(p: DebatePhase3RevisedProposal) -> CompressedProposal:
    """Compress a Phase-3 revised proposal for `others_phase3` in phase 4."""
    return CompressedProposal(
        predator_id=p.predator_id,
        philosophy="momentum",  # placeholder; overwritten below
        new_thesis_summary=[_summarize_thesis(t) for t in p.revised_new_theses],
        close_thesis_summary=[
            _summarize_close(tid) for tid in p.revised_close_theses
        ],
        order_summary=[_summarize_order(o) for o in p.revised_orders],
        rationale_summary=_truncate(p.rationale),
    )


# ── Phase instruction blocks (composed into signature docstrings) ──────


_PHASE1_INSTRUCTIONS = """\

PHASE 1 — INDEPENDENT PROPOSAL

You have NOT yet heard from the other predators. Draft your full
action plan from your own philosophy alone:

  - `new_theses`: InvestmentThesis objects you want to open today.
  - `close_theses`: thesis_ids in your active book you want to close
    today (matured, invalidated, or no-longer-relevant).
  - `views`: per-ticker forward-return forecasts at h1/h5/h20/h60 for
    every name you intend to act on.
  - `orders`: concrete BUY/SELL/HOLD/ROTATE orders sized as percent of
    YOUR slice's equity. Every order MUST commit to a primary_horizon
    that matches its TickerView and expected_alpha_bps it can defend.
  - `rationale`: one paragraph on the net thesis for today.

Your sizing budget is YOUR slice's available cash (see
`predator_state.cash` and `predator_state.invested_pct`). Stay within
your own budget; the other predators have their own slices.
"""


_PHASE2_INSTRUCTIONS = """\

PHASE 2 — STRUCTURED RESPONSES TO PEERS

You have seen all four predators' phase-1 proposals (compressed). Emit
one or more structured `Response`s targeted at each OTHER predator's
proposal.

This is **NOT** an adversarial round. Predators can agree, extend
each other's theses, or concede when another's case is stronger.
Honest agreement is valuable signal.

For each other predator, emit at least one Response with one of:

  - kind="agree":      You share their view (cite the specific
                       thesis/ticker you agree with).
  - kind="concede_to": You'd CHANGE your own plan because of theirs
                       (you'll formalize the change in phase 3).
  - kind="object_to":  You disagree; explain the substantive flaw.
  - kind="extend":     Their case applies to additional names (extend
                       to specific tickers).

Each Response must target `target_predator_id` ≠ your own predator_id
AND have at least one of `target_thesis_id` / `target_ticker`
populated. If you have nothing substantive on a predator, emit a
single `kind="agree"` Response with rationale exactly containing the
phrase "no objection" — that placeholder satisfies the validator
without forcing fake disagreement.

Be honest. Two non-objections per predator is fine; six fake
objections is signal-destroying noise.

OUTPUT SHAPE — IMPORTANT:

Return a SINGLE JSON object with this exact top-level shape:

    {"predator_id": "<your predator_id>",
     "responses": [ <Response>, <Response>, ... ]}

DO NOT return a bare list of Responses at the top level.
DO NOT return one DebatePhase2Responses-wrapper per peer (e.g.
[{predator_id, responses}, {predator_id, responses}, ...]). Emit ONE
wrapper, with one flat `responses` list inside that names every peer
you address. Multiple Responses with the same `target_predator_id` are
fine (one per thesis/ticker you address).
"""


_PHASE3_INSTRUCTIONS = """\

PHASE 3 — REVISED PROPOSAL WITH STRUCTURED CONCESSIONS / HELD-FIRMS

You see:
  (a) `own_phase1` — your original proposal.
  (b) `own_phase2` — your Responses to others.
  (c) `responses_directed_at_me` — every Response the other three
       predators emitted that named YOU as target.

Revise your proposal. Two structured fields are first-class:

  - `conceded`: list `Concession`s — concrete things you're CHANGING
    because of phase-2 input. Each must cite `target_predator_id`
    (who moved you) and `what_changed` (the change in your plan).
    A concession costs YOU capital.

  - `held_firm`: list `HeldFirm`s — concrete things you're KEEPING
    despite phase-2 pressure. Each cites the challenger (optional)
    and `what_held` (the kept position). A held_firm BETS your
    capital on your conviction.

Then emit your `revised_*` fields (new_theses, close_theses, views,
orders) reflecting the revision. The revised plan is what you'd
commit to today, modulo any further updates you make in phase 4.

Be honest. If nothing changed, return `conceded=[]` and surface every
disagreement as `held_firm`. If everything changed, surface every
flip as a Concession. The decomposer mines this — empty fields are
acceptable; lying fields are not.
"""


_PHASE4_INSTRUCTIONS = """\

PHASE 4 — FINAL INDEPENDENT COMMIT

You see all four predators' revised phase-3 proposals (compressed).
You now COMMIT your final orders against YOUR sub-portfolio.

You do NOT coordinate with other predators. Each predator acts on
its own conviction with its own slice of capital. The market
resolves the rest.

Emit:
  - `final_theses_to_open`
  - `final_theses_to_close`
  - `final_orders` — the concrete order set the runner will execute
    against your slice. You may merge BUY+SELL on the same ticker
    into a single ROTATE if both make sense to you. You may drop
    any orders from your phase-3 revised plan that no longer feel
    right after seeing the other predators' phase-3 commitments.
  - `rationale` — one paragraph on the net commit.
"""


_PHASE_INSTRUCTIONS: dict[int, str] = {
    1: _PHASE1_INSTRUCTIONS,
    2: _PHASE2_INSTRUCTIONS,
    3: _PHASE3_INSTRUCTIONS,
    4: _PHASE4_INSTRUCTIONS,
}


def compose_phase_instructions(philosophy: PredatorPhilosophy, phase: int) -> str:
    """Assemble the full instructions for one (philosophy, phase) pair:

        HEADER + _PM_TAX_FRAMING_BLOCK + PHASE_N_INSTRUCTIONS
    """
    if phase not in _PHASE_INSTRUCTIONS:
        raise ValueError(f"phase must be in {{1,2,3,4}}, got {phase}")
    return (
        header_for(philosophy)
        + _PM_TAX_FRAMING_BLOCK
        + _PHASE_INSTRUCTIONS[phase]
    )


# ── DSPy signatures ────────────────────────────────────────────────────


class DebatePhase1(dspy.Signature):
    """4-phase debate, phase 1: propose your action plan independently.

    The full per-philosophy instructions (header + tax framing + phase-1
    instructions) are attached at runtime via `with_instructions` in
    `_signature_for_phase`. This base docstring is a placeholder.
    """
    today: str = dspy.InputField()
    days_remaining: int = dspy.InputField()
    predator_state: PortfolioState = dspy.InputField()
    own_positions: list[PositionSnapshot] = dspy.InputField()
    own_thesis_book: list[InvestmentThesis] = dspy.InputField()
    watchlist: list[str] = dspy.InputField()
    observations: str = dspy.InputField()
    proposal: DebatePhase1Proposal = dspy.OutputField()


class DebatePhase2(dspy.Signature):
    """4-phase debate, phase 2: respond to every other predator's phase-1
    proposal with structured Responses.

    Full instructions attached at runtime.
    """
    today: str = dspy.InputField()
    own_phase1: DebatePhase1Proposal = dspy.InputField()
    others_phase1: list[CompressedProposal] = dspy.InputField()
    responses: DebatePhase2Responses = dspy.OutputField()


class DebatePhase3(dspy.Signature):
    """4-phase debate, phase 3: revise proposal with explicit Concession
    and HeldFirm fields.

    Full instructions attached at runtime.
    """
    today: str = dspy.InputField()
    predator_state: PortfolioState = dspy.InputField()
    own_phase1: DebatePhase1Proposal = dspy.InputField()
    own_phase2: DebatePhase2Responses = dspy.InputField()
    responses_directed_at_me: list[Response] = dspy.InputField()
    revised: DebatePhase3RevisedProposal = dspy.OutputField()


class DebatePhase4(dspy.Signature):
    """4-phase debate, phase 4: final independent commit.

    Full instructions attached at runtime.
    """
    today: str = dspy.InputField()
    predator_state: PortfolioState = dspy.InputField()
    own_phase3: DebatePhase3RevisedProposal = dspy.InputField()
    others_phase3: list[CompressedProposal] = dspy.InputField()
    commit: DebatePhase4Commit = dspy.OutputField()


_PHASE_BASE_SIGNATURES: dict[int, type[dspy.Signature]] = {
    1: DebatePhase1,
    2: DebatePhase2,
    3: DebatePhase3,
    4: DebatePhase4,
}


def _signature_for_phase(philosophy: PredatorPhilosophy, phase: int):
    """Return a DSPy Signature class with the philosophy-specific
    instructions attached for the given (philosophy, phase) pair.

    Uses `Signature.with_instructions(...)` so we don't need to
    multiply-subclass; one base signature per phase, decorated at
    runtime with the philosophy's header + tax framing + phase
    instructions.
    """
    base = _PHASE_BASE_SIGNATURES[phase]
    instructions = compose_phase_instructions(philosophy, phase)
    return base.with_instructions(instructions)


# ── Transcript logging ─────────────────────────────────────────────────


DEFAULT_TRANSCRIPT_DIR = Path("data/firehose_eval/debate_transcripts")


def _log_debate_transcript(
    today: str,
    phase1: dict[str, DebatePhase1Proposal],
    phase2: dict[str, DebatePhase2Responses],
    phase3: dict[str, DebatePhase3RevisedProposal],
    phase4: dict[str, DebatePhase4Commit],
    *,
    run_label: str = "debate",
    transcript_dir: Optional[Path] = None,
) -> Path:
    """Append one JSONL line per day with all four phases' Pydantic dumps.

    Returns the path written. Creates the directory if missing.
    """
    out_dir = transcript_dir if transcript_dir is not None else DEFAULT_TRANSCRIPT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_label}.jsonl"
    row: dict[str, Any] = {
        "date": today,
        "phase1": {pid: p.model_dump(mode="json") for pid, p in phase1.items()},
        "phase2": {pid: p.model_dump(mode="json") for pid, p in phase2.items()},
        "phase3": {pid: p.model_dump(mode="json") for pid, p in phase3.items()},
        "phase4": {pid: p.model_dump(mode="json") for pid, p in phase4.items()},
    }
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return path


# ── Orchestrator ───────────────────────────────────────────────────────


async def run_debate(
    apex_portfolio: ApexPortfolio,
    today: str,
    days_remaining: int,
    watchlist: list[str],
    observations: str,
    current_prices: dict[str, float],
    lm,
    *,
    run_label: str = "debate",
    transcript_dir: Optional[Path] = None,
    write_transcript: bool = True,
) -> dict[str, DebatePhase4Commit]:
    """Execute the 4-phase debate. Returns predator_id → final commit.

    Each phase fires N concurrent LM calls via `asyncio.gather`; the
    inter-phase barrier is strict. The caller is responsible for
    mutating each `PredatorSubPortfolio` based on the returned commits
    (e.g., `apex_portfolio.sub_portfolios[pid].buy(...)`).

    The full transcript is appended to
    `<transcript_dir>/<run_label>.jsonl` unless `write_transcript=False`.
    """
    if not apex_portfolio.is_debate_mode():
        raise ValueError("run_debate requires debate-mode portfolio")

    predator_ids = list(apex_portfolio.sub_portfolios.keys())
    philosophies: dict[str, PredatorPhilosophy] = {
        pid: apex_portfolio.sub_portfolios[pid].philosophy  # type: ignore[assignment]
        for pid in predator_ids
    }

    # ── Phase 1: independent proposals ────────────────────────────────
    async def _phase1(pid: str) -> tuple[str, DebatePhase1Proposal]:
        sub = apex_portfolio.sub_portfolios[pid]
        sig = _signature_for_phase(philosophies[pid], 1)
        pred_state, own_positions = apex_portfolio.state_for_predator(
            pid, today, current_prices,
        )
        own_book = sub.thesis_book.active_theses()

        def _call() -> DebatePhase1Proposal:
            with dspy.context(lm=lm):
                out = dspy.Predict(sig)(
                    today=today,
                    days_remaining=days_remaining,
                    predator_state=pred_state,
                    own_positions=own_positions,
                    own_thesis_book=own_book,
                    watchlist=watchlist,
                    observations=observations,
                )
            return _coerce_proposal(out.proposal, pid, philosophies[pid])

        proposal = await asyncio.to_thread(_call)
        return pid, proposal

    phase1_results: dict[str, DebatePhase1Proposal] = dict(
        await asyncio.gather(*(_phase1(pid) for pid in predator_ids))
    )

    # ── Phase 2: structured responses to peers ─────────────────────────
    async def _phase2(pid: str) -> tuple[str, DebatePhase2Responses]:
        sig = _signature_for_phase(philosophies[pid], 2)
        own_p1 = phase1_results[pid]
        others_compressed = [
            compress_proposal(phase1_results[k])
            for k in predator_ids if k != pid
        ]

        def _call() -> DebatePhase2Responses:
            with dspy.context(lm=lm):
                out = dspy.Predict(sig)(
                    today=today,
                    own_phase1=own_p1,
                    others_phase1=others_compressed,
                )
            return _coerce_responses(out.responses, pid)

        responses = await asyncio.to_thread(_call)
        return pid, responses

    phase2_results: dict[str, DebatePhase2Responses] = dict(
        await asyncio.gather(*(_phase2(pid) for pid in predator_ids))
    )

    # ── Phase 3: revise proposal with concessions/held-firms ──────────
    def _responses_at(target_pid: str) -> list[Response]:
        bag: list[Response] = []
        for pid, r in phase2_results.items():
            if pid == target_pid:
                continue
            for resp in r.responses:
                if resp.target_predator_id == target_pid:
                    bag.append(resp)
        return bag

    async def _phase3(pid: str) -> tuple[str, DebatePhase3RevisedProposal]:
        sig = _signature_for_phase(philosophies[pid], 3)
        pred_state, _ = apex_portfolio.state_for_predator(
            pid, today, current_prices,
        )
        directed_at_me = _responses_at(pid)

        def _call() -> DebatePhase3RevisedProposal:
            with dspy.context(lm=lm):
                out = dspy.Predict(sig)(
                    today=today,
                    predator_state=pred_state,
                    own_phase1=phase1_results[pid],
                    own_phase2=phase2_results[pid],
                    responses_directed_at_me=directed_at_me,
                )
            return _coerce_revised(out.revised, pid)

        revised = await asyncio.to_thread(_call)
        return pid, revised

    phase3_results: dict[str, DebatePhase3RevisedProposal] = dict(
        await asyncio.gather(*(_phase3(pid) for pid in predator_ids))
    )

    # ── Phase 4: final independent commit ─────────────────────────────
    async def _phase4(pid: str) -> tuple[str, DebatePhase4Commit]:
        sig = _signature_for_phase(philosophies[pid], 4)
        pred_state, _ = apex_portfolio.state_for_predator(
            pid, today, current_prices,
        )
        own_p3 = phase3_results[pid]
        others_p3 = []
        for k in predator_ids:
            if k == pid:
                continue
            cp = compress_phase3(phase3_results[k])
            cp.philosophy = philosophies[k]
            others_p3.append(cp)

        def _call() -> DebatePhase4Commit:
            with dspy.context(lm=lm):
                out = dspy.Predict(sig)(
                    today=today,
                    predator_state=pred_state,
                    own_phase3=own_p3,
                    others_phase3=others_p3,
                )
            return _coerce_commit(out.commit, pid)

        commit = await asyncio.to_thread(_call)
        return pid, commit

    phase4_results: dict[str, DebatePhase4Commit] = dict(
        await asyncio.gather(*(_phase4(pid) for pid in predator_ids))
    )

    if write_transcript:
        _log_debate_transcript(
            today,
            phase1_results,
            phase2_results,
            phase3_results,
            phase4_results,
            run_label=run_label,
            transcript_dir=transcript_dir,
        )

    return phase4_results


# ── Output coercion (DSPy may return dict or Pydantic) ─────────────────


def _coerce_proposal(
    raw: Any, pid: str, philosophy: PredatorPhilosophy,
) -> DebatePhase1Proposal:
    if isinstance(raw, DebatePhase1Proposal):
        return raw
    data = _to_dict(raw)
    data.setdefault("predator_id", pid)
    data.setdefault("philosophy", philosophy)
    return DebatePhase1Proposal(**data)


def _coerce_responses(raw: Any, pid: str) -> DebatePhase2Responses:
    """Coerce DSPy's Phase-2 output into a `DebatePhase2Responses` with
    the correct `predator_id`.

    The orchestrator (`run_debate._phase2`) is the single source of truth
    for which predator emitted the phase-2 output. The model itself
    frequently:
      - returns `predator_id=""` (we default the field to "" so DSPy's
        adapter can materialize the wrapper without crashing when Qwen
        emits a bare list of Responses), OR
      - returns the OutputField as a bare list of Responses with no
        wrapper, OR
      - returns a list of per-peer wrappers (one per other predator).

    We accept all three shapes and ALWAYS overwrite `predator_id` with
    the orchestrator-supplied `pid` — the model is not authoritative on
    its own identity in this seam.
    """
    if isinstance(raw, DebatePhase2Responses):
        if raw.predator_id != pid:
            return DebatePhase2Responses(
                predator_id=pid, responses=raw.responses
            )
        return raw
    if isinstance(raw, list):
        # Run the before-validator manually to gather responses into the
        # canonical dict, then inject the orchestrator-supplied pid.
        prepared = DebatePhase2Responses._normalize_shape(raw)
        if isinstance(prepared, dict):
            prepared = dict(prepared)  # don't mutate the input
            prepared["predator_id"] = pid  # orchestrator is authoritative
            return DebatePhase2Responses(**prepared)
        # Unknown shape — let Pydantic raise.
        return DebatePhase2Responses.model_validate(raw)
    data = _to_dict(raw)
    data["predator_id"] = pid  # orchestrator is authoritative
    return DebatePhase2Responses(**data)


def _coerce_revised(raw: Any, pid: str) -> DebatePhase3RevisedProposal:
    if isinstance(raw, DebatePhase3RevisedProposal):
        return raw
    data = _to_dict(raw)
    data.setdefault("predator_id", pid)
    return DebatePhase3RevisedProposal(**data)


def _coerce_commit(raw: Any, pid: str) -> DebatePhase4Commit:
    if isinstance(raw, DebatePhase4Commit):
        return raw
    data = _to_dict(raw)
    data.setdefault("predator_id", pid)
    return DebatePhase4Commit(**data)


def _to_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, BaseModel):
        return raw.model_dump()
    if isinstance(raw, dict):
        return dict(raw)
    if hasattr(raw, "model_dump"):
        return raw.model_dump()
    raise TypeError(
        f"cannot coerce {type(raw).__name__} into a debate Pydantic model; "
        f"DSPy returned an unexpected output type."
    )


__all__ = [
    "DebatePhase1Proposal",
    "Response",
    "DebatePhase2Responses",
    "Concession",
    "HeldFirm",
    "DebatePhase3RevisedProposal",
    "DebatePhase4Commit",
    "CompressedProposal",
    "ResponseKind",
    "compress_proposal",
    "compress_phase3",
    "compose_phase_instructions",
    "DebatePhase1",
    "DebatePhase2",
    "DebatePhase3",
    "DebatePhase4",
    "run_debate",
    "_signature_for_phase",
    "_log_debate_transcript",
    "DEFAULT_TRANSCRIPT_DIR",
]
