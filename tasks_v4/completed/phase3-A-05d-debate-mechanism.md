# Mission 05d: debate-mechanism (4-phase deliberation)

**Handle**: phase3-A-05d
**Phase**: 3 (sequential, after 05c)
**Mission file**: phase3-A-05d-debate-mechanism.md
**Dependencies**: phase3-A-05c (PredatorSubPortfolio)
**Blocks**: phase3-A-05e (runner integration), phase3-A-05f (smoke gate)

---

## Before You Start

```bash
grep -E "@all|@phase3|@phase3-A|@phase3-A-05d" tasks_v4/scratchpad.md | tail -30
```

Verify the contracts you depend on:
```bash
.venv/bin/python -c "
from trophic.beliefs.investment_thesis import InvestmentThesis, ThesisBook
from trophic.agents.predator_sub_portfolio import PredatorSubPortfolio
from trophic.agents.apex_portfolio import make_debate_portfolio
print('05b/05c contracts present')
"
```
If this fails, halt and post @all.

```bash
sed -i 's/phase3-A-05d:PENDING/phase3-A-05d:RUNNING/' tasks_v4/scratchpad.md 2>/dev/null || true
```

---

## Goal

The **4-phase deliberation mechanism**. This is the heart of the v4 phase 3
innovation. Per-predator budgets + a structured argument graph that the
decomposer can mine downstream.

**Critical framing**: this is NOT strictly adversarial. Predators can
agree with each other, extend each other's theses, concede to a better
case, OR object. The output is a four-mode response, not just "objections."

### The 4 phases

| Phase | What happens | LM calls | Parallelism |
|---|---|---|---|
| 1 | Each predator independently proposes their action plan | 4 | parallel |
| 2 | Each predator responds to every OTHER predator's proposal (agree/concede/object/extend) | 4 | parallel |
| 3 | Each predator sees responses directed at them + own responses to others; revises proposal | 4 | parallel |
| 4 | Each predator sees all revised proposals; commits final orders against own budget | 4 | parallel |

**Total per day**: 16 LM calls (4 predators × 4 phases). On Qwen-4B vLLM
at ~3-5s/call with phase parallelism, ≈ 12-20s/day. Over 66 days ≈ 13-22
minutes for the debate path alone. Bedrock paths run alongside.

### Why 4 phases and not 3

The mission was originally written with 3 rounds. We landed on 4 because:

- Phase 1 alone is "everyone proposes," which gives no argument structure.
- A naive phase 2 ("everyone revises seeing others") is parallel revision,
  not debate — nobody is forced to address specific others.
- **The new phase 2 forces each predator to engage every other predator's
  proposal directly** with a structured response (agree/concede/object/extend).
- Phase 3 is where the predator gets to *re-propose* having absorbed
  what others said about them and what they said back. The internal
  monologue is captured.
- Phase 4 is the final independent commit — predators don't have to
  agree; they each act with their own capital. They've now had two
  chances to absorb perspectives before committing.

**The output transcript is mineable**: each phase-2 Response targets a
specific other predator and a specific element (a thesis_id or ticker)
with a specific `kind`. The decomposer (deferred) can ask "which
predator most often persuades others?", "which philosophy most often
holds firm under pressure?", etc.

---

## Files to Create / Modify

**Create:**
- `trophic/beliefs/debate.py` — Pydantic types + DSPy signatures + `run_debate(...)` orchestrator.
- `trophic/beliefs/predator_prompts.py` — the 4 philosophy-shaped system prompts.
- `tests/test_debate_mechanism.py`.

**Do not modify** the runner (05e does that). Do not modify strategy_committee.

---

## Implementation Steps

### Step 1: philosophy prompts (`predator_prompts.py`)

Four distinct philosophy headers. Each is a paragraph (~150 words)
defining the predator's character. Compose with the existing tax-framing
block from `apex_signatures._PM_TAX_FRAMING_BLOCK`.

```python
MOMENTUM_HEADER = """\
You are the **momentum predator** in a 4-predator debate ecology. You
trade your own slice of the portfolio's capital independently.

You look for accelerating trends: positive earnings surprises with follow-
through buying, sector rotation into your watch names, persistent insider
buying, options flow that confirms trend, breakouts above prior resistance.

Your typical horizons: h5 (1-week trend rides on news momentum), h20
(1-month rides on earnings or product catalysts). You rarely use h1
(too noisy for a momentum thesis) or h60 (your edge decays past a month).

Your kryptonite: chasing tops. Be honest with yourself about whether
your entry is "early in the trend" vs "late chase". When other predators
flag this, listen.
"""

VALUE_HEADER = """\
You are the **value predator** in a 4-predator debate ecology...
"""

# ... mean_revert, event_driven similarly
```

Each header ≈150 words. Compose: full prompt for round N is
`HEADER + _PM_TAX_FRAMING_BLOCK + ROUND_N_INSTRUCTIONS`.

### Step 2: Pydantic types (in `debate.py`)

```python
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field
from trophic.beliefs.apex_signatures import TickerView, Order, PortfolioState, PositionSnapshot
from trophic.beliefs.investment_thesis import InvestmentThesis

PredatorPhilosophy = Literal["momentum", "value", "mean_revert", "event_driven"]
ResponseKind = Literal["agree", "concede_to", "object_to", "extend"]


class DebatePhase1Proposal(BaseModel):
    """Phase 1: each predator's independent action plan."""
    predator_id: str
    philosophy: PredatorPhilosophy
    new_theses: list[InvestmentThesis] = Field(default_factory=list)
    close_theses: list[str] = Field(default_factory=list)   # thesis_ids
    views: list[TickerView] = Field(default_factory=list)
    orders: list[Order] = Field(default_factory=list)
    rationale: str


class Response(BaseModel):
    """One predator addressing one specific element of another predator's proposal."""
    target_predator_id: str
    target_thesis_id: Optional[str] = None   # set if responding to a specific thesis
    target_ticker: Optional[str] = None       # set if responding to a specific order/view
    kind: ResponseKind
    rationale: str = Field(..., min_length=1)
    # Examples by kind:
    # "agree":      "I share your h20 momentum case on AAPL; my own watchlist flagged the same pattern."
    # "concede_to": "Conceding my AAPL SELL because your h60 value framework is sound — I'm withdrawing."
    # "object_to":  "Your h60 thesis on KO ignores the volume divergence I see; expect a 5d pullback."
    # "extend":     "Your NVDA momentum thesis applies to AMD too; same chip cycle catalyst."


class DebatePhase2Responses(BaseModel):
    """Phase 2: one predator's responses to every other predator's phase-1 proposal."""
    predator_id: str
    responses: list[Response] = Field(default_factory=list)
    # No constraint on length, but the prompt should request at least one
    # response per other predator (even if it's just "I have no strong
    # view on your proposal" — those are valid Responses with kind=agree
    # and rationale="no objection").


class Concession(BaseModel):
    """Phase 3: predator records something it changed in response to phase-2 input."""
    target_predator_id: str   # the predator whose response drove this change
    what_changed: str          # "withdrew SELL on AAPL" / "shortened horizon h60→h20 on KO" / etc.
    rationale: str             # why


class HeldFirm(BaseModel):
    """Phase 3: predator records something it kept despite phase-2 pressure."""
    target_predator_id: Optional[str] = None   # the predator who challenged this (None if held without specific challenge)
    what_held: str             # "kept BUY on NVDA at h20"
    rationale: str             # why the phase-2 objection didn't move them


class DebatePhase3RevisedProposal(BaseModel):
    """Phase 3: each predator's revised proposal after absorbing phase-2."""
    predator_id: str
    revised_new_theses: list[InvestmentThesis] = Field(default_factory=list)
    revised_close_theses: list[str] = Field(default_factory=list)
    revised_views: list[TickerView] = Field(default_factory=list)
    revised_orders: list[Order] = Field(default_factory=list)
    conceded: list[Concession] = Field(default_factory=list)
    held_firm: list[HeldFirm] = Field(default_factory=list)
    rationale: str


class DebatePhase4Commit(BaseModel):
    """Phase 4: each predator's final commit, executed against own sub-portfolio."""
    predator_id: str
    final_theses_to_open: list[InvestmentThesis] = Field(default_factory=list)
    final_theses_to_close: list[str] = Field(default_factory=list)
    final_orders: list[Order] = Field(default_factory=list)
    rationale: str
```

### Step 3: DSPy signatures (one per phase)

```python
import dspy

class DebatePhase1(dspy.Signature):
    """[PHILOSOPHY HEADER + TAX FRAMING + PHASE-1 INSTRUCTIONS]
    Propose how YOU would act with YOUR capital today. You have not
    yet heard from other predators. Lay out your full plan: theses to
    open, theses to close, views, orders. Justify each."""
    today: str = dspy.InputField()
    days_remaining: int = dspy.InputField()
    predator_state: PortfolioState = dspy.InputField()
    own_positions: list[PositionSnapshot] = dspy.InputField()
    own_thesis_book: list[InvestmentThesis] = dspy.InputField()
    watchlist: list[str] = dspy.InputField()
    observations: str = dspy.InputField()
    proposal: DebatePhase1Proposal = dspy.OutputField()


class DebatePhase2(dspy.Signature):
    """[PHILOSOPHY HEADER + TAX FRAMING + PHASE-2 INSTRUCTIONS]
    You have seen all four predators' phase-1 proposals (including
    your own). Respond to each OTHER predator's proposal with one
    or more structured Responses. Each Response targets a specific
    thesis or ticker.

    Modes: agree (share their view), concede_to (you'd change yours
    because of theirs), object_to (you disagree and explain why),
    extend (their case applies elsewhere too).

    This is NOT adversarial — predators can agree and extend each
    other. Be honest. If you have nothing to say to predator X,
    output one Response with kind=agree and rationale="no objection".

    IMPORTANT: target_predator_id must NOT equal your own predator_id."""
    today: str = dspy.InputField()
    own_phase1: DebatePhase1Proposal = dspy.InputField()
    others_phase1: list[DebatePhase1Proposal] = dspy.InputField()
    responses: DebatePhase2Responses = dspy.OutputField()


class DebatePhase3(dspy.Signature):
    """[PHILOSOPHY HEADER + TAX FRAMING + PHASE-3 INSTRUCTIONS]
    You see (a) your phase-1 proposal, (b) every response directed AT
    you in phase 2, and (c) every response YOU made to others. Revise
    your proposal.

    Required outputs:
      - conceded: list things you're changing because of phase-2 input
                  (cite which predator changed your mind on what)
      - held_firm: list things you're keeping despite phase-2 pressure
                   (cite who pushed back and why you held)
      - revised_*: your new theses / orders / views

    A concession costs YOU capital. A held_firm bets YOUR capital on
    your conviction. Be honest about both."""
    today: str = dspy.InputField()
    predator_state: PortfolioState = dspy.InputField()
    own_phase1: DebatePhase1Proposal = dspy.InputField()
    own_phase2: DebatePhase2Responses = dspy.InputField()
    responses_directed_at_me: list[Response] = dspy.InputField()
    revised: DebatePhase3RevisedProposal = dspy.OutputField()


class DebatePhase4(dspy.Signature):
    """[PHILOSOPHY HEADER + TAX FRAMING + PHASE-4 INSTRUCTIONS]
    Final commit. You see all four predators' revised phase-3
    proposals. Act INDEPENDENTLY against your own capital.

    Compile your final order list. You may merge BUY+SELL on the
    same ticker into a single ROTATE if both make sense. You may
    drop any orders that no longer fit given what you've heard.

    Do NOT try to coordinate with other predators. Each predator
    acts independently in phase 4. The market resolves the rest."""
    today: str = dspy.InputField()
    predator_state: PortfolioState = dspy.InputField()
    own_phase3: DebatePhase3RevisedProposal = dspy.InputField()
    others_phase3: list[DebatePhase3RevisedProposal] = dspy.InputField()
    commit: DebatePhase4Commit = dspy.OutputField()
```

### Step 4: orchestrator (`run_debate`)

```python
import asyncio
import dspy
from trophic.agents.apex_portfolio import ApexPortfolio


def _signature_for_phase(philosophy: str, phase: int):
    """Returns the DSPy signature class with the right philosophy header."""
    # Build a subclass per (philosophy, phase) at module load, OR
    # construct a Signature with __doc__ assembled from header + phase instructions.
    ...


async def run_debate(
    apex_portfolio: ApexPortfolio,
    today: str,
    days_remaining: int,
    watchlist: list[str],
    observations: str,
    current_prices: dict[str, float],
    lm,
) -> dict[str, DebatePhase4Commit]:
    """Executes 4-phase debate. Returns predator_id → final commit.
    Caller mutates each PredatorSubPortfolio based on the commits."""
    if not apex_portfolio.is_debate_mode():
        raise ValueError("run_debate requires debate-mode portfolio")

    predator_ids = list(apex_portfolio.sub_portfolios.keys())

    # Phase 1: parallel across predators
    async def _phase1(pid: str):
        sub = apex_portfolio.sub_portfolios[pid]
        sig = _signature_for_phase(sub.philosophy, 1)
        pred_state = apex_portfolio.state_for_predator(pid, today, current_prices)
        def _call():
            with dspy.context(lm=lm):
                return dspy.Predict(sig)(
                    today=today,
                    days_remaining=days_remaining,
                    predator_state=pred_state,
                    own_positions=sub.positions_as_snapshots(),
                    own_thesis_book=sub.thesis_book.active_theses(),
                    watchlist=watchlist,
                    observations=observations,
                )
        out = await asyncio.to_thread(_call)
        return pid, out.proposal

    phase1_results = dict(await asyncio.gather(*(_phase1(pid) for pid in predator_ids)))

    # Phase 2: parallel; each predator sees all phase-1
    async def _phase2(pid: str):
        sub = apex_portfolio.sub_portfolios[pid]
        sig = _signature_for_phase(sub.philosophy, 2)
        others = [p for k, p in phase1_results.items() if k != pid]
        def _call():
            with dspy.context(lm=lm):
                return dspy.Predict(sig)(
                    today=today,
                    own_phase1=phase1_results[pid],
                    others_phase1=others,
                )
        out = await asyncio.to_thread(_call)
        return pid, out.responses

    phase2_results = dict(await asyncio.gather(*(_phase2(pid) for pid in predator_ids)))

    # Phase 3: parallel; each predator sees (own phase-1, own phase-2 responses, responses directed at them)
    def _responses_at(target_pid: str) -> list[Response]:
        bag = []
        for pid, r in phase2_results.items():
            for resp in r.responses:
                if resp.target_predator_id == target_pid:
                    bag.append(resp)
        return bag

    async def _phase3(pid: str):
        sub = apex_portfolio.sub_portfolios[pid]
        sig = _signature_for_phase(sub.philosophy, 3)
        pred_state = apex_portfolio.state_for_predator(pid, today, current_prices)
        def _call():
            with dspy.context(lm=lm):
                return dspy.Predict(sig)(
                    today=today,
                    predator_state=pred_state,
                    own_phase1=phase1_results[pid],
                    own_phase2=phase2_results[pid],
                    responses_directed_at_me=_responses_at(pid),
                )
        out = await asyncio.to_thread(_call)
        return pid, out.revised

    phase3_results = dict(await asyncio.gather(*(_phase3(pid) for pid in predator_ids)))

    # Phase 4: parallel; each predator sees all phase-3 revised proposals
    async def _phase4(pid: str):
        sub = apex_portfolio.sub_portfolios[pid]
        sig = _signature_for_phase(sub.philosophy, 4)
        pred_state = apex_portfolio.state_for_predator(pid, today, current_prices)
        others = [p for k, p in phase3_results.items() if k != pid]
        def _call():
            with dspy.context(lm=lm):
                return dspy.Predict(sig)(
                    today=today,
                    predator_state=pred_state,
                    own_phase3=phase3_results[pid],
                    others_phase3=others,
                )
        out = await asyncio.to_thread(_call)
        return pid, out.commit

    phase4_results = dict(await asyncio.gather(*(_phase4(pid) for pid in predator_ids)))

    # Log full transcript
    _log_debate_transcript(today, phase1_results, phase2_results, phase3_results, phase4_results)

    return phase4_results
```

### Step 5: token-budget compression on cross-visibility

Phase 1 proposals can be verbose. By the time you concatenate 3 of them
into `others_phase1` for phase 2, plus the observation block + system
prompt + tax framing, you can blow past 16K on Qwen.

**Compression rule for `others_phase1`** (and `others_phase3` in phase 4):

Instead of passing the full `DebatePhase1Proposal` for each other predator,
pass a *compressed view*:

```python
class CompressedProposal(BaseModel):
    predator_id: str
    philosophy: PredatorPhilosophy
    # Theses summarized: ticker + primary_horizon + 1-line rationale
    new_thesis_summary: list[str]  # ["AAPL h20 momentum: earnings tailwind"]
    close_thesis_summary: list[str]  # ["thesis_id=abc12 — guidance cut invalidation"]
    # Orders summarized: side+ticker+horizon+expected_alpha_bps
    order_summary: list[str]  # ["BUY AAPL h20 350bps"]
    rationale_summary: str  # 1-2 sentences, NOT the full rationale
```

The phase 2/4 signatures take `list[CompressedProposal]` instead of
`list[DebatePhase1Proposal]`. Add a helper `compress_proposal(p) -> CompressedProposal`.

### Step 6: validation enforcement

Add a `model_validator(mode="after")` on `DebatePhase2Responses` that:

- Verifies no `Response.target_predator_id` equals the outer `predator_id`.
- Verifies every Response has either `target_thesis_id` OR `target_ticker`
  populated (not both None) — except for `kind="agree"` Responses with
  rationale containing "no objection", which can have both None.

### Step 7: transcript logging

Append a JSONL line per day to `data/firehose_eval/debate_transcripts/<run_label>.jsonl`:

```json
{
  "date": "2026-02-03",
  "phase1": {"momentum": {...}, "value": {...}, "mean_revert": {...}, "event_driven": {...}},
  "phase2": {...},
  "phase3": {...},
  "phase4": {...}
}
```

Each entry stores the full Pydantic dump. ≈30-80KB/day; 66 days ≈ 2-5MB.

### Step 8: validators run per predator

The phase-4 commit's `final_orders` go through the existing v4 validator
chain — `filter_tax_aware` then `validate_orders` (the scaler) — but
**scoped to that predator's sub-portfolio**.

Reuse the existing chain unchanged; just pass `state_for_predator(pid,...)`
and the predator's `positions` instead of aggregate state. Validator
rejections feed back to the predator only — they don't bubble up to
peers.

---

## Acceptance Criteria

- [ ] Four philosophy prompts defined in `predator_prompts.py`, each ≥120 words, each meaningfully distinct.
- [ ] Pydantic types `DebatePhase1Proposal`, `Response`, `DebatePhase2Responses`, `Concession`, `HeldFirm`, `DebatePhase3RevisedProposal`, `DebatePhase4Commit` all defined and round-trip.
- [ ] All four DSPy signatures defined, each composes (header + tax framing + phase instructions).
- [ ] `run_debate(...)` is async, executes 4 phases in series, each phase parallelizes across predators.
- [ ] Cross-visibility compression: phase 2/4 take `list[CompressedProposal]`, not full proposals.
- [ ] Validator `target_predator_id != own predator_id` enforced.
- [ ] Transcript JSONL written per day.
- [ ] Validators run per predator (no aggregate validation).
- [ ] Existing tests stay green (currently 214/1 after 05a+05b; will be higher after 05c).

---

## Testing Conditions

All offline — mock LM responses. Do not hit vLLM in the test suite.

### 1. Philosophy prompts distinct

```bash
.venv/bin/python -m pytest tests/test_debate_mechanism.py::test_philosophy_prompts_distinct -x -v
```
**Expected**: pass. Verifies the 4 headers have distinct content (e.g., "momentum" word count >0 in MOMENTUM_HEADER, ~0 in VALUE_HEADER; "valuation" or "intrinsic value" present in VALUE_HEADER only; etc.).

### 2. Phase types round-trip

```bash
.venv/bin/python -m pytest tests/test_debate_mechanism.py::test_phase_types_roundtrip -x -v
```
**Expected**: pass. Build phase-1/2/3/4 Pydantic objects with synthetic data, dump → validate → assert equal.

### 3. Response self-targeting rejected

```bash
.venv/bin/python -m pytest tests/test_debate_mechanism.py::test_self_targeting_rejected -x -v
```
**Expected**: pass. `DebatePhase2Responses(predator_id="momentum", responses=[Response(target_predator_id="momentum", ...)])` raises ValidationError.

### 4. Orchestration with mock LM

```bash
.venv/bin/python -m pytest tests/test_debate_mechanism.py::test_run_debate_orchestration -x -v
```
**Expected**: pass. Mock LM returns deterministic outputs; verify all 4 predators get phase 1, 2, 3, 4 calls (16 total mock invocations); verify `responses_directed_at_me` for predator A in phase 3 contains the responses targeting A from phases 2 of B/C/D.

### 5. Compression preserves intent

```bash
.venv/bin/python -m pytest tests/test_debate_mechanism.py::test_compression_preserves_intent -x -v
```
**Expected**: pass. `compress_proposal(p).order_summary` contains BUY/SELL + ticker + horizon for each order in `p.orders`; same for theses.

### 6. Per-predator validation scoping

```bash
.venv/bin/python -m pytest tests/test_debate_mechanism.py::test_per_predator_validation -x -v
```
**Expected**: pass. Predator A's phase-4 commit over-allocates against A's slice; validator rejects A's order; B/C/D unaffected.

### 7. Transcript JSONL

```bash
.venv/bin/python -m pytest tests/test_debate_mechanism.py::test_transcript_logging -x -v
```
**Expected**: pass. After a synthetic debate, JSONL line contains all 4 phases.

### 8. No regressions

```bash
.venv/bin/python -m pytest tests/ -x -q
```
**Expected**: green.

---

## Coordination

- **No side-effects** on `strategy_committee.py` (non-debate paths still use it).
- **No side-effects** on tax-aware validators (they run per-predator via `state_for_predator`, no changes needed).
- **No side-effects** on `apex_portfolio.py` (05c added what's needed).
- 05e (runner) wires this into the new `ECO-DEBATE-QWEN` path.
- 05f (smoke gate) tests the full mechanic end-to-end against vLLM.

## Performance notes for downstream

- **16 LM calls/day** at ~3-5s on Qwen-4B vLLM = 50-80s/day for the debate path. 66 days ≈ 55-90 min. The other 8 paths run alongside via `--async-paths`.
- If Phase 2 or Phase 4 prompts exceed 14K tokens (Qwen 16K minus 2K for output), the compression in Step 5 is the lever — tighten it.
- The transcript JSONL grows ≈ 30-80KB/day. Plan disk; not a problem at 66 days.

---

## When Done

```bash
grep -E "@all|@phase3" tasks_v4/scratchpad.md | tail -30
sed -i 's/phase3-A-05d:RUNNING/phase3-A-05d:DONE/' tasks_v4/scratchpad.md

# 05e was blocked on 05a (done) AND 05d (now done). If 05a is DONE, unblock 05e:
sed -i 's/phase3-A-05e:BLOCKED.*$/phase3-A-05e:PENDING/' tasks_v4/scratchpad.md

cat >> tasks_v4/scratchpad.md <<'EOF'

[YYYY-MM-DD HH:MM] phase3-A-05d > @phase3-A-05e,05f: 4-phase debate mechanism landed. trophic/beliefs/debate.py exports run_debate(apex_portfolio, today, ..., lm) → dict[predator_id, DebatePhase4Commit]. Predators agree/concede/object/extend in phase 2. Concession + HeldFirm are first-class fields in phase 3. Transcript JSONL written per day. Validators run per-predator unchanged.
EOF

mv tasks_v4/phase3-A-05d-debate-mechanism.md tasks_v4/completed/
```
