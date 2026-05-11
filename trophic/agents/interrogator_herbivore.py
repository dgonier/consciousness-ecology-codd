"""InterrogatorHerbivore — a quant-analyst-shaped herbivore.

Pipeline for one scenario:
  1. Read producer broadcasts (technical/quantitative inputs).
  2. Ask: Qwen3-4B formulates one or more math questions, embedding the
     relevant numbers from the producer payloads inline.
  3. Solve: Each question goes to MathHost (Qwen2.5-Math-1.5B) which
     returns numeric findings.
  4. Synthesize: Qwen3-4B writes a structured synthesis citing the math
     findings as the quantitative spine.

The output is a single herbivore broadcast (XML <synthesis kind="interrogator">)
that the predator's Channel can attend to like any other herbivore.

Diet: same as technical herbivore (price-event substrate). The math node
isn't an agent in the trophic stack — it's a tool the Interrogator calls
internally.

No cross-attention Channel on the producer→interrogator edge in v1; the
herbivore reads producer payloads directly. (We can add a Channel later
if it earns its compute, but the read-and-extract job is structurally
deterministic.)
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..cross_model_channel import CrossModelChannel
from ..math_host import MathHost, MathResult
from ..model_host import ModelHost
from ..types import Broadcast
from ..training.xml_schema import emit_synthesis
from .base import BaseAgent, new_agent_id


INTERROGATOR_KINDS = ("interrogator",)


# Diet: any quantitative substrate (OHLCV bars, quote series, options flow)
DIETS = {
    "interrogator": [
        "from_tickdelta",
        "from_quant_series",
    ],
}


PLAN_PROMPT = (
    "You are a quantitative-analysis planner. You have a small toolbox of"
    " mathematical operations and a math-specialist agent that can compute"
    " them. Given the data below, decide what would be useful to compute"
    " (1-3 questions max). Each question must be self-contained — paste the"
    " relevant numbers directly into it.\n\n"
    "Format your response as one or more lines of:\n"
    "ASK: <single math question with numbers inline>\n\n"
    "Examples of useful questions:\n"
    "- 'Open=100, close=102.5. Percent change?'\n"
    "- 'Series [100, 101, 102, 103]: slope of linear trend?'\n"
    "- 'Series [100.1, 100.3, ...]: standard deviation?'\n\n"
    "If the data has no quantitative content worth analyzing, output ABSTAIN."
)


SYNTH_PROMPT = (
    "You are an interrogator herbivore. You read producer data and consult"
    " a math specialist for exact arithmetic. Below you have the original"
    " data, the questions you posed, and the specialist's answers. Now"
    " synthesize a final XML output.\n\n"
    "Output format (no extra text, just the XML):\n"
    "<synthesis kind=\"interrogator\">\n"
    "  <ticker>TICKER</ticker>\n"
    "  <bias>up|down|flat</bias>\n"
    "  <pct_move>+1.50</pct_move>\n"
    "  <signal>strong|moderate|weak</signal>\n"
    "  <confidence>0.7</confidence>\n"
    "  <abstain>false</abstain>\n"
    "</synthesis>\n\n"
    "Use the EXACT numbers the math specialist returned. Do not invent."
)


def _extract_questions(plan_text: str) -> list[str]:
    if "ABSTAIN" in plan_text.upper():
        return []
    asks = re.findall(r"ASK:\s*(.+?)(?:\n|$)", plan_text)
    return [a.strip() for a in asks if a.strip()]


def _render_inputs(candidates: list[Broadcast]) -> str:
    """Compact text rendering of producer broadcasts for the Interrogator
    to read. Pulls numeric payload directly when present.
    """
    lines = []
    for c in candidates:
        ticker = c.payload.get("ticker", "?")
        if c.agent_kind == "tickdelta":
            o = c.payload.get("open")
            h = c.payload.get("high")
            l = c.payload.get("low")
            cl = c.payload.get("close")
            v = c.payload.get("volume")
            lines.append(
                f"OHLCV {ticker} window={c.payload.get('window','1m')}: "
                f"open={o} high={h} low={l} close={cl} volume={v}"
            )
        elif c.agent_kind == "quote_series":
            hist = c.payload.get("history", [])
            horizon = c.payload.get("horizon", 12)
            # Truncate history representation for prompt size
            if len(hist) > 16:
                hist_str = (
                    "[" + ", ".join(f"{x:.2f}" for x in hist[:8]) +
                    f", ... {len(hist) - 16} values ..., " +
                    ", ".join(f"{x:.2f}" for x in hist[-8:]) + "]"
                )
            else:
                hist_str = "[" + ", ".join(f"{x:.2f}" for x in hist) + "]"
            lines.append(
                f"QUOTE_SERIES {ticker}: history={hist_str} horizon={horizon}"
            )
        else:
            # Generic fallback for other producer kinds
            payload_str = ", ".join(f"{k}={v}" for k, v in c.payload.items())
            lines.append(f"{c.agent_kind.upper()} {payload_str}")
    return "\n".join(lines) if lines else "<no data>"


def _render_findings(questions: list[str], results: list[MathResult]) -> str:
    if not questions:
        return "(no math findings — question phase produced no asks)"
    parts = ["MATH FINDINGS:"]
    for q, r in zip(questions, results):
        ans = r.answer or "?"
        parts.append(f"  Q: {q}")
        parts.append(f"  A: {ans}")
    return "\n".join(parts)


@dataclass
class InterrogatorHerbivore(BaseAgent):
    role: str = "herbivore"
    capacity: int = 6
    role_prefix: Optional[torch.Tensor] = None
    # Cross-model channel — set when use_channel=True. Bridges Qwen3-4B
    # hidden states ↔ Qwen2.5-Math input embeddings, bypassing tokens on
    # the internal Analyst → Math call.
    channel: Optional[CrossModelChannel] = None
    use_channel: bool = False
    _diet_tags: list[str] = field(default_factory=list)

    @classmethod
    def make(
        cls,
        kind: str = "interrogator",
        capacity: int = 6,
        use_channel: bool = False,
    ) -> "InterrogatorHerbivore":
        assert kind in INTERROGATOR_KINDS, kind
        h = cls(
            id=new_agent_id("herbivore", kind),
            kind=kind,
            capacity=capacity,
            use_channel=use_channel,
        )
        h._diet_tags = list(DIETS[kind])
        return h

    def ensure_initialized(
        self,
        host: ModelHost,
        seed_base: int | None = None,
        math_host: MathHost | None = None,
    ) -> None:
        if self.role_prefix is None:
            self.role_prefix = host.encode_role_prefix(
                "You are an interrogator herbivore. Plan questions, consult"
                " the math specialist, then synthesize."
            )
        if self.use_channel and self.channel is None:
            mh = math_host or MathHost.get()
            seed = (seed_base * 31 if seed_base is not None else None)
            self.channel = CrossModelChannel.make(
                analyst_dim=host.hidden_size,
                math_dim=mh.hidden_size,
                analyst_target_norm=host.input_embedding_norm(),
                math_target_norm=1.0,  # math model embedding norm — refined below
                seed=seed,
            )
            # Refine math-side target_norm by sampling embed norms
            try:
                with torch.no_grad():
                    w = mh.embed_layer.weight.detach().float()
                    n = float(w.norm(dim=-1).mean().item())
                self.channel.a_to_m.target_norm = n
            except Exception:
                pass

    @property
    def diet_tags(self) -> list[str]:
        return self._diet_tags

    def _solve_via_channel(
        self,
        question: str,
        host: ModelHost,
        math_host: MathHost,
    ) -> MathResult:
        """Token-bypass solve: embed the question via Qwen3-4B's input
        embedding layer, project through E_a→m, splice as Math model's
        prefix. Math forwards on those embeddings and decodes an answer.

        The math model still produces tokens at output — the bypass only
        removes Math's tokenizer from the question path. Output is
        decoded text, parsed the usual way.
        """
        if self.channel is None:
            return math_host.solve(question)

        # Embed question in Qwen3-4B's input space
        q_ids = host._tok(
            question, return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(host.device)
        q_emb_a = host._embed_layer(q_ids)[0].detach()  # [Q, hidden_a]

        # Project to Math's space
        q_emb_a = q_emb_a.to(
            device=self.channel.a_to_m.proj_in.weight.device,
            dtype=self.channel.a_to_m.proj_in.weight.dtype,
        )
        q_emb_m = self.channel.a_to_m(q_emb_a)         # [Q, hidden_m]

        # Build a small text-side prefix so Math has a system prompt in
        # its own native space (frame the task) — these are normal
        # Math-tokenizer embeddings, not projected.
        sys_text = "Solve the following exactly. Output: ANSWER: <number>"
        sys_emb_m = math_host.text_to_input_embeds(sys_text)
        sys_emb_m = sys_emb_m.to(device=q_emb_m.device, dtype=q_emb_m.dtype)

        # Concatenate [sys_emb_m ⊕ q_emb_m] as Math's input embeds.
        prefix = torch.cat([sys_emb_m, q_emb_m], dim=0)
        decoded, _last_hidden = math_host.forward_with_prefix_embeds(
            prefix_embeds=prefix,
            max_new_tokens=128,
            return_hidden=False,
        )

        # Parse the decoded answer
        from ..math_host import _parse_math_response
        return _parse_math_response(decoded)

    async def hunt_and_synthesize(
        self,
        candidates: list[Broadcast],
        tick: int,
        host: ModelHost,
        math_host: MathHost | None = None,
    ) -> tuple[Broadcast, list[Broadcast], list[Broadcast], dict]:
        self.ensure_initialized(host)
        math_host = math_host or MathHost.get()

        # Diet pre-filter (cheap re-check)
        edible = [
            c for c in candidates
            if any(t in self._diet_tags for t in c.diet_tags)
        ]

        # Limit by capacity (most recent first)
        edible.sort(key=lambda c: c.created_tick, reverse=True)
        meal = edible[: self.capacity]
        rejected = edible[self.capacity:]

        if not meal:
            # No diet-matching food → abstain.
            null_emb = torch.zeros(host.hidden_size, dtype=torch.float32).tolist()
            br = Broadcast(
                id=str(uuid.uuid4()),
                tier="herbivore_broadcast",
                agent_id=self.id,
                agent_kind=self.kind,
                parent_input_ids=[],
                diet_tags=[f"from_{self.kind}_herbivore"],
                payload={"abstain": True, "reason": "no diet-matching candidates"},
                decoded_text=emit_synthesis(kind="interrogator", abstain=True),
                channel_embedding=null_emb,
                created_tick=tick,
                abstained=True,
            )
            return br, [], rejected, {
                "abstain": True, "n_meal": 0, "avg_null_prob": 1.0,
            }

        # ------------------------------------------------------------------
        # Phase 1: PLAN (Qwen3-4B reads + decides what to ask)
        # ------------------------------------------------------------------
        rendered = _render_inputs(meal)
        try:
            plan = host.generate_chat(
                system=PLAN_PROMPT,
                user=f"DATA:\n{rendered}\n\nWhat should we compute?",
                max_new_tokens=256,
                temperature=0.0,
            )
        except Exception as e:
            plan = f"ABSTAIN ({e})"

        questions = _extract_questions(plan)

        # ------------------------------------------------------------------
        # Phase 2: SOLVE
        # ------------------------------------------------------------------
        # Two paths:
        #   - use_channel=False (default): tokens go through Math's tokenizer.
        #     Standard chat completion; question-text in, answer-text out.
        #   - use_channel=True: bypass Math's tokenizer. We embed the
        #     question with Qwen3-4B's input embeddings, project through
        #     E_a→m to land in Math's embedding space, and forward Math
        #     directly on those projected embeddings. Math never sees the
        #     question as tokens — only as Analyst-derived hidden states.
        results: list[MathResult] = []
        for q in questions[:3]:
            try:
                if self.use_channel and self.channel is not None:
                    r = self._solve_via_channel(q, host, math_host)
                else:
                    r = math_host.solve(q)
            except Exception as e:
                r = MathResult(work="", answer="", answer_num=None,
                               raw=f"[error] {e}")
            results.append(r)

        # ------------------------------------------------------------------
        # Phase 3: SYNTHESIZE (Qwen3-4B writes the XML output)
        # ------------------------------------------------------------------
        findings_text = _render_findings(questions, results)
        try:
            synthesis_text = host.generate_chat(
                system=SYNTH_PROMPT,
                user=(
                    f"DATA:\n{rendered}\n\n"
                    f"PLAN:\n{plan}\n\n"
                    f"{findings_text}\n\n"
                    "Now produce the synthesis XML."
                ),
                max_new_tokens=192,
                temperature=0.0,
            )
        except Exception as e:
            synthesis_text = emit_synthesis(kind="interrogator", abstain=True)

        # Truncate at first </synthesis> close tag.
        for stop in ("</synthesis>",):
            idx = synthesis_text.find(stop)
            if idx >= 0:
                synthesis_text = synthesis_text[: idx + len(stop)]
                break

        # Compute the broadcast's channel_embedding from Qwen forward on
        # the synthesis text — this is what downstream Channels attend to.
        try:
            pooled = host.text_to_hidden(synthesis_text, pool="mean")
            emb = pooled.detach().cpu().tolist()
        except Exception:
            emb = torch.zeros(host.hidden_size, dtype=torch.float32).tolist()

        br = Broadcast(
            id=str(uuid.uuid4()),
            tier="herbivore_broadcast",
            agent_id=self.id,
            agent_kind=self.kind,
            parent_input_ids=[c.id for c in meal],
            diet_tags=[f"from_{self.kind}_herbivore"],
            payload={
                "n_questions": len(questions),
                "questions": questions,
                "answers": [r.answer for r in results],
                "answer_nums": [r.answer_num for r in results],
            },
            decoded_text=synthesis_text,
            channel_embedding=emb,
            created_tick=tick,
        )
        return br, meal, rejected, {
            "abstain": False,
            "n_meal": len(meal),
            "n_questions": len(questions),
            "avg_null_prob": 0.0,
        }


# ── PM-mode wrapper (v4 firehose runner) ──────────────────────────────
#
# The training-time `InterrogatorHerbivore.hunt_and_synthesize` above is
# tightly coupled to in-process ModelHost + MathHost (Qwen3-4B in-proc +
# Qwen2.5-Math-1.5B in-proc) and produces a `Broadcast` for the trophic
# stack to attach to other herbivore channels.
#
# In PM mode (`scripts/run_firehose_loop.py --apex-pm --apex-via-vllm`)
# the apex runs through DSPy + vLLM, not ModelHost. There's no
# Broadcast channel — observations are plain dicts that the apex
# `WatchlistFromObservationsPM` signature reads. Loading a second
# in-process model (Qwen2.5-Math-1.5B) just for the interrogator
# doubles the GPU footprint with no gain, since the apex itself is
# off-process.
#
# `PMInterrogator.predict_only` therefore reimplements the
# plan→solve→synthesize loop on top of DSPy LMs. Same conceptual
# pipeline — different transport. If anything in the toolchain is
# unhealthy (vLLM down, MathHost OOM, DSPy adapter error), the
# `predict_only` returns a no-op synthesis dict with `abstain=True`
# and a clear `reason` so the sweep continues without crashing.

import dspy as _dspy_for_pm
from pydantic import BaseModel as _BaseModel
from pydantic import Field as _Field


class _InterrogatorSynthesisOut(_BaseModel):
    """Pydantic output for the PM-mode interrogator synthesis call.

    Field choices mirror the training-time XML synthesis schema (ticker,
    bias, pct_move, signal, confidence, abstain) but are typed for DSPy
    consumption so the runner doesn't have to parse free-text XML.
    """
    ticker: str = _Field(default="", description="Focus ticker; empty if no single focus")
    bias: str = _Field(default="flat", description="up | down | flat")
    pct_move: float = _Field(default=0.0, description="Signed percentage move; e.g. +1.50 or -0.75")
    signal: str = _Field(default="weak", description="strong | moderate | weak")
    confidence: float = _Field(default=0.5, ge=0.0, le=1.0)
    abstain: bool = _Field(default=False)
    rationale: str = _Field(default="", description="One-paragraph quantitative rationale")


class _PMInterrogatorPlanSig(_dspy_for_pm.Signature):
    """PM-mode interrogator (planner phase).

    You are a quantitative-analysis planner. Given carnivore observations
    + focus tickers, decide what would be useful to compute (1-3 questions
    max). Each question must be self-contained — paste the relevant
    numbers from the observations directly into it.

    Format: one or more `ASK: <question with numbers inline>` lines, or
    the single word `ABSTAIN` if there's nothing quantitative to compute.
    """
    focus_tickers: list[str] = _dspy_for_pm.InputField()
    observations_summary: str = _dspy_for_pm.InputField(
        desc="Compact text rendering of the carnivore observations + key bars."
    )
    plan: str = _dspy_for_pm.OutputField(
        desc="One or more 'ASK: <question>' lines OR 'ABSTAIN'."
    )


class _PMInterrogatorSolveSig(_dspy_for_pm.Signature):
    """PM-mode interrogator (solver phase).

    You are a math specialist. Given a single quantitative question
    with the numbers inlined, answer it precisely. Output one line:
    `ANSWER: <number-with-units>`. No prose.
    """
    question: str = _dspy_for_pm.InputField()
    answer: str = _dspy_for_pm.OutputField(
        desc="`ANSWER: <number-with-units>` on one line."
    )


class _PMInterrogatorSynthesisSig(_dspy_for_pm.Signature):
    """PM-mode interrogator (synthesis phase).

    You are an interrogator herbivore. You read carnivore observations
    and consulted a math specialist for exact arithmetic. Synthesize a
    structured quantitative view: bias (up / down / flat), expected
    pct_move (signed), signal strength, and confidence. Cite the math
    findings — do not invent numbers. If the data is too thin or the
    math findings are inconsistent, set `abstain=true` and explain.
    """
    focus_tickers: list[str] = _dspy_for_pm.InputField()
    observations_summary: str = _dspy_for_pm.InputField()
    plan: str = _dspy_for_pm.InputField()
    findings: str = _dspy_for_pm.InputField(
        desc="MATH FINDINGS — one Q/A pair per question, or '(no findings)'."
    )
    synthesis: _InterrogatorSynthesisOut = _dspy_for_pm.OutputField()


def _render_observations_for_interrogator(
    observations: list[dict], focus_tickers: list[str], max_obs: int = 10,
) -> str:
    """Compact text rendering of carnivore observations + focus tickers,
    sized for the planner prompt (~1.5K chars).
    """
    if not observations:
        return "(no observations)"
    lines = []
    if focus_tickers:
        lines.append(f"FOCUS: {','.join(focus_tickers)}")
    for o in observations[:max_obs]:
        # `Observation.to_dict()` shape used by carnivore.aggregate:
        # {ticker, outcome_id, p_up, magnitude, reasoning, …}
        tk = o.get("ticker") or o.get("outcome_id", "?")
        p_up = o.get("p_up")
        mag = o.get("magnitude")
        reason = (o.get("reasoning") or o.get("statement") or "")[:120]
        parts = [str(tk)]
        if p_up is not None:
            parts.append(f"p_up={p_up:.2f}")
        if mag is not None:
            parts.append(f"mag={mag:+.3f}")
        if reason:
            parts.append(reason)
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _extract_pm_questions(plan_text: str, max_q: int = 3) -> list[str]:
    if not plan_text:
        return []
    if "ABSTAIN" in plan_text.upper() and "ASK:" not in plan_text.upper():
        return []
    asks = re.findall(r"ASK:\s*(.+?)(?:\n|$)", plan_text)
    return [a.strip() for a in asks if a.strip()][:max_q]


@dataclass
class PMInterrogator:
    """Lightweight, in-PM-pipeline interrogator herbivore.

    Built once per run (in `scripts/run_firehose_loop.py`); invoked per
    day per ECO-* path before the apex sees the observations. Returns a
    structured synthesis dict that the runner appends to the
    observations list as one additional `kind="interrogator"`
    observation.

    Never raises out of `predict_only`. If a phase fails, it logs and
    falls back to abstain-with-reason; the sweep keeps going.

    Constructor flag `enabled=False` is the explicit no-op mode
    (e.g. for BARE/ORACLE paths, or when MathHost/vLLM is unhealthy
    and the operator chose to skip the interrogator entirely).
    """
    enabled: bool = True
    label: str = "interrogator"
    # Capacity of the planner: max number of math questions per day.
    max_questions: int = 3
    # If `solver_lm` is None we use the same `dspy.context` LM as the
    # planner / synthesizer — i.e. the active apex LM. For a separate
    # Qwen2.5-Math-1.5B endpoint, set `solver_lm` to a dspy.LM and the
    # solve calls will route through it via `dspy.context(lm=solver_lm)`.
    solver_lm: Optional["object"] = None  # dspy.LM or None

    def predict_only(
        self,
        observations: list[dict],
        focus_tickers: list[str],
        *,
        date: str = "",
    ) -> dict:
        """Run plan → solve → synthesize against the active DSPy LM
        (and `solver_lm` for the math step if set). Returns a dict
        that the runner appends to the observations list.

        Return shape always has at minimum: `{kind: "interrogator",
        abstain: bool, reason: str}`. On success also carries
        `ticker / bias / pct_move / signal / confidence / rationale /
        n_questions / questions / answers`.
        """
        if not self.enabled:
            return {
                "kind": "interrogator",
                "ticker": "",
                "abstain": True,
                "reason": "interrogator disabled (--no-interrogator or fallback)",
            }
        if not observations:
            return {
                "kind": "interrogator",
                "ticker": "",
                "abstain": True,
                "reason": "no observations to interrogate",
            }

        obs_summary = _render_observations_for_interrogator(
            observations, focus_tickers,
        )

        # ── Plan ──────────────────────────────────────────────────────
        try:
            plan_pred = _dspy_for_pm.Predict(_PMInterrogatorPlanSig)(
                focus_tickers=list(focus_tickers or []),
                observations_summary=obs_summary,
            )
            plan_text = (getattr(plan_pred, "plan", "") or "").strip()
        except Exception as e:
            return {
                "kind": "interrogator",
                "ticker": "",
                "abstain": True,
                "reason": f"plan-phase error: {type(e).__name__}: {str(e)[:160]}",
            }

        questions = _extract_pm_questions(plan_text, max_q=self.max_questions)
        if not questions:
            return {
                "kind": "interrogator",
                "ticker": "",
                "abstain": True,
                "reason": "planner abstained (no math-worthy questions)",
                "plan": plan_text[:240],
            }

        # ── Solve ─────────────────────────────────────────────────────
        answers: list[str] = []
        solver_lm = self.solver_lm
        for q in questions:
            try:
                if solver_lm is not None:
                    with _dspy_for_pm.context(lm=solver_lm):
                        a_pred = _dspy_for_pm.Predict(_PMInterrogatorSolveSig)(question=q)
                else:
                    a_pred = _dspy_for_pm.Predict(_PMInterrogatorSolveSig)(question=q)
                answers.append((getattr(a_pred, "answer", "") or "").strip())
            except Exception as e:  # one bad solve doesn't kill synthesis
                answers.append(f"(solver error: {type(e).__name__})")

        findings_lines = ["MATH FINDINGS:"]
        for q, a in zip(questions, answers):
            findings_lines.append(f"  Q: {q}")
            findings_lines.append(f"  A: {a}")
        findings = "\n".join(findings_lines)

        # ── Synthesize ────────────────────────────────────────────────
        try:
            synth_pred = _dspy_for_pm.Predict(_PMInterrogatorSynthesisSig)(
                focus_tickers=list(focus_tickers or []),
                observations_summary=obs_summary,
                plan=plan_text,
                findings=findings,
            )
            synth = getattr(synth_pred, "synthesis", None)
            if synth is None:
                raise ValueError("DSPy returned no synthesis field")
            # synth may be the Pydantic model or a dict (depending on adapter)
            if hasattr(synth, "model_dump"):
                synth_dict = synth.model_dump()
            elif isinstance(synth, dict):
                synth_dict = dict(synth)
            else:
                synth_dict = _InterrogatorSynthesisOut().model_dump()
        except Exception as e:
            return {
                "kind": "interrogator",
                "ticker": "",
                "abstain": True,
                "reason": f"synthesis-phase error: {type(e).__name__}: {str(e)[:160]}",
                "plan": plan_text[:240],
                "n_questions": len(questions),
                "questions": questions,
                "answers": answers,
            }

        return {
            "kind": "interrogator",
            "ticker": synth_dict.get("ticker", ""),
            "bias": synth_dict.get("bias", "flat"),
            "pct_move": float(synth_dict.get("pct_move", 0.0)),
            "signal": synth_dict.get("signal", "weak"),
            "confidence": float(synth_dict.get("confidence", 0.5)),
            "abstain": bool(synth_dict.get("abstain", False)),
            "rationale": (synth_dict.get("rationale") or "")[:300],
            "n_questions": len(questions),
            "questions": questions,
            "answers": answers,
            "reason": "",
        }


def make_pm_interrogator(
    enabled: bool = True,
    solver_lm=None,
    label: str = "interrogator",
) -> "PMInterrogator":
    """Factory used by `scripts/run_firehose_loop.py`. Never raises.

    Pass `enabled=False` to get a no-op interrogator that always returns
    an abstain dict (useful when the operator wants to disable it
    without removing the wiring). When MathHost/vLLM health checks fail
    upstream, the runner should call this factory with `enabled=False`
    and log a warning rather than crash the sweep.
    """
    return PMInterrogator(enabled=enabled, label=label, solver_lm=solver_lm)


# Re-export so `from trophic.agents.interrogator_herbivore import …`
# picks up both the training-time and PM-mode entry points.
__all__ = [
    "InterrogatorHerbivore",
    "PMInterrogator",
    "make_pm_interrogator",
]
