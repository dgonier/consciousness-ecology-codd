"""Per-agent feedback derived by the decomposer from KG observations.

Hexis principle: every agent gets its OWN modulation, not a shared one.
The decomposer reads each agent's history (which scenarios it saw, what
it said, what was right) and derives:

  - prompt_modulation: a natural-language addendum the orchestrator
    appends to that agent's prompt next round. Tailored to the agent's
    recent failure modes / strengths.

  - diet_adjustments: per-tag scalar weights ∈ [0, 1]. The agent should
    weight upstream broadcasts with these tags more (>1) or less (<1).
    Used by trough-attention or by prompt-level "pay attention to X"
    instructions for API voters.

  - confidence_calibration: scalar bias to add/subtract from the agent's
    self-reported confidence. Use when the agent is systematically over-
    or under-confident.

  - m_tensor_hint: per-layer M-tensor delta for trained-LLM agents that
    have phi_mlp hooks installed. Optional; None for API voters that
    don't have a weight-modulation surface.

Feedback is REGENERATED each session (not accumulated indefinitely)
from the rolling KG window. This way an agent that fixes its mistakes
isn't permanently penalized for past failures.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .observation import Observation, ObservationWriter


@dataclass
class AgentFeedback:
    """Per-agent modulation packet. Decomposer outputs one of these per
    agent per session-boundary; orchestrator injects into next round."""
    agent_id: str
    prompt_modulation: str = ""
    diet_adjustments: dict[str, float] = field(default_factory=dict)
    confidence_calibration: float = 0.0  # added to self-reported confidence
    m_tensor_hint: dict | None = None    # serialized {layer: tensor} or None
    # Provenance: which observations informed this feedback (rolling window).
    derived_from: dict = field(default_factory=dict)


@dataclass
class FeedbackDeriver:
    """Reads observations, derives per-agent feedback packets.

    Three signals per agent over the rolling window:
      - SOLO accuracy by ticker / by scenario condition
      - over/under-confidence (does its self-confidence track its
        actual hit rate?)
      - per-citation accuracy (when it cites 'drift' is it more or less
        often right?)

    From those signals, the deriver writes:
      - prompt_modulation pointing out the strongest pattern
      - diet_adjustments biasing toward citations that worked
      - confidence_calibration zeroing out systematic bias
    """
    window: int = 200
    over_confidence_threshold: float = 0.10  # |actual - reported| >0.10 → adjust
    min_seen: int = 10                        # don't write hints with too little data

    def derive_all(
        self, observations: list[Observation],
    ) -> dict[str, AgentFeedback]:
        recent = observations[-self.window :]
        # collect per-agent samples
        per_agent: dict[str, list[dict]] = defaultdict(list)
        for obs in recent:
            target = obs.target_direction
            if target not in ("up", "down"):
                continue
            for v in obs.voter_responses:
                if v.get("direction") not in ("up", "down"):
                    continue
                ticker = obs.ticker
                per_agent[v["voter_id"]].append({
                    "scenario": obs.scenario_name,
                    "ticker": ticker,
                    "target": target,
                    "direction": v["direction"],
                    "correct": v["direction"] == target,
                    "confidence": v.get("confidence"),
                    "citations": v.get("evidence_citations") or [],
                    "reasoning": v.get("reasoning", ""),
                })
        out: dict[str, AgentFeedback] = {}
        for agent_id, samples in per_agent.items():
            if len(samples) < self.min_seen:
                continue
            out[agent_id] = self._derive_one(agent_id, samples)
        return out

    def _derive_one(self, agent_id: str, samples: list[dict]) -> AgentFeedback:
        n = len(samples)
        n_correct = sum(1 for s in samples if s["correct"])
        actual_acc = n_correct / n

        # per-ticker accuracy
        per_ticker = defaultdict(lambda: [0, 0])
        for s in samples:
            per_ticker[s["ticker"]][1] += 1
            if s["correct"]:
                per_ticker[s["ticker"]][0] += 1

        weakest_ticker, weakest_acc = None, 1.0
        strongest_ticker, strongest_acc = None, 0.0
        for tk, (c, t) in per_ticker.items():
            if t < 3:
                continue
            acc = c / t
            if acc < weakest_acc:
                weakest_acc = acc
                weakest_ticker = tk
            if acc > strongest_acc:
                strongest_acc = acc
                strongest_ticker = tk

        # Citation effectiveness
        citation_effect = defaultdict(lambda: [0, 0])
        for s in samples:
            for c in s["citations"]:
                citation_effect[c][1] += 1
                if s["correct"]:
                    citation_effect[c][0] += 1
        good_citations = []
        bad_citations = []
        for cite, (c, t) in citation_effect.items():
            if t < 3:
                continue
            r = c / t
            if r >= 0.6:
                good_citations.append((cite, r))
            elif r <= 0.4:
                bad_citations.append((cite, r))

        # Confidence calibration
        rep_confs = [s["confidence"] for s in samples if s["confidence"] is not None]
        avg_rep = sum(rep_confs) / len(rep_confs) if rep_confs else None
        cal_adjust = 0.0
        if avg_rep is not None and abs(avg_rep - actual_acc) > self.over_confidence_threshold:
            # If reported > actual, we want to downshift confidence
            cal_adjust = actual_acc - avg_rep

        # Diet adjustments — favor good citations, dampen bad
        diet = {}
        for cite, _ in good_citations:
            diet[f"cite.{cite}"] = 1.2
        for cite, _ in bad_citations:
            diet[f"cite.{cite}"] = 0.8

        # Prompt modulation — natural-language hint
        bullets = []
        if weakest_ticker and weakest_acc < 0.40:
            bullets.append(
                f"You've been weak on {weakest_ticker} (acc {weakest_acc:.0%}); "
                f"approach those scenarios with extra skepticism."
            )
        if strongest_ticker and strongest_acc > 0.60:
            bullets.append(
                f"You've done well on {strongest_ticker} (acc {strongest_acc:.0%}); "
                f"trust your read there."
            )
        if good_citations:
            cites = ", ".join(c for c, _ in good_citations[:3])
            bullets.append(f"Citations that have correlated with correct answers: {cites}.")
        if bad_citations:
            cites = ", ".join(c for c, _ in bad_citations[:3])
            bullets.append(f"Citations that have correlated with wrong answers: {cites}.")
        if avg_rep is not None and cal_adjust < -0.05:
            bullets.append(
                f"Your reported confidence ({avg_rep:.2f}) has overshot your actual"
                f" hit rate ({actual_acc:.2f}); be more cautious."
            )
        elif avg_rep is not None and cal_adjust > 0.05:
            bullets.append(
                f"Your reported confidence ({avg_rep:.2f}) has undershot your actual"
                f" hit rate ({actual_acc:.2f}); trust your reads more."
            )

        prompt_mod = ""
        if bullets:
            prompt_mod = (
                "DECOMPOSER FEEDBACK (from your prior runs on similar scenarios):\n"
                + "\n".join(f"  - {b}" for b in bullets)
                + "\n"
            )

        return AgentFeedback(
            agent_id=agent_id,
            prompt_modulation=prompt_mod,
            diet_adjustments=diet,
            confidence_calibration=cal_adjust,
            m_tensor_hint=None,  # only populated for trained-LLM agents
            derived_from={
                "n_observations": n,
                "actual_acc": actual_acc,
                "n_per_ticker": {k: v[1] for k, v in per_ticker.items()},
                "weakest_ticker": weakest_ticker,
                "strongest_ticker": strongest_ticker,
            },
        )


@dataclass
class FeedbackStore:
    """Persistent per-agent feedback. Lives at
    external/decomposer_kg/feedback/<agent_id>.json. Orchestrator reads
    these at session start; deriver overwrites them each session."""
    root: Path = Path("external/decomposer_kg/feedback")

    def __post_init__(self):
        if isinstance(self.root, str):
            self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, fb: AgentFeedback) -> Path:
        import json
        from dataclasses import asdict
        # Path-safe agent id
        safe = fb.agent_id.replace("/", "_").replace(":", "_")
        p = self.root / f"{safe}.json"
        p.write_text(json.dumps(asdict(fb), indent=2))
        return p

    def write_all(self, feedbacks: dict[str, AgentFeedback]) -> dict[str, Path]:
        return {aid: self.write(fb) for aid, fb in feedbacks.items()}

    def read(self, agent_id: str) -> AgentFeedback | None:
        import json
        safe = agent_id.replace("/", "_").replace(":", "_")
        p = self.root / f"{safe}.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return AgentFeedback(**d)

    def read_all(self) -> dict[str, AgentFeedback]:
        out: dict[str, AgentFeedback] = {}
        for p in self.root.glob("*.json"):
            import json
            d = json.loads(p.read_text())
            out[d["agent_id"]] = AgentFeedback(**d)
        return out
