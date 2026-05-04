"""Per-agent feedback derived by the decomposer from KG observations.

Hexis principle: every agent gets its OWN modulation, not a shared one.
This module covers ALL agent roles, not just apex voters:

  - producer (tickdelta, anomaly, disclosure, social_signal, quote_series)
  - herbivore (technical, fundamental, forecaster, interrogator)
  - apex_voter (OpenAI / Anthropic / Gemini / OpenRouter / local Qwen)
  - apex_aggregator (the soft-vote / agree-gate logic itself)
  - decomposer (self-feedback: am I deriving good guidance?)

Each role receives different feedback fields depending on what's
configurable about that agent:

  - prompt_modulation: NL addendum prepended to that agent's
    instructions. Used by herbivores (role_prefix prepend) and apex
    voters (system-message prepend).

  - diet_adjustments: per-tag scalar weights ∈ [0.5, 1.5]. The agent
    should weight upstream broadcasts with these tags more (>1) or less
    (<1). Used by trough-attention agents (predator + herbivores).

  - confidence_calibration: scalar bias added to the agent's self-
    reported confidence. Used by apex voters and the apex aggregator.

  - producer_config: per-producer config overrides — `{"pool":
    "last", "numeric_inject": true}`. Used by producer agents only.

  - aggregator_strategy: which strategy the apex aggregator should run
    by default — `plurality` | `confidence_weighted` |
    `perplexity_weighted` | `agree_gate`. Used by apex_aggregator.

  - m_tensor_hint: per-layer M-tensor delta for trained-LLM agents that
    have phi_mlp hooks. Used by herbivores (when ckpt loaded) and the
    apex predator's binary head. None for API voters.

Feedback is REGENERATED each session (not accumulated indefinitely)
from the rolling KG window. An agent that fixes its mistakes isn't
permanently penalized for past failures.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .observation import Observation, ObservationWriter


AgentRole = str  # 'producer' | 'herbivore' | 'apex_voter' | 'apex_aggregator' | 'decomposer'


@dataclass
class AgentFeedback:
    """Per-agent modulation packet. Decomposer outputs one of these per
    agent per session-boundary; orchestrator injects into next round.

    Different roles consume different fields:
      - producer:         producer_config
      - herbivore:        prompt_modulation, diet_adjustments, m_tensor_hint
      - apex_voter:       prompt_modulation, confidence_calibration
      - apex_aggregator:  aggregator_strategy, confidence_calibration
      - decomposer:       prompt_modulation (self-feedback)

    Unused fields stay at defaults — the agent ignores fields irrelevant
    to its role.
    """
    agent_id: str
    role: AgentRole = "apex_voter"   # backward-compat default
    # Apex voter / herbivore / decomposer
    prompt_modulation: str = ""
    # Herbivore / predator
    diet_adjustments: dict[str, float] = field(default_factory=dict)
    # Apex voter / apex aggregator
    confidence_calibration: float = 0.0
    # Trained-LLM agents (herbivore, apex predator binary head)
    m_tensor_hint: dict | None = None
    # Producer-only: pool strategy, numeric_inject, etc.
    producer_config: dict = field(default_factory=dict)
    # Apex-aggregator-only: which strategy to run by default
    aggregator_strategy: str | None = None
    # Provenance: which observations informed this feedback.
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
        """Derive feedback for ALL agent roles, not just voters.

        Returns one AgentFeedback per (agent_id, role) pair. Producers,
        herbivores, voters, and the aggregator are all covered.
        """
        recent = observations[-self.window :]
        out: dict[str, AgentFeedback] = {}
        # APEX VOTERS — derived from voter_responses on each Observation
        out.update(self._derive_voters(recent))
        # PRODUCERS — derived from inter_tier_signals (tier 1)
        out.update(self._derive_producers(recent))
        # HERBIVORES — derived from inter_tier_signals (tier 3)
        # Currently empty under the apex-only pipeline, but will populate
        # when a trained ckpt is wired into the voter path.
        out.update(self._derive_herbivores(recent))
        # APEX AGGREGATOR — derived from per-strategy ensemble outcomes
        agg = self._derive_aggregator(recent)
        if agg is not None:
            out[agg.agent_id] = agg
        return out

    def _derive_voters(self, recent: list[Observation]) -> dict[str, AgentFeedback]:
        per_agent: dict[str, list[dict]] = defaultdict(list)
        for obs in recent:
            target = obs.target_direction
            if target not in ("up", "down"):
                continue
            for v in obs.voter_responses:
                if v.get("direction") not in ("up", "down"):
                    continue
                per_agent[v["voter_id"]].append({
                    "scenario": obs.scenario_name,
                    "ticker": obs.ticker,
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
            out[agent_id] = self._derive_one_voter(agent_id, samples)
        return out

    def _derive_producers(self, recent: list[Observation]) -> dict[str, AgentFeedback]:
        """Per-producer feedback — measures whether each producer's
        signal correlated with correct ensemble decisions.

        For each tier-1 NodeSignal id (e.g. producer.ohlcv.AAPL,
        producer.forecast.AAPL), track the ensemble's correctness across
        scenarios where that producer was active. Producers whose
        presence correlates with bad outcomes get config nudges (try
        pool=last, try larger numeric inject, etc.).
        """
        per_producer: dict[str, list[dict]] = defaultdict(list)
        for obs in recent:
            target = obs.target_direction
            if target not in ("up", "down"):
                continue
            ens_dir = obs.ensemble.get("direction")
            ens_correct = (ens_dir == target) if ens_dir else None
            for ns in obs.inter_tier_signals:
                if ns.tier != 1:
                    continue
                # Strip per-ticker suffix to get the producer kind
                # ("producer.ohlcv.AAPL" → "producer.ohlcv")
                kind_id = ".".join(ns.node_id.split(".")[:2])
                per_producer[kind_id].append({
                    "ensemble_correct": ens_correct,
                    "scenario": obs.scenario_name,
                    "ticker": obs.ticker,
                    "norm": ns.norm,
                })
        out: dict[str, AgentFeedback] = {}
        for prod_id, samples in per_producer.items():
            if len(samples) < self.min_seen:
                continue
            n = len(samples)
            n_corr = sum(1 for s in samples if s.get("ensemble_correct"))
            corr_rate = n_corr / n
            cfg = {}
            bullets = []
            # The producer can't directly cause/fix correctness, but if
            # the ensemble is wrong much more often than the prior
            # suggests, signal that the producer's broadcast may be
            # uninformative or misleading.
            if corr_rate < 0.40:
                bullets.append(
                    f"Ensemble correctness is low ({corr_rate:.0%}) on scenarios"
                    f" where this producer is active. Consider pool='last' or"
                    f" stronger numeric injection."
                )
                cfg["pool"] = "last"
                cfg["numeric_inject_scale"] = 1.5
            out[prod_id] = AgentFeedback(
                agent_id=prod_id,
                role="producer",
                prompt_modulation="\n".join(bullets) if bullets else "",
                producer_config=cfg,
                derived_from={"n_observations": n, "ensemble_correctness": corr_rate},
            )
        return out

    def _derive_herbivores(self, recent: list[Observation]) -> dict[str, AgentFeedback]:
        """Per-herbivore feedback. Today the apex-only voter pipeline
        doesn't capture tier-3 herb broadcasts in Observations, so this
        returns empty. When a trained ckpt is wired in (so herb
        broadcasts get logit-lensed and stored as tier-3 NodeSignals),
        this will derive role_prefix prompt modulation + diet
        adjustments per herb species."""
        per_herb: dict[str, list[dict]] = defaultdict(list)
        for obs in recent:
            for ns in obs.inter_tier_signals:
                if ns.tier != 3:
                    continue
                per_herb[ns.node_id].append({
                    "scenario": obs.scenario_name,
                    "target": obs.target_direction,
                    "ensemble_correct": (
                        obs.ensemble.get("direction") == obs.target_direction
                        if obs.ensemble.get("direction") else None
                    ),
                })
        out: dict[str, AgentFeedback] = {}
        for herb_id, samples in per_herb.items():
            if len(samples) < self.min_seen:
                continue
            n = len(samples)
            n_corr = sum(1 for s in samples if s.get("ensemble_correct"))
            corr_rate = n_corr / n
            bullets = []
            if corr_rate < 0.40:
                bullets.append(
                    "When you contributed, ensemble was wrong"
                    f" {1-corr_rate:.0%} of the time. Increase weight on"
                    " quantitative features over text-derived sentiment."
                )
            out[herb_id] = AgentFeedback(
                agent_id=herb_id,
                role="herbivore",
                prompt_modulation="\n".join(bullets) if bullets else "",
                derived_from={"n_observations": n, "ensemble_correctness": corr_rate},
            )
        return out

    def _derive_aggregator(self, recent: list[Observation]) -> AgentFeedback | None:
        """Aggregator feedback — pick the best aggregation strategy.

        For each strategy (plurality, confidence_weighted,
        perplexity_weighted), compute MCC over the rolling window. The
        best strategy becomes the suggested default.
        """
        from collections import defaultdict as _dd
        decisions = _dd(list)  # method → list[(target, predicted)]
        for obs in recent:
            t = obs.target_direction
            if t not in ("up", "down"):
                continue
            # The Observation today only stores the plurality decision.
            # Future: capture all three strategies' decisions per
            # scenario so the aggregator's feedback is real. For now we
            # have only one method to evaluate.
            method = obs.ensemble.get("method", "plurality")
            decisions[method].append((t, obs.ensemble.get("direction")))
        if not decisions:
            return None
        # Score each method; pick the best by MCC.
        best_method, best_mcc = None, -2.0
        method_scores: dict[str, dict] = {}
        for method, decs in decisions.items():
            if len(decs) < self.min_seen:
                continue
            tp = tn = fp = fn = 0
            for tgt, pred in decs:
                if pred == "up" and tgt == "up": tp += 1
                elif pred == "down" and tgt == "down": tn += 1
                elif pred == "up" and tgt == "down": fp += 1
                elif pred == "down" and tgt == "up": fn += 1
            denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            mcc = (tp * tn - fp * fn) / max(denom_sq ** 0.5, 1e-9) if denom_sq else 0
            method_scores[method] = {
                "mcc": mcc, "n": len(decs), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            }
            if mcc > best_mcc:
                best_mcc = mcc
                best_method = method
        if best_method is None:
            return None
        bullets = [
            f"Best aggregation strategy on recent window: {best_method}"
            f" (MCC {best_mcc:+.3f})."
        ]
        # Calibrate ensemble confidence if it skews
        return AgentFeedback(
            agent_id="apex.aggregator",
            role="apex_aggregator",
            prompt_modulation="\n".join(bullets),
            aggregator_strategy=best_method,
            derived_from={"method_scores": method_scores},
        )

    def _derive_one_voter(self, agent_id: str, samples: list[dict]) -> AgentFeedback:
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
            role="apex_voter",
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
