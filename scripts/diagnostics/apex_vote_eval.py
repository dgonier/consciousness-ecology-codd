"""Run apex-voter ensemble across StockNet test scenarios.

Strategy:
  1. For each scenario, build an EvidencePacket (producer + herb +
     forecast snapshot rendered to text, with metadata).
  2. Each available voter (OpenAI / Anthropic / Gemini / OpenRouter /
     local Qwen) emits (direction, confidence, perplexity).
  3. Aggregate via plurality / confidence-weighted / perplexity-weighted
     (rank-vote on binary is just plurality).
  4. Score MCC against the StockNet labels.

Currently auto-skips voters whose API key isn't set, so today's run
will only use the local Qwen voter — but the architecture is in place
for the day API keys arrive.

Usage:
    .venv/bin/python -u scripts/diagnostics/apex_vote_eval.py \\
        --n-per-ticker 20 --tickers AAPL,GOOG,MSFT,AMZN,JPM
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Default to having Chronos features available so the evidence packet
# includes the forecast snapshot block.
os.environ.setdefault("TROPHIC_COMPUTE_FORECAST", "1")

from trophic.apex_voters import (
    AnthropicVoter, GeminiVoter, LocalQwenVoter, OpenAIVoter, OpenRouterVoter,
    build_evidence_packet, confidence_weighted, deliberation_packet,
    perplexity_weighted, plurality,
    build_panel_from_registry,
)
from trophic.decomposers import (
    FitnessTracker, KGWriter, PopulationManager,
    Observation, ObservationWriter, DecomposerJudgment,
    capture_evidence_signals,
    SpeciesRegistry, bootstrap_default_panel,
)
from trophic.decomposers.agent_feedback import (
    FeedbackDeriver, FeedbackStore, AgentFeedback,
)
from trophic.training.stocknet_loader import build_stocknet_scenarios
from trophic.training.xml_schema import parse_prediction


def mcc(tp, tn, fp, fn):
    n = tp + tn + fp + fn
    if n == 0:
        return 0.0
    return (tp * tn - fp * fn) / max(
        math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), 1e-9,
    )


def score_decisions(decisions):
    tp = tn = fp = fn = ab = 0
    for tgt, pred in decisions:
        if pred is None:
            ab += 1
            continue
        if pred == "up" and tgt == "up":
            tp += 1
        elif pred == "down" and tgt == "down":
            tn += 1
        elif pred == "up" and tgt == "down":
            fp += 1
        elif pred == "down" and tgt == "up":
            fn += 1
    dec = tp + tn + fp + fn
    return {
        "n": len(decisions), "decided": dec, "abstained": ab,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "acc": (tp + tn) / max(dec, 1),
        "mcc": mcc(tp, tn, fp, fn),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-ticker", type=int, default=20)
    ap.add_argument("--tickers", default="AAPL,GOOG,MSFT,AMZN,JPM")
    ap.add_argument("--out", default="logs/apex_vote_eval.jsonl")
    ap.add_argument("--max-scenarios", type=int, default=0,
                    help="0 = no cap; otherwise stop after this many")
    ap.add_argument("--decomposer", action="store_true",
                    help="Track per-voter fitness, write KG, emit evolution report at end.")
    ap.add_argument("--kg-path", default="external/decomposer_kg/kg.jsonl")
    ap.add_argument("--obs-path", default="external/decomposer_kg/observations.jsonl",
                    help="Full per-scenario observation: all inter-tier signals + voter reasoning + ensemble + judgments.")
    ap.add_argument("--feedback-root", default="external/decomposer_kg/feedback",
                    help="Per-agent feedback store. If files exist for a voter, that voter sees its own feedback in this run's prompts.")
    ap.add_argument("--no-feedback", action="store_true",
                    help="Skip reading per-agent feedback into prompts (useful for ablation).")
    ap.add_argument("--species-path", default="external/decomposer_kg/species.jsonl",
                    help="KG-backed species registry. Roster compiled at start.")
    ap.add_argument("--cycles", type=int, default=1,
                    help="Number of evolutionary cycles. Decomposer fires evolution updates between cycles.")
    ap.add_argument("--passes-per-cycle", type=int, default=0,
                    help="Scenarios per pass × passes-per-cycle = scenarios per cycle. 0 = use --max-scenarios as the cycle length.")
    args = ap.parse_args()

    print(f"[apex_vote] tickers={args.tickers}, n_per_ticker={args.n_per_ticker}")
    print(f"[apex_vote] cycles={args.cycles}, passes/cycle={args.passes_per_cycle or 'auto'}")

    # KG-driven panel: read alive species from the registry, build voter
    # for each. Bootstrap default 4-voter panel if registry is empty.
    registry = SpeciesRegistry(path=Path(args.species_path))
    seeded = bootstrap_default_panel(registry)
    if seeded:
        print(f"[apex_vote] bootstrapped {len(seeded)} default species → {registry.path}")
    voters = build_panel_from_registry(registry)
    print(f"[apex_vote] compiled panel: {[v.voter_id for v in voters]}")
    if not voters:
        print("[apex_vote] no voters available; aborting"); return 1

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    scens = build_stocknet_scenarios(
        split="test", tickers=tickers, max_per_ticker=args.n_per_ticker,
        compute_forecast=True,
    )
    if args.max_scenarios:
        scens = scens[: args.max_scenarios]
    print(f"[apex_vote] {len(scens)} scenarios")

    # Per-strategy decision lists (as we iterate)
    plur_decs, conf_decs, ppl_decs = [], [], []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("w")

    # Decomposer: KG writer + fitness tracker + population manager +
    # full Observation writer (every inter-tier signal + voter reasoning).
    kg = None
    obs_writer = None
    tracker = None
    if args.decomposer:
        kg = KGWriter(path=Path(args.kg_path))
        kg.open()
        obs_writer = ObservationWriter(Path(args.obs_path))
        obs_writer.open()
        tracker = FitnessTracker(window=200)
        print(f"[apex_vote] decomposer ON: KG={kg.path}, obs={obs_writer.path}, fitness window=200")

    # Per-agent feedback (Hexis-style: each agent gets its own modulation).
    feedback_store = None
    feedback_by_agent: dict = {}
    if not args.no_feedback:
        feedback_store = FeedbackStore(root=Path(args.feedback_root))
        feedback_by_agent = feedback_store.read_all()
        if feedback_by_agent:
            print(f"[apex_vote] loaded per-agent feedback for: {list(feedback_by_agent.keys())}")
        else:
            print(f"[apex_vote] no prior feedback found at {feedback_store.root}; first run")

    for i, sc in enumerate(scens):
        target = parse_prediction(sc.predator_target or "").direction
        # Default packet (no agent feedback) — used for the inter-tier
        # capture and as a fallback if a voter has no feedback yet.
        default_pkt = build_evidence_packet(sc, herb_broadcasts=None)
        responses = []
        for v in voters:
            # Per-agent packet: each voter gets its own decomposer feedback.
            voter_fb = feedback_by_agent.get(v.voter_id) if feedback_by_agent else None
            v_pkt = (
                build_evidence_packet(sc, herb_broadcasts=None, agent_feedback=voter_fb)
                if voter_fb is not None
                else default_pkt
            )
            try:
                r = v.vote(v_pkt)
            except Exception as e:
                print(f"  [voter {v.voter_id}] error: {e}")
                continue
            responses.append(r)
        # `pkt` for the rest of the loop = the default no-feedback packet
        # (used by capture_evidence_signals and Observation snapshot).
        pkt = default_pkt
        plur = plurality(responses)
        cw = confidence_weighted(responses)
        pw = perplexity_weighted(responses)

        plur_decs.append((target, plur.direction))
        conf_decs.append((target, cw.direction))
        ppl_decs.append((target, pw.direction))

        # Decomposer: per-voter fitness via leave-one-out drop test.
        if tracker is not None:
            for j, r in enumerate(responses):
                # Panel without voter j
                without_j = [responses[k] for k in range(len(responses)) if k != j]
                full_dir = plur.direction
                drop_dir = plurality(without_j).direction if without_j else None
                marginal_flip = (full_dir != drop_dir)
                marginal_correct = marginal_flip and (full_dir == target)
                voter_correct = (r.direction == target) if r.direction else None
                tracker.record(
                    voter_id=r.voter_id,
                    correct=voter_correct,
                    decisive=r.direction is not None,
                    marginal_flip=marginal_flip,
                    marginal_correct=marginal_correct,
                )
            if kg is not None and target in ("up", "down"):
                signature = {
                    "n_voters": len(responses),
                    "n_decisive": sum(1 for r in responses if r.direction),
                    "agreement": (
                        "all_agree" if len(set(r.direction for r in responses if r.direction)) <= 1
                        else "split"
                    ),
                }
                kg.write_observation(
                    scenario_name=sc.name,
                    ticker=sc.name.split("_")[2] if "_" in sc.name else "?",
                    target_direction=target,
                    ensemble_decision=plur,
                    signature=signature,
                )
            # Full Observation: every inter-tier signal + voter reasoning +
            # ensemble + per-voter decomposer judgments. Decomposer's
            # input for KG queries and evolution decisions.
            if obs_writer is not None:
                inter_tier = capture_evidence_signals(sc, pkt)
                judgments = []
                for j, r in enumerate(responses):
                    without_j = [responses[k] for k in range(len(responses)) if k != j]
                    drop_dir = plurality(without_j).direction if without_j else None
                    mflip = (plur.direction != drop_dir)
                    judgments.append(DecomposerJudgment(
                        voter_id=r.voter_id,
                        correct=(r.direction == target) if r.direction else None,
                        decisive=r.direction is not None,
                        marginal_flip=mflip,
                        marginal_correct=mflip and (plur.direction == target),
                    ))
                obs = Observation.from_voter_responses(
                    scenario_name=sc.name,
                    ticker=sc.name.split("_")[2] if "_" in sc.name else "?",
                    target_direction=target,
                    voters=responses,
                    ensemble=plur,
                    inter_tier=inter_tier,
                    signature={
                        "n_voters": len(responses),
                        "n_decisive": sum(1 for r in responses if r.direction),
                        "agreement": (
                            "all_agree" if len(set(r.direction for r in responses if r.direction)) <= 1
                            else "split"
                        ),
                    },
                )
                obs.decomposer_judgments = judgments
                obs_writer.write(obs)

        # Rolling stats so we can interrupt early and still see data
        rec = {
            "scenario": sc.name,
            "target": target,
            "voters": [
                {"id": r.voter_id, "direction": r.direction,
                 "confidence": r.confidence, "perplexity": r.perplexity,
                 "raw_text": r.raw_text[:200]}
                for r in responses
            ],
            "plurality": plur.direction,
            "confidence_weighted": cw.direction,
            "perplexity_weighted": pw.direction,
        }
        fout.write(json.dumps(rec) + "\n")
        fout.flush()

        if (i + 1) % 10 == 0 or (i + 1) == len(scens):
            ps = score_decisions(plur_decs)
            cs = score_decisions(conf_decs)
            ws = score_decisions(ppl_decs)
            print(
                f"  [{i+1:3d}/{len(scens)}] "
                f"plur MCC={ps['mcc']:+.3f} dec={ps['decided']}/{ps['n']}  "
                f"conf MCC={cs['mcc']:+.3f}  "
                f"ppl MCC={ws['mcc']:+.3f}"
            )

    fout.close()

    # Final report
    print()
    print("=== FINAL ===")
    for name, decs in [
        ("plurality", plur_decs),
        ("confidence_weighted", conf_decs),
        ("perplexity_weighted", ppl_decs),
    ]:
        s = score_decisions(decs)
        print(
            f"  {name:30s} n={s['n']} dec={s['decided']}/{s['n']} "
            f"ab={s['abstained']} tp={s['tp']} tn={s['tn']} fp={s['fp']} fn={s['fn']} "
            f"acc={s['acc']:.2%} MCC={s['mcc']:+.4f}"
        )
    print(f"\n[apex_vote] per-scenario records → {out_path}")

    if tracker is not None:
        print()
        print("=== PER-VOTER FITNESS ===")
        for vid, af in tracker.summary().items():
            print(
                f"  {vid:30s} seen={af.n_seen} dec={af.n_decisive} "
                f"acc={af.solo_accuracy:.2f} "
                f"marginal_flip={af.n_marginal_flip} "
                f"marginal={af.marginal_contribution:.2f} "
                f"freerider={af.freerider_score:+.2f}"
            )
        mgr = PopulationManager()
        decisions = mgr.decide(tracker)
        print()
        print(mgr.render_report(decisions))
    if kg is not None:
        kg.close()
        print(f"[apex_vote] KG records → {kg.path}")
    if obs_writer is not None:
        obs_writer.close()
        print(f"[apex_vote] full observations → {obs_writer.path}")

    # Derive per-agent feedback from this run's + prior observations and
    # write to the feedback store. Next session's voters will see their
    # own histories. Hexis-style: per-agent modulation, not shared.
    if feedback_store is not None and obs_writer is not None:
        from trophic.decomposers import ObservationWriter as _OW
        all_obs = _OW.read_all(Path(args.obs_path))
        if all_obs:
            min_seen = int(os.environ.get("TROPHIC_FEEDBACK_MIN_SEEN", "5"))
            deriver = FeedbackDeriver(window=200, min_seen=min_seen)
            new_feedbacks = deriver.derive_all(all_obs)
            if new_feedbacks:
                feedback_store.write_all(new_feedbacks)
                print(f"\n[apex_vote] derived per-agent feedback for {len(new_feedbacks)} voters → {feedback_store.root}")
                for aid, fb in new_feedbacks.items():
                    n = fb.derived_from.get("n_observations", 0)
                    acc = fb.derived_from.get("actual_acc", 0)
                    print(f"  {aid}: n={n} acc={acc:.2f}")
                    if fb.prompt_modulation:
                        for ln in fb.prompt_modulation.splitlines()[:6]:
                            print(f"    {ln}")
            else:
                print(f"\n[apex_vote] not enough data to derive feedback (need >= {deriver.min_seen} per voter)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
