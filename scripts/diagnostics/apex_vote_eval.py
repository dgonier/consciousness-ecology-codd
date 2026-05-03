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
)
from trophic.decomposers import (
    FitnessTracker, KGWriter, PopulationManager,
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
    args = ap.parse_args()

    print(f"[apex_vote] tickers={args.tickers}, n_per_ticker={args.n_per_ticker}")

    # Stand up all voters; only available ones will actually be called.
    candidates = [
        OpenAIVoter(),
        AnthropicVoter(),
        GeminiVoter(),
        OpenRouterVoter(),
        LocalQwenVoter(),
    ]
    voters = [v for v in candidates if v.is_available()]
    print(f"[apex_vote] available voters: {[v.voter_id for v in voters]}")
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

    # Decomposer: KG writer + fitness tracker + population manager.
    kg = None
    tracker = None
    if args.decomposer:
        kg = KGWriter(path=Path(args.kg_path))
        kg.open()
        tracker = FitnessTracker(window=200)
        print(f"[apex_vote] decomposer ON: KG={kg.path}, fitness window=200")

    for i, sc in enumerate(scens):
        target = parse_prediction(sc.predator_target or "").direction
        pkt = build_evidence_packet(sc, herb_broadcasts=None)
        responses = []
        for v in voters:
            try:
                r = v.vote(pkt)
            except Exception as e:
                print(f"  [voter {v.voter_id}] error: {e}")
                continue
            responses.append(r)
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
                kg.write_observation(
                    scenario_name=sc.name,
                    ticker=sc.name.split("_")[2] if "_" in sc.name else "?",
                    target_direction=target,
                    ensemble_decision=plur,  # use plurality as the recorded decision
                    signature={
                        "n_voters": len(responses),
                        "n_decisive": sum(1 for r in responses if r.direction),
                        "agreement": (
                            "all_agree" if len(set(r.direction for r in responses if r.direction)) <= 1
                            else "split"
                        ),
                    },
                )

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
