"""Sanity-check the MCC eval pipeline with synthetic predictors.

Both models reviewing the trophic stack (GPT-5.2 + Gemini 3.1 Pro) flagged
the bit-identical TP=36 TN=0 FP=14 FN=0 across 5 different checkpoints as
suspicious — could be evaluator bug, not just model collapse. This runs the
StockNet labels through the SAME matthews_corrcoef() function used in
scripts/eval_stocknet.py with three synthetic predictors:

  1. Constant-up (matches all 5 checkpoints) → expect ACC=0.72 MCC=0.0
  2. Random 50/50 → expect ACC≈0.5 MCC≈0
  3. Label-flipped (always inverts the truth) → expect ACC≈0.28 MCC=-1.0

If predictor 3 doesn't give MCC=-1.0, the evaluator is broken.
"""
from __future__ import annotations
import asyncio, builtins, functools, os, random, sys
from pathlib import Path
builtins.print = functools.partial(builtins.print, flush=True)
print = builtins.print

ROOT = Path("/home/dgonier/ecology_experiment/trophic")
sys.path.insert(0, str(ROOT))

from scripts.eval_stocknet import matthews_corrcoef
from trophic.training.stocknet_loader import build_stocknet_scenarios
from trophic.training.xml_schema import parse_prediction


def evaluate(predictions: list[str], targets: list[str]) -> dict:
    tp = tn = fp = fn = 0
    for p, t in zip(predictions, targets):
        if t == "up" and p == "up":
            tp += 1
        elif t == "down" and p == "down":
            tn += 1
        elif t == "down" and p == "up":
            fp += 1
        elif t == "up" and p == "down":
            fn += 1
    n = len(predictions)
    acc = (tp + tn) / n if n else 0.0
    mcc = matthews_corrcoef(tp, tn, fp, fn)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, acc=acc, mcc=mcc, n=n)


async def main() -> None:
    print("[mcc-sanity] loading scenarios...")
    scenarios = build_stocknet_scenarios(
        split="test",
        tickers=["AAPL", "GOOG", "MSFT", "AMZN", "JPM"],
        max_per_ticker=10,
    )
    targets = []
    for sc in scenarios:
        t = parse_prediction(sc.predator_target or "")
        targets.append(t.direction)
    print(f"[mcc-sanity] n={len(targets)}; up={targets.count('up')} down={targets.count('down')}")
    print()

    # Predictor 1: constant "up"
    pred_up = ["up"] * len(targets)
    r1 = evaluate(pred_up, targets)
    print(f"predictor=constant-up   TP={r1['tp']} TN={r1['tn']} FP={r1['fp']} FN={r1['fn']}  ACC={r1['acc']:.3f} MCC={r1['mcc']:+.3f}")

    # Predictor 2: constant "down"
    pred_down = ["down"] * len(targets)
    r2 = evaluate(pred_down, targets)
    print(f"predictor=constant-down TP={r2['tp']} TN={r2['tn']} FP={r2['fp']} FN={r2['fn']}  ACC={r2['acc']:.3f} MCC={r2['mcc']:+.3f}")

    # Predictor 3: random 50/50
    rng = random.Random(0)
    pred_rand = [rng.choice(["up", "down"]) for _ in targets]
    r3 = evaluate(pred_rand, targets)
    print(f"predictor=random50/50   TP={r3['tp']} TN={r3['tn']} FP={r3['fp']} FN={r3['fn']}  ACC={r3['acc']:.3f} MCC={r3['mcc']:+.3f}")

    # Predictor 4: label-flipped (always wrong)
    pred_flipped = ["down" if t == "up" else "up" for t in targets]
    r4 = evaluate(pred_flipped, targets)
    print(f"predictor=label-flipped TP={r4['tp']} TN={r4['tn']} FP={r4['fp']} FN={r4['fn']}  ACC={r4['acc']:.3f} MCC={r4['mcc']:+.3f}")

    # Predictor 5: oracle (always right)
    pred_oracle = list(targets)
    r5 = evaluate(pred_oracle, targets)
    print(f"predictor=oracle        TP={r5['tp']} TN={r5['tn']} FP={r5['fp']} FN={r5['fn']}  ACC={r5['acc']:.3f} MCC={r5['mcc']:+.3f}")

    # Predictor 6: weakly-informative (75% chance of right answer)
    rng2 = random.Random(0)
    pred_weak = [t if rng2.random() < 0.75 else ("down" if t == "up" else "up") for t in targets]
    r6 = evaluate(pred_weak, targets)
    print(f"predictor=75%-correct   TP={r6['tp']} TN={r6['tn']} FP={r6['fp']} FN={r6['fn']}  ACC={r6['acc']:.3f} MCC={r6['mcc']:+.3f}")

    print()
    print("=== INTERPRETATION ===")
    print("constant-up should match our 5 checkpoint runs (TP=36 TN=0 FP=14 FN=0)")
    print("label-flipped should give MCC=-1.0 if evaluator is correct")
    print("oracle should give MCC=+1.0 if evaluator is correct")
    print("75%-correct should give MCC > 0.4")


if __name__ == "__main__":
    asyncio.run(main())
