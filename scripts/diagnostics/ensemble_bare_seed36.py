"""Ensemble of bare Qwen + seed36 binary head — offline scoring.

Reads each system's per-scenario predictions from disk (no GPU needed)
and computes three ensemble strategies:

  1. AGREE: only commit when both agree, else abstain.
     Tighter precision, lower recall.
  2. MAJORITY: pick whichever direction the two predict more often.
     Since K=2, this is just AGREE with a tiebreak; falls back to bare's
     prediction on disagreement (bare's MCC is higher than seed36's
     individually, so it's a reasonable arbiter).
  3. BARE_OR: take bare's prediction, fall back to seed36 only when bare
     abstains. (StockNet bare always decides; this collapses to bare.)

For each strategy, prints n, decided, TP/TN/FP/FN, accuracy, MCC, and
class output distribution.

Usage:
    .venv/bin/python -u scripts/diagnostics/ensemble_bare_seed36.py \\
        --bare-log logs/baseline_promptonly_stocknet_v4.log \\
        --seed-log logs/loop2_seed36_full_v2.log
"""
from __future__ import annotations
import argparse, math, re, sys
from pathlib import Path


def parse_bare(path: Path) -> dict[str, tuple[str, str]]:
    """Returns {scenario_name: (target_dir, pred_dir)}."""
    out: dict[str, tuple[str, str]] = {}
    pat = re.compile(r"\[(stocknet_test_\S+)\]\s+target=(\w+)\s+pred=(\w+)")
    for ln in path.read_text().splitlines():
        m = pat.search(ln)
        if not m:
            continue
        sc, tgt, pred = m.group(1), m.group(2), m.group(3)
        if pred == "None":
            pred = None  # type: ignore
        out[sc] = (tgt, pred)
    return out


def parse_seed(path: Path) -> dict[str, tuple[str, str]]:
    """loop2_hooks_smoke logs: '[scenario  ] tgt=X pred=Y finite=Y | ...'"""
    out: dict[str, tuple[str, str]] = {}
    pat = re.compile(r"\[(\S+)\s*\]\s+tgt=(\w+)\s+pred=(\S+)")
    for ln in path.read_text().splitlines():
        m = pat.search(ln)
        if not m:
            continue
        sc, tgt, pred = m.group(1), m.group(2), m.group(3)
        if pred == "None":
            pred = None  # type: ignore
        out[sc] = (tgt, pred)
    return out


def score(decisions: list[tuple[str, str | None, str]]) -> dict:
    """Take list of (target, prediction, label) where label is for printing.

    Returns metrics dict.
    """
    n = tp = tn = fp = fn = ab = 0
    ups = downs = 0
    for tgt, pred, _ in decisions:
        n += 1
        if pred is None:
            ab += 1
            continue
        if pred == "up":
            ups += 1
        elif pred == "down":
            downs += 1
        if pred == "up" and tgt == "up":
            tp += 1
        elif pred == "down" and tgt == "down":
            tn += 1
        elif pred == "up" and tgt == "down":
            fp += 1
        elif pred == "down" and tgt == "up":
            fn += 1
    dec = tp + tn + fp + fn
    if dec == 0:
        mcc = 0.0
        acc = 0.0
    else:
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / max(denom, 1e-9)
        acc = (tp + tn) / dec
    return {
        "n": n, "decided": dec, "abstained": ab,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "ups": ups, "downs": downs, "acc": acc, "mcc": mcc,
    }


def fmt(name: str, m: dict) -> str:
    return (
        f"{name:30s} n={m['n']} dec={m['decided']}/{m['n']} ab={m['abstained']} "
        f"tp={m['tp']} tn={m['tn']} fp={m['fp']} fn={m['fn']} "
        f"acc={m['acc']:.2%} MCC={m['mcc']:+.4f} "
        f"classes={{up:{m['ups']},down:{m['downs']}}}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bare-log", required=True, type=Path)
    ap.add_argument("--seed-log", required=True, type=Path)
    args = ap.parse_args()

    bare = parse_bare(args.bare_log)
    seed = parse_seed(args.seed_log)

    common = sorted(set(bare) & set(seed))
    print(f"bare scenarios: {len(bare)}  seed scenarios: {len(seed)}  common: {len(common)}")
    print()

    # ---- Each system standalone ----
    bare_decs = [(bare[sc][0], bare[sc][1], sc) for sc in common]
    seed_decs = [(seed[sc][0], seed[sc][1], sc) for sc in common]
    print(fmt("bare alone", score(bare_decs)))
    print(fmt("seed36 alone", score(seed_decs)))
    print()

    # ---- AGREE: only commit when both agree ----
    agree = []
    for sc in common:
        tgt, pred_b = bare[sc]
        _, pred_s = seed[sc]
        if pred_b == pred_s and pred_b is not None:
            agree.append((tgt, pred_b, sc))
        else:
            agree.append((tgt, None, sc))
    print(fmt("agree (both must match)", score(agree)))

    # ---- MAJORITY: prefer match; on disagreement fall back to bare ----
    majority = []
    for sc in common:
        tgt, pred_b = bare[sc]
        _, pred_s = seed[sc]
        if pred_b == pred_s:
            pred = pred_b
        else:
            pred = pred_b  # bare arbiter (higher individual MCC)
        majority.append((tgt, pred, sc))
    print(fmt("majority (bare arbiter)", score(majority)))

    # ---- MAJORITY_SEED: on disagreement fall back to seed36 ----
    majority_seed = []
    for sc in common:
        tgt, pred_b = bare[sc]
        _, pred_s = seed[sc]
        pred = pred_b if pred_b == pred_s else pred_s
        majority_seed.append((tgt, pred, sc))
    print(fmt("majority (seed36 arbiter)", score(majority_seed)))

    # ---- AGREE shows precision under high-confidence regime ----
    print()
    a = score(agree)
    if a["decided"] > 0:
        print(f"agreement precision: {a['acc']:.2%} on {a['decided']}/{a['n']} "
              f"({a['decided']/a['n']:.0%}) decided")


if __name__ == "__main__":
    main()
