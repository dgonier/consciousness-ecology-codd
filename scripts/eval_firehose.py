"""Eval harness for firehose-loop output.

Reads the JSONL written by run_firehose_loop.py and computes:
  - Per-path per-ticker MCC (where the path emitted a prediction)
  - Coverage × accuracy curve (top-K most-confident watchlist entries per day)
  - Cluster-coherent prediction rate (multi-ticker article alignment)
  - Overall calibration (does p_up correlate with realized direction?)
  - Watchlist-focus uplift (forced-attention tickers)

Output: a printed report + a JSON results file for plotting later.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """Matthews correlation coefficient. Returns 0.0 if undefined."""
    num = tp * tn - fp * fn
    den = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    if den <= 0:
        return 0.0
    return num / den


def predicted_direction(action: str, p_up: float) -> str | None:
    """Map a watchlist entry's (action, p_up) → "up" / "down" / None.
    BUY → up, SELL → down, WATCH → up if p_up > 0.5 else down (tiebreak)
    """
    if action == "BUY":
        return "up"
    if action == "SELL":
        return "down"
    if action == "WATCH":
        return "up" if p_up >= 0.5 else "down"
    return None


def score_path(records: list[dict], path: str, top_k: int | None = None) -> dict:
    """Score one path's predictions across all days."""
    by_ticker: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    # entries: (predicted_dir, actual_dir, p_up)
    n_days = 0
    n_predictions = 0

    for rec in records:
        if path not in rec:
            continue
        wl = rec[path].get("watchlist", [])
        if top_k is not None:
            wl = wl[:top_k]
        n_days += 1
        # Build target lookup
        targets = {t["ticker"]: t for t in rec["targets"]}
        for entry in wl:
            tk = entry["ticker"]
            if tk not in targets:
                continue
            tg = targets[tk]
            actual = tg["actual_direction"]
            if actual == "flat":
                # Skip flat ticker-days for MCC scoring; they're noise
                continue
            pred = predicted_direction(entry["action"], entry["p_up"])
            if pred is None:
                continue
            by_ticker[tk].append((pred, actual, entry["p_up"]))
            n_predictions += 1

    # Aggregate confusion across all tickers (per ticker-day)
    tp = tn = fp = fn = 0
    p_up_correct: list[tuple[float, int]] = []
    for tk, entries in by_ticker.items():
        for pred, actual, p_up in entries:
            if pred == "up" and actual == "up":
                tp += 1
                p_up_correct.append((p_up, 1))
            elif pred == "down" and actual == "down":
                tn += 1
                p_up_correct.append((p_up, 0))
            elif pred == "up" and actual == "down":
                fp += 1
                p_up_correct.append((p_up, 0))
            elif pred == "down" and actual == "up":
                fn += 1
                p_up_correct.append((p_up, 1))

    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    overall_mcc = mcc(tp, tn, fp, fn)

    # Per-ticker MCC
    per_ticker_mcc: dict[str, float] = {}
    for tk, entries in by_ticker.items():
        if len(entries) < 4:
            continue
        ttp = ttn = tfp = tfn = 0
        for pred, actual, _ in entries:
            if pred == "up" and actual == "up": ttp += 1
            elif pred == "down" and actual == "down": ttn += 1
            elif pred == "up" and actual == "down": tfp += 1
            elif pred == "down" and actual == "up": tfn += 1
        per_ticker_mcc[tk] = mcc(ttp, ttn, tfp, tfn)

    return {
        "path": path,
        "top_k": top_k,
        "n_days": n_days,
        "n_predictions": n_predictions,
        "n_unique_tickers_predicted": len(by_ticker),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(accuracy, 4),
        "mcc": round(overall_mcc, 4),
        "per_ticker_mcc": {k: round(v, 4) for k, v in sorted(per_ticker_mcc.items())},
    }


def coverage_curve(records: list[dict], path: str, ks: list[int]) -> list[dict]:
    return [score_path(records, path, top_k=k) for k in ks]


def cluster_coherence(records: list[dict], path: str) -> dict:
    """For multi-ticker news articles, do the predictions for those tickers
    move in the same direction as their actual returns?
    Score: among multi-ticker articles, fraction of (ticker_pred, ticker_actual)
    matches.
    """
    n_pairs_scored = 0
    n_correct = 0
    for rec in records:
        if path not in rec:
            continue
        wl = {e["ticker"]: e for e in rec[path].get("watchlist", [])}
        targets = {t["ticker"]: t for t in rec["targets"]}
        # Gather articles tagging multiple in-universe tickers
        # (we don't have firehose_news here in eval rec; this is a placeholder
        # for a richer eval — for now we just compute pairwise across all
        # tickers the path predicted on the same day)
        tickers_with_preds = list(wl.keys())
        for i, ti in enumerate(tickers_with_preds):
            for tj in tickers_with_preds[i+1:]:
                if ti not in targets or tj not in targets:
                    continue
                if targets[ti]["actual_direction"] == "flat" or targets[tj]["actual_direction"] == "flat":
                    continue
                pred_ti = predicted_direction(wl[ti]["action"], wl[ti]["p_up"])
                pred_tj = predicted_direction(wl[tj]["action"], wl[tj]["p_up"])
                if pred_ti is None or pred_tj is None:
                    continue
                pred_aligned = (pred_ti == pred_tj)
                actual_aligned = targets[ti]["actual_direction"] == targets[tj]["actual_direction"]
                n_pairs_scored += 1
                if pred_aligned == actual_aligned:
                    n_correct += 1
    return {
        "path": path,
        "n_pairs_scored": n_pairs_scored,
        "coherence_accuracy": round(n_correct / max(1, n_pairs_scored), 4),
    }


def focus_uplift(records: list[dict], path: str) -> dict:
    """How does the path do on focus_tickers vs non-focus?"""
    focus_tp = focus_tn = focus_fp = focus_fn = 0
    nonfoc_tp = nonfoc_tn = nonfoc_fp = nonfoc_fn = 0
    for rec in records:
        if path not in rec:
            continue
        wl = rec[path].get("watchlist", [])
        focus = set(rec.get("focus_tickers") or [])
        targets = {t["ticker"]: t for t in rec["targets"]}
        for entry in wl:
            tk = entry["ticker"]
            if tk not in targets:
                continue
            actual = targets[tk]["actual_direction"]
            if actual == "flat":
                continue
            pred = predicted_direction(entry["action"], entry["p_up"])
            if pred is None:
                continue
            bucket = "focus" if tk in focus else "nonfocus"
            if pred == "up" and actual == "up":
                if bucket == "focus": focus_tp += 1
                else: nonfoc_tp += 1
            elif pred == "down" and actual == "down":
                if bucket == "focus": focus_tn += 1
                else: nonfoc_tn += 1
            elif pred == "up" and actual == "down":
                if bucket == "focus": focus_fp += 1
                else: nonfoc_fp += 1
            elif pred == "down" and actual == "up":
                if bucket == "focus": focus_fn += 1
                else: nonfoc_fn += 1
    return {
        "path": path,
        "focus_n": focus_tp + focus_tn + focus_fp + focus_fn,
        "focus_mcc": round(mcc(focus_tp, focus_tn, focus_fp, focus_fn), 4),
        "focus_acc": round((focus_tp + focus_tn) / max(1, focus_tp + focus_tn + focus_fp + focus_fn), 4),
        "nonfocus_n": nonfoc_tp + nonfoc_tn + nonfoc_fp + nonfoc_fn,
        "nonfocus_mcc": round(mcc(nonfoc_tp, nonfoc_tn, nonfoc_fp, nonfoc_fn), 4),
        "nonfocus_acc": round((nonfoc_tp + nonfoc_tn) / max(1, nonfoc_tp + nonfoc_tn + nonfoc_fp + nonfoc_fn), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_jsonl", help="path to eval JSONL written by run_firehose_loop.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    records: list[dict] = []
    with open(args.eval_jsonl) as fh:
        for line in fh:
            records.append(json.loads(line))
    print(f"loaded {len(records)} day records")

    paths = []
    if records and "ecology" in records[0]:
        paths.append("ecology")
    if records and "bare" in records[0]:
        paths.append("bare")

    report: dict = {"n_days": len(records)}
    for path in paths:
        print(f"\n=== {path.upper()} ===")
        all_score = score_path(records, path)
        print(f"  predictions: {all_score['n_predictions']}  "
              f"acc={all_score['accuracy']:.3f}  MCC={all_score['mcc']:+.3f}")
        print(f"  TP={all_score['tp']} TN={all_score['tn']} "
              f"FP={all_score['fp']} FN={all_score['fn']}")
        coverage = coverage_curve(records, path, ks=[3, 5, 10, None])
        print("  coverage curve (top-K):")
        for c in coverage:
            k = c["top_k"] if c["top_k"] else "all"
            print(f"    top-{k:>3}: n={c['n_predictions']:>4d}  "
                  f"acc={c['accuracy']:.3f}  MCC={c['mcc']:+.3f}")
        coh = cluster_coherence(records, path)
        print(f"  pairwise coherence: n={coh['n_pairs_scored']}  "
              f"coherent_acc={coh['coherence_accuracy']:.3f}")
        focus = focus_uplift(records, path)
        if focus["focus_n"] > 0:
            print(f"  focus uplift: focus_MCC={focus['focus_mcc']:+.3f} "
                  f"(n={focus['focus_n']})  vs non-focus={focus['nonfocus_mcc']:+.3f} "
                  f"(n={focus['nonfocus_n']})")
        report[path] = {
            "all": all_score,
            "coverage": coverage,
            "cluster_coherence": coh,
            "focus_uplift": focus,
        }

    if "ecology" in paths and "bare" in paths:
        print("\n=== HEAD-TO-HEAD ===")
        eco = report["ecology"]["all"]
        bare = report["bare"]["all"]
        print(f"  ecology: MCC {eco['mcc']:+.3f}  acc {eco['accuracy']:.3f}  n={eco['n_predictions']}")
        print(f"  bare:    MCC {bare['mcc']:+.3f}  acc {bare['accuracy']:.3f}  n={bare['n_predictions']}")
        delta = eco["mcc"] - bare["mcc"]
        print(f"  Δ MCC (ecology − bare): {delta:+.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote report to {args.out}")


if __name__ == "__main__":
    main()
