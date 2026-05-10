"""Inspect one day of eval output: ecology observations + watchlists vs targets.

Usage:
  .venv/bin/python -u scripts/inspect_eval_day.py <eval.jsonl> [--date 2026-02-03]
"""
from __future__ import annotations

import argparse
import json
import sys


def fmt_obs(o: dict) -> str:
    return (f"[{o['action']}] {o['ticker']:5s} p_up={o['p_up']:.3f} "
            f"sal={o['salience']:.3f} conf={o['confidence']}  "
            f"{o['reasoning_summary'][:120]}")


def fmt_chain(chain: list[dict]) -> str:
    parts = []
    for c in chain:
        tag = c.get("parent", "?").replace("belief.company.", "").replace("belief.market.", "mkt.")
        tag = tag.split("__")[0]
        sign = "↑" if c.get("delta", 0) > 0 else "↓"
        ls = c.get("link_strength")
        contrib = c.get("contribution_log_odds", 0.0)
        if ls is not None:
            parts.append(f"{sign}{tag}({c.get('delta', 0):+.2f}×{ls:.2f}={contrib:+.2f})")
        else:
            parts.append(f"{sign}{tag}(Δ={c.get('delta', 0):+.2f}, contrib={contrib:+.2f})")
    return " | ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_jsonl")
    ap.add_argument("--date", default=None)
    args = ap.parse_args()

    with open(args.eval_jsonl) as fh:
        records = [json.loads(line) for line in fh]

    if args.date:
        records = [r for r in records if r["date"] == args.date]
    print(f"records: {len(records)}")
    for rec in records:
        print(f"\n{'='*78}\nDATE: {rec['date']}  (prev: {rec['prev_trading_day']})")
        targets_by_tk = {t["ticker"]: t for t in rec["targets"]}

        print("\nTARGETS:")
        for tk in sorted(targets_by_tk.keys()):
            t = targets_by_tk[tk]
            mark = "↑" if t["actual_direction"] == "up" else "↓" if t["actual_direction"] == "down" else "·"
            print(f"  {mark} {tk:5s}  ret={t['actual_return']:+.4f}  {t['actual_direction']:5s}  "
                  f"news={t['was_mentioned_in_news']}")

        if "ecology" in rec:
            eco = rec["ecology"]
            print(f"\nECOLOGY  ({eco.get('elapsed_s', '?')}s)")
            print(f"  event_acts={eco['n_event_activations']}  "
                  f"cluster={eco['n_xcorr_cluster_activations']}  "
                  f"sympathy={eco['n_xcorr_sympathy_activations']}  "
                  f"observations={eco['n_observations']}")
            print("  observations to apex:")
            for o in eco.get("observations", []):
                print(f"    {fmt_obs(o)}")
                if o.get("causal_chain"):
                    print(f"      drivers: {fmt_chain(o['causal_chain'][:4])}")
                if o.get("conflicting_signals"):
                    print(f"      conflict: {fmt_chain(o['conflicting_signals'][:3])}")
            print("  watchlist:")
            for w in eco["watchlist"]:
                tk = w["ticker"]
                t = targets_by_tk.get(tk)
                actual_mark = "✓" if t and t["actual_direction"] == "up" and w["action"] == "BUY" else \
                              "✓" if t and t["actual_direction"] == "down" and w["action"] == "SELL" else \
                              "·" if t and t["actual_direction"] == "flat" else "✗" if t else "?"
                actual = f"actual={t['actual_direction']:5s}({t['actual_return']:+.3f})" if t else ""
                print(f"    {actual_mark} [{w['action']}] {tk:5s} p_up={w['p_up']:.2f}  {actual}  "
                      f"{w['reason'][:100]}")

        if "bare" in rec:
            bare = rec["bare"]
            print(f"\nBARE  ({bare.get('elapsed_s', '?')}s)")
            print(f"  watchlist:")
            for w in bare["watchlist"]:
                tk = w["ticker"]
                t = targets_by_tk.get(tk)
                actual_mark = "✓" if t and t["actual_direction"] == "up" and w["action"] == "BUY" else \
                              "✓" if t and t["actual_direction"] == "down" and w["action"] == "SELL" else \
                              "·" if t and t["actual_direction"] == "flat" else "✗" if t else "?"
                actual = f"actual={t['actual_direction']:5s}({t['actual_return']:+.3f})" if t else ""
                print(f"    {actual_mark} [{w['action']}] {tk:5s} p_up={w['p_up']:.2f}  {actual}  "
                      f"{w['reason'][:100]}")


if __name__ == "__main__":
    main()
