"""Compare two scenario JSONLs side-by-side at each tier.

Quick CLI for "is the herb output different across opposite-direction inputs?"
without firing up the viz.

Usage:
    .venv/bin/python -u scripts/diagnostics/compare_signals.py \\
        logs/signals/stocknet_test_AAPL_2015-10-01.jsonl \\
        logs/signals/stocknet_test_AAPL_2015-10-02.jsonl
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def lens(n: dict, k: int = 4) -> str:
    L = n.get("logit_lens", [])
    if not L:
        return ""
    return "  ".join(f"{repr(x['token'])}({x['prob']*100:.0f}%)" for x in L[:k])


def main():
    a = Path(sys.argv[1])
    b = Path(sys.argv[2])
    A = {n["id"]: n for n in load(a)}
    B = {n["id"]: n for n in load(b)}
    a_target = A.get("scenario", {}).get("target_direction")
    b_target = B.get("scenario", {}).get("target_direction")
    print(f"\nA = {a.stem}  target={a_target}")
    print(f"B = {b.stem}  target={b_target}")
    print()

    ids = list(A.keys())  # assume same set
    for i in ids:
        if i not in B:
            continue
        nA = A[i]
        nB = B[i]
        tier = nA.get("tier")
        sa = lens(nA)
        sb = lens(nB)
        diff = "≠" if sa != sb else "="
        print(f"T{tier} {i:30s}  {diff}")
        if sa or sb:
            print(f"   A: {sa}")
            print(f"   B: {sb}")
        if i == "apex.output":
            print(f"   A.parsed: {nA.get('parsed_ticker')} / {nA.get('parsed_direction')}")
            print(f"   B.parsed: {nB.get('parsed_ticker')} / {nB.get('parsed_direction')}")


if __name__ == "__main__":
    main()
