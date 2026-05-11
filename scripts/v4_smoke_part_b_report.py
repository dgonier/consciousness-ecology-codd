"""v4 phase-2.5 smoke gate — Part B report generator.

Reads the runner's per-day eval JSONL and writes tasks_v4/SMOKE_RESULTS.md
plus a final pass/fail verdict on stdout. Applies the mission's numeric
thresholds:

  - Hard fails: runner exception (non-zero rc), rejection rate > 50%.
  - Soft fails: order count > 12 in 3 days, no h20/h60 orders.

Exit codes:
  0 — PART B PASS or SOFT-FAIL (phase 3 can launch)
  1 — PART B HARD FAIL (do NOT unblock phase 3)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_eval_jsonl(p: Path) -> list[dict]:
    rows: list[dict] = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  warning: malformed jsonl line: {e}")
    return rows


def extract_eco_orders(record: dict) -> tuple[list[dict], list[dict], list[str], dict]:
    """Return (raw_orders, validated_orders, rejections, snapshot_post) for
    the ecology path only. Defensive — empty lists if absent.
    """
    eco = record.get("ecology") or {}
    raw = eco.get("pm_orders_raw") or []
    validated = eco.get("pm_orders_validated") or []
    rejected = eco.get("pm_orders_rejected") or []
    snap = eco.get("pm_snapshot_post") or {}
    return raw, validated, rejected, snap


def horizon_counts(orders: list[dict]) -> Counter:
    return Counter((o.get("primary_horizon") or "unknown") for o in orders)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--eval-jsonl", required=True, type=Path)
    ap.add_argument("--runner-rc", type=int, required=True)
    ap.add_argument("--elapsed-s", type=int, required=True)
    ap.add_argument("--report-out", required=True, type=Path)
    args = ap.parse_args()

    rows = load_eval_jsonl(args.eval_jsonl)
    n_days = len(rows)

    # Aggregate across days
    total_raw = 0
    total_validated = 0
    total_rejected = 0
    all_horizons: Counter = Counter()
    sample_orders: list[tuple[str, dict]] = []
    per_day: list[dict] = []
    last_equity = None
    first_equity = None
    starting_equity = 100_000.0

    for rec in rows:
        date = rec.get("date", "?")
        raw, validated, rejected, snap = extract_eco_orders(rec)
        total_raw += len(raw)
        total_validated += len(validated)
        total_rejected += len(rejected)
        all_horizons.update(horizon_counts(validated))
        if first_equity is None and snap.get("equity") is not None:
            first_equity = float(snap["equity"])
        if snap.get("equity") is not None:
            last_equity = float(snap["equity"])
        for o in validated:
            sample_orders.append((date, o))
        per_day.append({
            "date": date,
            "raw": len(raw),
            "validated": len(validated),
            "rejected": len(rejected),
            "equity": snap.get("equity"),
            "cash": snap.get("cash"),
            "invested_pct": snap.get("invested_pct"),
        })

    # Compute rejection rate as rejections / (validated + rejections). If
    # both are zero, treat as 0% rejection.
    attempted = total_validated + total_rejected
    rej_rate = (total_rejected / attempted) if attempted > 0 else 0.0

    # Did any h20/h60 orders fire?
    n_long = all_horizons.get("h20", 0) + all_horizons.get("h60", 0)

    # Verdict gating
    hard_fail = False
    soft_fail = False
    notes: list[str] = []

    if args.runner_rc != 0:
        hard_fail = True
        notes.append(
            f"HARD FAIL: runner exited rc={args.runner_rc} "
            f"(possible exception/timeout)."
        )
    if rej_rate > 0.50:
        hard_fail = True
        notes.append(
            f"HARD FAIL: rejection_rate={rej_rate:.1%} > 50%."
        )
    if total_validated > 12:
        soft_fail = True
        notes.append(
            f"SOFT FAIL: total validated orders={total_validated} > 12 "
            f"(ceiling). Apex over-trading despite v4 prompt."
        )
    if n_long == 0 and total_validated > 0:
        soft_fail = True
        notes.append(
            f"SOFT FAIL: no h20/h60 orders in {n_days} days "
            f"(only short horizons fired). Multi-horizon machinery may not "
            f"be biting the prompt."
        )

    if hard_fail:
        verdict = "HARD-FAIL"
    elif soft_fail:
        verdict = "SOFT-FAIL-OK-TO-CONTINUE"
    else:
        verdict = "PASS"

    # Build report markdown
    lines: list[str] = []
    lines.append(f"# v4 Phase-2.5 Smoke Gate — RESULTS")
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    lines.append(f"- runner rc: `{args.runner_rc}`")
    lines.append(f"- elapsed: `{args.elapsed_s}s`")
    lines.append(f"- eval jsonl: `{args.eval_jsonl}`")
    lines.append(f"- runner log: `{args.log}`")
    lines.append("")

    # PART A summary (re-run not necessary; the gate runs Part A separately,
    # and this report only covers Part B's contribution. The shell driver
    # gates on Part A's exit code so we know it passed if we get here.)
    lines.append("## PART A — contract verifier")
    lines.append("")
    lines.append(
        "Part A was run prior to Part B. See "
        "`scripts/v4_smoke_part_a.py` and its stdout for the full assertion "
        "list. Part A reports `SMOKE_PART_A: PASS` on exit 0 — that exit code "
        "is enforced by the smoke-gate driver before this report is generated."
    )
    lines.append("")
    lines.append("Assertions covered (18 total):")
    lines.append("- happy path h20 BUY (all 3 strategies BUY, aggregator BUY, all 4 validators accept)")
    lines.append("- happy path h20 SELL (4 validators accept; sizing skipped on SELL per phase2-D)")
    lines.append("- adversarial: h20 BUY size=50 → horizon_sizing rejects")
    lines.append("- adversarial: BUY committed to h5 with negative forecast.h5 → forecast_consistency rejects")
    lines.append("- adversarial: BUY h5 with alpha=50 below dynamic floor → edge_floor rejects")
    lines.append("- h60 edge-floor waiver: same alpha=50 with h60 → accepts (waived)")
    lines.append("- re-buy preservation: BUY h60, BUY h1 same ticker, advance 3d, SELL → min_hold rejects (committed to h60)")
    lines.append("- committee: 2BUY+1SELL → strict-majority BUY")
    lines.append("- committee: 1BUY+1SELL+1BUY equal-conviction → strict-majority BUY")
    lines.append("- committee: all votes <0.20 conviction → None (aggregator drops)")
    lines.append("- committee: all HOLD votes → None")
    lines.append("- kelly edge-floor end-to-end: kelly HOLDs on low alpha, 2/3 BUY → aggregator BUY")
    lines.append("- full run_tax_aware_chain on clean h20 BUY → accepts")
    lines.append("")

    # PART B
    lines.append("## PART B — 3-day ECO-QWEN mini-sweep")
    lines.append("")
    lines.append("### Headline numbers")
    lines.append("")
    lines.append(f"| metric | value |")
    lines.append(f"|---|---|")
    lines.append(f"| days run | {n_days} |")
    lines.append(f"| total raw orders (post-committee) | {total_raw} |")
    lines.append(f"| total validated orders | {total_validated} |")
    lines.append(f"| total rejections | {total_rejected} |")
    lines.append(f"| rejection rate (rej / (val+rej)) | {rej_rate:.1%} |")
    lines.append(f"| order ceiling (mission threshold) | 12 |")
    lines.append(f"| rejection ceiling (mission threshold) | 50% |")
    lines.append(f"| starting equity | ${starting_equity:,.0f} |")
    lines.append(
        f"| final equity (day-{n_days}) | "
        + (f"${last_equity:,.0f}" if last_equity is not None else "n/a")
        + " |"
    )
    if last_equity is not None:
        ret = (last_equity - starting_equity) / starting_equity
        lines.append(f"| day-{n_days} return | {ret:+.2%} |")
    lines.append("")

    lines.append("### Horizon distribution (validated orders)")
    lines.append("")
    lines.append("| horizon | count |")
    lines.append("|---|---|")
    for h in ("h1", "h5", "h20", "h60", "unknown"):
        c = all_horizons.get(h, 0)
        if c or h in ("h1", "h5", "h20", "h60"):
            lines.append(f"| {h} | {c} |")
    lines.append("")

    lines.append("### Per-day equity trajectory")
    lines.append("")
    lines.append("| date | raw | validated | rejected | equity | cash | inv% |")
    lines.append("|---|---|---|---|---|---|---|")
    for d in per_day:
        eq = d.get("equity")
        cash = d.get("cash")
        inv = d.get("invested_pct")
        lines.append(
            f"| {d['date']} | {d['raw']} | {d['validated']} | {d['rejected']} "
            f"| {(f'${eq:,.0f}' if eq is not None else 'n/a')} "
            f"| {(f'${cash:,.0f}' if cash is not None else 'n/a')} "
            f"| {(f'{inv:.0f}' if inv is not None else 'n/a')} |"
        )
    lines.append("")

    lines.append("### Sample validated orders (up to 5)")
    lines.append("")
    if sample_orders:
        lines.append("| date | side | ticker | size_pct | primary_horizon | expected_alpha_bps |")
        lines.append("|---|---|---|---|---|---|")
        for date, o in sample_orders[:5]:
            lines.append(
                f"| {date} "
                f"| {o.get('side', '?')} "
                f"| {o.get('ticker', '?')} "
                f"| {o.get('size_pct', '?')} "
                f"| {o.get('primary_horizon', '?')} "
                f"| {o.get('expected_alpha_bps', '?')} |"
            )
    else:
        lines.append("(no validated orders to sample)")
    lines.append("")

    lines.append("### Notes / flags")
    lines.append("")
    if notes:
        for n in notes:
            lines.append(f"- {n}")
    else:
        lines.append("- (no flags; clean pass)")
    lines.append("")
    lines.append("### Soft-fails for phase 3 to watch")
    lines.append("")
    soft_items = []
    if total_validated > 12:
        soft_items.append(
            f"`order_overshoot`: validated orders ({total_validated}) > 12 ceiling "
            f"in 3 days. Phase 3 should watch the 66-day sweep for "
            f"~4/day cap (264 ceiling). Likely cause: prompt directives "
            f"on ≤0.5 orders/day not biting for ECO-QWEN."
        )
    if n_long == 0 and total_validated > 0:
        soft_items.append(
            f"`short_horizon_bias`: no h20/h60 orders. v3.3-grade behavior; "
            f"committee horizon-majority tiebreak picks longer but apex is "
            f"only emitting h1/h5 to begin with. Watch phase 3 logs for the "
            f"same — if persistent, raise to @phase2-C."
        )
    if soft_items:
        for it in soft_items:
            lines.append(f"- {it}")
    else:
        lines.append("- (no soft-fail items)")
    lines.append("")

    args.report_out.write_text("\n".join(lines))
    print(f"Wrote {args.report_out}")
    print()
    print("=" * 72)
    print(f"SMOKE_PART_B: {verdict}")
    print("=" * 72)
    for n in notes:
        print("  " + n)
    print(
        f"orders: raw={total_raw} validated={total_validated} "
        f"rejected={total_rejected} rej_rate={rej_rate:.1%}"
    )
    print(f"horizons: {dict(all_horizons)}")
    if last_equity is not None:
        print(
            f"equity: start=${starting_equity:,.0f} "
            f"end=${last_equity:,.0f} "
            f"({((last_equity - starting_equity) / starting_equity):+.2%})"
        )

    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
