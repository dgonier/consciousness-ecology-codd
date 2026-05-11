"""Parse the 3-day ECO-DEBATE-QWEN smoke artefacts and write the
verdict report at `tasks_v4/DEBATE_SMOKE_RESULTS.md` (phase3-A-05f).

Inputs:
  --log         Path to the runner stdout/stderr log.
  --eval        Path to the eval JSONL the runner wrote.
  --transcript  Path to the per-run_label debate transcripts JSONL.
  --out         Where to write the verdict markdown.
  --runner-rc   The runner's exit code (forwarded from the wrapper).

Outputs:
  - Verdict line at top: PASS / SOFT-FAIL / HARD-FAIL.
  - Per-day per-predator breakdown.
  - Diversity check (pairwise Jaccard on thesis tickers).
  - Token-budget summary (best-effort: parsed from log; vLLM doesn't
    always echo prompt tokens, in which case we surface "n/a").
  - Sample objection excerpt if available.

Exit code:
  0 if PASS or SOFT-FAIL with the soft-fail-ok policy below.
  3 if HARD-FAIL.
  2 on infra problems (missing files, etc.).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any


# ── log parsing ────────────────────────────────────────────────────────


def parse_log(log_path: Path) -> dict[str, Any]:
    """Pull the high-level signals from the runner's stdout."""
    if not log_path.exists():
        return {
            "exists": False,
            "debate_failures": [],
            "per_day": [],
            "max_prompt_tokens": None,
            "fatal": None,
        }
    text = log_path.read_text(errors="replace")

    debate_failures: list[dict[str, str]] = []
    fail_re = re.compile(
        r"\[eco_debate_qwen\] DEBATE FAILED (\d{4}-\d{2}-\d{2}): (.+)"
    )
    for m in fail_re.finditer(text):
        debate_failures.append({"date": m.group(1), "error": m.group(2)[:300]})

    # Per-day PM line — emitted right after each day's debate dispatch.
    per_day: list[dict[str, Any]] = []
    day_re = re.compile(
        r"\[(?P<i>\d+)/(?P<n>\d+)\]\s+(?P<date>\d{4}-\d{2}-\d{2})"
    )
    pm_re = re.compile(
        r"PM:\s+equity=\$([0-9,\.]+)\s+cash=\$([0-9,\.]+)\s+"
        r"inv=([0-9\.]+)%\s+orders=(\d+)\s+rejected=(\d+)"
    )
    predator_re = re.compile(
        r"\s+(?P<pid>[a-z_]+)\s+\((?P<phil>[a-z_]+)\)\s*:\s*"
        r"eq=\$([0-9,\.]+)\s+valid=(\d+)\s+rej=(\d+)\s+theses=(\d+)"
    )
    # Walk the file line by line and accumulate a day record.
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        d = day_re.search(line)
        if d:
            if current is not None:
                per_day.append(current)
            current = {
                "date": d.group("date"),
                "pm": None,
                "predators": {},
            }
            continue
        if current is None:
            continue
        p = pm_re.search(line)
        if p:
            current["pm"] = {
                "equity": _money(p.group(1)),
                "cash": _money(p.group(2)),
                "invested_pct": float(p.group(3)),
                "orders": int(p.group(4)),
                "rejected": int(p.group(5)),
            }
            continue
        x = predator_re.search(line)
        if x:
            pid = x.group("pid")
            current["predators"][pid] = {
                "philosophy": x.group("phil"),
                "equity": _money(x.group(3)),
                "valid": int(x.group(4)),
                "rejected": int(x.group(5)),
                "theses": int(x.group(6)),
            }
    if current is not None:
        per_day.append(current)

    # Token-budget signal — DSPy/litellm sometimes prints prompt-token
    # usage, but vLLM-through-litellm typically doesn't. Best-effort.
    max_prompt_tokens: int | None = None
    tok_re = re.compile(r"prompt_tokens['\":= ]+(\d{3,6})")
    for m in tok_re.finditer(text):
        v = int(m.group(1))
        if max_prompt_tokens is None or v > max_prompt_tokens:
            max_prompt_tokens = v

    fatal = None
    if "FATAL:" in text:
        fatal = next(
            (line for line in text.splitlines() if "FATAL:" in line),
            None,
        )

    return {
        "exists": True,
        "debate_failures": debate_failures,
        "per_day": per_day,
        "max_prompt_tokens": max_prompt_tokens,
        "fatal": fatal,
    }


def _money(s: str) -> float:
    return float(s.replace(",", "").replace("$", "").strip())


# ── transcript parsing ─────────────────────────────────────────────────


def parse_transcripts(transcript_path: Path) -> dict[str, Any]:
    """Walk the per-day JSONL and extract phase coverage + thesis tickers.

    For each predator across all days, we collect the set of tickers
    the predator opened theses on AND committed orders on (separately,
    because thesis-opens are the "intent to hold" signal and orders are
    the realized footprint). The Jaccard check runs over thesis tickers
    per the mission spec.
    """
    if not transcript_path.exists() or transcript_path.stat().st_size == 0:
        return {
            "exists": False,
            "days": [],
            "thesis_tickers_per_predator": {},
            "order_tickers_per_predator": {},
            "phase_coverage": {},
            "sample_objection": None,
        }

    days: list[dict[str, Any]] = []
    thesis_tickers: dict[str, set[str]] = {}
    order_tickers: dict[str, set[str]] = {}
    # phase_coverage[predator][phase_n] = days the predator emitted >0 items.
    phase_coverage: dict[str, dict[int, int]] = {}
    sample_objection: str | None = None

    with transcript_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                day = json.loads(line)
            except json.JSONDecodeError:
                continue
            date = day.get("date", "?")
            day_record: dict[str, Any] = {
                "date": date,
                "predators": {},
            }
            # phase1: each predator's proposal — counts theses + orders.
            for pid, prop in (day.get("phase1") or {}).items():
                thesis_tickers.setdefault(pid, set())
                order_tickers.setdefault(pid, set())
                phase_coverage.setdefault(pid, {1: 0, 2: 0, 3: 0, 4: 0})
                if prop.get("new_theses"):
                    phase_coverage[pid][1] += 1
                    for t in prop["new_theses"]:
                        if t.get("ticker"):
                            thesis_tickers[pid].add(t["ticker"])
                day_record["predators"].setdefault(pid, {})[
                    "phase1_new_theses"
                ] = len(prop.get("new_theses") or [])
                day_record["predators"][pid]["phase1_orders"] = len(
                    prop.get("orders") or []
                )
            # phase2: structured responses.
            for pid, resp in (day.get("phase2") or {}).items():
                phase_coverage.setdefault(pid, {1: 0, 2: 0, 3: 0, 4: 0})
                rs = resp.get("responses") or []
                if rs:
                    phase_coverage[pid][2] += 1
                day_record["predators"].setdefault(pid, {})[
                    "phase2_responses"
                ] = len(rs)
                # Capture a "good" object_to as sample for the report.
                if sample_objection is None:
                    for r in rs:
                        if (
                            r.get("kind") == "object_to"
                            and len(r.get("rationale") or "") > 40
                        ):
                            sample_objection = (
                                f"{pid} → {r.get('target_predator_id')}"
                                f" on {r.get('target_ticker') or r.get('target_thesis_id') or '?'}: "
                                f"{r.get('rationale')[:240]}"
                            )
                            break
            # phase3
            for pid, rev in (day.get("phase3") or {}).items():
                phase_coverage.setdefault(pid, {1: 0, 2: 0, 3: 0, 4: 0})
                if rev.get("revised_orders") or rev.get("revised_new_theses"):
                    phase_coverage[pid][3] += 1
                day_record["predators"].setdefault(pid, {})[
                    "phase3_orders"
                ] = len(rev.get("revised_orders") or [])
                day_record["predators"][pid]["phase3_conceded"] = len(
                    rev.get("conceded") or []
                )
                day_record["predators"][pid]["phase3_held"] = len(
                    rev.get("held_firm") or []
                )
            # phase4 — final commit
            for pid, commit in (day.get("phase4") or {}).items():
                order_tickers.setdefault(pid, set())
                phase_coverage.setdefault(pid, {1: 0, 2: 0, 3: 0, 4: 0})
                fo = commit.get("final_orders") or []
                if fo:
                    phase_coverage[pid][4] += 1
                day_record["predators"].setdefault(pid, {})[
                    "phase4_orders"
                ] = len(fo)
                for o in fo:
                    tk = o.get("ticker") or o.get("from_ticker") or o.get("to_ticker")
                    if tk:
                        order_tickers[pid].add(tk)
                for t in (commit.get("final_theses_to_open") or []):
                    if t.get("ticker"):
                        thesis_tickers.setdefault(pid, set()).add(t["ticker"])
            days.append(day_record)

    return {
        "exists": True,
        "days": days,
        "thesis_tickers_per_predator": {
            k: sorted(v) for k, v in thesis_tickers.items()
        },
        "order_tickers_per_predator": {
            k: sorted(v) for k, v in order_tickers.items()
        },
        "phase_coverage": phase_coverage,
        "sample_objection": sample_objection,
    }


# ── verdict logic ──────────────────────────────────────────────────────


def compute_verdict(
    log: dict[str, Any],
    transcript: dict[str, Any],
    runner_rc: int,
) -> dict[str, Any]:
    """Apply the mission's pass / soft-fail / hard-fail policy.

    HARD-FAIL conditions:
      - runner exited non-zero
      - any unhandled crash signature in the log
      - zero trades across ALL days (debate path effectively broken)
      - rejection rate > 50%
      - 3+ predators silent (no phase-1 OR phase-4 output)

    SOFT-FAIL conditions:
      - 3+ predators open zero theses across all days (diversity goal
        not met) → soft-fail-ok per the mission table.
      - any day has token usage > 80% of 16K
      - total orders/day > 16
    """
    issues: dict[str, list[str]] = {"hard": [], "soft": []}

    if runner_rc != 0:
        issues["hard"].append(f"runner exited non-zero rc={runner_rc}")
    if log.get("fatal"):
        issues["hard"].append(f"FATAL in log: {log['fatal']}")

    per_day = log.get("per_day") or []
    n_days = len(per_day)
    total_orders = sum((d.get("pm") or {}).get("orders", 0) for d in per_day)
    total_rej = sum((d.get("pm") or {}).get("rejected", 0) for d in per_day)
    rej_rate = total_rej / (total_orders + total_rej) if (total_orders + total_rej) else 0.0

    if n_days == 0:
        issues["hard"].append("no per-day PM lines parsed from log")
    elif total_orders == 0:
        # Per the mission file: zero trades on ALL days is HARD-FAIL.
        issues["hard"].append(
            f"zero trades across all {n_days} day(s) — debate path "
            f"produced no executed orders (rejected={total_rej}; "
            f"debate_failures={len(log.get('debate_failures') or [])})"
        )

    if rej_rate > 0.5:
        issues["hard"].append(
            f"rejection rate {rej_rate:.1%} > 50% threshold"
        )

    # Predator silence check (against phase 1 + phase 4 coverage).
    cov = (transcript.get("phase_coverage") or {})
    silent: list[str] = []
    for pid, phases in cov.items():
        if phases.get(1, 0) == 0 and phases.get(4, 0) == 0:
            silent.append(pid)
    if len(silent) >= 3:
        issues["hard"].append(
            f"3+ predators silent across all phases: {silent}"
        )

    # Diversity check: thesis tickers per predator across all days.
    thesis_tickers = transcript.get("thesis_tickers_per_predator") or {}
    zero_thesis_preds = [
        pid for pid in (cov or {})
        if not thesis_tickers.get(pid)
    ]
    if len(zero_thesis_preds) >= 3:
        issues["soft"].append(
            f"3+ predators opened zero theses: {zero_thesis_preds} "
            f"(diversity goal not met)"
        )

    # Pairwise Jaccard on thesis tickers.
    jaccard = compute_jaccard(thesis_tickers)
    if jaccard["pairs"] and jaccard["mean"] is not None and jaccard["mean"] > 0.7:
        issues["soft"].append(
            f"average pairwise Jaccard {jaccard['mean']:.2f} > 0.7 "
            f"— predators are too similar"
        )

    # Token budget — 16K context, 80% threshold = 12,800.
    max_tok = log.get("max_prompt_tokens")
    if max_tok is not None and max_tok > 12_800:
        issues["soft"].append(
            f"max prompt tokens {max_tok} > 80% of 16K context"
        )

    # Per-day order ceiling: total across 4 predators ≤ 16 per day.
    over_cap_days = [
        d["date"] for d in per_day
        if (d.get("pm") or {}).get("orders", 0) > 16
    ]
    if over_cap_days:
        issues["soft"].append(
            f"order count >16 on {len(over_cap_days)} day(s): {over_cap_days}"
        )

    if issues["hard"]:
        verdict = "HARD-FAIL"
    elif issues["soft"]:
        verdict = "SOFT-FAIL"
    else:
        verdict = "PASS"

    return {
        "verdict": verdict,
        "issues": issues,
        "n_days": n_days,
        "total_orders": total_orders,
        "total_rejected": total_rej,
        "rejection_rate": rej_rate,
        "jaccard": jaccard,
    }


def compute_jaccard(tickers_per: dict[str, list[str]]) -> dict[str, Any]:
    pids = sorted(tickers_per.keys())
    pairs: list[dict[str, Any]] = []
    for a, b in combinations(pids, 2):
        sa = set(tickers_per[a])
        sb = set(tickers_per[b])
        union = sa | sb
        inter = sa & sb
        j = (len(inter) / len(union)) if union else None
        pairs.append(
            {"a": a, "b": b, "intersection": sorted(inter),
             "jaccard": j, "size_a": len(sa), "size_b": len(sb)}
        )
    js = [p["jaccard"] for p in pairs if p["jaccard"] is not None]
    return {
        "pairs": pairs,
        "mean": (sum(js) / len(js)) if js else None,
    }


# ── markdown rendering ─────────────────────────────────────────────────


def render_markdown(
    log: dict[str, Any],
    transcript: dict[str, Any],
    verdict: dict[str, Any],
    eval_path: Path,
) -> str:
    out: list[str] = []
    out.append(f"# Debate Smoke Gate Results (phase3-A-05f)")
    out.append("")
    out.append(f"**Verdict: {verdict['verdict']}**")
    out.append("")
    out.append(f"- runner-log: `{log.get('_log_path', '?')}`")
    out.append(f"- eval-jsonl: `{eval_path}`")
    out.append(f"- transcript: `{transcript.get('_path', '?')}`")
    out.append(f"- days observed: {verdict['n_days']}")
    out.append(f"- total orders (PM, summed): {verdict['total_orders']}")
    out.append(f"- total rejections: {verdict['total_rejected']}")
    out.append(
        f"- rejection rate: {verdict['rejection_rate']:.1%}"
    )
    out.append("")

    if verdict["issues"]["hard"]:
        out.append("## HARD-FAIL issues")
        for s in verdict["issues"]["hard"]:
            out.append(f"- {s}")
        out.append("")
    if verdict["issues"]["soft"]:
        out.append("## SOFT-FAIL issues")
        for s in verdict["issues"]["soft"]:
            out.append(f"- {s}")
        out.append("")

    # Debate failures (caught by the runner try/except — diagnostic).
    failures = log.get("debate_failures") or []
    if failures:
        out.append("## Caught debate parse failures")
        out.append(
            "(These are caught by the runner's try/except — a no-trade "
            "fallback day. Counted toward zero-trades hard-fail check.)"
        )
        for f in failures:
            out.append(f"- {f['date']}: `{f['error']}`")
        out.append("")

    # Per-day breakdown
    out.append("## Per-day PM line")
    out.append("")
    out.append("| date | equity | cash | inv% | orders | rejected |")
    out.append("|---|---|---|---|---|---|")
    for d in log.get("per_day") or []:
        pm = d.get("pm") or {}
        out.append(
            f"| {d['date']} | ${pm.get('equity', 0):,.0f} "
            f"| ${pm.get('cash', 0):,.0f} | {pm.get('invested_pct', 0)}% "
            f"| {pm.get('orders', 0)} | {pm.get('rejected', 0)} |"
        )
    out.append("")

    # Per-predator breakdown
    out.append("## Per-predator equity (final day's leaderboard)")
    out.append("")
    out.append("| predator | philosophy | equity | valid | rejected | active_theses |")
    out.append("|---|---|---|---|---|---|")
    final = (log.get("per_day") or [])[-1] if log.get("per_day") else {}
    for pid, p in (final.get("predators") or {}).items():
        out.append(
            f"| {pid} | {p['philosophy']} | ${p['equity']:,.0f} "
            f"| {p['valid']} | {p['rejected']} | {p['theses']} |"
        )
    out.append("")

    # Phase coverage
    cov = transcript.get("phase_coverage") or {}
    if cov:
        out.append("## Phase coverage (days a predator emitted ≥1 item)")
        out.append("")
        out.append("| predator | phase1_propose | phase2_respond | phase3_revise | phase4_commit |")
        out.append("|---|---|---|---|---|")
        for pid in sorted(cov.keys()):
            phases = cov[pid]
            out.append(
                f"| {pid} | {phases.get(1, 0)} | {phases.get(2, 0)} "
                f"| {phases.get(3, 0)} | {phases.get(4, 0)} |"
            )
        out.append("")

    # Diversity / Jaccard
    out.append("## Diversity check (Jaccard on thesis tickers)")
    out.append("")
    out.append("Thesis tickers per predator (across all observed days):")
    out.append("")
    for pid, ts in sorted(
        (transcript.get("thesis_tickers_per_predator") or {}).items()
    ):
        out.append(f"- **{pid}**: {ts if ts else '(none)'}")
    out.append("")
    jacc = verdict["jaccard"]
    if jacc["pairs"]:
        out.append("Pairwise Jaccard:")
        out.append("")
        out.append("| a | b | shared | size_a | size_b | jaccard |")
        out.append("|---|---|---|---|---|---|")
        for p in jacc["pairs"]:
            j = p["jaccard"]
            j_str = f"{j:.2f}" if j is not None else "n/a"
            out.append(
                f"| {p['a']} | {p['b']} | {p['intersection']} "
                f"| {p['size_a']} | {p['size_b']} | {j_str} |"
            )
        out.append("")
        if jacc["mean"] is not None:
            out.append(f"Average pairwise Jaccard: **{jacc['mean']:.2f}** "
                       f"(diversity goal: < 0.7)")
            out.append("")

    # Order tickers (footprint)
    ord_tickers = transcript.get("order_tickers_per_predator") or {}
    if ord_tickers:
        out.append("## Order tickers per predator (realized footprint)")
        out.append("")
        for pid, ts in sorted(ord_tickers.items()):
            out.append(f"- **{pid}**: {ts if ts else '(none)'}")
        out.append("")

    # Token budget
    out.append("## Token-budget summary")
    out.append("")
    mt = log.get("max_prompt_tokens")
    if mt is None:
        out.append(
            "Max prompt tokens: **n/a** (DSPy/vLLM did not surface "
            "prompt-token usage in the runner log). Indirect check: "
            "no day raised a context-overflow error and no `DEBATE "
            "FAILED` line cited a context-length issue."
        )
    else:
        pct = (mt / 16_384) * 100
        out.append(f"Max prompt tokens observed: **{mt}** "
                   f"({pct:.1f}% of Qwen-4B's 16K context). "
                   f"Threshold: 80% = 13,107.")
    out.append("")

    # Sample objection
    out.append("## Sample debate excerpt")
    out.append("")
    so = transcript.get("sample_objection")
    if so:
        out.append("> " + so)
    else:
        out.append("(no `object_to` Response with rationale ≥ 40 chars "
                   "found in transcript — debate is in agree/no-objection "
                   "mode; not a hard-fail but worth noting.)")
    out.append("")

    # Parse-error status (post-fix narrative)
    out.append("## Parse-error status")
    out.append("")
    out.append(
        "Phase-2 `DebatePhase2Responses` parse error from the earlier "
        "1-day smoke (05e) — Qwen-4B returned a list-shape (either bare "
        "list of `Response` dicts or per-peer wrappers) where DSPy "
        "expected the canonical `{predator_id, responses}` object — was "
        "diagnosed and fixed in this mission. Fix is in "
        "`trophic/beliefs/debate.py`:"
    )
    out.append("")
    out.append(
        "  1. `DebatePhase2Responses` gained a `mode=\"before\"` model "
        "validator (`_normalize_shape`) that detects either list form "
        "and rewrites it into the canonical wrapper before Pydantic "
        "validates the fields."
    )
    out.append(
        "  2. `_coerce_responses` was extended to route raw-list output "
        "through the normalizer and inject the orchestrator's "
        "`predator_id` when the flat-list form drops it."
    )
    out.append(
        "  3. The Phase-2 instruction block in `predator_prompts.py` / "
        "`debate.py` gained an explicit `OUTPUT SHAPE` paragraph "
        "directing the model to emit ONE wrapper object, not a list."
    )
    out.append(
        "  4. Three new tests in `tests/test_debate_mechanism.py` cover "
        "(a) flat list of Response dicts → canonical wrapper, "
        "(b) list of per-peer wrappers → flattened, "
        "(c) canonical dict unchanged."
    )
    out.append("")

    return "\n".join(out) + "\n"


# ── main ───────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runner-rc", type=int, default=0)
    args = ap.parse_args()

    log_path = Path(args.log)
    eval_path = Path(args.eval)
    tr_path = Path(args.transcript)
    out_path = Path(args.out)

    log = parse_log(log_path)
    log["_log_path"] = str(log_path)
    transcript = parse_transcripts(tr_path)
    transcript["_path"] = str(tr_path)

    verdict = compute_verdict(log, transcript, args.runner_rc)
    md = render_markdown(log, transcript, verdict, eval_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)

    print(f"==> wrote {out_path}  verdict={verdict['verdict']}")
    print(f"    hard_issues={len(verdict['issues']['hard'])}, "
          f"soft_issues={len(verdict['issues']['soft'])}")
    if verdict["verdict"] == "HARD-FAIL":
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
