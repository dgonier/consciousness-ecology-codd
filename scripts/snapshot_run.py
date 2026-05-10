"""Snapshot the current pass artifacts + timeline into a labeled run dir.

Lets us preserve each PM sweep's results before launching a follow-up
ablation. Each run dir is self-contained:

  data/firehose_eval/runs/<label>/
    passes/*.json.gz         — full pass artifacts (with pm_orders_*)
    timeline/                — per-day timeline files + MANIFEST
    run_meta.json            — cmd line, git sha, final equities, flags

Usage:
  .venv/bin/python -m scripts.snapshot_run --label run_2026-05-10_pm_v1_baseline \
      --note "First PM sweep: retry+yield+rotate available but no oracles, no Hexis. Local vLLM 8K context."
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PASS_DIR = ROOT / "data" / "firehose_eval" / "passes"
TIMELINE_DIR = ROOT / "data" / "firehose_eval" / "timeline"
RUNS_DIR = ROOT / "data" / "firehose_eval" / "runs"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def extract_pm_summary(pass_paths: list[Path]) -> dict:
    """Walk pass artifacts to extract final equity per pipeline."""
    eco_curve: list[tuple[str, float]] = []
    bare_curve: list[tuple[str, float]] = []
    oracle_qwen_curve: list[tuple[str, float]] = []
    oracle_sonnet_curve: list[tuple[str, float]] = []

    for p in pass_paths:
        rec = json.load(gzip.open(p, "rt"))
        d = rec["date"]
        for label, curve in [
            ("ecology", eco_curve),
            ("bare", bare_curve),
            ("oracle_qwen", oracle_qwen_curve),
            ("oracle_sonnet", oracle_sonnet_curve),
        ]:
            wl = rec.get(f"watchlist_{label}", [])
            # PM snapshot is in pass_record's pm_snapshot_post if present;
            # otherwise we just don't have it for this label.
        # Pass artifact doesn't carry pm snapshot directly; we need to
        # also check the eval JSONL or just skip. For now: count #orders.
    return {"note": "Per-day equity curves not pulled here; check timeline files."}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True,
                    help="Subdirectory under data/firehose_eval/runs/")
    ap.add_argument("--note", default="",
                    help="Free-form description of what this run tested")
    ap.add_argument("--cmd", default="",
                    help="The runner command line that produced these passes")
    ap.add_argument("--feature-flags", default="",
                    help="Comma-sep summary of feature flags, e.g. "
                         "'retry=2,yield=0.04,rotate=on,oracles=off'")
    ap.add_argument("--final-equities", default="",
                    help="Comma-sep label=equity pairs, e.g. "
                         "'ECOLOGY=103673,BARE=95743'")
    ap.add_argument("--clear-source", action="store_true",
                    help="After successful copy, DELETE the source pass + "
                         "timeline files so the next run starts fresh. "
                         "Skip this to keep them in place for inspection.")
    args = ap.parse_args()

    out_dir = RUNS_DIR / args.label
    if out_dir.exists():
        print(f"ERROR: {out_dir} already exists. Pick a different --label.")
        return

    out_dir.mkdir(parents=True)
    pass_out = out_dir / "passes"
    pass_out.mkdir()
    tl_out = out_dir / "timeline"
    tl_out.mkdir()

    # Copy pass artifacts
    pass_paths = sorted(PASS_DIR.glob("*.json.gz"))
    for p in pass_paths:
        shutil.copy2(p, pass_out / p.name)
    print(f"Copied {len(pass_paths)} pass artifacts → {pass_out}")

    # Copy timeline (if it exists)
    if TIMELINE_DIR.exists():
        n_tl = 0
        for p in TIMELINE_DIR.glob("*.json"):
            shutil.copy2(p, tl_out / p.name)
            n_tl += 1
        print(f"Copied {n_tl} timeline files → {tl_out}")
    else:
        print("(no timeline to snapshot — re-run build_timeline_data.py if desired)")

    # Run meta
    meta = {
        "label": args.label,
        "snapshotted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": git_sha(),
        "n_pass_artifacts": len(pass_paths),
        "first_date": pass_paths[0].stem.removesuffix(".json") if pass_paths else None,
        "last_date": pass_paths[-1].stem.removesuffix(".json") if pass_paths else None,
        "note": args.note,
        "cmd": args.cmd,
        "feature_flags": args.feature_flags,
        "final_equities": args.final_equities,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_dir / 'run_meta.json'}")

    if args.clear_source:
        for p in pass_paths:
            p.unlink()
        if TIMELINE_DIR.exists():
            for p in TIMELINE_DIR.glob("*.json"):
                p.unlink()
        print("Cleared source pass + timeline files")
    else:
        print("(source pass + timeline files preserved; pass --clear-source to remove)")


if __name__ == "__main__":
    main()
