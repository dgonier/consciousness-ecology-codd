"""Ping every voter with a tiny prompt to confirm credentials work
before running a full eval. Reports: (voter_id, available, ok, error).

Usage:
    .venv/bin/python -u scripts/diagnostics/voter_health_check.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from trophic.apex_voters import (
    AnthropicVoter, GeminiVoter, LocalQwenVoter, OpenAIVoter, OpenRouterVoter,
    EvidencePacket,
)


def make_test_packet() -> EvidencePacket:
    text = (
        "You are a short-horizon market direction predictor. The stock"
        " is TEST. Recent close 100, prior 99. Predict direction.\n\n"
        "REASONING: <brief>\n"
        "<prediction><ticker>TEST</ticker><direction>up|down</direction>"
        "<horizon_min>1440</horizon_min><confidence>0.55</confidence></prediction>"
    )
    return EvidencePacket(
        scenario_name="health_check",
        ticker="TEST",
        text=text,
        metadata={},
    )


def main():
    candidates = [
        OpenAIVoter(),
        AnthropicVoter(),
        GeminiVoter(),
        OpenRouterVoter(),
        LocalQwenVoter(),
    ]
    pkt = make_test_packet()
    print(f"{'voter_id':50s}  {'avail':6s} {'lat_s':8s}  result")
    print("-" * 90)
    for v in candidates:
        avail = v.is_available()
        if not avail:
            print(f"{v.voter_id:50s}  {'no':6s} {'-':8s}  (no credentials)")
            continue
        t0 = time.time()
        try:
            r = v.vote(pkt)
            elapsed = time.time() - t0
            ok = r.direction in ("up", "down")
            note = (
                f"direction={r.direction}  conf={r.confidence}  ppl={r.perplexity:.2f}"
                if ok
                else f"FAILED: {r.raw_text[:120]!r}"
            )
            print(f"{v.voter_id:50s}  {'yes':6s} {elapsed:7.2f}s  {note}")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"{v.voter_id:50s}  {'yes':6s} {elapsed:7.2f}s  EXC: {e}")


if __name__ == "__main__":
    main()
