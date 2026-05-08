"""Smoke test for Polygon-backed scenario loader.

If POLYGON_API_KEY is set:
  Fetch last 7 days × 3 tickers → build scenarios → print one packet.
If not:
  Show the expected structure and required env vars; exit cleanly.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Load .env
env = ROOT / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
            continue
        os.environ.setdefault(k, v)


def main() -> int:
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        print("[skip] POLYGON_API_KEY not in env")
        print()
        print("Set up Polygon (free tier: 5 calls/min, indefinite):")
        print("  1. Sign up: https://polygon.io/")
        print("  2. Dashboard → API Keys → copy key")
        print("  3. Add to /home/dgonier/ecology_experiment/trophic/.env:")
        print("     POLYGON_API_KEY=your_key_here")
        print()
        print("Then re-run: .venv/bin/python scripts/diagnostics/smoke_polygon_loader.py")
        return 0
    print(f"polygon key prefix: {key[:6]}…  len={len(key)}")

    from trophic.training.polygon_loader import build_polygon_scenarios

    # Tiny smoke: 3 tickers, last 14 days target window, 60-day lookback
    scens = build_polygon_scenarios(
        tickers=["AAPL", "MSFT", "JPM"],
        target_window_days=14,
        observation_lookback_days=60,
        history_days=5,
        max_per_ticker=10,
    )
    print(f"\nbuilt {len(scens)} scenarios")
    if not scens:
        print("[FAIL] no scenarios built — likely cache + API both empty")
        return 1

    # Inspect first 3
    for sc in scens[:3]:
        print(f"\n--- {sc.name} ---")
        print(f"  inputs: {[i.source for i in sc.inputs]}")
        for inp in sc.inputs:
            if inp.source == "ohlcv":
                bars = inp.payload.get("bars", [])
                print(f"  ohlcv: {len(bars)} bars")
                if bars:
                    last = bars[-1]
                    print(f"    last bar: open={last.get('open'):.2f} close={last.get('close'):.2f}")
            elif inp.source == "press":
                body = inp.payload.get("body", "")
                hl = inp.payload.get("headline", "")
                print(f"  press: {hl}")
                # First 2 headlines
                for line in body.split("\n")[:4]:
                    print(f"    {line[:120]}")
        print(f"  predator_target (label): {sc.predator_target[:140]}")
    print()
    print("✓ Polygon loader smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
