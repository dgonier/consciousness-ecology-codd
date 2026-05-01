"""Extract d* per-layer direction from StockNet train-set up vs down days.

Renders each scenario as a structured prompt (same shape used by
`scripts/diagnostics/baseline_promptonly_stocknet.py`) and feeds through
Qwen3-4B. Up-labeled days populate `pro_texts`; down-labeled days populate
`con_texts`. We then call `extract_dstar` to compute the per-layer
unit-norm direction `d* = normalize(mean(h_pro) - mean(h_con))` on
`DEFAULT_PATCHED_LAYERS` (stride-3, 11 layers).

Usage:
    .venv/bin/python -u scripts/extract_dstar_stocknet.py \
        --out checkpoints/dstar_stocknet.pt \
        --max-per-side 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from trophic.config import DEFAULT_CONFIG  # noqa: E402
from trophic.dstar import extract_dstar  # noqa: E402
from trophic.model_host import ModelHost  # noqa: E402
from trophic.phi_mlp import DEFAULT_PATCHED_LAYERS  # noqa: E402
from trophic.training.stocknet_loader import build_stocknet_scenarios  # noqa: E402
from trophic.training.xml_schema import parse_prediction  # noqa: E402


def render_scenario(sc) -> str:
    """Same prompt shape as the prompt-only baseline (MCC=0.292)."""
    parts = []
    # sc.name is "stocknet_<split>_<ticker>_<date>" — split on "_"
    name_parts = sc.name.split("_")
    ticker = name_parts[2] if len(name_parts) >= 3 else "?"
    for inp in sc.inputs:
        if inp.source == "ohlcv":
            bars = inp.payload.get("bars", [])
            parts.append(f"OHLCV history for {ticker} ({len(bars)} bars):")
            for b in bars[-5:]:
                parts.append(
                    f"  open={b.get('open', 0):.6f} high={b.get('high', 0):.6f} "
                    f"low={b.get('low', 0):.6f} close={b.get('close', 0):.6f} "
                    f"vol={b.get('volume', 0)}"
                )
        elif inp.source == "press":
            t = inp.payload.get("ticker") or ticker
            body = (inp.payload.get("body") or "").strip()
            parts.append(f"\nTweets for {t}:")
            parts.append(body[:1500])
    return "\n".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="checkpoints/dstar_stocknet.pt")
    p.add_argument(
        "--max-per-side",
        type=int,
        default=64,
        help="Cap on number of up days and down days each",
    )
    p.add_argument("--split", type=str, default="train")
    args = p.parse_args()

    host = ModelHost.get(DEFAULT_CONFIG.model)
    print(f"[dstar] hidden={host.hidden_size} device={host.device}")

    scenarios = build_stocknet_scenarios(
        split=args.split,
        tickers=["AAPL", "GOOG", "MSFT", "AMZN", "JPM"],
        max_per_ticker=None,
    )
    print(f"[dstar] {len(scenarios)} {args.split} scenarios")

    pro_texts: list[str] = []
    con_texts: list[str] = []
    for sc in scenarios:
        target = parse_prediction(sc.predator_target or "")
        if not target.direction:
            continue
        txt = render_scenario(sc)
        if target.direction.lower() == "up" and len(pro_texts) < args.max_per_side:
            pro_texts.append(txt)
        elif target.direction.lower() == "down" and len(con_texts) < args.max_per_side:
            con_texts.append(txt)

    print(f"[dstar] pro={len(pro_texts)}, con={len(con_texts)}")
    if len(pro_texts) < 8 or len(con_texts) < 8:
        raise RuntimeError(
            f"Not enough samples: pro={len(pro_texts)} con={len(con_texts)}"
        )

    dstar = extract_dstar(
        base_model=host._model,
        tokenizer=host._tok,
        pro_texts=pro_texts,
        con_texts=con_texts,
        patched_layers=DEFAULT_PATCHED_LAYERS,
        device=host.device,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dstar.save(out_path)
    print(f"[dstar] saved -> {out_path}")
    for l, v in dstar.directions.items():
        print(f"  layer {l:2d}: norm={v.norm().item():.4f}  (should be 1.0)")


if __name__ == "__main__":
    main()
