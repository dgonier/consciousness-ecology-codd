"""SFT seed24 with the per-layer M+E rewrite + ORPO + d*.

Sets TROPHIC_CONSUMER_INTERFACE=hooks and TROPHIC_DSTAR_PATH=checkpoints/dstar_stocknet.pt
before invoking the existing SFTRunner; everything else flows through the
existing infrastructure. Saves checkpoints to:
    checkpoints/sft_seed24_best.pt
    checkpoints/sft_seed24_final.pt

(Naming follows the existing seed{N} convention from train_sft.py so
eval_stocknet.py picks them up via TROPHIC_CKPT without any rename.)
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Activate the new architecture BEFORE any trophic imports load.
os.environ.setdefault("TROPHIC_CONSUMER_INTERFACE", "hooks")
os.environ.setdefault(
    "TROPHIC_DSTAR_PATH", str(ROOT / "checkpoints" / "dstar_stocknet.pt")
)
os.environ.setdefault("TROPHIC_SEED", "24")
os.environ.setdefault("TROPHIC_STEPS", "200")
# Cap eval-decode generation so per-eval wall-time stays reasonable in hooks
# mode (each ORPO step does TWO forward passes vs prefix mode's one).
os.environ.setdefault("TROPHIC_EVAL_MAX_TOKENS", "96")

from scripts.train_sft import main as sft_main  # noqa: E402


if __name__ == "__main__":
    asyncio.run(sft_main())
