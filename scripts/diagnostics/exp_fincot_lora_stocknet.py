"""Integration test for issue #10 phase 1: FinCoT LoRA + StockNet smoke.

After interrogator_fincot_lora training completes, this script:
  1. Loads the base Qwen3-4B via the standard ModelHost
  2. Attaches the FinCoT LoRA adapter on top
  3. Loads ipo_seed8_best.pt (our best pre-LoRA Channel weights)
  4. Runs StockNet smoke on ipo_seed8_best with FinCoT-trained Qwen3-4B underneath

Pre-LoRA baseline on ipo_seed8_best.pt: ACCURACY=0.720, MCC=0.000
                                        (predator emits constant "up")

Decision criteria:
  MCC > 0.10 → FinCoT LoRA broke direction collapse. Phase 1 of #10 succeeds.
              Paper claim: structured reasoning installed via LoRA solves #8.
  MCC > 0.05 → partial. Direction signal present but weak; may need more
              training steps or larger LoRA rank.
  MCC ≈ 0   → FinCoT didn't help even with training. Reasoning structure isn't
              the bottleneck; likely needs Disclosure/SocialSignal CPT (next
              species in #10's plan) for a different intervention angle.

Env vars:
  TROPHIC_FINCOT_LORA_DIR (default: checkpoints/interrogator_fincot_lora/seed11)
  TROPHIC_CKPT (default: checkpoints/ipo_seed8_best.pt)
  TROPHIC_SEED (default: 8 — to match ipo_seed8 Channel init seeding)
  STOCKNET_MAX_PER_TICKER (default: 10 — top-5 × 10 days = 50 scenarios)
  TROPHIC_EVAL_MAX_TOKENS (default: 96)

Note: this attaches LoRA to the SHARED ModelHost, so all Qwen3-4B forwards
(producers, herbivores, predator) go through the FinCoT adapter. That's
intentional for v1 — measures the architecture-wide effect of installing
financial reasoning as the model's default behavior. Per-species isolation
(only-Interrogator-uses-adapter) is a v2 ramp.
"""
from __future__ import annotations

import asyncio
import builtins
import functools
import os
import sys
from pathlib import Path

builtins.print = functools.partial(builtins.print, flush=True)
print = builtins.print

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def attach_lora_to_modelhost(host, lora_dir: str):
    """Attach a saved PEFT LoRA adapter to host._model in place.

    PEFT's PeftModel.from_pretrained wraps the existing model. After this
    call, all forward passes route through the adapter automatically.
    """
    from peft import PeftModel
    print(f"[exp-fincot-lora] attaching adapter from {lora_dir}")
    host._model = PeftModel.from_pretrained(host._model, lora_dir)
    host._model.eval()
    n_lora = sum(p.numel() for p in host._model.parameters() if p.requires_grad)
    print(f"[exp-fincot-lora] adapter active. trainable params (eval mode): {n_lora:,}")


async def main() -> None:
    lora_dir = os.environ.get(
        "TROPHIC_FINCOT_LORA_DIR",
        "checkpoints/interrogator_fincot_lora/seed11",
    )
    if not Path(lora_dir).exists():
        raise FileNotFoundError(f"LoRA adapter not found at {lora_dir}")
    print(f"[exp-fincot-lora] LoRA dir: {lora_dir}")

    os.environ.setdefault("TROPHIC_CKPT", "checkpoints/ipo_seed8_best.pt")
    os.environ.setdefault("TROPHIC_SEED", "8")
    os.environ.setdefault("STOCKNET_MAX_PER_TICKER", "10")
    os.environ.setdefault("TROPHIC_EVAL_MAX_TOKENS", "96")

    from trophic.config import DEFAULT_CONFIG
    from trophic.model_host import ModelHost
    host = ModelHost.get(DEFAULT_CONFIG.model)
    print(f"[exp-fincot-lora] base model loaded; hidden={host.hidden_size}")

    attach_lora_to_modelhost(host, lora_dir)

    print(f"[exp-fincot-lora] running StockNet smoke with FinCoT LoRA active")
    print(f"[exp-fincot-lora] reference: pre-LoRA baseline = ACC 0.720, MCC 0.000")
    print(f"[exp-fincot-lora] decision: MCC > 0.05 = interesting; > 0.10 = paper-grade")
    print()
    from scripts.eval_stocknet import main as eval_main
    await eval_main()


if __name__ == "__main__":
    asyncio.run(main())
