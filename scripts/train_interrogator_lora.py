"""Train a LoRA adapter on Qwen3-4B for the Interrogator's synthesizer phase.

Loss: teacher-forcing CE on `interrogator_target` text given a constructed
synthesizer prompt that contains the math findings teacher-style (derived
from the scenario's known target). This is supervised fine-tuning of the
adapter, not RL.

After this run, the LoRA can be loaded into ModelHost.get()._model via
peft.PeftModel.from_pretrained() to enable interrogator-improved Qwen
behavior at inference time.
"""
from __future__ import annotations

import asyncio
import functools
import os
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.training.interrogator_lora import (
    InterrogatorLoRAConfig,
    lora_train_synthesizer,
)
from trophic.training.scenarios import build_scenarios, split


async def main() -> None:
    cfg = DEFAULT_CONFIG
    lora_cfg = InterrogatorLoRAConfig(
        lr=float(os.environ.get("TROPHIC_LORA_LR", "5e-5")),
        steps=int(os.environ.get("TROPHIC_LORA_STEPS", "400")),
        log_every=int(os.environ.get("TROPHIC_LOG_EVERY", "25")),
        eval_every=int(os.environ.get("TROPHIC_EVAL_EVERY", "100")),
        rank=int(os.environ.get("TROPHIC_LORA_RANK", "16")),
        alpha=int(os.environ.get("TROPHIC_LORA_ALPHA", "32")),
        seed=int(os.environ.get("TROPHIC_SEED", "1")),
    )
    print(f"[init] lora_cfg={lora_cfg}")

    host = ModelHost.get(cfg.model)
    print(f"[init] hidden={host.hidden_size} dtype={host.dtype} device={host.device}")

    scenarios = build_scenarios()
    train, eval_ = split(scenarios)
    print(f"[init] scenarios: {len(train)} train, {len(eval_)} eval")

    ckpt_dir = ROOT / "checkpoints" / "interrogator_lora"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = ckpt_dir / f"seed{lora_cfg.seed}"

    result = lora_train_synthesizer(
        host=host,
        train=train,
        eval_=eval_,
        cfg=lora_cfg,
        ckpt_path=str(ckpt_best),
        print_fn=print,
    )

    print(f"\n[done] best_eval_loss={result['best_eval']:.4f}")
    print(f"       trainable_params={result['n_trainable']:,}")
    print(f"       LoRA adapter saved to {ckpt_best}")


if __name__ == "__main__":
    asyncio.run(main())
