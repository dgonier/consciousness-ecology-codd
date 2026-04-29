"""Eval ANY checkpoint against the 68 held-out scenarios. Set TROPHIC_CKPT + label."""
from __future__ import annotations
import asyncio, builtins, functools, os, sys
from pathlib import Path
builtins.print = functools.partial(builtins.print, flush=True)
print = builtins.print
ROOT = Path("/home/dgonier/ecology_experiment/trophic")
sys.path.insert(0, str(ROOT))
import torch
from trophic.agents.herbivore import Herbivore
from trophic.agents.predator import Predator
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.training.checkpoint import load_channels
from trophic.training.holdout_scenarios import build_holdout_scenarios
from trophic.training.sft import SFTConfig, SFTRunner
from trophic.training.xml_schema import parse_prediction, reward_prediction


async def main() -> None:
    cfg = DEFAULT_CONFIG
    seed = int(os.environ.get("TROPHIC_SEED", "1"))
    ckpt_path = os.environ.get("TROPHIC_CKPT", "checkpoints/sft_seed1_best.pt")
    label = os.environ.get("TROPHIC_LABEL", "unknown")
    print(f"[holdout] label={label} ckpt={ckpt_path} seed={seed}")
    host = ModelHost.get(cfg.model)
    _lora_dir = os.environ.get("TROPHIC_LORA_DIR", "")
    if _lora_dir:
        from peft import PeftModel
        print(f"[holdout] attaching LoRA from {_lora_dir}")
        host._model = PeftModel.from_pretrained(host._model, _lora_dir, is_trainable=False)
        host._model.eval()
    sft_cfg = SFTConfig(seed=seed)
    herbivores = [Herbivore.make(k, capacity=cfg.population.intake_budget) for k in ("technical", "fundamental")]
    predator = Predator.make("short_horizon")
    for h in herbivores: h.ensure_initialized(host, seed_base=seed)
    predator.ensure_initialized(host, seed_base=seed)
    for h in herbivores:
        for ch in h.channels.values(): ch.to(device=host.device, dtype=host.dtype)
    for ch in predator.channels.values(): ch.to(device=host.device, dtype=host.dtype)
    meta = load_channels(ckpt_path, herbivores=herbivores, predators=[predator])
    print(f"[holdout] loaded: {meta}")
    for ch in predator.channels.values(): ch.to(device=host.device, dtype=host.dtype)
    for h in herbivores:
        for ch in h.channels.values(): ch.to(device=host.device, dtype=host.dtype)
    holdout = build_holdout_scenarios()
    print(f"[holdout] {len(holdout)} scenarios")
    from trophic.agents.producer import Producer
    from trophic.agents.quant_producer import QuantitativeProducer
    producers = [Producer.make(k) for k in ("tickdelta", "disclosure", "anomaly")]
    producers.append(QuantitativeProducer.make("quote_series"))
    runner = SFTRunner(cfg=sft_cfg, host=host, producers=producers,
                       herbivores=herbivores, predator=predator, train=[], eval_=holdout)
    await runner._cache_producer_broadcasts()
    rewards = []
    print(f"\n=== HELDOUT [{label}] ===")
    for sc in holdout:
        out = runner.eval_decode(sc)
        pred_text = out.get("pred.short_horizon", "")
        target = parse_prediction(sc.predator_target or "")
        parsed = parse_prediction(pred_text)
        rew = reward_prediction(parsed, target)
        rewards.append(rew["total"])
    mean_r = sum(rewards) / max(1, len(rewards))
    print(f"\n=== MEAN [{label}] HELDOUT REWARD: {mean_r:.3f} over {len(rewards)} ===")


if __name__ == "__main__":
    asyncio.run(main())
