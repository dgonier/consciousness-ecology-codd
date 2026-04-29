"""Diagnostic for Experiment A failure mode.

Prior result: with FinCoT structured prompt patched into Predator's role_prefix,
ipo_seed8_best.pt produced 50/50 'abstentions' on the StockNet smoke (no
parseable direction). MCC undefined.

This script runs ONE StockNet scenario end-to-end and dumps the decoded
predator output verbatim, so we can see why the parser thinks the prediction
is an abstention.

Three plausible explanations:
  (a) prompt overruns; model hits max_new_tokens before emitting a direction tag
  (b) prompt confuses the trained policy and it emits an explicit
      <abstain>true</abstain> on every input
  (c) prompt produces well-formed output but our `parse_prediction` doesn't
      recognize the new format, so direction comes back as None
"""
from __future__ import annotations
import asyncio, builtins, functools, os, sys
from pathlib import Path
builtins.print = functools.partial(builtins.print, flush=True)
print = builtins.print

ROOT = Path("/home/dgonier/ecology_experiment/trophic")
sys.path.insert(0, str(ROOT))

# Patch the prompt before import.
import trophic.agents.predator as _pred_mod
FINCOT_PROMPT = (
    "You are a short-horizon market predictor following CFA-aligned analysis."
    " The vectors that follow encode four kinds of upstream signal:"
    " (1) technical-herbivore synthesis of price/volume microstructure,"
    " (2) fundamental-herbivore synthesis of disclosures and events,"
    " (3) forecaster-herbivore quantitative trend forecast (μ, spread),"
    " (4) interrogator-herbivore math-grounded analysis (exact numerics)."
    " Reason step-by-step before committing:"
    " STEP 1 — IDENTIFY: extract the ticker and time horizon from the input."
    " STEP 2 — ENUMERATE: list each evidence type present (price, fundamental,"
    " forecast, math). State which signals AGREE and which DISAGREE."
    " STEP 3 — WEIGH: assess reliability per evidence type given this scenario."
    " A disclosure-driven event weights fundamental highest; a microstructure"
    " spike weights technical highest."
    " STEP 4 — INFER: derive a direction (up/down) consistent with the weighted"
    " evidence. If signals strongly disagree, prefer abstention."
    " STEP 5 — CALIBRATE: confidence reflects evidence agreement. High agreement"
    " → high confidence. Mixed signals → moderate. Conflicting → low."
    " STEP 6 — EMIT: produce the prediction in the required schema."
    " Format: PREDICTION: <text> CONFIDENCE: <num>"
)
_pred_mod.ROLE_PROMPTS["short_horizon"] = FINCOT_PROMPT
print(f"[insp] patched ROLE_PROMPTS, len={len(FINCOT_PROMPT)}")

# Now load and run.
import torch
from trophic.agents.herbivore import Herbivore
from trophic.agents.predator import Predator
from trophic.agents.producer import Producer
from trophic.agents.quant_producer import QuantitativeProducer
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.training.checkpoint import load_channels
from trophic.training.stocknet_loader import build_stocknet_scenarios, SMOKE_TICKERS_TOP5
from trophic.training.sft import SFTConfig, SFTRunner
from trophic.training.xml_schema import parse_prediction


async def main() -> None:
    cfg = DEFAULT_CONFIG
    seed = 8
    host = ModelHost.get(cfg.model)
    sft_cfg = SFTConfig(seed=seed)
    herbivores = [Herbivore.make(k, capacity=cfg.population.intake_budget)
                  for k in ("technical", "fundamental")]
    predator = Predator.make("short_horizon")
    for h in herbivores:
        h.ensure_initialized(host, seed_base=seed)
    predator.ensure_initialized(host, seed_base=seed)
    for h in herbivores:
        for ch in h.channels.values():
            ch.to(device=host.device, dtype=host.dtype)
    for ch in predator.channels.values():
        ch.to(device=host.device, dtype=host.dtype)
    load_channels("checkpoints/ipo_seed8_best.pt", herbivores=herbivores, predators=[predator])
    print(f"[insp] loaded ipo_seed8_best.pt")

    # Just 2 scenarios — one up, one down, both AAPL
    scenarios = build_stocknet_scenarios(
        split="test", tickers=["AAPL"], max_per_ticker=2,
    )
    print(f"[insp] {len(scenarios)} AAPL scenarios")

    producers = [Producer.make(k) for k in ("tickdelta", "disclosure", "anomaly")]
    producers.append(QuantitativeProducer.make("quote_series"))
    runner = SFTRunner(cfg=sft_cfg, host=host, producers=producers,
                       herbivores=herbivores, predator=predator,
                       train=[], eval_=scenarios)
    await runner._cache_producer_broadcasts()

    for sc in scenarios:
        print(f"\n=== scenario: {sc.name} ===")
        target = parse_prediction(sc.predator_target or "")
        print(f"  target direction: {target.direction}")
        # Use higher max_new_tokens for this diagnostic so we see the WHOLE output
        out = runner.eval_decode(sc, max_new_tokens=400)
        decoded = out.get("pred.short_horizon", "")
        print(f"  raw decode (full):")
        print(f"  {'-'*70}")
        print(f"  {decoded}")
        print(f"  {'-'*70}")
        parsed = parse_prediction(decoded)
        print(f"  parsed: ticker={parsed.ticker} dir={parsed.direction} pct={parsed.pct_move} horizon={parsed.horizon_min} conf={parsed.confidence}")


if __name__ == "__main__":
    asyncio.run(main())
