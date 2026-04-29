"""Experiment A for issue #11: FinCoT-style structured CoT prompting probe.

Tests whether replacing the Predator's role_prefix with a CFA-aligned structured
reasoning blueprint moves StockNet MCC on `ipo_seed8_best.pt` (post-#6-fix
checkpoint, where MCC=0 was last measured).

This is a TIER-0 INTERVENTION — no training, prompt-only. Runs the existing
StockNet smoke harness with a monkey-patched ROLE_PROMPTS dict.

Source: Nitarach et al., FinCoT: Grounding Chain-of-Thought in Expert Financial
Reasoning. arXiv 2506.16123. Blueprint adapted from the paper's Equity
Investments and Quantitative Methods CFA-domain blueprints (page 15-16).

Reference baseline (greedy decoding, ipo_seed8_best.pt):
  ACCURACY=0.720, MCC=0.000 (constant-up predictor)

Decision criteria (from issue #11):
  MCC > 0.05 — interesting; structural reasoning helps without training
  MCC > 0.10 — paper-grade; one-line architecture intervention solves #8
  MCC ≈ 0.0  — falsifies prompting-only hypothesis; route to #10 training fix
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

# CRITICAL: monkey-patch ROLE_PROMPTS BEFORE importing anything that uses Predator.
# The role_prefix is encoded once via host.encode_role_prefix(...) at agent init,
# so the patch must be in place before Predator.ensure_initialized fires.
import trophic.agents.predator as _pred_mod

# FinCoT-blueprint-style prompt. Adapted from arXiv 2506.16123 page 15-16
# (Equity Investments + Quantitative Methods CFA blueprints), reformulated
# for our short-horizon directional prediction task.
#
# Key structural elements per the paper:
#   1. Numbered procedural steps (forces sequential reasoning)
#   2. Explicit reflection/verification stage before commit
#   3. Domain-keyed role identity (CFA analyst)
#   4. Structured output schema preserved at the end
FINCOT_PREDATOR_PROMPT = (
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

# Patch in-place so any subsequent Predator.ensure_initialized() sees this.
_pred_mod.ROLE_PROMPTS["short_horizon"] = FINCOT_PREDATOR_PROMPT

print(f"[exp-A] FinCoT prompt patched into ROLE_PROMPTS['short_horizon']")
print(f"[exp-A] new prompt length: {len(FINCOT_PREDATOR_PROMPT)} chars")

# Now run the standard StockNet eval harness against ipo_seed8_best.pt.
# We import the eval module's main() and call it with env vars set.
os.environ.setdefault("TROPHIC_CKPT", "checkpoints/ipo_seed8_best.pt")
os.environ.setdefault("TROPHIC_SEED", "8")
os.environ.setdefault("TROPHIC_LABEL", "ipo_seed8_FINCOT")
os.environ.setdefault("STOCKNET_MAX_PER_TICKER", "10")

# Import after monkey-patch is in place.
from scripts.eval_stocknet import main as eval_main  # type: ignore

if __name__ == "__main__":
    print(f"[exp-A] running StockNet smoke with FinCoT prompt + ipo_seed8_best.pt")
    print(f"[exp-A] reference: prior MCC=0.000, ACCURACY=0.720 (constant-up)")
    print(f"[exp-A] decision: MCC > 0.05 = interesting; > 0.10 = paper-grade")
    print()
    asyncio.run(eval_main())
