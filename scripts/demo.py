"""End-to-end demo for the three-tier hidden-state architecture.

Usage:
  # Mocked smoke run (CPU only, no Bedrock):
  TROPHIC_MOCK_MODELS=1 TROPHIC_MOCK_PREDATOR=1 python scripts/demo.py

  # Real run (loads Qwen3-4B + retrieval embedder; calls Bedrock):
  python scripts/demo.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trophic.config import TrophicConfig
from trophic.runner import Runner


def _short(s: str | None, n: int = 100) -> str:
    if not s:
        return "<none>"
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


async def main() -> None:
    cfg = TrophicConfig()
    cfg.runner.max_ticks = int(os.environ.get("TROPHIC_TICKS", "5"))
    cfg.runner.inputs_per_tick = int(os.environ.get("TROPHIC_INPUTS_PER_TICK", "6"))
    cfg.pool.db_path = ":memory:"

    runner = Runner(cfg=cfg)
    print(f"# Trophic demo — {cfg.runner.max_ticks} ticks, "
          f"{cfg.runner.inputs_per_tick} inputs/tick")
    print(f"#   model.mock={cfg.model.mock}  apex.mock={cfg.bedrock.mock}")
    print(f"#   bedrock.model_id={cfg.bedrock.model_id}")
    print()

    results = await runner.run()
    for r in results:
        m = runner.metrics.rows[r.tick - 1]
        print(f"=== TICK {r.tick} ===  "
              f"inputs={m.n_inputs} substrate={m.n_substrate_added} "
              f"eaten={m.n_eaten} rotted={m.n_rotted} "
              f"avg_judgment={m.avg_judgment}")
        # Tier 1 herbivore broadcasts
        for br in r.herbivore_broadcasts:
            tag = "ABSTAIN" if br.abstained else "EAT"
            print(f"  [H {br.agent_kind:<11}] {tag} "
                  f"avg_null={r.abstentions.get(br.agent_id, 0):.3f} "
                  f"parents={len(br.parent_input_ids)}")
            print(f"     {_short(br.decoded_text)}")
        # Tier 2 predator broadcasts
        for br in r.predator_broadcasts:
            tag = "ABSTAIN" if br.abstained else "PREDICT"
            j = r.judgments.get(br.id)
            jstr = f"score={j.score:.2f}" if j else "score=?"
            print(f"  [P {br.agent_kind:<11}] {tag} "
                  f"avg_null={r.abstentions.get(br.agent_id, 0):.3f} "
                  f"parents={len(br.parent_input_ids)} {jstr}")
            print(f"     {_short(br.decoded_text)}")
            if j:
                print(f"     JDG: {_short(j.rationale)}")
        # Population energies
        snap = r.population_snapshot
        line = " | ".join(
            f"{p['kind']}={p['energy']:.2f}" for p in snap["producers"].values()
        )
        print(f"  prods: {line}")
        line = " | ".join(
            f"{h['kind']}={h['energy']:.2f}" for h in snap["herbivores"].values()
        )
        print(f"  herbs: {line}")
        line = " | ".join(
            f"{p['kind']}={p['energy']:.2f}" for p in snap["predators"].values()
        )
        print(f"  preds: {line}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
