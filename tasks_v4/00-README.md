# v4 — Multi-horizon forecasting + strategy committee + tax-aware ECO

## What changed from v3.3

v3.3 (archived at `data/firehose_eval/runs/run_2026-05-10_pm_v3_3_9way/`) finished
with three uncomfortable conclusions:

1. **ECO produced more realized P&L than BARE on Qwen** (+$6,792 vs +$5,104)
   but lost the net race (+8.32% vs +9.67%) entirely to tax friction from
   extra trades.
2. **ORACLE-next-day was an architectural failure**. Even Oracle-Opus, which
   was given perfect 1-day foresight, finished at +0.17% net because tax
   ($9,998 owed) and slippage from 384 orders consumed all of its $9,422
   realized alpha. Pre-tax it would have been +10.17%.
3. **Sonnet was worst-of-3 in every pipeline** — a model-capability issue,
   not a pipeline issue.

The v4 architecture addresses each:

- **Multi-horizon forecasting** (h1/h5/h20/h60) committed at order time —
  forces the apex to commit to a holding period, which both surfaces ECO's
  multi-scale signal *and* makes tax-aware validation possible.
- **Full-window Oracle** sees all 66 days at once and is prompted to
  produce an infrequent buy-low-sell-high trajectory, not a per-day reaction.
- **Three-strategy committee** (conviction-weighted / tiered / Kelly-edge)
  votes on each order via weighted-average sizing, with philosophy weights
  configurable in `philosophy_weights.yaml`.
- **Tax-aware validators** enforce minimum holding by committed horizon and
  reject orders whose expected alpha doesn't clear the slippage+tax floor.
- **Interrogator herbivore** (already exists at
  `trophic/agents/interrogator_herbivore.py`, never wired into PM) gets
  plumbed into the firehose pipeline.

The multi-hop tool-use extension to the interrogator is **parked** as
`phase4-PARKED-06`. That's the next-next step.

## Missions

| Handle      | File                                          | Phase | Depends on   | Blocks       |
|-------------|-----------------------------------------------|-------|--------------|--------------|
| phase1-A:01 | phase1-A-01-multi-horizon-signatures.md       | 1     | none         | 02, 03, 04   |
| phase2-B:02 | phase2-B-02-full-window-oracle.md             | 2     | 01           | 05           |
| phase2-C:03 | phase2-C-03-strategy-committee.md             | 2     | 01           | 05           |
| phase2-D:04 | phase2-D-04-tax-aware-validators.md           | 2     | 01           | 05           |
| phase2_5:smoke | phase2_5-smoke-gate.md                     | 2.5   | 02, 03, 04   | 05           |
| phase3-A:05 | phase3-A-05-wire-interrogator-and-run.md      | 3     | 02, 03, 04, smoke | phase4 (parked) |
| phase4:PARKED | phase4-PARKED-06-multi-hop-tool-use.md     | 4     | 05 (when unparked) | future v5 work |

## STATUS at scaffold time

```
phase1-A:01:PENDING
phase2-B:02:BLOCKED  (needs phase1-A:01)
phase2-C:03:BLOCKED  (needs phase1-A:01)
phase2-D:04:BLOCKED  (needs phase1-A:01)
phase2_5:smoke:BLOCKED  (needs all of phase 2)
phase3-A:05:BLOCKED   (needs phase 2 + smoke gate)
phase4:PARKED:PARKED  (documentation only; deferred)
```

## Phase plan

```
  Phase 1 (foundation, sequential, agent A)
        phase1-A:01 — signatures, types, tax-framing
                 │
        ┌────────┴───────────────────────┐
        │              │                  │
  Phase 2 (parallel, agents B/C/D)
   phase2-B:02   phase2-C:03   phase2-D:04
   (oracle)     (committee)   (validators)
        │              │                  │
        └────────┬─────┴──────────────────┘
                 │
   Phase 2.5 (sequential smoke gate, any agent — verify parts compose)
        phase2_5:smoke
                 │
  Phase 3 (sequential, agent A, with heavy testing)
        phase3-A:05 — wire interrogator + run v4 sweep
                 │
  Phase 4 (PARKED, no work yet)
        phase4:06 — multi-hop tool-use vision doc
```

Phase 2.5 is non-negotiable. Phase 3 launches a 66-day GPU sweep — that's
an expensive failure if any of the phase-2 components produce orders the
others can't accept. The smoke gate runs a 3-day mini-sweep and verifies
every component contract holds end-to-end.

## Reading order for agents

1. Read this README.
2. Read `scratchpad.md` end-to-end (STATUS, INBOX PROTOCOL, INTERFACE CONTRACTS).
3. Open your mission file.
4. Run the inbox grep before starting work.
5. Run the inbox grep before flipping DONE.

## v3.3 reference artifacts

- `data/firehose_eval/runs/run_2026-05-10_pm_v3_3_9way/eval.jsonl` — full eval JSONL (6.1MB).
- `data/firehose_eval/runs/run_2026-05-10_pm_v3_3_9way/sweep.log` — full sweep log.
- `data/firehose_eval/runs/run_2026-05-10_pm_v3_3_9way/run_meta.json` — final equities (net and pre-tax), order counts, headline findings.

Compare any v4 run against these numbers; "+9.67% net (BARE-QWEN)" is
the bar to beat, and "+13.49% pre-tax (BARE-QWEN)" is the
upper-bound-without-changing-tax-policy.

## Working test command

```bash
cd /home/dgonier/ecology_experiment/trophic && .venv/bin/python -m pytest tests/ -x
```

## Author handoff at completion

When all phases land, write `data/firehose_eval/runs/run_<date>_pm_v4/run_meta.json`
mirroring the v3.3 archive format.
