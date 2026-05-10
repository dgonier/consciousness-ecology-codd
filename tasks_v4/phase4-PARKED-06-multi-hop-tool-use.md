# Mission 06 (PARKED): multi-hop tool-use for the interrogator herbivore

**Handle**: phase4 (PARKED — no agent assigned)
**Phase**: 4 (deferred until phase 3 lands and is analyzed)
**Mission file**: phase4-PARKED-06-multi-hop-tool-use.md
**Dependencies**: phase3-A:05 (must complete and produce headline numbers)
**Blocks**: nothing (this is the leading edge)

**STATUS**: PARKED. This is a vision doc for the next architectural
expansion, not an active mission. Do not start work on this until phase 3
has landed and there's an explicit decision to proceed.

---

## Why this is parked

The v3.3 → v4 transition is already a large architectural lift
(multi-horizon, committee voting, tax-aware validators, full-window
oracle, interrogator wiring). Adding tool-use to the interrogator in the
same release would muddy the signal: we wouldn't know whether v4's
performance came from multi-horizon, committee, validators, or
interrogator tool-use.

We need a clean v4 measurement first, then a v4 → v5 measurement that
isolates the tool-use contribution.

---

## The vision

Today, `InterrogatorHerbivore` (`trophic/agents/interrogator_herbivore.py`)
is a two-agent setup:

```
producer broadcasts → Qwen3-4B planner (asks math questions)
                    → Qwen2.5-Math-1.5B solver (answers)
                    → Qwen3-4B synthesizer (writes XML synthesis broadcast)
```

The planner emits questions like `"Open=100, close=102.5. Percent change?"`
which the math host answers numerically. The synthesizer then writes a
structured synthesis citing those answers.

**The limitation**: the planner can only ask math questions about numbers
that are *already in the producer broadcast*. It can't go look at the raw
price history, the news payload, options chain, fundamentals, or anything
else. It rewrites what it's handed.

The vision is **multi-hop investigation via tool calls**:

```
producer broadcasts → planner can call tools mid-thought
                       ├─ tool: get_price_history(ticker, days)
                       ├─ tool: get_news_for(ticker, window)
                       ├─ tool: get_fundamentals(ticker)
                       ├─ tool: get_correlated_tickers(ticker, threshold)
                       ├─ tool: ask_math_host(question)  (existing)
                       └─ tool: get_options_flow(ticker)  (later)
                    → tool results loop back into planner's context
                    → planner can chain calls; multiple hops per scenario
                    → synthesizer writes final synthesis citing tool calls
```

The agent investigates rather than rewrites. It pulls evidence it needs.

---

## Why this matters for performance

v4's headline finding (if the v4 thesis holds) will be that ECO's
multi-scale signal is finally getting through to the apex via horizon
forecasts. But the *quality* of those forecasts is still limited by what
the herbivore tier sees. The herbivore reads producer broadcasts —
summaries of summaries. The interrogator's planner asks math questions —
but only about numbers already extracted by producers.

**The real raw data — full price tapes, options flow, news bodies — is
never directly inspected** by the apex's reasoning chain. The producer
tier is a lossy compression layer.

Tool-use lets the herbivore tier *decompress on demand* when its current
state suggests a quantitative question that the available producer
summary can't answer.

Expected signal lift: **probably significant on event-driven days** (earnings,
guidance changes, news shocks) where the producer's summary loses
critical info; **probably small on quiet days**. So the alpha contribution
is uneven, but the days where it helps are exactly the days that matter.

---

## What a v5 mission would need to specify

When this is unparked, the mission needs:

1. **Tool registry design**: a structured way to declare tool name,
   signature, return shape, and cost (how many extra tokens/calls).
2. **Tool implementations** for at least 3 tools (price history, news,
   correlated tickers). Each must be deterministic and fast (< 100ms
   per call) — the apex runs hundreds of times per sweep day.
3. **Planner prompt redesign**: the planner needs to know what tools
   exist, when to call them, and when to STOP calling them. A budget
   (e.g., max 5 tool calls per scenario) is essential to bound runtime.
4. **Logging**: every tool call must be logged with arguments and result
   summary. This becomes training data for the decomposer.
5. **A measurement plan**: how do we know tool-use helped? Likely an A/B:
   `v5-with-tools` vs `v5-no-tools` on the same 66-day window.
6. **Cost analysis**: tool calls multiply the per-day token budget.
   Estimate latency and Bedrock cost before committing.

---

## Prior art in the codebase

- `trophic/agents/interrogator_herbivore.py` — current 2-agent ask→solve→synthesize.
- `trophic/math_host.py` — the math solver host (model: Qwen2.5-Math-1.5B).
- `trophic/cross_model_channel.py` — the cross-attention channel between
  the planner and synthesizer; tool calls would slot into this same
  pattern.
- `trophic/training/interrogator_lora.py` — there's already a LoRA
  training loop for the interrogator. If we add tools, this training
  loop is the natural place to teach the planner *when* to call them.

---

## Acceptance criteria (when unparked)

These are placeholders for the future mission spec, not for current work:

- [ ] Tool registry exists and is unit-tested.
- [ ] ≥ 3 tool implementations land with deterministic returns.
- [ ] Planner prompt teaches tool selection + budget compliance.
- [ ] A/B comparison vs v4 (same 66-day window, interrogator tools on/off)
      with headline lift measured.
- [ ] No regression in v4-style headline metrics.

---

## STATUS line

PARKED. No work. The file lives in `tasks_v4/` because it's the natural
follow-on, but it's intentionally not in any active phase. When unparked,
move it to `tasks_v5/` (or wherever the next round lives) and update its
STATUS to PENDING.
