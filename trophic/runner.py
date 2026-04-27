"""Async tick-driven runner — three-tier hidden-state pipeline.

Per tick:
  1. Pull N raw inputs from the synthetic feed.
  2. PRODUCERS forward-pass each attracted input → tier-0 broadcasts.
  3. HERBIVORES query the tier-0 pool (diet-filtered), each runs its
     Channels against the candidates. Null gate may fire → abstain.
     Else: forward Qwen with [role_prefix ⊕ channel_output ⊕ query] →
     tier-1 broadcast.
  4. PREDATORS query the tier-1 pool, same pattern → tier-2 broadcast.
  5. APEX JUDGE (Bedrock) scores each predator broadcast.
  6. Decomposer cascades the judgment back through 2 trophic edges.
  7. Population applies energy events; metrics recorded.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import os

import torch

from .agents.decomposer import Decomposer
from .agents.forecaster_herbivore import ForecasterHerbivore
from .agents.herbivore import Herbivore
from .agents.predator import Predator
from .agents.producer import Producer
from .agents.quant_producer import QuantitativeProducer
from .config import DEFAULT_CONFIG, TrophicConfig
from .decomposer import Decomposer as BiasDecomposer
from .ecology.metrics import MetricsCollector, TickMetrics
from .ecology.population import Population
from .inputs.synthetic import SyntheticFeed
from .model_host import ModelHost
from .predators.apex_judge import ApexJudge
from .substrate import SubstratePool
from .trough_attention import TroughAttention
from .types import Broadcast, PredatorJudgment


_DEBUG_DECOMPOSER = os.environ.get("TROPHIC_DEBUG_DECOMPOSER", "0") not in ("", "0", "false", "False")


@dataclass
class TickResult:
    tick: int
    substrate: list[Broadcast] = field(default_factory=list)
    herbivore_broadcasts: list[Broadcast] = field(default_factory=list)
    predator_broadcasts: list[Broadcast] = field(default_factory=list)
    judgments: dict[str, PredatorJudgment] = field(default_factory=dict)
    population_snapshot: dict = field(default_factory=dict)
    abstentions: dict = field(default_factory=dict)  # {agent_id: avg_null_prob}
    # Mission 02: per-(tier,kind) trough lifecycle stats from step_lifecycle()
    lifecycle_stats: list[dict] = field(default_factory=list)


@dataclass
class Runner:
    cfg: TrophicConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    pool: SubstratePool = field(default=None)  # type: ignore[assignment]
    feed: SyntheticFeed = field(default=None)  # type: ignore[assignment]
    population: Population = field(default=None)  # type: ignore[assignment]
    apex: ApexJudge = field(default=None)  # type: ignore[assignment]
    decomposer: Decomposer = field(default_factory=Decomposer)
    metrics: MetricsCollector = field(default_factory=MetricsCollector)
    host: ModelHost = field(default=None)  # type: ignore[assignment]

    # Lineage: maps a broadcast id back to its producing agent id, so the
    # decomposer can credit-assign across tiers.
    _broadcast_to_agent: dict[str, str] = field(default_factory=dict)

    # Mission 05 (phase2-D): one BiasDecomposer per trough (keyed by
    # (tier, kind)). Lazily allocated on first use so we honor the
    # trough's hidden_size / n_slots without having to know them up
    # front.
    _bias_decomposers: dict[tuple[str, str], BiasDecomposer] = field(default_factory=dict)
    # Diagnostic counter — bumped every time a decomposer bias is staged
    # via `set_pending_bias`. Read by --debug-decomposer in scripts.
    _bias_applied_count: int = 0
    _bias_max_magnitude: float = 0.0

    # Mission 06 (phase3-A): cross-attention softmax temperature for the
    # current tick. Set by `set_tau` (or directly) before each `step()`
    # call. Forwarded into every `attend()` call this tick (decomposer
    # side-effect attends, predator skip-connection attend).
    tau: float = 1.0

    def __post_init__(self):
        if self.pool is None:
            # SubstratePool is now an in-memory shim over TroughAttention
            # (Mission 01 of TROUGH_AS_TRANSFORMER). `db_path` is accepted
            # for backward-compat with config but ignored — there is no
            # SQLite layer anymore. Trough sizing is per-(tier,kind) and
            # set here so the runner has explicit control.
            self.pool = SubstratePool(
                db_path=self.cfg.pool.db_path,
                n_slots_per_kind=getattr(self.cfg.pool, "n_slots_per_kind", 256),
                trough_seed=self.cfg.runner.seed,
            )
        if self.feed is None:
            self.feed = SyntheticFeed(seed=self.cfg.runner.seed)
        if self.host is None:
            self.host = ModelHost.get(self.cfg.model)
        if self.population is None:
            self.population = Population(self.cfg.population)
            self._seed_agents()
        if self.apex is None:
            self.apex = ApexJudge.from_config(self.cfg.bedrock)

    def _seed_agents(self) -> None:
        # Qwen-text producers
        for kind in ("tickdelta", "disclosure", "anomaly"):
            for _ in range(self.cfg.runner.n_producers_per_kind):
                self.population.add_producer(Producer.make(kind))
        # Quantitative (numeric) producer for the Chronos path
        self.population.add_producer(QuantitativeProducer.make("quote_series"))
        # Qwen-text herbivores
        for kind in ("technical", "fundamental"):
            for _ in range(self.cfg.runner.n_herbivores_per_kind):
                self.population.add_herbivore(
                    Herbivore.make(kind, capacity=self.cfg.population.intake_budget)
                )
        # Forecaster (Chronos-backed) herbivore
        self.population.add_herbivore(ForecasterHerbivore.make("forecaster", capacity=4))
        # One predator kind for v1.5; eats from all three herbivore kinds.
        self.population.add_predator(Predator.make("short_horizon"))

    # ---------- skip-connection helper (Mission 06 / phase3-A) ----------

    def _first_producer_trough(self) -> TroughAttention | None:
        """Return the first producer-tier trough that has any alive slots,
        or None. Producer broadcasts live under tier="substrate" in the
        SubstratePool shim, with kind == producer agent_kind. This is
        the FIRST runtime call to a producer trough's attend() (per
        phase2-C's note); if no broadcasts have been deposited yet there
        is nothing to skip-attend over.
        """
        for (tier, _kind), trough in self.pool._troughs.items():
            if tier != "substrate":
                continue
            if int(trough.alive.sum().item()) > 0:
                return trough
        return None

    def set_tau(self, tau: float) -> None:
        """Set the softmax temperature for the next tick's attend() calls.

        Mission 06 (phase3-A): callers (the SFT/IPO/GRPO trainer scripts
        that drive Runner directly) compute the schedule and update this
        before each `step()` so the predator skip-attend and the
        decomposer side-effect attends share the same τ this tick.
        """
        self.tau = float(tau)

    # ---------- bias-decomposer helpers (Mission 05 / phase2-D) ----------

    def _get_bias_decomposer(self, trough: TroughAttention, key: tuple[str, str]) -> BiasDecomposer:
        """Lazily allocate a `BiasDecomposer` matched to this trough's dims."""
        d = self._bias_decomposers.get(key)
        if d is None:
            d = BiasDecomposer(
                hidden_size=trough.hidden_size,
                n_slots=trough.n_slots,
            )
            self._bias_decomposers[key] = d
        return d

    def _apply_decomposer_bias(
        self,
        pred_br: Broadcast,
        judgment: PredatorJudgment,
        tick: int,
    ) -> None:
        """Run the bias-decomposer for one judged predator broadcast and
        stage the resulting bias on the predator's tier-1 trough.

        Steps:
          1. Look up the predator's tier-1 trough (where its prey live).
             If none exists yet (no broadcasts of that kind deposited),
             skip — there's nothing to bias.
          2. Build a "judgment embedding" from the broadcast's decoded
             text via `host.encode_role_prefix`, mean-pooled. This is
             explicitly a placeholder per the mission file; future work
             can swap in the apex's real hidden state.
          3. Build a `slot_lineage` vector of shape `[n_slots]` by doing
             a side-effect `attend()` on the trough using the broadcast's
             channel embedding as the query. The resulting
             `per_slot_attention` is the lineage.
          4. Forward the decomposer; clamp to a sensible range; stage
             via `trough.set_pending_bias`.

        On any error, we log and continue — the decomposer is opt-in
        feedback; failures must not halt the runner.
        """
        if pred_br.abstained or not pred_br.channel_embedding:
            return
        # The predator's prey live in the (herbivore_broadcast, herb_kind)
        # troughs — but the predator queries them all via Channels. We
        # bias the trough that holds the predator's *own* output (tier
        # "predator_broadcast", kind = predator agent_kind). That's the
        # trough whose slot membership is most directly affected by what
        # the apex thought of THIS broadcast.
        kind_key = pred_br.agent_kind or pred_br.tier
        tier = pred_br.tier
        trough = self.pool.get_trough(tier, kind_key)
        if trough is None:
            return

        decomposer = self._get_bias_decomposer(trough, (tier, kind_key))

        try:
            # 1. Slot-lineage from a side-effect attend on the broadcast
            #    embedding. We use no_grad — this is a diagnostic readout,
            #    not a training pass.
            query = torch.tensor(pred_br.channel_embedding, dtype=torch.float32)
            with torch.no_grad():
                attend_out = trough.attend(query, tau=self.tau)
            lineage = attend_out.per_slot_attention.detach().clone()

            # 2. Judgment embedding — placeholder per mission spec:
            #    encode_role_prefix(decoded_text) → mean-pool.
            text = pred_br.decoded_text or "[no text]"
            try:
                role_seq = self.host.encode_role_prefix(text)  # [seq, hidden]
            except Exception:
                # Defensive: ModelHost may not be initialised in some
                # test configs. Use a zero embedding so the decomposer
                # path still exercises (with zero-init head it produces
                # zero bias regardless).
                role_seq = torch.zeros(1, trough.hidden_size, dtype=torch.float32)
            judgment_emb = role_seq.float().mean(dim=0).detach()
            if judgment_emb.shape[-1] != trough.hidden_size:
                # Skip if dims don't match (e.g. mismatched model host)
                return

            # 3. Decomposer forward.
            with torch.no_grad():
                bias = decomposer(judgment_emb, lineage)

            # 4. Stage on the trough for the next attend() call.
            trough.set_pending_bias(bias)
            self._bias_applied_count += 1
            mag = float(bias.abs().max().item())
            if mag > self._bias_max_magnitude:
                self._bias_max_magnitude = mag
            if _DEBUG_DECOMPOSER:
                print(
                    f"[tick {tick}] bias-applied trough({tier},{kind_key}) "
                    f"score={judgment.score:.3f} max|bias|={mag:.4f}"
                )
        except Exception as e:
            if _DEBUG_DECOMPOSER:
                print(f"[tick {tick}] decomposer-bias skipped ({e!r})")
            return

    # ---------- one tick ----------

    async def step(self, tick: int) -> TickResult:
        result = TickResult(tick=tick)
        inputs = self.feed.emit(self.cfg.runner.inputs_per_tick)

        # ---- 1. Producers ----
        async def _produce_for(p: Producer):
            made = []
            for inp in inputs:
                if not p.attracts(inp):
                    continue
                br = await p.produce(inp, tick, self.host)
                if br is not None:
                    made.append(br)
            return p, made

        produced = await asyncio.gather(
            *[_produce_for(p) for p in self.population.alive_producers()]
        )
        for prod, items in produced:
            for it in items:
                self.pool.add_broadcast(it)
                self._broadcast_to_agent[it.id] = prod.id
                result.substrate.append(it)
            self.population.credit_production(prod.id, len(items))

        # ---- 2. Herbivores hunt the tier-0 pool ----
        herbivore_results = []
        for h in self.population.alive_herbivores():
            candidates = self.pool.query_pool(
                tier="substrate",
                diet_tags=h.diet_tags,
                limit=self.cfg.pool.retrieval_k,
            )
            br, claimed, rejected, stats = await h.hunt_and_synthesize(
                candidates, tick, self.host
            )
            # Atomically claim what the Channel selected.
            if claimed:
                self.pool.claim([c.id for c in claimed], h.id, tick)
            self.pool.add_broadcast(br)
            self._broadcast_to_agent[br.id] = h.id
            result.herbivore_broadcasts.append(br)
            herbivore_results.append((h, claimed, rejected, br, stats))
            result.abstentions[h.id] = stats["avg_null_prob"]

            if not br.abstained:
                fill_ratio = len(claimed) / max(h.capacity, 1)
                self.population.credit_intake(h.id, fill_ratio, len(claimed))
                for c in claimed:
                    pid = self._broadcast_to_agent.get(c.id)
                    if pid:
                        self.population.credit_eaten(pid, 1)

        # ---- 3. Predators hunt the tier-1 pool ----
        predator_results = []
        for pred in self.population.alive_predators():
            candidates = self.pool.query_pool(
                tier="herbivore_broadcast",
                diet_tags=pred.diet_tags,
                limit=self.cfg.pool.retrieval_k,
            )
            # Mission 06 (phase3-A): cross-tier skip — pull a producer
            # trough so the predator can attend directly across the
            # herbivore tier as a residual path. We pick the first
            # populated producer trough (tier="substrate"); if multiple
            # producer kinds exist, the multi-head attention will route
            # within the trough rather than across kinds.
            producer_trough = self._first_producer_trough()
            br, claimed, rejected, stats = await pred.hunt_and_predict(
                candidates, tick, self.host,
                producer_trough=producer_trough,
                tau=self.tau,
            )
            if claimed:
                self.pool.claim([c.id for c in claimed], pred.id, tick)
            self.pool.add_broadcast(br)
            self._broadcast_to_agent[br.id] = pred.id
            result.predator_broadcasts.append(br)
            predator_results.append((pred, claimed, rejected, br, stats))
            result.abstentions[pred.id] = stats["avg_null_prob"]

            if not br.abstained:
                fill_ratio = len(claimed) / max(pred.capacity, 1)
                self.population.credit_intake(pred.id, fill_ratio, len(claimed))
                for c in claimed:
                    hid = self._broadcast_to_agent.get(c.id)
                    if hid:
                        self.population.credit_eaten(hid, 1)

        # ---- 4. Apex judges predator broadcasts ----
        judgments = await self.apex.judge_batch(result.predator_broadcasts, tick)
        for j in judgments:
            self.pool.add_judgment(j)
        judgments_by = {j.synthesis_id: j for j in judgments}
        result.judgments = judgments_by

        # Per-judgment predator reward.
        for br in result.predator_broadcasts:
            j = judgments_by.get(br.id)
            if j is not None:
                aid = self._broadcast_to_agent.get(br.id)
                if aid:
                    self.population.credit_judgment(aid, j.score)

        # ---- 5. Cascade credit back through 2 trophic edges ----
        # For each judged predator broadcast, the upstream herbivores it ate
        # share the credit, and through them, the producers those herbivores
        # ate also share.
        for pred, claimed, rejected, pred_br, _stats in predator_results:
            j = judgments_by.get(pred_br.id)
            if j is None:
                continue
            score = j.score
            per_item = (score - 0.5) * 2.0
            # Tier 1 attribution: herbivores fed by the predator
            for herb_br in claimed:
                hid = self._broadcast_to_agent.get(herb_br.id)
                if hid:
                    self.population.apply_decomposer(
                        producer_energy={},
                        producer_reputation={},
                        herbivore_energy={hid: per_item * 0.05},
                    )
                # Tier 0 attribution: producers fed by THAT herbivore
                #   herb_br.parent_input_ids = ids of producer broadcasts the
                #   herbivore ate.
                for prod_br_id in herb_br.parent_input_ids:
                    pid = self._broadcast_to_agent.get(prod_br_id)
                    if pid:
                        self.population.apply_decomposer(
                            producer_energy={pid: per_item * 0.02},
                            producer_reputation={pid: per_item * 0.01},
                            herbivore_energy={},
                        )

        # ---- 5c. Decomposer bias (Mission 05 — phase2-D) ----
        # For each judged predator broadcast, run the small bias-
        # decomposer and stage a per-slot bias on the predator's trough
        # that the NEXT tick's attend() will consume.
        for pred_br in result.predator_broadcasts:
            j = judgments_by.get(pred_br.id)
            if j is None:
                continue
            self._apply_decomposer_bias(pred_br, j, tick)

        # ---- 5b. Trough lifecycle (Mission 02 — slot decay/death) ----
        # After all attend() calls for the tick, fold the EMA forward
        # and kill any slot that has been below epsilon for n_patience
        # ticks. Stats are logged per (tier, kind) trough.
        lifecycle_stats: list[dict] = []
        for (tier, kind), trough in self.pool._troughs.items():
            stats = trough.step_lifecycle()
            stats["tier"] = tier
            stats["kind"] = kind
            lifecycle_stats.append(stats)
            if stats["n_killed"] > 0:
                print(
                    f"[tick {tick}] trough({tier},{kind}): "
                    f"n_alive={stats['n_alive']} n_killed={stats['n_killed']} "
                    f"killed_ids={stats['killed_ids']}"
                )
            else:
                print(
                    f"[tick {tick}] trough({tier},{kind}): "
                    f"n_alive={stats['n_alive']} n_killed=0"
                )
        result.lifecycle_stats = lifecycle_stats

        # ---- 6. Existence cost + census ----
        self.population.tick_existence_costs(tick)
        snapshot = self.population.census(tick)
        result.population_snapshot = snapshot

        # ---- 7. Reap unclaimed substrate this tick (item nobody bid on) ----
        all_this_tick = {it.id for it in result.substrate}
        # Anything not claimed by any herbivore is wasted.
        eaten_ids = set()
        for _h, claimed, _r, _b, _s in herbivore_results:
            for c in claimed:
                eaten_ids.add(c.id)
        ignored = list(all_this_tick - eaten_ids)
        n_rotted = self.pool.mark_rotted(ignored) if ignored else 0

        # ---- Metrics ----
        n_eaten_total = len(eaten_ids) + sum(
            len(claimed) for _p, claimed, _r, _b, _s in predator_results
        )
        n_abst = sum(
            1 for br in result.herbivore_broadcasts + result.predator_broadcasts
            if br.abstained
        )
        avg_j = (sum(j.score for j in judgments) / len(judgments)) if judgments else 0.0
        m = TickMetrics(
            tick=tick,
            n_inputs=len(inputs),
            n_substrate_added=len(result.substrate),
            n_eaten=n_eaten_total,
            n_rotted=n_rotted,
            n_syntheses=len(result.herbivore_broadcasts) + len(result.predator_broadcasts),
            n_judgments=len(judgments),
            avg_judgment=round(avg_j, 4),
            avg_confidence=round(
                sum(_safe_conf(b.decoded_text) for b in result.predator_broadcasts) /
                max(1, len(result.predator_broadcasts)), 4
            ),
            population=snapshot,
        )
        self.metrics.record(m)
        self.pool.record_tick(
            tick_id=tick,
            n_broadcasts=len(result.substrate) + len(result.herbivore_broadcasts) + len(result.predator_broadcasts),
            n_eaten=n_eaten_total,
            n_rotted=n_rotted,
            n_judgments=len(judgments),
            n_abstentions=n_abst,
        )
        return result

    async def run(self, n_ticks: int | None = None) -> list[TickResult]:
        n = n_ticks or self.cfg.runner.max_ticks
        out = []
        for t in range(1, n + 1):
            out.append(await self.step(t))
        return out


def _safe_conf(text: str | None) -> float:
    if not text:
        return 0.0
    import re
    m = re.search(r"CONFIDENCE:\s*([0-9.]+)", text, re.IGNORECASE)
    if not m:
        return 0.0
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return 0.0
