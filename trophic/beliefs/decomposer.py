"""LLM-driven recursive decomposer.

For each premise belief in an internal link, ask Bedrock Sonnet:
  - Is this proposition directly observable from one of our data sources?
  - If yes → terminal leaf, stop recursing.
  - If no → list its component assumptions; for each, recurse.

Hard call budget. Stops at depth 5 even if leaves not reached.

Concurrency: siblings at the same depth are fired in parallel via
ThreadPoolExecutor. boto3 is blocking but each call holds the GIL only
briefly during JSON ser/deser; threads handle the network wait
efficiently. With max_workers=8 and ~5s per Bedrock call, throughput
goes from ~12 calls/min serial to ~80+ calls/min parallel.

Usage:
    decomposer = BeliefDecomposer(
        store=neo4j_store,
        max_depth=5,
        max_calls=100,
        max_workers=8,
    )
    decomposer.decompose_all_seed_premises(seed_links)
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from ..decomposers.opus_judge import call_opus_for_json
from .schema import (
    DirectionalLean,
    EvidenceLevel,
    InternalLink,
    Scope,
    StateBelief,
    prior_p_from_categorical,
)
from .store import BeliefStore


# Bedrock Sonnet for decomposition (faster + cheaper than Opus, sufficient
# for structured-output classification work).
DEFAULT_SONNET_MODEL = os.environ.get(
    "DECOMPOSER_SONNET_MODEL", "us.anthropic.claude-sonnet-4-6",
)

# Concrete data sources we can directly observe propositions against.
# When the LLM says a proposition is observable from ONE of these, we
# stop recursing.
OBSERVABLE_SOURCES = [
    "polygon_news",         # news headline / article
    "polygon_ohlcv",        # OHLCV bars and derived quant signals
    "earnings_release",     # 8-K, 10-K, 10-Q filings
    "sec_filing",           # other SEC filings (13D, 13F, S-1)
    "polymarket",           # prediction market price
    "fed_announcement",     # FOMC statement / minutes
    "macro_release",        # CPI, PCE, NFP, GDP
    "options_flow",         # implied vol, open interest, unusual activity
    "social_media",         # Twitter/X, Reddit (when ingested)
    "press_release",        # company-issued PR
    "court_filing",         # litigation records
    "regulatory_action",    # FTC, DOJ, EU, etc.
]


SYSTEM_PROMPT = """You are decomposing a financial-market belief into its causal preconditions.

Given a proposition (a state belief), do TWO things:

1. Decide whether the proposition is DIRECTLY OBSERVABLE from one of these data sources:
   {observable_sources}

   "Directly observable" means: a single data record (one news headline, one
   filing, one OHLCV bar, one Polymarket price) would let you classify the
   proposition true/false WITHOUT needing to first verify multiple other facts.

2. If the proposition IS directly observable, return:
   {{
     "terminal": true,
     "observable_source": "<one of the sources above>",
     "reasoning": "<one sentence on why a single record from this source proves the proposition>"
   }}

3. If the proposition is NOT directly observable (it depends on multiple sub-facts
   to be evaluated), return its component assumptions — the propositions that, if
   ALL true, would make this proposition true:

   {{
     "terminal": false,
     "reasoning": "<one sentence on why this proposition decomposes>",
     "assumptions": [
       {{
         "statement": "<a sub-proposition this depends on>",
         "scope": "macro" | "sector" | "company",
         "evidence_level": "novel" | "weak" | "moderate" | "strong" | "well_established",
         "directional_lean": "strongly_false" | "leans_false" | "neutral" | "leans_true" | "strongly_true",
         "direction_to_parent": "positive" | "negative",
         "strength": "weak" | "medium" | "strong" | "decisive"
       }},
       ... up to 4 assumptions
     ]
   }}

CRITICAL RULES:
- Output JSON ONLY. No prose preamble.
- Maximum 4 assumptions per decomposition. Pick the LOAD-BEARING ones; ignore minor caveats.
- Each assumption MUST be at the same or coarser scope than the parent (macro > sector > company).
- direction_to_parent: "positive" if assumption being TRUE makes parent more likely true.
- strength: how reliably the assumption being violated would invalidate the parent.
- evidence_level + directional_lean → looked up against PRIOR_P_TABLE to set base rate.
"""


@dataclass
class DecomposerStats:
    n_llm_calls: int = 0
    n_terminal_leaves: int = 0
    n_intermediate_beliefs: int = 0
    n_links_created: int = 0
    max_depth_reached: int = 0
    budget_exhausted: bool = False


@dataclass
class BeliefDecomposer:
    """Recursive decomposer with hard call budget. Thread-safe under
    ThreadPoolExecutor — siblings at the same depth fire concurrently."""

    store: BeliefStore
    max_depth: int = 5
    max_calls: int = 100
    max_workers: int = 8
    model: str = DEFAULT_SONNET_MODEL
    # In-memory cache: don't re-decompose the same belief id twice
    _decomposed: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    stats: DecomposerStats = field(default_factory=DecomposerStats)

    def decompose_link(self, link: InternalLink) -> None:
        """Decompose the premise belief of a seed internal link."""
        premise = self.store.get_state_belief(link.premise_belief_id)
        if premise is None:
            print(f"[decomposer] link {link.id}: premise {link.premise_belief_id!r} "
                  f"not yet in store; skipping (load YAML inventory first)",
                  flush=True)
            return
        self.decompose_all_seed_premises([link])

    def decompose_all_seed_premises(
        self, seed_links: list[InternalLink],
    ) -> DecomposerStats:
        """Walk every premise belief in the seed inventory.

        Worklist pattern (no recursive thread-pool waiting): a single
        ThreadPoolExecutor processes a queue of (belief, depth) tuples.
        Each completed task adds its children's tuples back to the queue.
        This avoids the classic recursive-pool deadlock where parents
        block on children that need pool slots that are held by the
        parents.

        Idempotent: skips premises that already have decomposition edges
        in the store. To force re-decomposition, delete children first.
        """
        # Seed the in-memory _decomposed cache from existing graph state
        existing_decomposed = self._already_decomposed_ids()
        with self._lock:
            self._decomposed |= existing_decomposed
        if existing_decomposed:
            print(f"[decomposer] resume: {len(existing_decomposed)} beliefs already "
                  f"decomposed in prior run; skipping those", flush=True)

        seen = set()
        roots: list[StateBelief] = []
        for link in seed_links:
            if link.premise_belief_id in seen:
                continue
            seen.add(link.premise_belief_id)
            premise = self.store.get_state_belief(link.premise_belief_id)
            if premise is None:
                print(f"[decomposer] seed premise {link.premise_belief_id!r} "
                      f"not in store; skipping", flush=True)
                continue
            roots.append(premise)

        # Worklist drained by a thread pool. queue of (belief, depth).
        from queue import Queue, Empty
        work: Queue = Queue()
        for r in roots:
            work.put((r, 0))

        # Sentinel to count in-flight tasks; pool exits when both queue
        # is empty AND no tasks are running.
        in_flight = [0]

        def _process_one(belief: StateBelief, depth: int) -> list[StateBelief]:
            """Process one belief: returns list of children to enqueue."""
            with self._lock:
                self.stats.max_depth_reached = max(
                    self.stats.max_depth_reached, depth,
                )
                if belief.id in self._decomposed:
                    return []
                if depth >= self.max_depth:
                    print(f"[decomposer] {belief.id} at max_depth {self.max_depth}", flush=True)
                    self._decomposed.add(belief.id)
                    return []
                if self.stats.n_llm_calls >= self.max_calls:
                    if not self.stats.budget_exhausted:
                        self.stats.budget_exhausted = True
                        print(f"[decomposer] CALL BUDGET EXHAUSTED "
                              f"({self.max_calls}); halting", flush=True)
                    return []
                self._decomposed.add(belief.id)
                self.stats.n_llm_calls += 1
                call_n = self.stats.n_llm_calls

            try:
                result = self._call_llm(belief, depth)
            except Exception as e:
                print(f"[decomposer] LLM raised on {belief.id}: {e}", flush=True)
                with self._lock:
                    self.stats.n_terminal_leaves += 1
                return []

            if result is None:
                print(f"[decomposer] LLM returned None on {belief.id}", flush=True)
                with self._lock:
                    self.stats.n_terminal_leaves += 1
                return []

            if result.get("terminal"):
                obs = result.get("observable_source", "unknown")
                print(f"[decomposer] TERMINAL d={depth} ({call_n}/{self.max_calls}) "
                      f"{belief.id} → {obs}", flush=True)
                with self._lock:
                    self.stats.n_terminal_leaves += 1
                return []

            assumptions = result.get("assumptions", []) or []
            if not assumptions:
                with self._lock:
                    self.stats.n_terminal_leaves += 1
                return []

            print(f"[decomposer] DECOMPOSE d={depth} ({call_n}/{self.max_calls}) "
                  f"{belief.id} → {len(assumptions)} children", flush=True)

            children: list[StateBelief] = []
            for i, a in enumerate(assumptions[:4]):
                child = self._make_child_belief(belief, a, depth, i)
                if child is None:
                    continue
                self.store.upsert_state_belief(child)
                child_link = self._make_child_link(child, belief, a)
                if child_link is not None:
                    self.store.upsert_link(child_link)
                with self._lock:
                    self.stats.n_intermediate_beliefs += 1
                    if child_link is not None:
                        self.stats.n_links_created += 1
                children.append(child)
            return children

        # Worker function: pops from queue, processes, pushes children
        def _worker():
            while True:
                try:
                    belief, depth = work.get(timeout=2.0)
                except Empty:
                    # If no tasks running and queue empty, exit
                    with self._lock:
                        if in_flight[0] == 0:
                            return
                    continue
                with self._lock:
                    in_flight[0] += 1
                try:
                    children = _process_one(belief, depth)
                    if not self.stats.budget_exhausted:
                        for c in children:
                            work.put((c, depth + 1))
                finally:
                    with self._lock:
                        in_flight[0] -= 1
                    work.task_done()

        # Spin up max_workers worker threads
        threads = [
            threading.Thread(target=_worker, daemon=True)
            for _ in range(self.max_workers)
        ]
        for t in threads:
            t.start()
        # Wait for all to finish (they exit when queue empty AND in_flight=0)
        for t in threads:
            t.join()
        return self.stats

    def _already_decomposed_ids(self) -> set[str]:
        """Belief ids that already have at least one decomposition-style
        edge (state-belief → state-belief). Skipped on resume."""
        out: set[str] = set()
        all_links = self.store.all_links()
        belief_ids = {b.id for b in self.store.all_state_beliefs()}
        for l in all_links:
            if l.premise_belief_id in belief_ids and l.conclusion_belief_id in belief_ids:
                out.add(l.conclusion_belief_id)
        return out

    def _call_llm(self, belief: StateBelief, depth: int) -> Optional[dict]:
        system = SYSTEM_PROMPT.format(
            observable_sources=", ".join(OBSERVABLE_SOURCES),
        )
        ctx_str = (
            f" (context: {json.dumps(belief.context)})"
            if belief.context else ""
        )
        user = (
            f"Proposition to decompose:\n\n"
            f'  "{belief.statement_template}"{ctx_str}\n\n'
            f"Scope: {belief.scope}\n"
            f"Current depth in tree: {depth}\n\n"
            f"Output JSON only."
        )
        return call_opus_for_json(
            system=system, user=user,
            model=self.model,
            max_tokens=2000,
            temperature=0.2,
        )

    def _make_child_belief(
        self, parent: StateBelief, assumption: dict, depth: int, idx: int,
    ) -> Optional[StateBelief]:
        try:
            statement = assumption["statement"]
            scope: Scope = assumption.get("scope", parent.scope)
            evidence_level: EvidenceLevel = assumption.get("evidence_level", "weak")
            directional_lean: DirectionalLean = assumption.get("directional_lean", "neutral")
        except KeyError as e:
            print(f"[decomposer] malformed assumption (missing {e}): {assumption}")
            return None

        # Stable id from parent + statement hash for idempotence on re-runs
        import hashlib
        statement_hash = hashlib.md5(statement.encode()).hexdigest()[:8]
        # Inherit parent's company/sector context unless scope is broader
        ctx = dict(parent.context)
        if scope == "macro":
            ctx = {}
        elif scope == "sector" and "ticker" in ctx:
            ctx.pop("ticker", None)

        cid = f"belief.{scope}.decomposed.{statement_hash}"
        return StateBelief.from_categorical(
            id=cid,
            statement_template=statement,
            scope=scope,
            evidence_level=evidence_level,
            directional_lean=directional_lean,
            decay_class="normal",
            context=ctx,
        )

    def _make_child_link(
        self, child: StateBelief, parent: StateBelief, assumption: dict,
    ) -> Optional[InternalLink]:
        from .schema import MAGNITUDE_TO_LOG_ODDS  # noqa
        direction = assumption.get("direction_to_parent", "positive")
        strength_label = assumption.get("strength", "medium")
        strength_to_prior = {
            "weak": 0.4, "medium": 0.6, "strong": 0.75, "decisive": 0.9,
        }
        s = strength_to_prior.get(strength_label, 0.6)
        return InternalLink(
            id=f"link.{child.id}→{parent.id}",
            premise_belief_id=child.id,
            conclusion_belief_id=parent.id,
            scope=child.scope,
            direction=direction,
            strength_prior=s,
            strength_posterior=s,
            citation=f"LLM-decomposed at depth via Sonnet ({assumption.get('strength', 'medium')})",
            validation_status="unverified",
            created_at=time.time(),
            updated_at=time.time(),
        )
