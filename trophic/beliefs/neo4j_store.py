"""Neo4j-backed belief network store.

Same interface as JSONLBeliefStore. Uses MERGE on `id` so upserts are
idempotent and concurrent-safe.

**Namespace: every node gets the `:Ecology` label** in addition to its
type label. The shared DB hosts DebaterHub, Hexis, and Ecology data;
the `:Ecology` label is the namespace fence. Other systems ignore
`:Ecology` nodes; our queries always filter on it.

Schema (Cypher):
  (:Ecology:StateBelief {id, statement_template, scope, prior_p, current_p,
                         decay_class, context_json, last_updated,
                         evidence_log, polymarket_market_id})
  (:Ecology:OutcomeBelief {id, ticker, horizon_min, statement, p_up,
                           contributing_state_beliefs, contributing_links,
                           activated_at, resolved, actual_direction})
  (:Ecology:InternalLink {id, premise_belief_id, conclusion_belief_id, scope,
                          direction, strength_prior, strength_posterior,
                          n_validations, n_correct, citation,
                          validation_status, created_at, updated_at})
  (:Ecology:TrophicEvent {id, timestamp, source_kind, raw_content, ticker, sector})

  (:Ecology:StateBelief)-[:LINKS_TO {link_id, strength, direction, scope}]
                       ->(:Ecology:StateBelief|:Ecology:OutcomeBelief)
  (:Ecology:TrophicEvent)-[:ACTIVATED {magnitude, decay_class,
                                       self_rated_confidence, magnitude_logit,
                                       reasoning, species_id,
                                       direction_of_effect}]
                        ->(:Ecology:StateBelief)
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Optional

from neo4j import Driver, GraphDatabase

from .schema import (
    BeliefActivation,
    Event,
    InternalLink,
    OutcomeBelief,
    Scope,
    StateBelief,
)
from .store import BeliefStore


def _driver_from_env() -> Driver:
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    pw = os.environ.get("NEO4J_PASSWORD")
    if not (uri and user and pw):
        raise RuntimeError(
            "NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD must be set in env"
        )
    return GraphDatabase.driver(uri, auth=(user, pw))


# Constraint setup — runs once on first connection per database.
# Constraint named with `ecology_` prefix to avoid clashing with other
# systems' constraints in the shared DB.
_CONSTRAINTS = [
    "CREATE CONSTRAINT ecology_state_belief_id IF NOT EXISTS "
    "FOR (n:StateBelief) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT ecology_outcome_belief_id IF NOT EXISTS "
    "FOR (n:OutcomeBelief) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT ecology_internal_link_id IF NOT EXISTS "
    "FOR (n:InternalLink) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT ecology_event_id IF NOT EXISTS "
    "FOR (n:TrophicEvent) REQUIRE n.id IS UNIQUE",
]


class Neo4jBeliefStore(BeliefStore):
    """Neo4j-backed belief store. Shares the DB with DebaterHub and Hexis;
    every node carries the `:Ecology` label in addition to its type label
    so it's namespace-filterable.
    """

    NAMESPACE_LABEL = "Ecology"

    def __init__(self, driver: Optional[Driver] = None):
        self.driver = driver or _driver_from_env()
        self._ensure_constraints()

    def close(self) -> None:
        self.driver.close()

    def _ensure_constraints(self) -> None:
        with self.driver.session() as sess:
            for c in _CONSTRAINTS:
                sess.run(c)

    # ── Internal serialization helpers ──

    @staticmethod
    def _state_belief_props(b: StateBelief) -> dict:
        d = asdict(b)
        d["context_json"] = json.dumps(d.pop("context"))
        return d

    @staticmethod
    def _state_belief_from_node(node) -> StateBelief:
        d = dict(node)
        ctx_json = d.pop("context_json", "{}")
        d["context"] = json.loads(ctx_json) if ctx_json else {}
        d["evidence_log"] = list(d.get("evidence_log") or [])
        return StateBelief(**d)

    @staticmethod
    def _link_props(link: InternalLink) -> dict:
        return asdict(link)

    @staticmethod
    def _link_from_node(node) -> InternalLink:
        return InternalLink(**dict(node))

    @staticmethod
    def _outcome_props(o: OutcomeBelief) -> dict:
        d = asdict(o)
        d["contributing_state_beliefs"] = list(d.get("contributing_state_beliefs") or [])
        d["contributing_links"] = list(d.get("contributing_links") or [])
        return d

    @staticmethod
    def _event_props(e: Event) -> dict:
        return {
            "id": e.id,
            "timestamp": e.timestamp,
            "source_kind": e.source,
            "raw_content": e.raw_content,
            "ticker": e.ticker,
            "sector": e.sector,
        }

    @staticmethod
    def _activation_props(a: BeliefActivation) -> dict:
        d = asdict(a)
        return d

    # ── Links ──

    def upsert_link_if_acyclic(
        self,
        link: InternalLink,
        existing_links: Optional[list[InternalLink]] = None,
    ) -> bool:
        """Insert `link` only if it doesn't close a cycle in the link graph.

        Pass `existing_links` (already-loaded snapshot) to avoid hitting Neo4j
        for every check during a batch insert. Returns True if inserted,
        False if refused.
        """
        from .propagation import link_would_create_cycle
        if existing_links is None:
            existing_links = self.all_links()
        if link_would_create_cycle(
            link.premise_belief_id, link.conclusion_belief_id, existing_links,
        ):
            return False
        self.upsert_link(link)
        return True

    def upsert_link(self, link: InternalLink) -> None:
        props = self._link_props(link)
        with self.driver.session() as sess:
            sess.run(
                "MERGE (l:Ecology:InternalLink {id: $id}) SET l += $props",
                id=link.id, props=props,
            )
            # Edge from premise → conclusion. Conclusion may be a StateBelief
            # or an OutcomeBelief — both have the :Ecology label.
            sess.run(
                """
                MATCH (premise:Ecology:StateBelief {id: $premise_id})
                OPTIONAL MATCH (conclusion:Ecology {id: $conclusion_id})
                  WHERE conclusion:StateBelief OR conclusion:OutcomeBelief
                FOREACH (_ IN CASE WHEN conclusion IS NULL THEN [] ELSE [1] END |
                  MERGE (premise)-[r:LINKS_TO {link_id: $link_id}]->(conclusion)
                  SET r.strength = $strength,
                      r.direction = $direction,
                      r.scope = $scope
                )
                """,
                premise_id=link.premise_belief_id,
                conclusion_id=link.conclusion_belief_id,
                link_id=link.id,
                strength=link.strength_posterior,
                direction=link.direction,
                scope=link.scope,
            )

    def get_link(self, link_id: str) -> Optional[InternalLink]:
        with self.driver.session() as sess:
            row = sess.run(
                "MATCH (l:Ecology:InternalLink {id: $id}) RETURN l",
                id=link_id,
            ).single()
            return self._link_from_node(row["l"]) if row else None

    def all_links(self) -> list[InternalLink]:
        with self.driver.session() as sess:
            rows = sess.run("MATCH (l:Ecology:InternalLink) RETURN l")
            return [self._link_from_node(r["l"]) for r in rows]

    def delete_link(self, link_id: str) -> bool:
        """Delete an InternalLink node and its LINKS_TO edge. Returns True if deleted."""
        with self.driver.session() as sess:
            result = sess.run(
                """
                MATCH (l:Ecology:InternalLink {id: $id})
                OPTIONAL MATCH ()-[r:LINKS_TO {link_id: $id}]-()
                DELETE r
                WITH l
                DETACH DELETE l
                RETURN count(l) AS removed
                """,
                id=link_id,
            ).single()
            return bool(result and result["removed"])

    def links_into(self, conclusion_belief_id: str) -> list[InternalLink]:
        with self.driver.session() as sess:
            rows = sess.run(
                "MATCH (l:Ecology:InternalLink {conclusion_belief_id: $cid}) RETURN l",
                cid=conclusion_belief_id,
            )
            return [self._link_from_node(r["l"]) for r in rows]

    # ── State beliefs ──

    def upsert_state_belief(self, belief: StateBelief) -> None:
        props = self._state_belief_props(belief)
        with self.driver.session() as sess:
            sess.run(
                "MERGE (b:Ecology:StateBelief {id: $id}) SET b += $props",
                id=belief.id, props=props,
            )

    def get_state_belief(self, belief_id: str) -> Optional[StateBelief]:
        with self.driver.session() as sess:
            row = sess.run(
                "MATCH (b:Ecology:StateBelief {id: $id}) RETURN b",
                id=belief_id,
            ).single()
            return self._state_belief_from_node(row["b"]) if row else None

    def all_state_beliefs(self) -> list[StateBelief]:
        with self.driver.session() as sess:
            rows = sess.run("MATCH (b:Ecology:StateBelief) RETURN b")
            return [self._state_belief_from_node(r["b"]) for r in rows]

    def state_beliefs_in_scope(self, scope: Scope) -> list[StateBelief]:
        with self.driver.session() as sess:
            rows = sess.run(
                "MATCH (b:Ecology:StateBelief {scope: $scope}) RETURN b",
                scope=scope,
            )
            return [self._state_belief_from_node(r["b"]) for r in rows]

    # ── Outcomes ──

    def append_outcome_belief(self, outcome: OutcomeBelief) -> None:
        props = self._outcome_props(outcome)
        with self.driver.session() as sess:
            sess.run(
                "MERGE (o:Ecology:OutcomeBelief {id: $id}) SET o += $props",
                id=outcome.id, props=props,
            )

    # ── Events ──

    def append_event(self, event: Event) -> None:
        props = self._event_props(event)
        with self.driver.session() as sess:
            sess.run(
                "MERGE (e:Ecology:TrophicEvent {id: $id}) SET e += $props",
                id=event.id, props=props,
            )
            for act in event.activations:
                act_props = self._activation_props(act)
                sess.run(
                    """
                    MATCH (e:Ecology:TrophicEvent {id: $event_id})
                    MATCH (b:Ecology:StateBelief {id: $target})
                    MERGE (e)-[r:ACTIVATED {species_id: $species_id}]->(b)
                    SET r += $props
                    """,
                    event_id=event.id,
                    target=act.target_belief_id,
                    species_id=act.species_id,
                    props=act_props,
                )
