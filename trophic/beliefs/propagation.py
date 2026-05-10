"""Belief propagation math.

Three operations:
  1. apply_decay  — decay current_p toward prior_p over Δt
  2. apply_activation — apply an event's BeliefActivation to a state belief
  3. propagate_outcome — compute outcome.p_up from contributing state beliefs
                          and their internal links

All math runs in log-odds space; probabilities are converted at boundaries.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

from .schema import (
    DECAY_HALF_LIFE_HOURS,
    MAGNITUDE_TO_LOG_ODDS,
    BeliefActivation,
    InternalLink,
    OutcomeBelief,
    StateBelief,
)


# ───── Probability ↔ log-odds ─────

def log_odds(p: float) -> float:
    """Convert probability to log-odds. Clamps to avoid ±inf at p ∈ {0, 1}."""
    p = min(0.9999, max(0.0001, p))
    return math.log(p / (1.0 - p))


def prob_from_log_odds(lo: float) -> float:
    """Convert log-odds back to probability."""
    return 1.0 / (1.0 + math.exp(-lo))


# ───── Decay (toward prior_p) ─────

def apply_decay(belief: StateBelief, now_ts: float) -> StateBelief:
    """Decay belief.current_p toward belief.prior_p based on elapsed time.

    Math (in log-odds space, the deviation from prior decays toward 0):
        delta_log_odds(t) = delta_log_odds(t-Δt) × 0.5^(Δt / half_life)
        log_odds(t) = log_odds(prior_p) + delta_log_odds(t)

    Polymarket-bound beliefs do NOT decay — their current_p is read-through
    from the market price.
    """
    if belief.is_polymarket_bound:
        return belief
    if belief.last_updated <= 0.0:
        # Never been activated; nothing to decay.
        return belief

    elapsed_hours = max(0.0, (now_ts - belief.last_updated) / 3600.0)
    half_life = DECAY_HALF_LIFE_HOURS.get(belief.decay_class, 1.0)
    if half_life <= 0.0:
        # "instant" decay — snap to prior immediately.
        belief.current_p = belief.prior_p
        belief.last_updated = now_ts
        return belief
    if elapsed_hours <= 0.0:
        return belief

    lo_prior = log_odds(belief.prior_p)
    lo_current = log_odds(belief.current_p)
    delta = lo_current - lo_prior
    decay_factor = 0.5 ** (elapsed_hours / half_life)
    new_delta = delta * decay_factor
    belief.current_p = prob_from_log_odds(lo_prior + new_delta)
    belief.last_updated = now_ts
    return belief


# ───── Activation (one event updates one belief) ─────

def apply_activation(
    belief: StateBelief,
    activation: BeliefActivation,
    event_id: str,
    now_ts: float,
) -> StateBelief:
    """Apply a BeliefActivation to a state belief.

    Sequence:
      1. Decay belief to now_ts (so we apply the new evidence on top of
         already-aged credence).
      2. Compute log-odds shift: magnitude × direction × confidence.
      3. Add to current log-odds, clamp via probability conversion.
      4. Update decay_class to whatever the activation specified (LLM picks
         appropriate timescale per evidence type).
      5. Append event_id to evidence_log.

    Polymarket-bound beliefs ignore activations — Polymarket is the authority.
    """
    if belief.is_polymarket_bound:
        return belief

    # Step 1: age the belief to now
    belief = apply_decay(belief, now_ts)

    # Step 2: compute log-odds shift
    base_shift = MAGNITUDE_TO_LOG_ODDS.get(activation.magnitude, 1.0)
    sign = +1.0 if activation.direction_of_effect == "increases" else -1.0
    confidence_weight = activation.effective_confidence
    log_odds_delta = sign * base_shift * confidence_weight

    # Step 3: apply
    new_log_odds = log_odds(belief.current_p) + log_odds_delta
    belief.current_p = prob_from_log_odds(new_log_odds)

    # Step 4: update decay class to match activation's chosen timescale
    belief.decay_class = activation.decay_class

    # Step 5: log
    belief.last_updated = now_ts
    if event_id and event_id not in belief.evidence_log:
        belief.evidence_log.append(event_id)

    return belief


# ───── Outcome propagation ─────

def propagate_outcome(
    outcome_prior_p: float,
    state_beliefs: dict[str, StateBelief],
    links: Iterable[InternalLink],
    scaling_factor: float = 1.0,
) -> tuple[float, list[str], list[str]]:
    """Compute outcome.p_up from contributing state beliefs and their links.

    Single-step propagation into a single outcome node. For full network
    propagation across multi-step chains, use `propagate_network`.

    Math (log-odds-delta propagation):
        log_odds(outcome) = log_odds(prior)
                          + sum_i [
                                link_i.strength_posterior
                                × sign(link_i.direction)
                                × (log_odds(current_p(B_i)) − log_odds(prior_p(B_i)))
                                × scaling_factor
                            ]

    Each link contributes a shift proportional to (a) the link's reliability
    [strength_posterior], (b) the parent's deviation from prior in LOG-ODDS
    space (so confident beliefs propagate proportional to their certainty,
    not their probability-space distance), and (c) the direction.

    Why log-odds-delta and not probability-delta: probability-delta compresses
    signal near boundaries (a parent moving from 0.95 to 0.99 has small
    probability-delta but large log-odds-delta). For multi-hop chains, this
    means strong upstream evidence retains its punch through the chain.

    Returns (p_up, contributing_belief_ids, contributing_link_ids).
    """
    lo = log_odds(outcome_prior_p)
    contributing_beliefs: list[str] = []
    contributing_links: list[str] = []

    for link in links:
        b = state_beliefs.get(link.premise_belief_id)
        if b is None:
            continue
        lo_delta = log_odds(b.current_p) - log_odds(b.prior_p)
        if abs(lo_delta) < 1e-6:
            continue
        sign = +1.0 if link.direction == "positive" else -1.0
        shift = link.strength_posterior * sign * lo_delta * scaling_factor
        lo += shift
        contributing_beliefs.append(b.id)
        contributing_links.append(link.id)

    return prob_from_log_odds(lo), contributing_beliefs, contributing_links


# ───── Multi-step network propagation ─────

# Scope ordering enforces DAG-ness: macro/market < sector < company < outcome.
# Links can only go from a higher-or-equal scope to a lower-or-equal scope.
_SCOPE_ORDER: dict[str, int] = {"macro": 0, "market": 0, "sector": 1, "company": 2}
# Outcomes have an implicit scope_order of 3 (deepest leaf).


def _scope_order_of_node(node_id: str, state_beliefs: dict[str, "StateBelief"]) -> int:
    """Get topological order for a node id. State beliefs use their scope;
    outcomes (anything not in state_beliefs) get order 3 (leaf-ward)."""
    b = state_beliefs.get(node_id)
    if b is None:
        return 3  # outcome / unknown
    return _SCOPE_ORDER.get(b.scope, 2)


def link_would_create_cycle(
    new_premise_id: str,
    new_conclusion_id: str,
    existing_links: Iterable["InternalLink"],
) -> bool:
    """Check whether adding (premise → conclusion) would close a cycle.

    Walk forward from `conclusion` along existing edges; if we reach
    `premise`, the new link would create a cycle. O(V + E) DFS with a
    visited set so it terminates even when the existing graph has cycles.
    """
    if new_premise_id == new_conclusion_id:
        return True

    children: dict[str, list[str]] = {}
    for l in existing_links:
        children.setdefault(l.premise_belief_id, []).append(l.conclusion_belief_id)

    visited: set[str] = set()
    stack: list[str] = [new_conclusion_id]
    while stack:
        cur = stack.pop()
        if cur == new_premise_id:
            return True
        if cur in visited:
            continue
        visited.add(cur)
        stack.extend(children.get(cur, []))
    return False


def find_cycles(
    links: Iterable["InternalLink"],
    max_cycles: int = 50,
) -> list[list[str]]:
    """Return up to `max_cycles` simple cycles in the link graph.

    Useful for auditing existing stores. Returns each cycle as an ordered
    list of node ids.
    """
    children: dict[str, list[str]] = {}
    nodes: set[str] = set()
    for l in links:
        children.setdefault(l.premise_belief_id, []).append(l.conclusion_belief_id)
        nodes.add(l.premise_belief_id)
        nodes.add(l.conclusion_belief_id)

    cycles: list[list[str]] = []
    visited: set[str] = set()

    def dfs(node: str, path: list[str], on_path: set[str]) -> None:
        if len(cycles) >= max_cycles:
            return
        on_path.add(node)
        path.append(node)
        for c in children.get(node, []):
            if c in on_path:
                # Found a cycle from c back to itself
                idx = path.index(c)
                cycles.append(path[idx:] + [c])
                if len(cycles) >= max_cycles:
                    break
            elif c not in visited:
                dfs(c, path, on_path)
        on_path.remove(node)
        path.pop()
        visited.add(node)

    for n in list(nodes):
        if n not in visited and len(cycles) < max_cycles:
            dfs(n, [], set())
    return cycles


def topological_order(
    nodes: Iterable[str],
    links: Iterable["InternalLink"],
    state_beliefs: dict[str, "StateBelief"],
) -> list[str]:
    """Topological sort of node ids, respecting link parent→child edges.

    Falls back on scope_order when scopes alone disambiguate (which they do
    by design: macro before sector before company before outcome).
    """
    nodes = list(set(nodes))
    indeg: dict[str, int] = {n: 0 for n in nodes}
    children: dict[str, list[str]] = {n: [] for n in nodes}
    for link in links:
        if link.premise_belief_id in indeg and link.conclusion_belief_id in indeg:
            children[link.premise_belief_id].append(link.conclusion_belief_id)
            indeg[link.conclusion_belief_id] += 1

    # Kahn-style: queue nodes with indeg 0, sorted by scope_order for stability
    queue = sorted(
        [n for n, d in indeg.items() if d == 0],
        key=lambda n: _scope_order_of_node(n, state_beliefs),
    )
    out: list[str] = []
    while queue:
        n = queue.pop(0)
        out.append(n)
        for c in children[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)
        # Re-sort by scope each iteration (small cost; ensures stable order)
        queue.sort(key=lambda n: _scope_order_of_node(n, state_beliefs))

    if len(out) != len(nodes):
        raise ValueError(
            f"DAG validation failed: cycle detected. {len(nodes) - len(out)} "
            f"nodes unreachable in topological order."
        )
    return out


def propagate_network(
    state_beliefs: dict[str, "StateBelief"],
    outcome_beliefs: dict[str, "OutcomeBelief"],
    links: list["InternalLink"],
    max_iterations: int = 25,
    convergence_eps: float = 1e-4,
    scaling_factor: float = 1.0,
    require_dag: bool = False,
    clamped_ids: Optional[set[str]] = None,
) -> dict[str, list[tuple[str, str]]]:
    """Forward-propagate evidence through the full belief network.

    Cycle-tolerant by default: uses synchronous fixed-point iteration
    (Jacobi-style) — each pass reads from the previous iteration's snapshot
    and writes a fresh snapshot, then atomically swaps. This converges even
    when the link graph contains cycles (the consolidator can introduce them
    via merges; the decomposer occasionally too).

    Set `require_dag=True` for the old behavior: enforce topological order
    and raise on cycles.

    Each non-source state belief's recomputed value =
        log_odds(prior_p) + sum over incoming links of:
            link.strength_posterior
            × sign(direction)
            × (log_odds(parent_prev.current_p) − log_odds(parent.prior_p))
            × scaling_factor

    Log-odds-delta is used (not probability-delta) so signal magnitude
    survives multi-hop chains.

    Source state beliefs (no incoming links) are left at whatever value the
    event activations + decay have set. Their evidence is the OBSERVATION
    layer; everything downstream is inference.

    Mutates state_beliefs and outcome_beliefs in place. Returns a contribution
    map: {node_id: [(parent_id, link_id), ...]} from the final iteration.
    """
    # Index links by conclusion (target) for fast lookup
    incoming: dict[str, list["InternalLink"]] = {}
    for l in links:
        incoming.setdefault(l.conclusion_belief_id, []).append(l)

    contributions: dict[str, list[tuple[str, str]]] = {}
    clamped: set[str] = clamped_ids or set()

    if require_dag:
        all_ids = list(state_beliefs.keys()) + list(outcome_beliefs.keys())
        topological_order(all_ids, links, state_beliefs)  # raises on cycle

    # Snapshot of parent values at the start of each iteration (Jacobi
    # update). Cycles converge to a fixed point under bounded shifts because
    # log-odds-delta dampens with strength_posterior < 1.
    for it in range(max_iterations):
        prev_state_p: dict[str, float] = {
            bid: b.current_p for bid, b in state_beliefs.items()
        }
        max_change = 0.0

        # Recompute state beliefs from previous-iteration parent snapshot
        for nid, belief in state_beliefs.items():
            in_links = incoming.get(nid, [])
            if not in_links:
                continue  # source/leaf: event-driven, leave as-is
            if nid in clamped:
                continue  # this pass clamped this belief; do not overwrite

            lo = log_odds(belief.prior_p)
            local_contribs: list[tuple[str, str]] = []
            for link in in_links:
                parent_id = link.premise_belief_id
                parent = state_beliefs.get(parent_id)
                if parent is None:
                    continue
                parent_prev_p = prev_state_p[parent_id]
                lo_delta = log_odds(parent_prev_p) - log_odds(parent.prior_p)
                if abs(lo_delta) < 1e-6:
                    continue
                sign = +1.0 if link.direction == "positive" else -1.0
                shift = link.strength_posterior * sign * lo_delta * scaling_factor
                lo += shift
                local_contribs.append((parent_id, link.id))

            new_p = prob_from_log_odds(lo)
            contributions[nid] = local_contribs
            old = belief.current_p
            if abs(new_p - old) > convergence_eps:
                max_change = max(max_change, abs(new_p - old))
            belief.current_p = new_p

        # Recompute outcomes from the freshly-updated state beliefs
        for nid, outcome in outcome_beliefs.items():
            in_links = incoming.get(nid, [])
            if not in_links:
                continue
            lo = log_odds(0.5)
            local_contribs: list[tuple[str, str]] = []
            for link in in_links:
                parent = state_beliefs.get(link.premise_belief_id)
                if parent is None:
                    continue
                lo_delta = log_odds(parent.current_p) - log_odds(parent.prior_p)
                if abs(lo_delta) < 1e-6:
                    continue
                sign = +1.0 if link.direction == "positive" else -1.0
                shift = link.strength_posterior * sign * lo_delta * scaling_factor
                lo += shift
                local_contribs.append((parent.id, link.id))

            new_p = prob_from_log_odds(lo)
            contributions[nid] = local_contribs
            old = outcome.p_up
            if abs(new_p - old) > convergence_eps:
                max_change = max(max_change, abs(new_p - old))
            outcome.p_up = new_p
            outcome.contributing_state_beliefs = [pid for pid, _ in local_contribs]
            outcome.contributing_links = [lid for _, lid in local_contribs]

        if max_change < convergence_eps:
            break

    return contributions
