"""Consolidation pass for the belief network.

After recursive decomposition, multiple parents independently propose
their own versions of common assumptions ("market is not risk-off",
"implied vol skew didn't price the upside" etc.). Without consolidation
the graph fans out into a forest of redundant nodes.

Three-tier similarity check:
  1. Lexical fast-path (SequenceMatcher ratio > 0.85)
  2. Embedding cosine (Qwen3-Embedding-0.6B via Modal)
     - >= 0.92  → merge automatically
     - 0.75-0.92 → flag as borderline for LLM judge
  3. LLM judge (Bedrock Sonnet) on borderline pairs

Constraints:
  - Only consolidates within the same scope. Different scopes mean
    different context — never collapse.
  - Polymarket-bound beliefs are NEVER merged (they have a unique
    market_id; collapsing them loses the binding).
  - Decomposed beliefs are eligible by default; hand-seeded beliefs
    have priority as canonical (less likely to be merged away).

Merge semantics:
  - Pick canonical: prefer hand-seeded over decomposed; among same kind,
    prefer the one with more activations / further-from-0.5 prior.
  - Rewrite all incoming/outgoing links from non-canonical → canonical.
  - Union evidence_logs and contexts.
  - Delete the redundant belief.

Idempotent: re-running is a no-op once the graph is consolidated.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import combinations
from typing import Optional

from .embeddings import ModalEmbedder, cosine_sim
from .schema import StateBelief
from .store import BeliefStore

logger = logging.getLogger(__name__)


# Tunable thresholds
LEXICAL_RATIO_AUTO_MERGE = 0.85
EMBEDDING_COSINE_AUTO_MERGE = 0.92
EMBEDDING_COSINE_BORDERLINE = 0.75


@dataclass
class MergeCandidate:
    """One candidate pair for consolidation."""
    canonical_id: str
    redundant_id: str
    source: str  # "lexical" | "embedding" | "llm_confirmed"
    score: float
    reasoning: str = ""


@dataclass
class ConsolidationReport:
    n_total: int = 0
    n_pairs_evaluated: int = 0
    n_lexical_merges: int = 0
    n_embedding_merges: int = 0
    n_llm_borderline: int = 0
    n_llm_confirmed: int = 0
    n_llm_rejected: int = 0
    merges_applied: list[MergeCandidate] = field(default_factory=list)
    n_llm_calls: int = 0


def _pick_canonical(a: StateBelief, b: StateBelief) -> tuple[StateBelief, StateBelief]:
    """Return (canonical, redundant). Hand-seeded wins; otherwise more-evidenced
    or further-from-0.5 prior wins. Ties broken by lexicographic id."""
    a_decomposed = "decomposed" in a.id
    b_decomposed = "decomposed" in b.id
    if a_decomposed and not b_decomposed:
        return b, a
    if b_decomposed and not a_decomposed:
        return a, b
    a_evidence = len(a.evidence_log)
    b_evidence = len(b.evidence_log)
    if a_evidence != b_evidence:
        return (a, b) if a_evidence > b_evidence else (b, a)
    a_certainty = abs(a.prior_p - 0.5)
    b_certainty = abs(b.prior_p - 0.5)
    if abs(a_certainty - b_certainty) > 1e-9:
        return (a, b) if a_certainty > b_certainty else (b, a)
    return (a, b) if a.id < b.id else (b, a)


def _llm_pairwise_judge(
    a: StateBelief, b: StateBelief, model: Optional[str] = None,
) -> tuple[bool, str]:
    """Ask Sonnet whether two propositions are semantically equivalent.

    Returns (equivalent, reasoning). On any error, defaults to NOT
    equivalent (safer: avoids destroying a node we're unsure about).
    """
    from ..decomposers.opus_judge import call_opus_for_json
    import os as _os
    model = model or _os.environ.get(
        "DECOMPOSER_SONNET_MODEL", "us.anthropic.claude-sonnet-4-6",
    )
    system = """You judge whether two financial-market propositions are SEMANTICALLY EQUIVALENT.

Two propositions are equivalent if a single piece of evidence about the world
would update them both in the same direction by the same amount. They can be
phrased differently; what matters is the truth-conditions.

Return ONLY this JSON, nothing else:
{
  "equivalent": true | false,
  "reasoning": "<1-2 sentences on why>"
}

Be conservative: if there's a meaningful nuance distinguishing the two, return
false. Only return true if you'd genuinely treat them as the same node."""
    user = (
        f"Proposition A:\n  \"{a.statement_template}\"\n  scope: {a.scope}\n  context: {a.context}\n\n"
        f"Proposition B:\n  \"{b.statement_template}\"\n  scope: {b.scope}\n  context: {b.context}\n\n"
        f"Are these equivalent? JSON only."
    )
    try:
        result = call_opus_for_json(
            system=system, user=user,
            model=model, max_tokens=300, temperature=0.1,
        )
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return False, f"judge error: {e}"
    if not result:
        return False, "judge returned None"
    return bool(result.get("equivalent")), result.get("reasoning", "")


def _candidate_pairs(beliefs: list[StateBelief]) -> list[tuple[StateBelief, StateBelief]]:
    """Pairs eligible for consolidation: same scope, neither polymarket-bound."""
    out = []
    for a, b in combinations(beliefs, 2):
        if a.scope != b.scope:
            continue
        if a.is_polymarket_bound or b.is_polymarket_bound:
            continue
        out.append((a, b))
    return out


def _apply_merge(store: BeliefStore, canonical: StateBelief, redundant: StateBelief) -> None:
    """Rewrite all links pointing to/from `redundant` so they point to `canonical`,
    union evidence logs and contexts, then delete `redundant`."""
    # Rewrite links
    all_links = store.all_links()
    for link in all_links:
        changed = False
        if link.premise_belief_id == redundant.id:
            link.premise_belief_id = canonical.id
            changed = True
        if link.conclusion_belief_id == redundant.id:
            link.conclusion_belief_id = canonical.id
            changed = True
        if changed:
            store.upsert_link(link)

    # Union evidence_log and context
    canonical.evidence_log = list(set(canonical.evidence_log) | set(redundant.evidence_log))
    canonical.context = {**redundant.context, **canonical.context}  # canonical wins on conflict
    store.upsert_state_belief(canonical)

    # Delete redundant. The store ABC doesn't expose delete, so use the
    # backend-specific path. For JSONLBeliefStore: pop from cache + rewrite.
    # For Neo4jBeliefStore: DETACH DELETE.
    if hasattr(store, "_state_beliefs"):
        store._state_beliefs.pop(redundant.id, None)
        store._rewrite(
            store.STATE_BELIEFS_FILE,
            (
                __import__("dataclasses").asdict(b)
                for b in store._state_beliefs.values()
            ),
        )
    elif hasattr(store, "driver"):
        with store.driver.session() as sess:
            sess.run(
                "MATCH (b:Ecology:StateBelief {id: $id}) DETACH DELETE b",
                id=redundant.id,
            )


def consolidate(
    store: BeliefStore,
    embedder: Optional[ModalEmbedder] = None,
    llm_judge_borderline: bool = True,
    apply: bool = True,
    max_workers: int = 8,
    max_judge_calls: int = 100,
    verbose: bool = True,
) -> ConsolidationReport:
    """Run a full consolidation pass over the belief store.

    Args:
        store: belief store (any BeliefStore impl)
        embedder: ModalEmbedder; created from env if None
        llm_judge_borderline: if True, send borderline pairs to LLM judge.
            Set False for cheap dry-runs.
        apply: if True, actually merge. If False, return the report only.
    """
    report = ConsolidationReport()
    beliefs = store.all_state_beliefs()
    report.n_total = len(beliefs)
    if report.n_total < 2:
        return report

    if verbose:
        print(f"[consolidate] loaded {len(beliefs)} beliefs", flush=True)

    pairs = _candidate_pairs(beliefs)
    report.n_pairs_evaluated = len(pairs)
    if not pairs:
        return report

    if verbose:
        print(f"[consolidate] {len(pairs)} candidate pairs (same-scope, non-polymarket)",
              flush=True)

    # Sort beliefs by id for deterministic iteration
    by_id = {b.id: b for b in beliefs}

    candidates: dict[tuple[str, str], MergeCandidate] = {}

    # ── Tier 1: lexical ──
    if verbose:
        print(f"[consolidate] tier 1: lexical fast-path (SequenceMatcher >= {LEXICAL_RATIO_AUTO_MERGE})",
              flush=True)
    skip_for_embedding: set[tuple[str, str]] = set()
    for a, b in pairs:
        ratio = SequenceMatcher(
            None, a.statement_template.lower(), b.statement_template.lower(),
        ).ratio()
        if ratio >= LEXICAL_RATIO_AUTO_MERGE:
            canonical, redundant = _pick_canonical(a, b)
            key = (canonical.id, redundant.id)
            candidates[key] = MergeCandidate(
                canonical_id=canonical.id, redundant_id=redundant.id,
                source="lexical", score=ratio,
                reasoning=f"sequencematcher ratio {ratio:.3f}",
            )
            skip_for_embedding.add((a.id, b.id))
            skip_for_embedding.add((b.id, a.id))
    if verbose:
        print(f"[consolidate]   lexical merges: {len(candidates)}", flush=True)

    # ── Tier 2: embedding cosine ──
    embedder = embedder or ModalEmbedder()
    if embedder.is_available() and len(beliefs) >= 2:
        if verbose:
            print(f"[consolidate] tier 2: embedding {len(beliefs)} statements via Modal Qwen3 ...",
                  flush=True)
        statements = [b.statement_template for b in beliefs]
        try:
            embs = embedder.embed_texts(statements)
            emb_by_id = {b.id: e for b, e in zip(beliefs, embs)}
            if verbose:
                print(f"[consolidate]   embedded {len(emb_by_id)} statements (1024-dim each)",
                      flush=True)
        except Exception as e:
            logger.warning(f"Embedding batch failed; skipping embedding tier: {e}")
            print(f"[consolidate]   ⚠ embedding failed, skipping tier 2: {e}", flush=True)
            emb_by_id = {}

        if emb_by_id:
            if verbose:
                print(f"[consolidate]   computing cosine on {len(pairs)} pairs ...",
                      flush=True)
            borderline_pairs: list[tuple[StateBelief, StateBelief, float]] = []
            for a, b in pairs:
                if (a.id, b.id) in skip_for_embedding:
                    continue
                ea, eb = emb_by_id.get(a.id), emb_by_id.get(b.id)
                if not ea or not eb:
                    continue
                sim = cosine_sim(ea, eb)
                if sim >= EMBEDDING_COSINE_AUTO_MERGE:
                    canonical, redundant = _pick_canonical(a, b)
                    key = (canonical.id, redundant.id)
                    if key not in candidates:
                        candidates[key] = MergeCandidate(
                            canonical_id=canonical.id, redundant_id=redundant.id,
                            source="embedding", score=sim,
                            reasoning=f"cosine {sim:.3f} >= {EMBEDDING_COSINE_AUTO_MERGE}",
                        )
                elif sim >= EMBEDDING_COSINE_BORDERLINE:
                    borderline_pairs.append((a, b, sim))
            n_emb_merges_so_far = sum(
                1 for c in candidates.values() if c.source == "embedding"
            )
            if verbose:
                print(f"[consolidate]   embedding auto-merges: {n_emb_merges_so_far}",
                      flush=True)
                print(f"[consolidate]   borderline pairs (cosine "
                      f"{EMBEDDING_COSINE_BORDERLINE}-{EMBEDDING_COSINE_AUTO_MERGE}): "
                      f"{len(borderline_pairs)}", flush=True)

            # Tier 3: LLM judge on borderline (parallel, budget-capped)
            if llm_judge_borderline and borderline_pairs:
                # Sort borderline pairs by descending cosine — judge the
                # highest-similarity ones first; stop when budget hits.
                borderline_pairs.sort(key=lambda t: -t[2])
                if len(borderline_pairs) > max_judge_calls:
                    if verbose:
                        print(f"[consolidate] tier 3: capping LLM judge at "
                              f"{max_judge_calls}/{len(borderline_pairs)} borderline pairs "
                              f"(highest cosine first)", flush=True)
                    borderline_pairs = borderline_pairs[:max_judge_calls]
                else:
                    if verbose:
                        print(f"[consolidate] tier 3: judging all "
                              f"{len(borderline_pairs)} borderline pairs", flush=True)

                report.n_llm_borderline = len(borderline_pairs)

                def _judge_one(args):
                    a, b, sim = args
                    eq, reason = _llm_pairwise_judge(a, b)
                    return (a, b, sim, eq, reason)

                done_count = [0]
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [ex.submit(_judge_one, p) for p in borderline_pairs]
                    for f in as_completed(futures):
                        try:
                            a, b, sim, equivalent, reasoning = f.result()
                        except Exception as e:
                            logger.warning(f"judge worker raised: {e}")
                            continue
                        report.n_llm_calls += 1
                        done_count[0] += 1
                        if verbose and done_count[0] % 20 == 0:
                            print(f"[consolidate]   judge progress: "
                                  f"{done_count[0]}/{len(borderline_pairs)}",
                                  flush=True)
                        if equivalent:
                            report.n_llm_confirmed += 1
                            canonical, redundant = _pick_canonical(a, b)
                            key = (canonical.id, redundant.id)
                            if key not in candidates:
                                candidates[key] = MergeCandidate(
                                    canonical_id=canonical.id,
                                    redundant_id=redundant.id,
                                    source="llm_confirmed", score=sim,
                                    reasoning=reasoning,
                                )
                        else:
                            report.n_llm_rejected += 1
                if verbose:
                    print(f"[consolidate]   judge done: "
                          f"confirmed={report.n_llm_confirmed} "
                          f"rejected={report.n_llm_rejected}", flush=True)

    # Count by source
    for c in candidates.values():
        if c.source == "lexical":
            report.n_lexical_merges += 1
        elif c.source == "embedding":
            report.n_embedding_merges += 1

    # ── Apply merges ──
    if verbose:
        print(f"[consolidate] applying {len(candidates)} merges to store ...",
              flush=True)
    applied_redundant_ids: set[str] = set()
    if apply:
        for i, cand in enumerate(candidates.values()):
            if cand.canonical_id in applied_redundant_ids:
                continue
            if cand.redundant_id in applied_redundant_ids:
                continue
            canonical = store.get_state_belief(cand.canonical_id)
            redundant = store.get_state_belief(cand.redundant_id)
            if canonical is None or redundant is None:
                continue
            _apply_merge(store, canonical, redundant)
            applied_redundant_ids.add(cand.redundant_id)
            report.merges_applied.append(cand)
            if verbose and (i + 1) % 25 == 0:
                print(f"[consolidate]   applied {i+1}/{len(candidates)} ...",
                      flush=True)
    if verbose:
        print(f"[consolidate] done: {len(report.merges_applied)} merges applied",
              flush=True)

    return report
