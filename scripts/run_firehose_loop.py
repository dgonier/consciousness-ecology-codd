"""Firehose loop runner — both ecology and bare paths over the dataset.

For each trading day, runs both pipelines and writes a per-day JSONL
of watchlist outputs + targets for the eval harness to score.

Usage:
  .venv/bin/python -u scripts/run_firehose_loop.py [--limit N] [--days a,b,c] \
                                                   [--path bare|ecology|both]
                                                   [--focus AAPL,MSFT]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

env = ROOT / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
            continue
        os.environ.setdefault(k, v)

import dspy

from trophic.agents.apex_portfolio import ApexPortfolio, make_debate_portfolio
from trophic.agents.interrogator_herbivore import (
    PMInterrogator,
    make_pm_interrogator,
)
from trophic.beliefs.apex_signatures import (
    Order,
    TickerForwardReturns,
    TickerView,
    WatchlistFromFirehose,
    WatchlistFromFirehosePM,
    WatchlistFromObservations,
    WatchlistFromObservationsPM,
    WatchlistOracle,            # DEPRECATED in v4 — kept for back-compat tests
    WatchlistOracleFullWindow,  # v4: full-window forward-returns matrix
)
from trophic.beliefs.debate import (
    DebatePhase4Commit,
    run_debate,
)
from trophic.beliefs.strategy_committee import (
    aggregate_votes,
    conviction_weighted_vote,
    kelly_edge_vote,
    tiered_discrete_vote,
)
from trophic.beliefs.carnivore import aggregate
from trophic.beliefs.dspy_qwen_lm import QwenLocalLM
from trophic.beliefs.dspy_vllm_lm import make_vllm_lm, vllm_health_check
from trophic.beliefs.herbivores.cross_correlation import (
    CrossCorrelationHerbivore,
    update_state_from_bars,
)
from trophic.beliefs.herbivores.event_classifier import EventClassifierHerbivore
from trophic.beliefs.neo4j_store import Neo4jBeliefStore
from trophic.beliefs.pass_record import PassPersister
from trophic.beliefs.propagation import (
    apply_activation,
    log_odds,
    propagate_network,
)
from trophic.beliefs.schema import (
    BeliefActivation,
    Event,
    OutcomeBelief,
)
from trophic.config import ModelConfig
from trophic.data.dow30_universe import DOW30
from trophic.model_host import ModelHost


DATASET = ROOT / "data" / "firehose_dataset.jsonl"
EVAL_OUT = ROOT / "data" / "firehose_eval"
EVENT_CACHE = ROOT / "external" / "event_classifier_cache"
XCORR_STATE = ROOT / "data" / "cross_correlation_state.npz"

# v3.3 mixed-universe (20 tickers). Kept under --universe legacy for
# back-compat with v3.3-era runs. v4 default is Dow 30 (see DOW30 import).
LEGACY_UNIVERSE: list[str] = [
    "AAPL", "ABBV", "AMZN", "AVGO", "CSCO", "CVX", "GOOG", "HD", "JNJ", "JPM",
    "KO", "MA", "MCD", "MRK", "MSFT", "PEP", "PG", "UNH", "V", "WMT",
]

# Active universe — set by `main()` based on `--universe`. Default is
# DOW30 (v4 sweep). All module-level references downstream read through
# this binding; the choice is plumbed in one place.
UNIVERSE: list[str] = list(DOW30)


# ── v4 sweep path matrix ──────────────────────────────────────────────
#
# 9 paths total. ORACLE-SONNET intentionally dropped from `all-9` (it
# finished worst-of-9 in v3.3 at -6.73%). ECO-DEBATE-QWEN is new in
# v4 — 4-predator debate ecology, local Qwen-4B only.
#
# The canonical path names use the dashed form (e.g. `eco-debate-qwen`).
# Internal portfolio labels still use underscores (e.g.
# `eco_debate_qwen`) for filesystem / JSON-key cleanliness; the
# `_path_to_label` helper bridges the two.
ALL_PATHS_V4: tuple[str, ...] = (
    "bare-qwen",
    "bare-sonnet",
    "bare-opus",
    "ecology-qwen",
    "ecology-sonnet",
    "ecology-opus",
    "oracle-qwen",
    # NOTE: oracle-sonnet dropped in v4 (was -6.73% in v3.3).
    "oracle-opus",
    "eco-debate-qwen",   # NEW in v4 — 4-predator debate ecology
)

# v3.3 nine-way matrix preserved verbatim for back-compat re-runs.
V3_3_BASELINE_PATHS: tuple[str, ...] = (
    "bare-qwen",
    "bare-sonnet",
    "bare-opus",
    "ecology-qwen",
    "ecology-sonnet",
    "ecology-opus",
    "oracle-qwen",
    "oracle-sonnet",
    "oracle-opus",
)

PATH_GROUPS: dict[str, tuple[str, ...]] = {
    "all-9":         ALL_PATHS_V4,              # v4 sweep matrix
    "bare":          ALL_PATHS_V4[0:3],
    "ecology":       ALL_PATHS_V4[3:6],
    "oracle":        ("oracle-qwen", "oracle-opus"),
    "debate":        ("eco-debate-qwen",),
    "v3-3-baseline": V3_3_BASELINE_PATHS,
}


def _path_to_label(path: str) -> str:
    """Convert a dashed CLI path (`eco-debate-qwen`) into the underscored
    portfolio label used inside the runner (`eco_debate_qwen`)."""
    return path.replace("-", "_")


def hr() -> None:
    print("─" * 78)


def filter_news_to_universe(
    news: list[dict],
    universe: list[str] | set[str],
) -> list[dict]:
    """Drop articles whose tagged tickers are entirely outside the active
    universe (e.g. a TSLA-only headline when running --universe dow30).

    Articles that tag at least one in-universe ticker pass through
    untouched (we don't tamper with the article body — per the mission's
    coordination note, prefer pass over strip when an in-universe name
    is present alongside an out-of-universe name).

    Articles with no `tickers` / `all_tickers` field at all also pass
    through: they're broad-market commentary that downstream consumers
    can decide what to do with.
    """
    univ = set(universe) if not isinstance(universe, set) else universe
    out: list[dict] = []
    for art in news:
        tagged = art.get("all_tickers") or art.get("tickers") or []
        if not tagged:
            # Untagged article — let it through; downstream tier-1
            # classifiers will decide whether it's relevant.
            out.append(art)
            continue
        if any(t in univ for t in tagged):
            out.append(art)
        # else: entirely out-of-universe broadcast, drop.
    return out


def summarize_bars(firehose_bars: list[dict]) -> str:
    """Compact one-liner per ticker."""
    parts = []
    for b in firehose_bars:
        if b.get("close") is None:
            continue
        parts.append(
            f"{b['ticker']}: close={b['close']:.2f} vol={b.get('volume', 0):.0f}"
        )
    return "; ".join(parts)


def _validator_blocked_real_orders(rejections: list[str]) -> bool:
    """Did the validator block actual orders (vs. just emit advisory notes)?
    Heuristic: any rejection that names a ticker (\"AAPL: ...\") or contains
    'blocked', 'over budget', 'on unheld' is real. Pure scaling notes don't
    require a retry by themselves but rejections that mention specific
    failures do.
    """
    for r in rejections:
        rl = r.lower()
        if "blocked" in rl or "unheld" in rl or "not in universe" in rl:
            return True
        if "scaled by" in rl:
            continue
        # ticker-prefixed: usually real
        if ":" in r and r.split(":", 1)[0].strip().isupper():
            return True
    return False


def _order_to_dict(o) -> dict:
    """Coerce an Order Pydantic instance (or already-dict) into a plain
    dict the validator expects. Maps from_ticker→from, to_ticker→to since
    those are the alias names the validator already keys on.
    """
    if hasattr(o, "model_dump"):
        d = o.model_dump(by_alias=True)
        # by_alias=True emits 'from' / 'to' for the ticker pair fields
    elif isinstance(o, dict):
        d = dict(o)
    else:
        return {}
    return d


_PHILOSOPHY_WEIGHTS_PATH = ROOT / "tasks_v4" / "philosophy_weights.yaml"
_PHILOSOPHY_WEIGHTS_CACHE: dict | None = None
# Active weights path — settable from main() via --philosophy-weights.
_PHILOSOPHY_WEIGHTS_ACTIVE: Path = _PHILOSOPHY_WEIGHTS_PATH


def set_philosophy_weights_path(path: Path | str) -> None:
    """Override the weights path AND invalidate the cache so the next
    `get_philosophy_weights()` call re-reads."""
    global _PHILOSOPHY_WEIGHTS_ACTIVE, _PHILOSOPHY_WEIGHTS_CACHE
    _PHILOSOPHY_WEIGHTS_ACTIVE = Path(path)
    _PHILOSOPHY_WEIGHTS_CACHE = None


def get_philosophy_weights() -> dict[str, float]:
    """Load strategy committee weights from
    `tasks_v4/philosophy_weights.yaml` once and cache. The aggregator
    reads `weights[strategy]` for the three strategy names — extra keys
    (e.g. `philosophy_bias`) are ignored at the strategy-weight level.
    """
    global _PHILOSOPHY_WEIGHTS_CACHE
    if _PHILOSOPHY_WEIGHTS_CACHE is not None:
        return _PHILOSOPHY_WEIGHTS_CACHE
    active = _PHILOSOPHY_WEIGHTS_ACTIVE
    try:
        import yaml  # local import keeps yaml dep optional at module-import time
        with active.open() as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception as e:
        print(f"  [committee] WARN: could not load {active}: {e}; "
              f"falling back to equal strategy weights")
        raw = {}
    weights = {
        "conviction_weighted": float(raw.get("conviction_weighted", 0.34)),
        "tiered_discrete":     float(raw.get("tiered_discrete",     0.33)),
        "kelly_edge":          float(raw.get("kelly_edge",          0.33)),
    }
    _PHILOSOPHY_WEIGHTS_CACHE = weights
    return weights


def _run_strategy_committee(
    views: list,
    orders: list,
    weights: dict[str, float],
    label: str,
) -> list:
    """Replace the apex's `orders` with committee-aggregated orders.

    For each apex Order:
      - ROTATE: passed through unchanged (committee voting is per single
        ticker; ROTATE is a two-leg atomic primitive sized by the apex).
      - HOLD with no ticker: dropped (no-op).
      - Anything else: matched to its TickerView by ticker; three
        strategies vote; aggregator emits one Order (or None for
        "no consensus / all abstain", in which case the order is DROPPED).

    Errors from individual strategy votes / view lookups never abort —
    they just drop that one order with a debug log line.
    """
    if not orders:
        return []
    # Build a ticker → TickerView map. The phase1-A `ApexPMResponse`
    # validator already enforced one-view-per-order-ticker at the schema
    # level; this is a defensive fallback for paths where pred.views may
    # be missing.
    view_by_ticker: dict[str, TickerView] = {}
    for v in (views or []):
        if isinstance(v, TickerView):
            view_by_ticker[v.ticker.upper()] = v
        elif isinstance(v, dict) and "ticker" in v:
            try:
                view_by_ticker[str(v["ticker"]).upper()] = TickerView(**v)
            except Exception:
                continue

    out: list = []
    for o in orders:
        # Coerce dict-shaped orders (some apex paths emit dicts directly).
        if isinstance(o, dict):
            try:
                order = Order(**o)
            except Exception as e:
                print(f"  [{label} committee] skipping malformed order: {e}")
                continue
        elif isinstance(o, Order):
            order = o
        else:
            # Unknown shape — pass through untouched, let the validator decide.
            out.append(o)
            continue

        # ROTATE: pass through (committee is per-single-ticker).
        if order.side == "ROTATE":
            out.append(order)
            continue
        # HOLD with no ticker is a no-op the apex sometimes emits.
        if order.side == "HOLD" and not order.ticker:
            continue

        ticker_key = (order.ticker or "").upper()
        view = view_by_ticker.get(ticker_key)
        if view is None:
            # No view → committee can't vote. Drop the order rather than
            # let an un-voted size through. (ApexPMResponse validator
            # rejects this at the schema layer when used; this branch
            # is the runtime fallback.)
            print(f"  [{label} committee] no TickerView for {ticker_key}; dropping order")
            continue

        try:
            votes = [
                conviction_weighted_vote(view, order),
                tiered_discrete_vote(view, order),
                kelly_edge_vote(view, order),
            ]
            agg = aggregate_votes(votes, weights)
        except Exception as e:
            print(f"  [{label} committee] vote error on {ticker_key}: {e}; "
                  f"dropping order")
            continue
        if agg is None:
            # No consensus / all abstain. Drop silently — this is the
            # designed semantic for "committee says don't trade".
            continue
        out.append(agg)
    return out


def call_apex_pm_with_retry(
    apex_module: "dspy.Module",
    apex_kwargs: dict,
    portfolio: ApexPortfolio,
    universe: set[str],
    prices: dict[str, float],
    date: str,
    max_retries: int,
    label: str,
) -> dict:
    """Run the apex with up to `max_retries` re-tries when the validator
    rejects orders. Returns a result dict carrying the final accepted
    orders + the full per-attempt trace. Mutates `portfolio` only on the
    final accepted orders.
    """
    attempts: list[dict] = []
    final_validated: list[dict] = []
    final_raw: list[dict] = []
    final_rejected: list[str] = []
    final_rationale = ""
    last_error = None

    previous_attempts: list[dict] = []
    weights = get_philosophy_weights()
    for attempt_idx in range(max_retries + 1):
        try:
            pred = apex_module(
                **apex_kwargs,
                previous_attempts=previous_attempts,
            )
            # v4: route the apex's raw orders through the three-strategy
            # committee before validation. Aggregator may drop orders
            # (returns None for split-side / all-abstain); the runner
            # treats that as "committee says don't trade this ticker".
            apex_views = list(getattr(pred, "views", []) or [])
            apex_orders = list(pred.orders or [])
            committee_orders = _run_strategy_committee(
                apex_views, apex_orders, weights, label,
            )
            # Convert Pydantic Order instances → plain dicts for the validator
            raw_orders = [_order_to_dict(o) for o in committee_orders]
            rationale = (getattr(pred, "rationale", "") or "")[:300]
        except Exception as e:
            last_error = str(e)
            print(f"  [{label} apex-PM error attempt {attempt_idx}] {e}")
            raw_orders = []
            rationale = ""
            apex_views = []

        # v4 phase-2-fix: tax-aware filter (horizon sizing, forecast
        # consistency, min-hold, edge floor) now runs BEFORE the v3.3
        # budget scaler. Rationale: the v3.3 `validate_orders` shrinks
        # BUYs proportionally to fit the 100% budget; the v4 horizon
        # sizing validator then sees the post-scale size and rejects
        # it as off-tier. Running tax-aware FIRST means horizon sizing
        # checks the apex's HONEST (pre-scale) commitment — the scaler
        # afterward only shrinks survivors, and its own rejection
        # ("over-allocated, free capacity") still propagates into the
        # retry feedback below.
        from trophic.beliefs.validators import (
            validate_min_hold,
            validate_forecast_consistency,
            validate_horizon_sizing,
            validate_edge_floor,
        )  # noqa: F401 — re-imported here to satisfy mission-file grep test
        tax_survivors, tax_rejections = portfolio.filter_tax_aware(
            raw_orders, date, apex_views, universe=UNIVERSE,
        )
        validated, rejected = portfolio.validate_orders(
            tax_survivors, universe, prices,
        )
        if tax_rejections:
            rejected = list(tax_rejections) + list(rejected)
        attempts.append({
            "attempt": attempt_idx,
            "orders": raw_orders,
            "rejections": rejected,
            "n_validated": len(validated),
            "rationale": rationale,
        })

        # Persist the final state of these for the caller
        final_validated = validated
        final_raw = raw_orders
        final_rejected = rejected
        final_rationale = rationale

        # Decide whether to retry
        if attempt_idx >= max_retries:
            break
        if not _validator_blocked_real_orders(rejected):
            break  # pure advisory or no rejections — accept
        if not raw_orders:
            break  # apex emitted nothing; retrying won't help

        # Build prior-attempts feedback for next call. DSPy accepts list of
        # dicts or Pydantic PreviousAttempt; using dicts here is fine since
        # the signature coerces — but emit minimal data to keep tokens low.
        previous_attempts.append({
            "orders": [
                {k: v for k, v in {
                    "side": o.get("side"),
                    "ticker": o.get("ticker"),
                    "from": o.get("from"),
                    "to": o.get("to"),
                    "size_pct": o.get("size_pct"),
                }.items() if v is not None}
                for o in raw_orders
            ],
            "rejections": rejected,
        })
        print(f"  [{label} apex-PM retry {attempt_idx + 1}/{max_retries}] "
              f"{len(rejected)} rejections; revising")

    # Execute the final accepted orders ONCE
    portfolio.execute(date, final_validated, prices)
    return {
        "raw_orders": final_raw,
        "validated_orders": final_validated,
        "rejections": final_rejected,
        "rationale": final_rationale,
        "attempts_trace": attempts,
        "n_attempts": len(attempts),
    }


def orders_to_watchlist(raw_orders: list[dict], snapshot: dict) -> list[dict]:
    """Convert PM-mode orders back into a watchlist-shaped list so the
    existing eval harness, persister, and timeline viz can read it.
    BUY/SELL → action; size_pct repurposed as a confidence proxy via p_up.
    For SELL we set p_up<0.5 (bearish), BUY p_up>0.5 (bullish), HOLD p_up=0.5.
    """
    out: list[dict] = []
    for rank, o in enumerate(raw_orders, 1):
        side = (o.get("side") or "").upper()
        size = float(o.get("size_pct") or 0)
        # map size into a soft p_up: 5% → 0.55, 20% → 0.70, capped 0.85
        magnitude = min(0.35, size / 60.0)
        if side == "BUY":
            action = "BUY"
            p_up = 0.50 + magnitude
        elif side == "SELL":
            action = "SELL"
            p_up = 0.50 - magnitude
        else:
            action = "WATCH"
            p_up = 0.50
        out.append({
            "ticker": (o.get("ticker") or "").upper(),
            "action": action,
            "p_up": round(p_up, 4),
            "reason": (o.get("reasoning") or "")[:200],
            "rank": rank,
            "size_pct": round(size, 2),
        })
    return out


def normalize_watchlist_entry(entry: dict) -> dict | None:
    """Validate and clamp a single watchlist entry from the apex output."""
    if not isinstance(entry, dict):
        return None
    try:
        ticker = str(entry.get("ticker", "")).upper().strip()
        action_raw = str(entry.get("action", "")).upper().strip()
        p_up = float(entry.get("p_up", 0.5))
        reason = str(entry.get("reason", ""))[:300]
    except (TypeError, ValueError):
        return None
    if action_raw not in ("BUY", "SELL", "WATCH"):
        return None
    if not (1 < len(ticker) <= 5):
        return None
    p_up = max(0.0, min(1.0, p_up))
    return {"ticker": ticker, "action": action_raw, "p_up": p_up, "reason": reason}


def load_dataset() -> list[dict]:
    rows: list[dict] = []
    with DATASET.open() as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def reconstruct_bars_through(rows: list[dict], up_to_date: str) -> dict[str, dict[str, dict]]:
    """Rebuild bars_by_date_ticker from dataset rows (firehose_bars + targets)
    up to and including up_to_date. Used by the cross-correlation herb.
    """
    bars: dict[str, dict[str, dict]] = {}
    for row in rows:
        if row["date"] > up_to_date:
            break
        for fb in row["firehose_bars"]:
            bars.setdefault(fb["date"], {})[fb["ticker"]] = {"close": fb["close"]}
        for tg in row["targets"]:
            bars.setdefault(row["date"], {})[tg["ticker"]] = {"close": tg["actual_close"]}
    return bars


def fetch_outcomes_from_store(store: Neo4jBeliefStore) -> dict[str, OutcomeBelief]:
    out: dict[str, OutcomeBelief] = {}
    with store.driver.session() as sess:
        rows = sess.run("MATCH (o:Ecology:OutcomeBelief) RETURN o")
        for r in rows:
            n = r["o"]
            out[n["id"]] = OutcomeBelief(
                id=n["id"],
                ticker=n["ticker"],
                horizon_min=int(n.get("horizon_min", 1440)),
                statement=n.get("statement", ""),
                p_up=0.5,  # always start propagation from prior
            )
    return out


def reset_ticker_state(state: dict, universe: list[str]) -> None:
    """Reset all ticker-suffixed beliefs to their prior_p so each pass is
    independent. (We persist activated state in Neo4j across runs, but for
    the eval we want each day to be a clean pass.)"""
    for b in state.values():
        if "__ticker_" in b.id and any(
            b.id.endswith(f"__ticker_{tk}") for tk in universe
        ):
            b.current_p = b.prior_p
            b.evidence_log = []


def run_ecology_pass(
    row: dict,
    store: Neo4jBeliefStore,
    state: dict,
    links: list,
    xcorr_herb: CrossCorrelationHerbivore,
    event_herb: EventClassifierHerbivore,
    apex_observations: dspy.Module,
    focus_tickers: list[str],
    n_top_observations: int = 12,
    pass_persister: PassPersister | None = None,
    event_herb_lm: dspy.LM | None = None,
    portfolio: ApexPortfolio | None = None,
    pm_retries: int = 0,
    interrogator: PMInterrogator | None = None,
) -> dict:
    """Run the ecology pipeline for one trading day. Returns:
       {watchlist_raw, watchlist, observations, n_event_acts, n_xcorr_cluster, n_xcorr_sympathy}
    """
    NOW = time.time()

    # 0. Reset ticker beliefs to prior (clean pass)
    reset_ticker_state(state, UNIVERSE)

    # 1. Pull fresh outcomes (all at p_up=0.5)
    outcomes = fetch_outcomes_from_store(store)

    # 2. cross-correlation: cluster activations (FIRST, before event classifier)
    cluster_acts = xcorr_herb.emit_cluster_activations()

    # 3. event classifier on news. Also keep a side-table mapping each
    # activation back to its source article so the persister can record it.
    # Herbivore LM (typically vLLM) is swapped in via dspy.context.
    # Universe filter: drop articles tagged exclusively to out-of-universe
    # tickers before they reach the herbivore tier.
    news = filter_news_to_universe(row["firehose_news"], UNIVERSE)
    event_acts: list[BeliefActivation] = []
    article_id_by_act: dict[int, str] = {}  # id(activation) → article_id

    def _classify_news() -> None:
        for art in news:
            for a in event_herb.classify_to_activations(art):
                event_acts.append(a)
                article_id_by_act[id(a)] = art.get("id", "")

    if event_herb_lm is not None:
        with dspy.context(lm=event_herb_lm):
            _classify_news()
    else:
        _classify_news()

    # 4. cross-correlation: sympathy broadcast (AFTER event classifier so
    # we have ticker-specific activations to ripple)
    sympathy_acts = xcorr_herb.broadcast_sympathy(event_acts)

    all_acts = list(cluster_acts) + list(event_acts) + list(sympathy_acts)

    # 5. Apply activations to state. clamp event-activated leaves so
    # their decomposition children don't overwrite the direct evidence.
    clamped: set[str] = set()
    species_origin: dict[str, str] = {}
    fake_event = Event(
        id=f"firehose.{row['date']}.evt",
        timestamp=NOW,
        source="firehose_loop",
        raw_content=f"firehose for {row['date']}",
    )
    fake_event.activations = all_acts
    for a in all_acts:
        if a.target_belief_id in state:
            leaf_before = state[a.target_belief_id].current_p
            updated = apply_activation(state[a.target_belief_id], a, fake_event.id, NOW)
            state[a.target_belief_id] = updated
            leaf_after = updated.current_p
            applied_shift = log_odds(leaf_after) - log_odds(leaf_before) if leaf_before > 0 and leaf_before < 1 else None
            clamped.add(a.target_belief_id)
            species_origin[a.target_belief_id] = a.species_id
            if pass_persister is not None:
                pass_persister.add_activation(
                    target_belief_id=a.target_belief_id,
                    direction=a.direction_of_effect,
                    magnitude=a.magnitude,
                    decay_class=a.decay_class,
                    confidence=a.self_rated_confidence,
                    species_id=a.species_id,
                    reasoning=a.reasoning,
                    source_article_id=article_id_by_act.get(id(a)),
                    magnitude_logit=a.magnitude_logit,
                    applied_log_odds_shift=applied_shift,
                    leaf_p_before=leaf_before,
                    leaf_p_after=leaf_after,
                )

    # 6. Propagate
    propagate_network(
        state_beliefs=state,
        outcome_beliefs=outcomes,
        links=links,
        clamped_ids=clamped,
        max_iterations=25,
        scaling_factor=1.0,
    )

    # 7. Carnivore aggregation
    observations = aggregate(
        state_beliefs=state,
        outcome_beliefs=outcomes,
        links=links,
        triggering_events=[fake_event],
        watchlist_focus=focus_tickers,
        species_origin=species_origin,
    )
    top_obs = observations[:n_top_observations]

    # 8. Apex
    obs_dicts = [o.to_dict() for o in top_obs]

    # v4: optional InterrogatorHerbivore synthesis — invoked on the
    # carnivore observations BEFORE the apex sees them. The synthesis
    # is appended as one additional observation (kind="interrogator")
    # so downstream consumers (eval JSONL, decomposer) can identify
    # the source. Disabled / failed interrogators produce an abstain
    # entry; we never silently drop the synthesis call.
    interrogator_synth: dict | None = None
    if interrogator is not None and interrogator.enabled:
        try:
            interrogator_synth = interrogator.predict_only(
                obs_dicts, focus_tickers, date=row["date"],
            )
        except Exception as e:  # never crash the sweep
            interrogator_synth = {
                "kind": "interrogator",
                "abstain": True,
                "reason": f"runner-level error: {type(e).__name__}: {str(e)[:120]}",
            }
        obs_dicts = list(obs_dicts) + [_interrogator_observation_line(interrogator_synth)]

    if pass_persister is not None:
        pass_persister.add_observations(obs_dicts)

    # If apex_observations is None, the caller wants the ecology stack
    # only (no apex call) — used by the model-ablation runner to compute
    # observations once and feed them to multiple apex variants.
    if apex_observations is None:
        return {
            "watchlist": [],
            "watchlist_raw_count": 0,
            "n_event_activations": len(event_acts),
            "n_xcorr_cluster_activations": len(cluster_acts),
            "n_xcorr_sympathy_activations": len(sympathy_acts),
            "n_observations": len(observations),
            "n_top_observations_to_apex": len(top_obs),
            "observations": obs_dicts,
            "interrogator": interrogator_synth,
        }

    # PM mode: portfolio sees today's prices (post-advance), emits sized
    # orders. Non-PM mode: legacy watchlist signature.
    pm_mode = portfolio is not None
    pm_extras: dict = {}
    if pm_mode:
        prices_today = portfolio.advance_prices(row["date"], row.get("targets", []))
        portfolio_state, open_positions = portfolio.state_for_apex(row["date"], prices_today)
        result = call_apex_pm_with_retry(
            apex_module=apex_observations,
            apex_kwargs={
                "universe": UNIVERSE,
                "focus_tickers": focus_tickers,
                "observations": obs_dicts,
                "portfolio_state": portfolio_state,
                "open_positions": open_positions,
            },
            portfolio=portfolio,
            universe=set(UNIVERSE),
            prices=prices_today,
            date=row["date"],
            max_retries=pm_retries,
            label="ecology",
        )
        snapshot = portfolio.snapshot(row["date"], prices_today)
        watchlist_raw = result["raw_orders"]
        watchlist = orders_to_watchlist(watchlist_raw, snapshot)
        pm_extras = {
            "pm_orders_raw": watchlist_raw,
            "pm_orders_validated": result["validated_orders"],
            "pm_orders_rejected": result["rejections"],
            "pm_rationale": result["rationale"],
            "pm_portfolio_state_pre": portfolio_state,
            "pm_open_positions_pre": open_positions,
            "pm_snapshot_post": snapshot,
            "pm_retry_trace": result["attempts_trace"],
            "pm_n_attempts": result["n_attempts"],
        }
    else:
        try:
            pred = apex_observations(
                universe=UNIVERSE,
                focus_tickers=focus_tickers,
                observations=obs_dicts,
            )
            watchlist_raw = list(pred.watchlist or [])
        except Exception as e:
            watchlist_raw = []
            print(f"  [ecology apex error] {e}")

        watchlist = []
        for entry in watchlist_raw:
            n = normalize_watchlist_entry(entry)
            if n is not None and n["ticker"] in UNIVERSE:
                watchlist.append(n)

    return {
        "watchlist": watchlist,
        "watchlist_raw_count": len(watchlist_raw),
        "n_event_activations": len(event_acts),
        "n_xcorr_cluster_activations": len(cluster_acts),
        "n_xcorr_sympathy_activations": len(sympathy_acts),
        "n_observations": len(observations),
        "n_top_observations_to_apex": len(top_obs),
        "observations": obs_dicts,
        **pm_extras,
    }


async def _async_run_with_lm(fn, lm=None):
    """Run a sync portfolio-pass function (run_oracle_pass / run_bare_pass /
    run_ecology_apex_only) in a thread, optionally under a dspy.context(lm=...).

    Used by the --async-paths dispatcher so all 9 apex paths fire
    concurrently. Each path mutates only its own ApexPortfolio so no
    locking is needed.
    """
    import asyncio
    if lm is None:
        return await asyncio.to_thread(fn)
    def _wrapped():
        with dspy.context(lm=lm):
            return fn()
    return await asyncio.to_thread(_wrapped)


def run_ecology_apex_only(
    row: dict,
    obs_dicts: list[dict],
    apex_observations: dspy.Module,
    focus_tickers: list[str],
    portfolio: ApexPortfolio,
    pm_retries: int,
    label: str,
) -> dict:
    """Reuse pre-computed carnivore observations; run only the apex +
    portfolio leg. Used for ecology-sonnet / ecology-opus variants so we
    don't re-run the herbivore + propagation per model.
    """
    prices_today = portfolio.advance_prices(row["date"], row.get("targets", []))
    portfolio_state, open_positions = portfolio.state_for_apex(row["date"], prices_today)
    result = call_apex_pm_with_retry(
        apex_module=apex_observations,
        apex_kwargs={
            "universe": UNIVERSE,
            "focus_tickers": focus_tickers,
            "observations": obs_dicts,
            "portfolio_state": portfolio_state,
            "open_positions": open_positions,
        },
        portfolio=portfolio,
        universe=set(UNIVERSE),
        prices=prices_today,
        date=row["date"],
        max_retries=pm_retries,
        label=label,
    )
    snapshot = portfolio.snapshot(row["date"], prices_today)
    watchlist_raw = result["raw_orders"]
    watchlist = orders_to_watchlist(watchlist_raw, snapshot)
    return {
        "watchlist": watchlist,
        "watchlist_raw_count": len(watchlist_raw),
        "n_observations_seen": len(obs_dicts),
        "pm_orders_raw": watchlist_raw,
        "pm_orders_validated": result["validated_orders"],
        "pm_orders_rejected": result["rejections"],
        "pm_rationale": result["rationale"],
        "pm_portfolio_state_pre": portfolio_state,
        "pm_open_positions_pre": open_positions,
        "pm_snapshot_post": snapshot,
        "pm_retry_trace": result["attempts_trace"],
        "pm_n_attempts": result["n_attempts"],
    }


# ── Full-window Oracle (v4) helpers ────────────────────────────────────
#
# v3.3 had a per-day Oracle that saw only tomorrow_returns; v4 builds a
# `future_returns_matrix` covering EVERY day from `today+1` through window-end
# and shows it to the apex at once. The apex commits to a buy-low-sell-high
# trajectory of FEW infrequent trades instead of reacting bar-by-bar.

# Rough heuristic: each "0.0123" float is ~7 chars + comma/quote glue ≈ 8
# input tokens. 12K tokens worth of matrix ≈ 1500 floats. For 20 tickers
# * 66 days = 1320 floats, which fits, but if the watchlist grows we
# prune to the top-N by realized volatility.
_FUTURE_MATRIX_TOKEN_BUDGET: int = 12_000
_FUTURE_MATRIX_CHARS_PER_FLOAT: int = 8  # truncate-to-4-decimals + delimiters
_FUTURE_MATRIX_CHARS_PER_TOKEN: int = 4  # crude tokens-per-char approximation


def build_future_returns_matrix(
    today_idx: int,
    rows: list[dict],
    universe: list[str],
    days_remaining: int,
    token_budget: int = _FUTURE_MATRIX_TOKEN_BUDGET,
) -> tuple[list[TickerForwardReturns], list[str]]:
    """Build the per-ticker forward-returns matrix shown to the full-window
    Oracle. Returns (matrix, pruned_tickers).

    Inputs:
      - today_idx: index of today's row in the per-day dataset list.
      - rows: full dataset list (each row has 'targets' with actual_return).
      - universe: tickers to include in the matrix.
      - days_remaining: how many trading days are left in the window
        INCLUDING today. The matrix never extends past today_idx + days_remaining.
      - token_budget: rough cap; over this, prune to top-N most volatile.

    Returned forward_returns[ticker] is the list of close-to-close returns
    for days (today_idx+1 .. today_idx+days_remaining-1) — i.e. the moves
    from today's close into each subsequent day. Length ≤ days_remaining - 1
    (or 0 if today is the last day of the window). Floats are truncated to
    4 decimal places via `round(r, 4)`.

    HARD INVARIANT: no entry in any forward_returns array corresponds to a
    day beyond `today_idx + days_remaining - 1`. The caller's assertion in
    the runner ensures we never leak data past window-end.
    """
    assert days_remaining >= 0, f"days_remaining must be ≥ 0, got {days_remaining}"
    last_idx_exclusive = today_idx + days_remaining
    # The "forward" days are the rows whose targets describe the move from
    # the prior close to that row's close — i.e. rows[today_idx+1 ..
    # last_idx_exclusive-1]. The first such row's actual_return is the
    # close-to-close move from today's close to tomorrow's close.
    future_rows = rows[today_idx + 1 : last_idx_exclusive]
    # Assertion in the runner verifies last_idx_exclusive ≤ len(rows); here
    # we just assert we never sliced past it.
    assert len(future_rows) <= max(0, days_remaining - 1), (
        f"future_rows leak: got {len(future_rows)} rows, expected ≤ "
        f"{max(0, days_remaining - 1)} (days_remaining-1)"
    )

    per_ticker: dict[str, list[float]] = {t: [] for t in universe}
    for fr in future_rows:
        for tg in fr.get("targets", []):
            tk = tg.get("ticker")
            if tk in per_ticker:
                per_ticker[tk].append(round(float(tg["actual_return"]), 4))

    # Drop tickers with no future data at all (e.g. if a ticker is missing
    # from all future rows). They go to pruned_tickers so the apex knows
    # the omission was intentional.
    pruned: list[str] = []
    raw_matrix: list[tuple[str, list[float], float]] = []  # (ticker, returns, abs-vol)
    for tk in universe:
        rs = per_ticker[tk]
        if not rs:
            pruned.append(tk)
            continue
        # Realized vol proxy = stdev (no library — use pure python).
        if len(rs) >= 2:
            mean_r = sum(rs) / len(rs)
            var = sum((r - mean_r) ** 2 for r in rs) / len(rs)
            vol = var ** 0.5
        else:
            vol = abs(rs[0])
        raw_matrix.append((tk, rs, vol))

    # Token-budget check: prune lowest-volatility tickers if we exceed budget.
    def _estimate_tokens(entries: list[tuple[str, list[float], float]]) -> int:
        n_floats = sum(len(r) for _, r, _ in entries)
        # ~8 chars/float + 4 chars/ticker + delimiters ≈ char count → /4 → tokens
        chars = n_floats * _FUTURE_MATRIX_CHARS_PER_FLOAT + 12 * len(entries)
        return chars // _FUTURE_MATRIX_CHARS_PER_TOKEN

    if _estimate_tokens(raw_matrix) > token_budget:
        # Sort by descending vol (keep most volatile = most alpha-rich)
        raw_matrix.sort(key=lambda e: e[2], reverse=True)
        kept: list[tuple[str, list[float], float]] = []
        for entry in raw_matrix:
            kept.append(entry)
            if _estimate_tokens(kept) > token_budget:
                # Last add overflowed; back out and stop.
                pruned.append(kept.pop()[0])
                break
        # Any remaining tail past the budget-break also get pruned.
        already_kept = {e[0] for e in kept}
        for tk, _, _ in raw_matrix:
            if tk not in already_kept and tk not in pruned:
                pruned.append(tk)
        raw_matrix = kept

    matrix = [
        TickerForwardReturns(ticker=tk, forward_returns=rs)
        for tk, rs, _ in raw_matrix
    ]
    return matrix, sorted(pruned)


def run_oracle_pass(
    row: dict,
    apex_oracle: dspy.Module,
    focus_tickers: list[str],
    portfolio: ApexPortfolio,
    *,
    today_idx: int,
    rows: list[dict],
    pm_retries: int = 0,
    label: str = "oracle",
) -> dict:
    """Full-window Oracle pipeline (v4): apex sees the per-ticker forward-returns
    matrix from today+1 through window-end and allocates a multi-day trajectory.

    `today_idx` and `rows` are required so we can slice the dataset to build
    `future_returns_matrix`. `focus_tickers` is retained for API symmetry
    but unused by the full-window signature (the entire watchlist is
    surfaced via the matrix).
    """
    n_total = len(rows)
    # days_remaining counts today (inclusive) through window-end.
    days_remaining = n_total - today_idx
    assert days_remaining >= 1, (
        f"oracle dispatched with days_remaining={days_remaining}; today_idx="
        f"{today_idx} n_total={n_total}"
    )

    matrix, pruned_tickers = build_future_returns_matrix(
        today_idx=today_idx,
        rows=rows,
        universe=UNIVERSE,
        days_remaining=days_remaining,
    )
    # LEAK GUARD — assert matrix never extends past window-end.
    max_len_seen = max((len(m.forward_returns) for m in matrix), default=0)
    assert max_len_seen <= days_remaining - 1, (
        f"future_returns_matrix leak: max forward_returns length "
        f"{max_len_seen} > days_remaining-1 ({days_remaining - 1})"
    )

    prices_today = portfolio.advance_prices(row["date"], row.get("targets", []))
    portfolio_state, open_positions = portfolio.state_for_apex(row["date"], prices_today)

    watchlist_tickers = [m.ticker for m in matrix]

    result = call_apex_pm_with_retry(
        apex_module=apex_oracle,
        apex_kwargs={
            "today": row["date"],
            "days_remaining": days_remaining,
            "portfolio_state": portfolio_state,
            "positions": open_positions,
            "watchlist": watchlist_tickers,
            "future_returns_matrix": matrix,
            "pruned_tickers": pruned_tickers,
        },
        portfolio=portfolio,
        universe=set(UNIVERSE),
        prices=prices_today,
        date=row["date"],
        max_retries=pm_retries,
        label=label,
    )
    snapshot = portfolio.snapshot(row["date"], prices_today)
    watchlist_raw = result["raw_orders"]
    watchlist = orders_to_watchlist(watchlist_raw, snapshot)
    return {
        "watchlist": watchlist,
        "watchlist_raw_count": len(watchlist_raw),
        "n_news_used": 0,
        "days_remaining": days_remaining,
        "future_returns_matrix_size": len(matrix),
        "future_returns_matrix_pruned": pruned_tickers,
        "pm_orders_raw": watchlist_raw,
        "pm_orders_validated": result["validated_orders"],
        "pm_orders_rejected": result["rejections"],
        "pm_rationale": result["rationale"],
        "pm_portfolio_state_pre": portfolio_state,
        "pm_open_positions_pre": open_positions,
        "pm_snapshot_post": snapshot,
        "pm_retry_trace": result["attempts_trace"],
        "pm_n_attempts": result["n_attempts"],
    }


# ── ECO-DEBATE-QWEN (v4 new path) ──────────────────────────────────────


def _interrogator_observation_line(synth: dict) -> dict:
    """Render a `PMInterrogator.predict_only` synthesis dict into one
    additional observation-shaped entry the apex pipeline can read.
    Mirrors the carnivore observation shape minimally.
    """
    if synth.get("abstain"):
        return {
            "ticker": "",
            "outcome_id": "interrogator__abstain",
            "p_up": 0.5,
            "magnitude": 0.0,
            "reasoning": f"[interrogator abstained] {synth.get('reason', '')}",
            "species_id": "interrogator",
        }
    bias = (synth.get("bias") or "flat").lower()
    pct = float(synth.get("pct_move", 0.0))
    p_up = 0.5
    if bias == "up":
        p_up = min(0.95, 0.5 + max(0.05, abs(pct) / 10.0))
    elif bias == "down":
        p_up = max(0.05, 0.5 - max(0.05, abs(pct) / 10.0))
    return {
        "ticker": synth.get("ticker", ""),
        "outcome_id": f"interrogator__{synth.get('ticker', '?')}",
        "p_up": round(p_up, 4),
        "magnitude": round(pct, 4),
        "reasoning": (
            f"[interrogator {synth.get('signal', 'weak')}@"
            f"{synth.get('confidence', 0.5):.2f}] "
            f"{synth.get('rationale', '')}"
        )[:300],
        "species_id": "interrogator",
        "n_questions": synth.get("n_questions", 0),
    }


def _build_observations_blob(
    obs_dicts: list[dict],
    interrogator_synth: dict | None,
    bars_summary: str,
    news_count: int,
) -> str:
    """Compose the `observations: str` blob passed to the debate apex.

    The debate signature takes a single string (not a list); we serialize
    the (optional) interrogator synthesis + the top carnivore observations
    + a bars summary into a compact prompt-sized blob.
    """
    lines = []
    if interrogator_synth and not interrogator_synth.get("abstain"):
        lines.append("INTERROGATOR SYNTHESIS:")
        lines.append(
            f"  ticker={interrogator_synth.get('ticker','?')} "
            f"bias={interrogator_synth.get('bias','flat')} "
            f"pct_move={interrogator_synth.get('pct_move',0.0):+.2f} "
            f"signal={interrogator_synth.get('signal','weak')} "
            f"conf={interrogator_synth.get('confidence',0.5):.2f}"
        )
        rat = (interrogator_synth.get("rationale") or "")[:240]
        if rat:
            lines.append(f"  rationale: {rat}")
    elif interrogator_synth and interrogator_synth.get("abstain"):
        lines.append(
            f"INTERROGATOR: abstained ({interrogator_synth.get('reason','')[:120]})"
        )
    lines.append("")
    lines.append(f"BARS (news_count={news_count}):")
    lines.append(f"  {bars_summary}"[:600])
    lines.append("")
    lines.append("CARNIVORE OBSERVATIONS:")
    for o in obs_dicts[:12]:
        tk = o.get("ticker") or "?"
        p_up = o.get("p_up")
        mag = o.get("magnitude")
        reason = (o.get("reasoning") or "")[:140]
        parts = [str(tk)]
        if p_up is not None:
            parts.append(f"p_up={float(p_up):.2f}")
        if mag is not None:
            parts.append(f"mag={float(mag):+.3f}")
        if reason:
            parts.append(reason)
        lines.append("  " + " | ".join(parts))
    return "\n".join(lines)


def _order_dict_for_validators(o, *, default_horizon: str = "") -> dict:
    """Convert a debate `Order` Pydantic instance into the dict shape
    that `ApexPortfolio.validate_orders` and `filter_tax_aware` expect.
    Adds `from`/`to` aliases for ROTATEs (the validator keys on those).
    """
    if hasattr(o, "model_dump"):
        d = o.model_dump(by_alias=True)
    elif isinstance(o, dict):
        d = dict(o)
    else:
        return {}
    # Ensure primary_horizon and expected_alpha_bps are carried through
    if not d.get("primary_horizon") and default_horizon:
        d["primary_horizon"] = default_horizon
    return d


def _apply_predator_orders(
    sub,
    validated: list[dict],
    prices: dict[str, float],
    today: str,
    slip_bps: float,
    tax_fraction: float,
) -> None:
    """Apply validated orders to one PredatorSubPortfolio. Mirrors
    `ApexPortfolio.execute` semantics but writes to the per-predator
    slice (sub.buy / sub.sell / sub.rotate).
    """
    for o in validated:
        side = o.get("side")
        t = o.get("ticker")
        px = prices.get(t)
        if not px:
            continue
        if side == "BUY":
            dollars = float(o.get("dollars_intent") or 0.0)
            if dollars <= 0:
                continue
            sub.buy(
                t, dollars, slip_bps, today,
                primary_horizon=o.get("primary_horizon", ""),
                price=px,
            )
        elif side == "SELL":
            pos = sub.positions.get(t)
            if not pos or pos.shares <= 0:
                continue
            # validate_orders writes `size_pct` as % of position for SELLs.
            frac = float(o.get("size_pct") or 0.0) / 100.0
            sub.sell(
                t, frac, slip_bps, tax_fraction, today,
                price=px,
            )


def _validate_one_predator(
    sub,
    commit: "DebatePhase4Commit",
    apex_portfolio: ApexPortfolio,
    prices: dict[str, float],
    today: str,
    universe: set[str],
) -> tuple[list[dict], list[str]]:
    """Run the v4 validator chain against ONE predator's final_orders
    using the predator's sub-portfolio as the budget/positions source.

    The chain order matches `call_apex_pm_with_retry`: tax-aware filter
    first, then v3.3 budget scaler. We temporarily install the
    sub-portfolio's view onto a throw-away ApexPortfolio so the existing
    validator methods (which read `self.positions`, `self.cash`, etc.)
    run unmodified.

    Returns (validated_dicts_with_dollars_intent, rejections).
    """
    raw = [
        _order_dict_for_validators(o)
        for o in (commit.final_orders or [])
    ]
    if not raw:
        return [], []

    # Build a single-book proxy ApexPortfolio that views the predator's
    # slice as a stand-alone portfolio. This lets `filter_tax_aware`
    # and `validate_orders` work without modification.
    proxy = ApexPortfolio(
        label=f"debate_{sub.predator_id}",
        starting_cash=sub.starting_cash,
        slippage_bps=apex_portfolio.slippage_bps,
        tax_rate=apex_portfolio.tax_rate,
        cash_yield_annual=0.0,  # cash yield accrues at the parent level
    )
    proxy.cash = sub.cash
    proxy.tax_owed = sub.tax_owed_accrued
    proxy.positions = dict(sub.positions)
    proxy._last_price = dict(apex_portfolio._last_price)
    proxy._dates_seen = list(apex_portfolio._dates_seen)

    views_for_validators: list = []  # Phase4Commit doesn't carry views;
    # forecast-consistency / regime-invalidation in run_tax_aware_chain
    # skips when views are empty (verified in phase2-D contract).

    # Compute slice_fraction so horizon-sizing tier bases scale down
    # for the predator's $25K share of the $100K apex. Example:
    # h60 base 50% × 0.25 → 12.5% per-slice base, which matches
    # what the predator actually emits when sizing inside its slice.
    total_starting_cash = sum(
        s.starting_cash for s in apex_portfolio.sub_portfolios.values()
    )
    slice_fraction = (
        sub.starting_cash / total_starting_cash
        if total_starting_cash > 0
        else 1.0
    )

    tax_survivors, tax_rejections = proxy.filter_tax_aware(
        raw, today, views_for_validators,
        slice_fraction=slice_fraction,
        universe=universe,
    )
    validated, scaler_rejections = proxy.validate_orders(
        tax_survivors, universe, prices,
    )
    rejections = list(tax_rejections) + list(scaler_rejections)
    return validated, rejections


async def _run_debate_pass_async(
    row: dict,
    apex_portfolio: ApexPortfolio,
    *,
    today_idx: int,
    rows: list[dict],
    obs_dicts: list[dict],
    interrogator: PMInterrogator | None,
    focus_tickers: list[str],
    lm,
    run_label: str = "eco_debate_qwen",
    pm_retries: int = 0,
) -> dict:
    """Async-native debate dispatch. Same body as `run_debate_pass`
    below; factored so the async-paths dispatcher can await it directly
    (nesting `asyncio.run` inside the outer `asyncio.run(_dispatch())`
    is illegal). The sync wrapper `run_debate_pass` calls
    `asyncio.run(_run_debate_pass_async(...))` for non-async-paths
    sweeps.
    """
    if not apex_portfolio.is_debate_mode():
        raise ValueError("run_debate_pass requires debate-mode portfolio")

    today = row["date"]
    n_total = len(rows)
    days_remaining = max(1, n_total - today_idx)

    prices_today = apex_portfolio.advance_prices(today, row.get("targets", []))

    interrogator_synth: dict | None = None
    if interrogator is not None and interrogator.enabled:
        try:
            interrogator_synth = interrogator.predict_only(
                obs_dicts, focus_tickers, date=today,
            )
        except Exception as e:
            interrogator_synth = {
                "kind": "interrogator",
                "abstain": True,
                "reason": f"runner-level error: {type(e).__name__}: {str(e)[:120]}",
            }

    bars_summary = summarize_bars(row.get("firehose_bars", []))
    observations_blob = _build_observations_blob(
        obs_dicts, interrogator_synth, bars_summary, row.get("n_news", 0),
    )
    watchlist = list(UNIVERSE)
    universe_set = set(UNIVERSE)

    # Debate dispatch — 16 LM calls fired in 4 phase-batches. If any
    # phase's DSPy parse fails (e.g. Qwen returns a JSON list where a
    # single object is expected, an issue surfaced during 1-day smoke),
    # we degrade to a no-trade day rather than crashing the whole sweep.
    # Subsequent days dispatch fresh from phase 1 so a single bad day
    # doesn't poison the run.
    commits: dict[str, DebatePhase4Commit]
    debate_error: str | None = None
    try:
        commits = await run_debate(
            apex_portfolio,
            today,
            days_remaining,
            watchlist,
            observations_blob,
            prices_today,
            lm,
            run_label=run_label,
            write_transcript=True,
        )
    except Exception as e:  # never crash the sweep on a bad debate day
        debate_error = f"{type(e).__name__}: {str(e)[:240]}"
        print(f"  [eco_debate_qwen] DEBATE FAILED {today}: {debate_error}; "
              f"falling back to no-trade day")
        # Synthesize empty commits so the downstream loop just records
        # a flat day per predator.
        commits = {}
        for pid, sub in apex_portfolio.sub_portfolios.items():
            commits[pid] = DebatePhase4Commit(
                predator_id=pid,
                rationale=f"debate-fail-fallback: {debate_error}",
            )

    predator_results: dict[str, dict] = {}
    total_validated: list[dict] = []
    total_rejected: list[str] = []
    for pid, commit in commits.items():
        sub = apex_portfolio.sub_portfolios[pid]
        validated, rejections = _validate_one_predator(
            sub, commit, apex_portfolio, prices_today, today, universe_set,
        )
        _apply_predator_orders(
            sub, validated, prices_today, today,
            apex_portfolio.slippage_bps, apex_portfolio.tax_rate,
        )
        opened_now: list[str] = []
        for new_thesis in (commit.final_theses_to_open or []):
            try:
                sub.thesis_book.add(new_thesis)
                opened_now.append(new_thesis.thesis_id)
            except Exception as e:
                total_rejected.append(
                    f"[{pid}] thesis-open rejected: {type(e).__name__}: "
                    f"{str(e)[:140]}"
                )
        for tid in (commit.final_theses_to_close or []):
            try:
                sub.thesis_book.close(
                    tid, status="matured", closed_at_date=today,
                    realized_pnl=0.0, realized_return_pct=0.0, trigger=None,
                )
            except Exception as e:
                total_rejected.append(
                    f"[{pid}] thesis-close rejected: {type(e).__name__}: "
                    f"{str(e)[:140]}"
                )
        try:
            sub.thesis_book.sweep_overdue(today_idx, opened_idx_by_thesis={})
        except Exception:
            pass

        sub_eq = sub.equity(prices_today)
        predator_results[pid] = {
            "philosophy": sub.philosophy,
            "equity": round(sub_eq, 2),
            "cash": round(sub.cash, 2),
            "tax_owed_accrued": round(sub.tax_owed_accrued, 2),
            "n_validated": len(validated),
            "n_rejected": len(rejections),
            "n_orders_proposed": len(commit.final_orders or []),
            "n_theses_opened": len(opened_now),
            "n_theses_closed": len(commit.final_theses_to_close or []),
            "active_theses": len(sub.thesis_book.active_theses()),
        }
        total_validated.extend(validated)
        total_rejected.extend(rejections)

    snapshot = apex_portfolio.snapshot(today, prices_today)
    snapshot["equity"] = round(apex_portfolio.equity(prices_today), 2)
    snapshot["cash"] = round(apex_portfolio.total_cash, 2)
    snapshot["invested_pct"] = round(apex_portfolio.invested_pct(prices_today), 2)
    snapshot["tax_owed_accrued"] = round(apex_portfolio.total_tax_owed_accrued, 2)
    snapshot["predator_breakdown"] = predator_results

    watchlist_view = orders_to_watchlist(total_validated, snapshot)

    return {
        "watchlist": watchlist_view,
        "watchlist_raw_count": sum(
            len(c.final_orders or []) for c in commits.values()
        ),
        "pm_orders_raw": [
            _order_dict_for_validators(o)
            for c in commits.values()
            for o in (c.final_orders or [])
        ],
        "pm_orders_validated": total_validated,
        "pm_orders_rejected": total_rejected,
        "pm_rationale": " | ".join(
            f"{pid}: {(c.rationale or '')[:100]}"
            for pid, c in commits.items()
        )[:600],
        "pm_snapshot_post": snapshot,
        "predator_breakdown": predator_results,
        "interrogator": interrogator_synth or {"abstain": True, "reason": "disabled"},
        "n_predators": len(commits),
        "days_remaining": days_remaining,
        "debate_error": debate_error,
    }


def run_debate_pass(
    row: dict,
    apex_portfolio: ApexPortfolio,
    *,
    today_idx: int,
    rows: list[dict],
    obs_dicts: list[dict],
    interrogator: PMInterrogator | None,
    focus_tickers: list[str],
    lm,
    run_label: str = "eco_debate_qwen",
    pm_retries: int = 0,
) -> dict:
    """Per-day dispatch for the ECO-DEBATE-QWEN path (sync wrapper).

    Wires the 4-phase `run_debate` orchestrator through the runner:
      1. Advance prices on the parent ApexPortfolio.
      2. Run the (optional) PMInterrogator on the carnivore observations.
      3. Build the `observations: str` blob for the debate signature.
      4. `await run_debate(...)` to get per-predator `DebatePhase4Commit`s.
      5. For each predator: run the v4 validator chain against its
         sub-portfolio's view, apply the survivors, and update its
         thesis book.
      6. Build a snapshot with per-predator breakdown for the eval JSONL.

    `pm_retries` is currently unused for the debate path (each phase-4
    commit is already the predator's own retry-after-debate). Reserved
    for future per-predator retry-on-rejection logic.

    For async-paths dispatch the runner awaits
    `_run_debate_pass_async(...)` directly to avoid nesting
    `asyncio.run` inside the outer dispatcher.
    """
    import asyncio
    return asyncio.run(_run_debate_pass_async(
        row, apex_portfolio,
        today_idx=today_idx, rows=rows, obs_dicts=obs_dicts,
        interrogator=interrogator, focus_tickers=focus_tickers,
        lm=lm, run_label=run_label, pm_retries=pm_retries,
    ))


def run_bare_pass(
    row: dict,
    apex_firehose: dspy.Module,
    focus_tickers: list[str],
    max_news: int = 35,
    portfolio: ApexPortfolio | None = None,
    pm_retries: int = 0,
) -> dict:
    """Run the bare-Qwen pipeline for one trading day."""
    # Universe filter first, then cap at max_news. This way an article
    # budget of 35 isn't burned on irrelevant out-of-universe headlines.
    news = filter_news_to_universe(row["firehose_news"], UNIVERSE)[:max_news]
    headlines = []
    for art in news:
        title = art.get("title", "")
        desc = art.get("description") or ""
        tickers = [t for t in (art.get("all_tickers") or art.get("tickers") or []) if t in UNIVERSE]
        line = f"{title}"
        if tickers:
            line += f" [tickers: {','.join(tickers)}]"
        if desc:
            line += f" — {desc[:200]}"
        headlines.append(line)
    bars_summary = summarize_bars(row["firehose_bars"])

    pm_mode = portfolio is not None
    pm_extras: dict = {}
    if pm_mode:
        prices_today = portfolio.advance_prices(row["date"], row.get("targets", []))
        portfolio_state, open_positions = portfolio.state_for_apex(row["date"], prices_today)
        result = call_apex_pm_with_retry(
            apex_module=apex_firehose,
            apex_kwargs={
                "universe": UNIVERSE,
                "focus_tickers": focus_tickers,
                "news_headlines": headlines,
                "prev_bars_summary": bars_summary,
                "portfolio_state": portfolio_state,
                "open_positions": open_positions,
            },
            portfolio=portfolio,
            universe=set(UNIVERSE),
            prices=prices_today,
            date=row["date"],
            max_retries=pm_retries,
            label="bare",
        )
        snapshot = portfolio.snapshot(row["date"], prices_today)
        watchlist_raw = result["raw_orders"]
        watchlist = orders_to_watchlist(watchlist_raw, snapshot)
        pm_extras = {
            "pm_orders_raw": watchlist_raw,
            "pm_orders_validated": result["validated_orders"],
            "pm_orders_rejected": result["rejections"],
            "pm_rationale": result["rationale"],
            "pm_portfolio_state_pre": portfolio_state,
            "pm_open_positions_pre": open_positions,
            "pm_snapshot_post": snapshot,
            "pm_retry_trace": result["attempts_trace"],
            "pm_n_attempts": result["n_attempts"],
        }
    else:
        try:
            pred = apex_firehose(
                universe=UNIVERSE,
                focus_tickers=focus_tickers,
                news_headlines=headlines,
                prev_bars_summary=bars_summary,
            )
            watchlist_raw = list(pred.watchlist or [])
        except Exception as e:
            watchlist_raw = []
            print(f"  [bare apex error] {e}")

        watchlist = []
        for entry in watchlist_raw:
            n = normalize_watchlist_entry(entry)
            if n is not None and n["ticker"] in UNIVERSE:
                watchlist.append(n)

    return {
        "watchlist": watchlist,
        "watchlist_raw_count": len(watchlist_raw),
        "n_news_used": len(headlines),
        **pm_extras,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="run only first N days of dataset")
    ap.add_argument("--days", type=str, default=None,
                    help="comma-separated specific dates (overrides --limit)")
    # --path accepts either a single path from ALL_PATHS_V4 (or
    # ORACLE-SONNET via explicit opt-in) or a group name from PATH_GROUPS
    # (all-9, bare, ecology, oracle, debate, v3-3-baseline). Legacy v3.3
    # group names ("bare", "ecology", "both", "all-with-oracles",
    # "model-ablation") are still accepted for back-compat.
    _v4_path_choices = sorted(set(
        list(ALL_PATHS_V4)
        + list(PATH_GROUPS.keys())
        + [
            # Back-compat: ORACLE-SONNET dropped from all-9 default, but
            # still explicitly accessible as a single-path opt-in.
            "oracle-sonnet",
            # Legacy v3.3 aliases
            "bare", "ecology", "both",
            "ecology-sonnet", "ecology-opus",
            "bare-sonnet", "bare-opus",
            "all-with-oracles", "model-ablation",
        ]
    ))
    ap.add_argument("--path", choices=_v4_path_choices, default="all-9",
        help="v4 default: 'all-9' (9-path matrix; ORACLE-SONNET dropped vs "
             "v3.3, ECO-DEBATE-QWEN added). Single-path names accept the "
             "dashed form (bare-qwen, ecology-sonnet, oracle-opus, "
             "eco-debate-qwen). Group names: all-9, bare, ecology, oracle, "
             "debate, v3-3-baseline. Legacy v3.3 strings (bare, ecology, "
             "both, all-with-oracles, model-ablation, ecology-sonnet, "
             "ecology-opus, bare-sonnet, bare-opus) are preserved for "
             "back-compat.")
    ap.add_argument("--focus", type=str, default="",
                    help="comma-separated focus tickers")
    ap.add_argument("--universe", choices=["dow30", "legacy"], default="dow30",
                    help="Ticker universe. 'dow30' (default for v4 sweep): "
                         "30 Dow Jones Industrial components — blue-chip "
                         "names where 5bps slippage and h60 holds are "
                         "realistic. 'legacy': v3.3 mixed 20-ticker list "
                         "(AAPL/ABBV/AMZN/AVGO/CSCO/CVX/GOOG/HD/JNJ/JPM/"
                         "KO/MA/MCD/MRK/MSFT/PEP/PG/UNH/V/WMT) — back-compat "
                         "for re-running v3.3-era sweeps.")
    ap.add_argument("--out", type=str, default=str(EVAL_OUT))
    ap.add_argument("--vllm-base", type=str, default="http://localhost:8001/v1",
                    help="vLLM OpenAI-compat base URL for the herbivore tier "
                         "(set empty string to disable and use in-process Qwen)")
    ap.add_argument("--apex-via-vllm", action="store_true",
                    help="Route apex calls through vLLM too (saves GPU mem; "
                         "avoids loading the model twice)")
    ap.add_argument("--decomposer-each-day", action="store_true",
                    help="Run the decomposer on each day's pass before moving "
                         "to the next; updated link strengths flow into next "
                         "day's eval. Online learning loop.")
    ap.add_argument("--no-lm-cache", action="store_true",
                    help="Disable DSPy LM-level cache so reruns hit the model "
                         "fresh. Use when measuring online-learning effects.")
    ap.add_argument("--herbivore-via-hexis", action="store_true",
                    help="Route the event_classifier herbivore through the "
                         "deployed Hexis vLLM (per-species Mind Tree session) "
                         "instead of the bare local vLLM. Apex stays on the "
                         "bare path so the only variable in A/B is the "
                         "herbivore. Requires HEXIS_API_URL or default "
                         "(hexis-agentic-a100 modal deploy).")
    ap.add_argument("--no-event-classifier-cache", action="store_true",
                    help="Bypass the article-level disk cache at "
                         "external/event_classifier_cache/ so each article "
                         "re-classification hits the LM fresh. Critical when "
                         "A/B-testing different LM backends on the same days.")
    ap.add_argument("--apex-pm", action="store_true",
                    help="Run apex as a PORTFOLIO MANAGER: per-pipeline "
                         "$100K accounts, blind to each other; apex sees "
                         "cash/equity/positions and emits sized orders "
                         "(percent of equity). Settled at end-of-day on "
                         "today's actual_return. Each run starts fresh.")
    ap.add_argument("--starting-cash", type=float, default=100_000.0,
                    help="Starting cash for each PM portfolio (default $100K).")
    ap.add_argument("--slippage-bps", type=float, default=5.0,
                    help="Per-trade slippage in basis points (PM mode).")
    ap.add_argument("--tax-rate", type=float, default=0.37,
                    help="Short-term capital gains rate on realized gains "
                         "(PM mode). Set 0.0 to disable.")
    ap.add_argument("--apex-pm-retries", type=int, default=2,
                    help="When validator rejects orders, send rejection "
                         "feedback to apex and re-ask up to this many "
                         "times. Default 2 (so 1 initial + up to 2 retries "
                         "= 3 calls max on bad days). Set 0 to disable retry.")
    ap.add_argument("--cash-yield", type=float, default=0.04,
                    help="Annualized risk-free yield on idle cash (PM mode). "
                         "Default 0.04 (~3-month T-bill). Set 0.0 to disable.")
    ap.add_argument("--async-paths", action="store_true",
                    help="Dispatch all per-day apex paths concurrently via "
                         "asyncio.gather + dspy.asyncify. Saves ~50%% wall "
                         "time on the 9-way model-ablation sweep. The "
                         "ecology stack still runs sync (Neo4j ordering "
                         "matters); only the apex calls fan out.")
    # v4 toggles + sweep-log discoverability flags
    ap.add_argument("--use-interrogator", dest="use_interrogator",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Enable the PMInterrogator on ECO-* paths (default ON). "
                         "Each ECO day's carnivore observations get an extra "
                         "interrogator-synthesis observation appended before "
                         "the apex sees them. --no-use-interrogator disables. "
                         "Ignored for BARE/ORACLE paths.")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="Per-call max_tokens for the apex vLLM LM. Default 4096.")
    ap.add_argument("--max-days", type=int, default=None,
                    help="Alias for --limit; v4 sweep-script-friendly name. "
                         "Wins over --limit when both are set.")
    ap.add_argument("--output", type=str, default=None,
                    help="Optional path to tee the sweep log to. The runner "
                         "still writes its eval JSONL into `--out` regardless.")
    ap.add_argument("--run-label", type=str, default=None,
                    help="Sweep label (e.g. `run_2026-05-10_pm_v4`); flows "
                         "into the debate transcript filename and the "
                         "eval-JSONL run header. Default is auto-generated.")
    ap.add_argument("--use-strategy-committee", action="store_true",
                    help="No-op v4 toggle (strategy committee is always ON "
                         "in v4; flag preserved for sweep-log discoverability).")
    ap.add_argument("--use-tax-aware-validators", action="store_true",
                    help="No-op v4 toggle (tax-aware validators are always "
                         "ON in v4; flag preserved for sweep-log discoverability).")
    ap.add_argument("--philosophy-weights", type=str,
                    default=str(_PHILOSOPHY_WEIGHTS_PATH),
                    help="Path to philosophy_weights.yaml. Default points "
                         "at tasks_v4/philosophy_weights.yaml.")
    ap.add_argument("--debate-predators", type=int, default=4,
                    help="Number of predators in the ECO-DEBATE-QWEN path. "
                         "Default 4 (momentum/value/mean_revert/event_driven). "
                         "Currently only 4 is supported; flag is reserved "
                         "for future N-predator variants.")
    args = ap.parse_args()
    # Honor --max-days as alias for --limit (v4 sweep-script convention)
    if args.max_days is not None:
        args.limit = args.max_days

    # Universe selection. Mutates the module-level UNIVERSE so all
    # downstream references (carnivore filters, news filters, matrix
    # builder, ticker-belief reset) read the chosen list. Done once
    # before any pipeline construction.
    global UNIVERSE
    if args.universe == "dow30":
        UNIVERSE = list(DOW30)
    elif args.universe == "legacy":
        UNIVERSE = list(LEGACY_UNIVERSE)
    else:
        # argparse `choices=` should prevent this, but defensive.
        raise SystemExit(f"unknown --universe value: {args.universe!r}")

    # Wire --philosophy-weights through to the committee cache.
    if args.philosophy_weights:
        set_philosophy_weights_path(Path(args.philosophy_weights))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    focus_tickers = [t.strip().upper() for t in args.focus.split(",") if t.strip()]
    # Drop focus tickers that aren't in the active universe — they'd
    # silently produce nothing downstream anyway.
    _univ_set = set(UNIVERSE)
    dropped_focus = [t for t in focus_tickers if t not in _univ_set]
    focus_tickers = [t for t in focus_tickers if t in _univ_set]
    if dropped_focus:
        print(f"  [universe] dropped focus tickers not in {args.universe}: "
              f"{','.join(dropped_focus)}")
    pass_persister = PassPersister(out_dir, model_id=ModelConfig().chat_model_id)

    print("Firehose loop runner")
    print(f"path: {args.path}  focus: {focus_tickers}")
    print(f"universe: {args.universe} ({len(UNIVERSE)} tickers)")
    hr()

    rows = load_dataset()
    if args.days:
        wanted = set(args.days.split(","))
        rows = [r for r in rows if r["date"] in wanted]
    elif args.limit:
        rows = rows[: args.limit]
    print(f"running on {len(rows)} days from {rows[0]['date']} to {rows[-1]['date']}")

    cfg = ModelConfig()
    vllm_up = args.vllm_base.strip() and vllm_health_check(args.vllm_base)

    use_cache = not args.no_lm_cache
    if not use_cache:
        print("LM cache disabled (--no-lm-cache)")

    if args.apex_via_vllm and vllm_up:
        # Both tiers via vLLM. Avoids loading Qwen weights twice.
        print(f"using vLLM for BOTH apex and herbivore at {args.vllm_base}")
        vllm_lm = make_vllm_lm(
            model_name=cfg.chat_model_id,
            api_base=args.vllm_base,
            max_tokens=4096,
            temperature=0.0,
            cache=use_cache,
        )
        apex_lm = vllm_lm
        herb_lm = vllm_lm
        host = None
    else:
        print("loading in-process Qwen for apex...")
        host = ModelHost(cfg)
        apex_lm = QwenLocalLM(host, max_tokens=2048, temperature=0.0, cache=use_cache)
        if vllm_up:
            herb_lm = make_vllm_lm(
                model_name=cfg.chat_model_id,
                api_base=args.vllm_base,
                max_tokens=1024,
                temperature=0.0,
                cache=use_cache,
            )
            print(f"  herbivore LM: vLLM at {args.vllm_base}  ({cfg.chat_model_id})")
        else:
            herb_lm = apex_lm
            if args.vllm_base.strip():
                print(f"  vLLM unavailable at {args.vllm_base}; falling back to in-process")
            else:
                print(f"  herbivore LM: in-process Qwen (vllm disabled)")
        print(f"  apex LM: in-process Qwen ({cfg.chat_model_id})")

    # Optional: swap the herbivore LM to a per-species Hexis session.
    # Apex stays on the bare path so the only variable in this A/B is the
    # herbivore tier. Each species gets its own session_id from
    # SpeciesSessionManager (cached at data/hexis_sessions/{species_id}.json).
    if args.herbivore_via_hexis:
        from trophic.beliefs.hexis_client import (
            HexisClient, SpeciesSessionManager,
        )
        from trophic.beliefs.dspy_hexis_lm import (
            make_hexis_lm, hexis_health_check,
        )
        if not hexis_health_check():
            print("ERROR: --herbivore-via-hexis requested but Hexis deploy not healthy")
            sys.exit(2)
        hexis_client = HexisClient()
        species_mgr = SpeciesSessionManager(hexis_client)
        sid = species_mgr.get_or_create(
            species_id="event_classifier.v0",
            species_description=(
                "Event-classifier herbivore — DSPy ClassifyArticle Signature. "
                "Reads financial news articles, emits BeliefActivations per "
                "(article, ticker, template) on a 20-ticker universe."
            ),
            preload_node_ids=["root:finance:event_classifier.v0"],
        )
        herb_lm = make_hexis_lm(
            session_id=sid,
            api_base=hexis_client.base_url,
            model_name=cfg.chat_model_id,
            max_tokens=1024,
            temperature=0.0,
            cache=use_cache,
        )
        print(f"  herbivore LM: Hexis vLLM (session_id={sid}, "
              f"base={hexis_client.base_url})")

    dspy.configure(lm=apex_lm, adapter=dspy.JSONAdapter())
    hr()

    # Path participation — which pipelines run this sweep.
    #
    # `args.path` may be a v4 group name (all-9, bare, ecology, oracle,
    # debate, v3-3-baseline) OR a single dashed path (bare-qwen,
    # eco-debate-qwen, …). Legacy v3.3 strings (bare, ecology, both,
    # all-with-oracles, model-ablation, ecology-sonnet, …) resolve to
    # their pre-v4 path sets.
    LEGACY_PATH_GROUPS: dict[str, tuple[str, ...]] = {
        "bare":             ("bare-qwen",),
        "ecology":          ("ecology-qwen",),
        "both":             ("bare-qwen", "ecology-qwen"),
        "all-with-oracles": (
            "bare-qwen", "ecology-qwen",
            "oracle-qwen", "oracle-sonnet",
        ),
        # v3.3 9-way model-ablation matrix
        "model-ablation":   V3_3_BASELINE_PATHS,
        # Single-path legacy aliases
        "ecology-sonnet":   ("ecology-sonnet",),
        "ecology-opus":     ("ecology-opus",),
        "bare-sonnet":      ("bare-sonnet",),
        "bare-opus":        ("bare-opus",),
    }
    if args.path in PATH_GROUPS:
        active_paths: set[str] = set(PATH_GROUPS[args.path])
    elif args.path in LEGACY_PATH_GROUPS:
        active_paths = set(LEGACY_PATH_GROUPS[args.path])
    elif args.path in ALL_PATHS_V4 or args.path == "oracle-sonnet":
        active_paths = {args.path}
    else:
        raise SystemExit(f"unknown --path value: {args.path!r}")

    runs_bare           = "bare-qwen"        in active_paths
    runs_bare_sonnet    = "bare-sonnet"      in active_paths
    runs_bare_opus      = "bare-opus"        in active_paths
    # runs_ecology = ecology pipeline with the QWEN apex specifically
    runs_ecology        = "ecology-qwen"     in active_paths
    runs_ecology_sonnet = "ecology-sonnet"   in active_paths
    runs_ecology_opus   = "ecology-opus"     in active_paths
    runs_oracle_qwen    = "oracle-qwen"      in active_paths
    runs_oracle_sonnet  = "oracle-sonnet"    in active_paths
    runs_oracle_opus    = "oracle-opus"      in active_paths
    runs_debate_qwen    = "eco-debate-qwen"  in active_paths
    # The ecology STACK (herbivore + carnivore) is needed by any ECO-*
    # path (including the debate path, which feeds the apex observations
    # into the debate inputs).
    needs_ecology_stack = (
        runs_ecology or runs_ecology_sonnet or runs_ecology_opus
        or runs_debate_qwen
    )

    print(f"  active paths ({len(active_paths)}): "
          f"{','.join(sorted(active_paths))}")

    if args.apex_pm:
        print(f"Apex-PM mode ON  starting_cash=${args.starting_cash:,.0f} "
              f"slippage={args.slippage_bps}bps tax={args.tax_rate:.0%} "
              f"cash_yield={args.cash_yield:.1%}/yr "
              f"retries={args.apex_pm_retries}")
        apex_firehose = dspy.Predict(WatchlistFromFirehosePM) if runs_bare else None
        apex_obs      = dspy.Predict(WatchlistFromObservationsPM) if runs_ecology else None
        # Frontier-apex variants reuse the same signatures, just route
        # through dspy.context(lm=…) per call.
        apex_obs_sonnet = dspy.Predict(WatchlistFromObservationsPM) if runs_ecology_sonnet else None
        apex_obs_opus   = dspy.Predict(WatchlistFromObservationsPM) if runs_ecology_opus else None
        apex_firehose_sonnet = dspy.Predict(WatchlistFromFirehosePM) if runs_bare_sonnet else None
        apex_firehose_opus   = dspy.Predict(WatchlistFromFirehosePM) if runs_bare_opus else None
        # v4: full-window Oracle (multi-day forward-returns matrix). The
        # legacy per-day `WatchlistOracle` is deprecated.
        apex_oracle_qwen   = dspy.Predict(WatchlistOracleFullWindow) if runs_oracle_qwen else None
        apex_oracle_sonnet = dspy.Predict(WatchlistOracleFullWindow) if runs_oracle_sonnet else None
        apex_oracle_opus   = dspy.Predict(WatchlistOracleFullWindow) if runs_oracle_opus else None
    else:
        apex_firehose = dspy.Predict(WatchlistFromFirehose) if runs_bare else None
        apex_obs      = dspy.Predict(WatchlistFromObservations) if runs_ecology else None
        apex_obs_sonnet = None
        apex_obs_opus = None
        apex_firehose_sonnet = None
        apex_firehose_opus = None
        apex_oracle_qwen = None
        apex_oracle_sonnet = None
        apex_oracle_opus = None
        if any([runs_oracle_qwen, runs_oracle_sonnet, runs_oracle_opus,
                runs_ecology_sonnet, runs_ecology_opus,
                runs_bare_sonnet, runs_bare_opus]):
            print("Frontier/oracle paths require --apex-pm; ignoring")
            runs_oracle_qwen = runs_oracle_sonnet = runs_oracle_opus = False
            runs_ecology_sonnet = runs_ecology_opus = False
            runs_bare_sonnet = runs_bare_opus = False

    # Bedrock Sonnet/Opus LMs — shared across all paths that need them
    sonnet_lm = None
    opus_lm = None
    needs_sonnet = runs_oracle_sonnet or runs_ecology_sonnet or runs_bare_sonnet
    needs_opus = runs_oracle_opus or runs_ecology_opus or runs_bare_opus
    if needs_sonnet:
        from trophic.beliefs.dspy_bedrock_lm import make_bedrock_sonnet_lm
        sonnet_lm = make_bedrock_sonnet_lm(
            temperature=0.0, max_tokens=4096, cache=use_cache,
        )
        print(f"  Sonnet LM: Bedrock {os.environ.get('ORACLE_SONNET_MODEL_ID', 'us.anthropic.claude-sonnet-4-6')}")
    if needs_opus:
        from trophic.beliefs.dspy_bedrock_lm import make_bedrock_opus_lm
        opus_lm = make_bedrock_opus_lm(
            temperature=0.0, max_tokens=4096, cache=use_cache,
        )
        print(f"  Opus LM: Bedrock {os.environ.get('ORACLE_OPUS_MODEL_ID', 'us.anthropic.claude-opus-4-6-v1')}")

    # Per-pipeline portfolios. Blind to each other.
    eco_portfolio: ApexPortfolio | None = None
    bare_portfolio: ApexPortfolio | None = None
    eco_sonnet_portfolio: ApexPortfolio | None = None
    eco_opus_portfolio: ApexPortfolio | None = None
    bare_sonnet_portfolio: ApexPortfolio | None = None
    bare_opus_portfolio: ApexPortfolio | None = None
    oracle_qwen_portfolio: ApexPortfolio | None = None
    oracle_sonnet_portfolio: ApexPortfolio | None = None
    oracle_opus_portfolio: ApexPortfolio | None = None
    debate_qwen_portfolio: ApexPortfolio | None = None
    if args.apex_pm:
        if runs_ecology:
            eco_portfolio = ApexPortfolio(
                label="ecology",
                starting_cash=args.starting_cash,
                slippage_bps=args.slippage_bps,
                tax_rate=args.tax_rate,
                cash_yield_annual=args.cash_yield,
            )
        if runs_bare:
            bare_portfolio = ApexPortfolio(
                label="bare",
                starting_cash=args.starting_cash,
                slippage_bps=args.slippage_bps,
                tax_rate=args.tax_rate,
                cash_yield_annual=args.cash_yield,
            )
        if runs_oracle_qwen:
            oracle_qwen_portfolio = ApexPortfolio(
                label="oracle_qwen",
                starting_cash=args.starting_cash,
                slippage_bps=args.slippage_bps,
                tax_rate=args.tax_rate,
                cash_yield_annual=args.cash_yield,
            )
        if runs_oracle_sonnet:
            oracle_sonnet_portfolio = ApexPortfolio(
                label="oracle_sonnet",
                starting_cash=args.starting_cash,
                slippage_bps=args.slippage_bps,
                tax_rate=args.tax_rate,
                cash_yield_annual=args.cash_yield,
            )
        if runs_ecology_sonnet:
            eco_sonnet_portfolio = ApexPortfolio(
                label="ecology_sonnet",
                starting_cash=args.starting_cash,
                slippage_bps=args.slippage_bps,
                tax_rate=args.tax_rate,
                cash_yield_annual=args.cash_yield,
            )
        if runs_ecology_opus:
            eco_opus_portfolio = ApexPortfolio(
                label="ecology_opus",
                starting_cash=args.starting_cash,
                slippage_bps=args.slippage_bps,
                tax_rate=args.tax_rate,
                cash_yield_annual=args.cash_yield,
            )
        if runs_bare_sonnet:
            bare_sonnet_portfolio = ApexPortfolio(
                label="bare_sonnet",
                starting_cash=args.starting_cash,
                slippage_bps=args.slippage_bps,
                tax_rate=args.tax_rate,
                cash_yield_annual=args.cash_yield,
            )
        if runs_bare_opus:
            bare_opus_portfolio = ApexPortfolio(
                label="bare_opus",
                starting_cash=args.starting_cash,
                slippage_bps=args.slippage_bps,
                tax_rate=args.tax_rate,
                cash_yield_annual=args.cash_yield,
            )
        if runs_oracle_opus:
            oracle_opus_portfolio = ApexPortfolio(
                label="oracle_opus",
                starting_cash=args.starting_cash,
                slippage_bps=args.slippage_bps,
                tax_rate=args.tax_rate,
                cash_yield_annual=args.cash_yield,
            )
        if runs_debate_qwen:
            # ECO-DEBATE-QWEN: 4 predators × $25K each (starting cash
            # split evenly). Parent ApexPortfolio holds the price index
            # and cash-yield accrual; per-predator capital lives in
            # `sub_portfolios`. Per-predator tax isolation is enforced
            # by PredatorSubPortfolio.sell.
            debate_qwen_portfolio = make_debate_portfolio(
                total_starting_cash=args.starting_cash,
                slippage_bps=args.slippage_bps,
                tax_rate=args.tax_rate,
                cash_yield_annual=args.cash_yield,
            )
            debate_qwen_portfolio.label = "eco_debate_qwen"
            print(
                f"  ECO-DEBATE-QWEN: {len(debate_qwen_portfolio.sub_portfolios)} "
                f"predators @ ${args.starting_cash / len(debate_qwen_portfolio.sub_portfolios):,.0f} each"
            )

    # PMInterrogator — built once per run, used by ECO-* paths (Qwen,
    # Sonnet, Opus, and the debate path). For BARE/ORACLE paths the
    # interrogator is unused regardless of --use-interrogator. If the
    # interrogator can't be constructed (e.g. vLLM is the planner LM
    # and is unhealthy), fall back to a no-op interrogator with a
    # warning so the sweep continues.
    needs_interrogator = (
        args.use_interrogator
        and (runs_ecology or runs_ecology_sonnet or runs_ecology_opus
             or runs_debate_qwen)
    )
    if needs_interrogator:
        try:
            interrogator = make_pm_interrogator(enabled=True)
            print("  PMInterrogator: ENABLED (planner + synthesizer on apex_lm; "
                  "math step uses the active LM)")
        except Exception as e:
            print(f"  [interrogator] WARN: factory failed: {e}; "
                  f"falling back to no-op interrogator")
            interrogator = make_pm_interrogator(enabled=False)
    else:
        interrogator = make_pm_interrogator(enabled=False)
        if args.use_interrogator and not (
            runs_ecology or runs_ecology_sonnet or runs_ecology_opus
            or runs_debate_qwen
        ):
            print("  PMInterrogator: not needed (no ECO-* path active)")
        elif not args.use_interrogator:
            print("  PMInterrogator: disabled via --no-use-interrogator")

    # Ecology setup (herbivore + carnivore — runs once, feeds all ecology apex variants)
    store = None; state = None; links = None
    xcorr = None; event_herb = None
    if needs_ecology_stack:
        print("loading belief network from Neo4j...")
        store = Neo4jBeliefStore()
        state_list = store.all_state_beliefs()
        state = {b.id: b for b in state_list}
        links = store.all_links()
        print(f"  {len(state)} state beliefs, {len(links)} links")

        # Cross-correlation: build bars-through-end-of-dataset state once
        # (correlation of returns is independent across days; we use the
        # same matrix throughout for the eval to keep it deterministic.
        # In production this would be updated daily.)
        all_rows = load_dataset()
        bars_all = reconstruct_bars_through(all_rows, all_rows[-1]["date"])
        xcorr = CrossCorrelationHerbivore.with_state(UNIVERSE, XCORR_STATE)
        update_state_from_bars(xcorr.state, bars_all, all_rows[-1]["date"], xcorr.cfg)
        print(f"  cross-correlation: {xcorr.state.n_observations} obs in matrix")

        event_herb = EventClassifierHerbivore(
            universe=UNIVERSE, cache_dir=EVENT_CACHE,
            use_disk_cache=not args.no_event_classifier_cache,
        )
        cache_status = "disabled" if args.no_event_classifier_cache else f"enabled at {EVENT_CACHE}"
        print(f"  event classifier ready (cache={cache_status})")
    hr()

    out_lines = []
    for i, row in enumerate(rows):
        date = row["date"]
        print(f"\n[{i+1}/{len(rows)}] {date}  "
              f"news={row['n_news']}  mentioned={row['n_mentioned']}/20")
        record = {
            "date": date,
            "prev_trading_day": row["prev_trading_day"],
            "targets": row["targets"],
            "focus_tickers": focus_tickers,
            "n_news": row["n_news"],
        }

        eco_obs_dicts: list[dict] = []
        if needs_ecology_stack:
            t0 = time.time()
            # If runs_ecology, do stack+Qwen-apex+portfolio in one shot.
            # Otherwise just compute the stack so we can feed the obs to
            # the frontier-apex variants.
            eco = run_ecology_pass(
                row, store, state, links, xcorr, event_herb,
                apex_obs if runs_ecology else None,
                focus_tickers,
                pass_persister=pass_persister,
                event_herb_lm=(herb_lm if herb_lm is not apex_lm else None),
                portfolio=eco_portfolio if runs_ecology else None,
                pm_retries=(args.apex_pm_retries if eco_portfolio else 0),
                interrogator=interrogator,
            )
            eco["elapsed_s"] = round(time.time() - t0, 2)
            eco_obs_dicts = eco.get("observations", [])
            if runs_ecology:
                record["ecology"] = eco
                pass_persister.add_watchlist("ecology", eco["watchlist"])
                print(f"  ecology(qwen): {eco['n_event_activations']} event acts, "
                      f"{eco['n_xcorr_cluster_activations']} cluster, "
                      f"{eco['n_xcorr_sympathy_activations']} sympathy → "
                      f"{eco['n_observations']} obs → {len(eco['watchlist'])} watchlist  "
                      f"({eco['elapsed_s']}s)")
                if eco_portfolio is not None:
                    snap = eco["pm_snapshot_post"]
                    print(f"    PM: equity=${snap['equity']:,.0f} "
                          f"cash=${snap['cash']:,.0f} inv={snap['invested_pct']:.0f}% "
                          f"orders={len(eco.get('pm_orders_validated', []))} "
                          f"rejected={len(eco.get('pm_orders_rejected', []))}")
            else:
                # Stack-only: just print obs count
                print(f"  ecology stack only: {eco['n_observations']} obs "
                      f"({eco['elapsed_s']}s)")

        # ── ASYNC DISPATCH (--async-paths) ────────────────────────────
        # Fan out all remaining apex paths concurrently. Ecology(qwen)
        # already ran inline above as part of the stack call. The other
        # 8 paths are pure apex-and-portfolio work and run in threads.
        if args.async_paths:
            import asyncio

            async def _dispatch():
                tasks = []
                # Each entry: (name, async-coro). Build the lambdas with
                # default-arg captures to dodge late-binding gotchas.
                if runs_ecology_sonnet and eco_sonnet_portfolio is not None:
                    tasks.append(("ecology_sonnet", _async_run_with_lm(
                        lambda obs=eco_obs_dicts: run_ecology_apex_only(
                            row, obs, apex_obs_sonnet, focus_tickers,
                            eco_sonnet_portfolio, args.apex_pm_retries,
                            label="ecology_sonnet",
                        ),
                        lm=sonnet_lm,
                    )))
                if runs_ecology_opus and eco_opus_portfolio is not None:
                    tasks.append(("ecology_opus", _async_run_with_lm(
                        lambda obs=eco_obs_dicts: run_ecology_apex_only(
                            row, obs, apex_obs_opus, focus_tickers,
                            eco_opus_portfolio, args.apex_pm_retries,
                            label="ecology_opus",
                        ),
                        lm=opus_lm,
                    )))
                if runs_bare:
                    tasks.append(("bare", _async_run_with_lm(
                        lambda: run_bare_pass(
                            row, apex_firehose, focus_tickers,
                            portfolio=bare_portfolio,
                            pm_retries=(args.apex_pm_retries if bare_portfolio else 0),
                        ),
                    )))
                if runs_bare_sonnet and bare_sonnet_portfolio is not None:
                    tasks.append(("bare_sonnet", _async_run_with_lm(
                        lambda: run_bare_pass(
                            row, apex_firehose_sonnet, focus_tickers,
                            portfolio=bare_sonnet_portfolio,
                            pm_retries=args.apex_pm_retries,
                        ),
                        lm=sonnet_lm,
                    )))
                if runs_bare_opus and bare_opus_portfolio is not None:
                    tasks.append(("bare_opus", _async_run_with_lm(
                        lambda: run_bare_pass(
                            row, apex_firehose_opus, focus_tickers,
                            portfolio=bare_opus_portfolio,
                            pm_retries=args.apex_pm_retries,
                        ),
                        lm=opus_lm,
                    )))
                if runs_oracle_qwen and oracle_qwen_portfolio is not None:
                    tasks.append(("oracle_qwen", _async_run_with_lm(
                        lambda: run_oracle_pass(
                            row, apex_oracle_qwen, focus_tickers,
                            portfolio=oracle_qwen_portfolio,
                            today_idx=i, rows=rows,
                            pm_retries=args.apex_pm_retries,
                            label="oracle_qwen",
                        ),
                    )))
                if runs_oracle_sonnet and oracle_sonnet_portfolio is not None:
                    tasks.append(("oracle_sonnet", _async_run_with_lm(
                        lambda: run_oracle_pass(
                            row, apex_oracle_sonnet, focus_tickers,
                            portfolio=oracle_sonnet_portfolio,
                            today_idx=i, rows=rows,
                            pm_retries=args.apex_pm_retries,
                            label="oracle_sonnet",
                        ),
                        lm=sonnet_lm,
                    )))
                if runs_oracle_opus and oracle_opus_portfolio is not None:
                    tasks.append(("oracle_opus", _async_run_with_lm(
                        lambda: run_oracle_pass(
                            row, apex_oracle_opus, focus_tickers,
                            portfolio=oracle_opus_portfolio,
                            today_idx=i, rows=rows,
                            pm_retries=args.apex_pm_retries,
                            label="oracle_opus",
                        ),
                        lm=opus_lm,
                    )))
                if runs_debate_qwen and debate_qwen_portfolio is not None:
                    # Debate is async-native (`run_debate` returns a
                    # coroutine). Awaiting `_run_debate_pass_async`
                    # directly avoids nesting asyncio.run.
                    tasks.append(("eco_debate_qwen",
                        _run_debate_pass_async(
                            row, debate_qwen_portfolio,
                            today_idx=i, rows=rows,
                            obs_dicts=eco_obs_dicts,
                            interrogator=interrogator,
                            focus_tickers=focus_tickers,
                            lm=apex_lm,
                            run_label=(args.run_label or "eco_debate_qwen"),
                            pm_retries=args.apex_pm_retries,
                        )
                    ))
                names = [n for n, _ in tasks]
                results = await asyncio.gather(
                    *(t for _, t in tasks),
                    return_exceptions=True,
                )
                return dict(zip(names, results))

            t_async = time.time()
            results = asyncio.run(_dispatch())
            elapsed_async = round(time.time() - t_async, 2)

            for name, result in results.items():
                if isinstance(result, Exception):
                    print(f"  [{name} async error] {type(result).__name__}: {result}")
                    continue
                result["elapsed_s"] = elapsed_async  # shared parallel time
                record[name] = result
                pass_persister.add_watchlist(name, result["watchlist"])
                snap = result["pm_snapshot_post"]
                pretty = name.replace("_", "(").replace("_", " ") + ")"
                print(f"  {pretty}: equity=${snap['equity']:,.0f} "
                      f"cash=${snap['cash']:,.0f} inv={snap['invested_pct']:.0f}% "
                      f"orders={len(result.get('pm_orders_validated', []))} "
                      f"rejected={len(result.get('pm_orders_rejected', []))}")
            print(f"  [async] 8 paths in {elapsed_async}s (parallel)")

        # ECOLOGY-Sonnet — reuse obs_dicts, swap apex LM via dspy.context
        if not args.async_paths and runs_ecology_sonnet and eco_sonnet_portfolio is not None:
            t0 = time.time()
            with dspy.context(lm=sonnet_lm):
                eco_s = run_ecology_apex_only(
                    row, eco_obs_dicts, apex_obs_sonnet, focus_tickers,
                    eco_sonnet_portfolio, args.apex_pm_retries,
                    label="ecology_sonnet",
                )
            eco_s["elapsed_s"] = round(time.time() - t0, 2)
            record["ecology_sonnet"] = eco_s
            pass_persister.add_watchlist("ecology_sonnet", eco_s["watchlist"])
            snap = eco_s["pm_snapshot_post"]
            print(f"  ecology(sonnet): ({eco_s['elapsed_s']}s)")
            print(f"    PM: equity=${snap['equity']:,.0f} "
                  f"cash=${snap['cash']:,.0f} inv={snap['invested_pct']:.0f}% "
                  f"orders={len(eco_s.get('pm_orders_validated', []))} "
                  f"rejected={len(eco_s.get('pm_orders_rejected', []))}")

        # ECOLOGY-Opus — same pattern
        if not args.async_paths and runs_ecology_opus and eco_opus_portfolio is not None:
            t0 = time.time()
            with dspy.context(lm=opus_lm):
                eco_o = run_ecology_apex_only(
                    row, eco_obs_dicts, apex_obs_opus, focus_tickers,
                    eco_opus_portfolio, args.apex_pm_retries,
                    label="ecology_opus",
                )
            eco_o["elapsed_s"] = round(time.time() - t0, 2)
            record["ecology_opus"] = eco_o
            pass_persister.add_watchlist("ecology_opus", eco_o["watchlist"])
            snap = eco_o["pm_snapshot_post"]
            print(f"  ecology(opus): ({eco_o['elapsed_s']}s)")
            print(f"    PM: equity=${snap['equity']:,.0f} "
                  f"cash=${snap['cash']:,.0f} inv={snap['invested_pct']:.0f}% "
                  f"orders={len(eco_o.get('pm_orders_validated', []))} "
                  f"rejected={len(eco_o.get('pm_orders_rejected', []))}")

        if not args.async_paths and runs_bare:
            t0 = time.time()
            bare = run_bare_pass(
                row, apex_firehose, focus_tickers,
                portfolio=bare_portfolio,
                pm_retries=(args.apex_pm_retries if bare_portfolio else 0),
            )
            bare["elapsed_s"] = round(time.time() - t0, 2)
            record["bare"] = bare
            pass_persister.add_watchlist("bare", bare["watchlist"])
            print(f"  bare(qwen): {bare['n_news_used']} news → "
                  f"{len(bare['watchlist'])} watchlist  ({bare['elapsed_s']}s)")
            if bare_portfolio is not None:
                snap = bare["pm_snapshot_post"]
                print(f"    PM: equity=${snap['equity']:,.0f} "
                      f"cash=${snap['cash']:,.0f} inv={snap['invested_pct']:.0f}% "
                      f"orders={len(bare.get('pm_orders_validated', []))} "
                      f"rejected={len(bare.get('pm_orders_rejected', []))}")

        if not args.async_paths and runs_bare_sonnet and bare_sonnet_portfolio is not None:
            t0 = time.time()
            with dspy.context(lm=sonnet_lm):
                bs = run_bare_pass(
                    row, apex_firehose_sonnet, focus_tickers,
                    portfolio=bare_sonnet_portfolio,
                    pm_retries=args.apex_pm_retries,
                )
            bs["elapsed_s"] = round(time.time() - t0, 2)
            record["bare_sonnet"] = bs
            pass_persister.add_watchlist("bare_sonnet", bs["watchlist"])
            snap = bs["pm_snapshot_post"]
            print(f"  bare(sonnet): ({bs['elapsed_s']}s)")
            print(f"    PM: equity=${snap['equity']:,.0f} "
                  f"cash=${snap['cash']:,.0f} inv={snap['invested_pct']:.0f}% "
                  f"orders={len(bs.get('pm_orders_validated', []))} "
                  f"rejected={len(bs.get('pm_orders_rejected', []))}")

        if not args.async_paths and runs_bare_opus and bare_opus_portfolio is not None:
            t0 = time.time()
            with dspy.context(lm=opus_lm):
                bo = run_bare_pass(
                    row, apex_firehose_opus, focus_tickers,
                    portfolio=bare_opus_portfolio,
                    pm_retries=args.apex_pm_retries,
                )
            bo["elapsed_s"] = round(time.time() - t0, 2)
            record["bare_opus"] = bo
            pass_persister.add_watchlist("bare_opus", bo["watchlist"])
            snap = bo["pm_snapshot_post"]
            print(f"  bare(opus): ({bo['elapsed_s']}s)")
            print(f"    PM: equity=${snap['equity']:,.0f} "
                  f"cash=${snap['cash']:,.0f} inv={snap['invested_pct']:.0f}% "
                  f"orders={len(bo.get('pm_orders_validated', []))} "
                  f"rejected={len(bo.get('pm_orders_rejected', []))}")

        if not args.async_paths and runs_oracle_qwen and oracle_qwen_portfolio is not None:
            t0 = time.time()
            oq = run_oracle_pass(
                row, apex_oracle_qwen, focus_tickers,
                portfolio=oracle_qwen_portfolio,
                today_idx=i, rows=rows,
                pm_retries=args.apex_pm_retries,
                label="oracle_qwen",
            )
            oq["elapsed_s"] = round(time.time() - t0, 2)
            record["oracle_qwen"] = oq
            pass_persister.add_watchlist("oracle_qwen", oq["watchlist"])
            snap = oq["pm_snapshot_post"]
            print(f"  oracle_qwen:  ({oq['elapsed_s']}s)")
            print(f"    PM: equity=${snap['equity']:,.0f} "
                  f"cash=${snap['cash']:,.0f} inv={snap['invested_pct']:.0f}% "
                  f"orders={len(oq.get('pm_orders_validated', []))} "
                  f"rejected={len(oq.get('pm_orders_rejected', []))}")

        if not args.async_paths and runs_oracle_sonnet and oracle_sonnet_portfolio is not None:
            t0 = time.time()
            with dspy.context(lm=sonnet_lm):
                osn = run_oracle_pass(
                    row, apex_oracle_sonnet, focus_tickers,
                    portfolio=oracle_sonnet_portfolio,
                    today_idx=i, rows=rows,
                    pm_retries=args.apex_pm_retries,
                    label="oracle_sonnet",
                )
            osn["elapsed_s"] = round(time.time() - t0, 2)
            record["oracle_sonnet"] = osn
            pass_persister.add_watchlist("oracle_sonnet", osn["watchlist"])
            snap = osn["pm_snapshot_post"]
            print(f"  oracle_sonnet: ({osn['elapsed_s']}s)")
            print(f"    PM: equity=${snap['equity']:,.0f} "
                  f"cash=${snap['cash']:,.0f} inv={snap['invested_pct']:.0f}% "
                  f"orders={len(osn.get('pm_orders_validated', []))} "
                  f"rejected={len(osn.get('pm_orders_rejected', []))}")

        if not args.async_paths and runs_oracle_opus and oracle_opus_portfolio is not None:
            t0 = time.time()
            with dspy.context(lm=opus_lm):
                oo = run_oracle_pass(
                    row, apex_oracle_opus, focus_tickers,
                    portfolio=oracle_opus_portfolio,
                    today_idx=i, rows=rows,
                    pm_retries=args.apex_pm_retries,
                    label="oracle_opus",
                )
            oo["elapsed_s"] = round(time.time() - t0, 2)
            record["oracle_opus"] = oo
            pass_persister.add_watchlist("oracle_opus", oo["watchlist"])
            snap = oo["pm_snapshot_post"]
            print(f"  oracle_opus: ({oo['elapsed_s']}s)")
            print(f"    PM: equity=${snap['equity']:,.0f} "
                  f"cash=${snap['cash']:,.0f} inv={snap['invested_pct']:.0f}% "
                  f"orders={len(oo.get('pm_orders_validated', []))} "
                  f"rejected={len(oo.get('pm_orders_rejected', []))}")

        # ECO-DEBATE-QWEN (v4 NEW) — 4-predator debate ecology, local Qwen-4B.
        # Runs sync here when --async-paths is off; the async dispatcher
        # above handles the concurrent case.
        if not args.async_paths and runs_debate_qwen and debate_qwen_portfolio is not None:
            t0 = time.time()
            dq = run_debate_pass(
                row, debate_qwen_portfolio,
                today_idx=i, rows=rows,
                obs_dicts=eco_obs_dicts,
                interrogator=interrogator,
                focus_tickers=focus_tickers,
                lm=apex_lm,
                run_label=(args.run_label or "eco_debate_qwen"),
                pm_retries=args.apex_pm_retries,
            )
            dq["elapsed_s"] = round(time.time() - t0, 2)
            record["eco_debate_qwen"] = dq
            pass_persister.add_watchlist("eco_debate_qwen", dq["watchlist"])
            snap = dq["pm_snapshot_post"]
            print(f"  eco_debate_qwen: ({dq['elapsed_s']}s)")
            print(f"    PM: equity=${snap['equity']:,.0f} "
                  f"cash=${snap['cash']:,.0f} inv={snap['invested_pct']:.0f}% "
                  f"orders={len(dq.get('pm_orders_validated', []))} "
                  f"rejected={len(dq.get('pm_orders_rejected', []))}")
            for pid, pr in (dq.get("predator_breakdown") or {}).items():
                print(f"      {pid} ({pr.get('philosophy','?')}): "
                      f"eq=${pr['equity']:,.0f} "
                      f"valid={pr['n_validated']} rej={pr['n_rejected']} "
                      f"theses={pr['active_theses']}")

        out_lines.append(record)

        # Write the per-day Pass artifact (full causal trail for the
        # decomposer). Ground truth is the dataset's targets[] which
        # already carry actual_direction + magnitude_bucket.
        pass_persister.set_ground_truth(row["targets"])
        pass_path = pass_persister.write(
            date=date,
            prev_trading_day=row["prev_trading_day"],
            focus_tickers=focus_tickers,
            n_news=row["n_news"],
            elapsed_ecology_s=record.get("ecology", {}).get("elapsed_s"),
            elapsed_bare_s=record.get("bare", {}).get("elapsed_s"),
        )
        print(f"  pass artifact: {pass_path.name}")

        # Per-day decomposer interleave: process the just-written pass,
        # write link strength_posterior updates back to Neo4j, then
        # reload the in-memory `links` snapshot so the next day picks them up.
        if args.decomposer_each_day and needs_ecology_stack:
            from trophic.beliefs.decomposer_temporal import Decomposer
            from trophic.beliefs.pass_record import load_pass
            dec = Decomposer(root=ROOT, write_neo4j=True)
            rec_loaded = load_pass(pass_path)
            res = dec.process_one(rec_loaded)
            n_link_upd = res.get("neo4j_links_actually_updated", 0)
            print(f"  decomposer: hints={res.get('teacher_hints')}  "
                  f"link_updates={n_link_upd}  "
                  f"species={res.get('species_observed')}")
            # Refresh in-memory links snapshot so next day picks up new strengths
            if n_link_upd and store is not None and links is not None:
                links_new = store.all_links()
                links.clear()
                links.extend(links_new)
                print(f"  refreshed links snapshot ({len(links)})")

    # PM summary
    all_portfolios_named = [
        ("BARE-QWEN", bare_portfolio),
        ("BARE-SONNET", bare_sonnet_portfolio),
        ("BARE-OPUS", bare_opus_portfolio),
        ("ECOLOGY-QWEN", eco_portfolio),
        ("ECOLOGY-SONNET", eco_sonnet_portfolio),
        ("ECOLOGY-OPUS", eco_opus_portfolio),
        ("ORACLE-QWEN", oracle_qwen_portfolio),
        ("ORACLE-SONNET", oracle_sonnet_portfolio),
        ("ORACLE-OPUS", oracle_opus_portfolio),
        ("ECO-DEBATE-QWEN", debate_qwen_portfolio),
    ]
    if args.apex_pm and any(p for _, p in all_portfolios_named):
        hr()
        print("PORTFOLIO RESULTS")
        portfolios = all_portfolios_named
        for label, pf in portfolios:
            if pf is None:
                continue
            eq = pf.equity(pf._last_price)
            # In debate mode `pf.starting_cash` is 0 (capital lives in
            # sub_portfolios). Use the sum of sub-starting-cash instead.
            if pf.is_debate_mode():
                start = sum(s.starting_cash for s in pf.sub_portfolios.values())
                realized = sum(
                    sum(v for _, v in s.realized_pnl_history)
                    for s in pf.sub_portfolios.values()
                )
                tax_owed = pf.total_tax_owed_accrued
                n_buys = pf.n_buys  # parent-counted (currently 0 in debate)
                n_sells = pf.n_sells
            else:
                start = pf.starting_cash
                realized = pf.realized_gains_cum
                tax_owed = pf.tax_owed
                n_buys = pf.n_buys
                n_sells = pf.n_sells
            pnl = eq - start
            ret = pnl / start if start else 0.0
            print(f"  {label:<16} ${eq:>10,.0f}  pnl=${pnl:>+9,.0f}  "
                  f"return={ret:>+7.2%}  buys={n_buys}  sells={n_sells}  "
                  f"realized=${realized:>+8,.0f}  "
                  f"tax_owed=${tax_owed:>7,.0f}  "
                  f"yield_cum=${pf.cash_yield_cum:>5,.0f}")
            # In debate mode, also print per-predator leaderboard.
            if pf.is_debate_mode():
                for pid, sub in pf.sub_portfolios.items():
                    sub_eq = sub.equity(pf._last_price)
                    sub_pnl = sub_eq - sub.starting_cash
                    sub_ret = sub_pnl / sub.starting_cash if sub.starting_cash else 0.0
                    print(f"      {pid:<14} ({sub.philosophy:<13}) "
                          f"${sub_eq:>9,.0f}  pnl=${sub_pnl:>+8,.0f}  "
                          f"return={sub_ret:>+7.2%}  "
                          f"tax=${sub.tax_owed_accrued:>6,.0f}  "
                          f"active_theses={len(sub.thesis_book.active_theses())}")

    # Save eval JSONL — convert Pydantic instances (PortfolioState,
    # PositionSnapshot, etc.) into plain dicts so json.dumps doesn't choke.
    def _json_default(o):
        if hasattr(o, "model_dump"):
            return o.model_dump(by_alias=True)
        return str(o)

    out_path = out_dir / f"eval_{int(time.time())}.jsonl"
    with out_path.open("w") as fh:
        for rec in out_lines:
            fh.write(json.dumps(rec, default=_json_default) + "\n")
    hr()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
