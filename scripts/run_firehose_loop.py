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

from trophic.agents.apex_portfolio import ApexPortfolio
from trophic.beliefs.apex_signatures import (
    WatchlistFromFirehose,
    WatchlistFromFirehosePM,
    WatchlistFromObservations,
    WatchlistFromObservationsPM,
    WatchlistOracle,
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
from trophic.model_host import ModelHost


DATASET = ROOT / "data" / "firehose_dataset.jsonl"
EVAL_OUT = ROOT / "data" / "firehose_eval"
EVENT_CACHE = ROOT / "external" / "event_classifier_cache"
XCORR_STATE = ROOT / "data" / "cross_correlation_state.npz"

UNIVERSE = [
    "AAPL", "ABBV", "AMZN", "AVGO", "CSCO", "CVX", "GOOG", "HD", "JNJ", "JPM",
    "KO", "MA", "MCD", "MRK", "MSFT", "PEP", "PG", "UNH", "V", "WMT",
]


def hr() -> None:
    print("─" * 78)


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
    for attempt_idx in range(max_retries + 1):
        try:
            pred = apex_module(
                **apex_kwargs,
                previous_attempts=previous_attempts,
            )
            # Convert Pydantic Order instances → plain dicts for the validator
            raw_orders = [_order_to_dict(o) for o in (pred.orders or [])]
            rationale = (getattr(pred, "rationale", "") or "")[:300]
        except Exception as e:
            last_error = str(e)
            print(f"  [{label} apex-PM error attempt {attempt_idx}] {e}")
            raw_orders = []
            rationale = ""

        validated, rejected = portfolio.validate_orders(
            raw_orders, universe, prices,
        )
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
    news = row["firehose_news"]
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


def run_oracle_pass(
    row: dict,
    apex_oracle: dspy.Module,
    focus_tickers: list[str],
    portfolio: ApexPortfolio,
    pm_retries: int = 0,
    label: str = "oracle",
) -> dict:
    """Oracle pipeline: apex sees tomorrow_returns and allocates with foresight."""
    tomorrow_returns = [
        {"ticker": g["ticker"], "actual_return": g["actual_return"]}
        for g in row.get("targets", [])
    ]
    prices_today = portfolio.advance_prices(row["date"], row.get("targets", []))
    portfolio_state, open_positions = portfolio.state_for_apex(row["date"], prices_today)
    result = call_apex_pm_with_retry(
        apex_module=apex_oracle,
        apex_kwargs={
            "universe": UNIVERSE,
            "focus_tickers": focus_tickers,
            "tomorrow_returns": tomorrow_returns,
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
        "n_news_used": 0,
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


def run_bare_pass(
    row: dict,
    apex_firehose: dspy.Module,
    focus_tickers: list[str],
    max_news: int = 35,
    portfolio: ApexPortfolio | None = None,
    pm_retries: int = 0,
) -> dict:
    """Run the bare-Qwen pipeline for one trading day."""
    news = row["firehose_news"][:max_news]  # cap context for 4B model
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
    ap.add_argument("--path", choices=[
        "bare", "ecology", "both",
        "oracle-qwen", "oracle-sonnet", "oracle-opus",
        "all-with-oracles",
        "ecology-sonnet", "ecology-opus",
        "bare-sonnet", "bare-opus",
        "model-ablation",  # 9-way: 3 apex models × 3 pre-apex pipelines
    ], default="both",
        help="bare/ecology/both = standard A/B. "
             "oracle-qwen|sonnet|opus = single oracle path (apex sees tomorrow_returns). "
             "ecology-sonnet|opus, bare-sonnet|opus = frontier-apex variants. "
             "model-ablation = 3 apex models × 3 pre-apex pipelines (9-way matrix).")
    ap.add_argument("--focus", type=str, default="",
                    help="comma-separated focus tickers")
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
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    focus_tickers = [t.strip().upper() for t in args.focus.split(",") if t.strip()]
    pass_persister = PassPersister(out_dir, model_id=ModelConfig().chat_model_id)

    print("Firehose loop runner")
    print(f"path: {args.path}  focus: {focus_tickers}")
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
    ABL = "model-ablation"
    runs_bare           = args.path in ("bare", "both", "all-with-oracles", ABL)
    # runs_ecology = ecology pipeline with the QWEN apex specifically
    runs_ecology        = args.path in ("ecology", "both", "all-with-oracles", ABL)
    runs_ecology_sonnet = args.path in ("ecology-sonnet", ABL)
    runs_ecology_opus   = args.path in ("ecology-opus", ABL)
    # The ecology STACK (herbivore + carnivore) is needed by any of those
    needs_ecology_stack = runs_ecology or runs_ecology_sonnet or runs_ecology_opus
    runs_oracle_qwen    = args.path in ("oracle-qwen", "all-with-oracles", ABL)
    runs_oracle_sonnet  = args.path in ("oracle-sonnet", "all-with-oracles", ABL)
    runs_oracle_opus    = args.path in ("oracle-opus", ABL)
    runs_bare_sonnet    = args.path in ("bare-sonnet", ABL)
    runs_bare_opus      = args.path in ("bare-opus", ABL)

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
        apex_oracle_qwen   = dspy.Predict(WatchlistOracle) if runs_oracle_qwen else None
        apex_oracle_sonnet = dspy.Predict(WatchlistOracle) if runs_oracle_sonnet else None
        apex_oracle_opus   = dspy.Predict(WatchlistOracle) if runs_oracle_opus else None
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
                            pm_retries=args.apex_pm_retries,
                            label="oracle_qwen",
                        ),
                    )))
                if runs_oracle_sonnet and oracle_sonnet_portfolio is not None:
                    tasks.append(("oracle_sonnet", _async_run_with_lm(
                        lambda: run_oracle_pass(
                            row, apex_oracle_sonnet, focus_tickers,
                            portfolio=oracle_sonnet_portfolio,
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
                            pm_retries=args.apex_pm_retries,
                            label="oracle_opus",
                        ),
                        lm=opus_lm,
                    )))
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
    ]
    if args.apex_pm and any(p for _, p in all_portfolios_named):
        hr()
        print("PORTFOLIO RESULTS")
        portfolios = all_portfolios_named
        for label, pf in portfolios:
            if pf is None:
                continue
            eq = pf.equity(pf._last_price)
            pnl = eq - pf.starting_cash
            ret = pnl / pf.starting_cash
            print(f"  {label:<14} ${eq:>10,.0f}  pnl=${pnl:>+9,.0f}  "
                  f"return={ret:>+7.2%}  buys={pf.n_buys}  sells={pf.n_sells}  "
                  f"realized=${pf.realized_gains_cum:>+8,.0f}  "
                  f"tax_owed=${pf.tax_owed:>7,.0f}  "
                  f"yield_cum=${pf.cash_yield_cum:>5,.0f}")

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
