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

from trophic.beliefs.apex_signatures import (
    WatchlistFromFirehose,
    WatchlistFromObservations,
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
    }


def run_bare_pass(
    row: dict,
    apex_firehose: dspy.Module,
    focus_tickers: list[str],
    max_news: int = 35,
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
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="run only first N days of dataset")
    ap.add_argument("--days", type=str, default=None,
                    help="comma-separated specific dates (overrides --limit)")
    ap.add_argument("--path", choices=["bare", "ecology", "both"], default="both")
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
            max_tokens=2048,
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

    dspy.configure(lm=apex_lm, adapter=dspy.JSONAdapter())
    hr()

    apex_firehose = dspy.Predict(WatchlistFromFirehose) if args.path in ("bare", "both") else None
    apex_obs      = dspy.Predict(WatchlistFromObservations) if args.path in ("ecology", "both") else None

    # Ecology setup
    store = None; state = None; links = None
    xcorr = None; event_herb = None
    if args.path in ("ecology", "both"):
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
        )
        print(f"  event classifier ready (cache={EVENT_CACHE})")
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

        if args.path in ("ecology", "both"):
            t0 = time.time()
            eco = run_ecology_pass(
                row, store, state, links, xcorr, event_herb,
                apex_obs, focus_tickers,
                pass_persister=pass_persister,
                event_herb_lm=(herb_lm if herb_lm is not apex_lm else None),
            )
            eco["elapsed_s"] = round(time.time() - t0, 2)
            record["ecology"] = eco
            pass_persister.add_watchlist("ecology", eco["watchlist"])
            print(f"  ecology: {eco['n_event_activations']} event acts, "
                  f"{eco['n_xcorr_cluster_activations']} cluster, "
                  f"{eco['n_xcorr_sympathy_activations']} sympathy → "
                  f"{eco['n_observations']} obs → {len(eco['watchlist'])} watchlist  "
                  f"({eco['elapsed_s']}s)")
            for w in eco["watchlist"][:5]:
                print(f"    [{w['action']}] {w['ticker']}  p_up={w['p_up']:.2f}  {w['reason'][:80]}")

        if args.path in ("bare", "both"):
            t0 = time.time()
            bare = run_bare_pass(row, apex_firehose, focus_tickers)
            bare["elapsed_s"] = round(time.time() - t0, 2)
            record["bare"] = bare
            pass_persister.add_watchlist("bare", bare["watchlist"])
            print(f"  bare:    {bare['n_news_used']} news → "
                  f"{len(bare['watchlist'])} watchlist  ({bare['elapsed_s']}s)")
            for w in bare["watchlist"][:5]:
                print(f"    [{w['action']}] {w['ticker']}  p_up={w['p_up']:.2f}  {w['reason'][:80]}")

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
        if args.decomposer_each_day and args.path in ("ecology", "both"):
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

    # Save eval JSONL
    out_path = out_dir / f"eval_{int(time.time())}.jsonl"
    with out_path.open("w") as fh:
        for rec in out_lines:
            fh.write(json.dumps(rec) + "\n")
    hr()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
