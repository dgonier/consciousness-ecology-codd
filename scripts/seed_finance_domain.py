"""Seed the Hexis Mind Tree with the trophic finance domain.

Writes the canonical species/template hierarchy to Neo4j via the deployed
Hexis vLLM at /v1/mind_tree/node. Idempotent — safe to re-run; uses
path-based MERGE under the hood.

Topology written:

  root:finance
    └── event_classifier.v0
          ├── earnings_beat
          ├── earnings_miss
          ├── major_product_launch_positive_reception
          ├── ... (15 templates total)
    └── cross_correlation.v0
          ├── tech_cluster_high
          ├── financial_cluster_low
          ├── ... (cluster + sympathy nodes)

Each species is a parent node with a description; each template is a child
with a description that hints at what triggers + invariants apply.

Usage:
  .venv/bin/python -m scripts.seed_finance_domain
  .venv/bin/python -m scripts.seed_finance_domain --base-url <override>
  .venv/bin/python -m scripts.seed_finance_domain --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trophic.beliefs.hexis_client import HexisClient


# ── Species + templates (lifted from instantiate_ticker_universe.py) ──

EVENT_CLASSIFIER_TEMPLATES = {
    # bullish
    "earnings_beat": (
        "Quarterly earnings beat consensus by ≥5% on revenue OR EPS. Triggers "
        "a bullish activation. Watch for: explicit 'beat' / 'exceeded estimates' "
        "language with magnitude. Suppress when article is about a competitor's beat."
    ),
    "regulatory_clearance_received": (
        "FDA approval, antitrust clearance, or regulatory green-light for a "
        "specific product or transaction. Bullish. Watch for: 'approved', "
        "'cleared', 'authorization granted'. Suppress when clearance is conditional "
        "or for a competitor."
    ),
    "major_product_launch_positive_reception": (
        "Product launch with positive analyst response, stock movement, or "
        "demonstrable demand signal. Bullish. Watch for: launch + price-target "
        "increase or +N% reception. Suppress when reception is mixed or speculative."
    ),
    "activist_investor_takes_stake": (
        "Activist (Ackman, Icahn, Loeb, Elliott, etc.) takes ≥1% stake. Bullish "
        "short-term. Watch for: explicit named investor + named position size. "
        "Suppress when activist is exiting or position is purely passive."
    ),
    "target_in_announced_ma": (
        "Company is the announced TARGET (acquiree) in an M&A. Strongly "
        "bullish (deal premium). Watch for: 'to be acquired by', 'agreed to "
        "be bought'. Suppress when company is the acquirer."
    ),
    "dividend_increase_announced": (
        "Dividend per share raised vs prior. Modestly bullish (slow signal). "
        "Watch for: 'raises dividend', '+X% dividend'. Suppress when increase "
        "is below cost-of-capital growth rate."
    ),
    "buyback_program_announced": (
        "Share repurchase authorization. Modestly bullish. Watch for: '$NB "
        "buyback', 'repurchase authorization'. Suppress when buyback is "
        "purely offsetting dilution from option grants."
    ),
    # bearish
    "earnings_miss": (
        "Quarterly earnings missed consensus by ≥5% on revenue OR EPS. "
        "Strongly bearish. Watch for: 'missed', 'fell short', 'below estimates' "
        "with magnitude. Suppress when miss is from a one-time charge."
    ),
    "guidance_cut": (
        "Forward guidance lowered vs prior. Strongly bearish. Watch for: "
        "'lowered guidance', 'reduced outlook', 'cut forecast'. Suppress when "
        "cut is industry-wide and explicitly framed as macro."
    ),
    "regulatory_action_announced": (
        "DOJ, SEC, EU Commission, or other regulator opens an action. "
        "DISABLED 2026-05-09 — 29% accuracy in audit (anti-correlated). "
        "Strength=0 in graph; can re-learn via decomposer if precision recovers."
    ),
    "material_lawsuit_filed": (
        "Material class-action or securities lawsuit filed. Bearish. Watch "
        "for: 'filed lawsuit', 'class action', 'securities fraud claim'. "
        "Suppress when lawsuit is settled or dismissed."
    ),
    "ceo_departure_unplanned": (
        "Unplanned CEO exit (resignation, termination, sudden retirement). "
        "Bearish. Watch for: 'steps down', 'unexpected departure', 'effective "
        "immediately'. Suppress when departure is for promotion or planned succession."
    ),
    "acting_as_acquirer_in_announced_ma": (
        "Company is the announced ACQUIRER in an M&A (paying premium). "
        "Modestly bearish (acquirer underperforms in announcement-week). "
        "Watch for: 'agreed to acquire', 'to buy'. Suppress when target is small."
    ),
    "technical_support_broken": (
        "Stock breaks key technical support level (200-day MA, prior pivot). "
        "DISABLED 2026-05-09 — 41% accuracy in audit; event_classifier "
        "hallucinates TA signals from news headlines. Strength=0 in graph."
    ),
    "short_interest_high_and_rising": (
        "Short interest ≥15% of float and rising. Modestly bearish (suggests "
        "structural skepticism). Watch for: explicit short-interest data + "
        "trend. Suppress when short-squeeze setup is the article's framing."
    ),
}


CROSS_CORRELATION_NODES = {
    "tech_cluster_high": (
        "Tech-sector intra-correlation (60d window) running high. Suggests "
        "regime where idiosyncratic news on one tech name spreads to others. "
        "Used to broadcast sympathy activations."
    ),
    "tech_cluster_low": (
        "Tech-sector intra-correlation running low. Idiosyncratic news stays "
        "idiosyncratic. Sympathy broadcasts dampened."
    ),
    "financial_cluster_high": (
        "Financial-sector intra-correlation high. JPM/MA/V move together."
    ),
    "consumer_staples_cluster_high": (
        "Consumer-staples intra-correlation high. KO/PG/WMT/PEP move together."
    ),
    "cross_asset_dispersion_high": (
        "Inter-cluster dispersion high — sectors decoupled, regime is "
        "rotational rather than systemic."
    ),
    "sympathy_propagation": (
        "Single-ticker activation propagating to correlated peers via 60d "
        "Pearson + 1d lead-lag. Bug 2026-05-09: this can over-count — see "
        "task #139 (sympathy double-counting)."
    ),
}


# ── Failure-mode lessons drawn from the 2026-05-09 65-day audit ──
# These are written via write_lesson (Mind Tree's lesson-writing path)
# rather than as plain ensure_mind_node nodes, because they are the kind
# of guidance the M-channel + slot retrieval should pull at inference
# time. Format:
#   path: where the lesson attaches in the Mind Tree
#   text: the lesson body (what the species should attend to)
#   author: "audit_2026-05-09" so the decomposer can later distinguish
#     bootstrap-from-audit lessons from its own write-backs.
#   task_types: routing tags so /v1/remember can retrieve them when the
#     herbivore is classifying that template.
#   failure_modes: what to mark this lesson as guarding against.

EVENT_CLASSIFIER_FAILURE_LESSONS = [
    {
        "path": "root:finance:event_classifier.v0",
        "text": (
            "AUDIT 2026-05-09: across 754 hindsight-graded teacher hints "
            "from a 65-day Polygon news sweep, this species' overall "
            "directional precision was 47%. Two specific failure modes "
            "drove most errors: bullish bias (83% of fires were "
            "'increases' against a ground-truth class skew of only 51.6% "
            "up), and over-firing on two specific templates "
            "(earnings_beat at 24%, major_product_launch at 47%). "
            "Conservatism on bullish templates and stricter evidence "
            "requirements on those two templates would have lifted "
            "downstream MCC by ~+0.20."
        ),
        "task_types": ["financial_news", "event_classification"],
        "failure_modes": ["bullish_bias", "low_precision"],
    },
    {
        "path": "root:finance:event_classifier.v0:earnings_beat",
        "text": (
            "FAILURE MODE — earnings_beat over-fires on 'Q4 revenue grew "
            "X%' or 'analyst price target raised' phrasing in articles "
            "that don't actually describe a quarterly earnings BEAT vs "
            "consensus. 24% precision in 65-day audit (10 right / 31 wrong "
            "across 41 fires). REQUIRE explicit beat language ('beat "
            "consensus by X%', 'exceeded estimates', 'surpassed Wall "
            "Street expectations') AND a magnitude (revenue or EPS delta) "
            "before firing. Suppress on guidance-only articles, analyst-"
            "upgrade-only articles, or 'on track to beat'. Specifically "
            "watch out for confusing competitor-beat articles as the "
            "subject ticker's beat."
        ),
        "task_types": ["financial_news", "earnings"],
        "failure_modes": ["earnings_beat_overcall", "over_classification"],
    },
    {
        "path": "root:finance:event_classifier.v0:major_product_launch_positive_reception",
        "text": (
            "FAILURE MODE — fires on 'rumored launch', 'expected to "
            "announce', or 'analyst preview' articles that anticipate a "
            "launch but don't yet describe positive market reception. 47% "
            "precision in 65-day audit (23 right / 26 wrong across 49 "
            "fires). REQUIRE post-launch language: stock movement on the "
            "launch day, analyst price-target revisions citing the launch, "
            "or demand signals like preorder volume. Suppress speculative "
            "previews, calendar reminders, and articles where 'launch' is "
            "secondary to other news."
        ),
        "task_types": ["financial_news", "product_launch"],
        "failure_modes": ["product_launch_speculative_overcall"],
    },
    {
        "path": "root:finance:event_classifier.v0:activist_investor_takes_stake",
        "text": (
            "FAILURE MODE — over-fires on 'activist X is reportedly "
            "considering' or 'rumored to take a stake'. 12% precision in "
            "the audit (1 right / 7 wrong, n=8 too small for confidence "
            "but the directional signal is clear). REQUIRE explicit "
            "filing language ('13D filed', '5.x% stake disclosed') OR a "
            "named transaction ('agreed to buy X% of'). Treat activist "
            "rumor articles as no_signal."
        ),
        "task_types": ["financial_news", "activist"],
        "failure_modes": ["activist_rumor_overcall"],
    },
    {
        "path": "root:finance:event_classifier.v0:technical_support_broken",
        "text": (
            "DISABLED 2026-05-09 — strength=0 in carnivore propagation. "
            "AUDIT showed this template fires on news commentary ('AAPL "
            "is testing a critical uptrend line', '200-day MA approach') "
            "rather than actual technical breakdowns, hallucinating TA "
            "signals from price-action discussion in the article. 41% "
            "precision (10 wrong SELLs across 17 fires) was the single "
            "largest contributor to negative top-K MCC. Either suppress "
            "this template entirely until DSPy refinement lifts precision, "
            "or restrict to articles that EXPLICITLY say a support level "
            "broke (with a price level cited)."
        ),
        "task_types": ["financial_news", "technical_analysis"],
        "failure_modes": ["technical_support_broken_news_hallucination"],
    },
    {
        "path": "root:finance:event_classifier.v0:regulatory_action_announced",
        "text": (
            "DISABLED 2026-05-09 — strength=0 in carnivore propagation. "
            "AUDIT showed 29% precision (2 right / 5 wrong across 7 "
            "fires) — anti-correlated. The template confuses any "
            "regulatory mention (clearance, ongoing review, completed "
            "settlement) with adverse action. REQUIRE explicit adverse "
            "action language: 'lawsuit filed by SEC', 'DOJ investigation "
            "opened', 'CMA blocked', 'EU fined'. Suppress on neutral "
            "regulatory-news articles or articles where the action is "
            "favorable to the company."
        ),
        "task_types": ["financial_news", "regulatory"],
        "failure_modes": ["regulatory_action_anti_correlated"],
    },
    {
        "path": "root:finance:event_classifier.v0:dividend_increase_announced",
        "text": (
            "FAILURE MODE — fires on 'maintained dividend' or 'declared "
            "regular dividend' articles. 42% precision in 65-day audit "
            "(34 right / 47 wrong, n=81). REQUIRE an INCREASE vs prior "
            "with a magnitude ('raised dividend by X%', '+$Y per share'). "
            "Suppress when the article describes a routine quarterly "
            "declaration without comparison to prior."
        ),
        "task_types": ["financial_news", "capital_returns"],
        "failure_modes": ["dividend_routine_overcall"],
    },
    {
        "path": "root:finance:event_classifier.v0:acting_as_acquirer_in_announced_ma",
        "text": (
            "FAILURE MODE — confuses 'in talks to acquire' rumors with "
            "actual announced deals. 32% precision (7 right / 15 wrong, "
            "n=22). REQUIRE definitive language: 'agreed to buy', "
            "'announced acquisition of', 'tender offer at $X'. Speculative "
            "M&A coverage should be no_signal. Also: confirm the subject "
            "ticker is the ACQUIRER (paying premium, modestly bearish "
            "short-term) not the TARGET (receiving premium, strongly "
            "bullish — different template)."
        ),
        "task_types": ["financial_news", "ma"],
        "failure_modes": ["ma_speculation_overcall", "acquirer_target_confusion"],
    },
]


# ── Cross-species handoff hints ──
# Tells each species when to defer to a sibling. Read at session start so
# the herbivore knows when to emit no_signal vs fire.
HANDOFF_LESSONS = [
    {
        "path": "root:finance:event_classifier.v0",
        "text": (
            "HANDOFF — when an article describes a sector-wide pattern "
            "(multiple tickers in same sector all moving on a single "
            "regulatory or macro story), prefer to emit no_signal at the "
            "single-ticker level and let the cross_correlation.v0 species "
            "fire a cluster activation downstream. Single-ticker fires "
            "from sector-wide news double-count when sympathy "
            "propagation runs."
        ),
        "task_types": ["financial_news", "cross_species_routing"],
        "failure_modes": ["sector_wide_double_count"],
    },
    {
        "path": "root:finance:cross_correlation.v0",
        "text": (
            "HANDOFF — sympathy activations on correlated tickers should "
            "step DOWN in magnitude relative to the source activation "
            "(weak->very_weak, medium->weak, strong->medium) since the "
            "evidence is indirect. Current implementation does this but "
            "the carnivore aggregator may still over-count if multiple "
            "sympathy paths converge on the same ticker — see task #139."
        ),
        "task_types": ["cross_correlation", "cross_species_routing"],
        "failure_modes": ["sympathy_double_count"],
    },
]


# ── Calibration priors from the audit ──
EVIDENCE_PRIOR_LESSONS = [
    {
        "path": "root:finance:event_classifier.v0",
        "text": (
            "CALIBRATION 2026-05-09 — 65-day audit showed 83% of "
            "activations were directionally 'increases' vs an actual "
            "ground-truth up-rate of 51.6%. This species has a structural "
            "bullish bias. When choosing magnitude on bullish-template "
            "fires, prefer one bucket lower than initial assessment "
            "(strong→medium, medium→weak) until precision recalibrates. "
            "Do NOT downscale bearish template fires (earnings_miss, "
            "guidance_cut, ceo_departure) — these were 88%, 56%, 71% "
            "precision respectively and don't share the bias."
        ),
        "task_types": ["financial_news", "calibration"],
        "failure_modes": ["bullish_magnitude_inflation"],
    },
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=None,
                    help="Override HEXIS_API_URL")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be written without sending")
    args = ap.parse_args()

    client = HexisClient(base_url=args.base_url) if args.base_url else HexisClient()

    print(f"Seeding finance domain at {client.base_url}")
    print(f"  health check: ", end="", flush=True)
    if not args.dry_run:
        h = client.health()
        print(f"vllm_ready={h['vllm_ready']} mind_tree_enabled="
              f"{h['ablation_flags']['mind_tree_enabled']} bedrock_ready={h['bedrock_ready']}")
        if not h["vllm_ready"]:
            print("  ABORT: deploy not ready")
            return
        if not h["ablation_flags"]["mind_tree_enabled"]:
            print("  ABORT: mind tree disabled on deploy "
                  "(set HEXIS_MIND_TREE_ENABLED=true)")
            return
    else:
        print("(dry-run, skipped)")

    n_written = 0
    n_skipped = 0

    def write(path: str, description: str, **extra) -> None:
        nonlocal n_written, n_skipped
        if args.dry_run:
            print(f"  [DRY] {path}: {description[:80]}...")
            n_skipped += 1
            return
        try:
            res = client.write_mind_tree_node(
                path=path, description=description, **extra,
            )
            n_written += 1
            print(f"  + {path}: wrote_node={res['wrote_node']}")
        except Exception as e:
            print(f"  ! {path}: {type(e).__name__}: {str(e)[:120]}")

    # 1. Domain root
    write(
        "root:finance",
        "Trophic firehose finance domain. Hosts species (event_classifier, "
        "cross_correlation, ...) that classify financial news articles into "
        "directional belief activations on a 20-ticker universe (AAPL, ABBV, "
        "AMZN, AVGO, CSCO, CVX, GOOG, HD, JNJ, JPM, KO, MA, MCD, MRK, MSFT, "
        "PEP, PG, UNH, V, WMT). Decomposer feeds back hindsight hints every "
        "3 days based on realized 1-day-ahead returns.",
    )

    # 2. event_classifier.v0 species root
    write(
        "root:finance:event_classifier.v0",
        "Event-classifier herbivore species. Per-article DSPy 2-stage signature "
        "(IdentifySubjectTickers → ClassifyArticle). Emits BeliefActivation per "
        "(article, ticker, template). Current state (2026-05-09): 53% precision "
        "across 754 hints; major_product_launch and earnings_beat dominate "
        "activations but only ~50% accurate. Two templates "
        "(technical_support_broken, regulatory_action_announced) disabled "
        "after audit.",
    )

    # 3. event_classifier templates as children
    for tmpl, desc in EVENT_CLASSIFIER_TEMPLATES.items():
        write(
            f"root:finance:event_classifier.v0:{tmpl}",
            desc,
        )

    # 4. cross_correlation.v0 species root
    write(
        "root:finance:cross_correlation.v0",
        "Cross-correlation herbivore species. Pure-numpy: rolling 60d/20d "
        "Pearson + 1-day lead-lag matrices over the 20-ticker universe. "
        "Emits cluster regime activations + sympathy broadcasts on top of "
        "event_classifier activations. Known bug (#139, 2026-05-09): sympathy "
        "double-counting at the carnivore aggregator layer over-amplifies "
        "single-article evidence.",
    )

    for tmpl, desc in CROSS_CORRELATION_NODES.items():
        write(
            f"root:finance:cross_correlation.v0:{tmpl}",
            desc,
        )

    # 5. Failure-mode lessons from 2026-05-09 audit. These attach to the
    # already-existing nodes via the same path; the deploy treats the
    # presence of lesson_text as a write_lesson call on top of the node.
    n_lessons = 0
    n_lesson_fail = 0
    all_lessons = (
        EVENT_CLASSIFIER_FAILURE_LESSONS
        + HANDOFF_LESSONS
        + EVIDENCE_PRIOR_LESSONS
    )
    print()
    print(f"Writing {len(all_lessons)} audit-derived lessons (audit_2026-05-09)...")
    for lesson in all_lessons:
        if args.dry_run:
            print(f"  [DRY-LESSON] {lesson['path']}: {lesson['text'][:80]}...")
            continue
        try:
            res = client.write_mind_tree_node(
                path=lesson["path"],
                description="",
                lesson_text=lesson["text"],
                lesson_author="audit_2026-05-09",
                lesson_task_types=lesson["task_types"],
                lesson_failure_modes=lesson["failure_modes"],
            )
            lid = res.get("lesson_node_id")
            if lid:
                n_lessons += 1
                print(f"  + lesson@{lesson['path']}: "
                      f"lesson_node_id={lid}")
            else:
                n_lesson_fail += 1
                print(f"  ! lesson@{lesson['path']}: "
                      f"server did not write lesson (lesson_node_id=None)")
        except Exception as e:
            n_lesson_fail += 1
            print(f"  ! lesson@{lesson['path']}: {type(e).__name__}: "
                  f"{str(e)[:120]}")

    print()
    print(f"Done. Wrote {n_written} nodes + {n_lessons} lessons, "
          f"skipped {n_skipped}, failed {n_lesson_fail} "
          f"({len(EVENT_CLASSIFIER_TEMPLATES) + len(CROSS_CORRELATION_NODES) + 3} nodes + "
          f"{len(all_lessons)} lessons expected).")


if __name__ == "__main__":
    main()
