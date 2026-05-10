"""Event classifier herbivore — Polygon news → BeliefActivations.

LOCAL Qwen3-4B via DSPy. Same model the apex uses. This is what makes
the head-to-head benchmark apples-to-apples: bare-Qwen path and ecology
path both consume the firehose with the same single LLM, just differently
structured.

DSPy gives us:
  - Typed Signature (input/output fields with descriptions) so the prompt
    is auto-built and the parser is auto-built.
  - JSON adapter for structured output, with retries on schema failure.
  - Caching at the LM call level (DSPy_CACHE) so we don't re-classify the
    same article twice.

We additionally cache by article.id on disk (cheap to do, survives across
process restarts and across DSPy cache evictions).

Architecturally distinct from the previous Sonnet-based version. The old
one was scaffolding; this is production: every herbivore call hits the
same Qwen3-4B that the apex hits.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Optional

import dspy

from ..schema import BeliefActivation


SPECIES_ID = "event_classifier.v0.qwen-local"

# Templates this herb is allowed to activate. Must match TEMPLATE_BIAS in
# scripts/instantiate_ticker_universe.py.
ALLOWED_TEMPLATES: list[str] = [
    "earnings_beat",
    "earnings_miss",
    "guidance_cut",
    "regulatory_action_announced",
    "regulatory_clearance_received",
    "major_product_launch_positive_reception",
    "activist_investor_takes_stake",
    "ceo_departure_unplanned",
    "acting_as_acquirer_in_announced_ma",
    "target_in_announced_ma",
    "material_lawsuit_filed",
    "dividend_increase_announced",
    "buyback_program_announced",
    "technical_support_broken",
    "short_interest_high_and_rising",
]

ALLOWED_DIRECTIONS = {"increases", "decreases"}
ALLOWED_MAGNITUDES = {"weak", "medium", "strong", "decisive"}
ALLOWED_DECAYS = {"instant", "fast", "normal", "slow", "glacial"}
ALLOWED_CONFIDENCES = {"low", "medium", "high", "very_high"}


# ── DSPy signature ────────────────────────────────────────────────────

class IdentifySubjectTickers(dspy.Signature):
    """Identify which of the tagged tickers the article is PRIMARILY about.

    News articles are tagged with many tickers because they MENTION them,
    but the article is usually about a SUBSET. e.g., a PayPal earnings
    article tagged with AAPL/GOOG/PYPL is about PYPL — only PYPL gets an
    activation, not AAPL or GOOG.

    Return ONLY tickers from `tagged_tickers` that the article is
    making concrete claims about. Empty list if none qualify (e.g., the
    article is a generic market-overview piece that mentions tickers in
    passing).
    """
    title: str = dspy.InputField(desc="Article headline")
    description: str = dspy.InputField(desc="Article description / lede")
    tagged_tickers: list[str] = dspy.InputField(
        desc="Tickers the article is TAGGED with (may include tickers only mentioned)"
    )
    subject_tickers: list[str] = dspy.OutputField(
        desc=("Subset of tagged_tickers that the article is primarily about — "
              "tickers with concrete events, not mere mentions. Empty list if none.")
    )


class ClassifyArticle(dspy.Signature):
    """Classify a financial news article into BeliefActivation records.

    Each activation specifies a ticker (must be in SUBJECT_TICKERS — only
    tickers the article is primarily about, not mere mentions), a belief
    template (must be from ALLOWED_TEMPLATES), direction of effect,
    magnitude, decay class, and confidence. Activate beliefs only when
    the article gives concrete evidence about that specific ticker.

    direction:
      "increases" = the belief becomes MORE TRUE (e.g., article confirms
                    an earnings beat happened → earnings_beat increases)
      "decreases" = the belief becomes LESS TRUE

    magnitude scale: weak (1.5x odds), medium (2.7x), strong (5x),
                     decisive (10x). Reserve "decisive" for clear material
                     events with concrete numbers.

    decay_class: fast (intraday), normal (~1hr), slow (~1day, earnings,
                 regulatory), glacial (~1 week, M&A, structural).

    Use no_signal=true and empty activations if subject_tickers is empty
    or the article has no actionable belief signal.
    """
    title: str = dspy.InputField(desc="Article headline")
    description: str = dspy.InputField(desc="Article description / lede")
    subject_tickers: list[str] = dspy.InputField(
        desc="Tickers the article is PRIMARILY ABOUT — activations MUST use these"
    )
    allowed_templates: list[str] = dspy.InputField(
        desc="Belief templates allowed; activations MUST use these exact strings"
    )

    activations: list[dict] = dspy.OutputField(
        desc=(
            "List of activation dicts. Each dict has keys: "
            "ticker (str, must be in subject_tickers), belief_template (str), "
            "direction (increases|decreases), "
            "magnitude (weak|medium|strong|decisive), "
            "decay_class (fast|normal|slow|glacial), "
            "confidence (low|medium|high|very_high), reasoning (≤200 chars)"
        )
    )
    no_signal: bool = dspy.OutputField(
        desc="True if the article carries no actionable belief signal"
    )


# ── Validation ────────────────────────────────────────────────────────

def _validate_activation(act: dict, universe: set[str]) -> Optional[dict]:
    try:
        ticker = str(act["ticker"]).upper().strip()
        template = str(act["belief_template"]).strip()
        direction = str(act["direction"]).lower().strip()
        magnitude = str(act["magnitude"]).lower().strip()
        decay = str(act.get("decay_class", "normal")).lower().strip()
        confidence = str(act.get("confidence", "medium")).lower().strip()
        reasoning = str(act.get("reasoning", ""))[:300]
    except (KeyError, TypeError):
        return None
    if template not in ALLOWED_TEMPLATES:
        return None
    if direction not in ALLOWED_DIRECTIONS:
        return None
    if magnitude not in ALLOWED_MAGNITUDES:
        return None
    if decay not in ALLOWED_DECAYS:
        return None
    if confidence not in ALLOWED_CONFIDENCES:
        return None
    if ticker not in universe:
        return None
    return {
        "ticker": ticker,
        "belief_template": template,
        "direction": direction,
        "magnitude": magnitude,
        "decay_class": decay,
        "confidence": confidence,
        "reasoning": reasoning,
    }


# ── Herbivore ─────────────────────────────────────────────────────────

@dataclass
class EventClassifierHerbivore:
    """DSPy-driven herbivore. Configure dspy.configure(lm=...) before use
    so the local Qwen3-4B is selected as the LM."""
    universe: list[str]
    cache_dir: Path
    subject_filter: Optional[dspy.Module] = None  # stage 1: identify subject tickers
    classifier: Optional[dspy.Module] = None      # stage 2: classify activations
    use_disk_cache: bool = True                    # set False for fresh runs

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._universe_set = set(self.universe)
        if self.subject_filter is None:
            self.subject_filter = dspy.Predict(IdentifySubjectTickers)
        if self.classifier is None:
            self.classifier = dspy.Predict(ClassifyArticle)

    def _cache_path(self, article_id: str) -> Path:
        h = hashlib.md5(article_id.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{h}.json"

    def classify(self, article: dict) -> dict:
        """Classify a single article. Returns a dict with shape:
            {"activations": [...], "no_signal": bool, "_cache_hit": bool}
        """
        aid = article.get("id")
        if not aid:
            return {"activations": [], "no_signal": True,
                    "_cache_hit": False, "_error": "no id"}
        cp = self._cache_path(aid)
        if self.use_disk_cache and cp.exists():
            try:
                payload = json.loads(cp.read_text())
                payload["_cache_hit"] = True
                return payload
            except json.JSONDecodeError:
                pass

        title = (article.get("title") or "").strip()
        description = (article.get("description") or "").strip()
        tagged = [
            t.upper() for t in (article.get("all_tickers") or article.get("tickers") or [])
            if t.upper() in self._universe_set
        ]
        if not tagged:
            # No tagged universe ticker → no possible activation
            payload = {"activations": [], "no_signal": True, "_cache_hit": False}
            try:
                cp.write_text(json.dumps(payload))
            except Exception:
                pass
            return payload

        # STAGE 1: filter tagged_tickers → subject_tickers (which the article
        # is actually about, not just mentioning)
        try:
            subj_pred = self.subject_filter(
                title=title,
                description=description,
                tagged_tickers=tagged,
            )
            subject_tickers = [
                t.upper() for t in (subj_pred.subject_tickers or [])
                if isinstance(t, str) and t.upper() in self._universe_set
                   and t.upper() in [tt.upper() for tt in tagged]
            ]
        except Exception as e:
            payload = {
                "activations": [], "no_signal": True,
                "_cache_hit": False, "_error": f"stage1_failed: {e}",
            }
            try:
                cp.write_text(json.dumps(payload))
            except Exception:
                pass
            return payload

        if not subject_tickers:
            # Article doesn't make concrete claims about any in-universe ticker
            payload = {
                "activations": [], "no_signal": True, "_cache_hit": False,
                "_subject_tickers": [],
            }
            try:
                cp.write_text(json.dumps(payload))
            except Exception:
                pass
            return payload

        # STAGE 2: classify into BeliefActivations using only subject_tickers
        try:
            pred = self.classifier(
                title=title,
                description=description,
                subject_tickers=subject_tickers,
                allowed_templates=ALLOWED_TEMPLATES,
            )
            raw_acts = list(pred.activations or [])
            no_signal = bool(pred.no_signal)
        except Exception as e:
            payload = {
                "activations": [], "no_signal": True,
                "_cache_hit": False, "_error": f"stage2_failed: {e}",
                "_subject_tickers": subject_tickers,
            }
            try:
                cp.write_text(json.dumps(payload))
            except Exception:
                pass
            return payload

        # Validate: ticker must be in subject_tickers (defensive — Qwen can
        # still confuse this even after stage 1).
        subject_set = set(subject_tickers)
        validated = []
        for a in raw_acts:
            if not isinstance(a, dict):
                continue
            v = _validate_activation(a, self._universe_set)
            if v is None:
                continue
            if v["ticker"] not in subject_set:
                continue
            validated.append(v)
        payload = {
            "activations": validated,
            "no_signal": no_signal or len(validated) == 0,
            "_cache_hit": False,
            "_subject_tickers": subject_tickers,
        }
        try:
            cp.write_text(json.dumps(payload))
        except Exception:
            pass
        return payload

    def classify_to_activations(self, article: dict) -> list[BeliefActivation]:
        result = self.classify(article)
        out: list[BeliefActivation] = []
        for a in result.get("activations", []):
            target_id = (
                f"belief.company.{a['belief_template']}__ticker_{a['ticker']}"
            )
            out.append(BeliefActivation(
                target_belief_id=target_id,
                direction_of_effect=a["direction"],         # type: ignore[arg-type]
                magnitude=a["magnitude"],                   # type: ignore[arg-type]
                decay_class=a["decay_class"],               # type: ignore[arg-type]
                self_rated_confidence=a["confidence"],      # type: ignore[arg-type]
                reasoning=a.get("reasoning", ""),
                species_id=SPECIES_ID,
            ))
        return out

    def classify_batch(
        self, articles: Iterable[dict], verbose: bool = False,
    ) -> list[BeliefActivation]:
        out: list[BeliefActivation] = []
        n_total = 0
        n_cached = 0
        for art in articles:
            n_total += 1
            result = self.classify(art)
            if result.get("_cache_hit"):
                n_cached += 1
            for a in result.get("activations", []):
                target_id = (
                    f"belief.company.{a['belief_template']}__ticker_{a['ticker']}"
                )
                out.append(BeliefActivation(
                    target_belief_id=target_id,
                    direction_of_effect=a["direction"],         # type: ignore[arg-type]
                    magnitude=a["magnitude"],                   # type: ignore[arg-type]
                    decay_class=a["decay_class"],               # type: ignore[arg-type]
                    self_rated_confidence=a["confidence"],      # type: ignore[arg-type]
                    reasoning=a.get("reasoning", ""),
                    species_id=SPECIES_ID,
                ))
            if verbose and n_total % 50 == 0:
                print(f"  classified {n_total} articles "
                      f"({n_cached} cache hits, {len(out)} activations)")
        if verbose:
            print(f"  done: {n_total} articles, {n_cached} cache hits, "
                  f"{len(out)} total activations")
        return out
