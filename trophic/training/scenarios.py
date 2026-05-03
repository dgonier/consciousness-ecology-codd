"""Fixed scenario library for SFT and GRPO.

Each scenario bundles raw inputs (what producers see), per-herbivore-kind
target syntheses, and a target predator prediction. Targets are the SFT
supervision signal. Abstention expectations let us test the null gate.

Train/eval split is fixed by scenario name — names starting with 'eval_'
are held out. Everything else is training data.

v2.5: scenarios are now programmatically generated across (ticker ×
direction × source × diet shape) so we have 100+ training examples and
~10% held-out eval. The generation is deterministic (uses fixed seed
indices for RawInput ids) so the same set of scenarios is produced on
every run.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from ..types import RawInput
from .xml_schema import emit_prediction, emit_synthesis


@dataclass
class Scenario:
    name: str
    inputs: list[RawInput]
    technical_target: str | None
    fundamental_target: str | None
    predator_target: str | None
    forecaster_target: str | None = None       # multi-modal scenarios only
    interrogator_target: str | None = None     # set when scenario has quant inputs
    # 2026-05-03: numeric Chronos features extracted from the OHLCV history
    # at scenario-build time, fed through a learnable projection head into
    # the predator's herb-trough as a real numeric peer signal. None when
    # not computed (StockNet scenarios pre-2026-05-03 won't have these).
    forecaster_features: list[float] | None = None
    expected_abstain_technical: bool = False
    expected_abstain_fundamental: bool = False
    expected_abstain_forecaster: bool = False
    expected_abstain_interrogator: bool = False
    expected_abstain_predator: bool = False

    @property
    def is_eval(self) -> bool:
        return self.name.startswith("eval_")


# NOTE: The inline format-conversion helpers below (_ri + the per-source
# payload builders: _ohlcv_payload, _trade_payload, _options_payload,
# _filing_payload, _press_payload, _quote_series_payload) drive the
# *synthetic* training scenarios. They are intentionally NOT routed
# through `trophic/adapters/` because they're not adapting an external
# format — they're pseudo-random fixtures that imply a target direction
# from a seed. Real-data adapters live in `trophic/adapters/stocknet/`
# (phase2-C) and feed the StockNet loader (`stocknet_loader.py`).
def _ri(source: str, payload: dict, seed: int) -> RawInput:
    return RawInput(
        id=f"{source}.{seed:04d}.{uuid.UUID(int=seed).hex[:8]}",
        source=source,
        payload=payload,
    )


# ---------- target text templates ----------

def _technical_target(
    ticker: str, direction: str,
    pct_move: float | None = None, vol_note: str = "",
) -> str:
    """Technical synthesis as XML. vol_note is unused in XML form (the
    structured schema doesn't carry free-text justification at the herb
    tier; that's reserved for the predator's <evidence>).
    """
    signal = "strong" if (pct_move is not None and abs(pct_move) > 1.0) else "moderate"
    return emit_synthesis(
        kind="technical",
        ticker=ticker,
        bias=direction,
        pct_move=pct_move,
        signal=signal,
        confidence=0.70,
    )


def _fundamental_target(ticker: str, what: str, expected_dir: str | None = None) -> str:
    """Fundamental synthesis as XML. `what` (the disclosure summary) is
    captured via the bias direction; the actual narrative goes into the
    predator's <evidence> block.
    """
    return emit_synthesis(
        kind="fundamental",
        ticker=ticker,
        bias=expected_dir,
        signal="moderate",
        confidence=0.75,
    )


def _predator_target(
    ticker: str, direction: str, horizon_min: int = 60,
    pct_move: float | None = None, sigma: float | None = None,
    evidence: list[tuple[str, str]] | None = None,
) -> str:
    """Single-modal predator prediction as XML."""
    return emit_prediction(
        ticker=ticker,
        direction=direction,
        pct_move=pct_move,
        horizon_min=horizon_min,
        sigma_pct=sigma,
        confidence=0.65,
        evidence=evidence,
    )


def _abstain_target(kind: str = "technical") -> str:
    """Herbivore abstain. Default kind is 'technical' since the call site
    just needs *some* abstention; the actual kind matters for parsing.
    """
    return emit_synthesis(kind=kind, abstain=True)


def _abstain_pred_target() -> str:
    return emit_prediction(abstain=True)


def _ohlcv_pct_change(payload: dict) -> float:
    """Exact percent change from open to close in the OHLCV payload."""
    o = float(payload["open"])
    c = float(payload["close"])
    return (c - o) / o * 100.0


def _interrogator_target(
    ticker: str, direction: str,
    pct_move: float | None = None,
    sigma_pct: float | None = None,
) -> str:
    """Interrogator herbivore synthesis. Same XML schema as other herbivores
    but kind='interrogator'. The defining property: pct_move should be
    *exact* (computed via the math node), so during training the supervised
    target uses the exact computed value.
    """
    signal = "strong" if (pct_move is not None and abs(pct_move) > 1.0) else "moderate"
    return emit_synthesis(
        kind="interrogator",
        ticker=ticker,
        bias=direction,
        pct_move=pct_move,
        sigma_pct=sigma_pct,
        signal=signal,
        confidence=0.80,  # math-grounded → higher confidence
    )


def _fill_interrogator_targets(scenarios: list[Scenario]) -> None:
    """Mutate scenarios in place: any scenario whose inputs include OHLCV
    or quote_series gets an interrogator_target derived from the same
    numerical computations the predator_target already uses.
    """
    import re
    for sc in scenarios:
        if sc.interrogator_target is not None:
            continue  # already set
        # Find quant inputs.
        ohlcv_payload = None
        quote_payload = None
        for inp in sc.inputs:
            if inp.source == "ohlcv" and ohlcv_payload is None:
                ohlcv_payload = inp.payload
            elif inp.source == "quote_series" and quote_payload is None:
                quote_payload = inp.payload
        if ohlcv_payload is None and quote_payload is None:
            # Pure disclosure / press / book / halt — interrogator abstains.
            sc.interrogator_target = emit_synthesis(
                kind="interrogator", abstain=True,
            )
            sc.expected_abstain_interrogator = True
            continue
        # Extract ticker + direction from predator_target (ground truth).
        target = sc.predator_target or ""
        m = re.search(r"<ticker>([A-Z]+)</ticker>", target)
        ticker = m.group(1) if m else "?"
        m = re.search(r"<direction>(up|down|flat)</direction>", target)
        direction = m.group(1) if m else "flat"
        # Compute exact pct_move from whichever payload is available.
        if ohlcv_payload is not None:
            pct_move = _ohlcv_pct_change(ohlcv_payload)
            sigma_pct = None
        else:
            pct_move, sigma_pct = _quote_forecast_pct(quote_payload)
        sc.interrogator_target = _interrogator_target(
            ticker=ticker, direction=direction,
            pct_move=pct_move, sigma_pct=sigma_pct,
        )


# ---------- shared corpora ----------

# 16 tickers gives plenty of variety without being noise.
TICKERS = [
    "AAPL", "NVDA", "TSLA", "AMD", "GOOG", "MSFT", "META", "JPM",
    "AMZN", "NFLX", "AVGO", "ORCL", "CRM", "ADBE", "QCOM", "INTC",
]

FILING_TYPES = ["8-K", "10-Q", "13D", "S-1"]

# Disclosure summaries with implied direction
DISCLOSURE_CASES = [
    ("guidance reduction citing supply constraints",          "down"),
    ("revenue beat with strong ad pricing",                   "up"),
    ("material weakness in internal controls",                "down"),
    ("strategic acquisition pending review",                  "up"),
    ("cloud growth above consensus",                          "up"),
    ("cybersecurity incident with limited impact",            "down"),
    ("antitrust settlement reached",                          "up"),
    ("CFO departure announced effective end of quarter",      "down"),
    ("multi-year supply agreement with major customer",       "up"),
    ("regulatory probe into accounting practices opened",     "down"),
    ("dividend increase announced ahead of earnings",         "up"),
    ("data center capacity expansion in EMEA",                "up"),
    ("class-action lawsuit certified by district court",      "down"),
    ("activist investor discloses 5%+ stake",                 "up"),
    ("guidance reaffirmed despite macro headwinds",           "flat"),
    ("operational restructuring with workforce reduction",    "flat"),
]


# ---------- per-source builders ----------

def _ohlcv_payload(ticker: str, direction: str, seed: int) -> dict:
    if direction == "up":
        ch_lo, ch_hi = 0.005, 0.025
    elif direction == "down":
        ch_lo, ch_hi = -0.025, -0.005
    else:
        ch_lo, ch_hi = -0.002, 0.002
    # Deterministic pseudo-random in range from seed
    frac = ((seed * 2654435761) % 10_000) / 10_000.0
    ch = ch_lo + (ch_hi - ch_lo) * frac
    base = 100.0 + (seed % 80) * 5  # spread tickers across price range
    open_ = round(base, 2)
    close = round(base * (1 + ch), 2)
    high = round(max(open_, close) * 1.002, 2)
    low = round(min(open_, close) * 0.998, 2)
    vol = 2_000_000 + (seed * 31337) % 8_000_000
    return {
        "ticker": ticker, "open": open_, "high": high, "low": low, "close": close,
        "volume": int(vol), "window": "1m",
    }


def _trade_payload(ticker: str, direction: str, seed: int) -> dict:
    side = "B" if direction == "up" else "S"
    base = 100.0 + (seed % 80) * 5
    return {
        "ticker": ticker, "price": round(base * (1.0 + 0.001 * (1 if side == "B" else -1)), 2),
        "size": 1_000 + (seed * 7919) % 9_000, "side": side,
    }


def _options_payload(ticker: str, direction: str, seed: int) -> dict:
    side = "C" if direction == "up" else "P"
    return {
        "ticker": ticker, "strike": float(50 + (seed % 12) * 25),
        "expiry_days": [1, 7, 30, 90][seed % 4], "side": side,
        "premium": round(1.0 + (seed % 10) * 0.5, 2),
        "size": 500 + (seed * 6151) % 4500,
        "iv": round(0.2 + (seed % 8) * 0.1, 2),
    }


def _filing_payload(ticker: str, summary: str, ftype: str) -> dict:
    return {
        "ticker": ticker, "filing_type": ftype,
        "title": f"{ticker} files {ftype}",
        "summary": summary,
    }


def _press_payload(ticker: str, headline: str) -> dict:
    return {"ticker": ticker, "headline": headline}


def _quote_series_payload(ticker: str, direction: str, seed: int) -> dict:
    """64-step rolling history shaped to imply `direction`.

    The forecaster (Chronos-bolt) on this history will produce a near-linear
    extrapolation; the targets below are written assuming that. Also returns
    the slope used, so target builders can reference exact magnitude.
    """
    n = 64
    base = 100.0 + (seed % 80) * 5
    if direction == "up":
        slope = 0.05 + ((seed * 11) % 100) / 5000.0   # +0.05..+0.07
    elif direction == "down":
        slope = -0.05 - ((seed * 13) % 100) / 5000.0  # -0.07..-0.05
    else:
        slope = ((seed * 17) % 100) / 50000.0 - 0.001  # ~flat
    noise_amp = base * 0.005
    history = []
    for i in range(n):
        # Deterministic pseudo-noise from seed+i
        nseed = (seed * 2654435761 + i * 1664525) % 1_000_000
        noise = (nseed / 500_000.0 - 1.0) * noise_amp
        history.append(round(base + slope * i + noise, 3))
    return {
        "ticker": ticker,
        "window": "1m",
        "history": history,
        "horizon": 12,
        "_slope": slope,         # internal: per-step price change
        "_base": base,            # internal: starting level
        "_noise_amp": noise_amp,  # internal: spread proxy
    }


def _quote_forecast_pct(payload: dict) -> tuple[float, float]:
    """Project forward `horizon` steps and return (mean_pct, sigma_pct).

    Uses the synthetic data's own slope and noise to compute what Chronos
    *should* produce on this history if it's a good linear forecaster.
    """
    horizon = int(payload.get("horizon", 12))
    base = float(payload["_base"])
    slope = float(payload["_slope"])
    noise_amp = float(payload["_noise_amp"])
    last_history_price = base + slope * 63  # last index in 64-step history
    projected_end = last_history_price + slope * horizon
    pct_move = (projected_end - last_history_price) / last_history_price * 100.0
    # σ as percent-of-price, scaled by sqrt(horizon) for AR-noise
    sigma_pct = (noise_amp / last_history_price * 100.0) * (horizon ** 0.5)
    return pct_move, sigma_pct


# ---------- forecaster + integrated-predator targets ----------

def _forecaster_target(
    ticker: str, direction: str,
    pct_move: float | None = None, sigma_pct: float | None = None,
) -> str:
    """Forecaster-herbivore synthesis as XML."""
    if pct_move is not None and abs(pct_move) < 0.2:
        # effectively flat — overwrite direction
        bias = "flat"
        confidence = 0.4
    else:
        bias = direction
        confidence = 0.70
    return emit_synthesis(
        kind="forecaster",
        ticker=ticker,
        bias=bias,
        pct_move=pct_move,
        sigma_pct=sigma_pct,
        confidence=confidence,
    )


def _predator_integrated_target(
    ticker: str,
    direction: str,
    horizon_min: int,
    has_textual: bool,
    has_forecast: bool,
    pct_move: float | None = None,
    sigma_pct: float | None = None,
) -> str:
    """Integrated predator prediction as XML, with evidence referencing the
    upstream herbivores that contributed.
    """
    if not (has_textual or has_forecast):
        return emit_prediction(abstain=True)
    evidence: list[tuple[str, str]] = []
    if has_textual:
        evidence.append(("technical", f"tape consistent with {direction} bias"))
    if has_forecast:
        evidence.append(("forecaster", f"forecast trend {direction}"))
    confidence = 0.70 if (has_textual and has_forecast) else (0.60 if has_forecast else 0.65)
    return emit_prediction(
        ticker=ticker,
        direction=direction,
        pct_move=pct_move,
        horizon_min=horizon_min,
        sigma_pct=sigma_pct,
        confidence=confidence,
        evidence=evidence,
    )


# ---------- generation ----------

def build_scenarios() -> list[Scenario]:
    s: list[Scenario] = []
    seed_counter = [10_000]  # mutable counter for unique RawInput seeds

    def _next_seed() -> int:
        seed_counter[0] += 1
        return seed_counter[0]

    # --- TICKDELTA scenarios: tech eats, fundamental abstains ---
    # 16 tickers × 2 directions × 2 input shapes (ohlcv-only, ohlcv+trade) = 64
    for ticker in TICKERS:
        for direction in ("up", "down"):
            ohlcv_payload_a = _ohlcv_payload(ticker, direction, _next_seed())
            pct_a = _ohlcv_pct_change(ohlcv_payload_a)
            ohlcv = _ri("ohlcv", ohlcv_payload_a, _next_seed())
            s.append(Scenario(
                name=f"tickdelta_{direction}_{ticker}_ohlcv",
                inputs=[ohlcv],
                technical_target=_technical_target(ticker, direction, pct_move=pct_a, vol_note="1m bar movement"),
                fundamental_target=_abstain_target(kind="fundamental"),
                predator_target=_predator_target(ticker, direction, 60, pct_move=pct_a),
                expected_abstain_fundamental=True,
            ))
            ohlcv_payload_b = _ohlcv_payload(ticker, direction, _next_seed())
            pct_b = _ohlcv_pct_change(ohlcv_payload_b)
            ohlcv2 = _ri("ohlcv", ohlcv_payload_b, _next_seed())
            trade = _ri("trades", _trade_payload(ticker, direction, _next_seed()), _next_seed())
            note = "elevated buy-side flow" if direction == "up" else "aggressive sell-side flow"
            s.append(Scenario(
                name=f"tickdelta_{direction}_{ticker}_combo",
                inputs=[ohlcv2, trade],
                technical_target=_technical_target(ticker, direction, pct_move=pct_b, vol_note=note),
                fundamental_target=_abstain_target(kind="fundamental"),
                predator_target=_predator_target(ticker, direction, 60, pct_move=pct_b),
                expected_abstain_fundamental=True,
            ))

    # --- DISCLOSURE scenarios: fundamental eats, technical abstains ---
    # 16 disclosure cases × 4 tickers each = 64
    for i, (summary, direction) in enumerate(DISCLOSURE_CASES):
        # 4 tickers per disclosure case; rotate which tickers
        for j in range(4):
            ticker = TICKERS[(i * 4 + j) % len(TICKERS)]
            ftype = FILING_TYPES[(i + j) % len(FILING_TYPES)]
            filing = _ri("filing",
                         _filing_payload(ticker, summary, ftype),
                         _next_seed())
            horizon = 120 if direction != "flat" else 240
            s.append(Scenario(
                name=f"disclosure_{ticker}_{ftype}_{i:02d}{j}",
                inputs=[filing],
                technical_target=_abstain_target(kind="technical"),
                fundamental_target=_fundamental_target(ticker, summary),
                predator_target=(_predator_target(ticker, direction, horizon)
                                 if direction != "flat"
                                 else emit_prediction(
                                     ticker=ticker, direction="flat",
                                     horizon_min=horizon, confidence=0.5,
                                 )),
                expected_abstain_technical=True,
            ))

    # --- PRESS scenarios: fundamental eats, technical abstains ---
    # 16 press cases (positive + negative)
    PRESS_CASES = [
        ("announces flagship product line",                 "up"),
        ("CFO to depart at end of quarter",                 "down"),
        ("secures multi-year supply agreement",             "up"),
        ("expands data center capacity",                    "up"),
        ("warns of slower customer adoption",               "down"),
        ("partners with leading cloud provider",            "up"),
        ("recalls unit due to manufacturing defect",        "down"),
        ("reports record quarterly bookings",               "up"),
    ]
    for i, (head_tail, direction) in enumerate(PRESS_CASES):
        for j in range(2):
            ticker = TICKERS[(i * 2 + j) % len(TICKERS)]
            press = _ri("press",
                        _press_payload(ticker, f"{ticker} {head_tail}"),
                        _next_seed())
            s.append(Scenario(
                name=f"press_{ticker}_{i:02d}{j}",
                inputs=[press],
                technical_target=_abstain_target(kind="technical"),
                fundamental_target=_fundamental_target(ticker, head_tail),
                predator_target=_predator_target(ticker, direction, 90),
                expected_abstain_technical=True,
            ))

    # --- ANOMALY scenarios: both eat, contested resource ---
    # 16 tickers × 2 directions × options/halt = ~32
    for ticker in TICKERS:
        for direction in ("up", "down"):
            opts = _ri("options",
                       _options_payload(ticker, direction, _next_seed()),
                       _next_seed())
            tnote = ("call-side options flow" if direction == "up"
                     else "put-side options flow")
            fnote = f"options skew suggests information asymmetry on {ticker}"
            s.append(Scenario(
                name=f"anomaly_{ticker}_options_{direction}",
                inputs=[opts],
                # No magnitude available for options-only scenarios.
                technical_target=_technical_target(ticker, direction, vol_note=tnote),
                fundamental_target=_fundamental_target(ticker, fnote, expected_dir=direction),
                predator_target=_predator_target(ticker, direction, 30),
            ))

    # Halt anomalies (less common, only some tickers)
    for ticker in TICKERS[:6]:
        halt = _ri("halt", {"ticker": ticker, "reason": "LULD"}, _next_seed())
        s.append(Scenario(
            name=f"anomaly_{ticker}_halt",
            inputs=[halt],
            technical_target=_technical_target(ticker, "down", vol_note="trading halt event"),
            fundamental_target=_fundamental_target(ticker, f"{ticker} halted; pending material disclosure likely", expected_dir="down"),
            predator_target=_predator_target(ticker, "down", 30),
        ))

    # --- MIXED scenarios: tickdelta + disclosure together ---
    # 8 tickers × up/down = 16
    for i, ticker in enumerate(TICKERS[:8]):
        for direction in ("up", "down"):
            ohlcv_p = _ohlcv_payload(ticker, direction, _next_seed())
            pct = _ohlcv_pct_change(ohlcv_p)
            ohlcv = _ri("ohlcv", ohlcv_p, _next_seed())
            summary, _disc_dir = DISCLOSURE_CASES[(i * 2 + (0 if direction == "up" else 1)) % len(DISCLOSURE_CASES)]
            filing = _ri("filing", _filing_payload(ticker, summary, "8-K"), _next_seed())
            s.append(Scenario(
                name=f"mixed_{ticker}_{direction}",
                inputs=[ohlcv, filing],
                technical_target=_technical_target(ticker, direction, pct_move=pct, vol_note="tape consistent with disclosure"),
                fundamental_target=_fundamental_target(ticker, summary, expected_dir=direction),
                predator_target=_predator_target(ticker, direction, 60, pct_move=pct),
            ))

    # --- EMPTY scenarios: both abstain ---
    for ticker in TICKERS[:5]:
        book = _ri("book",
                   {"ticker": ticker, "bid": 100.0, "bid_size": 100,
                    "ask": 100.05, "ask_size": 100},
                   _next_seed())
        s.append(Scenario(
            name=f"empty_{ticker}_book",
            inputs=[book],
            technical_target=_abstain_target(kind="technical"),
            fundamental_target=_abstain_target(kind="fundamental"),
            predator_target=_abstain_pred_target(),
            expected_abstain_fundamental=True,
            expected_abstain_predator=True,
        ))

    # --- FORECAST-ONLY scenarios: forecaster eats, others abstain ---
    # 16 tickers × 2 directions = 32 baseline. Then a few flat ones.
    for ticker in TICKERS:
        for direction in ("up", "down"):
            qs_p = _quote_series_payload(ticker, direction, _next_seed())
            f_pct, f_sigma = _quote_forecast_pct(qs_p)
            qs = _ri("quote_series", qs_p, _next_seed())
            s.append(Scenario(
                name=f"forecast_only_{direction}_{ticker}",
                inputs=[qs],
                technical_target=_abstain_target(kind="technical"),
                fundamental_target=_abstain_target(kind="fundamental"),
                forecaster_target=_forecaster_target(ticker, direction, pct_move=f_pct, sigma_pct=f_sigma),
                predator_target=_predator_integrated_target(
                    ticker, direction, 60, has_textual=False, has_forecast=True,
                    pct_move=f_pct, sigma_pct=f_sigma,
                ),
                expected_abstain_technical=True,
                expected_abstain_fundamental=True,
            ))
    # Flat forecasts (low-conviction): 4 tickers
    for ticker in TICKERS[:4]:
        qs_p = _quote_series_payload(ticker, "flat", _next_seed())
        f_pct, f_sigma = _quote_forecast_pct(qs_p)
        qs = _ri("quote_series", qs_p, _next_seed())
        s.append(Scenario(
            name=f"forecast_only_flat_{ticker}",
            inputs=[qs],
            technical_target=_abstain_target(kind="technical"),
            fundamental_target=_abstain_target(kind="fundamental"),
            forecaster_target=_forecaster_target(ticker, "flat", pct_move=f_pct, sigma_pct=f_sigma),
            predator_target=emit_prediction(
                ticker=ticker, direction="flat", pct_move=f_pct,
                horizon_min=60, sigma_pct=f_sigma, confidence=0.4,
                evidence=[("forecaster", "forecast trend roughly flat")],
            ),
            expected_abstain_technical=True,
            expected_abstain_fundamental=True,
        ))

    # --- TICKDELTA + FORECAST scenarios: technical + forecaster integrate ---
    # 12 tickers × 2 directions = 24
    for ticker in TICKERS[:12]:
        for direction in ("up", "down"):
            ohlcv_p = _ohlcv_payload(ticker, direction, _next_seed())
            tape_pct = _ohlcv_pct_change(ohlcv_p)
            ohlcv = _ri("ohlcv", ohlcv_p, _next_seed())
            qs_p = _quote_series_payload(ticker, direction, _next_seed())
            f_pct, f_sigma = _quote_forecast_pct(qs_p)
            qs = _ri("quote_series", qs_p, _next_seed())
            note = "tape and forecast aligned"
            s.append(Scenario(
                name=f"tickdelta_forecast_{direction}_{ticker}",
                inputs=[ohlcv, qs],
                technical_target=_technical_target(ticker, direction, pct_move=tape_pct, vol_note=note),
                fundamental_target=_abstain_target(kind="fundamental"),
                forecaster_target=_forecaster_target(ticker, direction, pct_move=f_pct, sigma_pct=f_sigma),
                predator_target=_predator_integrated_target(
                    ticker, direction, 60, has_textual=True, has_forecast=True,
                    pct_move=f_pct, sigma_pct=f_sigma,
                ),
                expected_abstain_fundamental=True,
            ))

    # --- DISCLOSURE + FORECAST scenarios: fundamental + forecaster integrate ---
    # 8 disclosure cases × 2 tickers = 16
    for i in range(8):
        summary, direction = DISCLOSURE_CASES[i]
        if direction == "flat":
            direction = "up"  # bias for clearer learning signal
        for j in range(2):
            ticker = TICKERS[(i * 2 + j) % len(TICKERS)]
            filing = _ri("filing",
                         _filing_payload(ticker, summary, "8-K"),
                         _next_seed())
            qs_p = _quote_series_payload(ticker, direction, _next_seed())
            f_pct, f_sigma = _quote_forecast_pct(qs_p)
            qs = _ri("quote_series", qs_p, _next_seed())
            s.append(Scenario(
                name=f"disclosure_forecast_{ticker}_{i:02d}{j}",
                inputs=[filing, qs],
                technical_target=_abstain_target(kind="technical"),
                fundamental_target=_fundamental_target(ticker, summary, expected_dir=direction),
                forecaster_target=_forecaster_target(ticker, direction, pct_move=f_pct, sigma_pct=f_sigma),
                predator_target=_predator_integrated_target(
                    ticker, direction, 120, has_textual=True, has_forecast=True,
                    pct_move=f_pct, sigma_pct=f_sigma,
                ),
                expected_abstain_technical=True,
            ))

    # --- TRIFECTA scenarios: all three herbivore kinds eat ---
    # 6 cases — forecaster + technical + fundamental + predator integrates everything.
    TRIFECTA_CASES = [
        ("AAPL", "up",   "cloud growth above consensus"),
        ("NVDA", "up",   "data center capacity expansion in EMEA"),
        ("META", "down", "guidance reduction citing supply constraints"),
        ("MSFT", "up",   "multi-year supply agreement with major customer"),
        ("TSLA", "down", "regulatory probe into accounting practices opened"),
        ("AMZN", "up",   "revenue beat with strong ad pricing"),
    ]
    for ticker, direction, summary in TRIFECTA_CASES:
        ohlcv_p = _ohlcv_payload(ticker, direction, _next_seed())
        tape_pct = _ohlcv_pct_change(ohlcv_p)
        ohlcv = _ri("ohlcv", ohlcv_p, _next_seed())
        filing = _ri("filing", _filing_payload(ticker, summary, "8-K"), _next_seed())
        qs_p = _quote_series_payload(ticker, direction, _next_seed())
        f_pct, f_sigma = _quote_forecast_pct(qs_p)
        qs = _ri("quote_series", qs_p, _next_seed())
        sign = "+" if f_pct >= 0 else ""
        s.append(Scenario(
            name=f"trifecta_{ticker}_{direction}",
            inputs=[ohlcv, filing, qs],
            technical_target=_technical_target(ticker, direction, pct_move=tape_pct,
                                               vol_note="tape consistent with disclosure and forecast"),
            fundamental_target=_fundamental_target(ticker, summary, expected_dir=direction),
            forecaster_target=_forecaster_target(ticker, direction, pct_move=f_pct, sigma_pct=f_sigma),
            predator_target=emit_prediction(
                ticker=ticker, direction=direction, pct_move=f_pct,
                horizon_min=60, sigma_pct=f_sigma, confidence=0.8,
                evidence=[
                    ("technical", f"tape consistent with {direction} bias"),
                    ("fundamental", f"disclosure thesis: {summary}"),
                    ("forecaster", f"forecast trend {direction}"),
                ],
            ),
        ))

    # --- EVAL scenarios (held out, distinct combos not in training) ---
    eval_specs = [
        # tickdelta-only with unusual ticker rotation
        ("eval_tickdelta_up_AAPL_ohlcv", [
            _ri("ohlcv", _ohlcv_payload("AAPL", "up", 99001), 99001),
        ], "AAPL", "up", "fundamental_abstain"),
        ("eval_tickdelta_down_NVDA_combo", [
            _ri("ohlcv", _ohlcv_payload("NVDA", "down", 99002), 99002),
            _ri("trades", _trade_payload("NVDA", "down", 99003), 99003),
        ], "NVDA", "down", "fundamental_abstain"),
        # disclosure-only with held-out summaries
        ("eval_disclosure_GOOG_settlement", [
            _ri("filing", _filing_payload("GOOG", "antitrust settlement reached", "8-K"), 99004),
        ], "GOOG", "up", "technical_abstain_disclosure"),
        ("eval_disclosure_TSLA_recall", [
            _ri("filing", _filing_payload("TSLA", "recalls unit due to manufacturing defect", "8-K"), 99005),
        ], "TSLA", "down", "technical_abstain_disclosure"),
        # press
        ("eval_press_MSFT_partnership", [
            _ri("press", _press_payload("MSFT", "MSFT partners with leading cloud provider"), 99006),
        ], "MSFT", "up", "technical_abstain_press"),
        # anomaly options
        ("eval_anomaly_AMD_options_up", [
            _ri("options", _options_payload("AMD", "up", 99007), 99007),
        ], "AMD", "up", "anomaly"),
        ("eval_anomaly_META_options_down", [
            _ri("options", _options_payload("META", "down", 99008), 99008),
        ], "META", "down", "anomaly"),
        # mixed
        ("eval_mixed_AMZN_up", [
            _ri("ohlcv", _ohlcv_payload("AMZN", "up", 99009), 99009),
            _ri("filing", _filing_payload("AMZN", "cloud growth above consensus", "8-K"), 99010),
        ], "AMZN", "up", "mixed"),
        # empty
        ("eval_empty_NFLX", [
            _ri("book", {"ticker": "NFLX", "bid": 400.0, "bid_size": 200,
                         "ask": 400.10, "ask_size": 200}, 99011),
        ], "NFLX", "flat", "empty"),
        # forecaster-only (held out: tickers we didn't quote-train)
        ("eval_forecast_only_QCOM_up", [
            _ri("quote_series", _quote_series_payload("QCOM", "up", 99020), 99020),
        ], "QCOM", "up", "forecast_only"),
        ("eval_forecast_only_INTC_down", [
            _ri("quote_series", _quote_series_payload("INTC", "down", 99021), 99021),
        ], "INTC", "down", "forecast_only"),
        # tickdelta + forecast (held out)
        ("eval_tickdelta_forecast_AVGO_up", [
            _ri("ohlcv", _ohlcv_payload("AVGO", "up", 99022), 99022),
            _ri("quote_series", _quote_series_payload("AVGO", "up", 99023), 99023),
        ], "AVGO", "up", "tickdelta_forecast"),
        # disclosure + forecast (held out)
        ("eval_disclosure_forecast_ORCL_down", [
            _ri("filing", _filing_payload("ORCL", "operational restructuring with workforce reduction", "8-K"), 99024),
            _ri("quote_series", _quote_series_payload("ORCL", "down", 99025), 99025),
        ], "ORCL", "down", "disclosure_forecast"),
        # trifecta (held out)
        ("eval_trifecta_CRM_up", [
            _ri("ohlcv", _ohlcv_payload("CRM", "up", 99026), 99026),
            _ri("filing", _filing_payload("CRM", "dividend increase announced ahead of earnings", "8-K"), 99027),
            _ri("quote_series", _quote_series_payload("CRM", "up", 99028), 99028),
        ], "CRM", "up", "trifecta"),
    ]

    for name, inputs, ticker, direction, kind in eval_specs:
        # Try to extract magnitudes from the eval input payloads.
        tape_pct = None
        for inp in inputs:
            if inp.source == "ohlcv":
                tape_pct = _ohlcv_pct_change(inp.payload)
                break
        f_pct = f_sigma = None
        for inp in inputs:
            if inp.source == "quote_series":
                f_pct, f_sigma = _quote_forecast_pct(inp.payload)
                break

        if kind == "fundamental_abstain":
            note = "1m bar movement" if len(inputs) == 1 else "elevated flow"
            s.append(Scenario(
                name=name, inputs=inputs,
                technical_target=_technical_target(ticker, direction, pct_move=tape_pct, vol_note=note),
                fundamental_target=_abstain_target(kind="fundamental"),
                predator_target=_predator_target(ticker, direction, 60, pct_move=tape_pct),
                expected_abstain_fundamental=True,
            ))
        elif kind == "technical_abstain_disclosure":
            summary = inputs[0].payload["summary"]
            s.append(Scenario(
                name=name, inputs=inputs,
                technical_target=_abstain_target(kind="technical"),
                fundamental_target=_fundamental_target(ticker, summary, expected_dir=direction),
                predator_target=_predator_target(ticker, direction, 120),
                expected_abstain_technical=True,
            ))
        elif kind == "technical_abstain_press":
            head = inputs[0].payload["headline"].split(maxsplit=1)[1]
            s.append(Scenario(
                name=name, inputs=inputs,
                technical_target=_abstain_target(kind="technical"),
                fundamental_target=_fundamental_target(ticker, head, expected_dir=direction),
                predator_target=_predator_target(ticker, direction, 90),
                expected_abstain_technical=True,
            ))
        elif kind == "anomaly":
            tnote = "call-side options flow" if direction == "up" else "put-side options flow"
            fnote = f"options skew suggests information asymmetry on {ticker}"
            s.append(Scenario(
                name=name, inputs=inputs,
                technical_target=_technical_target(ticker, direction, vol_note=tnote),
                fundamental_target=_fundamental_target(ticker, fnote, expected_dir=direction),
                predator_target=_predator_target(ticker, direction, 30),
            ))
        elif kind == "mixed":
            summary = inputs[1].payload["summary"]
            s.append(Scenario(
                name=name, inputs=inputs,
                technical_target=_technical_target(ticker, direction, pct_move=tape_pct, vol_note="tape consistent with disclosure"),
                fundamental_target=_fundamental_target(ticker, summary, expected_dir=direction),
                predator_target=_predator_target(ticker, direction, 60, pct_move=tape_pct),
            ))
        elif kind == "empty":
            s.append(Scenario(
                name=name, inputs=inputs,
                technical_target=_abstain_target(kind="technical"),
                fundamental_target=_abstain_target(kind="fundamental"),
                predator_target=_abstain_pred_target(),
                expected_abstain_fundamental=True,
                expected_abstain_predator=True,
            ))
        elif kind == "forecast_only":
            s.append(Scenario(
                name=name, inputs=inputs,
                technical_target=_abstain_target(kind="technical"),
                fundamental_target=_abstain_target(kind="fundamental"),
                forecaster_target=_forecaster_target(ticker, direction, pct_move=f_pct, sigma_pct=f_sigma),
                predator_target=_predator_integrated_target(
                    ticker, direction, 60, has_textual=False, has_forecast=True,
                    pct_move=f_pct, sigma_pct=f_sigma,
                ),
                expected_abstain_technical=True,
                expected_abstain_fundamental=True,
            ))
        elif kind == "tickdelta_forecast":
            s.append(Scenario(
                name=name, inputs=inputs,
                technical_target=_technical_target(ticker, direction, pct_move=tape_pct, vol_note="tape and forecast aligned"),
                fundamental_target=_abstain_target(kind="fundamental"),
                forecaster_target=_forecaster_target(ticker, direction, pct_move=f_pct, sigma_pct=f_sigma),
                predator_target=_predator_integrated_target(
                    ticker, direction, 60, has_textual=True, has_forecast=True,
                    pct_move=f_pct, sigma_pct=f_sigma,
                ),
                expected_abstain_fundamental=True,
            ))
        elif kind == "disclosure_forecast":
            summary = inputs[0].payload["summary"]
            s.append(Scenario(
                name=name, inputs=inputs,
                technical_target=_abstain_target(kind="technical"),
                fundamental_target=_fundamental_target(ticker, summary, expected_dir=direction),
                forecaster_target=_forecaster_target(ticker, direction, pct_move=f_pct, sigma_pct=f_sigma),
                predator_target=_predator_integrated_target(
                    ticker, direction, 120, has_textual=True, has_forecast=True,
                    pct_move=f_pct, sigma_pct=f_sigma,
                ),
                expected_abstain_technical=True,
            ))
        elif kind == "trifecta":
            summary = inputs[1].payload["summary"]
            sign = "+" if (f_pct is not None and f_pct >= 0) else ""
            mag = f" {sign}{f_pct:.2f}%" if f_pct is not None else ""
            sig = f" σ≈{f_sigma:.2f}%" if f_sigma is not None else ""
            s.append(Scenario(
                name=name, inputs=inputs,
                technical_target=_technical_target(ticker, direction, pct_move=tape_pct,
                                                  vol_note="tape consistent with disclosure and forecast"),
                fundamental_target=_fundamental_target(ticker, summary, expected_dir=direction),
                forecaster_target=_forecaster_target(ticker, direction, pct_move=f_pct, sigma_pct=f_sigma),
                predator_target=emit_prediction(
                    ticker=ticker, direction=direction, pct_move=f_pct,
                    horizon_min=60, sigma_pct=f_sigma, confidence=0.8,
                    evidence=[
                        ("technical", f"tape consistent with {direction} bias"),
                        ("fundamental", f"disclosure thesis: {summary}"),
                        ("forecaster", f"forecast trend {direction}"),
                    ],
                ),
            ))

    # Post-process: every scenario gets an interrogator_target filled in
    # (computed exactly from quant payloads when present, abstain when not).
    _fill_interrogator_targets(s)
    return s


def split(scenarios: list[Scenario]) -> tuple[list[Scenario], list[Scenario]]:
    train = [s for s in scenarios if not s.is_eval]
    eval_ = [s for s in scenarios if s.is_eval]
    return train, eval_
