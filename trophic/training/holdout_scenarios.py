"""Held-out test scenarios — fresh tickers and disclosure cases not seen in
training. Used for the honest paper-grade reward number after early-stopping
based on the 14-scenario dev set.

Generation policy:
  - Tickers: 16 large-cap names DISJOINT from `scenarios.TICKERS`. This is the
    strongest train/test separation we have for a fintech-prediction task —
    the model must generalize to new symbols.
  - Disclosure cases: 16 fresh phrasings with the same directional grammar.
  - Same per-source builders (_ohlcv_payload, _trade_payload, _options_payload,
    _filing_payload, _press_payload, _quote_series_payload) reused so the
    distribution of producer outputs is matched — only the ticker/disclosure
    surface vocabulary differs.

Output is a list[Scenario] disjoint from `build_scenarios()`. Use it via:

    from trophic.training.scenarios import build_scenarios
    from trophic.training.holdout_scenarios import build_holdout_scenarios
    train, dev = split(build_scenarios())
    test = build_holdout_scenarios()
"""
from __future__ import annotations

from .scenarios import (
    Scenario,
    _abstain_pred_target,
    _abstain_target,
    _filing_payload,
    _fundamental_target,
    _interrogator_target,
    _ohlcv_payload,
    _ohlcv_pct_change,
    _options_payload,
    _predator_integrated_target,
    _predator_target,
    _press_payload,
    _quote_forecast_pct,
    _quote_series_payload,
    _ri,
    _technical_target,
    _trade_payload,
    _forecaster_target,
    _fill_interrogator_targets,
)


# ---------- held-out corpora ----------

# 16 large-cap tickers, fully DISJOINT from training TICKERS (16 there too).
# Mix of mega-cap, semis, financials, consumer, healthcare to match training
# diversity while remaining visually distinguishable from the train set.
HOLDOUT_TICKERS = [
    "BRK.B", "V", "JNJ", "WMT", "PG", "UNH", "HD", "BAC",
    "DIS", "PYPL", "T", "PFE", "KO", "PEP", "XOM", "CVX",
]

HOLDOUT_FILING_TYPES = ["8-K", "10-Q", "13D", "S-1"]

# 16 disclosure cases with fresh phrasing but the same directional grammar.
HOLDOUT_DISCLOSURE_CASES = [
    ("preliminary results exceed prior guidance",                "up"),
    ("inventory write-down disclosed in segment review",         "down"),
    ("FDA approval granted for lead therapy",                    "up"),
    ("audit committee identifies revenue recognition concern",   "down"),
    ("stock buyback authorization expanded materially",          "up"),
    ("regulatory order halts product line in key market",        "down"),
    ("strategic partnership signed with industry leader",        "up"),
    ("class-action settlement reached at favorable terms",       "up"),
    ("cyber breach disclosed affecting customer records",        "down"),
    ("patent infringement ruling against company filed",         "down"),
    ("operating margin guidance raised on cost discipline",      "up"),
    ("CEO succession plan announced amid health concerns",       "down"),
    ("manufacturing facility expansion in APAC region",          "up"),
    ("customer concentration risk flagged in risk factors",      "flat"),
    ("M&A target acquired at accretive multiple",                "up"),
    ("currency headwind reduces top-line by mid-single-digits",  "flat"),
]


# ---------- generation ----------

def build_holdout_scenarios() -> list[Scenario]:
    """Held-out test scenarios. Disjoint tickers + disclosures from training.

    Smaller than train (we don't need 285) — aim for ~60 cases that cover the
    same scenario archetypes (tickdelta / disclosure / press / anomaly /
    forecast / mixed / trifecta) so the test distribution mirrors training in
    *shape* but not *vocabulary*. All scenario names are prefixed with
    `holdout_` so they never collide with `eval_` or training names.
    """
    s: list[Scenario] = []
    seed_counter = [50_000]  # well above scenarios.py's 10_000 base to avoid id collisions

    def _next_seed() -> int:
        seed_counter[0] += 1
        return seed_counter[0]

    # --- TICKDELTA holdout: 8 tickers × 2 directions × 2 shapes = 32 ---
    # Use the first 8 tickers for tickdelta; remaining 8 split across the
    # other archetypes to keep the set compact (~60 total).
    for ticker in HOLDOUT_TICKERS[:8]:
        for direction in ("up", "down"):
            ohlcv_payload_a = _ohlcv_payload(ticker, direction, _next_seed())
            pct_a = _ohlcv_pct_change(ohlcv_payload_a)
            ohlcv = _ri("ohlcv", ohlcv_payload_a, _next_seed())
            s.append(Scenario(
                name=f"holdout_tickdelta_{direction}_{ticker}_ohlcv",
                inputs=[ohlcv],
                technical_target=_technical_target(
                    ticker, direction, pct_move=pct_a, vol_note="1m bar movement"
                ),
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
                name=f"holdout_tickdelta_{direction}_{ticker}_combo",
                inputs=[ohlcv2, trade],
                technical_target=_technical_target(
                    ticker, direction, pct_move=pct_b, vol_note=note
                ),
                fundamental_target=_abstain_target(kind="fundamental"),
                predator_target=_predator_target(ticker, direction, 60, pct_move=pct_b),
                expected_abstain_fundamental=True,
            ))

    # --- DISCLOSURE holdout: cycle through holdout disclosure cases ---
    # 8 cases × 1 ticker each = 8
    for case_idx, (summary, expected_dir) in enumerate(HOLDOUT_DISCLOSURE_CASES[:8]):
        ticker = HOLDOUT_TICKERS[8 + (case_idx % 8)]
        ftype = HOLDOUT_FILING_TYPES[case_idx % len(HOLDOUT_FILING_TYPES)]
        filing = _ri("filing", _filing_payload(ticker, summary, ftype), _next_seed())
        s.append(Scenario(
            name=f"holdout_disclosure_{ticker}_{ftype}_{case_idx:03d}",
            inputs=[filing],
            technical_target=_abstain_target(kind="technical"),
            fundamental_target=_fundamental_target(ticker, summary, expected_dir),
            predator_target=_predator_target(
                ticker,
                expected_dir if expected_dir != "flat" else "up",
                120,
            ) if expected_dir != "flat" else _abstain_pred_target(),
            expected_abstain_technical=True,
        ))

    # --- PRESS holdout: 4 cases ---
    press_cases = [
        ("BRK.B", "Berkshire commits $5B to renewable infrastructure", "up"),
        ("WMT", "Walmart reports record holiday e-commerce volume", "up"),
        ("PFE", "Pfizer pauses pivotal trial pending safety review", "down"),
        ("XOM", "Exxon raises capex guidance on basin acquisitions", "up"),
    ]
    for ticker, headline, expected_dir in press_cases:
        press = _ri("press", _press_payload(ticker, headline), _next_seed())
        s.append(Scenario(
            name=f"holdout_press_{ticker}_{headline.split()[0].lower()}",
            inputs=[press],
            technical_target=_abstain_target(kind="technical"),
            fundamental_target=_fundamental_target(ticker, headline, expected_dir),
            predator_target=_predator_target(ticker, expected_dir, 90),
            expected_abstain_technical=True,
        ))

    # --- ANOMALY (options) holdout: 4 cases ---
    for ticker in ["V", "DIS", "PYPL", "BAC"]:
        for direction in ("up", "down"):
            options_payload = _options_payload(ticker, direction, _next_seed())
            options = _ri("options", options_payload, _next_seed())
            s.append(Scenario(
                name=f"holdout_anomaly_{ticker}_options_{direction}",
                inputs=[options],
                technical_target=_technical_target(
                    ticker, direction,
                    pct_move=None,
                    vol_note=f"unusual options activity (premium {direction})",
                ),
                fundamental_target=_abstain_target(kind="fundamental"),
                predator_target=_predator_target(ticker, direction, 30),
                expected_abstain_fundamental=True,
            ))

    # --- FORECAST holdout: quantitative forecaster path ---
    # 8 cases (4 tickers × 2 directions)
    for ticker in HOLDOUT_TICKERS[:4]:
        for direction in ("up", "down"):
            qpayload = _quote_series_payload(ticker, direction, _next_seed())
            qts = _ri("quote_series", qpayload, _next_seed())
            mu_pct, spread_pct = _quote_forecast_pct(qpayload)
            s.append(Scenario(
                name=f"holdout_forecast_only_{direction}_{ticker}",
                inputs=[qts],
                technical_target=_abstain_target(kind="technical"),
                fundamental_target=_abstain_target(kind="fundamental"),
                forecaster_target=_forecaster_target(ticker, mu_pct, spread_pct, 60),
                predator_target=_predator_target(
                    ticker, direction, 60,
                    pct_move=mu_pct, sigma=spread_pct,
                ),
                expected_abstain_technical=True,
                expected_abstain_fundamental=True,
            ))

    # --- MIXED holdout: tickdelta + disclosure (both herbivores eat) ---
    # 4 cases
    mixed_cases = [
        ("KO",  "up",   "preliminary results exceed prior guidance"),
        ("PEP", "down", "inventory write-down disclosed in segment review"),
        ("UNH", "up",   "FDA approval granted for lead therapy"),
        ("HD",  "down", "regulatory order halts product line in key market"),
    ]
    for ticker, direction, summary in mixed_cases:
        ohlcv_p = _ohlcv_payload(ticker, direction, _next_seed())
        pct = _ohlcv_pct_change(ohlcv_p)
        ohlcv = _ri("ohlcv", ohlcv_p, _next_seed())
        filing = _ri("filing", _filing_payload(ticker, summary, "8-K"), _next_seed())
        s.append(Scenario(
            name=f"holdout_mixed_{ticker}_{direction}",
            inputs=[ohlcv, filing],
            technical_target=_technical_target(
                ticker, direction, pct_move=pct, vol_note="1m bar movement"
            ),
            fundamental_target=_fundamental_target(ticker, summary, direction),
            predator_target=_predator_target(ticker, direction, 60, pct_move=pct),
        ))

    # --- TRIFECTA holdout: tickdelta + disclosure + forecast (all three eat) ---
    # 4 cases
    tri_cases = [
        ("BRK.B", "up",   "stock buyback authorization expanded materially"),
        ("WMT",   "down", "audit committee identifies revenue recognition concern"),
        ("JNJ",   "up",   "patent infringement ruling against company filed"),  # contrived contrast — keep it
        ("CVX",   "down", "currency headwind reduces top-line by mid-single-digits"),
    ]
    for ticker, direction, summary in tri_cases:
        ohlcv_p = _ohlcv_payload(ticker, direction, _next_seed())
        pct = _ohlcv_pct_change(ohlcv_p)
        ohlcv = _ri("ohlcv", ohlcv_p, _next_seed())
        filing = _ri("filing", _filing_payload(ticker, summary, "10-Q"), _next_seed())
        qpayload = _quote_series_payload(ticker, direction, _next_seed())
        qts = _ri("quote_series", qpayload, _next_seed())
        mu_pct, spread_pct = _quote_forecast_pct(qpayload)
        s.append(Scenario(
            name=f"holdout_trifecta_{ticker}_{direction}",
            inputs=[ohlcv, filing, qts],
            technical_target=_technical_target(
                ticker, direction, pct_move=pct, vol_note="1m bar movement"
            ),
            fundamental_target=_fundamental_target(ticker, summary, direction),
            forecaster_target=_forecaster_target(ticker, mu_pct, spread_pct, 60),
            predator_target=_predator_integrated_target(
                ticker, direction, summary, mu_pct, spread_pct, 60,
            ),
        ))

    # Fill interrogator targets where applicable
    _fill_interrogator_targets(s)

    return s
