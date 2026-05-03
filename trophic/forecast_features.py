"""Numeric feature extraction from a Chronos-Bolt forecast.

Replaces the text-pool oracle path that fed predators a sentence describing
the forecast. Instead we expose the actual probabilistic forecast as a
fixed-length numeric vector that a learnable projection head can read
through the trough.

The features chosen below are the ones a directional predictor would
actually want:
  - point trajectory (mean over horizon)
  - uncertainty bands (q10/q90 spread per step)
  - aggregate stats: total drift, max-drawdown-of-mean,
    confidence-weighted direction score
  - signal-to-noise ratio (drift / spread)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forecaster_host import ForecasterHost, ForecastResult


# Number of features emitted per forecast — must match the projection head input dim.
N_FORECAST_FEATURES = 32


def features_from_forecast(fr: ForecastResult, last_price: float) -> list[float]:
    """Extract a fixed-length numeric vector from a ForecastResult.

    Args:
        fr: Chronos output (mean[H], quantiles[H, Q]).
        last_price: most recent observed close, used for normalization.

    Returns:
        32 floats. NaN-safe; missing data → zeros.
    """
    H = max(fr.horizon, 1)
    mean = np.asarray(fr.mean, dtype=np.float64)
    if mean.shape[0] == 0:
        return [0.0] * N_FORECAST_FEATURES
    qs = np.asarray(fr.quantiles, dtype=np.float64)  # [H, Q]

    # Normalize prices to relative-return scale so different tickers feed
    # the same head with comparable magnitudes.
    base = max(abs(last_price), 1e-6)

    rel_mean = mean / base - 1.0  # [H], in fractional return units
    if qs.size:
        rel_q = qs / base - 1.0  # [H, Q]
        try:
            i_lo = fr.quantile_levels.index(0.1)
            i_hi = fr.quantile_levels.index(0.9)
            spread = rel_q[:, i_hi] - rel_q[:, i_lo]  # [H]
        except (ValueError, IndexError):
            spread = np.zeros(H)
    else:
        spread = np.zeros(H)

    # Point-trajectory features (8): subsampled to fixed 8 steps regardless
    # of true horizon. Linear-interpolate so different horizons map cleanly.
    idxs = np.linspace(0, H - 1, num=8)
    traj = np.interp(idxs, np.arange(H), rel_mean)
    traj = np.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0)

    # Uncertainty-band features (8): same subsampling on spread.
    spread_traj = np.interp(idxs, np.arange(H), spread)
    spread_traj = np.nan_to_num(spread_traj, nan=0.0, posinf=0.0, neginf=0.0)

    # Aggregate features (16):
    drift = float(rel_mean[-1])                                # 1: total drift
    sign_drift = float(np.sign(drift))                          # 2: direction
    abs_drift = float(abs(drift))                               # 3: magnitude
    avg_spread = float(np.mean(spread)) if spread.size else 0.0 # 4: avg uncertainty
    final_spread = float(spread[-1]) if spread.size else 0.0    # 5: final-step uncertainty
    snr = drift / max(avg_spread, 1e-6)                          # 6: signal-to-noise
    drift_per_step = drift / H                                   # 7: per-step drift
    max_excursion = float(np.max(np.abs(rel_mean)))             # 8: max excursion mean
    early_drift = float(rel_mean[min(2, H - 1)])                # 9: drift at step 3
    late_drift = float(rel_mean[-1] - rel_mean[max(H // 2, 0)]) # 10: late-half drift
    monotonicity = float(np.mean(np.sign(np.diff(rel_mean))))    # 11: how monotone (-1..+1)
    volatility = float(np.std(np.diff(rel_mean))) if H > 1 else 0.0  # 12: realized vol of mean
    spread_growth = float(spread[-1] - spread[0]) if spread.size > 1 else 0.0  # 13
    confidence_weighted_dir = sign_drift * (1.0 / (1.0 + avg_spread))  # 14
    history_norm_factor = float(np.tanh(base / 100.0))           # 15: ticker-scale anchor
    horizon_norm = float(np.tanh(H / 12.0))                      # 16: horizon length anchor

    aggregates = [
        drift, sign_drift, abs_drift, avg_spread, final_spread, snr,
        drift_per_step, max_excursion, early_drift, late_drift,
        monotonicity, volatility, spread_growth, confidence_weighted_dir,
        history_norm_factor, horizon_norm,
    ]
    aggregates = [float(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)) for x in aggregates]

    out = list(traj.tolist()) + list(spread_traj.tolist()) + aggregates
    assert len(out) == N_FORECAST_FEATURES, f"expected {N_FORECAST_FEATURES}, got {len(out)}"
    return out


def chronos_features_from_bars(bars: list[dict]) -> list[float]:
    """Run Chronos on the closes from a bar list and extract features.

    Returns 32 floats. If Chronos fails or bars are empty, returns zeros.
    """
    if not bars:
        return [0.0] * N_FORECAST_FEATURES
    try:
        closes = [float(b.get("close", 0.0)) for b in bars]
    except (TypeError, ValueError):
        return [0.0] * N_FORECAST_FEATURES
    if len(closes) < 2:
        return [0.0] * N_FORECAST_FEATURES
    last_price = closes[-1]
    fc = ForecasterHost.get()
    fr = fc.forecast(closes)
    return features_from_forecast(fr, last_price)
