"""Singleton wrapper for chronos-bolt forecasting.

Mirrors ModelHost shape so the rest of the system can treat the forecaster
as just another model resource. Loaded once on cuda; multiple herbivores
share the instance.

forecast(history, horizon=None, quantiles=...) → ForecastResult
  - history: list of floats (or 1d tensor) of length >= 2
  - horizon: int prediction length, defaults to config
  - quantiles: levels to return (default [0.1, 0.5, 0.9])

Returns mean point forecast + per-quantile bands. The forecast is
*native numeric* — no tokens, no hidden states. Downstream agents render
it to text or feed it as numeric features depending on the consumer.
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .config import DEFAULT_CONFIG, ModelConfig


@dataclass
class ForecastResult:
    mean: list[float]                 # [horizon]
    quantile_levels: list[float]      # [Q]
    quantiles: list[list[float]]      # [horizon, Q]
    horizon: int
    history_len: int

    def render_text(self, ticker: str = "?", window: str = "1m") -> str:
        """Compact, deterministic textual rendering for the rendering bottleneck."""
        h = len(self.mean)
        last = self.mean[-1]
        first = self.mean[0]
        delta_pct = (last - first) / max(abs(first), 1e-6) * 100.0
        # Pick the q10/q90 of the final-step prediction for spread
        try:
            q_idx_lo = self.quantile_levels.index(0.1)
            q_idx_hi = self.quantile_levels.index(0.9)
            lo = self.quantiles[-1][q_idx_lo]
            hi = self.quantiles[-1][q_idx_hi]
            spread = hi - lo
        except ValueError:
            spread = 0.0
        return (
            f"FORECAST {ticker} {window} h={h}: "
            f"mean[0]={first:.3f} mean[-1]={last:.3f} "
            f"Δ={delta_pct:+.2f}% q10/q90 spread={spread:.3f}"
        )


class ForecasterHost:
    _instance: Optional["ForecasterHost"] = None
    _lock = threading.Lock()

    def __init__(self, cfg: ModelConfig | None = None):
        self.cfg = cfg or DEFAULT_CONFIG.model
        self._pipe = None
        if not self.cfg.mock:
            self._load()

    @classmethod
    def get(cls, cfg: ModelConfig | None = None) -> "ForecasterHost":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(cfg)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    def _load(self) -> None:
        from chronos import ChronosBoltPipeline

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = dtype_map.get(self.cfg.dtype, torch.bfloat16)
        self._pipe = ChronosBoltPipeline.from_pretrained(
            self.cfg.forecaster_model_id,
            device_map=self.cfg.device,
            torch_dtype=dtype,
        )

    def forecast(
        self,
        history: list[float] | torch.Tensor,
        horizon: int | None = None,
        quantile_levels: list[float] | None = None,
    ) -> ForecastResult:
        horizon = horizon or self.cfg.forecaster_horizon
        quantile_levels = quantile_levels or [0.1, 0.5, 0.9]

        if self.cfg.mock:
            return self._mock_forecast(history, horizon, quantile_levels)

        if isinstance(history, list):
            hist = torch.tensor(history, dtype=torch.float32)
        else:
            hist = history.float()
        if hist.dim() == 1:
            hist_in = hist
        else:
            hist_in = hist.flatten()

        quantiles, mean = self._pipe.predict_quantiles(
            inputs=hist_in,
            prediction_length=horizon,
            quantile_levels=quantile_levels,
        )
        # Shapes: mean=[1, horizon]  quantiles=[1, horizon, Q]
        mean_list = mean[0].float().cpu().tolist()
        q_list = quantiles[0].float().cpu().tolist()  # [horizon, Q]
        return ForecastResult(
            mean=mean_list,
            quantile_levels=quantile_levels,
            quantiles=q_list,
            horizon=horizon,
            history_len=int(hist_in.shape[0]),
        )

    # ---------- mock ----------

    def _mock_forecast(
        self,
        history: list[float] | torch.Tensor,
        horizon: int,
        quantile_levels: list[float],
    ) -> ForecastResult:
        if isinstance(history, torch.Tensor):
            hist = history.flatten().tolist()
        else:
            hist = list(history)
        if len(hist) < 2:
            base = hist[-1] if hist else 100.0
            slope = 0.0
        else:
            base = hist[-1]
            slope = (hist[-1] - hist[0]) / max(1, len(hist) - 1)
        # Deterministic: linear extrapolation + symmetric quantile bands.
        seed = int(hashlib.sha256(str(hist).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, max(abs(base) * 0.001, 1e-3), size=horizon)
        mean = [base + slope * (i + 1) + float(noise[i]) for i in range(horizon)]
        spread = max(abs(base) * 0.01, 0.1)
        q_list = []
        for i in range(horizon):
            row = []
            for q in quantile_levels:
                # Map quantile to a +/- offset (10/50/90 → -1/0/+1 spreads)
                z = (q - 0.5) * 2.0
                row.append(mean[i] + z * spread)
            q_list.append(row)
        return ForecastResult(
            mean=mean,
            quantile_levels=quantile_levels,
            quantiles=q_list,
            horizon=horizon,
            history_len=len(hist),
        )
