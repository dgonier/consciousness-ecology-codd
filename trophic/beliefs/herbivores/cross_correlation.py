"""Cross-correlation herbivore — pure-numpy species, no LLM.

Maintains rolling Pearson correlation matrices over the watchlist universe
across two timescales (60d "regime" and 20d "recent"), plus a 1-day lead-lag
matrix. Activates beliefs in two ways:

  (a) Sympathy: when another herbivore activated beliefs about ticker A,
      this herb broadcasts a weakened version of those activations to the
      tickers most-correlated with A. Direction follows the sign of the
      correlation (positive corr → same-direction sympathy, negative →
      opposite). Magnitude scales with |correlation|.

  (b) Cluster / regime beliefs: based on the structure of the correlation
      matrix itself, this herb activates beliefs like
      `belief.market.tech_cluster_correlated_high`, `risk_on_regime`,
      `dispersion_high` (alpha environment).

The correlation state is persisted to disk as a numpy archive so it survives
across passes; the herb mutates and saves it on each call.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..schema import BeliefActivation


SPECIES_ID = "cross_correlation.v0"

# Default thresholds (configurable via init)
DEFAULTS = {
    "lookback_60d": 60,
    "lookback_20d": 20,
    "lookback_lead_lag": 30,    # 1d lead-lag estimated over recent 30d
    # Sympathy uses the LONGER (60d) window — short-window correlations are
    # too noisy and produce spurious pairings (e.g. AAPL/MRK at +0.57 in
    # the smoke test that wouldn't survive on 60d data).
    "sympathy_window": "60d",
    "sympathy_min_abs_corr": 0.55,
    "sympathy_max_targets": 4,
    "sympathy_decay_factor": 0.7,
    # Cluster detection uses the SHORTER (20d) window — we want to detect
    # the *current* regime, not the long-term average.
    "cluster_window": "20d",
    "cluster_high_threshold": 0.40,    # post-COVID baseline; was 0.65
    "cluster_low_threshold":  0.10,    # cluster broken / dispersing
    "dispersion_low_threshold": 0.25,  # mean off-diag |corr| ≤ → high dispersion
}

# Sector / cluster definitions for the 20-ticker universe. These are the
# obvious ones; refine per actual data.
CLUSTERS: dict[str, list[str]] = {
    "tech":       ["AAPL", "MSFT", "GOOG", "AVGO", "CSCO", "AMZN"],
    "financial":  ["JPM", "MA", "V"],
    "consumer_staples": ["KO", "PEP", "PG", "WMT", "MCD"],
    "consumer_disc": ["HD", "AMZN"],
    "health":     ["JNJ", "MRK", "UNH", "ABBV"],
    "energy":     ["CVX"],
}


# Magnitude category transitions when broadcasting via sympathy.
# parent_magnitude → child_magnitude (one step weaker per default)
SYMPATHY_MAGNITUDE_STEPDOWN = {
    "decisive": "strong",
    "strong":   "medium",
    "medium":   "weak",
    "weak":     "weak",  # already at floor
}


@dataclass
class CorrelationState:
    """Persisted correlation matrices for the universe."""
    universe: list[str]
    rolling_60d: np.ndarray  # shape (N, N), symmetric, diag=1, NaN if not enough data
    rolling_20d: np.ndarray  # shape (N, N)
    lead_lag_1d: np.ndarray  # shape (N, N), NOT symmetric; entry [i,j] = corr(returns_i[t-1], returns_j[t])
    last_date: Optional[str] = None
    n_observations: int = 0

    @classmethod
    def empty(cls, universe: list[str]) -> "CorrelationState":
        n = len(universe)
        nan_mat = np.full((n, n), np.nan, dtype=np.float32)
        return cls(
            universe=list(universe),
            rolling_60d=nan_mat.copy(),
            rolling_20d=nan_mat.copy(),
            lead_lag_1d=nan_mat.copy(),
            last_date=None,
            n_observations=0,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            rolling_60d=self.rolling_60d,
            rolling_20d=self.rolling_20d,
            lead_lag_1d=self.lead_lag_1d,
        )
        meta = {
            "universe": self.universe,
            "last_date": self.last_date,
            "n_observations": self.n_observations,
        }
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> Optional["CorrelationState"]:
        if not path.exists():
            return None
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        npz = np.load(path)
        return cls(
            universe=meta["universe"],
            rolling_60d=npz["rolling_60d"],
            rolling_20d=npz["rolling_20d"],
            lead_lag_1d=npz["lead_lag_1d"],
            last_date=meta.get("last_date"),
            n_observations=int(meta.get("n_observations", 0)),
        )

    def idx(self, ticker: str) -> Optional[int]:
        try:
            return self.universe.index(ticker)
        except ValueError:
            return None


def _pearson_matrix(returns: np.ndarray) -> np.ndarray:
    """Compute symmetric Pearson correlation across columns (tickers).

    `returns` shape: (T, N). Returns (N, N).
    Handles missing columns (all-NaN) by leaving NaN entries.
    """
    if returns.ndim != 2:
        raise ValueError("returns must be 2D (T, N)")
    T, N = returns.shape
    out = np.full((N, N), np.nan, dtype=np.float32)
    if T < 3:
        return out
    # Ignore rows with any NaN for the simple shared-window estimator.
    mask = ~np.isnan(returns).any(axis=1)
    rs = returns[mask]
    if rs.shape[0] < 3:
        # Per-pair fallback: use pairwise complete obs
        for i in range(N):
            for j in range(i, N):
                m = ~(np.isnan(returns[:, i]) | np.isnan(returns[:, j]))
                if m.sum() < 3:
                    continue
                c = np.corrcoef(returns[m, i], returns[m, j])[0, 1]
                out[i, j] = out[j, i] = c
        return out
    cm = np.corrcoef(rs.T)  # (N, N)
    out[:] = cm.astype(np.float32)
    return out


def _lead_lag_matrix(returns: np.ndarray) -> np.ndarray:
    """1-day lead-lag: corr(returns_i[t-1], returns_j[t]) for each (i, j).

    Not symmetric. Entry [i, j] tells you how much i's PRIOR-day move predicts
    j's CURRENT-day move.
    """
    T, N = returns.shape
    out = np.full((N, N), np.nan, dtype=np.float32)
    if T < 4:
        return out
    lagged = returns[:-1, :]   # i at t-1
    current = returns[1:, :]   # j at t
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            li = lagged[:, i]
            cj = current[:, j]
            m = ~(np.isnan(li) | np.isnan(cj))
            if m.sum() < 5:
                continue
            c = np.corrcoef(li[m], cj[m])[0, 1]
            out[i, j] = c
    return out


def update_state_from_bars(
    state: CorrelationState,
    bars_by_date_ticker: dict[str, dict[str, dict]],
    as_of_date: str,
    cfg: Optional[dict] = None,
) -> CorrelationState:
    """Recompute correlation matrices over the trailing windows ending at as_of_date.

    bars_by_date_ticker: {date_iso: {ticker: bar_dict}} — bar_dict has 'close'
    """
    cfg = cfg or DEFAULTS
    dates = sorted(bars_by_date_ticker.keys())
    if as_of_date not in dates:
        # Use the latest date available <= as_of_date
        dates = [d for d in dates if d <= as_of_date]
        if not dates:
            return state
    end_idx = dates.index(as_of_date) if as_of_date in dates else len(dates) - 1
    T = end_idx + 1
    N = len(state.universe)
    closes = np.full((T, N), np.nan, dtype=np.float32)
    for t in range(T):
        d = dates[t]
        ticker_bars = bars_by_date_ticker.get(d, {})
        for i, tk in enumerate(state.universe):
            b = ticker_bars.get(tk)
            if b is not None and b.get("close") is not None:
                closes[t, i] = b["close"]
    # Daily simple returns
    rets = np.full_like(closes, np.nan)
    rets[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
    # Slice trailing windows
    win60 = rets[-cfg["lookback_60d"]:]
    win20 = rets[-cfg["lookback_20d"]:]
    winLL = rets[-cfg["lookback_lead_lag"]:]
    state.rolling_60d = _pearson_matrix(win60)
    state.rolling_20d = _pearson_matrix(win20)
    state.lead_lag_1d = _lead_lag_matrix(winLL)
    state.last_date = as_of_date
    state.n_observations = T
    return state


def _cluster_intra_corr(corr: np.ndarray, idxs: list[int]) -> float:
    """Mean pairwise correlation within an index set (off-diagonal only)."""
    if len(idxs) < 2:
        return float("nan")
    sub = corr[np.ix_(idxs, idxs)]
    n = len(idxs)
    triu = sub[np.triu_indices(n, k=1)]
    triu = triu[~np.isnan(triu)]
    if len(triu) == 0:
        return float("nan")
    return float(triu.mean())


def _global_off_diagonal_mean_abs(corr: np.ndarray) -> float:
    n = corr.shape[0]
    if n < 2:
        return float("nan")
    triu = corr[np.triu_indices(n, k=1)]
    triu = triu[~np.isnan(triu)]
    if len(triu) == 0:
        return float("nan")
    return float(np.mean(np.abs(triu)))


@dataclass
class CrossCorrelationHerbivore:
    """Cross-correlation species — emits sympathy + cluster/regime activations."""
    universe: list[str]
    state: CorrelationState
    cfg: dict = field(default_factory=lambda: dict(DEFAULTS))

    @classmethod
    def with_state(
        cls,
        universe: list[str],
        state_path: Optional[Path] = None,
    ) -> "CrossCorrelationHerbivore":
        state = None
        if state_path is not None:
            state = CorrelationState.load(state_path)
        if state is None or state.universe != list(universe):
            state = CorrelationState.empty(universe)
        return cls(universe=list(universe), state=state)

    def _window(self, key: str) -> np.ndarray:
        """Select rolling matrix by config key ('20d' or '60d')."""
        if key == "60d":
            return self.state.rolling_60d
        return self.state.rolling_20d

    # ── (b) Cluster / regime activations ──────────────────────────────

    def emit_cluster_activations(self) -> list[BeliefActivation]:
        """Look at correlation structure → emit market-scope cluster beliefs."""
        out: list[BeliefActivation] = []
        corr = self._window(self.cfg.get("cluster_window", "20d"))
        if np.isnan(corr).all():
            return out

        # Per-cluster: is intra-cluster correlation high?
        for cluster_name, tickers in CLUSTERS.items():
            idxs = [self.state.idx(t) for t in tickers]
            idxs = [i for i in idxs if i is not None]
            if len(idxs) < 3:
                continue
            mean_corr = _cluster_intra_corr(corr, idxs)
            if np.isnan(mean_corr):
                continue
            target_high = f"belief.market.cluster_correlation_{cluster_name}_high"
            target_low  = f"belief.market.cluster_correlation_{cluster_name}_low"
            if mean_corr >= self.cfg["cluster_high_threshold"]:
                gap = mean_corr - self.cfg["cluster_high_threshold"]
                mag = "strong" if gap > 0.20 else "medium"
                out.append(BeliefActivation(
                    target_belief_id=target_high,
                    direction_of_effect="increases",
                    magnitude=mag,
                    decay_class="normal",
                    self_rated_confidence="high",
                    reasoning=(
                        f"mean intra-{cluster_name} corr = {mean_corr:+.2f} "
                        f"(≥ {self.cfg['cluster_high_threshold']:.2f})"
                    ),
                    species_id=SPECIES_ID,
                ))
            elif mean_corr <= self.cfg["cluster_low_threshold"]:
                # Cluster has broken / dispersed
                gap = self.cfg["cluster_low_threshold"] - mean_corr
                mag = "medium" if gap > 0.10 else "weak"
                out.append(BeliefActivation(
                    target_belief_id=target_low,
                    direction_of_effect="increases",
                    magnitude=mag,
                    decay_class="normal",
                    self_rated_confidence="medium",
                    reasoning=(
                        f"mean intra-{cluster_name} corr = {mean_corr:+.2f} "
                        f"(≤ {self.cfg['cluster_low_threshold']:.2f}); "
                        f"cluster broken / single-stock environment"
                    ),
                    species_id=SPECIES_ID,
                ))

        # Global dispersion: low mean |corr| means alpha environment
        global_abs = _global_off_diagonal_mean_abs(corr)
        if not np.isnan(global_abs):
            if global_abs <= self.cfg["dispersion_low_threshold"]:
                out.append(BeliefActivation(
                    target_belief_id="belief.market.cross_asset_dispersion_high",
                    direction_of_effect="increases",
                    magnitude="medium",
                    decay_class="normal",
                    self_rated_confidence="medium",
                    reasoning=(
                        f"global mean |off-diag corr| = {global_abs:.2f} "
                        f"≤ {self.cfg['dispersion_low_threshold']:.2f}; "
                        f"single-stock dispersion regime"
                    ),
                    species_id=SPECIES_ID,
                ))
        return out

    # ── (a) Sympathy broadcast ────────────────────────────────────────

    def broadcast_sympathy(
        self,
        source_activations: Iterable[BeliefActivation],
        ticker_extractor=None,
    ) -> list[BeliefActivation]:
        """For each source activation that targets a ticker-specific belief,
        emit weakened sympathy activations on correlated tickers.

        `ticker_extractor`: callable(activation) -> Optional[ticker_str].
            By default we look for activation.target_belief_id ending in
            `.ticker_<TKR>` or context with key 'ticker' (we don't have
            ticker context wired into BeliefActivation yet — for now we
            depend on the caller passing an extractor).
        """
        if ticker_extractor is None:
            ticker_extractor = _default_ticker_extractor
        out: list[BeliefActivation] = []
        corr = self._window(self.cfg.get("sympathy_window", "60d"))
        if np.isnan(corr).all():
            return out
        max_k = self.cfg["sympathy_max_targets"]
        min_abs = self.cfg["sympathy_min_abs_corr"]

        for src in source_activations:
            if src.species_id == SPECIES_ID:
                continue  # no sympathy-of-sympathy
            tk = ticker_extractor(src)
            if tk is None:
                continue
            i = self.state.idx(tk)
            if i is None:
                continue
            row = corr[i, :].copy()
            row[i] = 0.0  # exclude self
            # Pick top-K by |corr|, gated by min threshold
            abs_row = np.abs(row)
            abs_row = np.where(np.isnan(abs_row), 0.0, abs_row)
            order = np.argsort(-abs_row)
            picked = 0
            for j in order:
                if picked >= max_k:
                    break
                c = row[j]
                if np.isnan(c) or abs(c) < min_abs:
                    continue
                # Direction: same as source if corr > 0, flipped if < 0
                src_sign = +1.0 if src.direction_of_effect == "increases" else -1.0
                eff_sign = src_sign * (+1.0 if c > 0 else -1.0)
                child_dir = "increases" if eff_sign > 0 else "decreases"
                child_mag = SYMPATHY_MAGNITUDE_STEPDOWN.get(src.magnitude, "weak")
                # Confidence: scale source confidence by |corr|
                child_conf = "low"
                if abs(c) >= 0.80:
                    child_conf = "medium"
                if abs(c) >= 0.90:
                    child_conf = "high"
                target_ticker = self.state.universe[j]
                # Sympathy targets the same belief template under the
                # correlated ticker. Belief id pattern matches what the
                # event_classifier herbivore will emit.
                target_id = _swap_ticker_in_belief_id(
                    src.target_belief_id, tk, target_ticker,
                )
                if target_id is None:
                    continue
                out.append(BeliefActivation(
                    target_belief_id=target_id,
                    direction_of_effect=child_dir,
                    magnitude=child_mag,
                    decay_class=src.decay_class,
                    self_rated_confidence=child_conf,
                    reasoning=(
                        f"sympathy from {tk} (20d corr {c:+.2f}); "
                        f"source magnitude={src.magnitude}, "
                        f"source dir={src.direction_of_effect}"
                    ),
                    species_id=SPECIES_ID,
                ))
                picked += 1
        return out


def _default_ticker_extractor(act: BeliefActivation) -> Optional[str]:
    """Extract a ticker from an activation's target_belief_id.

    Conventions supported:
      - `belief.company.{template}__ticker_{TKR}` (event_classifier herb)
      - `belief.company.{template}.{TKR}` (alternate)
    Returns None if no ticker can be parsed.
    """
    bid = act.target_belief_id
    # Pattern A: __ticker_TKR suffix
    if "__ticker_" in bid:
        return bid.rsplit("__ticker_", 1)[1].split("__")[0].upper()
    # Pattern B: trailing .TKR after company belief template
    parts = bid.split(".")
    if len(parts) >= 4 and parts[0] == "belief" and parts[1] == "company":
        last = parts[-1]
        if last.isupper() and 1 < len(last) <= 5:
            return last
    return None


def _swap_ticker_in_belief_id(
    belief_id: str, src_ticker: str, dst_ticker: str,
) -> Optional[str]:
    if "__ticker_" in belief_id:
        head, _ = belief_id.rsplit("__ticker_", 1)
        return f"{head}__ticker_{dst_ticker}"
    parts = belief_id.split(".")
    if len(parts) >= 4 and parts[-1] == src_ticker:
        parts[-1] = dst_ticker
        return ".".join(parts)
    return None
