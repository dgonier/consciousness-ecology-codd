"""Producers — forward-pass-and-pool only.

A producer takes a RawInput, renders it to a short text string, runs it
through Qwen, pools the last hidden state, and broadcasts. There is no
system prompt and no Channel — producers are the bottom of the trophic
stack and have no upstream tier to hunt.

Differentiation between producer kinds:
  - WAVELENGTHS class attribute (which raw inputs they ingest, by source tag)
  - the rendering template that turns structured payload → tokenizable text
  - the diet_tags they stamp on their broadcast (so hunters can filter)
  - the pooling strategy (mean / last)

That's it. No prompt engineering. Information is in the hidden state.

Wavelength contract (post phase2-D refactor): each Producer subclass
declares a `WAVELENGTHS: ClassVar[set[str]]` matching tags from
`SOURCE_TAGS_VOCAB`. `Producer.attracts(inp)` is `inp.source in
self.WAVELENGTHS`. Many-to-many is the rule: e.g. an `ohlcv` RawInput
feeds both `TickDelta` and `Anomaly`.

EnvironmentStream contract (Mission 05 — phase3-A): producers can also
consume an attended hidden-state output from the EnvironmentStream
(input-layer K/V substrate) via `produce_from_stream(out, tick, host)`.
That bypasses the per-RawInput text-rendering path entirely; the broadcast
embedding IS the attended substrate vector (mean-pooled over the
out_seq_len). `role_q()` provides a query vector keyed to the producer's
KIND so each producer attracts a different mixture of slots.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import ClassVar

import torch

from ..model_host import ModelHost
from ..types import Broadcast, RawInput
from .base import BaseAgent, new_agent_id


PRODUCER_KINDS = ("tickdelta", "disclosure", "anomaly")


# ---------- diet tags emitted ----------

def diet_tags_for(kind: str, src: str) -> list[str]:
    if kind == "tickdelta":
        return ["is_price_event", "from_tickdelta"]
    if kind == "disclosure":
        return ["is_disclosure_event", "from_disclosure"]
    if kind == "anomaly":
        tags = ["is_anomaly", "from_anomaly"]
        if src == "options":
            tags.append("is_options_flow")
        return tags
    return []


# ---------- rendering ----------

RENDER_PREAMBLE = {
    "tickdelta": "MARKET MICRO ",
    "disclosure": "CORP DISCLOSURE ",
    "anomaly": "ANOMALY OBS ",
}


def _render(kind: str, inp: RawInput) -> str:
    bits = [f"src={inp.source}"]
    for k, v in inp.payload.items():
        bits.append(f"{k}={v}")
    return RENDER_PREAMBLE.get(kind, "") + "; ".join(bits)


# ---------- numeric-feature path (#8 root-cause fix) ----------
#
# The diagnostic at scripts/diagnostics/diag_producer_pool_ab.py showed that
# OHLCV-shaped producers (tickdelta, anomaly) emit bit-identical embeddings
# (cosine 0.999 across StockNet days for the same ticker) regardless of pool
# strategy. The LLM hidden state for "bars=[{'open': 0.029, ...}]" doesn't
# carry the small float differences as separable signal.
#
# Fix: extract hand-engineered statistics from `bars`, project to hidden_size
# via a deterministic seeded random matrix, and ADD to the LLM-pooled vector.
# Preserves the LLM signal (so existing trained Channels still see something
# in-distribution) but injects strong input-specific variance.

import hashlib

_NUMERIC_PROJ_CACHE: dict[tuple[str, int], "torch.Tensor"] = {}
NUMERIC_FEATURE_DIM = 16


def _ohlcv_features(bars: list[dict]) -> list[float]:
    """Extract NUMERIC_FEATURE_DIM hand-engineered features from a bars list.

    Features chosen to be directionally informative for short-horizon
    prediction. Defaults to zeros if the bar list is too short or malformed.
    """
    feats = [0.0] * NUMERIC_FEATURE_DIM
    if not bars:
        return feats
    try:
        closes = [float(b.get("close", 0.0)) for b in bars]
        opens = [float(b.get("open", 0.0)) for b in bars]
        highs = [float(b.get("high", 0.0)) for b in bars]
        lows = [float(b.get("low", 0.0)) for b in bars]
        volumes = [float(b.get("volume", 0.0)) for b in bars]
    except (TypeError, ValueError):
        return feats
    n = len(bars)
    if n == 0:
        return feats
    # 0: total return  (close[-1] - close[0]) / close[0]
    feats[0] = (closes[-1] - closes[0]) / max(abs(closes[0]), 1e-9)
    # 1: last-bar return  (close[-1] - open[-1]) / open[-1]
    feats[1] = (closes[-1] - opens[-1]) / max(abs(opens[-1]), 1e-9)
    # 2: range-to-close  (high[-1] - low[-1]) / close[-1]
    feats[2] = (highs[-1] - lows[-1]) / max(abs(closes[-1]), 1e-9)
    # 3: avg return per bar
    rets = [(closes[i] - closes[i-1]) / max(abs(closes[i-1]), 1e-9) for i in range(1, n)]
    feats[3] = sum(rets) / max(len(rets), 1) if rets else 0.0
    # 4: return std (volatility proxy)
    if rets and len(rets) > 1:
        m = feats[3]
        feats[4] = (sum((r - m) ** 2 for r in rets) / (len(rets) - 1)) ** 0.5
    # 5: pct of bars that closed up
    feats[5] = sum(1 for r in rets if r > 0) / max(len(rets), 1) if rets else 0.5
    # 6: volume z-score for last bar (against the prior bars)
    if n >= 2:
        prior_vol = volumes[:-1]
        m = sum(prior_vol) / len(prior_vol) if prior_vol else 0.0
        s = (sum((v - m) ** 2 for v in prior_vol) / max(len(prior_vol) - 1, 1)) ** 0.5 if len(prior_vol) > 1 else 1.0
        feats[6] = (volumes[-1] - m) / max(s, 1e-9)
    # 7: high-low range over full window normalized by close[-1]
    feats[7] = (max(highs) - min(lows)) / max(abs(closes[-1]), 1e-9) if closes[-1] else 0.0
    # 8: close[-1] / mean(close)
    mean_close = sum(closes) / max(n, 1)
    feats[8] = closes[-1] / max(abs(mean_close), 1e-9) if mean_close else 0.0
    # 9: open[0] / close[-1]  (gap over window)
    feats[9] = opens[0] / max(abs(closes[-1]), 1e-9) if closes[-1] else 0.0
    # 10: number of bars (sequence length signal)
    feats[10] = float(n)
    # 11-15: last 5 bar returns padded with zeros
    last_rets = rets[-5:] if rets else []
    for i, r in enumerate(last_rets):
        feats[11 + i] = r
    return feats


def _numeric_projection(kind: str, hidden_size: int, dtype, device):
    """Get (or build) the deterministic random projection matrix for `kind`.

    Seeded from the kind name so all producers of the same kind share the
    same projection (meaning trained Channels see consistent encoding) but
    different kinds get different projections (so tickdelta and anomaly
    don't collide). Frozen — no gradients.
    """
    key = (kind, hidden_size)
    if key in _NUMERIC_PROJ_CACHE:
        cached = _NUMERIC_PROJ_CACHE[key]
        return cached.to(device=device, dtype=dtype)
    seed = int(hashlib.sha256(f"numeric_proj:{kind}".encode()).hexdigest(), 16) % (2**31)
    g = torch.Generator()
    g.manual_seed(seed)
    # Random Gaussian projection scaled to roughly match LLM hidden norms.
    # LLM pooled hidden norm is ~60 in our diagnostic; we scale to that band
    # so the numeric signal isn't dwarfed by (or doesn't dwarf) the LLM signal.
    proj = torch.randn(NUMERIC_FEATURE_DIM, hidden_size, generator=g) * (1.0 / NUMERIC_FEATURE_DIM ** 0.5)
    _NUMERIC_PROJ_CACHE[key] = proj
    return proj.to(device=device, dtype=dtype)


def _numeric_inject(kind: str, payload: dict, llm_pooled: torch.Tensor) -> torch.Tensor:
    """If the payload has `bars`, project numeric features into hidden_size and
    add to the LLM-pooled hidden state. Pass through unchanged otherwise.
    """
    bars = payload.get("bars") if isinstance(payload, dict) else None
    if not bars:
        return llm_pooled
    feats = _ohlcv_features(bars)
    feat_t = torch.tensor(feats, dtype=llm_pooled.dtype, device=llm_pooled.device)
    proj = _numeric_projection(kind, llm_pooled.shape[-1], llm_pooled.dtype, llm_pooled.device)
    numeric_h = feat_t @ proj  # [hidden_size]
    # Scale to match LLM-pooled norm so neither dominates the sum.
    n_llm = llm_pooled.norm().clamp_min(1e-6)
    n_num = numeric_h.norm().clamp_min(1e-6)
    numeric_h = numeric_h * (n_llm / n_num)
    return llm_pooled + numeric_h


# ---------- producer ----------

@dataclass
class Producer(BaseAgent):
    """Base producer. Subclasses declare WAVELENGTHS + KIND."""
    role: str = "producer"
    pool: str = "mean"  # mean | last

    # Subclasses override these. Empty WAVELENGTHS on the base means the base
    # class doesn't attract anything on its own — only the concrete subtypes do.
    WAVELENGTHS: ClassVar[set[str]] = set()
    KIND: ClassVar[str] = "producer"
    # Default pooling strategy for this producer kind. The diagnostic at
    # scripts/diagnostics/diag_producer_pool_ab.py showed that mean-pool over
    # 350-1300 token sequences crushes input-specific token variance — the
    # root of #8 direction collapse. Subclasses with text-shaped inputs
    # (disclosure: 716-1310 tokens of tweets / press text) override to "last"
    # which preserves variance (cosine 0.61 vs 0.98 for mean-pool).
    DEFAULT_POOL: ClassVar[str] = "mean"

    @classmethod
    def make(cls, kind: str, pool: str | None = None) -> "Producer":
        """Backward-compat factory. Dispatches to the correct subclass by
        the legacy `kind` string ("tickdelta" / "disclosure" / "anomaly")
        so existing call sites (`Producer.make("tickdelta")`) keep working.

        If `pool` is None, uses the subclass's DEFAULT_POOL.
        """
        # Walk all subclasses (including indirect, e.g. SocialSignal lives
        # in a sibling module that imports Producer).
        kind_to_cls: dict[str, type[Producer]] = {}

        def _walk(c: type[Producer]) -> None:
            for sub in c.__subclasses__():
                if sub.KIND:
                    kind_to_cls[sub.KIND] = sub
                _walk(sub)

        _walk(Producer)
        if kind in kind_to_cls:
            target = kind_to_cls[kind]
            chosen_pool = pool if pool is not None else target.DEFAULT_POOL
            return target(id=new_agent_id("producer", kind), kind=kind, pool=chosen_pool)
        raise ValueError(f"unknown producer kind: {kind!r} (known: {sorted(kind_to_cls)})")

    def attracts(self, inp: RawInput) -> bool:
        return inp.source in self.WAVELENGTHS

    async def produce(
        self,
        inp: RawInput,
        tick: int,
        host: ModelHost | None = None,
    ) -> Broadcast | None:
        if not self.attracts(inp):
            return None
        host = host or ModelHost.get()

        text = _render(self.kind, inp)
        # Forward through Qwen and pool. This IS the broadcast; tokens are
        # only the input to the model, not the output to downstream agents.
        pooled = host.text_to_hidden(text, pool=self.pool)
        # #8 root-cause fix part 2: for OHLCV-shaped inputs, inject numeric
        # features so per-day variance survives the LLM encoder. No-op when
        # payload doesn't contain `bars` (e.g. disclosure/press inputs).
        if self.kind in ("tickdelta", "anomaly"):
            pooled = _numeric_inject(self.kind, inp.payload, pooled)
        emb = pooled.detach().cpu().tolist()

        return Broadcast(
            id=str(uuid.uuid4()),
            tier="substrate",
            agent_id=self.id,
            agent_kind=self.kind,
            parent_input_ids=[inp.id],
            diet_tags=diet_tags_for(self.kind, inp.source),
            payload=inp.payload,
            decoded_text=text,  # the rendered input — useful for debug only
            channel_embedding=emb,
            created_tick=tick,
        )

    # ------------------------------------------------------------------
    # EnvironmentStream-driven path (Mission 05 — phase3-A)
    # ------------------------------------------------------------------

    def role_q(self, host: ModelHost | None = None) -> torch.Tensor:
        """Return a query vector for attending into EnvironmentStream.

        Cheap, deterministic, kind-keyed: we encode a short role marker
        ("PRODUCER ROLE=<kind>") through the host model and pool. Producers
        with different KINDs get different queries, so they attract
        different slot subsets even after the wavelength filter narrows
        the candidate set.

        Returns a 1-D tensor of length `host.hidden_size` on CPU/fp32 so
        it round-trips cleanly into TroughAttention.attend.
        """
        host = host or ModelHost.get()
        text = f"PRODUCER ROLE={self.KIND or self.kind}"
        return host.text_to_hidden(text, pool="mean").detach().float().cpu()

    def produce_from_stream(
        self,
        attend_output,
        tick: int,
        host: ModelHost | None = None,
        parent_input_ids: list[str] | None = None,
    ) -> Broadcast | None:
        """Produce a substrate broadcast from a `StreamAttendOutput`.

        The attended hidden-state vector (mean-pooled across the
        out_seq_len) becomes this producer's broadcast embedding. No text
        rendering; the substrate has already done the work of mixing the
        relevant slots. `decoded_text` is set to a debug-only summary of
        which slots dominated.

        Returns None if the attend output didn't actually attend to
        anything (all in-filter slots were dead or the wavelength was
        quiet). Callers should drop None returns.
        """
        host = host or ModelHost.get()
        if attend_output is None:
            return None
        # Skip producers that found nothing in their wavelength. The trough
        # still produces an output even with all -inf bias (NaN -> zeros
        # via softmax fallback), but we treat that as "abstain".
        attn = attend_output.per_slot_attention
        if attn is None or float(attn.sum().item()) <= 1e-6:
            return None

        out_seq = attend_output.output  # [out_seq_len, hidden]
        if out_seq.dim() == 2:
            pooled = out_seq.mean(dim=0)
        else:
            pooled = out_seq
        emb = pooled.detach().float().cpu().tolist()

        # Top-3 slot summary for legibility.
        topk = min(3, attn.shape[0])
        top_vals, top_idx = torch.topk(attn, topk)
        debug_summary = (
            f"ENVSTREAM kind={self.KIND or self.kind} "
            + " ".join(
                f"slot{int(top_idx[i].item())}={float(top_vals[i].item()):.3f}"
                for i in range(topk)
            )
        )

        return Broadcast(
            id=str(uuid.uuid4()),
            tier="substrate",
            agent_id=self.id,
            agent_kind=self.kind,
            parent_input_ids=parent_input_ids or [],
            diet_tags=diet_tags_for(self.kind, "ohlcv"),  # source-agnostic from stream perspective
            payload={"from_envstream": True, "kind": self.KIND or self.kind},
            decoded_text=debug_summary,
            channel_embedding=emb,
            created_tick=tick,
        )


@dataclass
class TickDelta(Producer):
    WAVELENGTHS: ClassVar[set[str]] = {"ohlcv", "trades", "book"}
    KIND: ClassVar[str] = "tickdelta"


@dataclass
class Disclosure(Producer):
    WAVELENGTHS: ClassVar[set[str]] = {"filing", "press"}
    KIND: ClassVar[str] = "disclosure"
    # Tweets/press text are 700-1300 tokens; mean-pool washes the
    # input-specific token variance (cosine 0.98 across days). pool="last"
    # drops cosine to 0.61 — input-responsiveness restored. (#8 root cause)
    DEFAULT_POOL: ClassVar[str] = "last"


@dataclass
class Anomaly(Producer):
    WAVELENGTHS: ClassVar[set[str]] = {"ohlcv", "options", "halt"}
    KIND: ClassVar[str] = "anomaly"
