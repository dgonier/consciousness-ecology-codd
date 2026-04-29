"""EnvironmentStream smoke diagnostic — cortical-column fan-out check.

Builds a single EnvironmentStream, deposits 5 days of AAPL OHLCV + tweets
through the new adapter chain, then has each producer attend and reports
which slot tags each producer attended to most strongly.

Pass criterion (manual review):
  - TickDelta highest weight on `ohlcv` slots
  - SocialSignal highest weight on `tweets` slots
  - QuantitativeProducer attends ~uniformly (no quote_series deposits)

This validates the wavelength-filter plumbing end-to-end:
  raw bytes -> Adapter.adapt -> RawInput -> Stream.deposit
            -> Producer.role_q + WAVELENGTHS -> Stream.attend
            -> Producer.produce_from_stream -> Broadcast
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Force mock model so the smoke is fast and CPU-only.
os.environ.setdefault("TROPHIC_MOCK_MODELS", "1")

from trophic.adapters.stocknet import OhlcvNormalizedAdapter, TokenizedTweetAdapter
from trophic.agents.producer import Anomaly, Disclosure, TickDelta
from trophic.agents.quant_producer import QuantitativeProducer
from trophic.agents.social_signal import SocialSignal
from trophic.environment_stream import EnvironmentStream
from trophic.model_host import ModelHost


CACHE = ROOT / "external" / "stocknet_cache"
TICKER = "AAPL"
N_DAYS = 5


def main() -> int:
    print(f"[smoke] cache={CACHE}")
    if not CACHE.exists():
        print(f"[smoke] missing cache directory; aborting", file=sys.stderr)
        return 2

    host = ModelHost.get()
    print(f"[smoke] hidden_size={host.hidden_size} mock={host.cfg.mock}")

    stream = EnvironmentStream(
        hidden_size=host.hidden_size,
        n_slots=64,
        n_heads=8 if host.hidden_size % 8 == 0 else 4,
        out_seq_len=8,
        seed=42,
    )

    # --- adapt OHLCV ---
    price_path = CACHE / f"price__preprocessed__{TICKER}.txt"
    if not price_path.exists():
        print(f"[smoke] price file missing: {price_path}", file=sys.stderr)
        return 2
    price_text = price_path.read_text()
    ohlcv_inputs = list(OhlcvNormalizedAdapter(TICKER).adapt(price_text))
    print(f"[smoke] adapted {len(ohlcv_inputs)} OHLCV rows")
    # Take the first N_DAYS that have a matching tweet file.
    tweet_files = sorted(CACHE.glob(f"tweet__preprocessed__{TICKER}__*"))
    tweet_dates = [p.name.rsplit("__", 1)[-1] for p in tweet_files]
    by_date = {ri.payload["date"]: ri for ri in ohlcv_inputs}
    selected = []
    for d in tweet_dates:
        if d in by_date:
            selected.append((d, by_date[d]))
        if len(selected) >= N_DAYS:
            break
    print(f"[smoke] selected {len(selected)} (ticker, date) pairs with both OHLCV+tweets")

    # --- deposit ---
    deposit_log: list[tuple[int, str]] = []
    for d, ohlcv_ri in selected:
        emb = host.text_to_hidden(
            f"INPUT src={ohlcv_ri.source}; ticker={TICKER}; date={d}", pool="mean"
        ).detach().float()
        sid = stream.deposit(ohlcv_ri, emb)
        deposit_log.append((sid, ohlcv_ri.source))
        # tweets
        tweet_path = CACHE / f"tweet__preprocessed__{TICKER}__{d}"
        if tweet_path.exists():
            tw_text = tweet_path.read_text()
            for tw_ri in TokenizedTweetAdapter(TICKER, d).adapt(tw_text):
                emb = host.text_to_hidden(
                    f"INPUT src={tw_ri.source}; ticker={TICKER}; date={d}",
                    pool="mean",
                ).detach().float()
                sid = stream.deposit(tw_ri, emb)
                deposit_log.append((sid, tw_ri.source))

    print(f"[smoke] deposited {len(deposit_log)} slots")
    src_counts: dict[str, int] = defaultdict(int)
    for _, src in deposit_log:
        src_counts[src] += 1
    print(f"[smoke] slot source distribution: {dict(src_counts)}")

    # --- producers attend ---
    producers = [
        TickDelta(id="prod_tickdelta", kind="tickdelta"),
        Disclosure(id="prod_disclosure", kind="disclosure"),
        Anomaly(id="prod_anomaly", kind="anomaly"),
        QuantitativeProducer.make(),
        SocialSignal.make(),
    ]

    print()
    print("=== PER-PRODUCER ATTENTION DISTRIBUTION ===")
    for prod in producers:
        kind = getattr(prod, "KIND", prod.kind)
        wavelengths = getattr(prod, "WAVELENGTHS", set())
        # QuantitativeProducer has no role_q on the LM path; synthesize a
        # simple zero-based query so it still exercises attend().
        if hasattr(prod, "role_q"):
            try:
                query = prod.role_q(host)
            except Exception:
                import torch
                query = torch.zeros(host.hidden_size)
        else:
            import torch
            query = torch.zeros(host.hidden_size)
        out = stream.attend(query=query, wavelength_filter=wavelengths, tau=1.0)
        # Report attention sum by source tag.
        by_tag: dict[str, float] = defaultdict(float)
        attn = out.per_slot_attention
        for sid in range(stream.n_slots):
            tag = stream._slot_source_tags[sid] or "<empty>"
            by_tag[tag] += float(attn[sid].item())
        # Format
        ranking = sorted(by_tag.items(), key=lambda x: -x[1])
        rank_str = " ".join(f"{tag}={w:.3f}" for tag, w in ranking if w > 1e-4)
        print(
            f"  {kind:18s} wavelengths={sorted(wavelengths) or '[]'} "
            f"masked_out={out.masked_out_count:3d} "
            f"top_tags: {rank_str or '(all zero)'}"
        )

    print()
    print("[smoke] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
