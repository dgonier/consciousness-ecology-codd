"""Snapshot of inter-tier signals for one scenario, as fed to apex
voters. Builds a list[NodeSignal] from a Scenario + EvidencePacket so
the decomposer's Observation captures the full chain, not just the
apex slice.

Today this captures producer-tier signals (raw renderings of OHLCV /
press / forecast) and the apex evidence packet. When a trained
checkpoint is loaded into the voter pipeline (future work), this
module will also capture herb broadcasts and predator pooled hidden
via the same logit-lens approach used by inspect_signals_jsonl.py.
"""
from __future__ import annotations

from .observation import NodeSignal


def capture_evidence_signals(scenario, evidence_packet) -> list[NodeSignal]:
    """Build NodeSignals for one scenario based on its raw inputs and
    the evidence packet that the apex voters receive."""
    signals: list[NodeSignal] = [
        NodeSignal(
            tier=0,
            node_id="scenario",
            agent_kind="scenario",
            raw_text=scenario.name,
        )
    ]

    ticker = scenario.name.split("_")[2] if "_" in scenario.name else "?"

    for inp in scenario.inputs:
        if inp.source == "ohlcv":
            bars = inp.payload.get("bars", []) or []
            last_bar_text = ""
            if bars:
                b = bars[-1]
                last_bar_text = (
                    f"open={b.get('open', 0):.4f} "
                    f"close={b.get('close', 0):.4f} "
                    f"vol={int(b.get('volume', 0))}"
                )
            signals.append(NodeSignal(
                tier=1,
                node_id=f"producer.ohlcv.{ticker}",
                parents=["scenario"],
                agent_kind="ohlcv_producer",
                norm=float(len(bars)),
                raw_text=last_bar_text,
                diet_tags=["from_ohlcv"],
            ))
        elif inp.source == "press":
            body = (inp.payload.get("body") or "").strip()
            signals.append(NodeSignal(
                tier=1,
                node_id=f"producer.press.{ticker}",
                parents=["scenario"],
                agent_kind="press_producer",
                norm=float(len(body)),
                raw_text=body[:200],
                diet_tags=["from_press"],
            ))

    feats = getattr(scenario, "forecaster_features", None)
    if feats and len(feats) >= 32:
        # Drift, snr, monotonicity (per forecast_features.py layout)
        drift, snr = feats[16], feats[21]
        monot = feats[26] if len(feats) > 26 else 0.0
        signals.append(NodeSignal(
            tier=1,
            node_id=f"producer.forecast.{ticker}",
            parents=["scenario"],
            agent_kind="chronos_forecaster",
            norm=float(abs(drift)),
            raw_text=f"drift={drift:+.4f} snr={snr:+.3f} monot={monot:+.3f}",
            diet_tags=["from_forecaster"],
        ))

    # Tier 4: the apex evidence packet — the full prompt the voters see.
    parent_ids = [s.node_id for s in signals if s.tier == 1]
    signals.append(NodeSignal(
        tier=4,
        node_id="apex.evidence_packet",
        parents=parent_ids,
        agent_kind="evidence_packet",
        norm=float(len(evidence_packet.text)),
        raw_text=evidence_packet.text[:600],
    ))
    return signals
