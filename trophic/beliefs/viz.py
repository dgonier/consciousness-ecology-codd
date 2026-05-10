"""Render belief network state to a JSON snapshot + self-contained HTML.

Pattern: after each pass completes, call `render_pass_viz(store, pass_meta,
out_dir)` and it writes:
  out_dir/pass_{N}.json   — belief state snapshot
  out_dir/pass_{N}.html   — self-contained react-flow viz (CDN, no build)
  out_dir/index.html      — symlink/copy of the latest

The html embeds the json (no fetch needed) so you can open the file
directly without running a server.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .schema import BeliefActivation, Event, InternalLink, OutcomeBelief, StateBelief
from .store import BeliefStore


def _belief_color(p: float) -> str:
    """Map probability [0,1] to a color. Below 0.5 = redshift, above = greenshift,
    0.5 itself = neutral grey."""
    if p < 0.5:
        # red side: scale 0.0 → strong red, 0.5 → grey
        intensity = (0.5 - p) * 2  # 0 to 1
        r = int(180 + 75 * intensity)
        g = int(80 - 30 * intensity)
        b = int(80 - 30 * intensity)
    else:
        # green side
        intensity = (p - 0.5) * 2
        r = int(80 - 30 * intensity)
        g = int(180 + 75 * intensity)
        b = int(80 - 30 * intensity)
    return f"rgb({r},{g},{b})"


def _compute_node_depths(
    state_belief_ids: list[str],
    outcome_ids: list[str],
    edges: list[tuple[str, str]],  # (premise, conclusion)
    max_depth_cap: int = 30,
) -> tuple[dict[str, int], int]:
    """Compute layout depth = longest path from leaf to outcome (or sink).

    Cycle-safe: uses BFS-with-visited-per-pass and a hard cap. Outcomes (or
    sinks if no outcomes) get depth 0. Leaves get the largest depth.
    """
    from collections import defaultdict, deque

    all_ids = set(state_belief_ids) | set(outcome_ids)
    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for src, dst in edges:
        if src in all_ids and dst in all_ids:
            outgoing[src].append(dst)
            incoming[dst].append(src)

    # Anchors: outcomes AND sinks (state beliefs with no downstream).
    # Without outcomes the sinks ARE the rightmost layer; with outcomes we
    # want both — outcomes pull their ancestors right, but the vast majority
    # of the network doesn't connect to the synthetic smoke outcome, so its
    # natural rightmost layer is its sinks.
    anchors_set: set[str] = set()
    for oid in outcome_ids:
        if oid in all_ids:
            anchors_set.add(oid)
    for bid in state_belief_ids:
        downstream = [c for c in outgoing.get(bid, []) if c in all_ids]
        if not downstream:
            anchors_set.add(bid)
    anchors = list(anchors_set)

    depth: dict[str, int] = {a: 0 for a in anchors}
    queue: deque = deque(anchors)
    safety = 0
    while queue and safety < 500_000:
        safety += 1
        cur = queue.popleft()
        cur_d = depth[cur]
        if cur_d >= max_depth_cap:
            continue
        for p in incoming.get(cur, []):
            new_d = cur_d + 1
            if new_d > max_depth_cap:
                continue
            if p not in depth or depth[p] < new_d:
                depth[p] = new_d
                queue.append(p)

    # Disconnected nodes → depth 0
    for bid in state_belief_ids:
        depth.setdefault(bid, 0)
    max_depth = max(depth.values()) if depth else 0
    return depth, max_depth


def snapshot_pass(
    store: BeliefStore,
    pass_id: str,
    pass_meta: dict,
    incoming_events: Optional[list[Event]] = None,
    outcome_predictions: Optional[list[OutcomeBelief]] = None,
) -> dict:
    """Build a JSON-serializable snapshot of the belief network state.

    Includes:
      - pass metadata (id, ticker, scenario, label, prediction, outcome correct?)
      - all state beliefs (with current_p, prior_p, deviation, scope)
      - all internal links (premise, conclusion, strength, direction)
      - this pass's activations (incoming evidence)
      - this pass's outcome predictions
    """
    state_beliefs = store.all_state_beliefs()
    links = store.all_links()

    # Pre-compute node depths (longest path to outcome/sink) — cycle-safe
    outcome_list = list(outcome_predictions or [])
    node_depths, max_depth = _compute_node_depths(
        state_belief_ids=[b.id for b in state_beliefs],
        outcome_ids=[o.id for o in outcome_list],
        edges=[(l.premise_belief_id, l.conclusion_belief_id) for l in links],
    )

    # Collect activations from incoming events
    activations: list[dict] = []
    if incoming_events:
        for e in incoming_events:
            for a in e.activations:
                activations.append({
                    "event_id": e.id,
                    "event_source": e.source,
                    "event_summary": (e.raw_content or ""),
                    "target_belief_id": a.target_belief_id,
                    "direction_of_effect": a.direction_of_effect,
                    "magnitude": a.magnitude,
                    "decay_class": a.decay_class,
                    "self_rated_confidence": a.self_rated_confidence,
                    "magnitude_logit": a.magnitude_logit,
                    "reasoning": (a.reasoning or ""),
                    "species_id": a.species_id,
                })

    return {
        "pass_id": pass_id,
        "pass_meta": pass_meta,
        "max_depth": max_depth,
        "state_beliefs": [
            {
                "id": b.id,
                "statement": b.statement_template,
                "scope": b.scope,
                "prior_p": b.prior_p,
                "current_p": b.current_p,
                "deviation": b.current_p - b.prior_p,
                "decay_class": b.decay_class,
                "context": b.context,
                "polymarket_market_id": b.polymarket_market_id,
                "color": _belief_color(b.current_p),
                "n_evidence": len(b.evidence_log),
                "depth": node_depths.get(b.id, 0),
            }
            for b in state_beliefs
        ],
        "internal_links": [
            {
                "id": l.id,
                "premise": l.premise_belief_id,
                "conclusion": l.conclusion_belief_id,
                "scope": l.scope,
                "direction": l.direction,
                "strength": l.strength_posterior,
                "validation_status": l.validation_status,
                "n_validations": l.n_validations,
                "n_correct": l.n_correct,
            }
            for l in links
        ],
        "activations_this_pass": activations,
        "outcomes_this_pass": [
            {
                "id": o.id,
                "ticker": o.ticker,
                "statement": o.statement,
                "p_up": o.p_up,
                "color": _belief_color(o.p_up),
                "contributing_state_beliefs": o.contributing_state_beliefs,
                "contributing_links": o.contributing_links,
                "resolved": o.resolved,
                "actual_direction": o.actual_direction,
            }
            for o in (outcome_predictions or [])
        ],
    }


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Ecology — belief network — {pass_id}</title>
<style>
  html, body, #root {{ height: 100%; margin: 0; font-family: ui-monospace, Menlo, monospace; background: #0e1116; color: #d6dbe1; }}
  .header {{ position: absolute; top: 0; left: 0; right: 0; padding: 8px 14px; background: #161b22; border-bottom: 1px solid #30363d; z-index: 100; font-size: 12px; }}
  .header b {{ color: #58a6ff; }}
  .panel {{ position: absolute; right: 12px; top: 60px; width: 360px; background: #161b22; border: 1px solid #30363d; padding: 10px; font-size: 11px; max-height: 80vh; overflow-y: auto; z-index: 100; border-radius: 4px; }}
  .panel h3 {{ margin: 0 0 6px; color: #58a6ff; font-size: 12px; }}
  .activation {{ padding: 6px; margin-bottom: 4px; background: #0d1117; border-left: 3px solid #f0883e; }}
  .activation .target {{ color: #79c0ff; }}
  .activation .magnitude {{ color: #f0883e; font-weight: bold; }}
  .activation .reasoning {{ color: #8b949e; font-style: italic; margin-top: 2px; }}
  .outcome-card {{ padding: 8px; margin-bottom: 6px; background: #0d1117; border-left: 3px solid; }}
  .outcome-card.up {{ border-left-color: #3fb950; }}
  .outcome-card.down {{ border-left-color: #f85149; }}
  .belief-node-content {{ padding: 6px 8px; border-radius: 4px; min-width: 200px; max-width: 280px; font-size: 11px; word-wrap: break-word; overflow-wrap: break-word; white-space: normal; }}
  .belief-node-content.delta-up {{ box-shadow: 0 0 0 3px #58a6ff, 0 0 18px 3px rgba(88,166,255,0.7); animation: pulse 2s infinite; }}
  .belief-node-content.delta-down {{ box-shadow: 0 0 0 3px #f78166, 0 0 18px 3px rgba(247,129,102,0.7); animation: pulse 2s infinite; }}
  .belief-node-content.delta-strong {{ box-shadow: 0 0 0 4px #ffd33d, 0 0 24px 4px rgba(255,211,61,0.8); animation: pulse 1.2s infinite; }}
  .delta-badge {{ display: inline-block; margin-left: 4px; padding: 1px 5px; border-radius: 8px; font-size: 9px; font-weight: bold; }}
  .delta-badge.up {{ background: #1f6feb; color: #fff; }}
  .delta-badge.down {{ background: #da3633; color: #fff; }}
  @keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.7; }} }}
  .delta-toggle {{ position: absolute; top: 40px; left: 14px; padding: 6px 10px; background: #161b22; border: 1px solid #30363d; color: #d6dbe1; font-size: 11px; cursor: pointer; z-index: 100; border-radius: 4px; }}
  .delta-toggle.active {{ background: #1f6feb; border-color: #1f6feb; }}
  .scope-tag {{ display: inline-block; padding: 1px 6px; background: #30363d; border-radius: 3px; font-size: 9px; margin-right: 4px; }}
</style>
<script crossorigin src="https://unpkg.com/react@18.3.1/umd/react.production.min.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js"></script>
<script src="https://unpkg.com/reactflow@11.11.4/dist/umd/index.js"></script>
<link rel="stylesheet" href="https://unpkg.com/reactflow@11.11.4/dist/style.css" />
</head>
<body>
<div id="root"></div>
<script>
  const PASS_DATA = {pass_data_json};
  const e = React.createElement;

  const SCOPE_COLOR = {{ macro: '#bb86fc', market: '#03dac6', sector: '#ffb74d', company: '#81d4fa', outcome: '#ff8a80' }};

  // Layout parameters
  const ROW_HEIGHT = 110;
  const NODE_WIDTH = 280;
  // Within a depth column we fan into N sub-columns (C_A..C_E). Each
  // sub-column is offset in X from the column base, AND each sub-column
  // starts at a different Y (staggered like a checkerboard). This gives
  // edges vertical gaps to sneak through.
  const SUBCOLS_PER_DEPTH = 5;
  const SUBCOL_DX = NODE_WIDTH * 1.25;                       // 350px — clears node width
  const COL_GAP = NODE_WIDTH * 2.5;                          // 700px — gap between depth columns
  const SUBCOL_Y_STAGGER = ROW_HEIGHT / SUBCOLS_PER_DEPTH;   // y-shift per sub-col

  function buildGraph(data) {{
    // Depths are precomputed in Python and embedded per-belief
    const maxDepth = data.max_depth || 0;
    const depth = new Map();
    data.state_beliefs.forEach(b => depth.set(b.id, b.depth || 0));
    data.outcomes_this_pass.forEach(o => depth.set(o.id, 0));

    // For X layout: depth 0 (outcomes) on the right, max depth on the left.
    // x = (maxDepth - depth) * COL_WIDTH
    const colCounts = new Map(); // depth -> running y index, used as a counter per column
    const nodes = [];

    // Sort beliefs by scope within each depth column for some visual grouping
    const SCOPE_ORDER = {{ macro: 0, market: 1, sector: 2, company: 3, outcome: 4 }};
    const sortedBeliefs = [...data.state_beliefs].sort((a, b) => {{
      const da = depth.get(a.id) || 0;
      const db = depth.get(b.id) || 0;
      if (da !== db) return db - da; // process leftmost (high-depth) first
      return (SCOPE_ORDER[a.scope] || 9) - (SCOPE_ORDER[b.scope] || 9);
    }});

    // Each depth column occupies: SUBCOLS_PER_DEPTH × SUBCOL_DX (sub-col fan-out)
    // + COL_GAP (gap before next column). Sub-columns are spaced by
    // NODE_WIDTH × 1.25 so they don't overlap; columns are spaced by
    // NODE_WIDTH × 2.5 between their fan-outs.
    const COL_TOTAL_WIDTH = SUBCOLS_PER_DEPTH * SUBCOL_DX + COL_GAP;

    sortedBeliefs.forEach(b => {{
      const d = depth.get(b.id) || 0;
      const col = maxDepth - d;
      const yIdx = colCounts.get(d) || 0;
      colCounts.set(d, yIdx + 1);
      // Round-robin into sub-columns so each sub-col fills evenly.
      const subCol = yIdx % SUBCOLS_PER_DEPTH;
      const rowInSub = Math.floor(yIdx / SUBCOLS_PER_DEPTH);
      const baseX = col * COL_TOTAL_WIDTH + 60;
      const x = baseX + subCol * SUBCOL_DX;
      // Each sub-col starts at a staggered y so adjacent sub-cols
      // checkerboard. Sub-col 0 starts at y=80, sub-col 1 at y=80 + stagger,
      // etc.; rows step by ROW_HEIGHT within each sub-col.
      const y = 80 + subCol * SUBCOL_Y_STAGGER + rowInSub * ROW_HEIGHT;
      const scopeColor = SCOPE_COLOR[b.scope] || '#888';
      const delta = (b.current_p ?? b.prior_p) - b.prior_p;
      const aDelta = Math.abs(delta);
      let deltaClass = '';
      if (aDelta >= 0.10) deltaClass = ' delta-strong';
      else if (delta > 0.02) deltaClass = ' delta-up';
      else if (delta < -0.02) deltaClass = ' delta-down';
      const deltaBadge = aDelta >= 0.02
        ? e('span', {{ key: 'db', className: `delta-badge ${{delta > 0 ? 'up' : 'down'}}` }},
            (delta > 0 ? '+' : '') + delta.toFixed(2))
        : null;
      nodes.push({{
        id: b.id,
        position: {{ x, y }},
        data: {{ label: e('div', {{ className: `belief-node-content${{deltaClass}}`, style: {{ background: b.color, borderLeft: `4px solid ${{scopeColor}}` }} }},
          e('div', {{ key: 't' }}, e('span', {{ className: 'scope-tag' }}, b.scope, ' d', d), b.statement, deltaBadge),
          e('div', {{ key: 'p', style: {{ marginTop: 4, color: '#0e1116', fontWeight: 'bold' }} }},
            `p=${{b.current_p.toFixed(3)}}  (prior ${{b.prior_p.toFixed(2)}})  Δ=${{(delta>=0?'+':'')+delta.toFixed(3)}}`),
          b.context && Object.keys(b.context).length ? e('div', {{ key: 'c', style: {{ color: '#0e1116', fontSize: 10 }} }},
            JSON.stringify(b.context)) : null,
        ) }},
        style: {{ background: 'transparent', border: 'none', width: NODE_WIDTH }},
        // Stash delta for client-side filtering
        // (reactflow keeps this in node.data; we read it for the toggle)
        // Note: must NOT shadow `data` field above. Use a separate key.
      }});
      // Augment last-pushed node with raw delta numeric for the filter
      nodes[nodes.length - 1].data._delta = delta;
      nodes[nodes.length - 1].data._aDelta = aDelta;
    }});

    // Outcome nodes — depth 0, rightmost column
    data.outcomes_this_pass.forEach((o, i) => {{
      const col = maxDepth;
      const baseX = col * COL_TOTAL_WIDTH + 60;
      nodes.push({{
        id: o.id,
        position: {{ x: baseX + 200, y: 80 + i * (ROW_HEIGHT + 20) }},
        data: {{ label: e('div', {{ className: 'belief-node-content', style: {{ background: o.color, border: '2px solid #58a6ff' }} }},
          e('div', {{ key: 't' }}, e('span', {{ className: 'scope-tag' }}, 'OUTCOME'), o.statement),
          e('div', {{ key: 'p', style: {{ marginTop: 4, color: '#0e1116', fontWeight: 'bold' }} }},
            `P(up) = ${{o.p_up.toFixed(3)}}`),
          o.actual_direction ? e('div', {{ key: 'a', style: {{ color: '#0e1116', fontSize: 10, marginTop: 2 }} }},
            `actual: ${{o.actual_direction}}`) : null,
        ) }},
        style: {{ background: 'transparent', border: 'none', width: NODE_WIDTH }},
      }});
    }});

    // Edges from links
    const nodeIds = new Set(nodes.map(n => n.id));
    const edges = data.internal_links
      .filter(l => nodeIds.has(l.premise) && nodeIds.has(l.conclusion))
      .map(l => ({{
        id: l.id,
        source: l.premise,
        target: l.conclusion,
        animated: false,
        style: {{ stroke: l.direction === 'positive' ? '#3fb950' : '#f85149', strokeWidth: 1 + 3 * l.strength }},
        label: `${{l.direction === 'positive' ? '+' : '−'}} ${{l.strength.toFixed(2)}}`,
        labelStyle: {{ fontSize: 10, fill: '#8b949e' }},
        labelBgStyle: {{ fill: '#0e1116', fillOpacity: 0.8 }},
      }}));

    return {{ nodes, edges, maxDepth }};
  }}

  function App() {{
    const data = PASS_DATA;
    const {{ nodes: allNodes, edges: allEdges }} = React.useMemo(() => buildGraph(data), []);
    const meta = data.pass_meta || {{}};
    const [deltaOnly, setDeltaOnly] = React.useState(false);
    const nMoved = allNodes.filter(n => (n.data._aDelta || 0) >= 0.02).length;

    const {{ nodes, edges }} = React.useMemo(() => {{
      if (!deltaOnly) return {{ nodes: allNodes, edges: allEdges }};
      // Moved set: nodes with |delta| >= 0.02
      const moved = new Set(allNodes.filter(n => (n.data._aDelta || 0) >= 0.02).map(n => n.id));
      // Snapshot before expanding so we don't transitively close the set
      const keep = new Set(moved);
      allEdges.forEach(e => {{
        if (moved.has(e.target)) keep.add(e.source);  // direct parents
        if (moved.has(e.source)) keep.add(e.target);  // direct children
      }});
      const nodes = allNodes.filter(n => keep.has(n.id));
      const edges = allEdges.filter(e => keep.has(e.source) && keep.has(e.target));
      return {{ nodes, edges }};
    }}, [deltaOnly]);

    return e(React.Fragment, null,
      e('div', {{ className: 'header' }},
        e('b', null, 'Ecology Belief Network — '),
        `pass: ${{data.pass_id}}`,
        meta.ticker ? `  ·  ticker: ${{meta.ticker}}` : '',
        meta.scenario ? `  ·  scenario: ${{meta.scenario}}` : '',
        meta.label ? `  ·  label: ${{meta.label}}` : '',
        meta.prediction ? `  ·  pred: ${{meta.prediction}}` : '',
        meta.correct !== undefined ? `  ·  ${{meta.correct ? '✓' : '✗'}}` : '',
        `  ·  moved: ${{nMoved}} / ${{allNodes.length}}`,
      ),
      e('button', {{
        className: `delta-toggle${{deltaOnly ? ' active' : ''}}`,
        onClick: () => setDeltaOnly(v => !v),
      }}, deltaOnly ? `showing moved + parents (${{nodes.length}})` : `show only moved (${{nMoved}})`),
      e('div', {{ style: {{ width: '100%', height: '100%' }} }},
        e(window.ReactFlow.default, {{
          nodes, edges,
          fitView: true,
          fitViewOptions: {{ padding: 0.2 }},
          minZoom: 0.02,
          maxZoom: 4.0,
          nodesDraggable: true,
          nodesConnectable: false,
          panOnScroll: true,
          zoomOnScroll: true,
          zoomOnPinch: true,
          panOnDrag: true,
          proOptions: {{ hideAttribution: true }},
        }},
          e(window.ReactFlow.Background, {{ color: '#21262d' }}),
          e(window.ReactFlow.Controls, {{ showInteractive: true }}),
          e(window.ReactFlow.MiniMap, {{ pannable: true, zoomable: true, nodeColor: (n) => (n.data._aDelta||0) >= 0.02 ? '#ffd33d' : '#58a6ff', maskColor: 'rgba(14,17,22,0.7)', style: {{ background: '#161b22', border: '1px solid #30363d' }} }}),
        )),
      e('div', {{ className: 'panel' }},
        e('h3', null, `incoming activations (${{data.activations_this_pass.length}})`),
        data.activations_this_pass.length === 0
          ? e('div', {{ style: {{ color: '#6e7681' }} }}, 'no activations this pass')
          : data.activations_this_pass.map((a, i) => e('div', {{ key: i, className: 'activation' }},
              e('div', null,
                e('span', {{ className: 'magnitude' }}, a.magnitude.toUpperCase()),
                ' ', a.direction_of_effect, ' → ',
                e('span', {{ className: 'target' }}, a.target_belief_id.slice(0, 50)),
              ),
              e('div', {{ style: {{ color: '#8b949e', fontSize: 10 }} }}, `species: ${{a.species_id}} · decay: ${{a.decay_class}} · conf: ${{a.self_rated_confidence}}`),
              a.reasoning ? e('div', {{ className: 'reasoning' }}, a.reasoning) : null,
            )),
        e('h3', {{ style: {{ marginTop: 12 }} }}, `outcomes (${{data.outcomes_this_pass.length}})`),
        data.outcomes_this_pass.map((o, i) => {{
          const cls = o.p_up >= 0.5 ? 'up' : 'down';
          return e('div', {{ key: i, className: `outcome-card ${{cls}}` }},
            e('div', null, e('b', null, `P(up) = ${{o.p_up.toFixed(3)}}`)),
            e('div', {{ style: {{ color: '#8b949e', fontSize: 10 }} }}, `${{o.contributing_state_beliefs.length}} contributing beliefs · ${{o.contributing_links.length}} links`),
            o.actual_direction ? e('div', {{ style: {{ marginTop: 4, color: o.actual_direction === (o.p_up >= 0.5 ? 'up' : 'down') ? '#3fb950' : '#f85149' }} }},
              `actual: ${{o.actual_direction}}`) : null,
          );
        }}),
      ),
    );
  }}

  ReactDOM.createRoot(document.getElementById('root')).render(e(App));
</script>
</body>
</html>
"""


def render_pass_viz(
    snapshot: dict,
    out_dir: Path,
    pass_id: str,
) -> tuple[Path, Path]:
    """Write snapshot json + self-contained html to out_dir.

    Returns (json_path, html_path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"pass_{pass_id}.json"
    html_path = out_dir / f"pass_{pass_id}.html"

    with json_path.open("w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    html = _HTML_TEMPLATE.format(
        pass_id=pass_id,
        pass_data_json=json.dumps(snapshot, default=str),
    )
    with html_path.open("w") as f:
        f.write(html)

    # Symlink-style "latest"
    latest = out_dir / "latest.html"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.write_text(html)

    return json_path, html_path
