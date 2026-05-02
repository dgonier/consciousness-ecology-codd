import React, { useEffect, useMemo, useState } from 'react';
import ReactFlow, {
  Background,
  Controls,
  Handle,
  Position,
  ReactFlowProvider,
} from 'reactflow';
import 'reactflow/dist/style.css';

const TIER_LABELS = {
  0: 'scenario',
  1: 'producers (Tier 1)',
  2: 'producer-trough (Tier 2)',
  3: 'herbivores (Tier 3)',
  4: 'apex inputs PRE-@E (Tier 4)',
  5: 'apex output POST-@E (Tier 5)',
};

const TIER_X = { 0: 60, 1: 360, 2: 720, 3: 1080, 4: 1480, 5: 1920 };
const TIER_COLOR = {
  0: '#2c3140',
  1: '#1e3a5f',
  2: '#1f4f3a',
  3: '#5f4a1e',
  4: '#5f1e3a',
  5: '#3a1e5f',
};

function logitLensSummary(lens) {
  if (!lens || !lens.length) return null;
  return lens.slice(0, 5).map(({ token, prob }) =>
    `${JSON.stringify(token)}(${(prob * 100).toFixed(0)}%)`
  ).join('  ');
}

function NodeBox({ data }) {
  const n = data.node;
  const lens = logitLensSummary(n.logit_lens);
  const isOutput = n.tier === 5;
  return (
    <div
      style={{
        background: TIER_COLOR[n.tier] || '#222',
        border: '1px solid #555',
        borderRadius: 6,
        padding: '8px 10px',
        minWidth: 280,
        maxWidth: 360,
        fontSize: 11,
        lineHeight: 1.4,
        color: '#e6e9ee',
      }}
    >
      <Handle type="target" position={Position.Left} style={{ background: '#888' }} />
      <div style={{ fontWeight: 700, marginBottom: 4 }}>{n.label || n.id}</div>
      <div style={{ opacity: 0.7, fontSize: 10 }}>id: {n.id}</div>
      {n.norm != null && <div>norm: {n.norm.toFixed(2)}</div>}
      {n.norm_M_A != null && (
        <div>
          ||M_A||: {n.norm_M_A.toFixed(2)} ||E_A||: {n.norm_E_A.toFixed(2)}<br />
          s_M: {n.s_M.toFixed(4)} s_E: {n.s_E.toFixed(4)}
        </div>
      )}
      {n.cumulative_attention != null && (
        <div>cum_attn: {n.cumulative_attention.toFixed(3)}</div>
      )}
      {n.diet_tags && n.diet_tags.length > 0 && (
        <div style={{ opacity: 0.75 }}>diet: {n.diet_tags.join(', ')}</div>
      )}
      {lens && (
        <div style={{ marginTop: 4, padding: '4px 6px', background: '#0e1116', borderRadius: 4 }}>
          <span style={{ opacity: 0.6 }}>logit-lens:</span><br />
          <span style={{ color: '#9ad29a' }}>{lens}</span>
        </div>
      )}
      {n.raw_text && (
        <div style={{ marginTop: 4, padding: '4px 6px', background: '#0e1116', borderRadius: 4, maxHeight: 80, overflow: 'auto' }}>
          <span style={{ opacity: 0.6 }}>raw_text:</span><br />
          <span style={{ color: '#dabb88' }}>{n.raw_text}</span>
        </div>
      )}
      {isOutput && (
        <div style={{ marginTop: 4, padding: '4px 6px', background: n.match ? '#15391f' : '#3a1515', borderRadius: 4 }}>
          parsed: {n.parsed_ticker || '?'} / {n.parsed_direction || '?'} (conf={n.parsed_confidence ?? '?'})<br />
          target: {n.target_direction || '?'} → {n.match ? 'CORRECT' : 'WRONG'}
        </div>
      )}
      <Handle type="source" position={Position.Right} style={{ background: '#888' }} />
    </div>
  );
}

const nodeTypes = { box: NodeBox };

function parseJsonl(text) {
  return text
    .split('\n')
    .map(l => l.trim())
    .filter(Boolean)
    .map(l => JSON.parse(l));
}

function layoutNodes(rawNodes) {
  const byTier = {};
  for (const n of rawNodes) {
    (byTier[n.tier] ||= []).push(n);
  }
  const nodes = [];
  for (const tier of Object.keys(byTier).map(Number).sort((a, b) => a - b)) {
    const list = byTier[tier];
    const yStep = 260;
    const yStart = 40;
    list.forEach((n, i) => {
      nodes.push({
        id: n.id,
        type: 'box',
        position: { x: TIER_X[tier] ?? (tier * 360), y: yStart + i * yStep },
        data: { node: n },
      });
    });
  }
  const edges = [];
  for (const n of rawNodes) {
    for (const p of (n.parents || [])) {
      edges.push({
        id: `${p}->${n.id}`,
        source: p,
        target: n.id,
        animated: false,
        style: { stroke: '#5b6677', strokeWidth: 1.2 },
      });
    }
  }
  return { nodes, edges };
}

export default function App() {
  const [files, setFiles] = useState([]);
  const [active, setActive] = useState(null);
  const [raw, setRaw] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/signals/')
      .then(r => r.json())
      .then(list => {
        setFiles(list);
        if (list.length > 0) setActive(list[0]);
      })
      .catch(e => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!active) return;
    fetch(`/signals/${active}`)
      .then(r => r.text())
      .then(t => setRaw(parseJsonl(t)))
      .catch(e => setError(String(e)));
  }, [active]);

  const { nodes, edges } = useMemo(() => layoutNodes(raw), [raw]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <div style={{ padding: 10, borderBottom: '1px solid #2c3140', background: '#161a22', display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <strong>Trophic Signal Viz</strong>
        <span style={{ opacity: 0.7 }}>scenario:</span>
        <select
          value={active || ''}
          onChange={e => setActive(e.target.value)}
          style={{ background: '#0e1116', color: '#d6dbe1', border: '1px solid #2c3140', padding: '4px 8px' }}
        >
          {files.map(f => (
            <option key={f} value={f}>{f}</option>
          ))}
        </select>
        <span style={{ opacity: 0.7 }}>nodes: {raw.length}</span>
        {error && <span style={{ color: '#ff8888' }}>err: {error}</span>}
        <span style={{ marginLeft: 'auto', display: 'flex', gap: 12 }}>
          {Object.entries(TIER_LABELS).map(([t, l]) => (
            <span key={t} style={{ fontSize: 11, padding: '2px 8px', background: TIER_COLOR[t], borderRadius: 4 }}>
              {l}
            </span>
          ))}
        </span>
      </div>
      <div style={{ flex: 1 }}>
        <ReactFlowProvider>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            fitView
            minZoom={0.2}
            maxZoom={1.5}
          >
            <Background color="#1f2530" gap={24} />
            <Controls />
          </ReactFlow>
        </ReactFlowProvider>
      </div>
    </div>
  );
}
