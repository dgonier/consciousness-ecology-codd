// Macro / Sector / Market context strip — sits above the heatmap.
// Renders the named (non-decomposed, non-stub) beliefs from the manifest plus a
// rolled-up count of placeholder/decomposed beliefs that aren't worth chipping.

function ContextStrip({ manifest, day, setHover }) {
  const nodes = manifest.belief_catalog.nodes;

  const isStub = (n) => n.statement_template.startsWith("(stub");
  const isDecomp = (n) => n.id.includes("decomposed");

  const named = React.useMemo(() => {
    return nodes.filter(n =>
      (n.scope === "macro" || n.scope === "sector" || n.scope === "market") &&
      !isStub(n) && !isDecomp(n)
    );
  }, [nodes]);

  const stubCount = nodes.filter(n =>
    (n.scope === "macro" || n.scope === "sector" || n.scope === "market") && isStub(n) && !isDecomp(n)
  ).length;
  const decompCount = nodes.filter(n => isDecomp(n)).length;

  // Index of activations for the day, by belief_id.
  const activeById = {};
  (day.belief_deltas || []).forEach(b => { activeById[b.belief_id] = b; });

  const groups = {
    macro:   named.filter(n => n.scope === "macro"),
    market:  named.filter(n => n.scope === "market"),
    sector:  named.filter(n => n.scope === "sector"),
  };

  // Categorize market clusters into "high" / "low" cluster pairs by sector
  const clusterBySector = {};
  groups.market.forEach(n => {
    const m = n.id.match(/cluster_correlation_(\w+)_(high|low)/);
    if (!m) return;
    const sec = m[1];
    (clusterBySector[sec] ||= {})[m[2]] = n;
  });

  const onChipEnter = (e, n) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const active = activeById[n.id];
    setHover({
      kind: "macro", id: n.id, scope: n.scope,
      statement: n.statement_template,
      p_after: active?.p_after, p_before: active?.p_before, delta: active?.delta,
      reason: active?.sample_reasoning,
      n_act: active?.n_activations,
      decay: n.decay_class, prior: n.prior_p,
      _cx: rect.left + rect.width / 2,
      _cy: rect.top,
    });
  };
  const onChipLeave = () => setHover(null);

  const Chip = ({ n }) => {
    const a = activeById[n.id];
    const fired = !!a;
    const dir = a ? (a.delta > 0 ? "up" : "dn") : null;
    return (
      <button
        className={"ctx-chip ctx-" + n.scope + (fired ? " fired " + dir : "")}
        onMouseEnter={(e) => onChipEnter(e, n)}
        onMouseLeave={onChipLeave}
      >
        <span className="ctx-chip-text">{n.statement_template}</span>
        {fired ? (
          <span className={"ctx-chip-p " + dir}>
            <span className="ctx-chip-arrow">{a.delta > 0 ? "↑" : "↓"}</span>
            {a.p_after.toFixed(2)}
          </span>
        ) : (
          <span className="ctx-chip-p dorm">p={n.prior_p.toFixed(2)}</span>
        )}
      </button>
    );
  };

  // Cluster pair render — each sector has a pair of opposing beliefs sharing one bar.
  const ClusterPair = ({ sector, pair }) => {
    const high = pair.high, low = pair.low;
    const aH = high && activeById[high.id], aL = low && activeById[low.id];
    const onPair = (e, target) => onChipEnter(e, target);
    const sectorLabel = { tech: "Tech", fin: "Financial", health: "Health", consumer_disc: "Consumer disc", consumer_staples: "Consumer staples", energy: "Energy" }[sector] || sector;
    return (
      <div className="ctx-cluster" onMouseLeave={onChipLeave}>
        <div className="ctx-cluster-label">{sectorLabel}</div>
        <div className="ctx-cluster-bar">
          {high && (
            <button
              className={"ctx-cluster-half hi" + (aH ? " fired" : "")}
              onMouseEnter={(e) => onPair(e, high)}
              title={high.statement_template}
            >
              <span className="ctx-cluster-half-l">correlated</span>
              <span className="ctx-cluster-half-p">{aH ? aH.p_after.toFixed(2) : high.prior_p.toFixed(2)}</span>
            </button>
          )}
          {low && (
            <button
              className={"ctx-cluster-half lo" + (aL ? " fired" : "")}
              onMouseEnter={(e) => onPair(e, low)}
              title={low.statement_template}
            >
              <span className="ctx-cluster-half-l">broken</span>
              <span className="ctx-cluster-half-p">{aL ? aL.p_after.toFixed(2) : low.prior_p.toFixed(2)}</span>
            </button>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="ctx-strip">
      <div className="ctx-row">
        <div className="ctx-row-label">
          <span className="ctx-row-l-1">macro</span>
          <span className="ctx-row-l-2">{groups.macro.length} named · {nodes.filter(n=>n.scope==='macro'&&isStub(n)).length} stub</span>
        </div>
        <div className="ctx-row-chips">
          {groups.macro.map(n => <Chip key={n.id} n={n} />)}
          {groups.sector.map(n => <Chip key={n.id} n={n} />)}
        </div>
      </div>

      <div className="ctx-row ctx-row-clusters">
        <div className="ctx-row-label">
          <span className="ctx-row-l-1">market clusters</span>
          <span className="ctx-row-l-2">{Object.keys(clusterBySector).length} sectors · correlated ↔ broken</span>
        </div>
        <div className="ctx-clusters">
          {Object.entries(clusterBySector).map(([sec, pair]) => (
            <ClusterPair key={sec} sector={sec} pair={pair} />
          ))}
        </div>
      </div>

      <div className="ctx-meta">
        <span className="ctx-meta-pill">
          <span className="ctx-meta-num">{stubCount}</span>
          <span className="ctx-meta-l">stub beliefs (placeholders, no statement yet)</span>
        </span>
        <span className="ctx-meta-pill">
          <span className="ctx-meta-num">{decompCount}</span>
          <span className="ctx-meta-l">decomposed beliefs (auto-generated hashes)</span>
        </span>
        <span className="ctx-meta-pill">
          <span className="ctx-meta-num">{manifest.n_state_beliefs}</span>
          <span className="ctx-meta-l">total state beliefs in manifest</span>
        </span>
      </div>
    </div>
  );
}

window.ContextStrip = ContextStrip;
