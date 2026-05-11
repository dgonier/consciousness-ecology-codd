// Holdings bar chart: 9 horizontal stacked bars (cash + positions) for every
// pipeline (3 apex models × {bare, ecology, oracle}). Scale fixed 0-$150K.

const HOLDINGS_MAX = 150000;

function HoldingsBar({ day, setHover }) {
  const fmt = (n) => n == null ? "—" : "$" + Math.round(n).toLocaleString();
  const pct = (n) => n == null ? "" : (n * 100).toFixed(1) + "%";

  const renderBar = (label, p, family, model) => {
    if (!p) {
      return (
        <div className={"hold-row hold-row-empty fam-" + family} key={label}>
          <div className="hold-row-l">
            <div className="hold-row-name">{label}</div>
            <div className="hold-row-meta">no snapshot</div>
          </div>
          <div className="hold-row-bar" />
        </div>
      );
    }
    const cash = p.cash || 0;
    // Stable canonical order across all 9 rows so the same ticker lands at
    // roughly the same x-position regardless of pipeline: sector first
    // (SECTOR_ORDER: tech → fin → health → consum → energy), then ticker
    // alphabetical within the sector. Cash is always rendered first by the
    // JSX below; this just orders the position segments after it.
    const sectorIdx = (t) => SECTOR_ORDER.indexOf(TICKER_SECTORS[t] || "other");
    const positions = (p.open_positions || []).slice().sort((a, b) => {
      const sa = sectorIdx(a.ticker), sb = sectorIdx(b.ticker);
      if (sa !== sb) return sa - sb;
      return a.ticker < b.ticker ? -1 : 1;
    });
    const totalMV = positions.reduce((s, p) => s + p.mv, 0);
    const equity = p.equity || (cash + totalMV);

    const onSegEnter = (e, kind, segLabel, value, extras) => {
      const rect = e.currentTarget.getBoundingClientRect();
      setHover({
        kind: "holding", segKind: kind, label: segLabel, value,
        portfolio: label,
        ...extras,
        _cx: rect.left + rect.width / 2,
        _cy: rect.top,
      });
    };
    const onSegLeave = () => setHover(null);

    return (
      <div className={"hold-row fam-" + family + " mod-" + model} key={label}>
        <div className="hold-row-l">
          <div className="hold-row-name">{label}</div>
          <div className="hold-row-eq">{fmt(equity)}</div>
          <div className="hold-row-meta">
            {p.n_open_positions} pos · {pct(p.invested_frac)} invested
          </div>
        </div>
        <div className="hold-row-bar">
          {cash > 0 && (
            <div
              className="hold-seg hold-seg-cash"
              style={{ width: `${(cash / HOLDINGS_MAX) * 100}%` }}
              onMouseEnter={(e) => onSegEnter(e, "cash", "CASH", cash, { invested_frac: p.invested_frac })}
              onMouseLeave={onSegLeave}
            >
              {cash / HOLDINGS_MAX > 0.04 && <span className="hold-seg-l">CASH</span>}
            </div>
          )}
          {positions.map(pos => {
            const sector = TICKER_SECTORS[pos.ticker] || "other";
            const w = (pos.mv / HOLDINGS_MAX) * 100;
            const showLabel = w > 2.5;
            return (
              <div
                key={pos.ticker}
                className={"hold-seg hold-seg-pos sector-" + sector + (pos.unrealized_pnl > 0 ? " up" : pos.unrealized_pnl < 0 ? " dn" : "")}
                style={{ width: `${w}%` }}
                onMouseEnter={(e) => onSegEnter(e, "pos", pos.ticker, pos.mv, {
                  cost_basis: pos.cost_basis, unrealized_pnl: pos.unrealized_pnl,
                  unrealized_pnl_pct: pos.unrealized_pnl_pct, shares: pos.shares,
                  sector,
                })}
                onMouseLeave={onSegLeave}
              >
                {showLabel && <span className="hold-seg-l">{pos.ticker}</span>}
              </div>
            );
          })}
          <div className="hold-seg-empty" style={{ width: `${(1 - equity / HOLDINGS_MAX) * 100}%` }} />
        </div>
      </div>
    );
  };

  const ticks = [0, 25000, 50000, 75000, 100000, 125000, 150000];

  const PIPELINES_DISPLAY = [
    { key: "bare_qwen",      label: "BARE-QWEN",      family: "bare",    model: "qwen"   },
    { key: "bare_sonnet",    label: "BARE-SONNET",    family: "bare",    model: "sonnet" },
    { key: "bare_opus",      label: "BARE-OPUS",      family: "bare",    model: "opus"   },
    { key: "ecology_qwen",   label: "ECO-QWEN",       family: "ecology", model: "qwen"   },
    { key: "ecology_sonnet", label: "ECO-SONNET",     family: "ecology", model: "sonnet" },
    { key: "ecology_opus",   label: "ECO-OPUS",       family: "ecology", model: "opus"   },
    { key: "oracle_qwen",    label: "ORACLE-QWEN",    family: "oracle",  model: "qwen"   },
    { key: "oracle_sonnet",  label: "ORACLE-SONNET",  family: "oracle",  model: "sonnet" },
    { key: "oracle_opus",    label: "ORACLE-OPUS",    family: "oracle",  model: "opus"   },
  ];

  // Today's orders preview — pull from each pipeline that has any.
  const ordersByFamily = { bare: [], ecology: [], oracle: [] };
  PIPELINES_DISPLAY.forEach(({ key, family, model }) => {
    const p = day.portfolio?.[key];
    (p?.orders_today || []).forEach(o => {
      ordersByFamily[family].push({ ...o, model });
    });
  });

  return (
    <div className="holdings">
      <div className="holdings-head">
        <div className="holdings-titlebar">
          <span className="holdings-eyebrow">portfolio · {day.date}</span>
          <span className="holdings-h">Holdings by day · 9-way ablation</span>
        </div>
        <div className="holdings-orders">
          {["bare", "ecology", "oracle"].map(fam => {
            const list = ordersByFamily[fam];
            return (
              <div key={fam} className={"ord-group fam-" + fam}>
                <span className={"ord-l " + fam}>{fam.toUpperCase()}</span>
                {list.length === 0 ? (
                  <span className="ord-empty">no trades</span>
                ) : list.map((o, i) => (
                  <span key={fam+i} className={"ord " + o.side.toLowerCase()} title={`${fam}-${o.model}`}>
                    <span className="ord-side">{o.side === "BUY" ? "+" : "−"}</span>
                    <span className="ord-tkr">{o.ticker}</span>
                    <span className="ord-amt">${Math.round(o.dollars_intent).toLocaleString()}</span>
                    <span className="ord-mod">{o.model[0]}</span>
                  </span>
                ))}
              </div>
            );
          })}
        </div>
      </div>

      <div className="hold-axis">
        <div className="hold-axis-spacer" />
        <div className="hold-axis-track">
          {ticks.map(t => (
            <div
              key={t}
              className={"hold-tick " + (t === 100000 ? "seed" : "")}
              style={{ left: `${(t / HOLDINGS_MAX) * 100}%` }}
            >
              <span className="hold-tick-l">${(t / 1000)}k{t === 100000 ? " · seed" : ""}</span>
            </div>
          ))}
        </div>
      </div>

      {PIPELINES_DISPLAY.map(({ key, label, family, model }) =>
        renderBar(label, day.portfolio?.[key], family, model)
      )}

      <PositionHeatmap day={day} pipelines={PIPELINES_DISPLAY} setHover={setHover} />
      <TopSignals day={day} setHover={setHover} />
    </div>
  );
}

// Top-3 signal cards. Picked from ECOLOGY-QWEN's observations (only pipeline
// with full causal-chain data) ranked by conviction = salience × |p_up - 0.5|.
// One card per signal. Re-renders every day as the scrubber moves.
function TopSignals({ day, setHover }) {
  const obs = day.observations || [];
  const ranked = obs
    .map(o => ({ ...o, _conv: (o.salience ?? 0) * Math.abs((o.p_up ?? 0.5) - 0.5) }))
    .sort((a, b) => b._conv - a._conv)
    .slice(0, 3);

  if (ranked.length === 0) {
    return (
      <div className="topsig-wrap">
        <div className="topsig-head">
          <span className="topsig-eyebrow">today's top signals</span>
          <span className="topsig-sub">no observations on this day</span>
        </div>
      </div>
    );
  }

  const onCardEnter = (e, o) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setHover({
      kind: "outcome",
      ticker: o.ticker,
      obs: o,
      golden: null,
      _cx: rect.left + rect.width / 2,
      _cy: rect.top,
      _x: 0, _y: 0,
    });
  };
  const onLeave = () => setHover(null);

  return (
    <div className="topsig-wrap">
      <div className="topsig-head">
        <span className="topsig-eyebrow">today's top signals</span>
        <span className="topsig-sub">ranked by salience × |p_up − 0.5| · ECOLOGY-QWEN</span>
      </div>
      <div className="topsig-cards">
        {ranked.map((o, i) => {
          const dir = o.p_up > 0.5 ? "up" : "dn";
          const action = o.action || "WATCH";
          const top = (o.causal_chain || [])[0];
          const topLabel = top ? (top.parent_id || "").replace(/^belief\.company\./, "").replace(/__ticker_.*$/, "").replace(/_/g, " ") : null;
          return (
            <div key={i} className={"topsig-card dir-" + dir + " act-" + action.toLowerCase()}
                 onMouseEnter={(e) => onCardEnter(e, o)} onMouseLeave={onLeave}>
              <div className="topsig-rank">#{i + 1}</div>
              <div className="topsig-body">
                <div className="topsig-l1">
                  <span className={"topsig-action act-" + action.toLowerCase()}>{action}</span>
                  <span className="topsig-tkr">{o.ticker}</span>
                  <span className={"topsig-pup " + dir}>p_up {o.p_up?.toFixed(2)}</span>
                </div>
                <div className="topsig-l2">
                  <span className="topsig-conf">{o.confidence || "—"}</span>
                  <span className="topsig-sal">salience {o.salience?.toFixed(2)}</span>
                  {topLabel && <span className="topsig-top">↑ {topLabel}</span>}
                </div>
                {o.reasoning_summary && (
                  <div className="topsig-reason" title={o.reasoning_summary}>
                    {o.reasoning_summary}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Position heatmap: rows = 9 strategies, columns = TICKERS (column-aligned with
// the belief grid below). Cell color encodes invested-percent of equity (the
// position's mv as a fraction of the portfolio's equity that day). Empty cell
// = no position; brighter = larger weight.
function PositionHeatmap({ day, pipelines, setHover }) {
  const wrapRef = React.useRef(null);
  const [containerW, setContainerW] = React.useState(1200);

  React.useEffect(() => {
    const el = wrapRef.current; if (!el) return;
    const ro = new ResizeObserver(entries => { for (const e of entries) setContainerW(e.contentRect.width); });
    ro.observe(el);
    setContainerW(el.getBoundingClientRect().width);
    return () => ro.disconnect();
  }, []);

  // Mirror viz-grid sizing exactly so columns line up with the belief grid.
  const labelW = 200;
  const padR = 16;
  const headerH = 22;
  const rowH = 18;
  const cellW = Math.max(28, Math.min(64, Math.floor((containerW - labelW - padR - 16) / Math.max(1, TICKERS.length))));
  const w = labelW + cellW * TICKERS.length + padR;
  const h = headerH + rowH * pipelines.length + 6;

  // White cells, opacity scales with weight %. Rescaled per day: the largest
  // single position across all 9 strategies × 20 tickers maps to opacity 1,
  // empty maps to opacity 0. So the busiest cell on the day always reads as
  // fully opaque regardless of whether the day's biggest bet is 12% or 40%.
  let dayMaxFrac = 0;
  pipelines.forEach(pipe => {
    const p = day.portfolio?.[pipe.key];
    const eq = p?.equity || 0;
    if (!eq) return;
    (p.open_positions || []).forEach(pos => {
      const f = pos.mv / eq;
      if (f > dayMaxFrac) dayMaxFrac = f;
    });
  });
  // Floor on the denominator so a quiet day (e.g., 5% max) doesn't blow up
  // tiny positions to opaque. Below 8% max we still use 8% as the scale.
  const denom = Math.max(0.08, dayMaxFrac);
  const cellFill = (frac) => {
    if (!frac || frac <= 0) return "rgba(255,255,255,0)";
    const a = Math.min(1, frac / denom);
    return `rgba(255,255,255,${a.toFixed(3)})`;
  };

  // Sector bands (mirror grid)
  const sectorBands = [];
  let curS = null, curStart = 0;
  TICKERS.forEach((t, i) => {
    const s = TICKER_SECTORS[t];
    if (s !== curS) {
      if (curS !== null) sectorBands.push({ s: curS, start: curStart, end: i });
      curS = s; curStart = i;
    }
  });
  sectorBands.push({ s: curS, start: curStart, end: TICKERS.length });

  const onCellEnter = (e, payload) => {
    const rect = e.currentTarget.ownerSVGElement.getBoundingClientRect();
    const cx = rect.left + payload._x + cellW / 2;
    const cy = rect.top + payload._y + rowH / 2;
    setHover({ ...payload, _cx: cx, _cy: cy });
  };
  const onLeave = () => setHover(null);

  return (
    <div className="pos-heatmap-wrap" ref={wrapRef}>
      <div className="pos-heatmap-head">
        <span className="pos-heatmap-eyebrow">positions · weight % of equity</span>
        <span className="pos-heatmap-sub">9 strategies × 20 tickers · empty = no position</span>
      </div>
      <svg width={w} height={h} className="pos-heatmap-svg">
        {sectorBands.map((b, i) => (
          <g key={i}>
            <rect x={labelW + b.start * cellW} y={2} width={(b.end - b.start) * cellW - 2} height={14}
              fill="var(--band-bg)" stroke="var(--band-edge)" strokeWidth="0.5" />
            <text x={labelW + b.start * cellW + ((b.end - b.start) * cellW) / 2} y={12}
              textAnchor="middle" className="band-label">{SECTOR_LABEL[b.s]}</text>
          </g>
        ))}
        {TICKERS.map((t, i) => (
          <text key={t} x={labelW + i * cellW + cellW / 2} y={headerH - 4}
            textAnchor="middle" className="tkr-label">{t}</text>
        ))}

        {pipelines.map((pipe, r) => {
          const p = day.portfolio?.[pipe.key];
          const equity = p?.equity || 1;
          const posByTkr = {};
          (p?.open_positions || []).forEach(pos => { posByTkr[pos.ticker] = pos; });
          const yRow = headerH + r * rowH;

          return (
            <g key={pipe.key}>
              <text x={labelW - 8} y={yRow + rowH / 2 + 3} textAnchor="end"
                className={"pos-row-label fam-" + pipe.family}>
                {pipe.label}
              </text>
              {TICKERS.map((tkr, c) => {
                const pos = posByTkr[tkr];
                const x = labelW + c * cellW;
                const y = yRow;
                const frac = pos ? pos.mv / equity : 0;
                const pct = frac * 100;
                const showNum = pct >= 4 && cellW >= 30;
                const payload = pos ? {
                  kind: "holding", segKind: "pos", label: tkr, value: pos.mv,
                  portfolio: pipe.label, sector: TICKER_SECTORS[tkr] || "other",
                  cost_basis: pos.cost_basis, unrealized_pnl: pos.unrealized_pnl,
                  unrealized_pnl_pct: pos.unrealized_pnl_pct, shares: pos.shares,
                  _x: x, _y: y,
                } : { kind: "pos-empty", label: tkr, portfolio: pipe.label, _x: x, _y: y };
                return (
                  <g key={tkr}
                    onMouseEnter={(e) => onCellEnter(e, payload)}
                    onMouseLeave={onLeave}>
                    <rect x={x + 1} y={y + 1} width={cellW - 2} height={rowH - 2}
                      fill={cellFill(frac)}
                      stroke="var(--cell-edge)" strokeWidth="0.5" rx="2" />
                    {showNum && (
                      <text x={x + cellW / 2} y={y + rowH / 2 + 3.5} textAnchor="middle" className="pos-cell-num">
                        {pct.toFixed(0)}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

window.HoldingsBar = HoldingsBar;
