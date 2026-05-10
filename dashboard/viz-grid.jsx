// Heatmap grid: templates × tickers + outcome strip + sector bands.
// Outcome strip shows predicted p_up dot WITH a separate "ideal" tick from goldens.

function Grid({ tickers, templates, beliefByCell, links, showLinkEdges, obsByTicker, goldenByTicker, showHalos, showIdeal, tickerFilter, setHover, mode }) {
  const wrapRef = React.useRef(null);
  const [containerW, setContainerW] = React.useState(1200);

  React.useEffect(() => {
    const el = wrapRef.current; if (!el) return;
    const ro = new ResizeObserver(entries => { for (const e of entries) setContainerW(e.contentRect.width); });
    ro.observe(el);
    setContainerW(el.getBoundingClientRect().width);
    return () => ro.disconnect();
  }, []);

  const vTickers = tickers.filter(t => tickerFilter.has(t));
  const labelW = 200;
  const headerH = 56;
  const outcomeH = 68;
  const padR = 16;
  const cellW = Math.max(28, Math.min(64, Math.floor((containerW - labelW - padR - 16) / Math.max(1, vTickers.length))));
  const cellH = Math.max(22, Math.min(34, Math.round(cellW * 0.78)));
  const w = labelW + cellW * vTickers.length + padR;
  const h = headerH + cellH * templates.length + outcomeH + 12;

  const cellFill = (p) => {
    if (p == null) return "var(--cell-empty)";
    const dev = p - 0.5;
    const intensity = Math.min(1, Math.abs(dev) * 2);
    if (Math.abs(dev) < 0.02) return "var(--cell-neutral)";
    if (dev > 0) return `oklch(${0.32 + intensity * 0.32} ${0.10 + intensity * 0.06} 145 / ${0.30 + intensity * 0.65})`;
    return `oklch(${0.32 + intensity * 0.30} ${0.10 + intensity * 0.07} 25 / ${0.30 + intensity * 0.65})`;
  };
  const outcomeFill = (p) => {
    const dev = p - 0.5;
    const intensity = Math.min(1, Math.abs(dev) * 2.5);
    if (Math.abs(dev) < 0.02) return "var(--cell-neutral)";
    if (dev > 0) return `oklch(${0.40 + intensity * 0.30} ${0.13 + intensity * 0.05} 145 / ${0.55 + intensity * 0.45})`;
    return `oklch(${0.40 + intensity * 0.28} ${0.13 + intensity * 0.06} 25 / ${0.55 + intensity * 0.45})`;
  };
  const haloColor = (g, predUp) => {
    if (!g || g.actual_direction === "flat") return "var(--halo-flat)";
    const correct = (g.actual_direction === "up") === predUp;
    return correct ? "var(--halo-good)" : "var(--halo-bad)";
  };

  const sectorBands = [];
  let curS = null, curStart = 0;
  vTickers.forEach((t, i) => {
    const s = TICKER_SECTORS[t];
    if (s !== curS) {
      if (curS !== null) sectorBands.push({ s: curS, start: curStart, end: i });
      curS = s; curStart = i;
    }
  });
  sectorBands.push({ s: curS, start: curStart, end: vTickers.length });

  const onCellEnter = (e, payload) => {
    const rect = e.currentTarget.ownerSVGElement.getBoundingClientRect();
    const cx = rect.left + payload._x + cellW / 2;
    const cy = rect.top + payload._y + cellH / 2;
    setHover({ ...payload, _cx: cx, _cy: cy });
  };
  const onLeave = () => setHover(null);

  return (
    <div className="grid-wrap" ref={wrapRef}>
      <svg width={w} height={h} className="grid-svg">
        {sectorBands.map((b, i) => (
          <g key={i}>
            <rect x={labelW + b.start * cellW} y={4} width={(b.end - b.start) * cellW - 2} height={18}
              fill="var(--band-bg)" stroke="var(--band-edge)" strokeWidth="0.5" />
            <text x={labelW + b.start * cellW + ((b.end - b.start) * cellW) / 2} y={16}
              textAnchor="middle" className="band-label">{SECTOR_LABEL[b.s]}</text>
          </g>
        ))}
        {vTickers.map((t, i) => (
          <text key={t} x={labelW + i * cellW + cellW / 2} y={headerH - 6} textAnchor="middle" className="tkr-label">{t}</text>
        ))}

        {templates.map((tpl, r) => (
          <g key={tpl}>
            <text x={labelW - 8} y={headerH + r * cellH + cellH / 2 + 4} textAnchor="end"
              className={"tpl-label dir-" + (TEMPLATE_DIR(tpl) === "+" ? "pos" : "neg")}>
              <tspan className="tpl-dir">{TEMPLATE_DIR(tpl)}</tspan> {TEMPLATE_LABEL(tpl)}
            </text>
            {vTickers.map((tkr, c) => {
              const cell = beliefByCell[`${tpl}|${tkr}`];
              const ls = links[`${tpl}|${tkr}`] ?? 0;
              const x = labelW + c * cellW;
              const y = headerH + r * cellH;
              const p = cell ? cell.p_after : 0.5;
              const delta = cell ? cell.delta : 0;
              const pulsing = cell && Math.abs(delta) > 0.05;
              const lsBorder = showLinkEdges ? Math.max(0.5, ls * 2.4) : 0.5;
              const payload = { kind: "cell", template: tpl, ticker: tkr, p, delta, ls, reason: cell?.reason, n_act: cell?.n_act, p_before: cell?.p_before, _x: x, _y: y };
              return (
                <g key={tkr} onMouseEnter={(e) => onCellEnter(e, payload)} onMouseLeave={onLeave}>
                  <rect x={x + 1} y={y + 1} width={cellW - 2} height={cellH - 2}
                    fill={cellFill(p)}
                    stroke={ls > 0.01 ? `oklch(0.78 0.14 ${TEMPLATE_DIR(tpl) === "+" ? 145 : 25} / ${Math.min(1, ls)})` : "var(--cell-edge)"}
                    strokeWidth={lsBorder} rx="2" />
                  {pulsing && (
                    <rect x={x + 1} y={y + 1} width={cellW - 2} height={cellH - 2} fill="none"
                      stroke={delta > 0 ? "var(--pulse-up)" : "var(--pulse-dn)"} strokeWidth="1.5" rx="2" className="pulse" />
                  )}
                  {cell && Math.abs(delta) > 0.04 && cellW >= 30 && (
                    <text x={x + cellW / 2} y={y + cellH / 2 + 3} textAnchor="middle" className="cell-num"
                      fill={delta > 0 ? "var(--num-up)" : "var(--num-dn)"}>
                      {(p).toFixed(2).replace(/^0/, "")}
                    </text>
                  )}
                </g>
              );
            })}
          </g>
        ))}

        {/* Outcome row: predicted dot + ideal tick from goldens */}
        <g>
          <text x={labelW - 8} y={headerH + templates.length * cellH + outcomeH / 2 + 4}
            textAnchor="end" className="outcome-row-label">
            <tspan x={labelW - 8} dy="-6">prediction</tspan>
            <tspan x={labelW - 8} dy="14" className="outcome-row-label-sub">vs reality →</tspan>
          </text>
          {vTickers.map((tkr, c) => {
            const o = obsByTicker[tkr];
            const g = goldenByTicker[tkr];
            const x = labelW + c * cellW;
            const y = headerH + templates.length * cellH + 6;
            const cy = y + outcomeH / 2 - 4;
            const pPred = o ? o.p_up : 0.5;
            const pIdeal = g?.ideal_p_up;
            const dev = pPred - 0.5;
            const sal = o?.salience ?? 0;
            const r = 4 + Math.min(13, Math.abs(dev) * 22 + sal * 6);
            const halo = haloColor(g, pPred > 0.5);
            const payload = { kind: "outcome", ticker: tkr, obs: o, golden: g, _x: x, _y: y };
            // Ideal tick: render as a small vertical bar at the ideal x-offset within the cell
            const idealX = pIdeal != null ? x + cellW * pIdeal : null;
            return (
              <g key={tkr} onMouseEnter={(e) => onCellEnter(e, payload)} onMouseLeave={onLeave}>
                {showHalos && g && (
                  <circle cx={x + cellW / 2} cy={cy} r={r + 4} fill="none" stroke={halo} strokeWidth="1.2" opacity="0.85" />
                )}
                {o && (
                  <circle cx={x + cellW / 2} cy={cy} r={r} fill={outcomeFill(pPred)} stroke="var(--outcome-edge)" strokeWidth="0.5" />
                )}
                {/* Predicted p_up number */}
                {o && (
                  <text x={x + cellW / 2} y={cy + 3} textAnchor="middle" className="outcome-num">
                    {pPred.toFixed(2).replace(/^0/, "")}
                  </text>
                )}
                {/* Ideal marker — small target chevron above the dot */}
                {showIdeal && pIdeal != null && (
                  <g>
                    <line x1={x + 2} x2={x + cellW - 2}
                      y1={y + outcomeH - 14} y2={y + outcomeH - 14}
                      stroke="var(--ideal-rail)" strokeWidth="0.5" />
                    <polygon
                      points={`${x + cellW * pIdeal},${y + outcomeH - 12} ${x + cellW * pIdeal - 3},${y + outcomeH - 16} ${x + cellW * pIdeal + 3},${y + outcomeH - 16}`}
                      fill={pIdeal > 0.5 ? "var(--ideal-up)" : "var(--ideal-dn)"}
                      stroke="var(--ideal-edge)" strokeWidth="0.5" />
                    <text x={x + cellW / 2} y={y + outcomeH - 1} textAnchor="middle" className="outcome-ideal-num">
                      {pIdeal.toFixed(2).replace(/^0/, "")}
                    </text>
                  </g>
                )}
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}

window.Grid = Grid;
