// Right-rail spotlight: portfolio, predictions vs reality, scoreboard.

function Spotlight({ day, date, dayIdx, mode, ecoStats, bareStats, links, totalCells, equitySeries, obsByTicker, goldenByTicker, beliefByCell, setHover }) {
  const allDeltas = (day.belief_deltas || [])
    .map(b => {
      const p = parseBeliefId(b.belief_id);
      return { ...b, _scope: p.scope, _template: p.template, _ticker: p.ticker };
    })
    .filter(b => b._scope === "company" && b._ticker)
    .slice()
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

  const ecoList = (day.watchlist_ecology || []).slice(0, 9);
  const bareList = (day.watchlist_bare || []).slice(0, 9);

  const verdict = (item) => {
    const g = goldenByTicker[item.ticker];
    if (!g || g.actual_direction === "flat") return { c: "flat", s: "·" };
    if (item.action === "WATCH") return { c: "watch", s: "?" };
    const predUp = item.action === "BUY";
    return ((g.actual_direction === "up") === predUp) ? { c: "good", s: "✓" } : { c: "bad", s: "✗" };
  };

  return (
    <aside className="spot">
      {/* Portfolio panel — equity curve + today's positions */}
      <section className="spot-sec port-sec">
        <PortfolioPanel day={day} dayIdx={dayIdx} equitySeries={equitySeries} />
      </section>

      {/* Day lede */}
      <section className="spot-sec lede-sec">
        <DateLede date={date} day={day} totalCells={totalCells} allDeltas={allDeltas} />
      </section>

      {/* Predictions vs reality (today's observations) */}
      <section className="spot-sec">
        <div className="sec-head">
          <span className="sec-title">predictions vs reality · today</span>
          <span className="sec-sub">{Object.keys(obsByTicker).length} calls</span>
        </div>
        <ul className="pvr-list">
          {Object.values(obsByTicker)
            .sort((a, b) => (b.salience || 0) - (a.salience || 0))
            .map(o => {
              const g = goldenByTicker[o.ticker];
              return <PredVsReal key={o.ticker} obs={o} golden={g} />;
            })}
          {Object.keys(obsByTicker).length === 0 && (
            <li className="delta-empty">no high-salience calls today.</li>
          )}
        </ul>
      </section>

      {/* All belief deltas (activated only) */}
      <section className="spot-sec deltas-sec">
        <div className="sec-head">
          <span className="sec-title">leaf activations · today</span>
          <span className="sec-sub">{allDeltas.length} of {totalCells} cells</span>
        </div>
        <div className="sec-note">
          firehose mode resets priors to 0.5 each trading day. only cells with news activity light up. cumulative learning lives in the link-strength borders. dormant beliefs appear in the "beliefs not fired" panel below the heatmap.
        </div>
        <ul className="delta-list">
          {allDeltas.map((b, i) => (
            <li key={i} className="delta-row act">
              <div className="delta-bar-wrap">
                <div className={"delta-bar " + (b.delta > 0 ? "up" : "dn")}
                  style={{ width: `${Math.min(100, Math.abs(b.delta) * 220)}%` }} />
              </div>
              <div className="delta-txt">
                <span className="d-tkr">{b._ticker}</span>
                <span className={"d-arrow " + (b.delta > 0 ? "up" : "dn")}>{b.delta > 0 ? "↑" : "↓"}</span>
                <span className="d-tpl">{TEMPLATE_LABEL(b._template)}</span>
                <span className="d-num">{b.p_before.toFixed(2)} → <strong>{b.p_after.toFixed(2)}</strong></span>
                {b.species === "cross_correlation.v0" && <span className="d-species">↻sympathy</span>}
              </div>
              {b.sample_reasoning && <div className="d-reason">{b.sample_reasoning}</div>}
            </li>
          ))}
          {allDeltas.length === 0 && <li className="delta-empty">no belief activations on this day.</li>}
        </ul>
      </section>

      {/* Apex watchlist */}
      <section className="spot-sec">
        <div className="sec-head">
          <span className="sec-title">apex watchlist <span className="sec-tag eco">ECOLOGY</span></span>
          <span className="sec-sub">{ecoList.length} ranked</span>
        </div>
        <ul className="wl">
          {ecoList.map(item => {
            const v = verdict(item);
            const g = goldenByTicker[item.ticker];
            return (
              <li key={item.ticker} className={"wl-row " + v.c}>
                <span className="wl-rank">{String(item.rank).padStart(2, "0")}</span>
                <span className="wl-tkr">{item.ticker}</span>
                <span className={"wl-act act-" + item.action.toLowerCase()}>{item.action}</span>
                <span className="wl-p">p={item.p_up.toFixed(2)}</span>
                {g && g.ideal_p_up != null && <span className="wl-ideal" title="ideal p_up from golden">→{g.ideal_p_up.toFixed(2)}</span>}
                <span className={"wl-vrd v-" + v.c}>{v.s}</span>
                {g && g.actual_direction !== "flat" && (
                  <span className={"wl-ret " + (g.actual_return > 0 ? "up" : "dn")}>
                    {(g.actual_return * 100).toFixed(2)}%
                  </span>
                )}
                <div className="wl-reason">{item.reason}</div>
              </li>
            );
          })}
        </ul>
      </section>

      {mode === "side-by-side" && (
        <section className="spot-sec">
          <div className="sec-head">
            <span className="sec-title">apex watchlist <span className="sec-tag bare">BARE</span></span>
            <span className="sec-sub">{bareList.length} ranked · no causal grounding</span>
          </div>
          <ul className="wl">
            {bareList.slice(0, 7).map(item => {
              const v = verdict(item);
              const g = goldenByTicker[item.ticker];
              return (
                <li key={item.ticker} className={"wl-row " + v.c}>
                  <span className="wl-rank">{String(item.rank).padStart(2, "0")}</span>
                  <span className="wl-tkr">{item.ticker}</span>
                  <span className={"wl-act act-" + item.action.toLowerCase()}>{item.action}</span>
                  <span className="wl-p">p={item.p_up.toFixed(2)}</span>
                  {g && g.ideal_p_up != null && <span className="wl-ideal">→{g.ideal_p_up.toFixed(2)}</span>}
                  <span className={"wl-vrd v-" + v.c}>{v.s}</span>
                  <div className="wl-reason">{item.reason}</div>
                </li>
              );
            })}
          </ul>
        </section>
      )}

      <section className="spot-sec score-sec">
        <div className="sec-head">
          <span className="sec-title">cumulative top-5 (commit only)</span>
          <span className="sec-sub">through {date}</span>
        </div>
        <div className="score-grid">
          <ScoreRow label="ECOLOGY" stats={ecoStats} cls="eco" />
          <ScoreRow label="BARE   " stats={bareStats} cls="bare" />
        </div>
        <div className="score-foot">
          <span className="hint">MCC ranges −1..+1; 0 = chance. Flat days excluded.</span>
        </div>
      </section>
    </aside>
  );
}

// Mini equity sparkline (9 lines + buyhold) + today's positions/orders.
function PortfolioPanel({ day, dayIdx, equitySeries }) {
  const W = 320, H = 110, padL = 6, padR = 6, padT = 8, padB = 14;
  const pts = (arr) => arr.map((v, i) => ({ x: i, y: v })).filter(p => p.y != null);

  const PIPELINES = [
    { k: "bare_qwen",      family: "bare",    model: "qwen"   },
    { k: "bare_sonnet",    family: "bare",    model: "sonnet" },
    { k: "bare_opus",      family: "bare",    model: "opus"   },
    { k: "ecology_qwen",   family: "ecology", model: "qwen"   },
    { k: "ecology_sonnet", family: "ecology", model: "sonnet" },
    { k: "ecology_opus",   family: "ecology", model: "opus"   },
    { k: "oracle_qwen",    family: "oracle",  model: "qwen"   },
    { k: "oracle_sonnet",  family: "oracle",  model: "sonnet" },
    { k: "oracle_opus",    family: "oracle",  model: "opus"   },
  ];
  const allLines = PIPELINES.map(p => ({ ...p, series: pts(equitySeries[p.k] || []) }));
  const bh = pts(equitySeries.buyhold || []);
  const allYs = [...allLines.flatMap(l => l.series.map(p => p.y)), ...bh.map(p => p.y)];
  const min = Math.min(...allYs, 100000), max = Math.max(...allYs, 100000);
  const range = max - min || 1;
  const N = (equitySeries.bare_qwen || equitySeries.ecology_qwen || []).length;
  const xScale = (i) => padL + (i / Math.max(1, N - 1)) * (W - padL - padR);
  const yScale = (v) => padT + (1 - (v - min) / range) * (H - padT - padB);
  const path = (arr) => arr.map((p, i) => (i === 0 ? "M" : "L") + xScale(p.x).toFixed(1) + "," + yScale(p.y).toFixed(1)).join(" ");
  const baselineY = yScale(100000);

  // Dash style by model: qwen solid, sonnet dashed, opus dotted.
  const dashFor = (model) => model === "qwen" ? "" : model === "sonnet" ? "4 2" : "1 2";

  return (
    <div className="port">
      <div className="sec-head">
        <span className="sec-title">portfolio · profit curve</span>
        <span className="sec-sub">$100k seed · 9 strategies + buyhold</span>
      </div>
      <svg className="eq-svg" width={W} height={H}>
        <line x1={padL} x2={W - padR} y1={baselineY} y2={baselineY} stroke="var(--eq-baseline)" strokeWidth="0.5" strokeDasharray="2 3" />
        {bh.length > 1 && <path d={path(bh)} className="eq-line bh" />}
        {allLines.map(line => line.series.length > 1 && (
          <path
            key={line.k}
            d={path(line.series)}
            className={"eq-line fam-" + line.family + " mod-" + line.model}
            strokeDasharray={dashFor(line.model)}
          />
        ))}
        <line x1={xScale(dayIdx)} x2={xScale(dayIdx)} y1={padT} y2={H - padB} stroke="var(--eq-cursor)" strokeWidth="1" />
        {allLines.map(line => {
          const v = equitySeries[line.k]?.[dayIdx];
          if (v == null) return null;
          return (
            <circle
              key={"d-" + line.k}
              cx={xScale(dayIdx)}
              cy={yScale(v)}
              r={line.model === "qwen" ? 2.6 : 2}
              className={"eq-dot fam-" + line.family + " mod-" + line.model}
            />
          );
        })}
        {equitySeries.buyhold[dayIdx] != null && (
          <circle cx={xScale(dayIdx)} cy={yScale(equitySeries.buyhold[dayIdx])} r="2.5" className="eq-dot bh" />
        )}
        <text x={padL} y={H - 2} className="eq-ax">${(min/1000).toFixed(0)}k</text>
        <text x={W - padR} y={H - 2} className="eq-ax" textAnchor="end">${(max/1000).toFixed(0)}k</text>
      </svg>
      <div className="eq-leg-grid">
        {["bare", "ecology", "oracle"].map(fam => (
          <div key={fam} className={"eq-leg-fam fam-" + fam}>
            <span className={"eq-leg-fam-l " + fam}>{fam}</span>
            {["qwen", "sonnet", "opus"].map(m => {
              const v = equitySeries[`${fam}_${m}`]?.[dayIdx];
              return (
                <span key={m} className={"eq-leg-mod mod-" + m} title={`${fam} · ${m}`}>
                  <span className="leg-sw"></span>
                  <span className="leg-l">{m[0]}</span>
                  {v != null && <span className="leg-v">{(v/1000).toFixed(1)}k</span>}
                </span>
              );
            })}
          </div>
        ))}
        <div className="eq-leg-fam fam-bh">
          <Leg cls="bh" label="buyhold" v={equitySeries.buyhold[dayIdx]} />
        </div>
      </div>
      <OrdersTable9 day={day} pipelines={PIPELINES} />
    </div>
  );
}

// Orders table — rows by (side, ticker), one mini-cell per pipeline.
// Shown as 3 family columns (bare/eco/oracle); each cell stacks 3 model dots
// (q/s/o) showing dollar amount or — for that pipeline.
function OrdersTable9({ day, pipelines }) {
  const byKey = new Map();
  pipelines.forEach(({ k }) => {
    const orders = day.portfolio?.[k]?.orders_today || [];
    orders.forEach(o => {
      const key = o.side + "|" + o.ticker;
      if (!byKey.has(key)) byKey.set(key, { side: o.side, ticker: o.ticker, by: {} });
      byKey.get(key).by[k] = o;
    });
  });
  const rows = [...byKey.values()].sort((a, b) => {
    if (a.side !== b.side) return a.side === "BUY" ? -1 : 1;
    return a.ticker < b.ticker ? -1 : 1;
  });
  const families = ["bare", "ecology", "oracle"];
  const models = ["qwen", "sonnet", "opus"];

  if (rows.length === 0) return <div className="port-orders-empty">no trades today (any pipeline)</div>;

  const fmtDollars = (n) => n >= 1000 ? `$${(n/1000).toFixed(1)}k` : `$${n.toFixed(0)}`;

  return (
    <div className="port-orders-tbl9">
      <div className="pot9-head">
        <span className="pot9-h pot9-h-side">side</span>
        <span className="pot9-h pot9-h-tkr">ticker</span>
        {families.map(f => (
          <span key={f} className={"pot9-h pot9-h-fam fam-" + f}>{f}</span>
        ))}
      </div>
      <div className="pot9-subhead">
        <span className="pot9-h pot9-h-side"></span>
        <span className="pot9-h pot9-h-tkr"></span>
        {families.map(f => (
          <span key={f} className="pot9-h-models">
            {models.map(m => (
              <span key={m} className={"pot9-h-mod mod-" + m}>{m[0]}</span>
            ))}
          </span>
        ))}
      </div>
      {rows.map((r, i) => (
        <div key={i} className={"pot9-row side-" + r.side.toLowerCase()}>
          <span className="pot9-side">{r.side}</span>
          <span className="pot9-tkr">{r.ticker}</span>
          {families.map(f => (
            <span key={f} className={"pot9-fam fam-" + f}>
              {models.map(m => {
                const o = r.by[`${f}_${m}`];
                return (
                  <span
                    key={m}
                    className={"pot9-cell mod-" + m + (o ? " on" : " off")}
                    title={o ? `${f}-${m}: ${fmtDollars(o.dollars_intent)}` : `${f}-${m}: —`}
                  >
                    {o ? fmtDollars(o.dollars_intent) : "—"}
                  </span>
                );
              })}
            </span>
          ))}
        </div>
      ))}
    </div>
  );
}

function Leg({ cls, label, v }) {
  return (
    <div className={"leg " + cls}>
      <span className="leg-sw"></span>
      <span className="leg-l">{label}</span>
      {v != null && <span className="leg-v">${(v/1000).toFixed(1)}k</span>}
    </div>
  );
}

// Per-call prediction vs reality row.
function PredVsReal({ obs, golden }) {
  const pred = obs.p_up;
  const ideal = golden?.ideal_p_up;
  const actualUp = golden?.actual_direction === "up";
  const ret = golden?.actual_return ?? 0;
  const mag = golden?.magnitude_bucket;
  const correct = ideal != null && Math.sign(pred - 0.5) === Math.sign(ideal - 0.5);
  const flat = !golden || golden.actual_direction === "flat";
  return (
    <li className={"pvr " + (flat ? "flat" : correct ? "good" : "bad")}>
      <div className="pvr-head">
        <span className="pvr-tkr">{obs.ticker}</span>
        <span className={"pvr-act act-" + obs.action.toLowerCase()}>{obs.action}</span>
        <span className="pvr-conf">{obs.confidence}</span>
        {!flat && (
          <span className="pvr-vrd">{correct ? "✓" : "✗"}</span>
        )}
        {!flat && (
          <span className={"pvr-ret " + (ret > 0 ? "up" : "dn")}>{(ret * 100).toFixed(2)}%</span>
        )}
        {mag && <span className={"pvr-mag mag-" + mag}>{mag}</span>}
      </div>
      <div className="pvr-bar-wrap">
        <div className="pvr-bar-rail">
          <div className="pvr-zero" />
          {/* predicted */}
          <div className="pvr-pred" style={{ left: `${pred * 100}%` }} title={`predicted ${pred.toFixed(2)}`}>
            <span className="pvr-pred-tag">PRED {pred.toFixed(2)}</span>
          </div>
          {/* ideal/golden */}
          {ideal != null && (
            <div className="pvr-ideal" style={{ left: `${ideal * 100}%` }} title={`ideal ${ideal.toFixed(2)}`}>
              <span className="pvr-ideal-tag">REAL {ideal.toFixed(2)}</span>
            </div>
          )}
        </div>
      </div>
    </li>
  );
}

function ScoreRow({ label, stats, cls }) {
  return (
    <div className={"score-row " + cls}>
      <div className="sr-label">{label}</div>
      <div className="sr-cells">
        <Stat l="MCC" v={stats.mcc.toFixed(3)} highlight={stats.mcc} />
        <Stat l="acc" v={(stats.acc * 100).toFixed(1) + "%"} />
        <Stat l="TP" v={stats.TP} />
        <Stat l="FP" v={stats.FP} />
        <Stat l="TN" v={stats.TN} />
        <Stat l="FN" v={stats.FN} />
        <Stat l="n" v={stats.n} />
      </div>
    </div>
  );
}
function Stat({ l, v, highlight }) {
  let cls = "stat";
  if (typeof highlight === "number") cls += highlight > 0.05 ? " hi-good" : highlight < -0.05 ? " hi-bad" : "";
  return (<div className={cls}><div className="stat-l">{l}</div><div className="stat-v">{v}</div></div>);
}

function DateLede({ date, day, totalCells, allDeltas }) {
  const top = day.watchlist_ecology?.[0];
  const recent = allDeltas.find(b => b.sample_reasoning);
  return (
    <div className="lede">
      <div className="lede-head">DAY SPOTLIGHT · {date}</div>
      {top && (
        <div className="lede-call">
          <span className="lede-rank">#1 ECOLOGY PICK</span>
          <span className="lede-tkr">{top.ticker}</span>
          <span className={"lede-act act-" + top.action.toLowerCase()}>{top.action}</span>
          <span className="lede-p">p_up = <strong>{top.p_up.toFixed(2)}</strong></span>
        </div>
      )}
      <div className="lede-stats">
        <span><strong>{day.n_news}</strong> news</span>
        <span><strong>{day.n_activations}</strong> activations</span>
        <span><strong>{allDeltas.length}</strong> beliefs moved</span>
      </div>
      {recent && (
        <div className="lede-belief">
          <span className="lede-belief-tag">strongest belief shift</span>
          <span className="lede-belief-text">"{recent.sample_reasoning}"</span>
        </div>
      )}
    </div>
  );
}

// Floating tooltip — pinned near the mouse.
function Tooltip({ hover, viewportW, viewportH }) {
  if (!hover) return null;
  const W = 320;
  let left = hover._cx + 16;
  if (left + W > viewportW - 8) left = hover._cx - W - 16;
  if (left < 8) left = 8;
  let top = hover._cy + 16;
  const Hest = hover.kind === "cell" ? (hover.reason ? 200 : 130) : hover.kind === "macro" ? 220 : hover.kind === "holding" ? 160 : 240;
  if (top + Hest > viewportH - 8) top = hover._cy - Hest - 12;
  if (top < 8) top = 8;
  return (
    <div className="tooltip" style={{ left, top, width: W }}>
      {hover.kind === "cell" ? <CellTip hover={hover} /> :
       hover.kind === "macro" ? <MacroTip hover={hover} /> :
       hover.kind === "holding" ? <HoldingTip hover={hover} /> :
       <OutcomeTip hover={hover} />}
    </div>
  );
}

function HoldingTip({ hover }) {
  const fmt = (n) => n == null ? "—" : "$" + Math.round(n).toLocaleString();
  return (
    <>
      <div className="tip-head">
        <span className={"tip-scope tip-scope-" + hover.portfolio}>{hover.portfolio}</span>
        <span className="tip-tkr">{hover.label}</span>
        {hover.segKind === "pos" && <span className="tip-tpl">{hover.sector}</span>}
      </div>
      <div className="tip-stats">
        <div className="tip-stat">
          <div className="tip-l">{hover.segKind === "cash" ? "cash held" : "market value"}</div>
          <div className="tip-v"><strong>{fmt(hover.value)}</strong></div>
        </div>
        {hover.segKind === "pos" && (
          <>
            <div className="tip-stat">
              <div className="tip-l">cost basis</div>
              <div className="tip-v"><strong>{fmt(hover.cost_basis)}</strong></div>
            </div>
            <div className="tip-stat">
              <div className="tip-l">unrealized P&amp;L</div>
              <div className="tip-v">
                <strong className={hover.unrealized_pnl > 0 ? "up" : hover.unrealized_pnl < 0 ? "dn" : ""}>
                  {hover.unrealized_pnl >= 0 ? "+" : ""}{fmt(hover.unrealized_pnl)}
                </strong>
                <span className={"tip-delta " + (hover.unrealized_pnl > 0 ? "up" : "dn")}>
                  {(hover.unrealized_pnl_pct * 100).toFixed(2)}%
                </span>
              </div>
            </div>
            <div className="tip-stat">
              <div className="tip-l">shares</div>
              <div className="tip-v"><strong>{hover.shares?.toFixed(2)}</strong></div>
            </div>
          </>
        )}
        {hover.segKind === "cash" && hover.invested_frac != null && (
          <div className="tip-stat">
            <div className="tip-l">invested fraction</div>
            <div className="tip-v"><strong>{(hover.invested_frac * 100).toFixed(1)}%</strong></div>
          </div>
        )}
      </div>
    </>
  );
}

function MacroTip({ hover }) {
  const fired = hover.delta != null;
  const dir = fired && hover.delta > 0 ? "up" : fired && hover.delta < 0 ? "dn" : null;
  return (
    <>
      <div className="tip-head">
        <span className={"tip-scope tip-scope-" + hover.scope}>{hover.scope}</span>
        <span className="tip-tpl">{hover.id.replace(/^belief\./, "")}</span>
      </div>
      <div className="tip-statement">{hover.statement}</div>
      <div className="tip-stats">
        <div className="tip-stat">
          <div className="tip-l">{fired ? "p (today)" : "prior p"}</div>
          <div className="tip-v">
            {fired && hover.p_before != null && <span className="tip-before">{hover.p_before.toFixed(2)} →</span>}
            <strong className={dir || ""}>{(fired ? hover.p_after : hover.prior).toFixed(3)}</strong>
            {fired && hover.delta !== 0 && (
              <span className={"tip-delta " + dir}>
                {hover.delta > 0 ? "+" : ""}{hover.delta.toFixed(3)}
              </span>
            )}
          </div>
        </div>
        <div className="tip-stat">
          <div className="tip-l">decay class</div>
          <div className="tip-v"><strong>{hover.decay}</strong></div>
        </div>
        {hover.n_act ? (
          <div className="tip-stat">
            <div className="tip-l">activations</div>
            <div className="tip-v"><strong>{hover.n_act}</strong></div>
          </div>
        ) : null}
      </div>
      {hover.reason ? <div className="tip-reason">"{hover.reason}"</div> :
        fired ? <div className="tip-empty">activated today — no sample reasoning attached.</div> :
        <div className="tip-empty">no news touched this belief today.</div>}
    </>
  );
}

function CellTip({ hover }) {
  const dir = TEMPLATE_DIR(hover.template);
  return (
    <>
      <div className="tip-head">
        <span className={"tip-dir " + (dir === "+" ? "pos" : "neg")}>{dir}</span>
        <span className="tip-tkr">{hover.ticker}</span>
        <span className="tip-x">×</span>
        <span className="tip-tpl">{TEMPLATE_LABEL(hover.template)}</span>
      </div>
      <div className="tip-stats">
        <div className="tip-stat">
          <div className="tip-l">leaf p</div>
          <div className="tip-v">
            {hover.p_before != null && <span className="tip-before">{hover.p_before.toFixed(2)} →</span>}
            <strong className={hover.delta > 0 ? "up" : hover.delta < 0 ? "dn" : ""}>{hover.p.toFixed(3)}</strong>
            {hover.delta !== 0 && (
              <span className={"tip-delta " + (hover.delta > 0 ? "up" : "dn")}>
                {hover.delta > 0 ? "+" : ""}{hover.delta.toFixed(3)}
              </span>
            )}
          </div>
        </div>
        <div className="tip-stat">
          <div className="tip-l">link strength</div>
          <div className="tip-v"><strong>{hover.ls.toFixed(3)}</strong></div>
        </div>
        {hover.n_act ? (
          <div className="tip-stat">
            <div className="tip-l">activations</div>
            <div className="tip-v"><strong>{hover.n_act}</strong></div>
          </div>
        ) : null}
      </div>
      {hover.reason ? <div className="tip-reason">"{hover.reason}"</div> : <div className="tip-empty">no news activated this belief today.</div>}
    </>
  );
}

function OutcomeTip({ hover }) {
  const o = hover.obs;
  const g = hover.golden;
  return (
    <>
      <div className="tip-head">
        <span className="tip-tkr">{hover.ticker}</span>
        <span className="tip-tpl">.next_day_direction</span>
      </div>
      {o ? (
        <>
          <div className="tip-stats">
            <div className="tip-stat">
              <div className="tip-l">action</div>
              <div className="tip-v"><span className={"tip-act act-" + o.action.toLowerCase()}>{o.action}</span></div>
            </div>
            <div className="tip-stat">
              <div className="tip-l">predicted p_up</div>
              <div className="tip-v"><strong>{o.p_up.toFixed(3)}</strong></div>
            </div>
            <div className="tip-stat">
              <div className="tip-l">salience</div>
              <div className="tip-v"><strong>{o.salience.toFixed(2)}</strong></div>
            </div>
          </div>
          {o.causal_chain && o.causal_chain.length > 0 && (
            <div className="tip-chain">
              <div className="tip-chain-head">causal chain</div>
              {o.causal_chain.slice(0, 4).map((c, i) => {
                const p = parseBeliefId(c.parent_id);
                return (
                  <div key={i} className="chain-row">
                    <span className={"chain-arrow " + (c.contribution_log_odds > 0 ? "up" : "dn")}>
                      {c.contribution_log_odds > 0 ? "↑" : "↓"}
                    </span>
                    <span className="chain-tpl">{TEMPLATE_LABEL(p.template)}</span>
                    <span className="chain-num">{c.contribution_log_odds.toFixed(2)} × {c.link_strength.toFixed(2)}</span>
                  </div>
                );
              })}
            </div>
          )}
          {o.conflicting_signals && o.conflicting_signals.length > 0 && (
            <div className="tip-chain conflict">
              <div className="tip-chain-head">conflicting signals</div>
              {o.conflicting_signals.slice(0, 3).map((c, i) => {
                const p = parseBeliefId(c.parent_id);
                return (
                  <div key={i} className="chain-row">
                    <span className={"chain-arrow " + (c.contribution_log_odds > 0 ? "up" : "dn")}>
                      {c.contribution_log_odds > 0 ? "↑" : "↓"}
                    </span>
                    <span className="chain-tpl">{TEMPLATE_LABEL(p.template)}</span>
                    <span className="chain-num">{c.contribution_log_odds.toFixed(2)}</span>
                  </div>
                );
              })}
            </div>
          )}
        </>
      ) : (
        <div className="tip-empty">no outcome update on this day (priors held at 0.5).</div>
      )}
      {g && (
        <div className="tip-real">
          <span className="tip-l">realized</span>
          <span className={"real-dir d-" + g.actual_direction}>{g.actual_direction}</span>
          <span className="tip-v">{(g.actual_return * 100).toFixed(2)}%</span>
          {g.ideal_p_up != null && (
            <span className="tip-ideal">ideal p_up = <strong>{g.ideal_p_up.toFixed(2)}</strong></span>
          )}
          {g.magnitude_bucket && <span className={"tip-mag mag-" + g.magnitude_bucket}>{g.magnitude_bucket}</span>}
        </div>
      )}
    </>
  );
}

window.Spotlight = Spotlight;
window.Tooltip = Tooltip;
