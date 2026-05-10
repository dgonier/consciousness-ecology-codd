// Below-grid sections (left column): "Beliefs Not Fired" cards + "Today's signals" feed.
// Both consume current day's data and update with the scrubber.

function BelowGrid({ day, beliefByCell, links, setHover, tickerFilter }) {
  const TPL = window.TEMPLATES;
  const TKR = window.TICKERS.filter(t => tickerFilter.has(t));
  const totalCells = TPL.length * TKR.length;

  // ─── Not-fired cards: one per cell where activated == false ────────
  const notFired = [];
  for (const tpl of TPL) {
    for (const tkr of TKR) {
      const key = `${tpl}|${tkr}`;
      if (!beliefByCell[key]) {
        notFired.push({
          _template: tpl, _ticker: tkr,
          ls: links[key] ?? 0,
        });
      }
    }
  }
  // Sort by link strength desc — "loudest dormant signal" first
  notFired.sort((a, b) => b.ls - a.ls);

  // ─── Group not-fired by template (one card per template, listing tickers) ──
  const byTemplate = {};
  notFired.forEach(b => {
    (byTemplate[b._template] ||= []).push(b);
  });

  const [tplFilter, setTplFilter] = React.useState("all");
  const [groupBy, setGroupBy] = React.useState("template"); // template | ticker | flat
  const [tickerFilt, setTickerFilt] = React.useState("all");

  let filteredNotFired = notFired;
  if (tplFilter !== "all") filteredNotFired = filteredNotFired.filter(b => b._template === tplFilter);
  if (tickerFilt !== "all") filteredNotFired = filteredNotFired.filter(b => b._ticker === tickerFilt);

  const onCardEnter = (e, b) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setHover({
      kind: "cell", template: b._template, ticker: b._ticker,
      p: 0.5, delta: 0, ls: b.ls,
      reason: null, n_act: 0, p_before: 0.5,
      _cx: rect.left + rect.width / 2, _cy: rect.top + rect.height / 2,
    });
  };
  const onCardLeave = () => setHover(null);

  // ─── Today's signals: activations → reasoning text + observations + sympathy ──
  const allDeltas = (day.belief_deltas || []).map(b => {
    const p = parseBeliefId(b.belief_id);
    return { ...b, _scope: p.scope, _template: p.template, _ticker: p.ticker };
  }).filter(b => b._scope === "company" && b._ticker)
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

  const obsList = (day.observations || []).slice().sort((a, b) => b.salience - a.salience);

  const tickersWithSignals = new Set(allDeltas.map(b => b._ticker));

  return (
    <div className="below">
      {/* ─── BELIEFS NOT FIRED ─────────────────────────────────────── */}
      <section className="below-sec nf-sec">
        <div className="below-head">
          <div className="below-title">
            <span className="below-eyebrow">manifest · {day.date}</span>
            <h2 className="below-h">Beliefs Not Fired</h2>
            <span className="below-sub">
              {filteredNotFired.length} of {totalCells} cells dormant today —
              priors held at 0.5, no news triggered the classifier.
            </span>
          </div>
          <div className="below-toolbar">
            <div className="nf-seg">
              <button className={"nf-seg-btn " + (groupBy === "template" ? "on" : "")} onClick={() => setGroupBy("template")}>by template</button>
              <button className={"nf-seg-btn " + (groupBy === "ticker" ? "on" : "")} onClick={() => setGroupBy("ticker")}>by ticker</button>
              <button className={"nf-seg-btn " + (groupBy === "flat" ? "on" : "")} onClick={() => setGroupBy("flat")}>flat</button>
            </div>
          </div>
        </div>

        {groupBy === "template" && (
          <div className="nf-grid">
            {TPL.map(tpl => {
              const items = (byTemplate[tpl] || []).filter(b => tickerFilt === "all" || b._ticker === tickerFilt);
              if (items.length === 0) return null;
              const dir = TEMPLATE_DIR(tpl);
              const fired = TKR.filter(t => beliefByCell[`${tpl}|${t}`]);
              return (
                <div key={tpl} className={"nf-card dir-" + (dir === "+" ? "pos" : "neg")}>
                  <div className="nf-card-head">
                    <span className={"nf-dir " + (dir === "+" ? "pos" : "neg")}>{dir}</span>
                    <span className="nf-tpl">{TEMPLATE_LABEL(tpl)}</span>
                    <span className="nf-count-pill" title="dormant tickers / total tickers">
                      {items.length}/{TKR.length}
                    </span>
                  </div>
                  {fired.length > 0 && (
                    <div className="nf-fired-row">
                      <span className="nf-fired-l">fired today:</span>
                      <span className="nf-fired-tkrs">
                        {fired.map(t => <span key={t} className="nf-fired-tkr">{t}</span>)}
                      </span>
                    </div>
                  )}
                  <div className="nf-tkrs">
                    {items.map(b => (
                      <button
                        key={b._ticker}
                        className="nf-chip"
                        onMouseEnter={(e) => onCardEnter(e, b)}
                        onMouseLeave={onCardLeave}
                      >
                        <span className="nf-chip-tkr">{b._ticker}</span>
                        {b.ls > 0.05 && (
                          <span className="nf-chip-ls" title={`learned link strength = ${b.ls.toFixed(2)}`}>
                            <span className="nf-chip-ls-bar" style={{ width: `${Math.min(100, b.ls * 100)}%` }} />
                          </span>
                        )}
                      </button>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {groupBy === "ticker" && (
          <div className="nf-grid">
            {TKR.map(tkr => {
              const items = notFired.filter(b => b._ticker === tkr).filter(b => tplFilter === "all" || b._template === tplFilter);
              if (items.length === 0) return null;
              const sector = TICKER_SECTORS[tkr];
              return (
                <div key={tkr} className={"nf-card sector-" + sector}>
                  <div className="nf-card-head">
                    <span className="nf-tkr-big">{tkr}</span>
                    <span className="nf-sector">{SECTOR_LABEL[sector]}</span>
                    <span className="nf-count-pill">{items.length}/{TPL.length}</span>
                  </div>
                  {tickersWithSignals.has(tkr) && (
                    <div className="nf-fired-row">
                      <span className="nf-fired-l">{allDeltas.filter(d => d._ticker === tkr).length} fired today</span>
                    </div>
                  )}
                  <div className="nf-tkrs">
                    {items.map(b => {
                      const dir = TEMPLATE_DIR(b._template);
                      return (
                        <button
                          key={b._template}
                          className={"nf-chip nf-chip-tpl dir-" + (dir === "+" ? "pos" : "neg")}
                          onMouseEnter={(e) => onCardEnter(e, b)}
                          onMouseLeave={onCardLeave}
                          title={`${dir} ${TEMPLATE_LABEL(b._template)} · link strength ${b.ls.toFixed(2)}`}
                        >
                          <span className="nf-chip-dir">{dir}</span>
                          <span className="nf-chip-tpl-l">{TEMPLATE_LABEL(b._template)}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {groupBy === "flat" && (
          <ul className="nf-flat">
            {filteredNotFired.slice(0, 200).map((b, i) => {
              const dir = TEMPLATE_DIR(b._template);
              return (
                <li key={i} className={"nf-row dir-" + (dir === "+" ? "pos" : "neg")}
                  onMouseEnter={(e) => onCardEnter(e, b)} onMouseLeave={onCardLeave}>
                  <span className={"nf-dir " + (dir === "+" ? "pos" : "neg")}>{dir}</span>
                  <span className="nf-tkr-row">{b._ticker}</span>
                  <span className="nf-tpl-row">{TEMPLATE_LABEL(b._template)}</span>
                  <span className="nf-ls-row">link {b.ls.toFixed(2)}</span>
                </li>
              );
            })}
            {filteredNotFired.length > 200 && (
              <li className="nf-trunc">+ {filteredNotFired.length - 200} more — switch to grouped view to see all</li>
            )}
          </ul>
        )}
      </section>

      {/* ─── DAILY SIGNALS / EVIDENCE FEED ──────────────────────────── */}
      <section className="below-sec sig-sec">
        <div className="below-head">
          <div className="below-title">
            <span className="below-eyebrow">evidence · {day.date}</span>
            <h2 className="below-h">Signals &amp; Sources</h2>
            <span className="below-sub">
              {day.n_news} news items · {day.n_activations} classifier hits · {day.n_observations} ranked outcomes ·
              ecology compute {day.elapsed?.ecology_s?.toFixed(1) || "—"}s
            </span>
          </div>
        </div>

        <div className="sig-grid">
          {/* Activation evidence column */}
          <div className="sig-col">
            <div className="sig-col-head">
              <span className="sig-col-title">classifier evidence</span>
              <span className="sig-col-sub">{allDeltas.length} activations · sample reasoning</span>
            </div>
            <div className="sig-list">
              {allDeltas.length === 0 && <div className="sig-empty">no activations today.</div>}
              {allDeltas.map((b, i) => (
                <article key={i} className="sig-item">
                  <div className="sig-item-head">
                    <span className="sig-tkr">{b._ticker}</span>
                    <span className={"sig-dir " + (b.delta > 0 ? "up" : "dn")}>
                      {b.delta > 0 ? "+" : ""}{(b.delta * 100).toFixed(0)}%
                    </span>
                    <span className="sig-tpl">{TEMPLATE_LABEL(b._template)}</span>
                    <span className="sig-meta">{b.n_activations || 1}× · p={b.p_after.toFixed(2)}</span>
                  </div>
                  {b.sample_reasoning && (
                    <blockquote className="sig-quote">"{b.sample_reasoning}"</blockquote>
                  )}
                  <div className="sig-foot">
                    <span className="sig-species">{b.species || "event_classifier"}</span>
                    {b.species === "cross_correlation.v0" && <span className="sig-tag">sympathy edge</span>}
                  </div>
                </article>
              ))}
            </div>
          </div>

          {/* Outcome reasoning column */}
          <div className="sig-col">
            <div className="sig-col-head">
              <span className="sig-col-title">outcome reasoning</span>
              <span className="sig-col-sub">{obsList.length} ranked calls</span>
            </div>
            <div className="sig-list">
              {obsList.length === 0 && <div className="sig-empty">no outcomes today.</div>}
              {obsList.map((o, i) => (
                <article key={i} className={"sig-item obs " + (o.action === "BUY" ? "buy" : o.action === "SELL" ? "sell" : "watch")}>
                  <div className="sig-item-head">
                    <span className="sig-tkr">{o.ticker}</span>
                    <span className={"sig-act act-" + o.action.toLowerCase()}>{o.action}</span>
                    <span className="sig-meta">p_up={o.p_up.toFixed(2)} · sal={o.salience.toFixed(2)} · {o.confidence}</span>
                  </div>
                  {o.reasoning_summary && (
                    <blockquote className="sig-quote">{o.reasoning_summary}</blockquote>
                  )}
                  {o.causal_chain && o.causal_chain.length > 0 && (
                    <div className="sig-chain">
                      {o.causal_chain.slice(0, 4).map((c, j) => {
                        const p = parseBeliefId(c.parent_id);
                        return (
                          <div key={j} className={"sig-chain-row " + (c.contribution_log_odds > 0 ? "up" : "dn")}>
                            <span className="sig-chain-arrow">{c.contribution_log_odds > 0 ? "↑" : "↓"}</span>
                            <span className="sig-chain-tpl">{TEMPLATE_LABEL(p.template)}</span>
                            <span className="sig-chain-num">{c.contribution_log_odds.toFixed(2)}</span>
                          </div>
                        );
                      })}
                      {o.causal_chain.length > 4 && (
                        <div className="sig-chain-more">+{o.causal_chain.length - 4} more parents</div>
                      )}
                    </div>
                  )}
                </article>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

window.BelowGrid = BelowGrid;
