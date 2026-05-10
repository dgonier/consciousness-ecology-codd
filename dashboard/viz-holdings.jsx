// Holdings bar chart: horizontal stacked bars showing cash + positions
// for ecology and bare portfolios on the current day. Scale fixed 0-$150K.

const HOLDINGS_MAX = 150000;

function HoldingsBar({ day, setHover }) {
  const eco = day.portfolio?.ecology;
  const bare = day.portfolio?.bare;

  const fmt = (n) => n == null ? "—" : "$" + Math.round(n).toLocaleString();
  const pct = (n) => n == null ? "" : (n * 100).toFixed(1) + "%";

  const renderBar = (label, p, cls) => {
    if (!p) return null;
    const cash = p.cash || 0;
    const positions = (p.open_positions || []).slice().sort((a, b) => b.mv - a.mv);
    const totalMV = positions.reduce((s, p) => s + p.mv, 0);
    const equity = p.equity || (cash + totalMV);

    const onSegEnter = (e, kind, label, value, extras) => {
      const rect = e.currentTarget.getBoundingClientRect();
      setHover({
        kind: "holding", segKind: kind, label, value,
        portfolio: cls,
        ...extras,
        _cx: rect.left + rect.width / 2,
        _cy: rect.top,
      });
    };
    const onSegLeave = () => setHover(null);

    return (
      <div className="hold-row">
        <div className="hold-row-l">
          <div className="hold-row-name">{label}</div>
          <div className="hold-row-eq">{fmt(equity)}</div>
          <div className="hold-row-meta">
            {p.n_open_positions} pos · {pct(p.invested_frac)} invested
          </div>
        </div>
        <div className="hold-row-bar">
          {/* Cash */}
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
          {/* Positions */}
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
          {/* Empty space to $150K */}
          <div className="hold-seg-empty" style={{ width: `${(1 - equity / HOLDINGS_MAX) * 100}%` }} />
        </div>
      </div>
    );
  };

  // Tick marks at 0 / 25 / 50 / 75 / 100 / 125 / 150K
  const ticks = [0, 25000, 50000, 75000, 100000, 125000, 150000];

  // Today's orders preview
  const ecoOrders = eco?.orders_today || [];
  const bareOrders = bare?.orders_today || [];

  return (
    <div className="holdings">
      <div className="holdings-head">
        <div className="holdings-title">
          <span className="holdings-eyebrow">portfolio · {day.date}</span>
          <span className="holdings-h">Holdings by day</span>
        </div>
        <div className="holdings-orders">
          {ecoOrders.length > 0 && (
            <div className="ord-group">
              <span className="ord-l eco">ECOLOGY trades:</span>
              {ecoOrders.map((o, i) => (
                <span key={"e"+i} className={"ord " + o.side.toLowerCase()}>
                  <span className="ord-side">{o.side === "BUY" ? "+" : "−"}</span>
                  <span className="ord-tkr">{o.ticker}</span>
                  <span className="ord-amt">${Math.round(o.dollars_intent).toLocaleString()}</span>
                </span>
              ))}
            </div>
          )}
          {bareOrders.length > 0 && (
            <div className="ord-group">
              <span className="ord-l bare">BARE trades:</span>
              {bareOrders.map((o, i) => (
                <span key={"b"+i} className={"ord " + o.side.toLowerCase()}>
                  <span className="ord-side">{o.side === "BUY" ? "+" : "−"}</span>
                  <span className="ord-tkr">{o.ticker}</span>
                  <span className="ord-amt">${Math.round(o.dollars_intent).toLocaleString()}</span>
                </span>
              ))}
            </div>
          )}
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

      {renderBar("ECOLOGY", eco, "eco")}
      {renderBar("BARE", bare, "bare")}
    </div>
  );
}

window.HoldingsBar = HoldingsBar;
