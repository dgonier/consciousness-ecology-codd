// Scrubber + tweaks host
function Scrubber({ dates, days, dayIdx, setDayIdx, playing, setPlaying, speed, setSpeed }) {
  const [drag, setDrag] = React.useState(false);
  const trackRef = React.useRef(null);

  React.useEffect(() => {
    const fn = (e) => {
      const d = e.detail;
      if (d === 0) setDayIdx(0);
      else if (d === 99999) setDayIdx(dates.length - 1);
      else setDayIdx(i => Math.max(0, Math.min(dates.length - 1, i + d)));
    };
    window.addEventListener("seek", fn);
    return () => window.removeEventListener("seek", fn);
  }, [dates.length, setDayIdx]);

  const onPointer = (e) => {
    const rect = trackRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const idx = Math.max(0, Math.min(dates.length - 1, Math.round(x * (dates.length - 1))));
    setDayIdx(idx);
  };

  // Activity sparkline: number of activations per day, normalized
  const max = Math.max(...days.map(d => d.n_activations || 0), 1);

  return (
    <footer className="scrub" >
      <div className="scrub-row">
        <button
          className={"play-big " + (playing ? "on" : "")}
          onClick={() => setPlaying(p => !p)}
          aria-label={playing ? "pause" : "play"}
        >
          <span className="play-glyph">{playing ? "❚❚" : "▶"}</span>
          <span className="play-label">{playing ? "PAUSE" : "PLAY"}</span>
          <span className="play-speed">{speed}×</span>
        </button>
        <div className="scrub-end l">{dates[0]}</div>
        <div
          className="track"
          ref={trackRef}
          onMouseDown={(e) => { setDrag(true); onPointer(e); }}
          onMouseMove={(e) => { if (drag) onPointer(e); }}
          onMouseUp={() => setDrag(false)}
          onMouseLeave={() => setDrag(false)}
        >
          {/* Activity bars */}
          <div className="activity">
            {days.map((d, i) => {
              const h = ((d.n_activations || 0) / max) * 100;
              return (
                <div
                  key={i}
                  className={"act-bar " + (i === dayIdx ? "cur" : i < dayIdx ? "past" : "future")}
                  style={{ height: `${Math.max(2, h)}%` }}
                  title={`${dates[i]} · ${d.n_activations} activations`}
                  onClick={() => setDayIdx(i)}
                />
              );
            })}
          </div>
          {/* Position marker */}
          <div
            className="cursor"
            style={{ left: `${(dayIdx / Math.max(1, dates.length - 1)) * 100}%` }}
          />
          {/* Month tick labels */}
          <div className="months">
            {dates.map((d, i) => {
              if (i === 0 || d.slice(8) === "01" || (d.slice(5, 7) !== dates[i - 1].slice(5, 7))) {
                return (
                  <div
                    key={i}
                    className="month-tick"
                    style={{ left: `${(i / Math.max(1, dates.length - 1)) * 100}%` }}
                  >{d.slice(0, 7)}</div>
                );
              }
              return null;
            })}
          </div>
        </div>
        <div className="scrub-end r">{dates[dates.length - 1]}</div>
      </div>
    </footer>
  );
}

// ─── Tweaks host (uses tweaks-panel.jsx) ────────────────────────────────
const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "showHalos": true,
  "showLinkEdges": true,
  "theme": "dark",
  "speed": 1,
  "mode": "daily"
}/*EDITMODE-END*/;

function TweaksHost({ showHalos, setShowHalos, showLinkEdges, setShowLinkEdges, theme, setTheme, speed, setSpeed, mode, setMode, tickerFilter, setTickerFilter }) {
  const [open, setOpen] = React.useState(false);

  React.useEffect(() => {
    const fn = (ev) => {
      const t = ev.data?.type;
      if (t === "__activate_edit_mode") setOpen(true);
      else if (t === "__deactivate_edit_mode") setOpen(false);
    };
    window.addEventListener("message", fn);
    window.parent.postMessage({ type: "__edit_mode_available" }, "*");
    return () => window.removeEventListener("message", fn);
  }, []);

  const dismiss = () => {
    setOpen(false);
    window.parent.postMessage({ type: "__edit_mode_dismissed" }, "*");
  };

  if (!open) return null;

  const toggleTicker = (t) => {
    setTickerFilter(prev => {
      const next = new Set(prev);
      if (next.has(t)) next.delete(t);
      else next.add(t);
      if (next.size === 0) return new Set([t]);
      return next;
    });
  };
  const allOn = () => setTickerFilter(new Set(TICKERS));
  const sectorOnly = (s) => setTickerFilter(new Set(TICKERS.filter(t => TICKER_SECTORS[t] === s)));

  return (
    <div className="tweaks-panel">
      <div className="tw-head">
        <span className="tw-title">Tweaks</span>
        <button className="tw-close" onClick={dismiss}>×</button>
      </div>
      <div className="tw-body">
        <div className="tw-sec">
          <div className="tw-label">display</div>
          <label className="tw-toggle">
            <input type="checkbox" checked={showHalos} onChange={e => setShowHalos(e.target.checked)} />
            <span>ground-truth halos</span>
          </label>
          <label className="tw-toggle">
            <input type="checkbox" checked={showLinkEdges} onChange={e => setShowLinkEdges(e.target.checked)} />
            <span>cumulative link edges</span>
          </label>
        </div>
        <div className="tw-sec">
          <div className="tw-label">theme</div>
          <div className="tw-radio">
            {["dark", "light", "print"].map(k => (
              <button key={k} className={"tw-rb " + (theme === k ? "on" : "")} onClick={() => setTheme(k)}>{k}</button>
            ))}
          </div>
        </div>
        <div className="tw-sec">
          <div className="tw-label">speed</div>
          <div className="tw-radio">
            {[0.25, 0.5, 1, 2, 5].map(s => (
              <button key={s} className={"tw-rb " + (speed === s ? "on" : "")} onClick={() => setSpeed(s)}>{s}x</button>
            ))}
          </div>
        </div>
        <div className="tw-sec">
          <div className="tw-label">mode</div>
          <div className="tw-radio">
            {[["daily", "daily"], ["drift", "drift"], ["side-by-side", "vs bare"]].map(([k, l]) => (
              <button key={k} className={"tw-rb " + (mode === k ? "on" : "")} onClick={() => setMode(k)}>{l}</button>
            ))}
          </div>
        </div>
        <div className="tw-sec">
          <div className="tw-label tw-label-row">
            <span>ticker filter</span>
            <button className="tw-mini" onClick={allOn}>all</button>
          </div>
          <div className="tw-radio sector-row">
            {SECTOR_ORDER.map(s => (
              <button key={s} className="tw-rb" onClick={() => sectorOnly(s)}>{SECTOR_LABEL[s]}</button>
            ))}
          </div>
          <div className="ticker-grid">
            {TICKERS.map(t => (
              <button
                key={t}
                className={"ti " + (tickerFilter.has(t) ? "on" : "off")}
                onClick={() => toggleTicker(t)}
              >{t}</button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

window.Scrubber = Scrubber;
window.TweaksHost = TweaksHost;
