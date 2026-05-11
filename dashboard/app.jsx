// Belief Timeline — main app, data loader, scrubber, transport.
// Schema 2.0: MANIFEST.belief_catalog + per-day {belief_deltas, observations, golden, portfolio}.
const { useState, useEffect, useRef, useMemo, useCallback } = React;

const DATES = [
  "2026-02-03","2026-02-04","2026-02-05","2026-02-06","2026-02-09","2026-02-10","2026-02-11","2026-02-12","2026-02-13","2026-02-17",
  "2026-02-18","2026-02-19","2026-02-20","2026-02-23","2026-02-24","2026-02-25","2026-02-26","2026-02-27","2026-03-02","2026-03-03",
  "2026-03-04","2026-03-05","2026-03-06","2026-03-09","2026-03-10","2026-03-11","2026-03-12","2026-03-13","2026-03-16","2026-03-17",
  "2026-03-18","2026-03-19","2026-03-20","2026-03-23","2026-03-24","2026-03-25","2026-03-26","2026-03-27","2026-03-30","2026-03-31",
  "2026-04-01","2026-04-02","2026-04-06","2026-04-07","2026-04-08","2026-04-09","2026-04-10","2026-04-13","2026-04-14","2026-04-15",
  "2026-04-16","2026-04-17","2026-04-20","2026-04-21","2026-04-22","2026-04-23","2026-04-24","2026-04-27","2026-04-28","2026-04-29",
  "2026-04-30","2026-05-01","2026-05-04","2026-05-05","2026-05-06","2026-05-07"
];

const TICKER_SECTORS = {
  AAPL: "tech",   AMZN: "tech",   AVGO: "tech",   CSCO: "tech",   GOOG: "tech",
  MSFT: "tech",
  ABBV: "health", JNJ: "health",  MRK: "health",  UNH: "health",
  CVX:  "energy",
  JPM:  "fin",    MA:   "fin",    V:    "fin",
  HD:   "consum", KO:   "consum", MCD:  "consum", PEP:  "consum", PG: "consum", WMT: "consum",
};
const SECTOR_ORDER = ["tech", "fin", "health", "consum", "energy"];
const SECTOR_LABEL = { tech: "TECH", fin: "FIN", health: "HEALTH", consum: "CONSUM", energy: "ENERGY" };

const TICKERS = Object.keys(TICKER_SECTORS).sort((a, b) => {
  const sa = SECTOR_ORDER.indexOf(TICKER_SECTORS[a]);
  const sb = SECTOR_ORDER.indexOf(TICKER_SECTORS[b]);
  if (sa !== sb) return sa - sb;
  return a < b ? -1 : 1;
});

const TEMPLATES = [
  "earnings_beat",
  "earnings_miss",
  "guidance_raised",
  "guidance_cut",
  "major_product_launch_positive_reception",
  "major_product_launch_negative_reception",
  "analyst_upgrade",
  "analyst_downgrade",
  "regulatory_clearance_received",
  "regulatory_action_announced",
  "technical_resistance_broken",
  "technical_support_broken",
  "buyback_program_announced",
  "dividend_increase_announced",
  "ceo_departure_unplanned",
  "activist_investor_takes_stake",
  "short_interest_high_and_rising",
  "material_lawsuit_filed",
  "acting_as_acquirer_in_announced_ma",
  "target_in_announced_ma",
  "partnership_announced",
  "supply_chain_disruption",
];

const TEMPLATE_LABEL = (t) => t.replace(/_/g, " ");
const TEMPLATE_DIR = (t) => {
  if (/(beat|raised|positive|upgrade|clearance|resistance_broken|buyback|dividend|partnership|target_in)/.test(t)) return "+";
  return "-";
};

// belief_id parser: returns {scope, template, ticker?} for company beliefs and macro/sector
function parseBeliefId(id) {
  // belief.company.<template>__ticker_<TKR>
  let m = id.match(/^belief\.company\.(.+)__ticker_([A-Z0-9.]+)$/);
  if (m) return { scope: "company", template: m[1], ticker: m[2] };
  m = id.match(/^belief\.(macro|sector|market)\.(.+)$/);
  if (m) return { scope: m[1], template: m[2], ticker: null };
  m = id.match(/^belief\.company\.(.+)$/);
  if (m) return { scope: "company", template: m[1], ticker: null }; // stub
  return { scope: "unknown", template: id, ticker: null };
}

// ─── Data loader ─────────────────────────────────────────────────────────
async function loadAll() {
  const manifest = await (await fetch("MANIFEST.json")).json();
  let runMeta = null;
  try { runMeta = await (await fetch("run_meta.json")).json(); } catch (e) {}
  const days = await Promise.all(
    DATES.map(async (d) => {
      const r = await fetch(`${d}.json`);
      return r.json();
    })
  );
  return { manifest, days, runMeta };
}

// ─── Build per-day link-strength snapshots from observations.causal_chain ─
// Each observation's causal_chain entry tells us the persistent link strength
// for parent_belief → outcome at that moment in time. Persist across days.
function buildLinkSnapshots(days) {
  const seedFor = (template) => {
    const seeds = {
      earnings_beat: 0.55, earnings_miss: 0.75, guidance_cut: 0.75, guidance_raised: 0.55,
      major_product_launch_positive_reception: 0.5, major_product_launch_negative_reception: 0.5,
      regulatory_action_announced: 0, technical_support_broken: 0, regulatory_clearance_received: 0.5,
      technical_resistance_broken: 0.5, buyback_program_announced: 0.4, dividend_increase_announced: 0.35,
      ceo_departure_unplanned: 0.65, activist_investor_takes_stake: 0.45,
      short_interest_high_and_rising: 0.45, material_lawsuit_filed: 0.65,
      acting_as_acquirer_in_announced_ma: 0.4, target_in_announced_ma: 0.75,
      analyst_upgrade: 0.5, analyst_downgrade: 0.5, partnership_announced: 0.4,
      supply_chain_disruption: 0.45,
    };
    return seeds[template] ?? 0.5;
  };
  const live = {};
  TEMPLATES.forEach(tpl => TICKERS.forEach(tkr => {
    live[`${tpl}|${tkr}`] = seedFor(tpl);
  }));
  const snapshots = [];
  days.forEach(day => {
    (day.observations || []).forEach(obs => {
      (obs.causal_chain || []).forEach(c => {
        const p = parseBeliefId(c.parent_id);
        if (p.scope === "company" && p.ticker && TEMPLATES.includes(p.template)) {
          live[`${p.template}|${p.ticker}`] = c.link_strength;
        }
      });
    });
    snapshots.push({ ...live });
  });
  return snapshots;
}

// ─── Scoring against goldens (with ideal_p_up) ───────────────────────────
function topKAccuracy(days, upTo, K, srcKey) {
  let TP = 0, FP = 0, TN = 0, FN = 0;
  for (let i = 0; i <= upTo; i++) {
    const d = days[i];
    if (!d.golden) continue;
    const real = {};
    d.golden.forEach(g => { real[g.ticker] = g; });
    const list = (d[srcKey] || []).slice(0, K);
    list.forEach(item => {
      const r = real[item.ticker];
      if (!r || r.actual_direction === "flat") return;
      if (item.action === "WATCH") return;
      const predUp = item.action === "BUY";
      const actualUp = r.actual_direction === "up";
      if (predUp && actualUp) TP++;
      else if (predUp && !actualUp) FP++;
      else if (!predUp && actualUp) FN++;
      else TN++;
    });
  }
  const n = TP + FP + TN + FN;
  const denom = Math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
  const mcc = denom > 0 ? (TP*TN - FP*FN) / denom : 0;
  const acc = n > 0 ? (TP + TN) / n : 0;
  return { mcc, acc, TP, FP, TN, FN, n };
}

// ─── Equity time-series for the curve panel ──────────────────────────────
// 9 pipelines (3 apex models × {bare, ecology, oracle}) + buyhold.
// Back-compat keys `ecology` / `bare` alias the qwen variants so existing
// surfaces (Header, PortfolioPanel) still work.
const PIPELINE_KEYS = [
  "bare_qwen", "bare_sonnet", "bare_opus",
  "ecology_qwen", "ecology_sonnet", "ecology_opus",
  "oracle_qwen", "oracle_sonnet", "oracle_opus",
];
const PIPELINE_FAMILY = {
  bare_qwen: "bare", bare_sonnet: "bare", bare_opus: "bare",
  ecology_qwen: "ecology", ecology_sonnet: "ecology", ecology_opus: "ecology",
  oracle_qwen: "oracle", oracle_sonnet: "oracle", oracle_opus: "oracle",
};
const PIPELINE_MODEL = {
  bare_qwen: "qwen", ecology_qwen: "qwen", oracle_qwen: "qwen",
  bare_sonnet: "sonnet", ecology_sonnet: "sonnet", oracle_sonnet: "sonnet",
  bare_opus: "opus", ecology_opus: "opus", oracle_opus: "opus",
};
const PIPELINE_LABEL = {
  bare_qwen: "BARE-QWEN", bare_sonnet: "BARE-SONNET", bare_opus: "BARE-OPUS",
  ecology_qwen: "ECO-QWEN", ecology_sonnet: "ECO-SONNET", ecology_opus: "ECO-OPUS",
  oracle_qwen: "ORACLE-QWEN", oracle_sonnet: "ORACLE-SONNET", oracle_opus: "ORACLE-OPUS",
};

function buildEquitySeries(days) {
  const series = { buyhold: [] };
  PIPELINE_KEYS.forEach(k => { series[k] = []; });
  days.forEach(d => {
    const p = d.portfolio || {};
    PIPELINE_KEYS.forEach(k => series[k].push(p[k]?.equity ?? null));
    series.buyhold.push(p.buyhold_equity ?? null);
  });
  // Aliases for back-compat (Header, sparkline, etc).
  series.ecology = series.ecology_qwen;
  series.bare = series.bare_qwen;
  return series;
}

window.PIPELINE_KEYS = PIPELINE_KEYS;
window.PIPELINE_FAMILY = PIPELINE_FAMILY;
window.PIPELINE_MODEL = PIPELINE_MODEL;
window.PIPELINE_LABEL = PIPELINE_LABEL;

// ─── App ─────────────────────────────────────────────────────────────────
function App() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [dayIdx, setDayIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [mode, setMode] = useState("daily");
  const [showHalos, setShowHalos] = useState(true);
  const [showLinkEdges, setShowLinkEdges] = useState(true);
  const [showIdeal, setShowIdeal] = useState(true);
  const [theme, setTheme] = useState("dark");
  const [tickerFilter, setTickerFilter] = useState(new Set(TICKERS));
  const [hover, setHover] = useState(null);
  const [vp, setVp] = useState({ w: window.innerWidth, h: window.innerHeight });

  useEffect(() => {
    const fn = () => setVp({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener("resize", fn);
    return () => window.removeEventListener("resize", fn);
  }, []);

  useEffect(() => {
    loadAll().then(setData).catch(e => setErr(String(e)));
  }, []);

  const linkSnapshots = useMemo(() => data ? buildLinkSnapshots(data.days) : null, [data]);
  const equitySeries = useMemo(() => data ? buildEquitySeries(data.days) : null, [data]);

  useEffect(() => {
    if (!playing || !data) return;
    const stepMs = mode === "drift" ? 200 / speed : 1000 / speed;
    const id = setInterval(() => {
      setDayIdx(i => {
        const next = i + 1;
        if (next >= DATES.length) { setPlaying(false); return i; }
        return next;
      });
    }, stepMs);
    return () => clearInterval(id);
  }, [playing, speed, mode, data]);

  if (err) return <div className="loading err">load error: {err}</div>;
  if (!data) return <div className="loading">loading timeline…</div>;

  const date = DATES[dayIdx];
  const day = data.days[dayIdx];
  const links = linkSnapshots[dayIdx];

  // belief_deltas → cell map. belief_id has no separate ticker field anymore.
  const beliefByCell = {};
  (day.belief_deltas || []).forEach(b => {
    const p = parseBeliefId(b.belief_id);
    if (p.scope !== "company" || !p.ticker || !TEMPLATES.includes(p.template)) return;
    const key = `${p.template}|${p.ticker}`;
    if (!beliefByCell[key] || Math.abs(b.delta) > Math.abs(beliefByCell[key].delta || 0)) {
      beliefByCell[key] = {
        p_after: b.p_after, p_before: b.p_before, delta: b.delta,
        n_act: b.n_activations, reason: b.sample_reasoning, species: b.species,
      };
    }
  });

  // observations: {ticker, action, p_up, salience, confidence, causal_chain[], conflicting_signals[]}
  const obsByTicker = {};
  (day.observations || []).forEach(o => { obsByTicker[o.ticker] = o; });

  // golden: {ticker, actual_direction, actual_return, magnitude_bucket, ideal_p_up, was_mentioned_in_news}
  const goldenByTicker = {};
  (day.golden || []).forEach(g => { goldenByTicker[g.ticker] = g; });

  const ecoStats = topKAccuracy(data.days, dayIdx, 5, "watchlist_ecology");
  const bareStats = topKAccuracy(data.days, dayIdx, 5, "watchlist_bare");

  return (
    <div className={`app theme-${theme}`}>
      <Header
        date={date} dayIdx={dayIdx} nDays={DATES.length}
        playing={playing} setPlaying={setPlaying}
        speed={speed} setSpeed={setSpeed}
        mode={mode} setMode={setMode}
        day={day} equitySeries={equitySeries}
        runMeta={data.runMeta}
      />
      <main className="main">
        <div className="main-l">
          <HoldingsBar day={day} setHover={setHover} />
          <ContextStrip manifest={data.manifest} day={day} setHover={setHover} />
          <Grid
            tickers={TICKERS} templates={TEMPLATES}
            beliefByCell={beliefByCell} links={links} showLinkEdges={showLinkEdges}
            obsByTicker={obsByTicker} goldenByTicker={goldenByTicker}
            showHalos={showHalos} showIdeal={showIdeal}
            tickerFilter={tickerFilter} setHover={setHover} mode={mode}
          />
          <BelowGrid
            day={day} beliefByCell={beliefByCell} links={links}
            setHover={setHover} tickerFilter={tickerFilter}
          />
        </div>
        <Spotlight
          day={day} date={date} dayIdx={dayIdx} mode={mode}
          ecoStats={ecoStats} bareStats={bareStats} links={links}
          totalCells={TEMPLATES.length * TICKERS.length}
          equitySeries={equitySeries}
          obsByTicker={obsByTicker} goldenByTicker={goldenByTicker}
          beliefByCell={beliefByCell} setHover={setHover}
        />
      </main>
      <Tooltip hover={hover} viewportW={vp.w} viewportH={vp.h} />
      <Scrubber
        dates={DATES} days={data.days} equitySeries={equitySeries}
        dayIdx={dayIdx} setDayIdx={setDayIdx}
        playing={playing} setPlaying={setPlaying}
        speed={speed} setSpeed={setSpeed}
      />
      <TweaksHost
        showHalos={showHalos} setShowHalos={setShowHalos}
        showLinkEdges={showLinkEdges} setShowLinkEdges={setShowLinkEdges}
        showIdeal={showIdeal} setShowIdeal={setShowIdeal}
        theme={theme} setTheme={setTheme}
        speed={speed} setSpeed={setSpeed}
        mode={mode} setMode={setMode}
        tickerFilter={tickerFilter} setTickerFilter={setTickerFilter}
      />
    </div>
  );
}

function Header({ date, dayIdx, nDays, playing, setPlaying, speed, setSpeed, mode, setMode, day, equitySeries, runMeta }) {
  // 3x3 grid of equity cells (rows = model, cols = pipeline family) + buyhold.
  const eBH = equitySeries.buyhold[dayIdx];
  const bhRet = eBH != null ? (eBH / 100000 - 1) * 100 : null;
  const families = ["bare", "ecology", "oracle"];
  const models = ["qwen", "sonnet", "opus"];
  return (
    <header className="hdr">
      <div className="hdr-grid">
      <div className="hdr-l">
        <div className="title">
          <span className="title-prefix">Tifin · </span>
          <span className="title-main">belief network timeline</span>
        </div>
        <div className="subtitle">
          firehose 66-day replay · 9-way ablation (3 apex × {`{bare,eco,oracle}`}) · $100k seed
        </div>
        {runMeta && (
          <div className="run-meta" title={runMeta.note}>
            <span className="rm-pill rm-label">{runMeta.label}</span>
            <span className="rm-pill rm-sha">git {runMeta.git_sha}</span>
            <span className="rm-pill rm-flags">{runMeta.feature_flags}</span>
            <span className="rm-pill rm-finals">final · {runMeta.final_equities}</span>
          </div>
        )}
      </div>

      <div className="hdr-c">
        <div className="date-block">
          <div className="date-label">DATE</div>
          <div className="date-val">{date}</div>
          <div className="date-sub">day {String(dayIdx + 1).padStart(2, "0")} / {nDays}</div>
        </div>
        <div className="counters">
          <Counter label="news" v={day.n_news} />
          <Counter label="activations" v={day.n_activations} />
          <Counter label="observations" v={day.n_observations} />
        </div>
      </div>

      <div className="hdr-r">
        <div className="equity-grid">
          <div className="eg-corner" />
          {families.map(f => (
            <div key={"hf-" + f} className={"eg-col-h fam-" + f}>{f.toUpperCase()}</div>
          ))}
          {models.map(m => (
            <React.Fragment key={"row-" + m}>
              <div className="eg-row-h">{m}</div>
              {families.map(f => {
                const k = `${f}_${m}`;
                const v = equitySeries[k]?.[dayIdx];
                const ret = v != null ? (v / 100000 - 1) * 100 : null;
                return <EqCellGrid key={k} v={v} ret={ret} family={f} model={m} />;
              })}
            </React.Fragment>
          ))}
          {/* CONTROL row: passive equal-weight buyhold of all 20 tickers.
              Sits under BARE since it's the no-trade baseline; ECOLOGY and
              ORACLE columns stay empty in this row. */}
          <div className="eg-row-h">control</div>
          <EqCellGrid v={eBH} ret={bhRet} family="bh" model="buyhold" />
          <div className="eg-cell empty" />
          <div className="eg-cell empty" />
        </div>
        <div className="hdr-r-bottom">
          <div className="mode-row">
            {[["daily", "daily"], ["drift", "drift"], ["side-by-side", "vs bare"]].map(([k, l]) => (
              <button key={k} className={"mbtn " + (mode === k ? "on" : "")} onClick={() => setMode(k)}>{l}</button>
            ))}
          </div>
        </div>
      </div>
      </div>
      <TickerStrip golden={day.golden || []} />
    </header>
  );
}

// Marquee-style live ticker. Scrolls today's per-ticker % change horizontally
// across the bottom of the header. CSS keyframes do the scroll; we duplicate
// the chip set so the animation seam is invisible.
function TickerStrip({ golden }) {
  if (!golden.length) return <div className="ticker-strip empty" />;
  // Sort by absolute move so the loudest moves bunch up — keeps the eye busy.
  const sorted = [...golden].sort((a, b) => Math.abs(b.actual_return) - Math.abs(a.actual_return));
  const renderChip = (g, key) => {
    const r = g.actual_return ?? 0;
    const dir = r > 0.001 ? "up" : r < -0.001 ? "dn" : "flat";
    const arrow = dir === "up" ? "▲" : dir === "dn" ? "▼" : "·";
    const pct = (r * 100).toFixed(2);
    const sign = r >= 0 ? "+" : "";
    return (
      <span key={key} className={"tk-chip dir-" + dir}>
        <span className="tk-tkr">{g.ticker}</span>
        <span className="tk-arr">{arrow}</span>
        <span className="tk-pct">{sign}{pct}%</span>
      </span>
    );
  };
  return (
    <div className="ticker-strip">
      <div className="ticker-track">
        {sorted.map((g, i) => renderChip(g, "a" + i))}
        {/* Duplicate set so the loop has no visible seam */}
        {sorted.map((g, i) => renderChip(g, "b" + i))}
      </div>
    </div>
  );
}

function EqCellGrid({ v, ret, family, model }) {
  if (v == null) return <div className={"eg-cell empty fam-" + family} />;
  const sign = ret >= 0 ? "+" : "";
  return (
    <div className={"eg-cell fam-" + family + " mod-" + model + " " + (ret >= 0 ? "up" : "dn")}>
      <div className="eg-v">${(v/1000).toFixed(1)}k</div>
      <div className="eg-r">{sign}{ret.toFixed(1)}%</div>
    </div>
  );
}

function EqCell({ label, v, ret, cls }) {
  if (v == null) return null;
  const sign = ret >= 0 ? "+" : "";
  return (
    <div className={"eq-cell " + cls}>
      <div className="eq-l">{label}</div>
      <div className="eq-v">${(v/1000).toFixed(1)}k</div>
      <div className={"eq-r " + (ret >= 0 ? "up" : "dn")}>{sign}{ret.toFixed(2)}%</div>
    </div>
  );
}

function Counter({ label, v }) {
  return (
    <div className="counter">
      <div className="counter-l">{label}</div>
      <div className="counter-v">{v ?? 0}</div>
    </div>
  );
}

window.App = App;
window.TICKERS = TICKERS;
window.TEMPLATES = TEMPLATES;
window.TICKER_SECTORS = TICKER_SECTORS;
window.SECTOR_ORDER = SECTOR_ORDER;
window.SECTOR_LABEL = SECTOR_LABEL;
window.TEMPLATE_LABEL = TEMPLATE_LABEL;
window.TEMPLATE_DIR = TEMPLATE_DIR;
window.parseBeliefId = parseBeliefId;
