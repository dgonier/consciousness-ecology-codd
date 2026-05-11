# Belief Timeline Dashboard

A static React+SVG visualization of a 66-day firehose-eval run. Rooted in
`trophic/dashboard/`, served as plain files — no build step.

This is the visualization-only counterpart to:
- `trophic/docs/SIGNAL_FLOW_2026_05_09.md` — what the data means.
- `trophic/docs/TIMELINE_VIZ_HANDOFF.md` — the original animation spec
  (the dashboard implements and extends it).
- `trophic/design_docs/belief_network_schema.md` — the upstream schema.

**Don't read this doc to learn the belief network**; read the three above.
This doc only covers what's in `trophic/dashboard/` itself.

## Run

```bash
cd trophic/dashboard
python3 -m http.server 8766
# open http://localhost:8766/index.html
```

React 18, ReactDOM, and Babel-standalone are loaded from unpkg via CDN.
Source is `.jsx` compiled in the browser by Babel — slow first paint,
fine for a single-user analysis tool.

## Loading a different run

Per-day JSON, `MANIFEST.json`, and `run_meta.json` in this directory are
the dashboard's data. To load a fresh `firehose_eval` run:

```bash
python3 _convert_run.py /path/to/runs/run_<label>
# wipes and rewrites 2026-*.json + MANIFEST.json + run_meta.json
```

`_convert_run.py` translates from the v3 9-way ablation schema
(`eval.jsonl` with one line per day, 9 sub-pipelines per line) into the
per-day v2 schema the dashboard reads. See `_convert_run.py` for the
exact field mapping; the salient choices:

- ECOLOGY-QWEN is wired to the back-compat `portfolio.ecology` alias and
  is the only pipeline whose belief-network internals (`belief_deltas`,
  `causal_chain`) populate the heatmap. The other 8 pipelines surface
  via holdings + sparkline + orders only.
- `golden[].ideal_p_up` is **derived** from the realized return via a
  sigmoid (±3% saturates to 0.05/0.95). The v3 source has no
  ground-truth ideal probability; this is a calibrated proxy used only
  for the ideal-tick chevron in the grid.
- `portfolio.buyhold_equity` is computed in-converter as an
  equal-weighted basket of all 20 tickers from day 1 prev_close. The v3
  source has no buyhold field.

If your run uses a different schema, edit `_convert_run.py` rather than
the React code.

## File layout

| File | Role |
|---|---|
| `index.html` | Entry point. Loads React, Babel, and every `viz-*.jsx` in order. |
| `styles.css` | All styling. Theme tokens at the top; component blocks below. |
| `app.jsx` | Top-level App, data loader, header, equity-grid (3×3 model×family), pipeline metadata constants. |
| `viz-grid.jsx` | 22 templates × 20 tickers belief heatmap + outcome strip with ideal-p_up chevrons. ECOLOGY-QWEN only. |
| `viz-holdings.jsx` | Top-of-left-column panel: 9 stacked horizontal bars (cash + positions, $0–$150k axis) + position weight heatmap (red→green by % of equity). |
| `viz-context.jsx` | Macro/sector/market context strip above the belief grid. Empty for v3 source (no macro beliefs). |
| `viz-spotlight.jsx` | Right rail. 9-line equity sparkline, predictions vs reality, today's orders table (3 family columns × 3 model dots), apex watchlists, MCC scoreboard, leaf activations. |
| `viz-below.jsx` | Below the belief grid: beliefs-not-fired groups + signals & sources columns. |
| `viz-scrubber.jsx` | Bottom playback bar with activity sparkline; tweaks panel host. |
| `_convert_run.py` | One-shot converter. Re-run when swapping data. |

## What's on screen

**Header.** Title + subtitle + run-meta pills + a 3×3 equity grid (rows =
apex model, cols = pipeline family). BUYHOLD + mode buttons sit below the
grid.

**Holdings panel (top of left column).**
- 9 stacked horizontal bars on a shared $0–$150k axis. Cash on the left,
  positions to the right sorted by market value. Green/red bottom inset
  on each position segment shows unrealized P&L.
- **Position weight heatmap** below the bars. Rows = the 9 strategies in
  the same order. Columns = the 20 tickers in the same sector-grouped
  order as the belief grid below it (same `labelW=200` and
  `cellW = clamp(28, floor((containerW − 200 − 32) / 20), 64)`). Color =
  weight % of equity, on a single red→amber→green ramp where empty = no
  fill, ~5% = dim red, ~10% = amber, 20%+ = full green. The integer %
  renders inside cells ≥ 4%. The ramp is identical across all 9 rows so
  cross-strategy alignment reads at a glance.

**Belief grid.** 22 templates × 20 tickers. Cell fill = `p_after`;
border thickness = cumulative link strength; pulse = today's `delta`.
Outcome strip below shows predicted `p_up` with an ideal-tick chevron
underneath.

**Below the grid.** Beliefs-not-fired groups (dormant cells), and a
two-column signals & sources view (classifier evidence + outcome
reasoning). All ECOLOGY-QWEN.

**Right rail.** Equity sparkline (9 lines + buyhold; family hue, model
dash style: qwen solid, sonnet dashed, opus dotted), predictions vs
reality, watchlists, scoreboard.

**Scrubber.** Bottom bar. Click or drag to seek. Activity sparkline =
`n_activations` per day.

## Family colors

The three pipeline families have vivid identity colors (set via CSS
custom props near the top of `styles.css`):

- `--fam-bare` — neon orange (hue 30) — the null hypothesis
- `--fam-ecology` — hot magenta (hue 320) — the candidate
- `--fam-oracle` — electric blue (hue 220) — peeks at the future

These show up on row labels (holdings + heatmap), header equity-grid
borders, sparkline lines + dots, and order-pill labels. The position
heatmap is the **one place** that does not use family colors — it uses
the red→green weight ramp so alignment patterns across strategies are
visible.

## Caveats

- Only ECOLOGY-QWEN populates the belief grid, predictions panel, and
  signals & sources. The v3 source carries observations + causal_chains
  for that pipeline only.
- The `portfolio.ecology` and `portfolio.bare` keys are aliases pointing
  at the qwen variants for back-compat with the original 2-strategy
  prototype. The PortfolioPanel header strip ("ecology"/"bare" pills)
  duplicates info now also shown in the 3×3 header grid; safe to remove
  if it bothers you.
- `ideal_p_up` is a derived proxy, not a model output. Useful for visual
  sanity but don't read deep meaning into per-day chevron positions.
- The dashboard is **single-horizon**. Every prediction is implicitly
  next-day (`obs.horizon_min: 1440`). v4 plans to add h1/h5/h20/h60
  forecasts; the dashboard will need new surfaces when that data lands.

## Adding a new run-derived field

1. Add the extraction to `_convert_run.py` and re-run it.
2. Add a renderer to whichever `viz-*.jsx` owns the surface, plus styles
   in `styles.css` near the related block.
3. Bump the `?v=<tag>` cache token on the affected `<script>` lines in
   `index.html` so browsers don't serve stale Babel-compiled cache.

## Adding a new pipeline

If a future run adds a 10th sub-pipeline, you'll touch:

- `_convert_run.py` — add the source key → dashboard key mapping in `PIPELINES`.
- `app.jsx` — add to `PIPELINE_KEYS` + the family/model/label tables.
- `viz-holdings.jsx` — add to `PIPELINES_DISPLAY`.
- `viz-spotlight.jsx` — add to the `PIPELINES` array in `PortfolioPanel`.
- `styles.css` — add a `--fam-<name>` token if the new pipeline is a new
  family; otherwise existing family CSS picks it up.
