# Belief Timeline Animation — Handoff

Goal: turn the 66-day belief-network sweep into an animation that shows
**how leaf beliefs and outcome p_up change day-by-day**, and how the
decomposer's link-strength updates accumulate over time.

This doc is self-contained. A new conversation can pick it up cold and
implement the viz without needing the upstream architecture context.

---

## Data already extracted (1.3 MB)

All files live in `data/firehose_eval/timeline/`.

```
data/firehose_eval/timeline/
├── MANIFEST.json                    # schema + ordered list of dates
├── 2026-02-03.json … 2026-05-07.json   # one per trading day, 66 files
├── link_strength_trajectory.json    # decomposer updates per day
└── link_final_state.json            # final link strengths after 66-day replay
```

### Per-day file (`{date}.json`) shape

```json
{
  "date": "2026-02-18",
  "prev_trading_day": "2026-02-17",
  "n_news": 43,
  "n_activations": 25,
  "n_observations": 12,
  "belief_deltas": [
    {
      "belief_id": "belief.company.major_product_launch_positive_reception__ticker_AAPL",
      "p_before": 0.5,
      "p_after": 0.868,
      "delta": 0.368,
      "n_activations": 2,
      "ticker": "AAPL",
      "sample_reasoning": "Stock rose 3.12% on announcements of new Macs, AI devices, and podcast features..."
    }
  ],
  "outcome_deltas": [
    {
      "outcome_id": "outcome.AAPL.next_day_direction",
      "ticker": "AAPL",
      "p_up_before": 0.5,
      "p_up_after": 0.8625,
      "delta": 0.3625,
      "salience": 0.7766,
      "action": "BUY",
      "top_chain_summary": [
        { "parent_template": "major_product_launch_positive_reception",
          "contribution_log_odds": 1.142, "link_strength": 0.61 },
        { "parent_template": "earnings_beat",
          "contribution_log_odds": 0.694, "link_strength": 0.54 },
        { "parent_template": "technical_support_broken",
          "contribution_log_odds": -0.0,  "link_strength": 0.00 }
      ]
    }
  ],
  "realized_ground_truth": [
    {
      "ticker": "AAPL",
      "actual_direction": "flat",
      "actual_return": 0.00178,
      "magnitude_bucket": "flat",
      "ideal_p_up": 0.5
    }
  ],
  "watchlist_ecology": [ /* apex output, ranked */ ],
  "watchlist_bare":    [ /* bare apex output, for comparison */ ],
  "elapsed": { "ecology_s": 38.2, "bare_s": 14.74 }
}
```

### Link-strength trajectory (`link_strength_trajectory.json`)

```json
{
  "schema_version": "1.0",
  "prior_weight": 3.0,
  "days": [
    {
      "date": "2026-02-03",
      "n_link_updates": 13,
      "link_updates": [
        {
          "template": "earnings_beat", "ticker": "WMT",
          "before": 0.55, "after": 0.687, "delta": 0.137,
          "was_correct": true
        }
      ]
    }
  ]
}
```

**Note**: this is a **pure-Python replay** of the decomposer with
`prior_weight=3` against the existing pass artifacts. It is NOT what's
currently in Neo4j — Neo4j has the live values from the original sweep
(`prior_weight=15`, plus the 42 force-zeroed bad-template links). The
trajectory is the "what the decomposer would have done with the new
settings" story. Ground it as a synthetic-but-realistic timeline for the
animation.

### Final state (`link_final_state.json`)

After replaying the 66 days, 163 of 1681 links moved. Top 5 movers:

```
  technical_support_broken    UNH    0.000 → 0.500  (+0.500)
  technical_support_broken    MSFT   0.000 → 0.382  (+0.382)
  major_product_launch       MA     0.500 → 0.876  (+0.376)
  technical_support_broken    AAPL   0.000 → 0.363  (+0.363)
  regulatory_action           JNJ    0.000 → 0.360  (+0.360)
```

Notable: the disabled templates (`technical_support_broken`,
`regulatory_action_announced`) **rediscover their strengths** through online
learning — the decomposer can re-activate them when their predictions
correlate with reality on specific tickers. This is a feature: the
zero-out is a "demote, don't delete" — and the trajectory file shows the
demotion working as intended.

---

## Suggested animation design

### Storyboard

A horizontal timeline scrubber at the top of the existing viz HTML
(`trophic/beliefs/viz.py` output). Three concurrent visualization layers:

1. **Belief nodes**: opacity/color modulated by `(p_after - 0.5)`. Pulse
   when `delta` is large that day. Blue = bullish lean, red = bearish.
2. **Outcome nodes** (the 20 ticker outcomes): radius modulated by
   `|p_up - 0.5|`, color by direction. Halo color reflects realized
   ground-truth (green if outcome direction matched realized direction,
   red if not, gray for flat days).
3. **Edge thickness** (template→outcome links): tied to the *cumulative*
   `link_strength_trajectory` up to the currently-displayed day. So as
   you scrub forward, link thickness drifts visibly — that's the
   decomposer learning.

### Three playback modes

| Mode | Speed | Key story |
|------|-------|-----------|
| **Daily-by-daily** | 1 day / 1 sec | Watch belief activations land, outcomes change, ground truth resolve. Most readable for explaining what one pass does. |
| **Cumulative drift** | 5 days / 1 sec | Watch link strengths slowly diverge from seeds. Best for showing the decomposer learning regimes. |
| **Side-by-side ecology vs bare** | 1 day / 2 sec | Two columns showing the apex watchlists from each path. Highlights where they agree/disagree against ground truth. |

### Implementation hints

- Vanilla JS + d3 or vis-network is enough — no need for a build step. The
  existing `viz.py` already emits an HTML file with a static graph; add a
  `<script>` block that loads the timeline JSON files via `fetch()` and a
  `<input type="range">` scrubber.
- The 66 day files can be loaded all at once (1.3 MB total) into a big
  `Map<date, snapshot>`; or lazily. All-at-once is simpler.
- Cubic interpolation on `current_p` between adjacent days for smooth
  motion. Snap to discrete keyframes on `link_strength` (it only updates
  end-of-day).
- For ground truth halos: load `realized_ground_truth` from the day file
  and color each ticker outcome's halo on the SAME day (since p_up is the
  prediction for *next* trading session, the realization actually happens
  at the next day's prev_trading_day; the realized_ground_truth field
  encodes this offset).

### Suggested controls

```
[◀◀] [◀] [▶/❚❚] [▶] [▶▶]    speed: [0.5x][1x][2x][5x]
─────────────────●─────────────────  scrubber
  2026-02-03  ◯  ◯◯●◯◯◯◯◯◯  2026-05-07
                  current: 2026-02-18   action: BUY AAPL p_up=0.86

[ ] show belief deltas only (current day)
[x] show cumulative link strength
[x] show ground truth halos
[ ] side-by-side ecology vs bare
```

---

## Architectural details to preserve

When the viz reads belief nodes:

- **belief.company.{template}__ticker_{TKR}** — leaf beliefs, one per
  (template, ticker). 15 templates × 20 tickers = 300 leaves.
- **outcome.{TKR}.next_day_direction** — 20 outcomes.
- Sympathy (cross_correlation) activations also land on these same leaves
  (no separate node), so they'll show as additional `n_activations` in the
  belief_deltas. Add a tooltip showing `species` if you want to distinguish
  which herbivore fired.

---

## Open questions for the new conversation

1. Should the link-strength trajectory be the **replay** version (clean
   prior_weight=3, post-fix) or the **live Neo4j** version (prior_weight=15
   pre-fix, then prior_weight=3 post-fix)? The replay is cleaner; the live
   version is the historical truth. Suggest the replay for storytelling.
2. Whether to animate **intra-day** keyframes (initial → herb activations →
   propagation → decomposer end-state). Currently the data has only
   end-of-day snapshots. If you want intra-day, the runner needs to dump
   pre-propagation and post-propagation `current_p` separately — small change
   to `pass_record.py` (~15 lines).
3. UI library: vanilla JS+d3 vs Vite+React. The existing viz is vanilla;
   sticking with vanilla keeps the file portable as a single HTML.

---

## What lives where

| File | Purpose |
|------|---------|
| `data/firehose_eval/timeline/{date}.json` | Per-day belief + outcome deltas |
| `data/firehose_eval/timeline/link_strength_trajectory.json` | Decomposer updates per day (replay) |
| `data/firehose_eval/timeline/link_final_state.json` | Final link strengths summary |
| `data/firehose_eval/timeline/MANIFEST.json` | Index + schema |
| `data/firehose_eval/passes/{date}.json.gz` | Source of truth (raw pass artifacts) — re-run extraction script if schema changes |
| `data/firehose_eval/passes_pre_template_fix/` | Pre-fix snapshots for 5 confirmation days |
| `trophic/beliefs/viz.py` | Existing static graph viz — extend this for the animation |
| `docs/SIGNAL_FLOW_2026_05_09.md` | Architecture walkthrough — useful background reading |

The extraction logic is in this conversation's history (it's a small
`.venv/bin/python -c '...'` block reading pass artifacts and writing the
timeline files). If you need to regenerate after a fresh sweep, the
relevant code is:
- per-day deltas: aggregate `activations[i].leaf_p_before/after` per
  `target_belief_id`, then read `observations[i].p_up` directly
- trajectory: replay `decomposer_temporal._bayes_update_strength` with the
  current `prior_weight` over `observations[i].causal_chain[j].contribution_log_odds`

Both are <50 lines each.
