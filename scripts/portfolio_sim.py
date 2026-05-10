"""Portfolio sim: $100K → 64 days → ecology vs bare.

Reads data/firehose_eval/passes/*.json.gz. Each day's pass has:
  - watchlist_ecology / watchlist_bare: [{ticker, action, p_up, ...}, ...]
  - ground_truth: [{ticker, actual_return}, ...]   (next-day close-to-close)

Trade timing convention: signal is end-of-day on day N (news already
priced into day-N close). We trade at day-N close, hold overnight, mark
to market at day-(N+1) close. The pass artifact's `actual_return` is
exactly that day-N → day-(N+1) close-to-close move, so a BUY on day N
captures it.

Apex policy (the portfolio manager):
  edge      = 2 * p_up - 1            (+1 = certain up, −1 = certain down)
  - BUY  when edge >= +ENTRY  AND cash available — size = TARGET_FRAC
    of EQUITY, capped at PER_NAME_CAP, capped by available cash.
  - SELL when edge <= −EXIT  AND we hold the name — close the full position.
    (long-only by default; shorts gated behind --allow-short.)
  - HOLD otherwise.
  Cap aggregate exposure at MAX_INVESTED of equity (no margin).

Costs (toggleable):
  --slippage-bps N   : N bps of notional debited per BUY/SELL trade
  --tax-rate F       : F (e.g. 0.37) of net realized gains per closed position
                        accrued to a tax liability bucket; subtracted at end
  --no-frictions     : zero slippage, zero tax (sanity check)

Output: per-day equity curves + summary table. Optional CSV via --csv.
"""
from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
PASS_DIR = ROOT / "data" / "firehose_eval" / "passes"


# ── Parameters (apex policy) ─────────────────────────────────────────────

ENTRY_EDGE = 0.10        # |2p-1| above this triggers an order
EXIT_EDGE = 0.05         # close existing if edge falls below this
TARGET_FRAC = 0.10       # target % of equity per new BUY
PER_NAME_CAP = 0.20      # never let a single name exceed this fraction
MAX_INVESTED = 1.00      # no margin
DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_TAX_RATE = 0.37  # short-term cap gains (federal-only, simplified)


@dataclass
class Position:
    shares: float = 0.0
    cost_basis: float = 0.0   # total $ paid (after slippage)


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    realized_gains: float = 0.0
    tax_owed: float = 0.0
    n_buys: int = 0
    n_sells: int = 0
    notional_traded: float = 0.0

    def equity(self, prices: dict[str, float]) -> float:
        held = sum(
            p.shares * prices.get(t, 0.0)
            for t, p in self.positions.items()
        )
        return self.cash + held - self.tax_owed

    def invested_frac(self, prices: dict[str, float]) -> float:
        eq = self.equity(prices)
        if eq <= 0:
            return 0.0
        held = sum(
            p.shares * prices.get(t, 0.0)
            for t, p in self.positions.items()
        )
        return held / eq


# ── Price reconstruction ─────────────────────────────────────────────────
# We don't have absolute prices; we have day-to-day pct returns. So we
# track each ticker's price as an indexed series starting at $100, then
# compounding by `1 + actual_return` each day. This preserves the relative
# math (Kelly sizing, equity curves) but makes "shares" abstract — they're
# really fractions of the $100-indexed series. That's fine for an A/B.

def reconstruct_prices(pass_paths: list[Path]) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Return (sorted_dates, prices[date][ticker]).

    prices[date][ticker] is the close on `date`. Each ticker starts at
    $100 on the first date that has it in ground_truth.
    """
    series: dict[str, dict[str, float]] = {}  # ticker -> date -> close
    last_price: dict[str, float] = {}
    dates: list[str] = []

    for path in pass_paths:
        rec = json.load(gzip.open(path, "rt"))
        date = rec["date"]
        dates.append(date)
        for g in rec.get("ground_truth", []):
            t = g["ticker"]
            ret = float(g.get("actual_return") or 0.0)
            if t not in last_price:
                last_price[t] = 100.0
            else:
                last_price[t] *= (1.0 + ret)
            series.setdefault(t, {})[date] = last_price[t]

    # Build prices[date][ticker] map (last-known forward-fill)
    prices: dict[str, dict[str, float]] = {}
    last: dict[str, float] = {}
    for date in dates:
        prices[date] = {}
        for t, ser in series.items():
            if date in ser:
                last[t] = ser[date]
            if t in last:
                prices[date][t] = last[t]
    return dates, prices


# ── Apex policy ──────────────────────────────────────────────────────────

def decide_orders(
    watchlist: list[dict],
    portfolio: Portfolio,
    prices: dict[str, float],
    allow_short: bool,
) -> list[tuple[str, str, float]]:
    """Return [(ticker, side, dollars), ...].
    side ∈ {"BUY", "SELL"}. Dollars is notional intent; capped/clamped later.
    """
    equity = portfolio.equity(prices)
    if equity <= 0:
        return []
    invested_frac = portfolio.invested_frac(prices)
    cash_room = max(0.0, MAX_INVESTED * equity - invested_frac * equity)

    orders: list[tuple[str, str, float]] = []

    # First pass: SELLs (free up cash before BUYs)
    for entry in watchlist:
        t = entry["ticker"]
        p_up = float(entry.get("p_up", 0.5))
        edge = 2 * p_up - 1
        action = entry.get("action", "")

        held = portfolio.positions.get(t)
        if held and held.shares > 0 and edge <= -EXIT_EDGE:
            # Close long
            mv = held.shares * prices.get(t, 0.0)
            orders.append((t, "SELL", mv))

    # Second pass: BUYs
    for entry in watchlist:
        t = entry["ticker"]
        if t not in prices:
            continue
        p_up = float(entry.get("p_up", 0.5))
        edge = 2 * p_up - 1

        if edge >= ENTRY_EDGE:
            # Size: edge-scaled, capped by per-name and cash room
            target_dollars = TARGET_FRAC * equity * min(1.0, edge / 0.40)
            held = portfolio.positions.get(t)
            current_mv = (held.shares * prices[t]) if held else 0.0
            cap_room = PER_NAME_CAP * equity - current_mv
            buy_dollars = max(0.0, min(target_dollars, cap_room, cash_room))
            if buy_dollars > 1.0:  # ignore <$1 orders
                orders.append((t, "BUY", buy_dollars))
                cash_room -= buy_dollars
        elif allow_short and edge <= -ENTRY_EDGE:
            # Short sizing path (off by default)
            held = portfolio.positions.get(t)
            if not held or held.shares >= 0:
                target_dollars = TARGET_FRAC * equity * min(1.0, -edge / 0.40)
                short_dollars = min(target_dollars, cash_room)
                if short_dollars > 1.0:
                    orders.append((t, "BUY", -short_dollars))  # negative = short
                    cash_room -= short_dollars
    return orders


# ── Execution ────────────────────────────────────────────────────────────

def execute(
    portfolio: Portfolio,
    orders: list[tuple[str, str, float]],
    prices: dict[str, float],
    slippage_bps: float,
    tax_rate: float,
) -> None:
    bps = slippage_bps / 1e4
    for ticker, side, dollars in orders:
        px = prices.get(ticker)
        if not px:
            continue
        if side == "BUY":
            cost_with_slip = abs(dollars) * (1 + bps)
            if cost_with_slip > portfolio.cash:
                cost_with_slip = portfolio.cash
                dollars = cost_with_slip / (1 + bps)
            shares = abs(dollars) / px
            portfolio.cash -= cost_with_slip
            pos = portfolio.positions.setdefault(ticker, Position())
            pos.shares += shares
            pos.cost_basis += cost_with_slip
            portfolio.n_buys += 1
            portfolio.notional_traded += abs(dollars)
        elif side == "SELL":
            pos = portfolio.positions.get(ticker)
            if not pos or pos.shares <= 0:
                continue
            shares_to_sell = min(pos.shares, abs(dollars) / px)
            proceeds_gross = shares_to_sell * px
            proceeds_net = proceeds_gross * (1 - bps)
            cost_for_those = (shares_to_sell / pos.shares) * pos.cost_basis
            gain = proceeds_net - cost_for_those
            portfolio.realized_gains += gain
            if gain > 0 and tax_rate > 0:
                portfolio.tax_owed += gain * tax_rate
            portfolio.cash += proceeds_net
            pos.shares -= shares_to_sell
            pos.cost_basis -= cost_for_those
            if pos.shares < 1e-9:
                del portfolio.positions[ticker]
            portfolio.n_sells += 1
            portfolio.notional_traded += proceeds_gross


# ── Run sim ──────────────────────────────────────────────────────────────

def simulate(
    label: str,
    watchlist_key: str,
    pass_paths: list[Path],
    dates: list[str],
    prices: dict[str, dict[str, float]],
    starting_cash: float,
    slippage_bps: float,
    tax_rate: float,
    allow_short: bool,
    verbose: bool,
) -> dict:
    pf = Portfolio(cash=starting_cash)
    equity_curve: list[tuple[str, float]] = []
    daily_states: list[dict] = []  # full per-day snapshot for timeline merge

    for i, path in enumerate(pass_paths):
        rec = json.load(gzip.open(path, "rt"))
        date = rec["date"]
        watchlist = rec.get(watchlist_key, [])
        day_prices = prices.get(date, {})

        # Capture pre-trade state for the apex's decision context
        pre_cash = pf.cash
        pre_equity = pf.equity(day_prices)

        orders = decide_orders(watchlist, pf, day_prices, allow_short)

        # Snapshot orders before execute() can clip them
        order_snap = []
        for ticker, side, dollars in orders:
            px = day_prices.get(ticker)
            if not px:
                continue
            order_snap.append({
                "ticker": ticker,
                "side": side,
                "dollars_intent": round(abs(dollars), 2),
                "px": round(px, 4),
            })

        execute(pf, orders, day_prices, slippage_bps, tax_rate)

        # Mark to market at end of day (uses *this* day's prices since
        # actual_return advances the price index between days)
        eq = pf.equity(day_prices)
        equity_curve.append((date, eq))

        # Per-day state for the timeline. open_positions reflects POST-trade
        # holdings; orders_today is what just executed (intent ≈ executed
        # for non-clipped orders).
        open_positions = []
        for t, p in pf.positions.items():
            mv = p.shares * day_prices.get(t, 0.0)
            unrealized = mv - p.cost_basis
            open_positions.append({
                "ticker": t,
                "shares": round(p.shares, 6),
                "cost_basis": round(p.cost_basis, 2),
                "mv": round(mv, 2),
                "unrealized_pnl": round(unrealized, 2),
                "unrealized_pnl_pct": round(unrealized / p.cost_basis, 4)
                if p.cost_basis > 0 else 0.0,
            })
        daily_states.append({
            "date": date,
            "equity": round(eq, 2),
            "cash": round(pf.cash, 2),
            "invested_frac": round(pf.invested_frac(day_prices), 4),
            "tax_owed_accrued": round(pf.tax_owed, 2),
            "realized_gains_cum": round(pf.realized_gains, 2),
            "n_open_positions": len(pf.positions),
            "pre_cash": round(pre_cash, 2),
            "pre_equity": round(pre_equity, 2),
            "orders_today": order_snap,
            "open_positions": open_positions,
        })

        if verbose:
            n_orders = len(orders)
            n_pos = len(pf.positions)
            inv = pf.invested_frac(day_prices)
            print(f"  [{label}] {date}: equity=${eq:>10,.0f} "
                  f"cash=${pf.cash:>10,.0f} pos={n_pos:>2} "
                  f"inv={inv:5.1%} orders={n_orders}")

    final_eq = equity_curve[-1][1]
    total_return = (final_eq - starting_cash) / starting_cash
    return {
        "label": label,
        "starting": starting_cash,
        "final": final_eq,
        "total_return": total_return,
        "equity_curve": equity_curve,
        "daily_states": daily_states,
        "n_buys": pf.n_buys,
        "n_sells": pf.n_sells,
        "notional_traded": pf.notional_traded,
        "realized_gains": pf.realized_gains,
        "tax_owed": pf.tax_owed,
        "n_open_positions": len(pf.positions),
    }


def run_full(
    starting_cash: float = 100_000.0,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    tax_rate: float = DEFAULT_TAX_RATE,
    allow_short: bool = False,
    verbose: bool = False,
) -> dict:
    """Programmatic entrypoint. Returns dict with eco/bare/buyhold + dates."""
    pass_paths = sorted(PASS_DIR.glob("*.json.gz"))
    dates, prices = reconstruct_prices(pass_paths)

    eco = simulate(
        "ECOLOGY", "watchlist_ecology", pass_paths, dates, prices,
        starting_cash, slippage_bps, tax_rate, allow_short, verbose,
    )
    bare = simulate(
        "BARE", "watchlist_bare", pass_paths, dates, prices,
        starting_cash, slippage_bps, tax_rate, allow_short, verbose,
    )
    # Buy-and-hold baseline equity curve, day by day
    bh_pf = Portfolio(cash=starting_cash)
    first_prices = prices[dates[0]]
    n_tickers = len(first_prices)
    per_name = starting_cash / n_tickers
    for t, px in first_prices.items():
        cost = per_name * (1 + slippage_bps / 1e4)
        bh_pf.cash -= cost
        bh_pf.positions[t] = Position(shares=per_name / px, cost_basis=cost)
    bh_pf.n_buys = n_tickers
    bh_curve = [(d, bh_pf.equity(prices[d])) for d in dates]
    bh = {
        "label": "BUY_AND_HOLD",
        "starting": starting_cash,
        "final": bh_curve[-1][1],
        "total_return": (bh_curve[-1][1] - starting_cash) / starting_cash,
        "equity_curve": bh_curve,
        "n_buys": n_tickers,
        "n_sells": 0,
    }

    return {
        "dates": dates,
        "ecology": eco,
        "bare": bare,
        "buyhold": bh,
        "policy": {
            "entry_edge": ENTRY_EDGE,
            "exit_edge": EXIT_EDGE,
            "target_frac": TARGET_FRAC,
            "per_name_cap": PER_NAME_CAP,
            "max_invested": MAX_INVESTED,
        },
        "frictions": {
            "slippage_bps": slippage_bps,
            "tax_rate": tax_rate,
        },
        "starting_cash": starting_cash,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--starting-cash", type=float, default=100_000.0)
    ap.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    ap.add_argument("--tax-rate", type=float, default=DEFAULT_TAX_RATE)
    ap.add_argument("--no-frictions", action="store_true",
                    help="zero slippage, zero tax (sanity test)")
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--csv", default=None,
                    help="Write equity curves to this CSV path")
    ap.add_argument("--json-out", default=None,
                    help="Write full sim result (incl. per-day states) to JSON")
    args = ap.parse_args()

    if args.no_frictions:
        slip = 0.0
        tax = 0.0
        print("Mode: FRICTIONLESS (no slippage, no tax)")
    else:
        slip = args.slippage_bps
        tax = args.tax_rate
        print(f"Mode: REALISTIC (slippage={slip} bps, tax={tax:.0%} on realized gains)")

    pass_paths = sorted(PASS_DIR.glob("*.json.gz"))
    print(f"Loaded {len(pass_paths)} pass artifacts: "
          f"{pass_paths[0].stem.removesuffix('.json')} → {pass_paths[-1].stem.removesuffix('.json')}")

    dates, prices = reconstruct_prices(pass_paths)
    print(f"Reconstructed prices for {len(prices)} days, "
          f"{len(set().union(*[set(p.keys()) for p in prices.values()]))} tickers")
    print()

    print("Running ECOLOGY...")
    eco = simulate(
        "ECOLOGY", "watchlist_ecology", pass_paths, dates, prices,
        args.starting_cash, slip, tax, args.allow_short, args.verbose,
    )
    print()
    print("Running BARE...")
    bare = simulate(
        "BARE", "watchlist_bare", pass_paths, dates, prices,
        args.starting_cash, slip, tax, args.allow_short, args.verbose,
    )
    print()

    # Buy-and-hold baseline: equal-weight all 20 tickers on day 0, hold to end
    print("Running BUY-AND-HOLD (equal-weight all 20)...")
    bh_pf = Portfolio(cash=args.starting_cash)
    first_prices = prices[dates[0]]
    n_tickers = len(first_prices)
    per_name = args.starting_cash / n_tickers
    for t, px in first_prices.items():
        cost = per_name * (1 + slip / 1e4)
        bh_pf.cash -= cost
        bh_pf.positions[t] = Position(shares=per_name / px, cost_basis=cost)
    bh_pf.n_buys = n_tickers
    bh_eq = bh_pf.equity(prices[dates[-1]])
    bh = {
        "label": "BUY_AND_HOLD",
        "starting": args.starting_cash,
        "final": bh_eq,
        "total_return": (bh_eq - args.starting_cash) / args.starting_cash,
        "n_buys": n_tickers,
        "n_sells": 0,
    }
    print()

    # ── Summary ─────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"{'STRATEGY':<14} {'FINAL EQUITY':>14} {'P&L':>12} {'RETURN':>10} "
          f"{'#BUY':>6} {'#SELL':>6}")
    print("-" * 70)
    for r in [eco, bare, bh]:
        pnl = r["final"] - r["starting"]
        print(f"{r['label']:<14} ${r['final']:>13,.0f} "
              f"${pnl:>+11,.0f} {r['total_return']:>+9.2%} "
              f"{r.get('n_buys', 0):>6} {r.get('n_sells', 0):>6}")
    print("=" * 70)

    delta = eco["final"] - bare["final"]
    print(f"\nECOLOGY − BARE = ${delta:+,.0f}  "
          f"({(eco['total_return'] - bare['total_return'])*100:+.2f} pp)")
    if not args.no_frictions:
        print(f"  ECOLOGY tax owed: ${eco['tax_owed']:,.0f}  "
              f"realized gains: ${eco['realized_gains']:,.0f}")
        print(f"  BARE    tax owed: ${bare['tax_owed']:,.0f}  "
              f"realized gains: ${bare['realized_gains']:,.0f}")
    print(f"  ECOLOGY notional traded: ${eco['notional_traded']:,.0f}")
    print(f"  BARE    notional traded: ${bare['notional_traded']:,.0f}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "ecology_equity", "bare_equity"])
            eco_dict = dict(eco["equity_curve"])
            bare_dict = dict(bare["equity_curve"])
            for d in dates:
                w.writerow([d, f"{eco_dict.get(d,''):.2f}", f"{bare_dict.get(d,''):.2f}"])
        print(f"\nEquity curves written to {args.csv}")

    if args.json_out:
        out = {
            "dates": dates,
            "ecology": eco,
            "bare": bare,
            "buyhold": bh,
            "policy": {
                "entry_edge": ENTRY_EDGE,
                "exit_edge": EXIT_EDGE,
                "target_frac": TARGET_FRAC,
                "per_name_cap": PER_NAME_CAP,
                "max_invested": MAX_INVESTED,
            },
            "frictions": {"slippage_bps": slip, "tax_rate": tax},
            "starting_cash": args.starting_cash,
        }
        Path(args.json_out).write_text(json.dumps(out, indent=1))
        print(f"\nFull sim result written to {args.json_out}")


if __name__ == "__main__":
    main()
